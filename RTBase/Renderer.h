#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include "Scene.h"
#include "GamesEngineeringBase.h"
#include <thread>
#include <functional>
#include <mutex>
#include <queue>
#include <OpenImageDenoise/oidn.hpp>
#include <algorithm>

class VPL {
public:
	ShadingData shadingData;
	Colour Ls; // VPL的辐射亮度
	Vec3 position() const { return shadingData.x; }
	void init(ShadingData _shadingData, Colour c) {
		shadingData = _shadingData;
		Ls = c;
	}
};

class RayTracer
{
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom *samplers;
	std::thread **threads;
	int numProcs;
	std::vector<VPL> vpls;

	struct BlockStats {
		std::vector<Colour> pixelSamples;
		Colour sum;
		Colour sumSquares;
		float variance = 0.0f;
		int allocatedSamples = 4;
		int x0, y0, x1, y1;
	};

	int blockSize = 32;
	int numBlocksX = 0;         // 横向分块数
	int numBlocksY = 0;         // 纵向分块数
	std::vector<BlockStats> blocks;
	int totalSamples = 256;     // 总样本预算

	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new MitchellNetravaliFilter());
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		numProcs = sysInfo.dwNumberOfProcessors;
		threads = new std::thread*[numProcs];
		samplers = new MTRandom[numProcs];
		clear();

		for (int i = 0; i < numProcs; i++) {
			samplers[i] = MTRandom(i + 1);
		}

		blockSize = 32;
		numBlocksX = (scene->camera.width + blockSize - 1) / blockSize;
		numBlocksY = (scene->camera.height + blockSize - 1) / blockSize;
		blocks.resize(numBlocksX * numBlocksY);

		for (int by = 0; by < numBlocksY; ++by) {
			for (int bx = 0; bx < numBlocksX; ++bx) {
				auto& block = blocks[by * numBlocksX + bx];
				block.x0 = bx * blockSize;
				block.y0 = by * blockSize;
				block.x1 = min(block.x0 + blockSize, scene->camera.width);
				block.y1 = min(block.y0 + blockSize, scene->camera.height);
				block.pixelSamples.resize((block.x1 - block.x0) * (block.y1 - block.y0), Colour(0, 0, 0));
			}
		}
	}
	void clear()
	{
		film->clear();
		for (auto& block : blocks) {
			block.sum = Colour(0, 0, 0);
			block.sumSquares = Colour(0, 0, 0);
			block.variance = 0.0f;
			block.allocatedSamples = 4;
			std::fill(block.pixelSamples.begin(), block.pixelSamples.end(), Colour(0, 0, 0));
		}
	}

#define MAX_DEPTH 8

	void traceVPLs(Sampler* sampler, int N_VPLs) {
		vpls.clear();

		for (int i = 0; i < N_VPLs; ++i) {
			// 1. 采样光源
			float lightPmf;
			Light* light = scene->sampleLight(sampler, lightPmf);
			if (!light || lightPmf <= 1e-6f) continue;

			// 2. 采样光源位置和方向
			float pdfPosition, pdfDirection;
			Vec3 lightPos = light->samplePositionFromLight(sampler, pdfPosition);
			Vec3 lightDir = light->sampleDirectionFromLight(sampler, pdfDirection);

			// 3. 初始化路径吞吐量
			Colour Le = light->evaluate(ShadingData(), -lightDir);
			Colour throughput = Le * std::abs(Dot(lightDir, light->normal(ShadingData(), lightDir)))
				/ (lightPmf * pdfPosition * pdfDirection);

			// 4. 开始路径追踪（直接调用VPLTracePath）
			Ray ray(lightPos + lightDir * EPSILON, lightDir);
			VPLTracePath(ray, throughput, sampler, 0);
		}
	}

	void VPLTracePath(Ray& r, Colour pathThroughput, Sampler* sampler, int depth) {
		if (depth >= 5) return; // 限制最大深度

		IntersectionData isect = scene->traverse(r);
		if (isect.t >= FLT_MAX) return;

		ShadingData sd = scene->calculateShadingData(isect, r);

		// 只在非镜面表面生成VPL
		if (!sd.bsdf->isPureSpecular()) {
			VPL newVPL;
			newVPL.shadingData = sd;
			newVPL.Ls = pathThroughput; // 路径吞吐量直接作为辐射亮度
			vpls.push_back(newVPL);
		}

		// 俄罗斯轮盘终止
		float rrProb = min(pathThroughput.Lum(), 0.95f);
		if (sampler->next() > rrProb) return;
		pathThroughput = pathThroughput / rrProb;

		// 采样BSDF方向
		Colour bsdfVal;
		float pdf;
		Vec3 wi = sd.bsdf->sample(sd, sampler, bsdfVal, pdf);
		if (pdf < 1e-6f) return;

		// 更新路径吞吐量
		pathThroughput = pathThroughput * bsdfVal * std::abs(Dot(wi, sd.sNormal)) / pdf;

		// 继续追踪
		Ray nextRay(sd.x + wi * EPSILON, wi);
		VPLTracePath(nextRay, pathThroughput, sampler, depth + 1);
	}

	Colour ncomputeDirect(ShadingData shadingData, Sampler* sampler)
	{
		Colour direct;
		Colour result;
		if (shadingData.bsdf->isPureSpecular() == true)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		// Sample a light
		float pmf;
		Light* light = scene->sampleLight(sampler, pmf);
		// Sample a point on the light
		float pdf;
		Colour emitted;
		Vec3 p = light->sample(shadingData, sampler, emitted, pdf);
		if (light->isArea())
		{
			// Calculate GTerm
			Vec3 wi = p - shadingData.x;
			float l = wi.lengthSq();
			wi = wi.normalize();
			float GTerm = (max(Dot(wi, shadingData.sNormal), 0.0f) * max(-Dot(wi, light->normal(shadingData, wi)), 0.0f)) / l;
			if (GTerm > 0)
			{
				// Trace
				if (scene->visible(shadingData.x, p))
				{
					// Shade
					direct = shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
				}
			}
		}
		else
		{
			// Calculate GTerm
			Vec3 wi = p;
			float GTerm = max(Dot(wi, shadingData.sNormal), 0.0f);
			if (GTerm > 0)
			{
				// Trace
				if (scene->visible(shadingData.x, shadingData.x + (p * 10000.0f)))
				{
					// Shade
					direct = shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
				}
			}
		}
		for (const VPL& vpl : vpls) {
			Vec3 x_i = vpl.position();
			Vec3 wi = (x_i - shadingData.x).normalize();

			// 几何项计算
			float dist2 = (x_i - shadingData.x).lengthSq();
			float cosTheta = std::abs(Dot(wi, shadingData.sNormal));
			float cosThetaVPL = std::abs(Dot(-wi, vpl.shadingData.sNormal));
			float G = cosTheta * cosThetaVPL / dist2;
			if (G < 1e-6f) continue;
			// 可见性测试
			if (scene->visible(shadingData.x, x_i)) {
				// 双向BSDF评估
				Colour frCamera = shadingData.bsdf->evaluate(shadingData, wi);
				Colour frVPL = vpl.shadingData.bsdf->evaluate(vpl.shadingData, -wi);
				result = result + frCamera * frVPL * G * vpl.Ls;
			}
		}
		return  direct + result;
	}

	Colour computeDirect(ShadingData shadingData, Sampler* sampler)
	{
		if (shadingData.bsdf->isPureSpecular() == true)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		// Sample a light
		float pmf;
		Light* light = scene->sampleLight(sampler, pmf);
		// Sample a point on the light
		float pdf;
		Colour emitted;
		Vec3 p = light->sample(shadingData, sampler, emitted, pdf);
		if (light->isArea())
		{
			// Calculate GTerm
			Vec3 wi = p - shadingData.x;
			float l = wi.lengthSq();
			wi = wi.normalize();
			float GTerm = (max(Dot(wi, shadingData.sNormal), 0.0f) * max(-Dot(wi, light->normal(shadingData, wi)), 0.0f)) / l;
			if (GTerm > 0)
			{
				// Trace
				if (scene->visible(shadingData.x, p))
				{
					// Shade
					return shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
				}
			}
		}
		else
		{
			// Calculate GTerm
			Vec3 wi = p;
			float GTerm = max(Dot(wi, shadingData.sNormal), 0.0f);
			if (GTerm > 0)
			{
				// Trace
				if (scene->visible(shadingData.x, shadingData.x + (p * 10000.0f)))
				{
					// Shade
					return shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
				}
			}
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler, bool canHitLight = true)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				if (canHitLight == true)
				{
					return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
				}
				else
				{
					return Colour(0.0f, 0.0f, 0.0f);
				}
			}
			Colour direct = pathThroughput * computeDirect(shadingData, sampler);
			if (depth > MAX_DEPTH)
			{
				return direct;
			}
			float russianRouletteProbability = min(pathThroughput.Lum(), 0.9f);
			if (sampler->next() < russianRouletteProbability)
			{
				pathThroughput = pathThroughput / russianRouletteProbability;
			}
			else
			{
				return direct;
			}
			Colour bsdf;
			float pdf;
			Vec3 wi = shadingData.bsdf->sample(shadingData, sampler, bsdf, pdf);

			pathThroughput = pathThroughput * bsdf * fabsf(Dot(wi, shadingData.sNormal)) / pdf;
			r.init(shadingData.x + (wi * EPSILON), wi);
			
			return (direct + pathTrace(r, pathThroughput, depth + 1, sampler, shadingData.bsdf->isPureSpecular()));
		}
		return scene->background->evaluate(shadingData, r.dir);
	}
	Colour direct(Ray& r, Sampler* sampler)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return computeDirect(shadingData, sampler);
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}	

	Colour albedo(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return shadingData.bsdf->evaluate(shadingData, Vec3(0, 1, 0));
		}
		return scene->background->evaluate(shadingData, r.dir);
	}
	Colour viewNormals(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX)
		{
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.sNormal.x), fabsf(shadingData.sNormal.y), fabsf(shadingData.sNormal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	void computeVarianceAndAllocate() {
		float totalVariance = 0.0f;

		// 计算每个块的方差
		for (auto& block : blocks) {
			int numPixels = (block.x1 - block.x0) * (block.y1 - block.y0);
			if (numPixels == 0) continue;

			Colour mean = block.sum / numPixels;
			Colour meanSquares = block.sumSquares / numPixels;
			block.variance = (meanSquares - mean * mean).Lum();
			totalVariance += block.variance;
		}

		// 分配样本
		const int remainingSamples = totalSamples - film->SPP;
		for (auto& block : blocks) {
			float weight = totalVariance > 0 ? block.variance / totalVariance : 1.0f / blocks.size();
			block.allocatedSamples = max(1, (int)std::round(weight * remainingSamples));
		}
	}
	void render() {
		const int initialSamples = 4;
		// 初始化分配
		for (auto& block : blocks) {
			block.allocatedSamples = initialSamples;
		}

		for (int iter = 0; iter < totalSamples / initialSamples; ++iter) {
			film->incrementSPP();

			// 多线程渲染代码（修改为按块索引）
			auto renderBlock = [&](int blockIdx, int threadId) {
				auto& block = blocks[blockIdx];
				if (block.allocatedSamples <= 0) return;

				Sampler* localSampler = &samplers[threadId];

				for (int y = block.y0; y < block.y1; y++) {
					for (int x = block.x0; x < block.x1; x++) {
						for (int s = 0; s < block.allocatedSamples; s++) {

							float px = std::clamp(
								x + localSampler->next(),
								0.0f,
								static_cast<float>(scene->camera.width - 1)
							);

							float py = std::clamp(
								y + localSampler->next(),
								0.0f,
								static_cast<float>(scene->camera.height - 1)
							);

							Ray ray = scene->camera.generateRay(px, py);
							Colour pathThroughput(1.0f, 1.f, 1.f);
							Colour col = pathTrace(ray, pathThroughput, 5, localSampler);

							int idx = (y - block.y0) * (block.x1 - block.x0) + (x - block.x0);
							block.pixelSamples[idx] = block.pixelSamples[idx] + col;
							block.sum = block.sum + col;
							block.sumSquares = block.sumSquares + col * col;

							film->splat(px, py, col);
							unsigned char r = (unsigned char)(col.r * 255);
							unsigned char g = (unsigned char)(col.g * 255);
							unsigned char b = (unsigned char)(col.b * 255);
							film->tonemap(px, py, r, g, b);
							canvas->draw(x, y, r, g, b);

						}
					}
				}
				block.allocatedSamples = 0;
				};

			// 启动多线程渲染
			std::vector<std::thread> workers;
			for (int i = 0; i < numProcs; i++) {
				workers.emplace_back([&, i]() {
					for (int b = i; b < blocks.size(); b += numProcs) {
						renderBlock(b, i);
					}
					});
			}
			for (auto& w : workers) w.join();

			// 计算方差并分配下一批样本
			computeVarianceAndAllocate();
		}
	}
	void nrender()
	{
		static const int TILE_SIZE = 32;
		film->incrementSPP();

		int numThreads = numProcs;
		std::vector<std::thread> workers;
		workers.reserve(numThreads);

		int numTilesX = (film->width + TILE_SIZE - 1) / TILE_SIZE;
		int numTilesY = (film->height + TILE_SIZE - 1) / TILE_SIZE;
		std::vector<float> hdrpixels(film->width * film->height * 3, 0.0f);

		auto renderTile = [&](int tileX, int tileY, int threadId)
			{
				int startX = tileX * TILE_SIZE;
				int startY = tileY * TILE_SIZE;
				int endX = min(startX + TILE_SIZE, static_cast<int>(film->width));
				int endY = min(startY + TILE_SIZE, static_cast<int>(film->height));
				Sampler* localSampler = &samplers[threadId];
				for (int y = startY; y < endY; y++)
				{
					for (int x = startX; x < endX; x++)
					{
						float px = x + 0.5f;
						float py = y + 0.5f;

						Ray ray = scene->camera.generateRay(px, py);

						Colour pathThroughput(1.0f, 1.0f, 1.0f);
						Colour col = pathTrace(ray, pathThroughput, 5, localSampler);

						film->splat(px, py, col);

						unsigned char r = static_cast<unsigned char>(min(col.r, 1.0f) * 255);
						unsigned char g = static_cast<unsigned char>(min(col.g, 1.0f) * 255);
						unsigned char b = static_cast<unsigned char>(min(col.b, 1.0f) * 255);

						int globalIndex = (y * film->width + x) * 3;
						hdrpixels[globalIndex + 0] = r / 255.0f;
						hdrpixels[globalIndex + 1] = g / 255.0f;
						hdrpixels[globalIndex + 2] = b / 255.0f;
					}
				}
			};

		auto workerFunc = [&](int threadId)
			{
				for (int tileY = 0; tileY < numTilesY; tileY++)
				{
					for (int tileX = 0; tileX < numTilesX; tileX++)
					{
						if (((tileY * numTilesX) + tileX) % numThreads == threadId)
						{
							renderTile(tileX, tileY, threadId);
						}
					}
				}
			};

		for (int i = 0; i < numThreads; i++)
		{
			workers.emplace_back(workerFunc, i);
		}
		for (auto& w : workers) {
			w.join();
		}

		//Denoise
		oidn::DeviceRef device = oidn::newDevice();
		device.commit();
		oidn::BufferRef buffer = device.newBuffer(film->width * film->height * 3 * sizeof(float));
		std::memcpy(buffer.getData(), hdrpixels.data(), buffer.getSize());
		oidn::FilterRef filter = device.newFilter("RT");
		filter.setImage("color", buffer, oidn::Format::Float3, film->width, film->height);
		filter.setImage("output", buffer, oidn::Format::Float3, film->width, film->height);
		filter.set("hdr", true);
		filter.commit();
		filter.execute();
		// Store the output
		std::memcpy(hdrpixels.data(), buffer.getData(), buffer.getSize());

		// Draw final frame
		for (int y = 0; y < film->height; y++)
		{
			for (int x = 0; x < film->width; x++)
			{
				int index = (y * film->width + x) * 3;
				float r_value = hdrpixels[index + 0] * 255.0f;
				float g_value = hdrpixels[index + 1] * 255.0f;
				float b_value = hdrpixels[index + 2] * 255.0f;

				r_value = std::clamp(r_value, 0.0f, 255.0f);
				g_value = std::clamp(g_value, 0.0f, 255.0f);
				b_value = std::clamp(b_value, 0.0f, 255.0f);

				unsigned char r = static_cast<unsigned char>(r_value);
				unsigned char g = static_cast<unsigned char>(g_value);
				unsigned char b = static_cast<unsigned char>(b_value);

				canvas->draw(x, y, r, g, b);
			}
		}
	}

	void trender()
	{
		film->incrementSPP();
		for (unsigned int y = 0; y < film->height; y++)
		{
			for (unsigned int x = 0; x < film->width; x++)
			{
				float px = x + 0.5f;
				float py = y + 0.5f;
				Ray ray = scene->camera.generateRay(px, py);
				Sampler* localSampler = &samplers[0];
				Colour pathThroughput(1.0f, 1.0f, 1.0f);
				Colour col = pathTrace(ray, pathThroughput, 5, localSampler);
				//Colour col = viewNormals(ray);
				//Colour col = albedo(ray);
				film->splat(px, py, col);
				unsigned char r = (unsigned char)(col.r * 255);
				unsigned char g = (unsigned char)(col.g * 255);
				unsigned char b = (unsigned char)(col.b * 255);
				film->tonemap(px, py, r, g, b);
				canvas->draw(x, y, r, g, b);
			}
		}
	}
	int getSPP()
	{
		return film->SPP;
	}
	void saveHDR(std::string filename)
	{
		film->save(filename);
	}
	void savePNG(std::string filename)
	{
		stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3);
	}
};