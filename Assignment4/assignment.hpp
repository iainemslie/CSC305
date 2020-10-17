#pragma once

#include "paths.hpp"

#include <atlas/core/Float.hpp>
#include <atlas/core/Timer.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/math/Ray.hpp>
#include <atlas/utils/LoadObjFile.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <limits>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <iostream>

atlas::math::Normal face_normal;

using atlas::core::areEqual;

using Colour = atlas::math::Vector;

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image);

// Declarations
class BRDF;
class Camera;
class Material;
class Light;
class Shape;
class Sampler;
class BBox;
class Semaphore;
class Tracer;

struct World
{
	std::size_t width, height;
	Colour background;
	std::vector <std::shared_ptr<Shape>> scene;
	std::vector<Colour> image;
	std::vector<std::shared_ptr<Light>> lights;
	std::shared_ptr<Light> ambient;
	std::shared_ptr<Tracer> tracer;
	int maxDepth;
};

struct ShadeRec
{
	float t;
	atlas::math::Normal normal;
	atlas::math::Ray<atlas::math::Vector> ray;
	std::shared_ptr<Material> material;
	std::shared_ptr<World> world;
	atlas::math::Point hitPoint;
	float u;
	float v;
	atlas::math::Point areaSamplePoint;
	atlas::math::Normal areaLightNormal;
	atlas::math::Vector areaWi;
	atlas::math::Point occluderSamplePoint;
	atlas::math::Vector occluderU;
	atlas::math::Vector occluderV;
	atlas::math::Vector occluderW;
	int threadNum;
	int depth;
};

// Borrowed from https://riptutorial.com/cplusplus/example/30142/semaphore-cplusplus-11
class Semaphore
{
public:
	Semaphore::Semaphore(int count = 0) : mCount{ count }
	{}

	inline void Semaphore::notify()
	{
		std::unique_lock<std::mutex> lock(mMtx);
		mCount++;
		mCv.notify_one();
	}

	inline void Semaphore::wait()
	{
		std::unique_lock<std::mutex> lock(mMtx);
		while (mCount == 0) {
			mCv.wait(lock);
		}
		mCount--;
	}

private:
	std::mutex mMtx;
	std::condition_variable mCv;
	int mCount;
};

// Abstract classes defining the interfaces for concrete entities

class Image
{
public:

	Image::Image() : hres{ 100 }, vres{ 100 }
	{}

	int Image::getVres()
	{
		return vres;
	}

	int Image::getHres()
	{
		return hres;
	}

	void Image::readImageFile(std::string const& filename)
	{
		std::cout << "Reading Image: " << filename << std::endl;
		unsigned char* image = stbi_load(filename.c_str(), &hres, &vres, &channels, STBI_rgb);
		Colour pixel;
		std::cout << "Image Resolution: " << hres << " x " << vres << std::endl;
		pixels.reserve(hres * vres);

		for (std::size_t i{ 0 }, k{ 0 }; i < (hres * vres); ++i, k += 3)
		{
			pixel.r = static_cast<float>(image[k + 0]) / 255.0f;
			pixel.g = static_cast<float>(image[k + 1]) / 255.0f;
			pixel.b = static_cast<float>(image[k + 2]) / 255.0f;
			pixels.push_back(pixel);
		}

		stbi_image_free(image);
	}

	Colour Image::getColour(const int row, const int column) const
	{
		int index = column + hres * (vres - row - 1);
		size_t pixels_size = pixels.size();

		if (index < pixels_size)
			return pixels[index];
		else
			return { 1,0,0 };	//Return red for debugging
	}

private:
	int hres;
	int vres;
	int channels;
	std::vector<Colour> pixels;
};

class Tracer
{
public:
	Tracer::Tracer()
	{}

	Tracer::Tracer(std::shared_ptr<World> w) : world_ptr{ w }
	{}

	virtual Colour Tracer::traceRay(atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] const int depth, [[maybe_unused]] const int threadNum) const = 0;

protected:
	std::shared_ptr<World> world_ptr;
};

class Camera
{
public:
	Camera::Camera() :
		mEye{ 0.0f, 0.0f, 500.0f },
		mLookAt{ 0.0f },
		mUp{ 0.0f, 1.0f, 0.0f },
		mU{ 1.0f, 0.0f, 0.0f },
		mV{ 0.0f, 1.0f, 0.0f },
		mW{ 0.0f, 0.0f, 1.0f }
	{}

	virtual ~Camera() = default;

	virtual void renderScene(std::shared_ptr<World> world, std::shared_ptr<Sampler> sampler,
		std::size_t width_start, std::size_t width_end,
		std::size_t height_start, std::size_t height_end, int threadNum, std::shared_ptr<Semaphore> sem) const = 0;

	void Camera::setEye(atlas::math::Point const& eye)
	{
		mEye = eye;
	}

	void Camera::setLookAt(atlas::math::Point const& lookAt)
	{
		mLookAt = lookAt;
	}

	void Camera::setUpVector(atlas::math::Vector const& up)
	{
		mUp = up;
	}

	void Camera::computeUVW()
	{
		mW = glm::normalize(mEye - mLookAt);
		mU = glm::normalize(glm::cross(mUp, mW));
		mV = glm::cross(mW, mU);

		if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) &&
			mEye.y > mLookAt.y)
		{
			mU = { 0.0f, 0.0f, 1.0f };
			mV = { 1.0f, 0.0f, 0.0f };
			mW = { 0.0f, 1.0f, 0.0f };
		}

		if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) &&
			mEye.y < mLookAt.y)
		{
			mU = { 1.0f, 0.0f, 0.0f };
			mV = { 0.0f, 0.0f, 1.0f };
			mW = { 0.0f, -1.0f, 0.0f };
		}
	}

protected:
	atlas::math::Point mEye;
	atlas::math::Point mLookAt;
	atlas::math::Point mUp;
	atlas::math::Vector mU, mV, mW;
};

class Sampler
{
public:
	Sampler(size_t numSamples, size_t numSets) :
		mNumSamples{ numSamples }, mNumSets{ numSets }, mCount{ 0 }, mJump{ 0 }
	{
		mSamples.reserve(mNumSets* mNumSamples);
		setupShuffledIndices();
	}

	virtual ~Sampler() = default;

	size_t getNumSamples() const
	{
		return mNumSamples;
	}

	void setupShuffledIndices()
	{
		mShuffledIndices.reserve(mNumSamples * mNumSets);
		std::vector<int> indices;

		std::random_device d;
		std::mt19937 generator(d());

		for (int j = 0; j < mNumSamples; ++j)
		{
			indices.push_back(j);
		}

		for (int p = 0; p < mNumSets; ++p)
		{
			std::shuffle(indices.begin(), indices.end(), generator);

			for (int j = 0; j < mNumSamples; ++j)
			{
				mShuffledIndices.push_back(indices[j]);
			}
		}
	}

	virtual void generateSamples() = 0;

	atlas::math::Point sampleUnitSquare()
	{
		if (mCount % mNumSamples == 0)
		{
			atlas::math::Random<int> engine;
			mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
		}

		return mSamples[mJump + mShuffledIndices[mJump + (mCount++) % mNumSamples]];
	}

	atlas::math::Point sampleHemisphere()
	{
		if (mCount % mNumSamples == 0)
		{
			atlas::math::Random<int> engine;
			mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
		}

		return mHemisphereSamples[mJump + mShuffledIndices[mJump + (mCount++) % mNumSamples]];
	}

	atlas::math::Point2 sampleUnitDisk()
	{
		if (mCount % mNumSamples == 0)
		{
			atlas::math::Random<int> engine;
			mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
		}

		return mDiskSamples[mJump + mShuffledIndices[mJump + mCount++ % mNumSamples]];
	}

	void mapSamplesToHemisphere(const float exp)
	{
		size_t size = mSamples.size();
		mHemisphereSamples.reserve(mNumSamples * mNumSets);

		for (int j = 0; j < size; j++)
		{
			float cosPhi = glm::cos(glm::two_pi<float>() * mSamples[j].x);
			float sinPhi = glm::sin(glm::two_pi<float>() * mSamples[j].x);
			float cosTheta = glm::pow((1.0f - mSamples[j].y), 1.0f / (exp + 1.0f));
			float sinTheta = glm::sqrt(1.0f - cosTheta * cosTheta);
			float pu = sinTheta * cosPhi;
			float pv = sinTheta * sinPhi;
			float pw = cosTheta;

			mHemisphereSamples.push_back(atlas::math::Point{ pu,pv,pw });
		}
	}

	void mapSamplesToUnitDisk()
	{
		auto size{ mSamples.size() };
		float r, phi;
		atlas::math::Point sp;

		mDiskSamples.resize((int)size);

		for (int j = 0; j < size; ++j)
		{
			sp.x = 2.0f * mSamples[j].x - 1.0f;
			sp.y = 2.0f * mSamples[j].y - 1.0f;

			if (sp.x > -sp.y) {
				if (sp.x > sp.y) {
					r = sp.x;
					phi = sp.y / sp.x;
				}
				else {
					r = sp.y;
					phi = 2 - sp.x / sp.y;
				}
			}
			else {
				if (sp.x < sp.y) {
					r = -sp.x;
					phi = 4 + sp.y / sp.x;
				}
				else {
					r = -sp.y;
					if (!atlas::core::areEqual(sp.y, 0.0f))
						phi = 6 - sp.x / sp.y;
					else
						phi = 0.0f;
				}
			}
			phi *= glm::pi<float>() / 4.0f;

			mDiskSamples[j].x = r * glm::cos(phi);
			mDiskSamples[j].y = r * glm::sin(phi);
		}
		mSamples.erase(mSamples.begin(), mSamples.end());
	}

protected:
	std::vector<atlas::math::Point> mSamples;
	std::vector<atlas::math::Point2> mDiskSamples;
	std::vector<atlas::math::Point> mHemisphereSamples;
	std::vector<int> mShuffledIndices;

	size_t mNumSamples;
	size_t mNumSets;
	unsigned long mCount;
	size_t mJump;
};

class BBox
{
public:
	BBox::BBox() : mP0{ -1, -1, -1 }, mP1{ 1, 1, 1 }
	{}

	BBox::BBox(atlas::math::Point p0, atlas::math::Point p1) : mP0{ p0 }, mP1{ p1 }
	{}

	BBox::BBox(float x0, float x1, float y0, float y1, float z0, float z1)
	{
		mP0.x = x0;
		mP0.y = y0;
		mP0.z = z0;
		mP1.x = x1;
		mP1.y = y1;
		mP1.z = z1;
	}

	BBox::~BBox() = default;

	bool BBox::hit(atlas::math::Ray<atlas::math::Vector> const& ray) const
	{
		float ox = ray.o.x, oy = ray.o.y, oz = ray.o.z;
		float dx = ray.d.x, dy = ray.d.y, dz = ray.d.z;

		float tx_min, ty_min, tz_min;
		float tx_max, ty_max, tz_max;

		float a = 1.0f / dx;
		if (a >= 0.0f)
		{
			tx_min = (mP0.x - ox) * a;
			tx_max = (mP1.x - ox) * a;
		}
		else
		{
			tx_min = (mP1.x - ox) * a;
			tx_max = (mP0.x - ox) * a;
		}

		float b = 1.0f / dy;
		if (b >= 0.0f)
		{
			ty_min = (mP0.y - oy) * b;
			ty_max = (mP1.y - oy) * b;
		}
		else
		{
			ty_min = (mP1.y - oy) * b;
			ty_max = (mP0.y - oy) * b;
		}

		float c = 1.0f / dz;
		if (c >= 0.0f)
		{
			tz_min = (mP0.z - oz) * c;
			tz_max = (mP1.z - oz) * c;
		}
		else
		{
			tz_min = (mP1.z - oz) * c;
			tz_max = (mP0.z - oz) * c;
		}

		float t0, t1;

		// find largest entering t value
		if (tx_min > ty_min)
			t0 = tx_min;
		else
			t0 = ty_min;

		if (tz_min > t0)
			t0 = tz_min;

		// find smallest exiting t value
		if (tx_max < ty_max)
			t1 = tx_max;
		else
			t1 = ty_max;

		if (tz_max < t1)
			t1 = tz_max;

		return (t0 < t1 && t1 > kEpsilon);
	}

	bool BBox::inside(const atlas::math::Point& p) const
	{
		return ((p.x > mP0.x&& p.x < mP1.x) && (p.y > mP0.y&& p.y < mP1.y) && (p.z > mP0.z&& p.z < mP1.z));
	}

	atlas::math::Point mP0;
	atlas::math::Point mP1;

private:
	float kEpsilon{ 0.001f };
};

class Shape
{
public:
	Shape::Shape()
	{}

	virtual ~Shape() = default;

	// if t computed is less than the t in sr, it and the color should be updated in sr
	virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const = 0;

	virtual bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const = 0;

	virtual void Shape::setMaterial(std::shared_ptr<Material> const& material) const
	{
		mMaterial = material;
	}

	virtual std::shared_ptr<Material> Shape::getMaterial() const
	{
		return mMaterial;
	}

	virtual void Shape::addObject(std::shared_ptr<Shape> object)
	{}

	void Shape::setBoundingBox()
	{}

	virtual BBox Shape::getBoundingBox() const = 0;

	virtual void setShadows(bool shadows)
	{
		mShadows = shadows;
	}

	virtual bool Shape::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const = 0;

	// For area lights

	virtual atlas::math::Point Shape::sample([[maybe_unused]] const ShadeRec& sr) const
	{
		return { 0,0,0 };
	}

	virtual float Shape::pdf([[maybe_unused]] const ShadeRec& sr) const
	{
		return 0.0f;
	}

	virtual atlas::math::Normal Shape::getNormal([[maybe_unused]] const atlas::math::Point& p) const
	{
		return atlas::math::Normal();
	}

protected:
	bool mShadows;
	BBox mBBox;
	mutable std::shared_ptr<Material> mMaterial;

};

class Instance : public Shape
{
public:

	Instance::Instance() : object_ptr{ nullptr }, mBBox{}, transform_the_texture{ false }
	{
		inv_matrix = glm::mat4(1.0f);
		forward_matrix = glm::mat4(1.0f);
	}

	Instance::Instance(const std::shared_ptr<Shape> obj_ptr) : object_ptr{ obj_ptr }, inv_matrix{}, mBBox{},
		transform_the_texture{ false }
	{
		inv_matrix = glm::mat4(1.0f);
		forward_matrix = glm::mat4(1.0f);
	}

	void Instance::setObject(std::shared_ptr<Shape> obj_ptr)
	{
		object_ptr = obj_ptr;
	}

	std::shared_ptr<Material> Instance::getMaterial() const
	{
		return object_ptr->getMaterial();
	}

	void Instance::setMaterial(std::shared_ptr<Material> m_ptr)
	{
		// This is wrong
		object_ptr->setMaterial(m_ptr);
		mMaterial = m_ptr;
	}

	BBox Instance::getBoundingBox() const
	{
		return mBBox;
	}

	void Instance::computeBoundingBox()
	{
		BBox objectBBox = object_ptr->getBoundingBox();

		std::vector<atlas::math::Point4> vertices;
		vertices.reserve(8);
		vertices[0].x = objectBBox.mP0.x; vertices[0].z = objectBBox.mP0.z; vertices[0].z = objectBBox.mP0.z; vertices[0].w = 1;
		vertices[1].x = objectBBox.mP0.x; vertices[1].z = objectBBox.mP0.z; vertices[1].z = objectBBox.mP0.z; vertices[1].w = 1;
		vertices[2].x = objectBBox.mP0.x; vertices[2].z = objectBBox.mP0.z; vertices[2].z = objectBBox.mP0.z; vertices[2].w = 1;
		vertices[3].x = objectBBox.mP0.x; vertices[3].z = objectBBox.mP0.z; vertices[3].z = objectBBox.mP0.z; vertices[3].w = 1;

		vertices[4].x = objectBBox.mP0.x; vertices[4].z = objectBBox.mP0.z; vertices[4].z = objectBBox.mP0.z; vertices[4].w = 1;
		vertices[5].x = objectBBox.mP0.x; vertices[5].z = objectBBox.mP0.z; vertices[5].z = objectBBox.mP0.z; vertices[5].w = 1;
		vertices[6].x = objectBBox.mP0.x; vertices[6].z = objectBBox.mP0.z; vertices[6].z = objectBBox.mP0.z; vertices[6].w = 1;
		vertices[7].x = objectBBox.mP0.x; vertices[7].z = objectBBox.mP0.z; vertices[7].z = objectBBox.mP0.z; vertices[7].w = 1;

		vertices[0] = forward_matrix * vertices[0];
		vertices[1] = forward_matrix * vertices[1];
		vertices[2] = forward_matrix * vertices[2];
		vertices[3] = forward_matrix * vertices[3];
		vertices[4] = forward_matrix * vertices[4];
		vertices[5] = forward_matrix * vertices[5];
		vertices[6] = forward_matrix * vertices[6];
		vertices[7] = forward_matrix * vertices[7];

		forward_matrix = glm::mat4(1.0f);

		float x0 = std::numeric_limits<float>::max();
		float y0 = std::numeric_limits<float>::max();
		float z0 = std::numeric_limits<float>::max();

		for (int j = 0; j <= 7; j++) {
			if (vertices[j].x < x0)
				x0 = vertices[j].x;
		}

		for (int j = 0; j <= 7; j++) {
			if (vertices[j].y < y0)
				y0 = vertices[j].y;
		}

		for (int j = 0; j <= 7; j++) {
			if (vertices[j].z < z0)
				z0 = vertices[j].z;
		}

		// Compute the minimum values

		float x1 = -std::numeric_limits<float>::max();
		float y1 = -std::numeric_limits<float>::max();
		float z1 = -std::numeric_limits<float>::max();

		for (int j = 0; j <= 7; j++) {
			if (vertices[j].x > x1)
				x1 = vertices[j].x;
		}

		for (int j = 0; j <= 7; j++) {
			if (vertices[j].y > y1)
				y1 = vertices[j].y;
		}

		for (int j = 0; j <= 7; j++) {
			if (vertices[j].z > z1)
				z1 = vertices[j].z;
		}

		// Assign values to the bounding box

		mBBox.mP0.x = x0;
		mBBox.mP0.y = y0;
		mBBox.mP0.z = z0;
		mBBox.mP1.x = x1;
		mBBox.mP1.y = y1;
		mBBox.mP1.z = z1;
	}

	void Instance::scale(atlas::math::Vector s)
	{
		glm::mat4 inv_scaling_matrix = glm::mat4(1.0f);
		inv_scaling_matrix = glm::affineInverse(glm::scale(inv_scaling_matrix, { s.x, s.y, s.z }));
		inv_matrix = inv_matrix * inv_scaling_matrix;

		glm::mat4 scaling_matrix = glm::mat4(1.0f);
		scaling_matrix = glm::scale(inv_scaling_matrix, { s.x, s.y, s.z });
		forward_matrix = scaling_matrix * forward_matrix;
	}

	void Instance::translate(atlas::math::Vector s)
	{
		glm::mat4 inv_translation_matrix = glm::mat4(1.0f);
		inv_translation_matrix = glm::affineInverse(glm::translate(inv_translation_matrix, { s.x, s.y, s.z }));
		inv_matrix = inv_matrix * inv_translation_matrix;

		glm::mat4 translation_matrix = glm::mat4(1.0f);
		translation_matrix = glm::translate(inv_translation_matrix, { s.x, s.y, s.z });
		forward_matrix = translation_matrix * forward_matrix;
	}

	void Instance::rotateX(const float theta)
	{
		glm::mat4 inv_rotation_matrix = glm::mat4(1.0f);
		inv_rotation_matrix = glm::affineInverse(glm::rotate(inv_rotation_matrix, glm::radians(theta), glm::vec3(1.0f, 0.0f, 0.0f)));
		inv_matrix = inv_matrix * inv_rotation_matrix;

		glm::mat4 rotation_matrix = glm::mat4(1.0f);
		rotation_matrix = glm::rotate(rotation_matrix, glm::radians(theta), glm::vec3(1.0f, 0.0f, 0.0f));
		forward_matrix = rotation_matrix * forward_matrix;
	}

	void Instance::rotateY(const float theta)
	{
		glm::mat4 inv_rotation_matrix = glm::mat4(1.0f);
		inv_rotation_matrix = glm::affineInverse(glm::rotate(inv_rotation_matrix, glm::radians(theta), glm::vec3(0.0f, 1.0f, 0.0f)));
		inv_matrix = inv_matrix * inv_rotation_matrix;

		glm::mat4 rotation_matrix = glm::mat4(1.0f);
		rotation_matrix = glm::rotate(rotation_matrix, glm::radians(theta), glm::vec3(0.0f, 1.0f, 0.0f));
		forward_matrix = rotation_matrix * forward_matrix;
	}

	void Instance::rotateZ(const float theta)
	{
		glm::mat4 inv_rotation_matrix = glm::mat4(1.0f);
		inv_rotation_matrix = glm::affineInverse(glm::rotate(inv_rotation_matrix, glm::radians(theta), glm::vec3(0.0f, 0.0f, 1.0f)));
		inv_matrix = inv_matrix * inv_rotation_matrix;

		glm::mat4 rotation_matrix = glm::mat4(1.0f);
		rotation_matrix = glm::rotate(rotation_matrix, glm::radians(theta), glm::vec3(0.0f, 0.0f, 1.0f));
		forward_matrix = rotation_matrix * forward_matrix;
	}

	void Instance::rotate(const float theta, glm::vec3 axes)
	{
		glm::mat4 inv_rotation_matrix = glm::mat4(1.0f);
		inv_rotation_matrix = glm::affineInverse(glm::rotate(inv_rotation_matrix, glm::radians(theta), axes));
		inv_matrix = inv_matrix * inv_rotation_matrix;

		glm::mat4 rotation_matrix = glm::mat4(1.0f);
		rotation_matrix = glm::rotate(rotation_matrix, glm::radians(theta), glm::vec3(0.0f, 0.0f, 1.0f));
		forward_matrix = rotation_matrix * forward_matrix;
	}

	bool Instance::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const
	{
		atlas::math::Ray<atlas::math::Vector4> inv_ray4;
		inv_ray4.o = glm::vec4(ray.o, 1.0);
		inv_ray4.d = glm::vec4(ray.d, 0.0);
		inv_ray4.o = inv_matrix * inv_ray4.o;
		inv_ray4.d = inv_matrix * inv_ray4.d;

		atlas::math::Ray<atlas::math::Vector> inv_ray3;
		inv_ray3.o = glm::vec3(inv_ray4.o.x, inv_ray4.o.y, inv_ray4.o.z);
		inv_ray3.d = glm::vec3(inv_ray4.d.x, inv_ray4.d.y, inv_ray4.d.z);

		if (object_ptr->hit(inv_ray3, sr))
		{
			atlas::math::Normal4 temp_norm4(sr.normal, 0.0);
			glm::mat4 tmp_inv = glm::transpose(inv_matrix);
			temp_norm4 = tmp_inv * temp_norm4;
			atlas::math::Normal temp_norm3{ temp_norm4.x, temp_norm4.y, temp_norm4.z };
			sr.normal = glm::normalize(temp_norm3);

			if (object_ptr->getMaterial() != nullptr) {
				mMaterial = object_ptr->getMaterial();
				sr.material = mMaterial;
			}
			if (mMaterial) {
				sr.material = mMaterial;
			}
			//else {
			//	sr.material = mMaterial;
			//}

			if (!transform_the_texture)
				sr.hitPoint = ray.o + sr.t * ray.d;

			return true;
		}
		return false;
	}

	void Instance::transformTexture(const bool transform)
	{
		transform_the_texture = transform;
	}

	bool Instance::shadowHit([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] float& tMin) const
	{
		atlas::math::Ray<atlas::math::Vector4> inv_ray4;
		inv_ray4.o = glm::vec4(ray.o, 1.0);
		inv_ray4.d = glm::vec4(ray.d, 0.0);
		inv_ray4.o = inv_matrix * inv_ray4.o;
		inv_ray4.d = inv_matrix * inv_ray4.d;

		atlas::math::Ray<atlas::math::Vector> inv_ray3;
		inv_ray3.o = glm::vec3(inv_ray4.o.x, inv_ray4.o.y, inv_ray4.o.z);
		inv_ray3.d = glm::vec3(inv_ray4.d.x, inv_ray4.d.y, inv_ray4.d.z);

		if (object_ptr->shadowHit(inv_ray3, tMin))
		{
			return true;
		}
		return false;
	}

	bool Instance::intersectRay([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] float& tMin) const
	{
		return false;
	}

private:
	std::shared_ptr<Shape> object_ptr;
	glm::mat4 inv_matrix;
	inline static glm::mat4 Instance::forward_matrix;
	mutable BBox mBBox;
	bool transform_the_texture;
};

class BRDF
{
public:
	virtual ~BRDF() = default;

	virtual Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const = 0;

	virtual Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const = 0;

	virtual Colour BRDF::sample_f([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector& incoming,
		[[maybe_unused]] float& pdf) const
	{
		return { 0,0,0 };
	}

	virtual Colour BRDF::sample_f([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector& incoming) const
	{
		return { 0,0,0 };
	}
};

class BTDF
{
public:

	virtual ~BTDF() = default;

	virtual Colour BTDF::fn([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector const& incoming) const
	{
		return { 0,0,0 };
	}

	virtual Colour BTDF::sample_f([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector& incoming) const
	{
		return { 0,0,0 };
	}

	virtual Colour BTDF::rho([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected) const
	{
		return { 0,0,0 };
	}

	virtual bool tir([[maybe_unused]] ShadeRec const& sr) const = 0;

};

class Material
{
public:
	virtual ~Material() = default;

	virtual Colour Material::shade([[maybe_unused]] ShadeRec& sr) = 0;

	virtual Colour Material::getEmittedRadiance([[maybe_unused]] ShadeRec& sr)
	{
		return { 0,0,0 };
	}

	virtual Colour Material::areaLightShade([[maybe_unused]] ShadeRec& sr)
	{
		return { 0,0,0 };
	}
};

/****************************************************************/
/*					TEXTURE MAPPINGS							*/
/***************************************************************/

class Mapping
{
public:
	Mapping::Mapping()
	{}

	virtual void getTexelCoordinates(const atlas::math::Point localHitPoint,
		const int hres, const int vres, int& row, int& column) const = 0;
};

class RectangularMap : public Mapping
{
public:
	RectangularMap::RectangularMap()
	{}

	void RectangularMap::getTexelCoordinates(const atlas::math::Point localHitPoint,
		const int hres, const int vres, int& row, int& column) const
	{
		float u = (localHitPoint.z + 1) / 2;
		float v = (localHitPoint.x + 1) / 2;

		column = (int)((hres - 1) * u);
		row = (int)((vres - 1) * v);
	}
};

class SphericalMap : public Mapping
{
public:
	SphericalMap::SphericalMap()
	{}

	void SphericalMap::getTexelCoordinates(const atlas::math::Point localHitPoint,
		const int hres, const int vres, int& row, int& column) const
	{
		// first compute theta and phi
		float theta = glm::acos(localHitPoint.y);
		float phi = glm::atan(localHitPoint.x, localHitPoint.z);

		if (phi < 0.0)
			phi += glm::two_pi<float>();

		// map theta and pi to (u, v)
		float u = phi * glm::one_over_two_pi<float>();
		float v = 1.0f - theta * glm::one_over_pi<float>();

		// map u and v to the texel coords
		column = (int)((hres - 1) * u);
		row = (int)((vres - 1) * v);
	}
};

class CylinderMap : public Mapping
{
public:
	CylinderMap::CylinderMap()
	{}

	void CylinderMap::getTexelCoordinates(const atlas::math::Point localHitPoint,
		const int hres, const int vres, int& row, int& column) const
	{
		float phi = glm::atan(localHitPoint.x / localHitPoint.z);

		float u = phi / glm::two_pi<float>();
		float v = (localHitPoint.y + 1) / 2.0f;

		// map u and v to the texel coords
		column = (int)((hres - 1) * u);
		row = (int)((vres - 1) * v);
	}
};

/****************************************************************/
/*						 TEXTURES								*/
/***************************************************************/

class Texture
{
public:
	// constructors, etc.
	Texture::Texture()
	{}

	virtual Colour getColour(const ShadeRec& sr) const = 0;
};

class ImageTexture : public Texture
{
public:

	ImageTexture::ImageTexture() : Texture(), hres{ 100 }, vres{ 100 },
		image_ptr{ nullptr }, mapping_ptr{ nullptr }
	{}

	ImageTexture::ImageTexture(std::shared_ptr<Image> image) : Texture(),
		hres{ image->getHres() }, vres{ image->getVres() }, image_ptr{ image }, mapping_ptr{ nullptr }
	{}

	Colour ImageTexture::getColour([[maybe_unused]] const ShadeRec& sr) const
	{
		int row;
		int column;

		if (mapping_ptr != nullptr)
			mapping_ptr->getTexelCoordinates(sr.hitPoint, hres, vres, row, column);
		else {
			row = (int)(sr.v * (vres - 1));
			column = (int)(sr.u * (hres - 1));
		}

		return image_ptr->getColour(row, column);
	}

	void ImageTexture::setImage(std::shared_ptr<Image> image)
	{
		image_ptr = image;
		hres = image->getHres();
		vres = image->getVres();
	}

	void ImageTexture::setMapping(std::shared_ptr<Mapping> mapping)
	{
		mapping_ptr = mapping;
	}

private:
	int hres;
	int vres;
	std::shared_ptr<Image> image_ptr;
	std::shared_ptr<Mapping> mapping_ptr;
};

class ConstantTexture : public Texture
{
public:

	// constructors, etc.

	void ConstantTexture::setColour(Colour const& c)
	{
		mColour = c;
	}

	Colour ConstantTexture::getColour([[maybe_unused]] const ShadeRec& sr) const
	{
		return mColour;
	}

private:
	Colour mColour;
};

class Light
{
public:
	virtual atlas::math::Vector getDirection([[maybe_unused]] ShadeRec& sr) const = 0;

	virtual bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const = 0;

	virtual bool castsShadows() const = 0;

	virtual Colour Light::L([[maybe_unused]] ShadeRec& sr) const
	{
		return mRadiance * mColour;
	}

	void Light::scaleRadiance(float b)
	{
		mRadiance = b;
	}

	void Light::setColour(Colour const& c)
	{
		mColour = c;
	}

	void Light::setShadows(bool shadows)
	{
		mShadows = shadows;
	}

	virtual float Light::getDirection([[maybe_unused]] const ShadeRec& sr) const
	{
		return 1.0f;
	}

	virtual float Light::G([[maybe_unused]] const ShadeRec& sr) const
	{
		return 1.0f;
	}

	virtual float Light::pdf([[maybe_unused]] const ShadeRec& sr) const
	{
		return 1.0f;
	}

protected:
	bool mShadows;
	Colour mColour;
	float mRadiance;
};

/****************************************************************/
/*						  SHAPES								*/
/***************************************************************/

class Compound : public Shape
{
public:
	Compound::Compound()
	{}

	void Compound::setMaterial(std::shared_ptr<Material> const& material)
	{
		for (auto object : objects)
		{
			object->setMaterial(material);
		}
	}

	void Compound::addObject(std::shared_ptr<Shape> object)
	{
		objects.push_back(object);
	}

	void Compound::setBoundingBox(atlas::math::Point p0, atlas::math::Point p1)
	{
		float kEpsilon = 0.0000001f;
		mBBox.mP0 = p0 - kEpsilon;
		mBBox.mP1 = p1 + kEpsilon;
	}

	BBox Compound::getBoundingBox() const
	{
		return mBBox;
	}

	bool Compound::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const
	{
		//LOOK AT ADDING THIS BACK ***!!!!
		//if (!mBBox.hit(ray))
		//	return false;

		bool hit{};

		for (auto const& obj : objects)
		{
			hit |= obj->hit(ray, sr);
		}
		return hit;
	}

	bool Compound::shadowHit([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] float& tMin) const
	{
		if (!mShadows)
			return false;

		bool hit{};

		for (auto const& obj : objects)
		{
			hit |= obj->shadowHit(ray, tMin);
		}
		return hit;
	}

	void setShadows(bool shadows)
	{
		mShadows = shadows;
		for (auto obj : objects)
		{
			obj->setShadows(shadows);
		}
	}

	bool intersectRay([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] float& tMin) const
	{
		return false;
	}

protected:
	std::vector<std::shared_ptr<Shape>> objects;
};

class Sphere : public Shape
{
public:
	Sphere::Sphere() : mCentre{ 0,0,0 }, mRadius{ 1.0f }, mRadiusSqr{ 1.0f }
	{}

	Sphere::Sphere(atlas::math::Point center, float radius) :
		mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
	{}

	void Sphere::setBoundingBox(atlas::math::Point p0, atlas::math::Point p1)
	{
		mBBox.mP0 = p0;
		mBBox.mP1 = p1;
	}

	BBox Sphere::getBoundingBox() const
	{
		return BBox{
			{ mCentre.x - mRadius, mCentre.y - mRadius, mCentre.z - mRadius },
			{ mCentre.x + mRadius, mCentre.y + mRadius, mCentre.z + mRadius }
		};
	}

	bool Sphere::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const
	{
		atlas::math::Vector tmp = ray.o - mCentre;
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.normal = (tmp + t * ray.d) / mRadius;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}

		return intersect;
	}

	bool Sphere::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	bool Sphere::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const
	{
		const auto tmp{ ray.o - mCentre };
		const auto a{ glm::dot(ray.d, ray.d) };
		const auto b{ 2.0f * glm::dot(ray.d, tmp) };
		const auto c{ glm::dot(tmp, tmp) - mRadiusSqr };
		const auto disc{ (b * b) - (4.0f * a * c) };

		if (atlas::core::geq(disc, 0.0f))
		{
			const float e{ std::sqrt(disc) };
			const float denom{ 2.0f * a };

			// Look at the negative root first
			float t = (-b - e) / denom;
			if (atlas::core::geq(t, kEpsilon))
			{
				tMin = t;
				return true;
			}

			// Now the positive root
			t = (-b + e);
			if (atlas::core::geq(t, kEpsilon))
			{
				tMin = t;
				return true;
			}
		}
		return false;
	}

private:
	atlas::math::Point mCentre;
	float mRadius;
	float mRadiusSqr;
	const float kEpsilon{ 0.01f };
};

class Plane : public Shape
{
public:
	Plane(atlas::math::Point point, atlas::math::Normal normal) :
		mPoint{ point }, mNormal{ normal }
	{}

	BBox Plane::getBoundingBox() const
	{
		return BBox{
			{ 0,0,0 },
			{ 0,0,0 }
		};
	}

	bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}

		return intersect;
	}

	bool Plane::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		float t{ glm::dot(mPoint - ray.o, mNormal) / glm::dot(ray.d, mNormal) };

		if (t > kEpsilon)
		{
			tMin = t;
			return true;
		}
		else
		{
			return false;
		}
	}

private:
	atlas::math::Point mPoint;
	atlas::math::Normal mNormal;
	const float kEpsilon{ 0.1f };
};

class Disk : public Shape
{
public:
	Disk::Disk(atlas::math::Point point, atlas::math::Normal normal, float radius)
		: mCentre{ point }, mNormal{ normal }, mRadius{ radius }
	{}

	BBox Disk::getBoundingBox() const
	{
		return BBox{
			{ 0,0,0 },
			{ 0,0,0 }
		};
	}

	bool Disk::hit(atlas::math::Ray <atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}

		return intersect;
	}

	bool Disk::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	bool Disk::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		float t{ glm::dot(mCentre - ray.o, mNormal) / glm::dot(ray.d, mNormal) };

		if (t <= kEpsilon)
			return false;

		atlas::math::Point p = ray.o + t * ray.d;

		float distance = glm::distance(mCentre, p);

		if ((distance * distance) < (mRadius * mRadius))
		{
			tMin = t;
			return true;
		}
		return false;
	}

private:
	atlas::math::Point mCentre;
	atlas::math::Normal mNormal;
	float mRadius;
	float kEpsilon{ 0.1f };
};

class Rectangle : public Shape
{
public:
	Rectangle::Rectangle(atlas::math::Point p0, atlas::math::Vector a, atlas::math::Vector b)
		: mP0{ p0 }, mA{ a }, mB{ b },
		mALenSquared{ glm::dot(a,a) * glm::dot(a,a) }, mBLenSquared{ glm::dot(b,b) * glm::dot(b,b) },
		mArea{ glm::length(a) * glm::length(b) }, mInvArea{ 1.0f / mArea }
	{
		mNormal = mNormal = glm::normalize(glm::cross(a, b));
	}

	Rectangle::Rectangle(atlas::math::Point p0, atlas::math::Vector a, atlas::math::Vector b, atlas::math::Normal normal)
		: mP0{ p0 }, mA{ a }, mB{ b }, mNormal{ normal },
		mALenSquared{ glm::dot(a,a) * glm::dot(a,a) }, mBLenSquared{ glm::dot(b,b) * glm::dot(b,b) },
		mArea{ glm::length(a) * glm::length(b) }, mInvArea{ 1.0f / mArea }
	{}

	// Generic Rectangle
	Rectangle::Rectangle() : mP0{ -1,0,-1 }, mA{ 0,0,1 }, mB{ 1,0,0 }, mNormal{ 0,1,0 },
		mArea{ 1.0f }, mInvArea{ 1.0f }, mALenSquared{ glm::dot(mA, mA) },
		mBLenSquared{ glm::dot(mB, mB) }
	{}

	BBox Rectangle::getBoundingBox() const
	{
		float delta = 0.0001f;
		return BBox{
			  { glm::min(mP0.x, mP0.x + mA.x + mB.x) - delta,
				glm::min(mP0.y, mP0.y + mA.y + mB.y) - delta,
				glm::min(mP0.z, mP0.z + mA.z + mB.z) - delta },
			  { glm::max(mP0.x, mP0.x + mA.x + mB.x) + delta,
				glm::max(mP0.y, mP0.y + mA.y + mB.y) + delta,
				glm::max(mP0.z, mP0.z + mA.z + mB.z) + delta }
		};
	}

	bool Rectangle::hit(atlas::math::Ray <atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}

		return intersect;
	}

	bool Rectangle::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	void Rectangle::setSampler(std::vector<std::shared_ptr<Sampler>> const& occludersamplerVec)
	{
		samplerVec = occludersamplerVec;
	}

	atlas::math::Normal Rectangle::getNormal([[maybe_unused]] const atlas::math::Point& p) const
	{
		return mNormal;
	}

	float Rectangle::pdf([[maybe_unused]] const ShadeRec& sr) const override
	{
		return mInvArea;
	}

	atlas::math::Point Rectangle::sample([[maybe_unused]] const ShadeRec& sr) const
	{
		atlas::math::Point2 samplePoint = samplerVec.at(sr.threadNum)->sampleUnitSquare();
		return (mP0 + samplePoint.x * mA + samplePoint.y * mB);
	}

	bool Rectangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		float t = glm::dot((mP0 - ray.o), mNormal) / glm::dot(ray.d, mNormal);

		if (t <= kEpsilon)
			return false;

		atlas::math::Point p = ray.o + t * ray.d;
		atlas::math::Vector d = p - mP0;

		float ddota = glm::dot(d, mA);

		if (ddota < 0.0f || ddota > mALenSquared)
			return false;

		float ddotb = glm::dot(d, mB);

		if (ddotb < 0.0f || ddotb > mBLenSquared)
			return false;

		tMin = t;
		return true;
	}

private:
	atlas::math::Point mP0;
	atlas::math::Vector mA;
	atlas::math::Vector mB;
	atlas::math::Normal mNormal;
	float mALenSquared;
	float mBLenSquared;
	float kEpsilon = 0.01f;
	std::vector<std::shared_ptr<Sampler>> samplerVec;
	float mArea;
	float mInvArea;
};

class OpenCylinder : public Shape
{
public:
	OpenCylinder::OpenCylinder() : mY0{ -1.0f }, mY1{ 1.0f }, mRadius{ 1.0f }, mInvRadius{ 1.0f }
	{}

	OpenCylinder::OpenCylinder(const float bottom, const float top, const float radius)
		: mY0{ bottom }, mY1{ top }, mRadius{ radius }, mInvRadius{ 1.0f / radius }
	{}

	BBox OpenCylinder::getBoundingBox() const
	{
		return BBox{};
	}

	bool OpenCylinder::hit(atlas::math::Ray <atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = mHitPoint;
		}

		return intersect;
	}

	bool OpenCylinder::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	bool OpenCylinder::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		float t;
		float ox = ray.o.x; float oy = ray.o.y; float oz = ray.o.z;
		float dx = ray.d.x; float dy = ray.d.y; float dz = ray.d.z;

		float a = dx * dx + dz * dz;
		float b = 2.0f * (ox * dx + oz * dz);
		float c = ox * ox + oz * oz - mRadius * mRadius;
		float disc = b * b - 4.0f * a * c;

		if (disc < 0.0f)
			return false;
		else
		{
			float e = glm::sqrt(disc);
			float denom = 2.0f * a;
			t = (-b - e) / denom;

			if (t > kEpsilon)
			{
				float yhit = oy + t * dy;

				if (yhit > mY0&& yhit < mY1)
				{
					tMin = t;
					mNormal = atlas::math::Normal((ox + t * dx) * mInvRadius, 0.0f, (oz + t * dz) * mInvRadius);

					if (glm::dot(-ray.d, mNormal) < 0.0f)
						mNormal = -mNormal;

					return true;
				}
			}
		}
		return false;
	}

protected:
	float mY0;
	float mY1;
	float mRadius;
	float mInvRadius;
	mutable atlas::math::Normal mNormal;
	mutable atlas::math::Point mHitPoint;
	float kEpsilon = 0.35f;
};

class SolidCylinder : public Compound
{
public:

	SolidCylinder::SolidCylinder()
	{}

	SolidCylinder::SolidCylinder(const float bottom, const float top, const float radius) : Compound{}
	{
		objects.push_back(std::make_shared<Disk>(atlas::math::Point{ 0, bottom, 0 },
			atlas::math::Normal{ 0, -1, 0 }, radius));
		objects.push_back(std::make_shared<Disk>(atlas::math::Point{ 0, top, 0 },
			atlas::math::Normal{ 0, 1, 0 }, radius));
		objects.push_back(std::make_shared<OpenCylinder>(bottom, top, radius));
	}

	bool SolidCylinder::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		if (!mBBox.hit(ray))
			return false;

		bool hit{};

		for (auto const& obj : objects)
		{
			hit |= obj->hit(ray, sr);
		}
		return hit;
	}
};

class Triangle : public Shape
{
public:
	Triangle(atlas::math::Point v0, atlas::math::Point v1, atlas::math::Point v2) : mV0{ v0 }, mV1{ v1 }, mV2{ v2 }
	{
		mNormal = glm::normalize(glm::cross(mV1 - mV0, mV2 - mV0));
	}

	void Triangle::setBoundingBox(atlas::math::Point p0, atlas::math::Point p1)
	{
		mBBox.mP0 = p0;
		mBBox.mP1 = p1;
	}

	BBox Triangle::getBoundingBox() const
	{
		float delta = 0.00001f;
		return BBox{
			atlas::math::Point{
				glm::min(glm::min(mV0.x, mV1.x), mV2.x) - delta,
				glm::min(glm::min(mV0.y, mV1.y), mV2.y) - delta,
				glm::min(glm::min(mV0.z, mV1.z), mV2.z) - delta,
			},
			atlas::math::Point{
				glm::max(glm::max(mV0.x, mV1.x), mV2.x) + delta,
				glm::max(glm::max(mV0.y, mV1.y), mV2.y) + delta,
				glm::max(glm::max(mV0.z, mV1.z), mV2.z) + delta,
			}
		};
	}

	bool Triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}
		return intersect;
	}

	bool Triangle::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		const float a = mV0.x - mV1.x, b = mV0.x - mV2.x, c = ray.d.x, d = mV0.x - ray.o.x;
		const float e = mV0.y - mV1.y, f = mV0.y - mV2.y, g = ray.d.y, h = mV0.y - ray.o.y;
		const float i = mV0.z - mV1.z, j = mV0.z - mV2.z, k = ray.d.z, l = mV0.z - ray.o.z;

		const float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
		const float q = g * i - e * k, s = e * j - f * i;

		const float inv_denom = 1.0f / (a * m + b * q + c * s);

		const float e1 = d * m - b * n - c * p;
		const float beta = e1 * inv_denom;

		if (beta < 0.0f)
			return false;

		const float r = e * l - h * i;
		const float e2 = a * n + d * q + c * r;
		const float gamma = e2 * inv_denom;

		if (gamma < 0.0f)
			return false;

		if (beta + gamma > 1.0f)
			return false;

		const float e3 = a * p - b * r + d * s;
		const float t = e3 * inv_denom;

		if (t < kEpsilon)
			return false;

		tMin = t;
		return true;
	}

private:
	atlas::math::Point mV0;
	atlas::math::Point mV1;
	atlas::math::Point mV2;
	atlas::math::Normal mNormal;
	const float kEpsilon{ 0.0000001f };
};

class MeshTriangle : public Shape
{
public:
	// Default constructor
	MeshTriangle::MeshTriangle() :
		mMesh_ptr{ NULL }, mIndex0{ 0 }, mIndex1{ 0 }, mIndex2{ 0 }, mNormal{ 0,0,0 }, mShapeIndex{ 0 }
	{}

	MeshTriangle::MeshTriangle(std::shared_ptr<atlas::utils::ObjMesh> mesh_ptr,
		std::size_t index0, std::size_t index1, std::size_t index2, std::size_t shapeIndex)
		: mMesh_ptr{ mesh_ptr }, mIndex0{ index0 }, mIndex1{ index1 }, mIndex2{ index2 },
		mShapeIndex{ shapeIndex }
	{
		ComputeNormal();
	}

	void MeshTriangle::ComputeNormal()
	{
		mNormal =
			glm::normalize(glm::cross(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex1].position - mMesh_ptr->shapes[mShapeIndex].vertices[mIndex0].position,
				mMesh_ptr->shapes[mShapeIndex].vertices[mIndex2].position - mMesh_ptr->shapes[mShapeIndex].vertices[mIndex0].position));
	}

	float MeshTriangle::interpolateU(const float beta, const float gamma) const
	{
		return ((1 - beta - gamma) * mMesh_ptr->shapes[mShapeIndex].vertices[mIndex0].texCoord[0]
			+ beta * mMesh_ptr->shapes[mShapeIndex].vertices[mIndex1].texCoord[0]
			+ gamma * mMesh_ptr->shapes[mShapeIndex].vertices[mIndex2].texCoord[0]);
	}

	float MeshTriangle::interpolateV(const float beta, const float gamma) const
	{
		return ((1 - beta - gamma) * mMesh_ptr->shapes[mShapeIndex].vertices[mIndex0].texCoord[1]
			+ beta * mMesh_ptr->shapes[mShapeIndex].vertices[mIndex1].texCoord[1]
			+ gamma * mMesh_ptr->shapes[mShapeIndex].vertices[mIndex2].texCoord[1]);
	}

	BBox MeshTriangle::getBoundingBox() const
	{
		float delta = 0.00001f;
		atlas::math::Point v0(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex0].position);
		atlas::math::Point v1(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex1].position);
		atlas::math::Point v2(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex2].position);

		return BBox{
			atlas::math::Point{
				glm::min(glm::min(v0.x, v1.x), v2.x) - delta,
				glm::min(glm::min(v0.y, v1.y), v2.y) - delta,
				glm::min(glm::min(v0.z, v1.z), v2.z) - delta,
			},
			atlas::math::Point{
				glm::max(glm::max(v0.x, v1.x), v2.x) + delta,
				glm::max(glm::max(v0.y, v1.y), v2.y) + delta,
				glm::max(glm::max(v0.z, v1.z), v2.z) + delta,
			}
		};
	}

	virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const = 0;

	std::shared_ptr<atlas::utils::ObjMesh> mMesh_ptr;
	std::size_t mShapeIndex;	//Index to shape in mesh object
	std::size_t mIndex0, mIndex1, mIndex2;	// index to vertices of that shape
	mutable atlas::math::Normal mNormal;
	mutable float mU;
	mutable float mV;
	float kEpsilon = 0.1f;
};

class SmoothMeshTriangle : public MeshTriangle
{
public:
	SmoothMeshTriangle::SmoothMeshTriangle()
	{}

	SmoothMeshTriangle::SmoothMeshTriangle(std::shared_ptr<atlas::utils::ObjMesh> mesh_ptr,
		std::size_t index0, std::size_t index1, std::size_t index2, std::size_t shapeIndex)
	{
		mMesh_ptr = mesh_ptr;
		mIndex0 = index0;
		mIndex1 = index1;
		mIndex2 = index2;
		mShapeIndex = shapeIndex;
	}

	bool SmoothMeshTriangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}
		return intersect;
	}

	bool SmoothMeshTriangle::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	bool SmoothMeshTriangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		atlas::math::Point v0(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex0].position);
		atlas::math::Point v1(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex1].position);
		atlas::math::Point v2(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex2].position);

		const float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.d.x, d = v0.x - ray.o.x;
		const float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.d.y, h = v0.y - ray.o.y;
		const float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.d.z, l = v0.z - ray.o.z;

		const float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
		const float q = g * i - e * k, s = e * j - f * i;

		const float inv_denom = 1.0f / (a * m + b * q + c * s);

		const float e1 = d * m - b * n - c * p;
		const float beta = e1 * inv_denom;

		if (beta < 0.0f)
			return false;

		const float r = e * l - h * i;
		const float e2 = a * n + d * q + c * r;
		const float gamma = e2 * inv_denom;

		if (gamma < 0.0f)
			return false;

		if (beta + gamma > 1.0f)
			return false;

		const float e3 = a * p - b * r + d * s;
		const float t = e3 * inv_denom;

		if (t < kEpsilon)
			return false;

		tMin = t;
		mNormal = interpolateNormal(beta, gamma);
		return true;
	}

protected:
	atlas::math::Normal SmoothMeshTriangle::interpolateNormal(const float beta, const float gamma) const
	{
		atlas::math::Normal v0(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex0].normal);
		atlas::math::Point v1(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex1].normal);
		atlas::math::Point v2(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex2].normal);

		atlas::math::Normal normal((1.0f - beta - gamma) * v0 + beta * v1 + gamma * v2);
		return glm::normalize(normal);
	}
};

class FlatMeshTriangle : public MeshTriangle
{
public:
	FlatMeshTriangle::FlatMeshTriangle()
	{}

	FlatMeshTriangle::FlatMeshTriangle(std::shared_ptr<atlas::utils::ObjMesh> mesh_ptr,
		std::size_t index0, std::size_t index1, std::size_t index2, std::size_t shapeIndex)
	{
		mMesh_ptr = mesh_ptr;
		mIndex0 = index0;
		mIndex1 = index1;
		mIndex2 = index2;
		mShapeIndex = shapeIndex;
		ComputeNormal();
	}

	bool FlatMeshTriangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}
		return intersect;
	}

	bool FlatMeshTriangle::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	bool FlatMeshTriangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		atlas::math::Point v0(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex0].position);
		atlas::math::Point v1(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex1].position);
		atlas::math::Point v2(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex2].position);

		const float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.d.x, d = v0.x - ray.o.x;
		const float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.d.y, h = v0.y - ray.o.y;
		const float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.d.z, l = v0.z - ray.o.z;

		const float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
		const float q = g * i - e * k, s = e * j - f * i;

		const float inv_denom = 1.0f / (a * m + b * q + c * s);

		const float e1 = d * m - b * n - c * p;
		const float beta = e1 * inv_denom;

		if (beta < 0.0f)
			return false;

		const float r = e * l - h * i;
		const float e2 = a * n + d * q + c * r;
		const float gamma = e2 * inv_denom;

		if (gamma < 0.0f)
			return false;

		if (beta + gamma > 1.0f)
			return false;

		const float e3 = a * p - b * r + d * s;
		const float t = e3 * inv_denom;

		if (t < kEpsilon)
			return false;

		tMin = t;
		return true;
	}
};

class SmoothUVMeshTriangle : public SmoothMeshTriangle
{
public:
	SmoothUVMeshTriangle::SmoothUVMeshTriangle(std::shared_ptr<atlas::utils::ObjMesh> mesh_ptr,
		std::size_t index0, std::size_t index1, std::size_t index2, std::size_t shapeIndex)
	{
		mMesh_ptr = mesh_ptr;
		mIndex0 = index0;
		mIndex1 = index1;
		mIndex2 = index2;
		mShapeIndex = shapeIndex;
	}

	bool SmoothUVMeshTriangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
			sr.u = mU;
			sr.v = mV;
		}
		return intersect;
	}

	bool SmoothUVMeshTriangle::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	bool SmoothUVMeshTriangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		atlas::math::Point v0(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex0].position);
		atlas::math::Point v1(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex1].position);
		atlas::math::Point v2(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex2].position);

		const float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.d.x, d = v0.x - ray.o.x;
		const float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.d.y, h = v0.y - ray.o.y;
		const float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.d.z, l = v0.z - ray.o.z;

		const float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
		const float q = g * i - e * k, s = e * j - f * i;

		const float inv_denom = 1.0f / (a * m + b * q + c * s);

		const float e1 = d * m - b * n - c * p;
		const float beta = e1 * inv_denom;

		if (beta < 0.0f)
			return false;

		const float r = e * l - h * i;
		const float e2 = a * n + d * q + c * r;
		const float gamma = e2 * inv_denom;

		if (gamma < 0.0f)
			return false;

		if (beta + gamma > 1.0f)
			return false;

		const float e3 = a * p - b * r + d * s;
		const float t = e3 * inv_denom;

		if (t < kEpsilon)
			return false;

		tMin = t;
		mNormal = interpolateNormal(beta, gamma);
		mU = interpolateU(beta, gamma);
		mV = interpolateV(beta, gamma);
		return true;
	}
};

class FlatUVMeshTriangle : public FlatMeshTriangle
{
public:
	FlatUVMeshTriangle::FlatUVMeshTriangle(std::shared_ptr<atlas::utils::ObjMesh> mesh_ptr,
		std::size_t index0, std::size_t index1, std::size_t index2, std::size_t shapeIndex)
	{
		mMesh_ptr = mesh_ptr;
		mIndex0 = index0;
		mIndex1 = index1;
		mIndex2 = index2;
		mShapeIndex = shapeIndex;
	}

	bool FlatUVMeshTriangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
			sr.u = mU;
			sr.v = mV;
		}
		return intersect;
	}

	bool FlatUVMeshTriangle::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	bool FlatUVMeshTriangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		atlas::math::Point v0(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex0].position);
		atlas::math::Point v1(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex1].position);
		atlas::math::Point v2(mMesh_ptr->shapes[mShapeIndex].vertices[mIndex2].position);

		const float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.d.x, d = v0.x - ray.o.x;
		const float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.d.y, h = v0.y - ray.o.y;
		const float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.d.z, l = v0.z - ray.o.z;

		const float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
		const float q = g * i - e * k, s = e * j - f * i;

		const float inv_denom = 1.0f / (a * m + b * q + c * s);

		const float e1 = d * m - b * n - c * p;
		const float beta = e1 * inv_denom;

		if (beta < 0.0f)
			return false;

		const float r = e * l - h * i;
		const float e2 = a * n + d * q + c * r;
		const float gamma = e2 * inv_denom;

		if (gamma < 0.0f)
			return false;

		if (beta + gamma > 1.0f)
			return false;

		const float e3 = a * p - b * r + d * s;
		const float t = e3 * inv_denom;

		if (t < kEpsilon)
			return false;

		mU = interpolateU(beta, gamma);
		mV = interpolateV(beta, gamma);
		tMin = t;
		return true;
	}
};

class AxisBox : public Shape
{
public:
	AxisBox::AxisBox() : mP0{ -1,-1,-1 }, mP1{ 1,1,1 }
	{}

	AxisBox::AxisBox(atlas::math::Point p0, atlas::math::Point p1) : mP0{ p0 }, mP1{ p1 }
	{}

	BBox AxisBox::getBoundingBox() const
	{
		return BBox{
			mP0, mP1
		};
	}

	bool AxisBox::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.normal = face_normal;
			sr.ray = ray;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}
		return intersect;
	}

	bool AxisBox::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
			return false;

		return intersectRay(ray, tMin);
	}

	bool AxisBox::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		float ox = ray.o.x, oy = ray.o.y, oz = ray.o.z;
		float dx = ray.d.x, dy = ray.d.y, dz = ray.d.z;

		float tx_min, ty_min, tz_min;
		float tx_max, ty_max, tz_max;

		float a = 1.0f / dx;
		if (a >= 0.0f)
		{
			tx_min = (mP0.x - ox) * a;
			tx_max = (mP1.x - ox) * a;
		}
		else
		{
			tx_min = (mP1.x - ox) * a;
			tx_max = (mP0.x - ox) * a;
		}

		float b = 1.0f / dy;
		if (b >= 0.0f)
		{
			ty_min = (mP0.y - oy) * b;
			ty_max = (mP1.y - oy) * b;
		}
		else
		{
			ty_min = (mP1.y - oy) * b;
			ty_max = (mP0.y - oy) * b;
		}

		float c = 1.0f / dz;
		if (c >= 0.0f)
		{
			tz_min = (mP0.z - oz) * c;
			tz_max = (mP1.z - oz) * c;
		}
		else
		{
			tz_min = (mP1.z - oz) * c;
			tz_max = (mP0.z - oz) * c;
		}

		float t0, t1;
		int face_in, face_out;

		// find largest entering t value
		if (tx_min > ty_min)
		{
			t0 = tx_min;
			face_in = (a >= 0.0f) ? 0 : 3;
		}
		else
		{
			t0 = ty_min;
			face_in = (b >= 0.0f) ? 1 : 4;
		}

		if (tz_min > t0)
		{
			t0 = tz_min;
			face_in = (c >= 0.0f) ? 2 : 5;
		}

		// find smallest exiting t value
		if (tx_max < ty_max)
		{
			t1 = tx_max;
			face_out = (a >= 0.0f) ? 3 : 0;
		}
		else
		{
			t1 = ty_max;
			face_out = (b >= 0.0f) ? 4 : 1;
		}

		if (tz_max < t1)
		{
			t1 = tz_max;
			face_out = (c >= 0.0f) ? 5 : 2;
		}

		if (t0 < t1 && t1 > kEpsilon) // condition for a hit
		{
			if (t0 > kEpsilon)
			{
				tMin = t0;	// ray hits outside surface
				face_normal = getNormal(face_in);
			}
			else
			{
				tMin = t1; // ray hits inside surface
				face_normal = getNormal(face_out);
			}
			return true;
		}
		else
		{
			return false;
		}
	}

private:
	atlas::math::Normal AxisBox::getNormal(const int face_hit) const
	{
		switch (face_hit)
		{
		case 0:
			return { -1,0,0 }; // -x face
		case 1:
			return { 0,-1,0 }; // -y face
		case 2:
			return { 0,0,-1 }; // -z face
		case 3:
			return { 1,0,0 }; // +x face
		case 4:
			return { 0,1,0 }; // +y face
		default:
			return { 0,0,1 }; // +z face
		}
	}

	atlas::math::Point mP0;
	atlas::math::Point mP1;
	//atlas::math::Normal mNormal;
	float kEpsilon{ 0.0001f };
};

class Grid : public Compound
{
public:
	Grid::Grid() : nx{ 0 }, ny{ 0 }, nz{ 0 }
	{};

	// other constructors

	//virtual BBox Grid::getBoundingBox()
	//{
		//return mBBox;
	//}

	void Grid::setupCells()
	{
		// Find the minimum and maximum coordinates of the grid
		atlas::math::Point p0 = minCoordinates();
		atlas::math::Point p1 = maxCoordinates();

		mBBox.mP0 = p0;
		mBBox.mP1 = p1;

		// Compute the number of grid cells in the x, y, and z directions
		auto numObjects = objects.size();
		float wx = p1.x - p0.x;
		float wy = p1.y - p0.y;
		float wz = p1.z - p0.z;
		float multiplier = 2.0f;
		float s = glm::pow(wx * wy * wz / numObjects, 0.3333333f);
		nx = static_cast<int> (multiplier * wx / s + 1);
		ny = static_cast<int> (multiplier * wy / s + 1);
		nz = static_cast<int> (multiplier * wz / s + 1);

		// Set up the array of grid cells with null pointers
		int numCells = nx * ny * nz;
		cells.reserve(numObjects);
		for (int j = 0; j < numCells; j++)
			cells.push_back(nullptr);

		// Set up a temporary array to hold the number of objects stored in each cell
		std::vector<int> counts;
		counts.reserve(numCells);
		for (int j = 0; j < numCells; j++)
			counts.push_back(0);

		// Put the objects into the cells
		BBox objectBBox;
		int index;

		for (std::shared_ptr<Shape> object : objects)
		{
			objectBBox = object->getBoundingBox();

			// Compute the cell indices at the corners of the bounding box of the object
			int ixmin = (int)glm::clamp(nx * (objectBBox.mP0.x - p0.x) / (p1.x - p0.x), 0.0f, nx - 1.0f);
			int iymin = (int)glm::clamp(ny * (objectBBox.mP0.y - p0.y) / (p1.y - p0.y), 0.0f, ny - 1.0f);
			int izmin = (int)glm::clamp(nz * (objectBBox.mP0.z - p0.z) / (p1.z - p0.z), 0.0f, nz - 1.0f);

			int ixmax = (int)glm::clamp(nx * (objectBBox.mP1.x - p0.x) / (p1.x - p0.x), 0.0f, nx - 1.0f);
			int iymax = (int)glm::clamp(ny * (objectBBox.mP1.y - p0.y) / (p1.y - p0.y), 0.0f, ny - 1.0f);
			int izmax = (int)glm::clamp(nz * (objectBBox.mP1.z - p0.z) / (p1.z - p0.z), 0.0f, nz - 1.0f);

			// Add the object to the cells
			for (int iz = izmin; iz <= izmax; iz++)				// cells in z direction
				for (int iy = iymin; iy <= iymax; iy++)			// cells in y direction
					for (int ix = ixmin; ix <= ixmax; ix++)		// cells in x direction
					{
						index = ix + nx * iy + nx * ny * iz;

						if (counts[index] == 0) {
							cells[index] = object;
							counts[index] += 1;
						}
						else {
							if (counts[index] == 1) {
								std::shared_ptr<Compound> compound_ptr{ std::make_shared <Compound>() };
								compound_ptr->addObject(cells[index]);
								compound_ptr->addObject(object);
								cells[index] = compound_ptr;
								counts[index] += 1;
							}
							else {
								cells[index]->addObject(object);
								counts[index] += 1;
							}
						}
					}
		}
		//objects.erase(objects.begin(), objects.end());

		int num_zeroes = 0;
		int num_ones = 0;
		int num_twos = 0;
		int num_threes = 0;
		int num_greater = 0;

		for (int j = 0; j < numCells; j++) {
			if (counts[j] == 0)
				num_zeroes += 1;
			if (counts[j] == 1)
				num_ones += 1;
			if (counts[j] == 2)
				num_twos += 1;
			if (counts[j] == 3)
				num_threes += 1;
			if (counts[j] > 3)
				num_greater += 1;
		}

		std::cout << "num_cells = " << numCells << std::endl;
		std::cout << "numZeroes = " << num_zeroes << "  numOnes = " << num_ones << "  numTwos = " << num_twos << std::endl;
		std::cout << "numThrees = " << num_threes << "  numGreater = " << num_greater << std::endl;

		//counts.erase(counts.begin(), counts.end());
	}

	bool Grid::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float ox = ray.o.x, oy = ray.o.y, oz = ray.o.z;
		float dx = ray.d.x, dy = ray.d.y, dz = ray.d.z;

		float x0 = mBBox.mP0.x, y0 = mBBox.mP0.y, z0 = mBBox.mP0.z;
		float x1 = mBBox.mP1.x, y1 = mBBox.mP1.y, z1 = mBBox.mP1.z;

		float tx_min, ty_min, tz_min;
		float tx_max, ty_max, tz_max;

		float a = 1.0f / dx;
		if (atlas::core::geq(a, 0.0f))
		{
			tx_min = (x0 - ox) * a;
			tx_max = (x1 - ox) * a;
		}
		else
		{
			tx_min = (x1 - ox) * a;
			tx_max = (x0 - ox) * a;
		}

		float b = 1.0f / dy;
		if (atlas::core::geq(b, 0.0f))
		{
			ty_min = (y0 - oy) * b;
			ty_max = (y1 - oy) * b;
		}
		else
		{
			ty_min = (y1 - oy) * b;
			ty_max = (y0 - oy) * b;
		}

		float c = 1.0f / dz;
		if (atlas::core::geq(c, 0.0f))
		{
			tz_min = (z0 - oz) * c;
			tz_max = (z1 - oz) * c;
		}
		else
		{
			tz_min = (z1 - oz) * c;
			tz_max = (z0 - oz) * c;
		}

		float t0, t1;

		if (tx_min > ty_min)
			t0 = tx_min;
		else
			t0 = ty_min;

		if (tz_min > t0)
			t0 = tz_min;

		if (tx_max < ty_max)
			t1 = tx_max;
		else
			t1 = ty_max;

		if (tz_max < t1)
			t1 = tz_max;

		if (t0 > t1)
			return false;

		int ix, iy, iz;

		if (mBBox.inside(ray.o)) {
			ix = (int)glm::clamp((ox - x0) * nx / (x1 - x0), 0.0f, nx - 1.0f);
			iy = (int)glm::clamp((oy - y0) * ny / (y1 - y0), 0.0f, ny - 1.0f);
			iz = (int)glm::clamp((oz - z0) * nz / (z1 - z0), 0.0f, nz - 1.0f);
		}
		else {
			atlas::math::Point p = ray.o + t0 * ray.d;
			ix = (int)glm::clamp((p.x - x0) * nx / (x1 - x0), 0.0f, nx - 1.0f);
			iy = (int)glm::clamp((p.y - y0) * ny / (y1 - y0), 0.0f, ny - 1.0f);
			iz = (int)glm::clamp((p.z - z0) * nz / (z1 - z0), 0.0f, nz - 1.0f);
		}

		float dtx = (tx_max - tx_min) / nx;
		float dty = (ty_max - ty_min) / ny;
		float dtz = (tz_max - tz_min) / nz;

		float 	tx_next, ty_next, tz_next;
		int 	ix_step, iy_step, iz_step;
		int 	ix_stop, iy_stop, iz_stop;

		if (atlas::core::geq(dx, 0.0f)) {
			tx_next = tx_min + (ix + 1) * dtx;
			ix_step = +1;
			ix_stop = nx;
		}
		else {
			tx_next = tx_min + (nx - ix) * dtx;
			ix_step = -1;
			ix_stop = -1;
		}

		if (atlas::core::isZero(dx)) {
			tx_next = std::numeric_limits<float>::max();
			ix_step = -1;
			ix_stop = -1;
		}

		if (atlas::core::geq(dy, 0.0f)) {
			ty_next = ty_min + (iy + 1) * dty;
			iy_step = +1;
			iy_stop = ny;
		}
		else {
			ty_next = ty_min + (ny - iy) * dty;
			iy_step = -1;
			iy_stop = -1;
		}

		if (atlas::core::isZero(dx)) {
			ty_next = std::numeric_limits<float>::max();
			iy_step = -1;
			iy_stop = -1;
		}

		if (atlas::core::geq(dz, 0.0f)) {
			tz_next = tz_min + (iz + 1) * dtz;
			iz_step = +1;
			iz_stop = nz;
		}
		else {
			tz_next = tz_min + (nz - iz) * dtz;
			iz_step = -1;
			iz_stop = -1;
		}

		if (atlas::core::isZero(dx)) {
			tz_next = std::numeric_limits<float>::max();
			iz_step = -1;
			iz_stop = -1;
		}

		while (true) {
			std::shared_ptr<Shape> object_ptr = cells[ix + nx * iy + nx * ny * iz];

			if (tx_next < ty_next && tx_next < tz_next) {
				if (object_ptr != nullptr && object_ptr->hit(ray, sr)) {
					return (true);
				}

				tx_next += dtx;
				ix += ix_step;

				if (ix == ix_stop)
					return (false);
			}
			else {
				if (ty_next < tz_next) {
					if (object_ptr != nullptr && object_ptr->hit(ray, sr)) {
						return (true);
					}

					ty_next += dty;
					iy += iy_step;

					if (iy == iy_stop)
						return (false);
				}
				else {
					if (object_ptr != nullptr && object_ptr->hit(ray, sr)) {
						return (true);
					}

					tz_next += dtz;
					iz += iz_step;

					if (iz == iz_stop)
						return (false);
				}
			}
		}
	}

	bool Grid::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		float ox = ray.o.x, oy = ray.o.y, oz = ray.o.z;
		float dx = ray.d.x, dy = ray.d.y, dz = ray.d.z;

		float x0 = mBBox.mP0.x, y0 = mBBox.mP0.y, z0 = mBBox.mP0.z;
		float x1 = mBBox.mP1.x, y1 = mBBox.mP1.y, z1 = mBBox.mP1.z;

		float tx_min, ty_min, tz_min;
		float tx_max, ty_max, tz_max;

		float a = 1.0f / dx;
		if (atlas::core::geq(a, 0.0f))
		{
			tx_min = (x0 - ox) * a;
			tx_max = (x1 - ox) * a;
		}
		else
		{
			tx_min = (x1 - ox) * a;
			tx_max = (x0 - ox) * a;
		}

		float b = 1.0f / dy;
		if (atlas::core::geq(b, 0.0f))
		{
			ty_min = (y0 - oy) * b;
			ty_max = (y1 - oy) * b;
		}
		else
		{
			ty_min = (y1 - oy) * b;
			ty_max = (y0 - oy) * b;
		}

		float c = 1.0f / dz;
		if (atlas::core::geq(c, 0.0f))
		{
			tz_min = (z0 - oz) * c;
			tz_max = (z1 - oz) * c;
		}
		else
		{
			tz_min = (z1 - oz) * c;
			tz_max = (z0 - oz) * c;
		}

		float t0, t1;

		if (tx_min > ty_min)
			t0 = tx_min;
		else
			t0 = ty_min;

		if (tz_min > t0)
			t0 = tz_min;

		if (tx_max < ty_max)
			t1 = tx_max;
		else
			t1 = ty_max;

		if (tz_max < t1)
			t1 = tz_max;

		if (t0 > t1)
			return false;

		int ix, iy, iz;

		if (mBBox.inside(ray.o)) {
			ix = (int)glm::clamp((ox - x0) * nx / (x1 - x0), 0.0f, nx - 1.0f);
			iy = (int)glm::clamp((oy - y0) * ny / (y1 - y0), 0.0f, ny - 1.0f);
			iz = (int)glm::clamp((oz - z0) * nz / (z1 - z0), 0.0f, nz - 1.0f);
		}
		else {
			atlas::math::Point p = ray.o + t0 * ray.d;
			ix = (int)glm::clamp((p.x - x0) * nx / (x1 - x0), 0.0f, nx - 1.0f);
			iy = (int)glm::clamp((p.y - y0) * ny / (y1 - y0), 0.0f, ny - 1.0f);
			iz = (int)glm::clamp((p.z - z0) * nz / (z1 - z0), 0.0f, nz - 1.0f);
		}

		float dtx = (tx_max - tx_min) / nx;
		float dty = (ty_max - ty_min) / ny;
		float dtz = (tz_max - tz_min) / nz;

		float 	tx_next, ty_next, tz_next;
		int 	ix_step, iy_step, iz_step;
		int 	ix_stop, iy_stop, iz_stop;

		if (atlas::core::geq(dx, 0.0f)) {
			tx_next = tx_min + (ix + 1) * dtx;
			ix_step = +1;
			ix_stop = nx;
		}
		else {
			tx_next = tx_min + (nx - ix) * dtx;
			ix_step = -1;
			ix_stop = -1;
		}

		if (atlas::core::isZero(dx)) {
			tx_next = std::numeric_limits<float>::max();
			ix_step = -1;
			ix_stop = -1;
		}

		if (atlas::core::geq(dy, 0.0f)) {
			ty_next = ty_min + (iy + 1) * dty;
			iy_step = +1;
			iy_stop = ny;
		}
		else {
			ty_next = ty_min + (ny - iy) * dty;
			iy_step = -1;
			iy_stop = -1;
		}

		if (atlas::core::isZero(dx)) {
			ty_next = std::numeric_limits<float>::max();
			iy_step = -1;
			iy_stop = -1;
		}

		if (atlas::core::geq(dz, 0.0f)) {
			tz_next = tz_min + (iz + 1) * dtz;
			iz_step = +1;
			iz_stop = nz;
		}
		else {
			tz_next = tz_min + (nz - iz) * dtz;
			iz_step = -1;
			iz_stop = -1;
		}

		if (atlas::core::isZero(dx)) {
			tz_next = std::numeric_limits<float>::max();
			iz_step = -1;
			iz_stop = -1;
		}

		while (true) {
			std::shared_ptr<Shape> object_ptr = cells[ix + nx * iy + nx * ny * iz];

			if (tx_next < ty_next && tx_next < tz_next) {
				if (object_ptr != nullptr && object_ptr->shadowHit(ray, tMin)) {
					return (true);
				}

				tx_next += dtx;
				ix += ix_step;

				if (ix == ix_stop)
					return (false);
			}
			else {
				if (ty_next < tz_next) {
					if (object_ptr != nullptr && object_ptr->shadowHit(ray, tMin)) {
						return (true);
					}

					ty_next += dty;
					iy += iy_step;

					if (iy == iy_stop)
						return (false);
				}
				else {
					if (object_ptr != nullptr && object_ptr->shadowHit(ray, tMin)) {
						return (true);
					}

					tz_next += dtz;
					iz += iz_step;

					if (iz == iz_stop)
						return (false);
				}
			}
		}
	}

private:
	std::vector<std::shared_ptr<Shape>> cells; //cells are stored in a 1D array
	BBox mBBox;     // bounding box
	int nx, ny, nz; // number cells in x, y, z directions
	float kEpsilon = 0.0001f;

	atlas::math::Point Grid::minCoordinates()
	{
		BBox bbox;
		atlas::math::Point p0{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };

		for (auto object : objects)
		{
			bbox = object->getBoundingBox();

			if (bbox.mP0.x < p0.x)
				p0.x = bbox.mP0.x;
			if (bbox.mP0.y < p0.y)
				p0.y = bbox.mP0.y;
			if (bbox.mP0.z < p0.z)
				p0.z = bbox.mP0.z;
		}

		p0.x -= kEpsilon; p0.y -= kEpsilon; p0.z -= kEpsilon;

		return p0;
	}

	atlas::math::Point Grid::maxCoordinates()
	{
		BBox bbox;
		atlas::math::Point p1{ -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };

		for (auto object : objects)
		{
			bbox = object->getBoundingBox();

			if (bbox.mP1.x > p1.x)
				p1.x = bbox.mP1.x;
			if (bbox.mP1.y > p1.y)
				p1.y = bbox.mP1.y;
			if (bbox.mP1.z > p1.z)
				p1.z = bbox.mP1.z;
		}

		p1.x += kEpsilon; p1.y += kEpsilon; p1.z += kEpsilon;

		return p1;
	}
};

/****************************************************************/
/*						   BRDFs								*/
/***************************************************************/

class Lambertian : public BRDF
{
public:
	Lambertian::Lambertian() : mDiffuseColour{}, mDiffuseReflection{}
	{}

	Lambertian::Lambertian(Colour diffuseColor, float diffuseReflection) :
		mDiffuseColour{ diffuseColor }, mDiffuseReflection{ diffuseReflection }
	{}

	Colour Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector const& incoming) const
	{
		return mDiffuseColour * mDiffuseReflection * glm::one_over_pi<float>();
	}

	Colour Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected) const
	{
		return mDiffuseColour * mDiffuseReflection;
	}

	void Lambertian::setDiffuseReflection(float kd)
	{
		mDiffuseReflection = kd;
	}

	void Lambertian::setDiffuseColour(Colour const& colour)
	{
		mDiffuseColour = colour;
	}

private:
	Colour mDiffuseColour;
	float mDiffuseReflection;
};

class Glossy : public BRDF
{
public:
	Glossy::Glossy() : mGlossyColour{}, mGlossyReflection{}, mExponent{}, mSampler{ nullptr }
	{}

	Glossy::Glossy(Colour glossyColor, float glossyReflection, float exp) :
		mGlossyColour{ glossyColor }, mGlossyReflection{ glossyReflection }, mExponent{ exp },
		mSampler{ nullptr }
	{}

	Colour
		Glossy::fn([[maybe_unused]] ShadeRec const& sr,
			[[maybe_unused]] atlas::math::Vector const& reflected,
			[[maybe_unused]] atlas::math::Vector const& incoming) const
	{
		Colour L{};
		float nDotWi = glm::dot(sr.normal, incoming);
		atlas::math::Vector r{ -incoming + 2.0f * sr.normal * nDotWi };
		float rDotWo = glm::dot(r, reflected);

		if (rDotWo > 0.0f)
			L = mGlossyColour * mGlossyReflection * glm::pow(rDotWo, mExponent);

		return L;
	}

	/* Blinn-Phong
		Colour L{0,0,0};
		float nDotWi = glm::dot(sr.normal, incoming);
		atlas::math::Vector r{ -incoming + 2.0f * sr.normal * nDotWi };
		float rDotWo = glm::dot(r, reflected);

		if (rDotWo > 0.0f)
			L += mGlossyReflection * glm::pow(rDotWo, mExponent);

		return L;
	*/

	virtual Colour Glossy::sample_f([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector& incoming,
		[[maybe_unused]] float& pdf) const
	{
		using atlas::math::Vector;

		float nDotWo = glm::dot(sr.normal, reflected);
		Vector r{ -reflected + 2.0f * sr.normal * nDotWo };

		Vector w{ r };
		Vector u = glm::cross(Vector{ 0.00424f, 1.0f, 0.00764f }, w);
		u = glm::normalize(u);
		Vector v = glm::cross(u, w);

		atlas::math::Point sp = mSampler->sampleHemisphere();
		incoming = sp.x * u + sp.y * v + sp.z * w;

		if (glm::dot(sr.normal, incoming) < 0.0f)
			incoming = -sp.x * u - sp.y * v + sp.z * w;

		float phongLobe = glm::pow(glm::dot(r, incoming), mExponent);
		pdf = phongLobe * glm::dot(sr.normal, incoming);

		return mGlossyColour * mGlossyReflection * phongLobe;
	}

	Colour
		Glossy::rho([[maybe_unused]] ShadeRec const& sr,
			[[maybe_unused]] atlas::math::Vector const& reflected) const
	{
		return { 0,0,0 };
	}

	void Glossy::setGlossyReflection(float kd)
	{
		mGlossyReflection = kd;
	}

	void Glossy::setGlossyColour(Colour const& colour)
	{
		mGlossyColour = colour;
	}

	void Glossy::setGlossyExponent(float exp)
	{
		mExponent = exp;
	}

	void Glossy::setSampler(std::shared_ptr<Sampler> sampler, const float exp) {
		mSampler = sampler;
		mSampler->mapSamplesToHemisphere(exp);
	}

private:
	Colour mGlossyColour;
	float mGlossyReflection;
	float mExponent;
	std::shared_ptr<Sampler> mSampler;
};

class PerfectSpecular : public BRDF
{
public:
	PerfectSpecular::PerfectSpecular() : mColour{ 0,0,0 }, mReflection{ 0.50f }
	{}

	PerfectSpecular::PerfectSpecular(Colour colour, float reflection) : BRDF{},
		mColour{ colour }, mReflection{ reflection }
	{}

	Colour
		PerfectSpecular::fn([[maybe_unused]] ShadeRec const& sr,
			[[maybe_unused]] atlas::math::Vector const& reflected,
			[[maybe_unused]] atlas::math::Vector const& incoming) const
	{
		return { 0,0,0 };
	}

	Colour
		PerfectSpecular::rho([[maybe_unused]] ShadeRec const& sr,
			[[maybe_unused]] atlas::math::Vector const& reflected) const
	{
		return { 0,0,0 };
	}

	Colour
		PerfectSpecular::sample_f([[maybe_unused]] ShadeRec const& sr,
			[[maybe_unused]] atlas::math::Vector const& reflected,
			[[maybe_unused]] atlas::math::Vector& incoming) const
	{
		float nDotWo = glm::dot(sr.normal, reflected);
		incoming = -reflected + (sr.normal * (2.0f * nDotWo));
		return mReflection * mColour / glm::dot(sr.normal, incoming);
	}

	void PerfectSpecular::setPerfSpecReflection(float kd)
	{
		mReflection = kd;
	}

	void PerfectSpecular::setPerfSpecColour(Colour const& colour)
	{
		mColour = colour;
	}

	float PerfectSpecular::getReflection()
	{
		return mReflection;
	}

	Colour PerfectSpecular::getColour()
	{
		return mColour;
	}

private:
	Colour mColour;
	float mReflection;
};

// SPATIALLY VARYING BRDFs //

class SV_Lambertian : public BRDF
{
public:

	// constructors, etc.
	SV_Lambertian::SV_Lambertian() : mDiffuseColour{}, mDiffuseReflection{ 0.0f }
	{}

	SV_Lambertian::SV_Lambertian(std::shared_ptr<Texture> diffuseColor, float diffuseReflection)
		: mDiffuseColour{ diffuseColor }, mDiffuseReflection{ diffuseReflection }
	{}

	Colour SV_Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected) const
	{
		return mDiffuseReflection * mDiffuseColour->getColour(sr);
	}

	Colour SV_Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector const& incoming) const
	{
		return mDiffuseReflection * mDiffuseColour->getColour(sr) * glm::one_over_pi<float>();
	}

	void SV_Lambertian::setDiffuseReflection(float kd)
	{
		mDiffuseReflection = kd;
	}

	void SV_Lambertian::setDiffuseColour(std::shared_ptr<Texture> texture_ptr)
	{
		mDiffuseColour = texture_ptr;
	}

private:
	std::shared_ptr<Texture> mDiffuseColour;
	float mDiffuseReflection;
};

class SV_Glossy : public BRDF
{
public:

	// constructors, etc.
	SV_Glossy::SV_Glossy() : mGlossyColour{}, mGlossyReflection{}, mExponent{}, mSampler{ nullptr }
	{}

	SV_Glossy::SV_Glossy(std::shared_ptr<Texture> glossyColour, float glossyReflection, float exp) :
		mGlossyColour{ glossyColour }, mGlossyReflection{ glossyReflection }, mExponent{ exp },
		mSampler{ nullptr }
	{}

	Colour SV_Glossy::rho([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected) const
	{
		return { 0,0,0 };
	}

	Colour SV_Glossy::fn([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector const& incoming) const
	{
		Colour L{ 0,0,0 };
		float nDotWi = glm::dot(sr.normal, incoming);
		atlas::math::Vector r{ -incoming + 2.0f * sr.normal * nDotWi };
		float rDotWo = glm::dot(r, reflected);

		if (rDotWo > 0.0f)
			L += mGlossyReflection * glm::pow(rDotWo, mExponent);

		return L;
	}

	virtual Colour SV_Glossy::sample_f([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector& incoming,
		[[maybe_unused]] float& pdf) const
	{
		using atlas::math::Vector;

		float nDotWo = glm::dot(sr.normal, reflected);
		Vector r{ -reflected + 2.0f * sr.normal * nDotWo };

		Vector w{ r };
		Vector u = glm::cross(Vector{ 0.00424f, 1.0f, 0.00764f }, w);
		u = glm::normalize(u);
		Vector v = glm::cross(u, w);

		atlas::math::Point sp = mSampler->sampleHemisphere();
		incoming = sp.x * u + sp.y * v + sp.z * w;

		if (glm::dot(sr.normal, incoming) < 0.0f)
			incoming = -sp.x * u - sp.y * v + sp.z * w;

		float phongLobe = glm::pow(glm::dot(r, incoming), mExponent);
		pdf = phongLobe * glm::dot(sr.normal, incoming);

		return mGlossyColour->getColour(sr) * mGlossyReflection * phongLobe;
	}

	void SV_Glossy::setGlossyReflection(float kd)
	{
		mGlossyReflection = kd;
	}

	void SV_Glossy::setGlossyColour(std::shared_ptr<Texture> texture_ptr)
	{
		mGlossyColour = texture_ptr;
	}

	void SV_Glossy::setGlossyExponent(float exp)
	{
		mExponent = exp;
	}

	void SV_Glossy::setSampler(std::shared_ptr<Sampler> sampler, const float exp) {
		mSampler = sampler;
		mSampler->mapSamplesToHemisphere(exp);
	}

private:
	std::shared_ptr<Texture> mGlossyColour;
	float mGlossyReflection;
	float mExponent;
	std::shared_ptr<Sampler> mSampler;
};

/****************************************************************/
/*						   BTDFs								*/
/***************************************************************/

class PerfectTransmitter : public BTDF
{
public:
	PerfectTransmitter::PerfectTransmitter() : BTDF{}, mKt{ 1.0f }, mIor{ 1.0f }
	{}

	PerfectTransmitter::PerfectTransmitter(float kt, float ior) :
		mKt{ kt }, mIor{ ior }
	{}

	void setKt(const float kt)
	{
		mKt = kt;
	}

	void setIor(const float ior)
	{
		mIor = ior;
	}

	Colour PerfectTransmitter::fn([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector const& incoming) const
	{
		return { 0,0,0 };
	}

	Colour PerfectTransmitter::rho([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected) const
	{
		return { 0,0,0 };
	}

	Colour PerfectTransmitter::sample_f([[maybe_unused]] ShadeRec const& sr,
		[[maybe_unused]] atlas::math::Vector const& reflected,
		[[maybe_unused]] atlas::math::Vector& transmitted) const
	{
		atlas::math::Normal n{ sr.normal };
		float cosThetaI = glm::dot(n, reflected);
		float eta = mIor;

		if (cosThetaI < 0.0f) {
			cosThetaI = -cosThetaI;
			n = -n;
			eta = 1.0f / eta;
		}

		float temp = 1.0f - (1.0f - cosThetaI * cosThetaI) / (eta * eta);
		float cosTheta2 = glm::sqrt(temp);
		transmitted = (-reflected / eta) - (cosTheta2 - cosThetaI / eta) * n;

		Colour white{ 1,1,1 };
		return (mKt / (eta * eta) * (white / glm::abs(glm::dot(sr.normal, transmitted))));

	}

	bool tir([[maybe_unused]] ShadeRec const& sr) const
	{

		atlas::math::Vector wo{ -sr.ray.d };
		float cosThetaI = glm::dot(sr.normal, wo);
		float eta = mIor;

		if (cosThetaI < 0.0f)
			eta = 1.0f / eta;

		return (1.0f - (1.0f - cosThetaI * cosThetaI) / (eta * eta) < 0.0f);
	}

private:
	float mKt;
	float mIor;
};

/****************************************************************/
/*						  MATERIALS								*/
/***************************************************************/

class Matte : public Material
{
public:
	Matte::Matte() :
		Material{},
		mDiffuseBRDF{ std::make_shared<Lambertian>() },
		mAmbientBRDF{ std::make_shared<Lambertian>() }
	{}

	Matte::Matte(float kd, float ka, Colour colour) : Matte{}
	{
		setAmbientReflection(ka);
		setDiffuseReflection(kd);
		setAmbientColour(colour);
		setDiffuseColour(colour);
	}

	Matte::Matte(float kd, float ka, Colour aColour, Colour dColour) : Matte{}
	{
		setAmbientReflection(ka);
		setDiffuseReflection(kd);
		setAmbientColour(aColour);
		setDiffuseColour(dColour);
	}

	void Matte::setAmbientReflection(float k)
	{
		mAmbientBRDF->setDiffuseReflection(k);
	}

	void Matte::setDiffuseReflection(float k)
	{
		mDiffuseBRDF->setDiffuseReflection(k);
	}

	void Matte::setAmbientColour(Colour aColour)
	{
		mAmbientBRDF->setDiffuseColour(aColour);
	}

	void Matte::setDiffuseColour(Colour dColour)
	{
		mDiffuseBRDF->setDiffuseColour(dColour);
	}

	Colour Matte::shade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		Vector wo = -sr.ray.o;
		Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
		size_t numLights = sr.world->lights.size();

		for (size_t i{ 0 }; i < numLights; ++i)
		{
			Vector wi = glm::normalize(sr.world->lights[i]->getDirection(sr));
			float nDotWi = glm::dot(sr.normal, wi);

			if (nDotWi > 0.0f)
			{
				bool inShadow = false;

				if (sr.world->lights[i]->castsShadows())
				{
					Ray<Vector> shadowRay{ sr.hitPoint, wi };
					inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
				}

				if (!inShadow)
				{
					L += mDiffuseBRDF->fn(sr, wo, wi) * (sr.world->lights[i]->L(sr) *
						nDotWi);
				}
			}
		}

		return L;
	}

	Colour Matte::areaLightShade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		atlas::math::Vector wo = -sr.ray.d;
		Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
		size_t numLights = sr.world->lights.size();

		for (size_t i{ 0 }; i < numLights; i++)
		{
			atlas::math::Vector wi = sr.world->lights[i]->getDirection(sr);
			float nDotWi = glm::dot(sr.normal, wi);

			if (nDotWi > 0.0f)
			{
				bool inShadow = false;

				if (sr.world->lights[i]->castsShadows())
				{
					Ray<Vector> shadowRay{ sr.hitPoint, wi };
					inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
				}

				if (!inShadow)
				{
					L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr)
						* sr.world->lights[i]->G(sr) * nDotWi / sr.world->lights[i]->pdf(sr);
				}
			}
		}
		return L;
	}

private:
	std::shared_ptr<Lambertian> mDiffuseBRDF;
	std::shared_ptr<Lambertian> mAmbientBRDF;
};

class Phong : public Material
{
public:

	Phong::Phong() :
		Material{},
		mDiffuseBRDF{ std::make_shared<Lambertian>() },
		mAmbientBRDF{ std::make_shared<Lambertian>() },
		mSpecularBRDF{ std::make_shared<Glossy>() }
	{}

	Phong::Phong(float kd, float ka, float ks, float exp, Colour color) : Phong{}
	{
		setDiffuseReflection(kd);
		setAmbientReflection(ka);
		setSpecularReflection(ks);
		setAmbientColour(color);
		setDiffuseColour(color);
		setSpecularColour(color);
		setSpecularExponent(exp);
	}

	Phong::Phong(float kd, float ka, float ks, float exp, Colour aColour, Colour dColour, Colour sColour) : Phong{}
	{
		setDiffuseReflection(kd);
		setAmbientReflection(ka);
		setSpecularReflection(ks);
		setAmbientColour(aColour);
		setDiffuseColour(dColour);
		setSpecularColour(sColour);
		setSpecularExponent(exp);
	}

	void Phong::setAmbientReflection(float k)
	{
		mAmbientBRDF->setDiffuseReflection(k);
	}

	void Phong::setDiffuseReflection(float k)
	{
		mDiffuseBRDF->setDiffuseReflection(k);
	}

	void Phong::setSpecularReflection(float k)
	{
		mSpecularBRDF->setGlossyReflection(k);
	}

	void Phong::setAmbientColour(Colour aColour)
	{
		mAmbientBRDF->setDiffuseColour(aColour);
	}

	void Phong::setDiffuseColour(Colour dColour)
	{
		mDiffuseBRDF->setDiffuseColour(dColour);
	}

	void Phong::setSpecularColour(Colour sColour)
	{
		mSpecularBRDF->setGlossyColour(sColour);
	}

	void Phong::setSpecularExponent(float exp)
	{
		mSpecularBRDF->setGlossyExponent(exp);
	}

	Colour Phong::shade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		Vector wo = -sr.ray.o;
		Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
		size_t numLights = sr.world->lights.size();

		for (size_t i{ 0 }; i < numLights; ++i)
		{
			Vector wi = sr.world->lights[i]->getDirection(sr);
			float nDotWi = glm::dot(sr.normal, wi);

			if (nDotWi > 0.0f)
			{
				bool inShadow = false;

				if (sr.world->lights[i]->castsShadows())
				{
					Ray<Vector> shadowRay{ sr.hitPoint, wi };
					inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
				}

				if (!inShadow)
				{
					L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi))
						* (sr.world->lights[i]->L(sr) * nDotWi);
				}

			}
		}

		return L;
	}

	Colour Phong::areaLightShade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		atlas::math::Vector wo = -sr.ray.d;
		Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
		size_t numLights = sr.world->lights.size();

		for (size_t i{ 0 }; i < numLights; i++)
		{
			atlas::math::Vector wi = sr.world->lights[i]->getDirection(sr);
			float nDotWi = glm::dot(sr.normal, wi);

			if (nDotWi > 0.0f)
			{
				bool inShadow = false;

				if (sr.world->lights[i]->castsShadows())
				{
					Ray<Vector> shadowRay{ sr.hitPoint, wi };
					inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
				}

				if (!inShadow)
				{
					L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi))
						* sr.world->lights[i]->L(sr) * sr.world->lights[i]->G(sr)
						* nDotWi / sr.world->lights[i]->pdf(sr);
				}
			}
		}
		return L;
	}

protected:
	std::shared_ptr<Lambertian> mDiffuseBRDF;
	std::shared_ptr<Lambertian> mAmbientBRDF;
	std::shared_ptr<Glossy> mSpecularBRDF;
};

class Reflective : public Phong
{
public:

	Reflective::Reflective() : Phong{}, mReflectiveBRDF{ new PerfectSpecular() }
	{}

	void Reflective::setReflectiveColour(Colour const c)
	{
		mReflectiveBRDF->setPerfSpecColour(c);
	}

	void Reflective::setReflectiveReflection(float const r)
	{
		mReflectiveBRDF->setPerfSpecReflection(r);
	}

	Colour Reflective::shade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		Colour L{ Phong::shade(sr) };

		Vector wo = -sr.ray.d;
		Vector wi;

		Colour fr = mReflectiveBRDF->sample_f(sr, wo, wi);
		Ray<Vector> reflectedRay{ sr.hitPoint, wi };

		L += fr * sr.world->tracer->traceRay(reflectedRay, sr.depth + 1, sr.threadNum)
			* glm::dot(sr.normal, wi);

		return L;
	}

	Colour Reflective::areaLightShade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		Colour L{ Phong::areaLightShade(sr) };

		Vector wo{ -sr.ray.d };
		Vector wi;

		Colour fr{ mReflectiveBRDF->sample_f(sr, wo, wi) };
		Ray<Vector> reflectedRay{ sr.hitPoint, wi };

		L += fr * sr.world->tracer->traceRay(reflectedRay, sr.depth + 1, sr.threadNum)
			* glm::dot(sr.normal, wi);

		return L;
	}

private:
	std::shared_ptr<PerfectSpecular> mReflectiveBRDF;
};

class GlossyReflector : public Phong
{
public:

	GlossyReflector::GlossyReflector() : Phong{}, mGlossyBRDF{ new Glossy() }
	{}

	Colour GlossyReflector::areaLightShade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		Colour L{ Phong::areaLightShade(sr) };
		Vector wo{ -sr.ray.d };
		Vector wi;
		float pdf;
		Colour fr{ mGlossyBRDF->sample_f(sr, wo, wi, pdf) };
		Ray<Vector> reflectedRay{ sr.hitPoint, wi };

		L += fr * sr.world->tracer->traceRay(reflectedRay, sr.depth + 1, sr.threadNum)
			* glm::dot(sr.normal, wi) / pdf;

		return L;
	}

	void GlossyReflector::setGlossyReflectorReflection(const float k)
	{
		mGlossyBRDF->setGlossyReflection(k);
	}

	void GlossyReflector::setGlossyReflectorColour(Colour const c)
	{
		mGlossyBRDF->setGlossyColour(c);
	}

	void GlossyReflector::setGlossyReflectorExponent(const float exp)
	{
		mGlossyBRDF->setGlossyExponent(exp);
	}

	void GlossyReflector::setSampler(std::shared_ptr<Sampler> sampler, const float exp)
	{
		mGlossyBRDF->setSampler(sampler, exp);
	}

private:
	std::shared_ptr<Glossy> mGlossyBRDF;
};

class Transparent : public Phong
{
public:

	Transparent::Transparent() :
		Phong{},
		mReflectiveBRDF{ std::make_shared<PerfectSpecular>() },
		mSpecularBTDF{ std::make_shared<PerfectTransmitter>() }
	{}

	void Transparent::setIor(float ior)
	{
		mSpecularBTDF->setIor(ior);
	}

	void Transparent::setKt(float kt)
	{
		mSpecularBTDF->setKt(kt);
	}

	void Transparent::setKr(float kr)
	{
		mReflectiveBRDF->setPerfSpecReflection(kr);
	}

	Colour Transparent::shade(ShadeRec& sr)
	{
		using atlas::math::Vector;
		using atlas::math::Ray;

		Colour L{ Phong::shade(sr) };

		Vector wo = -sr.ray.d;
		Vector wi;
		Colour fr = mReflectiveBRDF->sample_f(sr, wo, wi);
		Ray<Vector> reflectedRay{ sr.hitPoint, wi };

		if (mSpecularBTDF->tir(sr)) {
			L += sr.world->tracer->traceRay(reflectedRay, sr.depth + 1, sr.threadNum);
		}
		else {
			Vector wt;
			Colour ft = mSpecularBTDF->sample_f(sr, wo, wt);
			Ray <Vector> transmittedRay{ sr.hitPoint, wt };

			L += fr * sr.world->tracer->traceRay(reflectedRay, sr.depth + 1, sr.threadNum)
				* glm::abs(glm::dot(sr.normal, wi));
			L += ft * sr.world->tracer->traceRay(transmittedRay, sr.depth + 1, sr.threadNum)
				* glm::abs(glm::dot(sr.normal, wt));
		}

		return L;
	}

	// Maybe need to make this different from shade
	Colour Transparent::areaLightShade(ShadeRec& sr)
	{
		using atlas::math::Vector;
		using atlas::math::Ray;

		Colour L{ Phong::areaLightShade(sr) };

		Vector wo = -sr.ray.d;
		Vector wi;
		Colour fr = mReflectiveBRDF->sample_f(sr, wo, wi);
		Ray<Vector> reflectedRay{ sr.hitPoint, wi };

		if (mSpecularBTDF->tir(sr)) {
			L += sr.world->tracer->traceRay(reflectedRay, sr.depth + 1, sr.threadNum);
		}
		else {
			Vector wt;
			Colour ft = mSpecularBTDF->sample_f(sr, wo, wt);
			Ray <Vector> transmittedRay{ sr.hitPoint, wt };

			L += fr * sr.world->tracer->traceRay(reflectedRay, sr.depth + 1, sr.threadNum)
				* glm::abs(glm::dot(sr.normal, wi));
			L += ft * sr.world->tracer->traceRay(transmittedRay, sr.depth + 1, sr.threadNum)
				* glm::abs(glm::dot(sr.normal, wt));
		}

		return L;
	}

private:
	std::shared_ptr<PerfectSpecular> mReflectiveBRDF;
	std::shared_ptr<PerfectTransmitter> mSpecularBTDF;
};

class SV_Matte : public Material
{
public:

	// constructors, etc.
	SV_Matte::SV_Matte() :
		Material{},
		mDiffuseBRDF{ std::make_shared<SV_Lambertian>() },
		mAmbientBRDF{ std::make_shared<SV_Lambertian>() }
	{}

	SV_Matte::SV_Matte(float kd, float ka, std::shared_ptr<Texture> color) : SV_Matte{}
	{
		setDiffuseReflection(kd);
		setAmbientReflection(ka);
		setDiffuseColour(color);
	}

	void SV_Matte::setDiffuseReflection(float k)
	{
		mDiffuseBRDF->setDiffuseReflection(k);
	}

	void SV_Matte::setAmbientReflection(float k)
	{
		mAmbientBRDF->setDiffuseReflection(k);
	}

	void SV_Matte::setDiffuseColour(std::shared_ptr<Texture> texture_ptr)
	{
		mDiffuseBRDF->setDiffuseColour(texture_ptr);
	}

	void SV_Matte::setAmbientColour(std::shared_ptr<Texture> texture_ptr)
	{
		mAmbientBRDF->setDiffuseColour(texture_ptr);
	}

	Colour SV_Matte::shade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		Vector wo = -sr.ray.o;
		Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
		size_t numLights = sr.world->lights.size();

		for (size_t i{ 0 }; i < numLights; ++i)
		{
			Vector wi = glm::normalize(sr.world->lights[i]->getDirection(sr));
			float nDotWi = glm::dot(sr.normal, wi);
			//float nDotWo = glm::dot(sr.normal, wo);

			if (nDotWi > 0.0f)
			{
				bool inShadow = false;

				if (sr.world->lights[i]->castsShadows())
				{
					Ray<Vector> shadowRay{ sr.hitPoint, wi };
					inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
				}

				if (!inShadow)
					L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) * nDotWi;
			}
		}

		return L;
	}

	Colour SV_Matte::areaLightShade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		Vector wo = -sr.ray.o;
		Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
		size_t numLights = sr.world->lights.size();

		for (size_t i{ 0 }; i < numLights; ++i)
		{
			Vector wi = glm::normalize(sr.world->lights[i]->getDirection(sr));
			float nDotWi = glm::dot(sr.normal, wi);
			//float nDotWo = glm::dot(sr.normal, wo);

			if (nDotWi > 0.0f)
			{
				bool inShadow = false;

				if (sr.world->lights[i]->castsShadows())
				{
					Ray<Vector> shadowRay{ sr.hitPoint, wi };
					inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
				}

				if (!inShadow)
				{
					L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr)
						* sr.world->lights[i]->G(sr) * nDotWi / sr.world->lights[i]->pdf(sr);
				}
			}
		}

		return L;
	}

private:
	std::shared_ptr<SV_Lambertian> mDiffuseBRDF;
	std::shared_ptr<SV_Lambertian> mAmbientBRDF;
};

class SV_Phong : public Material
{
public:

	// constructors, etc.
	SV_Phong::SV_Phong() :
		Material{},
		mDiffuseBRDF{ std::make_shared<SV_Lambertian>() },
		mAmbientBRDF{ std::make_shared<SV_Lambertian>() },
		mSpecularBRDF{ std::make_shared<SV_Glossy>() }
	{}

	SV_Phong::SV_Phong(float kd, float ka, float ks, float exp, std::shared_ptr<Texture> color) : SV_Phong{}
	{
		setDiffuseReflection(kd);
		setAmbientReflection(ka);
		setSpecularReflection(ks);
		setAmbientColour(color);
		setDiffuseColour(color);
		setSpecularColour(color);
		setSpecularExponent(exp);
	}

	void SV_Phong::setDiffuseReflection(float k)
	{
		mDiffuseBRDF->setDiffuseReflection(k);
	}

	void SV_Phong::setAmbientReflection(float k)
	{
		mAmbientBRDF->setDiffuseReflection(k);
	}

	void SV_Phong::setDiffuseColour(std::shared_ptr<Texture> texture_ptr)
	{
		mDiffuseBRDF->setDiffuseColour(texture_ptr);
	}

	void SV_Phong::setAmbientColour(std::shared_ptr<Texture> texture_ptr)
	{
		mAmbientBRDF->setDiffuseColour(texture_ptr);
	}

	void SV_Phong::setSpecularReflection(float k)
	{
		mSpecularBRDF->setGlossyReflection(k);
	}

	void SV_Phong::setSpecularColour(std::shared_ptr<Texture> texture_ptr)
	{
		mSpecularBRDF->setGlossyColour(texture_ptr);
	}

	void SV_Phong::setSpecularExponent(float exp)
	{
		mSpecularBRDF->setGlossyExponent(exp);
	}

	Colour SV_Phong::shade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		Vector wo = -sr.ray.o;
		Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
		size_t numLights = sr.world->lights.size();

		for (size_t i{ 0 }; i < numLights; ++i)
		{
			Vector wi = glm::normalize(sr.world->lights[i]->getDirection(sr));
			float nDotWi = glm::dot(sr.normal, wi);

			if (nDotWi > 0.0f)
			{
				bool inShadow = false;

				if (sr.world->lights[i]->castsShadows())
				{
					Ray<Vector> shadowRay{ sr.hitPoint, wi };
					inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
				}

				if (!inShadow)
					L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi))
					* sr.world->lights[i]->L(sr) * nDotWi;
			}
		}

		return L;
	}

	Colour SV_Phong::areaLightShade(ShadeRec& sr)
	{
		using atlas::math::Ray;
		using atlas::math::Vector;

		Vector wo = -sr.ray.o;
		Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
		size_t numLights = sr.world->lights.size();

		for (size_t i{ 0 }; i < numLights; ++i)
		{
			Vector wi = glm::normalize(sr.world->lights[i]->getDirection(sr));
			float nDotWi = glm::dot(sr.normal, wi);

			if (nDotWi > 0.0f)
			{
				bool inShadow = false;

				if (sr.world->lights[i]->castsShadows())
				{
					Ray<Vector> shadowRay{ sr.hitPoint, wi };
					inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
				}

				if (!inShadow)
				{
					L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi))
						* sr.world->lights[i]->L(sr) * sr.world->lights[i]->G(sr)
						* nDotWi / sr.world->lights[i]->pdf(sr);
				}
			}
		}

		return L;
	}

private:
	std::shared_ptr<SV_Lambertian> mDiffuseBRDF;
	std::shared_ptr<SV_Lambertian> mAmbientBRDF;
	std::shared_ptr<SV_Glossy> mSpecularBRDF;
};

class Emissive : public Material
{
public:
	//constructors, setters, etc.
	Emissive::Emissive() : Material{}, mRadiance{ 0.50f }, mColour{ 1,1,1 }
	{}

	void Emissive::scaleRadiance(const float radiance)
	{
		mRadiance = radiance;
	}

	void Emissive::setColour(const Colour colour)
	{
		mColour = colour;
	}

	Colour Emissive::getEmittedRadiance([[maybe_unused]] ShadeRec& sr)
	{
		return mRadiance * mColour;
	}

	Colour Emissive::shade([[maybe_unused]] ShadeRec& sr)
	{
		return { 0,0,0 };
	}

	Colour Emissive::areaLightShade(ShadeRec& sr)
	{
		if (glm::dot(-sr.normal, sr.ray.d) > 0.0f)
			return (mRadiance * mColour);
		else
			return { 0,0,0 };
	}

private:
	float mRadiance;
	Colour mColour;
};

/****************************************************************/
/*						 LIGHTS 								*/
/***************************************************************/

class Directional : public Light
{
public:
	Directional::Directional() : Light{}
	{}

	Directional::Directional(atlas::math::Vector const& d, bool shadows) : Light{}
	{
		mShadows = shadows;
		setDirection(d);
	}

	void Directional::setDirection(atlas::math::Vector const& d)
	{
		mDirection = glm::normalize(d);
	}

	atlas::math::Vector Directional::getDirection([[maybe_unused]] ShadeRec& sr) const
	{
		return mDirection;
	}

	bool Directional::inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		size_t numObjects = sr.world->scene.size();

		for (size_t j{ 0 }; j < numObjects; j++)
		{
			if (sr.world->scene[j]->shadowHit(ray, t))
			{
				return true;
			}
		}
		return false;
	}

	bool Directional::castsShadows() const
	{
		return mShadows;
	}

private:
	atlas::math::Vector mDirection;
};

class Ambient : public Light
{
public:
	Ambient::Ambient() : Light{}
	{
		mShadows = false;
	}

	atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec& sr) const
	{
		return atlas::math::Vector{ 0.0f };
	}

	bool Ambient::inShadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] ShadeRec& sr) const
	{
		return false;
	}

	bool Ambient::castsShadows() const
	{
		return mShadows;
	}

private:
	atlas::math::Vector mDirection;
};

class AmbientOccluder : public Light
{
public:

	AmbientOccluder::AmbientOccluder()
	{}

	void AmbientOccluder::setSampler(std::vector<std::shared_ptr<Sampler>> occlusionSamplerVec)
	{
		samplerVec = occlusionSamplerVec;
		for (auto item : samplerVec)
			item->mapSamplesToHemisphere(1);
	}

	void AmbientOccluder::setMinAmount(Colour min)
	{
		mMinAmount = min;
	}

	Colour AmbientOccluder::L([[maybe_unused]] ShadeRec& sr) const
	{
		sr.occluderW = sr.normal;
		sr.occluderV = glm::cross(sr.occluderW, atlas::math::Vector{ 0.00072f, 1.0f, 0.0034f });
		sr.occluderV = glm::normalize(sr.occluderV);
		sr.occluderU = glm::cross(sr.occluderV, sr.occluderW);
		atlas::math::Ray<atlas::math::Vector> shadowRay{};
		shadowRay.o = sr.hitPoint;
		shadowRay.d = getDirection(sr);

		if (inShadow(shadowRay, sr))
			return mMinAmount * mRadiance * mColour;
		else
			return mRadiance * mColour;
	}

	atlas::math::Vector AmbientOccluder::getDirection([[maybe_unused]] ShadeRec& sr) const
	{
		atlas::math::Point samplePoint = samplerVec[sr.threadNum]->sampleHemisphere();
		return (samplePoint.x * sr.occluderU + samplePoint.y * sr.occluderV + samplePoint.z * sr.occluderW);
	}

	bool AmbientOccluder::inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t;
		size_t numObjects = sr.world->scene.size();
		for (int j = 0; j < numObjects; j++)
			if (sr.world->scene[j]->shadowHit(ray, t))
				return true;

		return false;
	}

	void AmbientOccluder::setShadows(bool shadows)
	{
		mShadows = shadows;
	}

	bool AmbientOccluder::castsShadows() const
	{
		return mShadows;
	}

private:
	std::vector<std::shared_ptr<Sampler>> samplerVec;
	Colour mMinAmount;
};

class PointLight : public Light
{
public:
	PointLight::PointLight() : Light{}, mPoint{ 0,0,0 }
	{}

	PointLight::PointLight(atlas::math::Point const& p, bool shadows) : Light{}
	{
		mShadows = shadows;
		setLocation(p);
	}

	Colour L(ShadeRec& sr) const
	{
		atlas::math::Point hitPoint{ sr.ray.o + sr.t * sr.ray.d };
		atlas::math::Vector mDirection{ mPoint - hitPoint };
		float r{ glm::dot(mDirection, mDirection) };
		float rSquared = r * r;
		return mRadiance * mColour / rSquared;
	}

	void PointLight::setLocation(atlas::math::Vector const& p)
	{
		mPoint = p;
	}

	atlas::math::Vector PointLight::getDirection([[maybe_unused]] ShadeRec& sr) const
	{
		atlas::math::Point hitPoint = sr.ray.o + sr.t * sr.ray.d;
		atlas::math::Vector mDirection = mPoint - hitPoint;
		mDirection = glm::normalize(mDirection);
		return mDirection;
	}

	bool PointLight::inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t;
		size_t numObjects = sr.world->scene.size();
		float d = glm::length(ray.o - mPoint);

		for (size_t j{ 0 }; j < numObjects; j++)
		{
			if (sr.world->scene[j]->shadowHit(ray, t) && atlas::core::geq(d, t))
			{
				return true;
			}
		}
		return false;
	}

	bool PointLight::castsShadows() const
	{
		return mShadows;
	}

private:
	atlas::math::Point mPoint;
};

class AreaLight : public Light
{
public:

	AreaLight::AreaLight() : Light{}, mShape_ptr{ nullptr }, mMaterial_ptr{ nullptr }
	{}

	void AreaLight::setObject(std::shared_ptr<Shape> shape_ptr)
	{
		mShape_ptr = shape_ptr;
		mMaterial_ptr = mShape_ptr->getMaterial();
	}

	Colour AreaLight::L([[maybe_unused]] ShadeRec& sr) const
	{
		float nDotD = glm::dot(-sr.areaLightNormal, sr.areaWi);

		if (nDotD > 0.0f)
			return mMaterial_ptr->getEmittedRadiance(sr);
		else
			return { 0,0,0 };
	}

	float AreaLight::G(const ShadeRec& sr) const
	{
		float nDotD = glm::dot(-sr.areaLightNormal, sr.areaWi);
		float d2 = glm::distance2(sr.areaSamplePoint, sr.hitPoint);

		return (nDotD / d2);
	}

	float AreaLight::pdf(const ShadeRec& sr) const
	{
		return mShape_ptr->pdf(sr);
	}

	atlas::math::Vector AreaLight::getDirection([[maybe_unused]] ShadeRec& sr) const
	{
		sr.areaSamplePoint = mShape_ptr->sample(sr);
		sr.areaLightNormal = mShape_ptr->getNormal(sr.areaSamplePoint);
		sr.areaWi = sr.areaSamplePoint - sr.hitPoint;
		sr.areaWi = glm::normalize(sr.areaWi);

		return sr.areaWi;
	}

	bool AreaLight::inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t;
		size_t numObjects = sr.world->scene.size();
		float ts = glm::dot((sr.areaSamplePoint - ray.o), ray.d);

		for (size_t i = 0; i < numObjects; i++)
		{
			if (sr.world->scene[i]->shadowHit(ray, t) && t < ts)
				return true;
		}

		return false;
	}

	bool AreaLight::castsShadows() const
	{
		return mShadows;
	}

private:
	std::shared_ptr<Shape> mShape_ptr;
	std::shared_ptr<Material> mMaterial_ptr;
};

/****************************************************************/
/*                       TRACERS								*/
/****************************************************************/

class RayCast : public Tracer
{
public:
	RayCast::RayCast() : Tracer{}
	{}

	RayCast::RayCast(std::shared_ptr<World> w) : Tracer{ w }
	{}

	Colour RayCast::traceRay(atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] const int depth, [[maybe_unused]] const int threadNum) const
	{
		ShadeRec trace_data{};
		trace_data.world = world_ptr;
		trace_data.t = std::numeric_limits<float>::max();
		bool hit{};

		for (auto const& obj : world_ptr->scene)
		{
			hit |= obj->hit(ray, trace_data);
		}

		if (hit)
			return trace_data.material->shade(trace_data);
		else
			return world_ptr->background;
	}
};

class AreaLighting : public Tracer
{
public:
	AreaLighting::AreaLighting() : Tracer{}
	{}

	AreaLighting::AreaLighting(std::shared_ptr <World> w) : Tracer{ w }
	{}

	Colour AreaLighting::traceRay(atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] const int depth, [[maybe_unused]] const int threadNum) const
	{

		if (depth > world_ptr->maxDepth)
			return world_ptr->background;
		else {
			ShadeRec trace_data{};
			trace_data.world = world_ptr;
			trace_data.t = std::numeric_limits<float>::max();
			trace_data.threadNum = threadNum;
			bool hit{};

			for (auto const& obj : world_ptr->scene)
			{
				hit |= obj->hit(ray, trace_data);
			}

			if (hit) {
				trace_data.depth = depth;
				trace_data.ray = ray;
				return trace_data.material->areaLightShade(trace_data);
			}
			else
				return world_ptr->background;
		}
	}
};

class Whitted : public Tracer
{
public:
	Whitted::Whitted() : Tracer{}
	{}

	Whitted::Whitted(std::shared_ptr <World> w) : Tracer{ w }
	{}

	Colour Whitted::traceRay(atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] const int depth, [[maybe_unused]] const int threadNum) const
	{

		if (depth > world_ptr->maxDepth)
			return world_ptr->background;
		else {
			ShadeRec trace_data{};
			trace_data.world = world_ptr;
			trace_data.t = std::numeric_limits<float>::max();
			trace_data.threadNum = threadNum;
			bool hit{};

			for (auto const& obj : world_ptr->scene)
				hit |= obj->hit(ray, trace_data);

			if (hit) {
				trace_data.depth = depth;
				trace_data.ray = ray;
				return trace_data.material->shade(trace_data);
			}
			else
				return world_ptr->background;
		}

	}
};

/****************************************************************/
/*						 CAMERAS								*/
/***************************************************************/

class Pinhole : public Camera
{
public:
	Pinhole::Pinhole() : Camera{}, mDistance{ 500.0f }, mZoom{ 1.0f }
	{}

	void Pinhole::setDistance(float distance)
	{
		mDistance = distance;
	}

	void Pinhole::setZoom(float zoom)
	{
		mZoom = zoom;
	}

	atlas::math::Vector Pinhole::rayDirection(atlas::math::Point2 const& p) const
	{
		const auto dir = p.x * mU + p.y * mV - mDistance * mW;
		return glm::normalize(dir);
	}

	void Pinhole::renderScene(std::shared_ptr<World> world, std::shared_ptr<Sampler> sampler,
		std::size_t width_start, std::size_t width_end,
		std::size_t height_start, std::size_t height_end, int threadNum, std::shared_ptr<Semaphore> sem) const
	{
		sem->wait();
		using atlas::math::Point2;
		using atlas::math::Ray;
		using atlas::math::Vector;

		atlas::math::Point samplePoint{}, pixelPoint{};
		Ray<atlas::math::Vector> ray{};

		//AreaLighting tracer{ world };
		int depth = 0;
		ray.o = mEye;
		float avg{ 1.0f / sampler->getNumSamples() };
		//height_start = 300; width_start = 300;
		for (std::size_t r{ height_start }; r < height_end; ++r)
		{
			for (std::size_t c{ width_start }; c < width_end; ++c)
			{
				Colour pixelAverage{ 0, 0, 0 };

				for (int j = 0; j < sampler->getNumSamples(); ++j)
				{
					samplePoint = sampler->sampleUnitSquare();
					pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
					pixelPoint.y = -(r - 0.5f * world->height + samplePoint.y);
					ray.d = rayDirection(pixelPoint);

					pixelAverage += world->tracer->traceRay(ray, depth, threadNum);
				}

				pixelAverage = pixelAverage * avg;

				// Clamp colours
				float max_value = std::max(pixelAverage.r, std::max(pixelAverage.g, pixelAverage.b));
				if (max_value > 1.0f)
				{
					pixelAverage = pixelAverage / max_value;
				}

				world->image[c + r * world->height] = pixelAverage;
			}
		}
		std::cout << "Thread " << threadNum << " finished" << std::endl;
		sem->notify();
	}

private:
	float mDistance;
	float mZoom;
};

/****************************************************************/
/*						SAMPLERS								*/
/***************************************************************/

class Regular : public Sampler
{
public:
	Regular::Regular(int numSamples, int numSets) : Sampler{ numSamples, numSets }
	{
		generateSamples();
	}

	void Regular::generateSamples()
	{
		int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

		for (int j{ 0 }; j < mNumSets; ++j)
		{
			for (int p{ 0 }; p < n; ++p)
			{
				for (int q{ 0 }; q < n; ++q)
				{
					mSamples.push_back(
						atlas::math::Point{ (q + 0.5f) / n, (p + 0.5f) / n, 0.0f });
				}
			}
		}
	}
};

class Random : public Sampler
{
public:
	Random::Random(int numSamples, int numSets) : Sampler{ numSamples, numSets }
	{
		generateSamples();
	}

	void Random::generateSamples()
	{
		atlas::math::Random<float> engine;
		for (int p = 0; p < mNumSets; ++p)
		{
			for (int q = 0; q < mNumSamples; ++q)
			{
				mSamples.push_back(atlas::math::Point{
					engine.getRandomOne(), engine.getRandomOne(), 0.0f });
			}
		}
	}
};

class Jittered : public Sampler
{
public:
	Jittered(int numSamples, int numSets) : Sampler{ numSamples, numSets }
	{
		generateSamples();
	}

	void generateSamples()
	{
		atlas::math::Random<float> engine;
		int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

		for (int j = 0; j < mNumSets; ++j)
		{
			for (int p = 0; p < n; ++p)
			{
				for (int q = 0; q < n; ++q)
				{
					mSamples.push_back(
						atlas::math::Point{ (q + engine.getRandomOne()) / n, (p + engine.getRandomOne()) / n, 0.0f });
				}
			}
		}
	}
};

class NRooks : public Sampler
{
public:
	NRooks(int numSamples, int numSets) : Sampler{ numSamples, numSets }
	{
		generateSamples();
	}

	void generateSamples()
	{
		atlas::math::Random<float> engine;

		// generate samples along main diagonal
		for (int p{ 0 }; p < mNumSets; ++p)
		{
			for (int j{ 0 }; j < mNumSamples; ++j)
			{
				mSamples.push_back(
					atlas::math::Point{ (j + engine.getRandomOne()) / mNumSamples, (j + engine.getRandomOne()) / mNumSamples, 0.0f });
			}

		}

		shuffle_x_coordinates();
		shuffle_y_coordinates();
	}

	void shuffle_x_coordinates()
	{
		atlas::math::Random<int> engine;

		for (int p{ 0 }; p < mNumSets - 1; ++p)
		{
			for (int i{ 0 }; i < mNumSamples; ++i)
			{
				size_t target{ engine.getRandomMax() % mNumSamples + p * mNumSamples };
				float temp{ mSamples[i + p * mNumSamples + 1].x };
				mSamples[i + p * mNumSamples + 1].x = mSamples[target].x;
				mSamples[target].x = temp;
			}
		}
	}

	void shuffle_y_coordinates()
	{
		atlas::math::Random<int> engine;

		for (int p{ 0 }; p < mNumSets - 1; ++p)
		{
			for (int i{ 0 }; i < mNumSamples; ++i)
			{
				size_t target{ engine.getRandomMax() % mNumSamples + p * mNumSamples };
				float temp{ mSamples[i + p * mNumSamples + 1].y };
				mSamples[i + p * mNumSamples + 1].y = mSamples[target].y;
				mSamples[target].y = temp;
			}
		}
	}
};

class MultiJittered : public Sampler
{
public:
	MultiJittered(int numSamples, int numSets) : Sampler{ numSamples, numSets }
	{
		generateSamples();
	}

	void generateSamples()
	{
		atlas::math::Random<int> engine;
		atlas::math::Random<float> float_eng;

		int n = (int)sqrt((float)mNumSamples);
		float subcell_width{ 1.0f / ((float)mNumSamples) };

		// fill the samples array with dummy points to allow us to use the [] notation
		// when  we set the initial patterns

		atlas::math::Point fill_point;
		for (int j{ 0 }; j < mNumSamples * mNumSets; j++)
		{
			mSamples.push_back(fill_point);
		}

		for (int p{ 0 }; p < mNumSets; ++p)
		{
			for (int i{ 0 }; i < n; i++)
			{
				for (int j{ 0 }; j < n; j++)
				{

					mSamples[i * n + j + p * mNumSamples].x = (i * n + j) * subcell_width + float_eng.getRandomRange(0, subcell_width);
					mSamples[i * n + j + p * mNumSamples].y = (j * n + i) * subcell_width + float_eng.getRandomRange(0, subcell_width);
				}
			}
		}

		// shuffle x coordinates
		for (int p{ 0 }; p < mNumSets; p++)
		{
			for (int i{ 0 }; i < n; i++)
			{
				for (int j{ 0 }; j < n; j++)
				{
					int k{ engine.getRandomRange(j, n - 1) };
					float t = mSamples[i * n + j + p * mNumSamples].x;
					mSamples[i * n + j + p * mNumSamples].x = mSamples[i * n + k + p * mNumSamples].x;
					mSamples[i * n + k + p * mNumSamples].x = t;
				}
			}
		}

		// shuffle y coordinates
		for (int p{ 0 }; p < mNumSets; ++p)
		{
			for (int i{ 0 }; i < n; ++i)
			{
				for (int j{ 0 }; j < n; ++j)
				{
					int k{ engine.getRandomRange(j, n - 1) };
					float t = mSamples[i * n + j + p * mNumSamples].x;
					mSamples[j * n + i + p * mNumSamples].x = mSamples[i * n + k + p * mNumSamples].x;
					mSamples[k * n + i + p * mNumSamples].x = t;
				}
			}
		}
	}
};

class Hammersley : public Sampler
{
public:
	Hammersley(int numSamples, int numSets) : Sampler{ numSamples, numSets }
	{
		generateSamples();
	}

	void generateSamples()
	{
		for (int p{ 0 }; p < mNumSets; ++p)
		{
			for (int j{ 0 }; j < mNumSamples; j++)
			{
				mSamples.push_back(atlas::math::Point{ (float)j / (float)mNumSamples, phi(j), 0.0f });
			}
		}
	}

	float phi(int j)
	{
		float x = 0.0f;
		float f = 0.5f;

		while (j)
		{
			x += f * (float)(j % 2);
			j /= 2;
			f *= 0.5;
		}
		return (x);
	}
};
