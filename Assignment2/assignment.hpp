#pragma once

#include <atlas/core/Float.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/math/Ray.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <limits>
#include <memory>
#include <vector>

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

struct World
{
	std::size_t width, height;
	Colour background;
	std::shared_ptr<Sampler> sampler;
	std::vector<std::shared_ptr<Shape>> scene;
	std::vector<Colour> image;
	std::vector<std::shared_ptr<Light>> lights;
	std::shared_ptr<Light> ambient;
};

struct ShadeRec
{
	Colour color;
	float t;
	atlas::math::Normal normal;
	atlas::math::Point hitPoint;
	atlas::math::Ray<atlas::math::Vector> ray;
	std::shared_ptr<Material> material;
	std::shared_ptr<World> world;
};

// Abstract classes defining the interfaces for concrete entities

class Sampler
{
public:
	Sampler(int numSamples, int numSets) :
		mNumSamples{ numSamples }, mNumSets{ numSets }, mCount{ 0 }, mJump{ 0 }
	{
		mSamples.reserve(mNumSets* mNumSamples);
		setupShuffledIndeces();
	}

	virtual ~Sampler() = default;

	int getNumSamples() const
	{
		return mNumSamples;
	}

	void setupShuffledIndeces()
	{
		mShuffledIndeces.reserve(mNumSamples * mNumSets);
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
				mShuffledIndeces.push_back(indices[j]);
			}
		}
	}

	virtual void generateSamples() = 0;

	atlas::math::Point2 sampleUnitSquare()
	{
		if (mCount % mNumSamples == 0)
		{
			atlas::math::Random<int> engine;
			mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
		}

		return mSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
	}

	atlas::math::Point2 sampleUnitDisk()
	{
		if (mCount % mNumSamples == 0)
		{
			atlas::math::Random<int> engine;
			mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
		}

		return mDiskSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
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
	std::vector<atlas::math::Point2> mSamples;
	std::vector<atlas::math::Point2> mDiskSamples;
	std::vector<int> mShuffledIndeces;

	int mNumSamples;
	int mNumSets;
	unsigned long mCount;
	int mJump;
};

class Shape
{
public:
	Shape::Shape() : mColour{ 0, 0, 0 }, mShadows{ true }
	{}

	virtual ~Shape() = default;

	// if t computed is less than the t in sr, it and the color should be
	// updated in sr
	virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const = 0;

	virtual bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const = 0;

	void setColour(Colour const& col)
	{
		mColour = col;
	}

	Colour getColour() const
	{
		return mColour;
	}

	void setMaterial(std::shared_ptr<Material> const& material)
	{
		mMaterial = material;
	}

	std::shared_ptr<Material> getMaterial() const
	{
		return mMaterial;
	}

	void setShadows(bool option)
	{
		mShadows = option;
	}

protected:
	virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const = 0;

	bool mShadows;
	Colour mColour;
	std::shared_ptr<Material> mMaterial;
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
};

class Material
{
public:
	virtual ~Material() = default;

	virtual Colour shade(ShadeRec& sr) = 0;
};

class Light
{
public:
	virtual atlas::math::Vector getDirection(ShadeRec& sr) = 0;

	virtual bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) = 0;

	virtual bool castsShadows() = 0;

	Colour L([[maybe_unused]] ShadeRec& sr)
	{
		return mRadiance * mColour;
	}

	void scaleRadiance(float b)
	{
		mRadiance = b;
	}

	void setColour(Colour const& c)
	{
		mColour = c;
	}

	void setShadows(bool shadows)
	{
		mShadows = shadows;
	}

protected:
	bool mShadows;
	Colour mColour;
	float mRadiance;
};

// Concrete classes which we can construct and use in our ray tracer

class Sphere : public Shape
{
public:
	Sphere(atlas::math::Point center, float radius) :
		mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
	{}

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
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
			sr.color = mColour;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}

		return intersect;
	}

	bool shadowHit([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray,
		[[maybe_unused]] float& tMin) const
	{
		if (!mShadows)
		{
			return false;
		}

		return intersectRay(ray, tMin);
	}

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
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
			if (t > kEpsilon)
			{
				tMin = t;
				return true;
			}

			// Now the positive root
			t = (-b + e);
			if (t > kEpsilon)
			{
				tMin = t;
				return true;
			}
		}
		return false;
	}

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

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.color = mColour;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}

		return intersect;
	}


	bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
	{
		if (!mShadows)
		{
			return false;
		}

		return intersectRay(ray, tMin);
	}

private:
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

	atlas::math::Point mPoint;
	atlas::math::Normal mNormal;
	const float kEpsilon{ 0.1f };
};

class Triangle : public Shape
{
public:
	Triangle(atlas::math::Point v0, atlas::math::Point v1, atlas::math::Point v2) : mV0{ v0 }, mV1{ v1 }, mV2{ v2 }
	{
		mNormal = glm::normalize(glm::cross(mV1 - mV0, mV2 - mV0));
	}

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.normal = mNormal;
			sr.ray = ray;
			sr.color = mColour;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}
		return intersect;
	}

	bool shadowHit([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray,
		[[maybe_unused]] float& tMin) const
	{
		if (!mShadows)
		{
			return false;
		}

		return intersectRay(ray, tMin);
	}

private:
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

	atlas::math::Point mV0;
	atlas::math::Point mV1;
	atlas::math::Point mV2;
	atlas::math::Normal mNormal;
	const float kEpsilon{ 0.01f };
};

class AxisBox : public Shape
{
public:
	AxisBox::AxisBox(atlas::math::Point p0, atlas::math::Point p1) : mP0{ p0 }, mP1{ p1 }
	{}

	bool AxisBox::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
	{
		float t{ std::numeric_limits<float>::max() };
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.normal = face_normal;
			sr.ray = ray;
			sr.color = mColour;
			sr.t = t;
			sr.material = mMaterial;
			sr.hitPoint = ray.o + t * ray.d;
		}
		return intersect;
	}

	bool shadowHit([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray,
		[[maybe_unused]] float& tMin) const
	{
		if (!mShadows)
		{
			return false;
		}

		return intersectRay(ray, tMin);
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

	atlas::math::Point mP0;
	atlas::math::Point mP1;
	atlas::math::Normal mNormal;
	float kEpsilon{ 0.001f };
};

class Regular : public Sampler
{
public:
	Regular(int numSamples, int numSets) : Sampler{ numSamples, numSets }
	{
		generateSamples();
	}

	void generateSamples()
	{
		int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

		for (int j = 0; j < mNumSets; ++j)
		{
			for (int p = 0; p < n; ++p)
			{
				for (int q = 0; q < n; ++q)
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
	Random(int numSamples, int numSets) : Sampler{ numSamples, numSets }
	{
		generateSamples();
	}

	void generateSamples()
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
				int target{ engine.getRandomOne() % mNumSamples + p * mNumSamples };
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
				int target{ engine.getRandomOne() % mNumSamples + p * mNumSamples };
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
			for (int i{ 0 }; i < n; ++i)
			{
				for (int j{ 0 }; j < n; j++)
				{
					mSamples[i * n + j + p * mNumSamples].x = (i * n + j) * subcell_width + float_eng.getRandomRange(0, subcell_width);
					mSamples[i * n + j + p * mNumSamples].y = (i * n + j) * subcell_width + float_eng.getRandomRange(0, subcell_width);
				}
			}
		}

		// shuffle x coordinates
		for (int p{ 0 }; p < mNumSamples; ++p)
		{
			for (int i{ 0 }; i < n; ++i)
			{
				for (int j{ 0 }; j < n; ++j)
				{
					int k{ engine.getRandomRange(j, n - 1) };
					float t{ mSamples[i * n + j + p * mNumSamples].x };
					mSamples[i * n + j + p * mNumSamples].x = mSamples[i * n + k + p * mNumSamples].x;
					mSamples[i * n + k + p * mNumSamples].x = t;
				}
			}
		}

		// shuffle y coordinates
		for (int p{ 0 }; p < mNumSamples; ++p)
		{
			for (int i{ 0 }; i < n; ++i)
			{
				for (int j{ 0 }; j < n; ++j)
				{
					int k{ engine.getRandomRange(j, n - 1) };
					float t{ mSamples[i * n + j + p * mNumSamples].y };
					mSamples[i * n + j + p * mNumSamples].y = mSamples[i * n + k + p * mNumSamples].y;
					mSamples[i * n + k + p * mNumSamples].y = t;
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

class Lambertian : public BRDF
{
public:
	Lambertian::Lambertian() : mDiffuseColour{}, mDiffuseReflection{}
	{}

	Lambertian::Lambertian(Colour diffuseColor, float diffuseReflection) :
		mDiffuseColour{ diffuseColor }, mDiffuseReflection{ diffuseReflection }
	{}

	Colour
		Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
			[[maybe_unused]] atlas::math::Vector const& reflected,
			[[maybe_unused]] atlas::math::Vector const& incoming) const
	{
		return mDiffuseColour * mDiffuseReflection * glm::one_over_pi<float>();
	}

	Colour
		Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
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
	Glossy::Glossy() : mGlossyColour{}, mGlossyReflection{}, mExponent{}
	{}

	Glossy::Glossy(Colour glossyColor, float glossyReflection, float exp) :
		mGlossyColour{ glossyColor }, mGlossyReflection{ glossyReflection }, mExponent{ exp }
	{}

	Colour
		Glossy::fn([[maybe_unused]] ShadeRec const& sr,
			[[maybe_unused]] atlas::math::Vector const& reflected,
			[[maybe_unused]] atlas::math::Vector const& incoming) const
	{
		Colour L;
		atlas::math::Vector h = (incoming + reflected) / glm::length(incoming + reflected);

		L = mGlossyReflection * mGlossyColour * glm::pow(glm::dot(sr.normal, h), mExponent);

		return L;
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

private:
	Colour mGlossyColour;
	float mGlossyReflection;
	float mExponent;
};

class Matte : public Material
{
public:
	Matte::Matte() :
		Material{},
		mDiffuseBRDF{ std::make_shared<Lambertian>() },
		mAmbientBRDF{ std::make_shared<Lambertian>() }
	{}

	Matte::Matte(float kd, float ka, Colour color) : Matte{}
	{
		setDiffuseReflection(kd);
		setAmbientReflection(ka);
		setDiffuseColour(color);
	}

	void Matte::setDiffuseReflection(float k)
	{
		mDiffuseBRDF->setDiffuseReflection(k);
	}

	void Matte::setAmbientReflection(float k)
	{
		mAmbientBRDF->setDiffuseReflection(k);
	}

	void Matte::setDiffuseColour(Colour colour)
	{
		mDiffuseBRDF->setDiffuseColour(colour);
		mAmbientBRDF->setDiffuseColour(colour);
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
			Vector wi = sr.world->lights[i]->getDirection(sr);
			float nDotWi = glm::dot(sr.normal, wi);

			if (nDotWi > 0.0f)
			{
				bool inShadow = false;

				if (sr.world->lights[i]->castsShadows())
				{
					Ray<atlas::math::Vector> shadowRay{};
					shadowRay.o = sr.hitPoint;
					shadowRay.d = wi;
					inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
				}

				if (!inShadow)
				{
					L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) * nDotWi;
				}

			}
		}

		return L;
	}

protected:
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
		setDiffuseColour(color);
		setSpecularColour(color);
		setSpecularExponent(exp);
	}

	void Phong::setDiffuseReflection(float k)
	{
		mDiffuseBRDF->setDiffuseReflection(k);
	}

	void Phong::setAmbientReflection(float k)
	{
		mAmbientBRDF->setDiffuseReflection(k);
	}

	void Phong::setDiffuseColour(Colour colour)
	{
		mDiffuseBRDF->setDiffuseColour(colour);
		mAmbientBRDF->setDiffuseColour(colour);
	}

	void Phong::setSpecularReflection(float k)
	{
		mSpecularBRDF->setGlossyReflection(k);
	}

	void Phong::setSpecularColour(Colour colour)
	{
		mSpecularBRDF->setGlossyColour(colour);
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
					Ray<atlas::math::Vector> shadowRay{};
					shadowRay.o = sr.hitPoint;
					shadowRay.d = wi;
					inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
				}

				if (!inShadow)
				{
					L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi)) * sr.world->lights[i]->L(sr) * nDotWi;
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


class Directional : public Light
{
public:
	Directional::Directional() : Light{}
	{}

	Directional::Directional(atlas::math::Vector const& d, bool shadows) : Light{}
	{
		mShadows = shadows;
		atlas::math::Vector dir;
		dir.x = d.x; dir.y = -d.y; dir.z = d.z;
		setDirection(dir);
	}

	void Directional::setDirection(atlas::math::Vector const& d)
	{
		mDirection = glm::normalize(d);
	}

	atlas::math::Vector Directional::getDirection([[maybe_unused]] ShadeRec& sr)
	{
		return mDirection;
	}

	bool Directional::inShadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] ShadeRec& sr)
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

	bool Directional::castsShadows()
	{
		return mShadows;
	}

private:
	atlas::math::Vector mDirection;
};

class PointLight : public Light
{
public:
	PointLight::PointLight() : Light{}
	{}

	PointLight::PointLight(atlas::math::Point const& p, bool shadows) : Light{}
	{
		mShadows = shadows;
		setLocation(p);
	}

	Colour L([[maybe_unused]] ShadeRec& sr)
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

	atlas::math::Vector PointLight::getDirection(ShadeRec& sr)
	{
		atlas::math::Point hitPoint = sr.ray.o + sr.t * sr.ray.d;
		atlas::math::Vector mDirection = mPoint - hitPoint;
		mDirection = glm::normalize(mDirection);
		return mDirection;
	}

	bool PointLight::inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr)
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

	bool PointLight::castsShadows()
	{
		return mShadows;
	}

private:
	atlas::math::Point mPoint;
};

class Ambient : public Light
{
public:
	Ambient::Ambient() : Light{}
	{
		mShadows = false;
	}

	atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec& sr)
	{
		return atlas::math::Vector{ 0.0f };
	}

	bool Ambient::inShadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] ShadeRec& sr)
	{
		return false;
	}

	bool Ambient::castsShadows()
	{
		return mShadows;
	}

private:
	atlas::math::Vector mDirection;
};

/****************************************/
/*              Camera                  */
/****************************************/

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

	virtual void renderScene(std::shared_ptr<World> world) const = 0;

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

class Pinhole : public Camera
{
public:
	Pinhole() : Camera{}, mDistance{ 500.0f }, mZoom{ 1.0f }
	{}

	void setDistance(float distance)
	{
		mDistance = distance;
	}

	void setZoom(float zoom)
	{
		mZoom = zoom;
	}

	atlas::math::Vector rayDirection(atlas::math::Point2 const& p) const
	{
		const auto dir = p.x * mU + p.y * mV - mDistance * mW;
		return glm::normalize(dir);
	}

	void renderScene(std::shared_ptr<World> world) const
	{
		using atlas::math::Point2;
		using atlas::math::Ray;
		using atlas::math::Vector;

		Point2 samplePoint{}, pixelPoint{};
		Ray<atlas::math::Vector> ray{};

		ray.o = mEye;
		float avg{ 1.0f / world->sampler->getNumSamples() };

		for (int r{ 0 }; r < world->height; ++r)
		{
			for (int c{ 0 }; c < world->width; ++c)
			{
				Colour pixelAverage{ 0, 0, 0 };

				for (int j = 0; j < world->sampler->getNumSamples(); ++j)
				{
					ShadeRec trace_data{};
					trace_data.world = world;
					trace_data.t = std::numeric_limits<float>::max();
					samplePoint = world->sampler->sampleUnitSquare();
					pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
					pixelPoint.y = -(r - 0.5f * world->height + samplePoint.y);
					ray.d = rayDirection(pixelPoint);
					bool hit{};

					for (auto const& obj : world->scene)
					{
						hit |= obj->hit(ray, trace_data);
					}

					if (hit)
					{
						pixelAverage += trace_data.material->shade(trace_data);
					}
				}

				pixelAverage.r = pixelAverage.r * avg;
				pixelAverage.g = pixelAverage.g * avg;
				pixelAverage.b = pixelAverage.b * avg;

				float max_value = std::max(pixelAverage.r, std::max(pixelAverage.g, pixelAverage.b));
				if (max_value > 1.0f)
				{
					pixelAverage = pixelAverage / max_value;
				}

				world->image.push_back({ pixelAverage.r, pixelAverage.g, pixelAverage.b });
			}
		}
	}

private:
	float mDistance;
	float mZoom;
};

class ThinLens : public Camera
{
public:
	ThinLens() : Camera{}, mLensRadius{ 10.0f }, mDistance{ 500.0f }, mFocal{ 100.0f }, mZoom{ 1.0f }
	{}

	void setLensRadius(float radius)
	{
		mLensRadius = radius;
	}

	void setDistance(float distance)
	{
		mDistance = distance;
	}

	void setZoom(float zoom)
	{
		mZoom = zoom;
	}

	void setFocal(float focal)
	{
		mFocal = focal;
	}

	void setSampler(std::shared_ptr<Sampler> sampler)
	{
		mDiskSampler = sampler;
		mDiskSampler->mapSamplesToUnitDisk();
	}

	atlas::math::Vector rayDirection(atlas::math::Point2 pixelPoint, atlas::math::Point2 lensPoint) const
	{
		atlas::math::Point2 p; // hit point on focal plane
		p.x = pixelPoint.x * mFocal / mDistance;
		p.y = pixelPoint.y * mFocal / mDistance;

		atlas::math::Vector dir = (p.x - lensPoint.x) * mU + (p.y - lensPoint.y) * mV - mFocal * mW;
		return glm::normalize(dir);
	}

	void renderScene(std::shared_ptr<World> world) const
	{
		using atlas::math::Point2;
		using atlas::math::Ray;
		using atlas::math::Vector;

		Ray<atlas::math::Vector> ray{};
		//int depth{ 0 };

		Point2 samplePoint{}, pixelPoint{}, diskPoint{}, lensPoint{};
		float avg{ 1.0f / world->sampler->getNumSamples() };

		for (int r{ 0 }; r < world->height; ++r)
		{
			for (int c{ 0 }; c < world->width; ++c)
			{
				Colour pixelAverage{ 0, 0, 0 };

				for (int j = 0; j < world->sampler->getNumSamples(); ++j)
				{
					ShadeRec trace_data{};
					trace_data.world = world;
					trace_data.t = std::numeric_limits<float>::max();
					samplePoint = world->sampler->sampleUnitSquare();
					pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
					pixelPoint.y = -(r - 0.5f * world->height + samplePoint.y);

					diskPoint = mDiskSampler->sampleUnitDisk();
					lensPoint = diskPoint * mLensRadius;

					ray.o = mEye + lensPoint.x * mU + lensPoint.y * mV;
					ray.d = rayDirection(pixelPoint, lensPoint);

					bool hit{};

					for (auto const& obj : world->scene)
					{
						hit |= obj->hit(ray, trace_data);
					}

					if (hit)
					{
						pixelAverage += trace_data.material->shade(trace_data);
					}
				}

				pixelAverage.r = pixelAverage.r * avg;
				pixelAverage.g = pixelAverage.g * avg;
				pixelAverage.b = pixelAverage.b * avg;

				float max_value = std::max(pixelAverage.r, std::max(pixelAverage.g, pixelAverage.b));
				if (max_value > 1.0f)
				{
					pixelAverage = pixelAverage / max_value;
				}

				world->image.push_back({ pixelAverage.r, pixelAverage.g, pixelAverage.b });
			}
		}
	}

private:
	std::shared_ptr<Sampler> mDiskSampler;

	float mLensRadius;
	float mDistance;
	float mFocal;
	float mZoom;
};


//Not working for some reason
class Orthographic : public Camera
{
public:

	void renderScene(std::shared_ptr<World> world) const
	{
		using atlas::math::Point2;
		using atlas::math::Ray;
		using atlas::math::Vector;

		Point2 samplePoint{}, pixelPoint{};
		Ray<atlas::math::Vector> ray{ {0,0,0},{0,0,-1} };

		float avg{ 1.0f / world->sampler->getNumSamples() };

		for (int row{ 0 }; row < world->height; ++row) {
			for (int col{ 0 }; col < world->width; ++col) {

				Colour pixelAverage{ 0,0,0 };

				for (int j{ 0 }; j < world->sampler->getNumSamples(); ++j)
				{
					ShadeRec trace_data{};
					trace_data.world = world;
					trace_data.t = std::numeric_limits<float>::max();
					samplePoint = world->sampler->sampleUnitSquare();
					pixelPoint.x = col - 0.5f * world->width + samplePoint.x;
					pixelPoint.y = -(row - 0.5f * world->height + samplePoint.y);
					ray.o = atlas::math::Vector{ pixelPoint.x, pixelPoint.y, 0 };

					bool hit{};

					for (auto const& obj : world->scene)
					{
						hit |= obj->hit(ray, trace_data);
					}

					if (hit)
					{
						pixelAverage += trace_data.material->shade(trace_data);
					}
				}

				pixelAverage.r = pixelAverage.r * avg;
				pixelAverage.g = pixelAverage.g * avg;
				pixelAverage.b = pixelAverage.b * avg;

				float max_value = std::max(pixelAverage.r, std::max(pixelAverage.g, pixelAverage.b));
				if (max_value > 1.0f)
				{
					pixelAverage = pixelAverage / max_value;
				}

				world->image.push_back({ pixelAverage.r, pixelAverage.g, pixelAverage.b });
			}
		}
	}
};
