#pragma once

#include <atlas/core/Float.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Ray.hpp>
#include <atlas/math/Solvers.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <vector>
//I added these
#include <limits>
#include <cstdlib>

using Colour = atlas::math::Vector;

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image);

// Your code here.

struct ShadeRec
{
	atlas::math::Vector colour;
	float t;
};


class SceneObject
{
public:

	// Default constructor
	SceneObject()
	{}

	// Destructor
	~SceneObject() {};


	virtual bool intersectRayWithObject(atlas::math::Ray<atlas::math::Vector> r, ShadeRec& sr)
	{
		r = r;
		sr = sr;
		return false;
	}

private:
};


class Sphere : public SceneObject
{
public:

	// Default constructor
	Sphere() : centre{ 0, 0, 0 }, radius{ 60 }, colour{ 1,1,1 }
	{}

	// Custom constructor
	Sphere(atlas::math::Point my_centre, float my_radius, atlas::math::Vector my_colour) : centre{ my_centre }, radius{ my_radius }, colour{ my_colour }
	{
	}

	// Destructor
	~Sphere() {};

	bool Sphere::intersectRayWithObject(atlas::math::Ray<atlas::math::Vector> r, ShadeRec& sr) override
	{
		atlas::math::Vector temp = r.o - centre;
		float a = glm::dot(r.d, r.d);
		float b = 2.0f * glm::dot(temp, r.d);
		float c = glm::dot(temp, temp) - (radius * radius);
		float disc = b * b - 4.0f * a * c;

		if (disc < 0.0f)
		{
			return false;
		}
		else
		{
			//If hit then calculate point of hit and return as t in ShadeRec
			float e = sqrt(disc);
			float denom = 2.0f * a;
			sr.t = (-b - e) / denom;

			sr.colour = colour;
			return true;
		}
	}

private:
	atlas::math::Point centre;
	float radius;
	atlas::math::Vector colour;
};

class Plane : public SceneObject
{
public:

	// Default constructor
	Plane() : point{ 0, 0, 0 }, normal{ 1,0,0 }, colour{ 1,1,1 }
	{}

	// Custom constructor
	Plane(atlas::math::Point my_point, atlas::math::Vector my_normal, atlas::math::Vector my_colour) : point{ my_point }, normal{ my_normal }, colour{ my_colour }
	{
	}

	// Destructor
	~Plane() {};


	// Add Hit Function Here
	bool intersectRayWithObject(atlas::math::Ray<atlas::math::Vector> r, ShadeRec& sr)
	{
		atlas::math::Vector temp = point - r.o;
		float numerator = glm::dot(temp, normal);
		float denominator = glm::dot(r.d, normal);
		float t = numerator / denominator;

		sr.t = t;
		sr.colour = colour;
		return true;
	}


private:
	atlas::math::Point point;
	atlas::math::Vector normal;
	atlas::math::Vector colour;
};


class Torus : public SceneObject
{
public:
	// Default constructor
	Torus() : centre{ 0, 0, 0 }, swept_radius{ 15.0f }, tube_radius{ 30.0f }, colour{ 1.0f,1.0f,1.0f }
	{}

	// Custom constructor
	Torus(atlas::math::Point my_centre, float my_swept_radius, float my_tube_radius, atlas::math::Vector my_colour) : centre{ my_centre }, swept_radius{ my_swept_radius }, tube_radius{ my_tube_radius }, colour{ my_colour }
	{
	}

	// Destructor
	~Torus() {};


	bool intersectRayWithObject(atlas::math::Ray<atlas::math::Vector> r, ShadeRec& sr)
	{
		atlas::math::Vector temp = r.o - centre;

		//float x1 = temp.y;
		auto y1 = temp.x;
		//float z1 = temp.z;

		//float d1 = r.d.x;
		auto d2 = r.d.y;
		//float d3 = r.d.z;

		std::vector<float> coeffs;
		std::vector<float> roots;

		// Define the coefficients
		auto sum_d_sqrd = glm::dot(r.d, r.d);
		auto e = glm::dot(temp, temp) - (swept_radius * swept_radius + tube_radius * tube_radius);
		auto f = glm::dot(temp, r.d);
		auto four_a_sqrd = 4.0f * swept_radius * swept_radius;

		auto E = e * e - four_a_sqrd * (tube_radius * tube_radius - y1 * y1);
		auto D = (4.0f * f * e) + (2.0f * four_a_sqrd * y1 * d2);
		auto C = (2.0f * sum_d_sqrd * e) + (4.0f * f * f) + (four_a_sqrd * d2 * d2);
		auto B = 4.0f * sum_d_sqrd * f;
		auto A = sum_d_sqrd * sum_d_sqrd;

		coeffs.emplace_back(E);	// constant term
		coeffs.emplace_back(D);
		coeffs.emplace_back(C);
		coeffs.emplace_back(B);
		coeffs.emplace_back(A);	//coefficient of t^4

		// Find the roots

		std::size_t num_roots{ 0 };
		num_roots = atlas::math::solveQuartic(coeffs, roots);

		bool intersected = false;

		float t_torus = std::numeric_limits<float>::infinity();

		if (num_roots == 0) {
			return false;
		}

		for (auto root : roots) {
			if (root > 0.0f) {
				intersected = true;
				if (root < t_torus) {
					t_torus = root;
				}
			}
		}

		if (!intersected) {
			return false;
		}

		sr.t = t_torus;
		sr.colour = colour;
		return true;
	}

private:
	atlas::math::Vector centre;
	float swept_radius;
	float tube_radius;
	atlas::math::Vector colour;
};
