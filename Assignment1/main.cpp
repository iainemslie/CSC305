#include "lab.hpp"


int main()
{
	//Create the image canvas
	const std::size_t imageWidth{ 600 };
	const std::size_t imageHeight{ 600 };
	std::vector<Colour> image(imageWidth * imageHeight);
	constexpr Colour background{ 0, 0, 0 };

	constexpr float max_value = std::numeric_limits<float>::max();
	constexpr std::size_t num_samples{ 4 };
	int n = (int)sqrt((float)num_samples);

	atlas::math::Ray<atlas::math::Vector> ray;
	ray.o = { 0,0,0 };
	ray.d = { 0,0,-1 };

	std::vector<SceneObject*> ObjectContainer;

	/* Create Sphere Objects for the Scene */
	Sphere sphere1{ {imageWidth / 2, imageHeight / 2, -240}, 120, {1, 1, 1} };
	Sphere sphere2{ {imageWidth / 4, imageHeight / 8, 0}, 60, {0, 0, 1} };
	Sphere sphere3{ {imageWidth / 2, 360, 0}, 30, {0, 1, 0} };
	Sphere sphere4{ {200, 120, 0}, 45, {0, 1, 1} };
	Sphere sphere5{ {540, 540, 0}, 15, {1, 0, 0} };
	Sphere sphere6{ {120, 540, -60}, 45, {1, 0, 1} };

	/* Create Plane Objects for the Scene */
	Plane plane1{ {16,14,-1000},{2,0.5,0.5},{1,1,0} };
	Plane plane2{ {900,50,-1500}, {3,5,-2}, {0.7,0.1,0.4} };

	/* Create a Torus Object for the Scene */
	Torus torus1{ {540,140,-20}, 75.0f, 30.0f, {0.1f,0.0f,0.3f} };

	/* Add Scene Objects to a vector container */
	auto it = ObjectContainer.begin();
	it = ObjectContainer.insert(it, &sphere1);
	it = ObjectContainer.insert(it, &sphere2);
	it = ObjectContainer.insert(it, &sphere3);
	it = ObjectContainer.insert(it, &sphere4);
	it = ObjectContainer.insert(it, &sphere5);
	it = ObjectContainer.insert(it, &sphere6);
	it = ObjectContainer.insert(it, &plane1);
	it = ObjectContainer.insert(it, &plane2);
	it = ObjectContainer.insert(it, &torus1);

	constexpr Colour backgroundColour{ 0.0f,0.0f,0.0f };

	ShadeRec trace_data{};

	atlas::math::Vector pixel_colour;

	for (std::size_t y{ 0 }; y < imageHeight; ++y)
	{
		for (std::size_t x{ 0 }; x < imageWidth; ++x)
		{
			Colour sum = background;
			Colour temp;

			float t_value = max_value;

			temp = background;
			for (std::size_t y_pixel{ 0 }; y_pixel < n; ++y_pixel)
			{
				for (std::size_t x_pixel{ 0 }; x_pixel < n; ++x_pixel) {
					//ray.o = { x + 0.5f + x_pixel + (0.5f / n), y + 0.5f + y_pixel + (0.5f / n), 0 }; // Regular Sampling
					ray.o = { x + 0.5f + x_pixel + ((rand() % n) / n), y + 0.5f + y_pixel + ((rand() % n) / n), 0 }; // Jittered Sampling
					// check if ray hit the objects
					for (auto object : ObjectContainer)
					{
						if (object->intersectRayWithObject(ray, trace_data))
						{
							if (trace_data.t < t_value) {
								t_value = trace_data.t;
								temp = trace_data.colour;
							}
						}
					}
					sum += temp;
				}
			}
			sum /= num_samples;
			// set pixel in image
			image[x + y * imageHeight] = sum;
		}
	}

	saveToFile("test_sphere.bmp", imageWidth, imageHeight, image);

	return 0;
}

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image)
{
	std::vector<unsigned char> data(image.size() * 3);

	for (std::size_t i{ 0 }, k{ 0 }; i < image.size(); ++i, k += 3)
	{
		Colour pixel = image[i];
		data[k + 0] = static_cast<unsigned char>(pixel.r * 255);
		data[k + 1] = static_cast<unsigned char>(pixel.g * 255);
		data[k + 2] = static_cast<unsigned char>(pixel.b * 255);
	}

	stbi_write_bmp(filename.c_str(),
		static_cast<int>(width),
		static_cast<int>(height),
		3,
		data.data());
}
