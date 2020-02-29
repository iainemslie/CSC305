#include "lab.hpp"

int main()
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	std::shared_ptr<World> world{ std::make_shared<World>() };

	world->width = 600;
	world->height = 600;
	world->background = { 0, 0, 0 };
	world->sampler = std::make_shared<MultiJittered>(64, 83);

	world->scene.push_back(
		std::make_shared<Plane>(atlas::math::Point(1, 0, 0), atlas::math::Normal(0, 1, 0)));
	world->scene[0]->setMaterial(
		std::make_shared<Matte>(0.40f, 0.03f, Colour{ 0.9f, 0.9f, 0.9f }));
	world->scene[0]->setColour({ 0.9f, 0.9f, 0.9f });
	world->scene[0]->setShadows(false);

	world->scene.push_back(
		std::make_shared<AxisBox>(atlas::math::Point{ -50, 0, -100 }, atlas::math::Point{ 50, 100, 0 }));
	world->scene[1]->setMaterial(
		std::make_shared<Matte>(0.40f, 0.025f, Colour{ 0.8f, 0.8f, 0.8f }));
	world->scene[1]->setColour({ 0.8f, 0.8f, 0.8f });
	world->scene[1]->setShadows(true);

	world->scene.push_back(
		std::make_shared<AxisBox>(atlas::math::Point{ -250, 0, -500 }, atlas::math::Point{ -150, 100, -400 }));
	world->scene[2]->setMaterial(
		std::make_shared<Matte>(0.40f, 0.025f, Colour{ 0.8f, 0.8f, 0.8f }));
	world->scene[2]->setColour({ 0.8f, 0.8f, 0.8f });
	world->scene[2]->setShadows(true);

	world->scene.push_back(
		std::make_shared<AxisBox>(atlas::math::Point{ 150, 0, 300 }, atlas::math::Point{ 250, 100, 400 }));
	world->scene[3]->setMaterial(
		std::make_shared<Matte>(0.40f, 0.025f, Colour{ 0.8f, 0.8f, 0.8f }));
	world->scene[3]->setColour({ 0.8f, 0.8f, 0.8f });
	world->scene[3]->setShadows(true);

	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 0, 150, -50 }, 50.0f));
	world->scene[4]->setMaterial(
		std::make_shared<Matte>(0.40f, 0.03f,Colour{ 1, 0, 0 }));
	world->scene[4]->setColour({ 1, 0, 0 });
	world->scene[4]->setShadows(true);

	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ -200, 150, -450 }, 50.0f));
	world->scene[5]->setMaterial(
		std::make_shared<Matte>(0.40f, 0.03f, Colour{ 1, 1, 0 }));
	world->scene[5]->setColour({ 1, 1, 0 });
	world->scene[5]->setShadows(true);

	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 200, 150, 350 }, 50.0f));
	world->scene[6]->setMaterial(
		std::make_shared<Phong>(0.40f, 0.03f, 0.1f, 1000.0f, Colour{ 0, 1, 0 }));
	world->scene[6]->setColour({ 0, 1, 0 });
	world->scene[6]->setShadows(true);

	world->scene.push_back(
		std::make_shared<AxisBox>(atlas::math::Point{ -350, 0, -20000 }, atlas::math::Point{ -325, 200, 500 }));
	world->scene[7]->setMaterial(
		std::make_shared<Matte>(0.40f, 0.025f, Colour{ 0.8f, 0.8f, 0.8f }));
	world->scene[7]->setColour({ 0.8f, 0.8f, 0.8f });
	world->scene[7]->setShadows(true);

	world->ambient = std::make_shared<Ambient>();
	world->ambient->setColour({ 1, 1, 1 });
	world->ambient->scaleRadiance(6.0f);

	world->lights.push_back(
		std::make_shared<Directional>(Directional{ {1, -0.25, 1}, true }));
	world->lights[0]->setColour({ 1, 1, 1 });
	world->lights[0]->scaleRadiance(7.0f);
	
	world->lights.push_back(
		std::make_shared<PointLight>(PointLight{ {-800, 600, 0}, true }));
	world->lights[1]->setColour({ 1, 1, 1 });
	world->lights[1]->scaleRadiance(0.8f);
	
	/*
	Orthographic camera{};
	camera.renderScene(world);
	*/
	/*
	Pinhole camera{};
	camera.setEye({ 0.0f, 300.0f, 800.0f });
	camera.setLookAt({ 0.0f, 50.0f, 200.0f });
	camera.computeUVW();
	camera.renderScene(world);
	*/
	
	ThinLens camera{};
	camera.setEye({ 0.0f, 300.0f, 800.0f });
	camera.setLookAt({ 0.0f, 50.0f, 200.0f });
	camera.computeUVW();
	camera.setFocal(800.0f);
	camera.setLensRadius(8.0f);
	camera.setSampler(std::make_shared<NRooks>(128, 83));
	camera.renderScene(world);
	
	saveToFile("raytrace.bmp", world->width, world->height, world->image);

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