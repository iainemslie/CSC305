#include "assignment.hpp"

// ******* Driver Code *******

#define smoothTri 0
#define flatTri 1
#define smoothUVTri 2
#define flatUVTri 3

void computeMeshNormals(std::shared_ptr<atlas::utils::ObjMesh> mesh, [[maybe_unused]] std::size_t shapeIndex)
{
	std::cout << "Inside computeMeshNormals" << std::endl;
}

//LOAD MESH WITH SMOOTH TRIANGLES + TRANSFORMATIONS
int loadMesh(std::shared_ptr<World> world, std::string filePath, std::string matPath, char triType,
	atlas::math::Vector scale, atlas::math::Vector translate, float rotateX, float rotateY, float rotateZ)
{
	// LOAD MESH
	auto result = atlas::utils::loadObjMesh(filePath);
	if (!result)
	{
		return 0;
	}

	atlas::utils::ObjMesh mesh = result.value();
	std::shared_ptr<atlas::utils::ObjMesh> mesh_ptr = std::make_shared<atlas::utils::ObjMesh>(mesh);

	std::cout << "Num Shapes " << mesh_ptr->shapes.size() << std::endl;

	std::shared_ptr<Compound> compoundObj{ std::make_shared<Compound>() };
	std::shared_ptr<Instance> compoundObjInstance_ptr = std::make_shared<Instance>(compoundObj);

	// Set up materials in vector that are assigned
	// to indices based on their material value
	std::vector<std::shared_ptr<Material>> matVec;

	for (int i = 0; i < mesh.materials.size(); i++)
	{
		switch (mesh_ptr->materials[i].illum)
		{
			case 0:
			{
				// Colour on and Ambient Off
				break;
			}
			case 1:
			{
				// Colour on and Ambient on
				if (mesh_ptr->materials[i].diffuse_texname.size()) {
					// Load Texture from File
					std::shared_ptr<SV_Phong> mat_ptr = std::make_shared<SV_Phong>();
					std::shared_ptr<Image> image_ptr = std::make_shared<Image>();
					std::string imgPath{ matPath + mesh_ptr->materials[i].diffuse_texname };
					image_ptr->readImageFile(imgPath);
					std::shared_ptr<ImageTexture> image_texture_ptr = std::make_shared<ImageTexture>();
					image_texture_ptr->setImage(image_ptr);

					mat_ptr->setAmbientColour(image_texture_ptr);
					mat_ptr->setDiffuseColour(image_texture_ptr);
					mat_ptr->setSpecularColour(image_texture_ptr);
					mat_ptr->setAmbientReflection(mesh_ptr->materials[i].ambient[0]);
					mat_ptr->setDiffuseReflection(mesh_ptr->materials[i].diffuse[0]);
					mat_ptr->setSpecularReflection(mesh_ptr->materials[i].specular[0]);
					mat_ptr->setSpecularExponent(mesh_ptr->materials[i].shininess);
					matVec.push_back(mat_ptr);
				}
				else {
					// Default Material
					std::shared_ptr<Phong> mat_ptr = std::make_shared<Phong>();
					mat_ptr->setAmbientColour({ mesh_ptr->materials[i].ambient[0],
												mesh_ptr->materials[i].ambient[1],
												mesh_ptr->materials[i].ambient[2] });
					mat_ptr->setDiffuseColour({ mesh_ptr->materials[i].diffuse[0],
												mesh_ptr->materials[i].diffuse[1],
												mesh_ptr->materials[i].diffuse[2] });
					mat_ptr->setSpecularColour({ mesh_ptr->materials[i].specular[0],
												mesh_ptr->materials[i].specular[1],
												mesh_ptr->materials[i].specular[2] });
					mat_ptr->setAmbientReflection(0.05f);
					mat_ptr->setDiffuseReflection(0.50f);
					mat_ptr->setSpecularReflection(0.10f);
					mat_ptr->setSpecularExponent(mesh_ptr->materials[i].shininess);
					matVec.push_back(mat_ptr);
				}
				break;
			}
			case 2:
			{
				// Highlight on
				if (mesh_ptr->materials[i].diffuse_texname.size()) {
					// Load Texture from File
					std::shared_ptr<SV_Phong> mat_ptr = std::make_shared<SV_Phong>();
					std::shared_ptr<Image> image_ptr = std::make_shared<Image>();
					std::string imgPath{ matPath + mesh_ptr->materials[i].diffuse_texname };
					image_ptr->readImageFile(imgPath);
					std::shared_ptr<ImageTexture> image_texture_ptr = std::make_shared<ImageTexture>();
					image_texture_ptr->setImage(image_ptr);

					mat_ptr->setAmbientColour(image_texture_ptr);
					mat_ptr->setDiffuseColour(image_texture_ptr);
					mat_ptr->setSpecularColour(image_texture_ptr);
					mat_ptr->setAmbientReflection(mesh_ptr->materials[i].ambient[0]);
					mat_ptr->setDiffuseReflection(mesh_ptr->materials[i].diffuse[0]);
					mat_ptr->setSpecularReflection(mesh_ptr->materials[i].specular[0]);
					mat_ptr->setSpecularExponent(mesh_ptr->materials[i].shininess);
					matVec.push_back(mat_ptr);
				}
				else {
					// Default Material
					std::shared_ptr<Phong> mat_ptr = std::make_shared<Phong>();
					mat_ptr->setAmbientColour({ mesh_ptr->materials[i].ambient[0],
												mesh_ptr->materials[i].ambient[1],
												mesh_ptr->materials[i].ambient[2] });
					mat_ptr->setDiffuseColour({ mesh_ptr->materials[i].diffuse[0],
												mesh_ptr->materials[i].diffuse[1],
												mesh_ptr->materials[i].diffuse[2] });
					mat_ptr->setSpecularColour({ mesh_ptr->materials[i].specular[0],
												mesh_ptr->materials[i].specular[1],
												mesh_ptr->materials[i].specular[2] });
					mat_ptr->setAmbientReflection(0.05f);
					mat_ptr->setDiffuseReflection(0.50f);
					mat_ptr->setSpecularReflection(0.10f);
					mat_ptr->setSpecularExponent(mesh_ptr->materials[i].shininess);
					matVec.push_back(mat_ptr);
				}
				break;
			}
			case 3:
			{
				// Reflection on and Ray Trace on
				std::shared_ptr<Reflective> reflect_mat = std::make_shared<Reflective>();
				reflect_mat->setAmbientReflection(0.25f);
				reflect_mat->setDiffuseReflection(0.50f);
				reflect_mat->setDiffuseColour({ 1.0f, 1.0f, 1.0f });
				reflect_mat->setSpecularReflection(0.15f);
				reflect_mat->setSpecularExponent(10.0f);
				reflect_mat->setSpecularColour({ 1,1,1 });
				reflect_mat->setReflectiveReflection(0.75f);
				reflect_mat->setReflectiveColour({ 1,1,1 });
				matVec.push_back(reflect_mat);
				break;
			}
			case 4:
			{
				// Transparency: Glass on - Reflection: Ray trace on
				std::shared_ptr<Transparent> glass_ptr = std::make_shared<Transparent>();
				glass_ptr->setSpecularReflection(0.05f);
				glass_ptr->setSpecularExponent(10.0f);
				glass_ptr->setIor(1.0f);
				glass_ptr->setKr(0.1f);
				glass_ptr->setKt(0.9f);
				glass_ptr->setSpecularColour({ 1,1,1 });
				matVec.push_back(glass_ptr);
				break;
			}
			case 5:
			{
				// Reflection: Fresnel on and Ray trace on
				break;
			}
			case 6:
			{
				// Transparency: Refraction on - Reflection: Fresnel offand Ray trace on
				break;
			}
			case 7:
			{
				// Transparency: Refraction on - Reflection: Fresnel onand Ray trace on
				break;
			}
			case 8:
			{
				// Reflection on and Ray trace off
				break;
			}
			case 9:
			{
				// Transparency: Glass on - Reflection: Ray trace off
				break;
			}
			case 10:
			{
				// Casts shadows onto invisible surfaces
				break;
			}
			default:
			{
				break;
			}
		}

	}

	std::size_t index0, index1, index2;

	for (size_t shapeIndex = 0; shapeIndex < mesh_ptr->shapes.size(); shapeIndex++)
	{
		std::cout << "Shape: " << shapeIndex << std::endl;
		std::cout << "Normals Defined?: " << mesh_ptr->shapes[shapeIndex].hasNormals << std::endl;
		std::cout << "Texture Coordinates?: " << mesh_ptr->shapes[shapeIndex].hasTextureCoords << std::endl;

		if (!mesh.shapes[shapeIndex].hasNormals)
			computeMeshNormals(mesh_ptr, shapeIndex);

		std::shared_ptr<Grid> grid{ std::make_shared<Grid>() };

		for (size_t k = 0, i = 0; k < mesh_ptr->shapes[shapeIndex].indices.size(); k += 3, i++)
		{
			index0 = mesh_ptr->shapes[shapeIndex].indices[k + 0];
			index1 = mesh_ptr->shapes[shapeIndex].indices[k + 1];
			index2 = mesh_ptr->shapes[shapeIndex].indices[k + 2];

			std::shared_ptr<MeshTriangle> tri;
			if (triType == smoothUVTri) {
				tri = std::make_shared<SmoothUVMeshTriangle>(mesh_ptr,
					index0, index1, index2, shapeIndex);
				tri->setMaterial(matVec[mesh_ptr->shapes[shapeIndex].materialIds[i]]);
				tri->setShadows(true);
			}
			else if (triType == flatUVTri) {
				tri = std::make_shared<FlatUVMeshTriangle>(mesh_ptr,
					index0, index1, index2, shapeIndex);
				tri->setMaterial(matVec[mesh_ptr->shapes[shapeIndex].materialIds[i]]);
				tri->setShadows(true);
			}
			if (triType == smoothTri) {
				tri = std::make_shared<SmoothMeshTriangle>(mesh_ptr,
					index0, index1, index2, shapeIndex);
				tri->setMaterial(matVec[mesh_ptr->shapes[shapeIndex].materialIds[i]]);
				tri->setShadows(true);
			}
			else if (triType == flatTri) {
				tri = std::make_shared<FlatMeshTriangle>(mesh_ptr,
					index0, index1, index2, shapeIndex);
				tri->setMaterial(matVec[mesh_ptr->shapes[shapeIndex].materialIds[i]]);
				tri->setShadows(true);
			}
			grid->addObject(tri);
			grid->addObject(tri);
		}
		grid->setupCells();
		grid->setShadows(true);
		compoundObj->addObject(grid);
		compoundObj->setShadows(true);
	}
	// scale, rotate, translate
	compoundObjInstance_ptr->scale(scale);
	compoundObjInstance_ptr->rotateX(rotateX);
	compoundObjInstance_ptr->rotateY(rotateY);
	compoundObjInstance_ptr->rotateZ(rotateZ);
	compoundObjInstance_ptr->translate(translate);
	compoundObjInstance_ptr->setShadows(true);
	world->scene.push_back(compoundObjInstance_ptr);
	return -1;
}

void render0Threads(std::shared_ptr<World> world, std::shared_ptr<Camera> camera)
{
	// Set up samplers
	std::shared_ptr<Sampler> sampler1 = std::make_shared<MultiJittered>(64, 83);

	world->image.resize(world->height * world->width);

	std::shared_ptr<Semaphore> sem{ std::make_shared <Semaphore>(4) };

	camera->renderScene(world, sampler1, 0, world->width, 0, world->height, 1, sem);
}

//numThreads must be a perfect square
void renderNThreads(std::shared_ptr<World> world, std::shared_ptr<Camera> camera, int numThreads, int numSamples, int sem_num)
{
	world->image.resize(world->height * world->width);
	size_t h_inc = world->height / (int)glm::sqrt(numThreads);
	size_t w_inc = world->width / (int)glm::sqrt(numThreads);
	int thread_num = 0;

	std::shared_ptr<Semaphore> sem{ std::make_shared <Semaphore>(sem_num) };

	std::vector<std::thread> threadVec;
	for (size_t h = 0; h <= world->height - h_inc; h += h_inc) {
		for (size_t w = 0; w <= world->width - w_inc; w += w_inc) {
			threadVec.push_back(std::thread(&Camera::renderScene, camera, std::ref(world), std::make_shared<MultiJittered>(numSamples, 83), w, w + w_inc, h, h + h_inc, thread_num, std::ref(sem)));
			thread_num++;
		}
	}

	for (std::thread& thr : threadVec)
	{
		if (thr.joinable())
			thr.join();
	}
}

int main()
{
	std::shared_ptr<World> world{ std::make_shared <World>() };

	// provide world data
	world->width = 1000;
	world->height = 1000;
	world->background = { 1,1,1 };
	world->tracer = std::make_shared<AreaLighting>(world);
	world->maxDepth = 2;

	int numThreads = 32;

	atlas::core::Timer<float> timer;
	timer.start();

	
	std::string meshPath{ ShaderPath };
	std::string matPath{ "Meshes/MyScene/" };
	std::string meshName{ "lampglassseats.obj" };
	matPath = meshPath + matPath;
	meshPath = matPath + meshName;
	if (!loadMesh(std::ref(world), meshPath, matPath, smoothUVTri, { 0.1,0.1,0.1 }, { 0,0,0 }, 0.0f, 0.0f, 0.0f))
	{
		return -100;
	}

	meshPath = { ShaderPath };
	matPath = { "Meshes/MyScene/" };
	meshName = { "windowglass.obj" };
	matPath = meshPath + matPath;
	meshPath = matPath + meshName;
	if (!loadMesh(std::ref(world), meshPath, matPath, smoothTri, { 0.1,0.1,0.1 }, { 0,0,0 }, 0.0f, 0.0f, 0.0f))
	{
		return -100;
	}
	
	meshPath = { ShaderPath };
	matPath = { "Meshes/MyScene/" };
	meshName = { "lamp.obj" };
	matPath = meshPath + matPath;
	meshPath = matPath + meshName;
	if (!loadMesh(std::ref(world), meshPath, matPath, smoothTri, { 0.1,0.1,0.1 }, { 0,0,0 }, 0.0f, 0.0f, 0.0f))
	{
		return -100;
	}

	meshPath = { ShaderPath };
	matPath = { "Meshes/MyScene/" };
	meshName = { "chair.obj" };
	matPath = meshPath + matPath;
	meshPath = matPath + meshName;
	if (!loadMesh(std::ref(world), meshPath, matPath, smoothTri, { 0.1,0.1,0.1 }, { 0,0,0 }, 0.0f, 0.0f, 0.0f))
	{
		return -100;
	}
	
	meshPath = { ShaderPath };
	matPath = { "Meshes/MyScene/" };
	meshName = { "mirrorframe.obj" };
	matPath = meshPath + matPath;
	meshPath = matPath + meshName;
	if (!loadMesh(std::ref(world), meshPath, matPath, smoothTri, { 0.1,0.1,0.1 }, { 0,0,0 }, 0.0f, 0.0f, 0.0f))
	{
		return -100;
	}

	meshPath = { ShaderPath };
	matPath = { "Meshes/MyScene/" };
	meshName = { "mirrorglass.obj" };
	matPath = meshPath + matPath;
	meshPath = matPath + meshName;
	if (!loadMesh(std::ref(world), meshPath, matPath, smoothTri, { 0.1,0.1,0.1 }, { 0,0,0 }, 0.0f, 0.0f, 0.0f))
	{
		return -100;
	}
	
	float t = timer.elapsed();
	fmt::print("load mesh time: {}\n", t);


	/****************************************************************************/
	/*                               LIGHTS                                     */
	/****************************************************************************/
	
	
	std::shared_ptr<Emissive> emissive_ptr = std::make_shared<Emissive>();
	emissive_ptr->scaleRadiance(120.0f);
	emissive_ptr->setColour({ 0.968f, 0.934f, 0.713f });

	std::vector<std::shared_ptr<Sampler>> rectangleSamplerVec;
	rectangleSamplerVec.reserve(numThreads);
	for (int i = 0; i < numThreads; i++) {
		rectangleSamplerVec.push_back(std::make_shared<MultiJittered>(64, 83));
	}

	std::shared_ptr<Rectangle> rectangle_ptr = std::make_shared<Rectangle>(
		atlas::math::Point{ 0, 8.5f, 0 }, glm::normalize(atlas::math::Vector{ 1, 2, 0 } /2.0f),
		glm::normalize(atlas::math::Vector{ -9,0, 8 }/2.0f));
	rectangle_ptr->setMaterial(emissive_ptr);
	rectangle_ptr->setSampler(rectangleSamplerVec);
	rectangle_ptr->setShadows(false);
	world->scene.push_back(rectangle_ptr);

	std::shared_ptr<AreaLight> areaLight_ptr = std::make_shared<AreaLight>();
	areaLight_ptr->setObject(rectangle_ptr);
	areaLight_ptr->setShadows(true);
	world->lights.push_back(areaLight_ptr);
	
	std::vector<std::shared_ptr<Sampler>> occlusionSamplerVec;
	occlusionSamplerVec.reserve(numThreads);
	for (int i = 0; i < numThreads; i++) {
		occlusionSamplerVec.push_back(std::make_shared<MultiJittered>(64, 83));
	}
	
	std::shared_ptr<Sampler> occluderSampler_ptr = std::make_shared<MultiJittered>(64, 83);
	std::shared_ptr<AmbientOccluder> occluder_ptr = std::make_shared<AmbientOccluder>();
	occluder_ptr->scaleRadiance(3.0);
	occluder_ptr->setColour({ 1,1,1 });
	occluder_ptr->setMinAmount({ 0.1,0.1,0.1 });
	occluder_ptr->setSampler(occlusionSamplerVec);
	world->ambient = occluder_ptr;

	world->lights.push_back(
		std::make_shared<PointLight>(atlas::math::Point{ 15, 9, 10 }, true));
	world->lights[0]->setColour({ .8, .8, 1 });
	world->lights[0]->scaleRadiance(4.0f);

	world->lights.push_back(
		std::make_shared<PointLight>(atlas::math::Point{ 0, 10, 20 }, true));
	world->lights[1]->setColour({ 1.0, 0.8, 0.3 });
	world->lights[1]->scaleRadiance(2.0f);

	//CAMERA
	std::shared_ptr<Pinhole> camera{ std::make_shared<Pinhole>() };
	camera->setEye({ -10.0f, 10.0f, 10.0f });
	camera->setLookAt({ 8, 0, -5.0f });
	camera->computeUVW();

	std::cout << "trace" << std::endl;
	timer.reset();
	timer.start();

	//render0Threads(world, camera);
	renderNThreads(world, camera, numThreads, 64, 8);

	t = timer.elapsed();
	fmt::print("render time: {}\n", t);

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
