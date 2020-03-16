/* Iain's 305 Assignment 3 README */

I based my program on the class tutorials and learnopengl.com

I have implemented the following features:

1. A scene containing a cube - the cube vertices and normals are hard coded as 
   a vector that I borrowed from learnopengl.com.
   This cube has its own class so that multiple instances of a cube can be created 
   each having their own shaders, colours, or other settings available.
   This inherits from a base class SceneObject which is the basis for all objects
   that are rendered in a scene.
   
2. I have implemented a pinhole camera class in assignment.hpp that uses perspective projection

3. I created a point light class. This can be used to render an object
   in the scene to represent the position of the light. However you can simply specify
   the position of a point light in each SceneObject as well.
   
4. Diffuse shading is implemented in my cube.frag and cube.vert shaders.
   It uses the normal for each vertex to calculate diffuse shading.
   The logic is similar to that used in the ray tracing assignment.
   We sum up the ambient and diffuse components for the final output result.
   
Advanced Features

- Specular Reflection - is implemented in the cube.frag shader. It uses the same logic
talked about in class and simply adds the specular component to the overall colour value
for each fragment. It was again based on learnopengl.com as well. It is possible to turn
this feature off for SceneObjects 0, 1 and 2 using the number (not numpad) keys 0-3.
This works by loading in a new shader file. This prints errors to console but works fine.

- First Person View Camera - The camera is implemented as specified and uses the WASD keys for
movement forwards, left, backwards and right respectively. The mouse is used to change the pitch
and yaw angles. This was based on the camera from learnopengl.com as well as the tutorial to create
the lambda function used for mouse input. Keyboard input was implemented differently so that it is
checked for every iteration of the render loop. This made it easier to have smooth camera movement.

- Load and render a simple mesh - It is possible for the user to load the suzanne.obj and
bunny.obj into the scene. It is assumed that they are contained in the same folder as the 
shaders. I have hardcoded their paths into the paths.hpp file which is included.
It is possible to turn on and off rendering for each object in the scene using the numpad keys from 0 - 3
Wireframe mode can be turned on and off by using the spacebar.
These are implemented using the keyboard callback since it gives better responsiveness for single key presses.
It is possible to turn each SceneObject's rendering off using the numpad 0, 1, 2 and 3 keys.
