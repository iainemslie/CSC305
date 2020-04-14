CSC 305 Spring 2020 Assignment4 ReadMe
Iain Emslie

My raytracing program implements all of the basic features listed below:
 - A virtual pinhole camera
 - Point Lights - shining light from window
 - Diffuse shading
 - Multisampling - all of the techniques discussed in class
 - Out of gamut colour handling through clamping
 - A variety of geometries. Planes, spheres, cylinders, axisboxes, triangles (uv mesh with smooth and flat shading)
   Disks, Rectangles. I only used triangle meshes in my .bmp file.
   
The following advanced features are implemented:

 - Intermediate Parallelization - The user is able to specify how many threads to use to
 render the image. The image is then divided into segments based on this number. This allows
 the user to specify the number of threads to use which can be more than the system can run at
 one time. A semaphore is used to only allow access to the rendering functions to
 to a specified number of threads at one time.
 
 - Regular Grids - These are used in triangle meshes to speed up the rendering times.
   While implementing this feature I loaded the Suzanne mesh. Before using the grids it took
   2 hours to render (while using threads). After implementing the grid it took 30 seconds.
   All of the meshes in my scene use regular grids.
 
 - Mirror Reflections - Mirror reflections are implemented as a planar mesh object. It is featured
   in the picture frame on the wall.
 
 - Glossy Reflections - Glossy reflections are implemented, however they do not appear in this scene.
 
 - Simple Transparency - Simple transparency is implemented as the window pane.
 
 - Regular Texture Mapping - Many of the objects in the scene feature texture mapping.
 
 - Shadows - Shadows are implemented
 
 - Ambient Occlusion - Ambient occlusion is implemented.
 
 - Area Lights - Area lights are used for the lamp light on the table
 
 - Rendering a Mesh - Every object in the scene is a mesh.
 
 
 All of my meshes were downloaded for free from turbosquid.combined
 The photos I used were free and found at the following locations:
 
 https://unsplash.com/s/photos/night-skyline
 
 /****************************************************************************/
 The following section includes the notes I took while implementing the threads.
 
 This file contains a list of timing tests using no parallelism, 4 threads and 8 threads.
My personal computer has an Intel Core i7-3615QM with
4 cores and supports up to 8 threads using hyperthreading.
I first implemented my multithreading using the Lab3 solution code and these numbers are based on that.
This simply divides each image into either 4 or 8 rectangles and has each thread work on them.
No mutual exclusion is necessary in this case because they are each working on their own part of the image.

Without Multithreading:
Test 1: 21.112982
Test 2: 19.119055
Test 3: 19.112171
Test 4: 19.145771
Test 5: 19.235718

4 Threads:
Test 1: 9.248477
Test 2: 9.419649
Test 3: 9.327262
Test 4: 9.824896
Test 5: 9.427071

8 Threads:
Test 1: 16.171965
Test 2: 15.287174
Test 3: 15.080109
Test 4: 15.159712
Test 5: 15.321374

It's clear that in this case 4 threads is the best option with almost a 50% speedup compared to 
not using multithreading.
I'm interested to see whether using 8 threads is more efficient on more busier scenes because
the lab3 test I used only has 2 spheres in one part of the scene.

I thought there might be problems using the same sampler in multiple threads. I thought this was
due to a limited number of samples being generated. From Physically Based Rendering it notes that
sharing a sampler can actually make the rendering slower. I had originally made one sampler for each
thread but only used one of them for each when I found it worked without problem. However after
using one sampler for each thread I have got another speedup on the existing code to around 6 seconds.

I combined Labs 3 and 4 together with threading and tested it to get the following results:

Without Multithreading:
Test 1: 33.519695
Test 2: 33.428356
Test 3: 34.17953
Test 4: 33.36997
Test 5: 33.37724

4 Threads:
Test 1: 10.50768
Test 2: 10.520283
Test 3: 9.943089
Test 4: 10.051146
Test 5: 9.986922

8 Threads:
Test 1: 20.123848
Test 2: 20.063202
Test 3: 19.962122
Test 4: 20.032042
Test 5: 21.047014

It's clear that using 4 threads is still the best option and around 3 times as fast as
using a single thread of execution.

To satisfy the 2nd requirement of the threading component I have implemented a semaphore class.
This splits the image into 8 slabs with 4 threads working at a time. When a thread has finished
execution it notifies the other threads and allows the waiting threads to begin executing.

So far using 4 threads is still working the best. But I think it may be better in busier scenes.
The implementation using semaphores works well when dividing my test image into 16 quadrants.
My first result was comparable to just using 4 threads.
Using 16 threads with a semaphore allowing 8 at a time (hyperthreading) I got my best result yet.
Better than just 4 threads at a time.
I tried with 32 threads and got a similar result to using 16.

I'm going to try implementing an initial trace to calculate the busiest quadrant and then subdivide
that quadrant further.
 
