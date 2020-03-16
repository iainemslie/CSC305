#version 450 core

out vec4 fragColour;

uniform vec3 objectColor;
uniform vec3 lightColor;

void main()
{
    //fragColour = vec4(1.0, 1.0, 1.0, 1.0);
	fragColour = vec4(objectColor, 1.0);
}
