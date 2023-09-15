#version 410

layout (location=0) in vec4 vPosition;
layout (location=1) in vec2 vTexCoord;
layout (location=2) in vec4 vNormal;
layout (location=3) in vec4 vWeight;
layout (location=4) in vec4 vWID;

const int MAX_BONES = 100;
const int MAX_BONES_INF = 4;

uniform mat4 BB[MAX_BONES];
uniform mat4 MM[MAX_BONES];

uniform mat4 modelViewProj;
uniform mat4 model;

out     vec4 pos;
out     vec2 fragmentTexCoord;
out     vec3 normal;

void main()
{         
    vec4 newv = vec4(0.0f);

    for (int i = 0; i < MAX_BONES_INF; i++) 
    {
        if(int(vWID[i]) >= 0) 
        {   
            mat4 mat = BB[int(vWID[i])] * MM[int(vWID[i])] ;
            vec4 temp = vPosition * mat;
            newv += vWeight[i] * temp;
            //normal = mat3(transpose(inverse(model*mat))) * vNormal.xyz;
            normal = mat3(transpose(inverse(model)))*mat3(mat) * vNormal.xyz;
        }
    }

    gl_Position = modelViewProj * newv;
    pos = model * newv;
    fragmentTexCoord = vTexCoord;
}
