// RUN: clang -emit-llvm %s

typedef __attribute__(( ocu_vector_type(4) )) float float4;
//typedef __attribute__(( ocu_vector_type(3) )) float float3;
typedef __attribute__(( ocu_vector_type(2) )) float float2;


float4 test1(float4 V) {
  return V.wzyx+V;
}

float2 vec2, vec2_2;
float4 vec4, vec4_2;
float f;

static void test2() {
    vec2 = vec4.rg;  // shorten
    f = vec2.x;      // extract elt
    vec4 = vec4.yyyy;  // splat
    
    vec2.x = f;      // insert one.
    vec2.yx = vec2; // reverse
}

