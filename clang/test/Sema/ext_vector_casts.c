// RUN: clang-cc -fsyntax-only -verify %s

typedef __attribute__(( ext_vector_type(2) )) float float2;
typedef __attribute__(( ext_vector_type(4) )) int int4;
typedef __attribute__(( ext_vector_type(4) )) float float4;
typedef float t3 __attribute__ ((vector_size (16)));

static void test() {
    float2 vec2;
    float4 vec4, vec4_2;
    int4 ivec4;
    t3 vec4_3;
    
    vec4 = (float4)5.0f;
    vec4 = (float4)5;
    vec4 = (float4)vec4_3;
    
    ivec4 = (int4)5.0f;
    ivec4 = (int4)5;
    ivec4 = (int4)vec4_3;
    
    vec4 = (float4)vec2; // expected-error {{invalid conversion between ext-vector type 'float4' and 'float2'}}
}
