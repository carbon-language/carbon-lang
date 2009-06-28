// RUN: clang-cc -fsyntax-only -verify %s

typedef __attribute__(( ext_vector_type(2) )) float float2;
typedef __attribute__(( ext_vector_type(4) )) int int4;
typedef __attribute__(( ext_vector_type(8) )) short short8;
typedef __attribute__(( ext_vector_type(4) )) float float4;
typedef float t3 __attribute__ ((vector_size (16)));

static void test() {
    float2 vec2;
    float4 vec4, vec4_2;
    int4 ivec4;
    short8 ish8;
    t3 vec4_3;
    int *ptr;
    int i;
    
    vec4 = 5.0f;
    vec4 = (float4)5.0f;
    vec4 = (float4)5;
    vec4 = (float4)vec4_3;
    
    ivec4 = (int4)5.0f;
    ivec4 = (int4)5;
    ivec4 = (int4)vec4_3;
    
    i = (int)ivec4; // expected-error {{invalid conversion between vector type 'int4' and integer type 'int' of different size}}
    i = ivec4; // expected-error {{incompatible type assigning 'int4', expected 'int'}}
    
    ivec4 = (int4)ptr; // expected-error {{invalid conversion between vector type 'int4' and scalar type 'int *'}}
    
    vec4 = (float4)vec2; // expected-error {{invalid conversion between ext-vector type 'float4' and 'float2'}}
    
    ish8 += 5; // expected-error {{can't convert between vector values of different size ('short8' and 'int')}}
    ish8 += (short)5;
    ivec4 *= 5;
     vec4 /= 5.2f;
     vec4 %= 4; // expected-error {{invalid operands to binary expression ('float4' and 'int')}}
    ivec4 %= 4;
    ivec4 += vec4; // expected-error {{can't convert between vector values of different size ('float4' and 'int4')}}
    ivec4 += (int4)vec4;
    ivec4 -= ivec4;
    ivec4 |= ivec4;
    ivec4 += ptr; // expected-error {{can't convert between vector values of different size ('int4' and 'int *')}}
}
