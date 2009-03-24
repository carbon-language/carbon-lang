// RUN: clang-cc -fsyntax-only -verify %s

typedef __attribute__(( ext_vector_type(2) )) float float2;
typedef __attribute__(( ext_vector_type(3) )) float float3;
typedef __attribute__(( ext_vector_type(4) )) float float4;

static void test() {
    float2 vec2, vec2_2;
    float3 vec3;
    float4 vec4, vec4_2, *vec4p;
    float f;

    vec2.z; // expected-error {{vector component access exceeds type 'float2'}}
    vec2.xyzw; // expected-error {{vector component access exceeds type 'float2'}}
    vec4.xyzw; // expected-warning {{expression result unused}}
    vec4.xyzc; // expected-error {{illegal vector component name 'c'}}
    vec4.s01z; // expected-error {{illegal vector component name 'z'}}
    vec2 = vec4.s01; // legal, shorten
    
    vec3 = vec4.xyz; // legal, shorten
    f = vec2.x; // legal, shorten
    f = vec4.xy.x; // legal, shorten

    vec2 = vec3.hi; // expected-error {{vector component access invalid for odd-sized type 'float3'}}
    
    vec4_2.xyzx = vec4.xyzw; // expected-error {{vector is not assignable (contains duplicate components)}}
    vec4_2.xyzz = vec4.xyzw; // expected-error {{vector is not assignable (contains duplicate components)}}
    vec4_2.xyyw = vec4.xyzw; // expected-error {{vector is not assignable (contains duplicate components)}}
    vec2.x = f;
    vec2.xx = vec2_2.xy; // expected-error {{vector is not assignable (contains duplicate components)}}
    vec2.yx = vec2_2.xy;
    vec4 = (float4){ 1,2,3,4 };
    vec4.xy.w; // expected-error {{vector component access exceeds type 'float2'}}
    vec4.s06; // expected-error {{vector component access exceeds type 'float4'}}
  
    vec4p->yz = vec4p->xy;
}
