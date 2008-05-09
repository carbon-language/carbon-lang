// RUN: clang -fsyntax-only -verify %s

typedef __attribute__(( ext_vector_type(2) )) float float2;
typedef __attribute__(( ext_vector_type(3) )) float float3;
typedef __attribute__(( ext_vector_type(4) )) float float4;

static void test() {
    float2 vec2, vec2_2;
    float3 vec3;
    float4 vec4, vec4_2;
    float f;

    vec2.z; // expected-error {{vector component access exceeds type 'float2'}}
    vec2.rgba; // expected-error {{vector component access exceeds type 'float2'}}
    vec4.rgba; // expected-warning {{expression result unused}}
    vec4.rgbc; // expected-error {{illegal vector component name 'c'}}
    vec3 = vec4.rgb; // legal, shorten
    f = vec2.x; // legal, shorten
    
    vec4_2.rgbr = vec4.rgba; // expected-error {{vector is not assignable (contains duplicate components)}}
    vec4_2.rgbb = vec4.rgba; // expected-error {{vector is not assignable (contains duplicate components)}}
    vec4_2.rgga = vec4.rgba; // expected-error {{vector is not assignable (contains duplicate components)}}
    vec2.x = f;
    vec2.xx = vec2_2.xy; // expected-error {{vector is not assignable (contains duplicate components)}}
    vec2.yx = vec2_2.xy;
    vec4 = (float4){ 1,2,3,4 };
    vec4.rg.a; // expected-error {{vector component access exceeds type 'float2'}}
}
