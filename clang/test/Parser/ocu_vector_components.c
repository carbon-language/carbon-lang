// RUN: clang -parse-ast-check %s

typedef __attribute__(( ocu_vector_type(2) )) float float2;
typedef __attribute__(( ocu_vector_type(3) )) float float3;
typedef __attribute__(( ocu_vector_type(4) )) float float4;

static void test() {
    float2 vec2;
    float3 vec3;
    float4 vec4;
    float f;

    vec2.z; // // expected-error {{vector component access exceeds type 'float2'}}
    vec2.rgba; // // expected-error {{vector component access exceeds type 'float2'}}
    vec4.rgba;
    vec4.rgbc; // // expected-error {{illegal vector component name 'c'}}
    vec3 = vec4.rgb; // legal, shorten
    f = vec2.x; // legal, shorten
    vec4 = (float4){ 1,2,3,4 };
}
