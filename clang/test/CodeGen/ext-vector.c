// RUN: %clang_cc1 -emit-llvm %s -o %t

typedef __attribute__(( ext_vector_type(4) )) float float4;
typedef __attribute__(( ext_vector_type(2) )) float float2;
typedef __attribute__(( ext_vector_type(4) )) int int4;

float4 foo = (float4){ 1.0, 2.0, 3.0, 4.0 };

const float4 bar = (float4){ 1.0, 2.0, 3.0, __builtin_inff() };

float4 test1(float4 V) {
  return V.wzyx+V;
}

float2 vec2, vec2_2;
float4 vec4, vec4_2;
float f;

void test2() {
    vec2 = vec4.xy;  // shorten
    f = vec2.x;      // extract elt
    vec4 = vec4.yyyy;  // splat
    
    vec2.x = f;      // insert one.
    vec2.yx = vec2; // reverse
}

void test3(float4 *out) {
  *out = ((float4) {1.0f, 2.0f, 3.0f, 4.0f });
}

void test4(float4 *out) {
  float a = 1.0f;
  float b = 2.0f;
  float c = 3.0f;
  float d = 4.0f;
  *out = ((float4) {a,b,c,d});
}

void test5(float4 *out) {
  float a;
  float4 b;
  
  a = 1.0f;
  b = a;
  b = b * 5.0f;
  b = 5.0f * b;
  b *= a;
  
  *out = b;
}

void test6(float4 *ap, float4 *bp, float c) {
  float4 a = *ap;
  float4 b = *bp;
  
  a = a + b;
  a = a - b;
  a = a * b;
  a = a / b;
  
  a = a + c;
  a = a - c;
  a = a * c;
  a = a / c;

  a += b;
  a -= b;
  a *= b;
  a /= b;
  
  a += c;
  a -= c;
  a *= c;
  a /= c;

  // Vector comparisons can sometimes crash the x86 backend: rdar://6326239,
  // reject them until the implementation is stable.
#if 0
  int4 cmp;
  cmp = a < b;
  cmp = a <= b;
  cmp = a < b;
  cmp = a >= b;
  cmp = a == b;
  cmp = a != b;
#endif
}

void test7(int4 *ap, int4 *bp, int c) {
  int4 a = *ap;
  int4 b = *bp;
  
  a = a + b;
  a = a - b;
  a = a * b;
  a = a / b;
  a = a % b;
  
  a = a + c;
  a = a - c;
  a = a * c;
  a = a / c;
  a = a % c;

  a += b;
  a -= b;
  a *= b;
  a /= b;
  a %= b;
  
  a += c;
  a -= c;
  a *= c;
  a /= c;
  a %= c;

  // Vector comparisons.
  int4 cmp;
  cmp = a < b;
  cmp = a <= b;
  cmp = a < b;
  cmp = a >= b;
  cmp = a == b;
  cmp = a != b;
}

void test8(float4 *ap, float4 *bp, int c) {
  float4 a = *ap;
  float4 b = *bp;

  // Vector comparisons.
  int4 cmp;
  cmp = a < b;
  cmp = a <= b;
  cmp = a < b;
  cmp = a >= b;
  cmp = a == b;
  cmp = a != b;
}
