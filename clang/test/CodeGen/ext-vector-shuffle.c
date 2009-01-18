// RUN: clang %s -emit-llvm -o - | not grep 'extractelement'
// RUN: clang %s -emit-llvm -o - | not grep 'insertelement'
// RUN: clang %s -emit-llvm -o - | grep 'shufflevector'

typedef __attribute__(( ext_vector_type(2) )) float float2;
typedef __attribute__(( ext_vector_type(4) )) float float4;

float2 test1(float4 V) {
  return V.xy + V.wz;
}

float4 test2(float4 V) {
  float2 W = V.ww;
  return W.xyxy + W.yxyx;
}
