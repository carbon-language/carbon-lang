#include <clc/clc.h>

_CLC_OVERLOAD _CLC_DEF float dot(float p0, float p1) {
  return p0*p1;
}

_CLC_OVERLOAD _CLC_DEF float dot(float2 p0, float2 p1) {
  return p0.x*p1.x + p0.y*p1.y;
}

_CLC_OVERLOAD _CLC_DEF float dot(float3 p0, float3 p1) {
  return p0.x*p1.x + p0.y*p1.y + p0.z*p1.z;
}

_CLC_OVERLOAD _CLC_DEF float dot(float4 p0, float4 p1) {
  return p0.x*p1.x + p0.y*p1.y + p0.z*p1.z + p0.w*p1.w;
}
