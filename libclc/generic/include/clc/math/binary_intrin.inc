_CLC_OVERLOAD float FUNCTION(float, float) __asm(INTRINSIC ".f32");
_CLC_OVERLOAD float2 FUNCTION(float2, float2) __asm(INTRINSIC ".v2f32");
_CLC_OVERLOAD float3 FUNCTION(float3, float3) __asm(INTRINSIC ".v3f32");
_CLC_OVERLOAD float4 FUNCTION(float4, float4) __asm(INTRINSIC ".v4f32");
_CLC_OVERLOAD float8 FUNCTION(float8, float8) __asm(INTRINSIC ".v8f32");
_CLC_OVERLOAD float16 FUNCTION(float16, float16) __asm(INTRINSIC ".v16f32");

#ifdef cl_khr_fp64
_CLC_OVERLOAD double FUNCTION(double, double) __asm(INTRINSIC ".f64");
_CLC_OVERLOAD double2 FUNCTION(double2, double2) __asm(INTRINSIC ".v2f64");
_CLC_OVERLOAD double3 FUNCTION(double3, double3) __asm(INTRINSIC ".v3f64");
_CLC_OVERLOAD double4 FUNCTION(double4, double4) __asm(INTRINSIC ".v4f64");
_CLC_OVERLOAD double8 FUNCTION(double8, double8) __asm(INTRINSIC ".v8f64");
_CLC_OVERLOAD double16 FUNCTION(double16, double16) __asm(INTRINSIC ".v16f64");
#endif

#undef FUNCTION
#undef INTRINSIC
