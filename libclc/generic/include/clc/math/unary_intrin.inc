_CLC_OVERLOAD float FUNCTION(float f) __asm(INTRINSIC ".f32");
_CLC_OVERLOAD float2 FUNCTION(float2 f) __asm(INTRINSIC ".v2f32");
_CLC_OVERLOAD float3 FUNCTION(float3 f) __asm(INTRINSIC ".v3f32");
_CLC_OVERLOAD float4 FUNCTION(float4 f) __asm(INTRINSIC ".v4f32");
_CLC_OVERLOAD float8 FUNCTION(float8 f) __asm(INTRINSIC ".v8f32");
_CLC_OVERLOAD float16 FUNCTION(float16 f) __asm(INTRINSIC ".v16f32");

#ifdef cl_khr_fp64
_CLC_OVERLOAD double FUNCTION(double d) __asm(INTRINSIC ".f64");
_CLC_OVERLOAD double2 FUNCTION(double2 d) __asm(INTRINSIC ".v2f64");
_CLC_OVERLOAD double3 FUNCTION(double3 d) __asm(INTRINSIC ".v3f64");
_CLC_OVERLOAD double4 FUNCTION(double4 d) __asm(INTRINSIC ".v4f64");
_CLC_OVERLOAD double8 FUNCTION(double8 d) __asm(INTRINSIC ".v8f64");
_CLC_OVERLOAD double16 FUNCTION(double16 d) __asm(INTRINSIC ".v16f64");
#endif

#undef FUNCTION
#undef INTRINSIC
