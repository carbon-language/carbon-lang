#define __CLC_GENTYPE float
#include BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float2
#include BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float3
#include BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float4
#include BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float8
#include BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float16
#include BODY
#undef __CLC_GENTYPE

#ifdef cl_khr_fp64
#define __CLC_GENTYPE double
#include BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double2
#include BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double3
#include BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double4
#include BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double8
#include BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double16
#include BODY
#undef __CLC_GENTYPE
#endif

#undef BODY
