#define SCALAR_GENTYPE float

#define GENTYPE float
#define SCALAR
#include BODY
#undef GENTYPE
#undef SCALAR

#define GENTYPE float2
#include BODY
#undef GENTYPE

#define GENTYPE float3
#include BODY
#undef GENTYPE

#define GENTYPE float4
#include BODY
#undef GENTYPE

#define GENTYPE float8
#include BODY
#undef GENTYPE

#define GENTYPE float16
#include BODY
#undef GENTYPE

#undef SCALAR_GENTYPE

#ifdef cl_khr_fp64
#define SCALAR_GENTYPE double

#define SCALAR
#define GENTYPE double
#include BODY
#undef GENTYPE
#undef SCALAR

#define GENTYPE double2
#include BODY
#undef GENTYPE

#define GENTYPE double3
#include BODY
#undef GENTYPE

#define GENTYPE double4
#include BODY
#undef GENTYPE

#define GENTYPE double8
#include BODY
#undef GENTYPE

#define GENTYPE double16
#include BODY
#undef GENTYPE

#undef SCALAR_GENTYPE
#endif

#undef BODY
