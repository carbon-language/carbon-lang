#define FLOAT float

#define FLOATN float
#include BODY
#undef FLOATN

#define FLOATN float2
#include BODY
#undef FLOATN

#define FLOATN float3
#include BODY
#undef FLOATN

#define FLOATN float4
#include BODY
#undef FLOATN

#undef FLOAT

#ifdef cl_khr_fp64

#define FLOAT double

#define FLOATN double
#include BODY
#undef FLOATN

#define FLOATN double2
#include BODY
#undef FLOATN

#define FLOATN double3
#include BODY
#undef FLOATN

#define FLOATN double4
#include BODY
#undef FLOATN

#undef FLOAT

#endif

#undef BODY
