#undef fabs
#define fabs __clc_fabs

#define FUNCTION __clc_fabs
#define INTRINSIC "llvm.fabs"
#include <clc/math/unary_intrin.inc>
