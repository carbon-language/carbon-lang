#undef cos
#define cos __clc_cos

#define __CLC_FUNCTION __clc_cos
#define __CLC_INTRINSIC "llvm.cos"
#include <clc/math/unary_intrin.inc>
