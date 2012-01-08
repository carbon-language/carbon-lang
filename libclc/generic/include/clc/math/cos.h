#undef cos
#define cos __clc_cos

#define FUNCTION __clc_cos
#define INTRINSIC "llvm.cos"
#include <clc/math/unary_intrin.inc>
