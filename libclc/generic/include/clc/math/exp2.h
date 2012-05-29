#undef exp2
#define exp2 __clc_exp2

#define FUNCTION __clc_exp2
#define INTRINSIC "llvm.exp2"
#include <clc/math/unary_intrin.inc>
