#undef sqrt
#define sqrt __clc_sqrt

#define FUNCTION __clc_sqrt
#define INTRINSIC "llvm.sqrt"
#include <clc/math/unary_intrin.inc>
