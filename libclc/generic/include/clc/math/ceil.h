#undef ceil
#define ceil __clc_ceil

#define FUNCTION __clc_ceil
#define INTRINSIC "llvm.ceil"
#include <clc/math/unary_intrin.inc>
