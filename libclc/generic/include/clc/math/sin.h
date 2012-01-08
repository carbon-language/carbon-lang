#undef sin
#define sin __clc_sin

#define FUNCTION __clc_sin
#define INTRINSIC "llvm.sin"
#include <clc/math/unary_intrin.inc>
