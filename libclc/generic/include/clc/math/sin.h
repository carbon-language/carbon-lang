#undef sin
#define sin __clc_sin

#define __CLC_FUNCTION __clc_sin
#define __CLC_INTRINSIC "llvm.sin"
#include <clc/math/unary_intrin.inc>
