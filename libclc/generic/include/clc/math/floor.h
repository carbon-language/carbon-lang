#undef floor
#define floor __clc_floor

#define FUNCTION __clc_floor
#define INTRINSIC "llvm.floor"
#include <clc/math/unary_intrin.inc>
