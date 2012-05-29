#undef pow
#define pow __clc_pow

#define FUNCTION __clc_pow
#define INTRINSIC "llvm.pow"
#include <clc/math/binary_intrin.inc>
