#undef fma
#define fma __clc_fma

#define FUNCTION __clc_fma
#define INTRINSIC "llvm.fma"
#include <clc/math/ternary_intrin.inc>
