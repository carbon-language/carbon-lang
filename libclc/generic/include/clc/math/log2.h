#undef log2
#define log2 __clc_log2

#define FUNCTION __clc_log2
#define INTRINSIC "llvm.log2"
#include <clc/math/unary_intrin.inc>
