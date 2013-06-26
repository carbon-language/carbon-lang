#undef fmin
#define fmin __clc_fmin

#define BODY <clc/math/binary_decl.inc>
#define FUNCTION __clc_fmin

#include <clc/math/gentype.inc>

#undef BODY
#undef FUNCTION

