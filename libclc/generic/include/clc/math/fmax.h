#undef fmax
#define fmax __clc_fmax

#define BODY <clc/math/binary_decl.inc>
#define FUNCTION __clc_fmax

#include <clc/math/gentype.inc>

#undef BODY
#undef FUNCTION

