// RUN: %check_clang_tidy -std=c++11-or-later %s modernize-deprecated-headers %t -- -extra-arg-before=-isystem%S/Inputs/modernize-deprecated-headers

#include <assert.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'assert.h'; consider using 'cassert' instead [modernize-deprecated-headers]
// CHECK-FIXES: {{^}}#include <cassert>{{$}}
#include <complex.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'complex.h'; consider using 'complex' instead
// CHECK-FIXES: {{^}}#include <complex>{{$}}
#include <ctype.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'ctype.h'; consider using 'cctype' instead
// CHECK-FIXES: {{^}}#include <cctype>{{$}}
#include <errno.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'errno.h'; consider using 'cerrno' instead
// CHECK-FIXES: {{^}}#include <cerrno>{{$}}
#include <fenv.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'fenv.h'; consider using 'cfenv' instead
// CHECK-FIXES: {{^}}#include <cfenv>{{$}}
#include <float.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'float.h'; consider using 'cfloat' instead
// CHECK-FIXES: {{^}}#include <cfloat>{{$}}
#include <inttypes.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'inttypes.h'; consider using 'cinttypes' instead
// CHECK-FIXES: {{^}}#include <cinttypes>{{$}}
#include <limits.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'limits.h'; consider using 'climits' instead
// CHECK-FIXES: {{^}}#include <climits>{{$}}
#include <locale.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'locale.h'; consider using 'clocale' instead
// CHECK-FIXES: {{^}}#include <clocale>{{$}}
#include <math.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'math.h'; consider using 'cmath' instead
// CHECK-FIXES: {{^}}#include <cmath>{{$}}
#include <setjmp.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'setjmp.h'; consider using 'csetjmp' instead
// CHECK-FIXES: {{^}}#include <csetjmp>{{$}}
#include <signal.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'signal.h'; consider using 'csignal' instead
// CHECK-FIXES: {{^}}#include <csignal>{{$}}
#include <stdarg.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'stdarg.h'; consider using 'cstdarg' instead
// CHECK-FIXES: {{^}}#include <cstdarg>{{$}}
#include <stddef.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'stddef.h'; consider using 'cstddef' instead
// CHECK-FIXES: {{^}}#include <cstddef>{{$}}
#include <stdint.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'stdint.h'; consider using 'cstdint' instead
// CHECK-FIXES: {{^}}#include <cstdint>{{$}}
#include <stdio.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'stdio.h'; consider using 'cstdio' instead
// CHECK-FIXES: {{^}}#include <cstdio>{{$}}
#include <stdlib.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'stdlib.h'; consider using 'cstdlib' instead
// CHECK-FIXES: {{^}}#include <cstdlib>{{$}}
#include <string.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'string.h'; consider using 'cstring' instead
// CHECK-FIXES: {{^}}#include <cstring>{{$}}
#include <tgmath.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'tgmath.h'; consider using 'ctgmath' instead
// CHECK-FIXES: {{^}}#include <ctgmath>{{$}}
#include <time.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'time.h'; consider using 'ctime' instead
// CHECK-FIXES: {{^}}#include <ctime>{{$}}
#include <uchar.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'uchar.h'; consider using 'cuchar' instead
// CHECK-FIXES: {{^}}#include <cuchar>{{$}}
#include <wchar.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'wchar.h'; consider using 'cwchar' instead
// CHECK-FIXES: {{^}}#include <cwchar>{{$}}
#include <wctype.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'wctype.h'; consider using 'cwctype' instead
// CHECK-FIXES: {{^}}#include <cwctype>

// Headers that have no effect in C++; remove them
#include <stdalign.h> // <stdalign.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdalign.h' has no effect in C++; consider removing it
// CHECK-FIXES: {{^}}// <stdalign.h>{{$}}
#include <stdbool.h> // <stdbool.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdbool.h' has no effect in C++; consider removing it
// CHECK-FIXES: {{^}}// <stdbool.h>{{$}}
#include <iso646.h> // <iso646.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'iso646.h' has no effect in C++; consider removing it
// CHECK-FIXES: {{^}}// <iso646.h>{{$}}

#include "assert.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'assert.h'; consider using 'cassert' instead
// CHECK-FIXES: {{^}}#include <cassert>{{$}}
#include "complex.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'complex.h'; consider using 'complex' instead
// CHECK-FIXES: {{^}}#include <complex>{{$}}
#include "ctype.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'ctype.h'; consider using 'cctype' instead
// CHECK-FIXES: {{^}}#include <cctype>{{$}}
#include "errno.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'errno.h'; consider using 'cerrno' instead
// CHECK-FIXES: {{^}}#include <cerrno>{{$}}
#include "fenv.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'fenv.h'; consider using 'cfenv' instead
// CHECK-FIXES: {{^}}#include <cfenv>{{$}}
#include "float.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'float.h'; consider using 'cfloat' instead
// CHECK-FIXES: {{^}}#include <cfloat>{{$}}
#include "inttypes.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'inttypes.h'; consider using 'cinttypes' instead
// CHECK-FIXES: {{^}}#include <cinttypes>{{$}}
#include "limits.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'limits.h'; consider using 'climits' instead
// CHECK-FIXES: {{^}}#include <climits>{{$}}
#include "locale.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'locale.h'; consider using 'clocale' instead
// CHECK-FIXES: {{^}}#include <clocale>{{$}}
#include "math.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'math.h'; consider using 'cmath' instead
// CHECK-FIXES: {{^}}#include <cmath>{{$}}
#include "setjmp.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'setjmp.h'; consider using 'csetjmp' instead
// CHECK-FIXES: {{^}}#include <csetjmp>{{$}}
#include "signal.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'signal.h'; consider using 'csignal' instead
// CHECK-FIXES: {{^}}#include <csignal>{{$}}
#include "stdarg.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'stdarg.h'; consider using 'cstdarg' instead
// CHECK-FIXES: {{^}}#include <cstdarg>{{$}}
#include "stddef.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'stddef.h'; consider using 'cstddef' instead
// CHECK-FIXES: {{^}}#include <cstddef>{{$}}
#include "stdint.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'stdint.h'; consider using 'cstdint' instead
// CHECK-FIXES: {{^}}#include <cstdint>{{$}}
#include "stdio.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'stdio.h'; consider using 'cstdio' instead
// CHECK-FIXES: {{^}}#include <cstdio>{{$}}
#include "stdlib.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'stdlib.h'; consider using 'cstdlib' instead
// CHECK-FIXES: {{^}}#include <cstdlib>{{$}}
#include "string.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'string.h'; consider using 'cstring' instead
// CHECK-FIXES: {{^}}#include <cstring>{{$}}
#include "tgmath.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'tgmath.h'; consider using 'ctgmath' instead
// CHECK-FIXES: {{^}}#include <ctgmath>{{$}}
#include "time.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'time.h'; consider using 'ctime' instead
// CHECK-FIXES: {{^}}#include <ctime>{{$}}
#include "uchar.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'uchar.h'; consider using 'cuchar' instead
// CHECK-FIXES: {{^}}#include <cuchar>{{$}}
#include "wchar.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'wchar.h'; consider using 'cwchar' instead
// CHECK-FIXES: {{^}}#include <cwchar>{{$}}
#include "wctype.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'wctype.h'; consider using 'cwctype' instead
// CHECK-FIXES: {{^}}#include <cwctype>

// Headers that have no effect in C++; remove them
#include "stdalign.h" // "stdalign.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdalign.h' has no effect in C++; consider removing it
// CHECK-FIXES: {{^}}// "stdalign.h"{{$}}
#include "stdbool.h" // "stdbool.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdbool.h' has no effect in C++; consider removing it
// CHECK-FIXES: {{^}}// "stdbool.h"{{$}}
#include "iso646.h" // "iso646.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'iso646.h' has no effect in C++; consider removing it
// CHECK-FIXES: {{^}}// "iso646.h"{{$}}
