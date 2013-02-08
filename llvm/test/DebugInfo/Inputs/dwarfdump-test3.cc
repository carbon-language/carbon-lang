#include "dwarfdump-test3-decl.h"

C::C(bool a, bool b) {}

// Built with gcc 4.6.3
// $ mkdir -p /tmp/dbginfo/include
// $ mkdir -p /tmp/include
// $ cp dwarfdump-test3.cc /tmp/dbginfo
// $ cp dwarfdump-test3-decl.h /tmp/include
// $ cp dwarfdump-test3-decl2.h /tmp/dbginfo/include
// $ cd /tmp/dbginfo
// $ gcc dwarfdump-test3.cc -g -I/tmp/include -Iinclude -fPIC -shared -o <output>
