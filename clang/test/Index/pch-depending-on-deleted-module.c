#include "a.h"

// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %clang_cc1 -x c-header -fmodules -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -emit-pch -I %S/Inputs/Headers -o %t/use_LibA.pch %s
// RUN: %clang_cc1 -fmodules -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -I %S/Inputs/Headers -verify-pch %t/use_LibA.pch
// RUN: rm -f %t/modules-cache/LibA.pcm
// RUN: not %clang_cc1 -fmodules -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -I %S/Inputs/Headers -verify-pch %t/use_LibA.pch 2>&1 | FileCheck -check-prefix=VERIFY %s
// RUN: not c-index-test -test-load-source all -x c -fmodules -Xclang -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -I %S/Inputs/Headers -include-pch %t/use_LibA.pch %s 2>&1 | FileCheck -check-prefix=INDEX %s

// VERIFY: fatal error: malformed or corrupted AST file: 'Unable to load module
// INDEX: {{^}}Failure: AST deserialization error occurred{{$}}

