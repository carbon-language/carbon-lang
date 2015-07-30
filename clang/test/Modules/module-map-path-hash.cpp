// rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -Rmodule-build -I%S/Inputs/module-map-path-hash -fmodules-cache-path=%t -fsyntax-only %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -Rmodule-build -I%S/Inputs//module-map-path-hash -fmodules-cache-path=%t -fsyntax-only %s 2>&1 | FileCheck -allow-empty %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -Rmodule-build -I%S/Inputs/./module-map-path-hash -fmodules-cache-path=%t -fsyntax-only %s 2>&1 | FileCheck -allow-empty %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -Rmodule-build -I%S/Inputs/../Inputs/module-map-path-hash -fmodules-cache-path=%t -fsyntax-only %s 2>&1 | FileCheck -allow-empty %s

#include "a.h"

// CHECK-NOT: remark: building module
