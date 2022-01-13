// RUN: %clang --system-header-prefix=libs/ --no-system-header-prefix=libs/mylib/ -I%S/Inputs/SystemHeaderPrefix -Wundef -E %s 2>&1 | FileCheck %s
// RUN: %clang --system-header-prefix libs/ --no-system-header-prefix libs/mylib/ -I%S/Inputs/SystemHeaderPrefix -Wundef -E %s 2>&1 | FileCheck %s

#include "src/all.h"

// CHECK-NOT: BOOST
// CHECK: libs{{/|\\}}mylib{{/|\\}}warn.h:1:5: warning: 'MYLIB' is not defined, evaluates to 0
// CHECK-NOT: BOOST
// CHECK: libs{{/|\\}}mylib{{/|\\}}warn.h:1:5: warning: 'MYLIB' is not defined, evaluates to 0
// CHECK-NOT: BOOST
// CHECK: src{{/|\\}}warn.h:1:5: warning: 'SRC' is not defined, evaluates to 0
// CHECK-NOT: BOOST
