// The 1 and # should not go on the same line.
// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s
// CHECK-NOT: 1{{.*}}#
// CHECK: {{^1$}}
// CHECK-NOT: 1{{.*}}#
// CHECK: {{^      #$}}
// CHECK-NOT: 1{{.*}}#
1
#define EMPTY
EMPTY #

