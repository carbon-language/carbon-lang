// RUN: %clang_cc1 -include Inputs/test3.h -E -H -o %t.out %s 2> %t.err
// RUN: FileCheck < %t.err %s

// CHECK-NOT: test3.h
// CHECK: . {{.*test.h}}
// CHECK: .. {{.*test2.h}}

// RUN: %clang_cc1 -include Inputs/test3.h -E --show-includes -o %t.out %s 2> %t.err
// RUN: FileCheck --check-prefix=MS < %t.err %s
// MS-NOT: test3.h
// MS: Note: including file: {{.*test.h}}
// MS: Note: including file:  {{.*test2.h}}
// MS-NOT: Note

#include "Inputs/test.h"
