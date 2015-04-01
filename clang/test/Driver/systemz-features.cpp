
// RUN: %clang -target s390x-unknown-linux-gnu %s -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-target-feature" "+transactional-execution"
// CHECK-DEFAULT-NOT: "-target-feature" "-transactional-execution"

// RUN: %clang -target s390x-unknown-linux-gnu %s -mhtm -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-HTM %s
// RUN: %clang -target s390x-unknown-linux-gnu %s -mno-htm -mhtm -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-HTM %s
// CHECK-HTM: "-target-feature" "+transactional-execution"
// CHECK-HTM-NOT: "-target-feature" "-transactional-execution"

// RUN: %clang -target s390x-unknown-linux-gnu %s -mno-htm -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOHTM %s
// RUN: %clang -target s390x-unknown-linux-gnu %s -mhtm -mno-htm -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOHTM %s
// CHECK-NOHTM: "-target-feature" "-transactional-execution"
// CHECK-NOHTM-NOT: "-target-feature" "+transactional-execution"

