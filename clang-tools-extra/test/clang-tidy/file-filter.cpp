// RUN: clang-tidy -checks=google-explicit-constructor -disable-checks='' -header-filter='' %s -- -I %S/Inputs/file-filter | FileCheck %s
// RUN: clang-tidy -checks=google-explicit-constructor -disable-checks='' -header-filter='.*' %s -- -I %S/Inputs/file-filter | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-tidy -checks=google-explicit-constructor -disable-checks='' -header-filter='header2\.h' %s -- -I %S/Inputs/file-filter | FileCheck --check-prefix=CHECK3 %s

#include "header1.h"
// CHECK-NOT: warning:
// CHECK2: header1.h:1:12: warning: Single-argument constructors must be explicit [google-explicit-constructor]
// CHECK3-NOT: warning:

#include "header2.h"
// CHECK-NOT: warning:
// CHECK2: header2.h:1:12: warning: Single-argument constructors {{.*}}
// CHECK3: header2.h:1:12: warning: Single-argument constructors {{.*}}

class A { A(int); };
// CHECK: :[[@LINE-1]]:11: warning: Single-argument constructors {{.*}}
// CHECK2: :[[@LINE-2]]:11: warning: Single-argument constructors {{.*}}
// CHECK3: :[[@LINE-3]]:11: warning: Single-argument constructors {{.*}}

// CHECK-NOT: warning:
// CHECK2-NOT: warning:
// CHECK3-NOT: warning:
