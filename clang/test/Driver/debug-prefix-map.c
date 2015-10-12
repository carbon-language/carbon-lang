// RUN: %clang -### -fdebug-prefix-map=old %s 2>&1 | FileCheck %s -check-prefix CHECK-INVALID
// RUN: %clang -### -fdebug-prefix-map=old=new %s 2>&1 | FileCheck %s -check-prefix CHECK-SIMPLE
// RUN: %clang -### -fdebug-prefix-map=old=n=ew %s 2>&1 | FileCheck %s -check-prefix CHECK-COMPLEX
// RUN: %clang -### -fdebug-prefix-map=old= %s 2>&1 | FileCheck %s -check-prefix CHECK-EMPTY

// CHECK-INVALID: error: invalid argument 'old' to -fdebug-prefix-map
// CHECK-SIMPLE: fdebug-prefix-map=old=new
// CHECK-COMPLEX: fdebug-prefix-map=old=n=ew
// CHECK-EMPTY: fdebug-prefix-map=old=
