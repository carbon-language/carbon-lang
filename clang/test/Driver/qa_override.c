// RUN: env QA_OVERRIDE_GCC3_OPTIONS="#+-Os +-Oz +-O +-O3 +-Oignore +a +b +c xb Xa Omagic ^-ccc-print-options  " %clang x -O2 b -O3 2>&1 | FileCheck %s
// RUN: env QA_OVERRIDE_GCC3_OPTIONS="x-Werror +-mfoo" %clang -Werror %s -c 2>&1 | FileCheck %s -check-prefix=RM-WERROR

// FIXME: It seems doesn't work with gcc-driver.
// REQUIRES: clang-driver

// CHECK-NOT: ###
// CHECK: Option 0 - Name: "-ccc-print-options", Values: {}
// CHECK-NEXT: Option 1 - Name: "<input>", Values: {"x"}
// CHECK-NEXT: Option 2 - Name: "-O", Values: {"ignore"}
// CHECK-NEXT: Option 3 - Name: "-O", Values: {"magic"}

// RM-WERROR: ### QA_OVERRIDE_GCC3_OPTIONS: x-Werror +-mfoo
// RM-WERROR-NEXT: ### Deleting argument -Werror
// RM-WERROR-NEXT: ### Adding argument -mfoo at end
// RM-WERROR-NEXT: warning: argument unused during compilation: '-mfoo'
