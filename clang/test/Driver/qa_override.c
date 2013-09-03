// RUN: env QA_OVERRIDE_GCC3_OPTIONS="#+-Os +-Oz +-O +-O3 +-Oignore +a +b +c xb Xa Omagic ^-###  " %clang %s -O2 b -O3 2>&1 | FileCheck %s
// RUN: env QA_OVERRIDE_GCC3_OPTIONS="x-Werror +-mfoo" %clang -Werror %s -c -### 2>&1 | FileCheck %s -check-prefix=RM-WERROR

// CHECK: "-cc1"
// CHECK-NOT: "-Oignore"
// CHECK: "-Omagic"
// CHECK-NOT: "-Oignore"

// RM-WERROR: ### QA_OVERRIDE_GCC3_OPTIONS: x-Werror +-mfoo
// RM-WERROR-NEXT: ### Deleting argument -Werror
// RM-WERROR-NEXT: ### Adding argument -mfoo at end
// RM-WERROR: warning: argument unused during compilation: '-mfoo'
// RM-WERROR-NOT: "-Werror"
