// RUN: %clang -### -c -integrated-as %s 2>&1 | FileCheck %s

// CHECK: cc1as
// CHECK-NOT: -relax-all

// RUN: not %clang -c -integrated-as -Wa,--compress-debug-sections %s 2>&1 | FileCheck --check-prefix=INVALID %s
// INVALID: error: unsupported argument '--compress-debug-sections' to option 'Wa,'

// RUN: %clang -### -target x86_64-linux-gnu -c -integrated-as %s -fsanitize=address 2>&1 %s | FileCheck --check-prefix=SANITIZE %s
// SANITIZE: argument unused during compilation: '-fsanitize=address'
