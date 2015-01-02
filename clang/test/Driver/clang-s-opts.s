// RUN: %clang -### -c -Werror -flto -fno-lto %s 2>&1 | FileCheck %s

// CHECK-NOT: argument unused during compilation
