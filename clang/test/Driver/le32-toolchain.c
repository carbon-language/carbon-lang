// RUN: %clang -### -target le32-unknown-nacl %s 2>&1 | FileCheck -check-prefix=CHECK-DEFAULT %s

// CHECK-DEFAULT: "-cc1" {{.*}} "-fno-math-builtin"

