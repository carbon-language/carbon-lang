// RUN: %clang -target xcore %s -### -o %t.o 2>&1 | FileCheck %s

// CHECK: "-momit-leaf-frame-pointer"
// CHECK-NOT: "-mdisable-fp-elim"
// CHECK: "-fno-signed-char"
// CHECK: "-fno-common"

