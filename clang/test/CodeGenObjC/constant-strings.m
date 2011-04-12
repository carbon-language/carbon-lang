// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=CHECK-NEXT < %t %s

// Check that we set alignment 1 on the string.
//
// CHECK-NEXT: @.str = {{.*}}constant [13 x i8] c"Hello World!\00", align 1

// RUN: %clang_cc1 -fgnu-runtime -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=CHECK-GNU < %t %s
// CHECK-GNU: NXConstantString
// CHECK-GNU-NOT: NXConstantString

// RUN: %clang_cc1 -fgnu-runtime -fconstant-string-class NSConstantString -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=CHECK-GNU-WITH-CLASS < %t %s
// CHECK-GNU-WITH-CLASS: NSConstantString
// CHECK-GNU-WITH-CLASS-NOT: NSConstantString
id a = @"Hello World!";

