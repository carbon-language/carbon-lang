// RUN: %clang_cc1 -triple i386-apple-darwin9 -emit-llvm -o - %s | FileCheck %s --check-prefix=REG
// RUN: %clang_cc1 -triple i386-apple-darwin9 -fpcc-struct-return -emit-llvm -o - %s | FileCheck %s --check-prefix=PCC
// RUN: %clang_cc1 -triple i386-apple-darwin9 -freg-struct-return -emit-llvm -o - %s | FileCheck %s --check-prefix=REG
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=PCC
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -fpcc-struct-return -emit-llvm -o - %s | FileCheck %s --check-prefix=PCC
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -freg-struct-return -emit-llvm -o - %s | FileCheck %s --check-prefix=REG
// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm -o - %s | FileCheck %s --check-prefix=REG
// RUN: %clang_cc1 -triple i386-pc-win32 -fpcc-struct-return -emit-llvm -o - %s | FileCheck %s --check-prefix=PCC
// RUN: %clang_cc1 -triple i386-pc-win32 -freg-struct-return -emit-llvm -o - %s | FileCheck %s --check-prefix=REG

typedef struct { int a,b,c,d; } Big;
typedef struct { int i; } Small;
typedef struct { short s; } Short;
typedef struct { } ZeroSized;

// CHECK: define void @returnBig
// CHECK: ret void
Big returnBig(Big x) { return x; }

// CHECK-PCC: define void @returnSmall
// CHECK-PCC: ret void
// CHECK-REG: define i32 @returnSmall
// CHECK-REG: ret i32
Small returnSmall(Small x) { return x; }

// CHECK-PCC: define void @returnShort
// CHECK-PCC: ret void
// CHECK-REG: define i16 @returnShort
// CHECK-REG: ret i16
Short returnShort(Short x) { return x; }

// CHECK: define void @returnZero()
// CHECK: ret void
ZeroSized returnZero(ZeroSized x) { return x; }
