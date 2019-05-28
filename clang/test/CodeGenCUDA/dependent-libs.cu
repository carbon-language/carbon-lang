// RUN: %clang_cc1 -emit-llvm -o - -fcuda-is-device -x hip %s | FileCheck --check-prefix=DEV %s
// RUN: %clang_cc1 -emit-llvm -o - -x hip %s | FileCheck --check-prefix=HOST %s

// DEV-NOT: llvm.dependent-libraries
// HOST: llvm.dependent-libraries
#pragma comment(lib, "libabc")
