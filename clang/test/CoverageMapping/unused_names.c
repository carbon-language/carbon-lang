// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -emit-llvm -o - %s | FileCheck %s

// Since foo is never emitted, there should not be a profile name for it.

// CHECK-NOT: @__llvm_profile_name_foo =
// CHECK: @__llvm_profile_name_bar =
// CHECK-NOT: @__llvm_profile_name_foo =

#ifdef IS_SYSHEADER

#pragma clang system_header
inline int foo() { return 0; }

#else

#define IS_SYSHEADER
#include __FILE__

int bar() { return 0; }

#endif
