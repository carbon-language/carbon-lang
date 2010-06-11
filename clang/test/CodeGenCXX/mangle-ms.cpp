// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-apple-darwin10 | FileCheck %s

int a;
// CHECK: @"\01?a@@"

namespace N { int b; }
// CHECK: @"\01?b@N@@"
