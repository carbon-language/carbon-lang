// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-apple-darwin10 | FileCheck %s

//int a; // FIXME: All names not in an extern "C" block are mangled

namespace N { int b; }
// CHECK: @"\01?b@N@@"
