// RUN: rm %S/cl-strict-aliasing.ll
// RUN: %clang_cc1 -x cl -emit-llvm -cl-strict-aliasing < %s
