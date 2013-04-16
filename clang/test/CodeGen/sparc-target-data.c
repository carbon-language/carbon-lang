// RUN: %clang -target sparc-sun-solaris   -o - -emit-llvm -S %s | FileCheck %s -check-prefix=V8
// RUN: %clang -target sparcv9-sun-solaris -o - -emit-llvm -S %s | FileCheck %s -check-prefix=V9

// V8: E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64
// V9: E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32:64-S128
