// RUN: %clang -target mipsel-linux-gnu -o - -emit-llvm -S %s |\
// RUN: FileCheck %s -check-prefix=32EL
// RUN: %clang -target mips-linux-gnu -o - -emit-llvm -S %s |\
// RUN: FileCheck %s -check-prefix=32EB
// RUN: %clang -target mips64el-linux-gnu -o - -emit-llvm -S %s |\
// RUN: FileCheck %s -check-prefix=64EL
// RUN: %clang -target mips64-linux-gnu -o - -emit-llvm -S %s |\
// RUN: FileCheck %s -check-prefix=64EB
// RUN: %clang -target mipsel-linux-gnu -o - -emit-llvm -S -mfp64 %s |\
// RUN: FileCheck %s -check-prefix=32EL

// 32EL: e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64
// 32EB: E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64
// 64EL: e-p:64:64:64-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v64:64:64-n32:64-S128
// 64EB: E-p:64:64:64-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v64:64:64-n32:64-S128

