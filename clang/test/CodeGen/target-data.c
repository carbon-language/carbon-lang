// RUN: %clang_cc1 -triple i686-unknown-unknown -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=I686-UNKNOWN %s
// I686-UNKNOWN: target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a:0:64-f80:32:32-n8:16:32-S128"

// RUN: %clang_cc1 -triple i686-apple-darwin9 -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=I686-DARWIN %s
// I686-DARWIN: target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a:0:64-f80:128:128-n8:16:32-S128"

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=X86_64 %s
// X86_64: target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a:0:64-s:64:64-f80:128:128-n8:16:32:64-S128"

// RUN: %clang_cc1 -triple xcore-unknown-unknown -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=XCORE %s
// XCORE: target datalayout = "e-p:32:32:32-a:0:32-n32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f16:16:32-f32:32:32-f64:32:32"

// RUN: %clang_cc1 -triple sparc-sun-solaris -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefix=SPARC-V8
// SPARC-V8: target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"

// RUN: %clang_cc1 -triple sparcv9-sun-solaris -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefix=SPARC-V9
// SPARC-V9: target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32:64-S128"

// RUN: %clang_cc1 -triple mipsel-linux-gnu -o - -emit-llvm %s |     \
// RUN: FileCheck %s -check-prefix=MIPS-32EL
// MIPS-32EL: target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"

// RUN: %clang_cc1 -triple mips-linux-gnu -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MIPS-32EB
// MIPS-32EB: target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"

// RUN: %clang_cc1 -triple mips64el-linux-gnu -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MIPS-64EL
// MIPS-64EL: target datalayout = "e-p:64:64:64-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v64:64:64-n32:64-S128"

// RUN: %clang_cc1 -triple mips64-linux-gnu -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MIPS-64EB
// MIPS-64EB: target datalayout = "E-p:64:64:64-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v64:64:64-n32:64-S128"
