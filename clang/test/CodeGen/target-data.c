// RUN: %clang_cc1 -triple i686-unknown-unknown -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=I686-UNKNOWN %s
// I686-UNKNOWN: target datalayout = "e-p:32:32:32-f64:32:64-f80:32:32-n8:16:32-S128"

// RUN: %clang_cc1 -triple i686-apple-darwin9 -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=I686-DARWIN %s
// I686-DARWIN: target datalayout = "e-p:32:32:32-f64:32:64-f80:128:128-n8:16:32-S128"

// RUN: %clang_cc1 -triple i686-unknown-win32 -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=I686-WIN32 %s
// I686-WIN32: target datalayout = "e-p:32:32:32-i64:64:64-f80:128:128-f80:32:32-n8:16:32-S32"

// RUN: %clang_cc1 -triple i686-unknown-cygwin -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=I686-CYGWIN %s
// I686-CYGWIN: target datalayout = "e-p:32:32:32-i64:64:64-f80:32:32-n8:16:32-S32"

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=X86_64 %s
// X86_64: target datalayout = "e-p:64:64:64-i64:64:64-s:64:64-f80:128:128-n8:16:32:64-S128"

// RUN: %clang_cc1 -triple xcore-unknown-unknown -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=XCORE %s
// XCORE: target datalayout = "e-p:32:32:32-a:0:32-n32-i1:8:32-i8:8:32-i16:16:32-i64:32:32-f16:16:32-f64:32:32"

// RUN: %clang_cc1 -triple sparc-sun-solaris -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefix=SPARC-V8
// SPARC-V8: target datalayout = "E-p:32:32:32-i64:64:64-n32-S64"

// RUN: %clang_cc1 -triple sparcv9-sun-solaris -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefix=SPARC-V9
// SPARC-V9: target datalayout = "E-p:64:64:64-i64:64:64-n32:64-S128"

// RUN: %clang_cc1 -triple mipsel-linux-gnu -o - -emit-llvm %s |     \
// RUN: FileCheck %s -check-prefix=MIPS-32EL
// MIPS-32EL: target datalayout = "e-p:32:32:32-i8:8:32-i16:16:32-i64:64:64-n32-S64"

// RUN: %clang_cc1 -triple mips-linux-gnu -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MIPS-32EB
// MIPS-32EB: target datalayout = "E-p:32:32:32-i8:8:32-i16:16:32-i64:64:64-n32-S64"

// RUN: %clang_cc1 -triple mips64el-linux-gnu -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MIPS-64EL
// MIPS-64EL: target datalayout = "e-p:64:64:64-i8:8:32-i16:16:32-i64:64:64-f128:128:128-n32:64-S128"

// RUN: %clang_cc1 -triple mips64el-linux-gnu -o - -emit-llvm -target-abi n32 \
// RUN: %s | FileCheck %s -check-prefix=MIPS-64EL-N32
// MIPS-64EL-N32: target datalayout = "e-p:32:32:32-i8:8:32-i16:16:32-i64:64:64-f128:128:128-n32:64-S128"

// RUN: %clang_cc1 -triple mips64-linux-gnu -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MIPS-64EB
// MIPS-64EB: target datalayout = "E-p:64:64:64-i8:8:32-i16:16:32-i64:64:64-f128:128:128-n32:64-S128"

// RUN: %clang_cc1 -triple mips64-linux-gnu -o - -emit-llvm %s -target-abi n32 \
// RUN: | FileCheck %s -check-prefix=MIPS-64EB-N32
// MIPS-64EB-N32: target datalayout = "E-p:32:32:32-i8:8:32-i16:16:32-i64:64:64-f128:128:128-n32:64-S128"

// RUN: %clang_cc1 -triple powerpc64-lv2 -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PS3
// PS3: target datalayout = "E-p:32:32:32-i64:64:64-n32"

// RUN: %clang_cc1 -triple i686-nacl -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=NACL
// RUN: %clang_cc1 -triple le32-nacl -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=NACL
// NACL: target datalayout = "e-i64:64:64-p:32:32:32-v128:32:32"

// RUN: %clang_cc1 -triple powerpc-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC
// PPC: target datalayout = "E-p:32:32:32-i64:64:64-n32"

// RUN: %clang_cc1 -triple powerpc64-freebsd -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC64-FREEBSD
// PPC64-FREEBSD: target datalayout = "E-p:64:64:64-i64:64:64-n32:64"

// RUN: %clang_cc1 -triple powerpc64-linux -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC64-LINUX
// PPC64-LINUX: target datalayout = "E-p:64:64:64-i64:64:64-f128:128:128-n32:64"

// RUN: %clang_cc1 -triple powerpc-darwin -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC32-DARWIN
// PPC32-DARWIN: target datalayout = "E-p:32:32:32-n32"

// RUN: %clang_cc1 -triple powerpc64-darwin -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC64-DARWIN
// PPC64-DARWIN: target datalayout = "E-p:64:64:64-i64:64:64-n32:64"

// RUN: %clang_cc1 -triple nvptx-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=NVPTX
// NVPTX: target datalayout = "e-p:32:32:32-i64:64:64-v16:16:16-v32:32:32-n16:32:64"

// RUN: %clang_cc1 -triple nvptx64-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=NVPTX64
// NVPTX64: target datalayout = "e-p:64:64:64-i64:64:64-v16:16:16-v32:32:32-n16:32:64"

// RUN: %clang_cc1 -triple r600-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=R600
// R600: target datalayout = "e-p:32:32:32-i64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v96:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-v2048:2048:2048-n32:64"

// RUN: %clang_cc1 -triple r600-unknown -target-cpu cayman -o - -emit-llvm %s \
// RUN: | FileCheck %s -check-prefix=R600D
// R600D: target datalayout = "e-p:32:32:32-i64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v96:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-v2048:2048:2048-n32:64"

// RUN: %clang_cc1 -triple r600-unknown -target-cpu hawaii -o - -emit-llvm %s \
// RUN: | FileCheck %s -check-prefix=R600SI
// R600SI: target datalayout = "e-p:64:64:64-p3:32:32:32-i64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v96:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-v2048:2048:2048-n32:64"

// RUN: %clang_cc1 -triple aarch64-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=AARCH64
// AARCH64: target datalayout = "e-p:64:64-i64:64:64-i128:128:128-f128:128:128-n32:64-S128"

// RUN: %clang_cc1 -triple thumb-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=THUMB
// THUMB: target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64:64-v128:64:128-a:0:32-n32-S64"

// RUN: %clang_cc1 -triple arm-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=ARM
// ARM: target datalayout = "e-p:32:32:32-i64:64:64-v128:64:128-n32-S64"

// RUN: %clang_cc1 -triple thumb-unknown -o - -emit-llvm -target-abi apcs-gnu \
// RUN: %s | FileCheck %s -check-prefix=THUMB-GNU
// THUMB-GNU: target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"

// RUN: %clang_cc1 -triple arm-unknown -o - -emit-llvm -target-abi apcs-gnu \
// RUN: %s | FileCheck %s -check-prefix=ARM-GNU
// ARM-GNU: target datalayout = "e-p:32:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"

// RUN: %clang_cc1 -triple hexagon-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=HEXAGON
// HEXAGON: target datalayout = "e-p:32:32:32-i64:64:64-i1:32:32-a:0-n32"

// RUN: %clang_cc1 -triple s390x-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=SYSTEMZ
// SYSTEMZ: target datalayout = "E-p:64:64:64-i1:8:16-i8:8:16-i16:16-i32:32-i64:64-f32:32-f64:64-f128:64-a:8:16-n32:64"

// RUN: %clang_cc1 -triple msp430-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MSP430
// MSP430: target datalayout = "e-p:16:16:16-i32:16:32-n8:16"

// RUN: %clang_cc1 -triple tce-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=TCE
// TCE: target datalayout = "E-p:32:32:32-i8:8:32-i16:16:32-i64:32:32-f64:32:32-v64:32:32-v128:32:32-a:0:32-n32"

// RUN: %clang_cc1 -triple spir-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=SPIR
// SPIR: target datalayout = "e-p:32:32:32-i64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v96:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"

// RUN: %clang_cc1 -triple spir64-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=SPIR64
// SPIR64: target datalayout = "e-p:64:64:64-i64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v96:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
