// REQUIRES: powerpc-registered-target
// RUN: not %clang -fsyntax-only -mcpu=power8 -std=c++11 %s 2>&1 | \
// RUN: FileCheck %s -check-prefix=CHECK-DEFAULT

// RUN: not %clang -fsyntax-only -mcpu=power8 -std=c++11 \
// RUN: -mno-vsx -mpower8-vector %s 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-NVSX-P8V

// RUN: not %clang -fsyntax-only -mcpu=power8 -std=c++11 \
// RUN: -mno-vsx -mdirect-move %s 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-NVSX-DMV

// RUN: not %clang -fsyntax-only -mcpu=power8 -std=c++11 \
// RUN: -mno-vsx -mpower8-vector -mvsx %s 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-DEFAULT

// RUN: not %clang -fsyntax-only -mcpu=power8 -std=c++11 \
// RUN: -mno-vsx -mdirect-move -mvsx %s 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-DEFAULT

// RUN: not %clang -fsyntax-only -mcpu=power8 -std=c++11 \
// RUN: -mpower8-vector -mno-vsx %s 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-NVSX-P8V

// RUN: not %clang -fsyntax-only -mcpu=power8 -std=c++11 \
// RUN: -mdirect-move -mno-vsx %s 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-NVSX-DMV

// RUN: not %clang -fsyntax-only -mcpu=power8 -std=c++11 \
// RUN: -mno-vsx %s 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-NVSX

// RUN: not %clang -fsyntax-only -mcpu=power6 -std=c++11 %s 2>&1 | \
// RUN: FileCheck %s -check-prefix=CHECK-NVSX

// RUN: not %clang -fsyntax-only -mcpu=power6 -std=c++11 \
// RUN: -mpower8-vector %s 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-DEFAULT

// RUN: not %clang -fsyntax-only -mcpu=power6 -std=c++11 \
// RUN: -mdirect-move %s 2>&1 | FileCheck %s \
// RUN: -check-prefix=CHECK-VSX

#ifdef __VSX__
static_assert(false, "VSX enabled");
#endif

#ifdef __POWER8_VECTOR__
static_assert(false, "P8V enabled");
#endif

#if !defined(__VSX__) && !defined(__POWER8_VECTOR__)
static_assert(false, "Neither enabled");
#endif

// CHECK-DEFAULT: VSX enabled
// CHECK-DEFAULT: P8V enabled
// CHECK-NVSX-P8V: error: option '-mpower8-vector' cannot be specified with '-mno-vsx'
// CHECK-NVSX-DMV: error: option '-mdirect-move' cannot be specified with '-mno-vsx'
// CHECK-NVSX: Neither enabled
// CHECK-VSX: VSX enabled
