// RUN: not %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-config unix.mallo:Optimistic=true 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-config uni:Optimistic=true 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-config uni.:Optimistic=true 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-config ..:Optimistic=true 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-config unix.:Optimistic=true 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-config unrelated:Optimistic=true 2>&1 | FileCheck %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-config unix.Malloc:Optimistic=true

// Just to test clang is working.
# foo

// CHECK: error:
