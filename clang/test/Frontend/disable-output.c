// FIXME: this test produces a leak report in clang.
// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -emit-llvm-only -triple=i386-apple-darwin -o %t
// RUN: not rm %t
// FIXME: %clang_cc1 %s -emit-codegen-only -triple=i386-apple-darwin -o %t
// FIXME: not rm %t

// Test that output is not generated when emission is disabled.
