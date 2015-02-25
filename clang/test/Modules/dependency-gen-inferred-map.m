// Test that the virtual file "__inferred_module.map" doesn't show up as dependency.

// RUN: rm -rf %t-mcp
// RUN: %clang_cc1 -isysroot %S/Inputs/System -triple x86_64-apple-darwin10 -dependency-file %t.d -MT %s.o -F %S/Inputs -fsyntax-only -fmodules -fmodules-cache-path=%t-mcp %s
// RUN: FileCheck %s < %t.d
// CHECK-NOT: __inferred_module

@import Module;
