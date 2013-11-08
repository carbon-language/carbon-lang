// RUN: clang-query -c "match functionDecl()" %s -- | FileCheck %s
// REQUIRES: libedit

// CHECK: function-decl.c:5:1: note: "root" binds here
void foo(void) {}
