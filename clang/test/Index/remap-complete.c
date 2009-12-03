// RUN: c-index-test -code-completion-at=%s:1:12 -remap-file="%s;%S/Inputs/remap-complete-to.c" %s | FileCheck %s
// XFAIL: win32

// CHECK: FunctionDecl:{TypedText f0}{LeftParen (}{RightParen )}
void f() { }
