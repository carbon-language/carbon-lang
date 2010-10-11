// RUN: c-index-test -code-completion-at=%s:6:2 -remap-file="%s;%S/Inputs/remap-complete-to.c" %s 2> %t.err | FileCheck %s
// RUN: FileCheck -check-prefix=CHECK-DIAGS %s < %t.err

// CHECK: FunctionDecl:{ResultType int}{TypedText f0}{LeftParen (}
void f() { }

// CHECK-DIAGS: remap-complete.c:2:19
