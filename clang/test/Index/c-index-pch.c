// RUN: clang-cc -emit-pch -x c -o %t.pch %S/c-index-pch.h &&
// RUN: clang-cc -include-pch %t.pch -x c -emit-pch -o %t.ast %s &&
// RUN: c-index-test %t.ast all | FileCheck -check-prefix=ALL %s
// CHECK-ALL: FunctionDecl=foo
// CHECK-ALL: VarDecl=bar
// CHECK-ALL: FunctionDecl=wibble
void wibble(int i);
