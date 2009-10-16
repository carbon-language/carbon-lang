// RUN: clang-cc -emit-pch -x c -o %t.pch %S/c-index-pch.h &&
// RUN: clang-cc -include-pch %t.pch -x c -emit-pch -o %t.ast %s &&
// RUN: c-index-test %t.ast all | FileCheck -check-prefix=ALL %s &&
// RUN: c-index-test %t.ast local | FileCheck -check-prefix=LOCAL %s
// ALL: FunctionDecl=foo
// ALL: VarDecl=bar
// ALL: FunctionDecl=wibble
// ALL: FunctionDecl=wonka
// LOCAL-NOT: FunctionDecl=foo
// LOCAL-NOT: VarDecl=bar
// LOCAL: FunctionDecl=wibble
// LOCAL: FunctionDecl=wonka
void wibble(int i);
void wonka(float);
