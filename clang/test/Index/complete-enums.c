// Note: the run lines follow their respective tests, since line/column
// matter in this test.

enum {
  Red = 17,
  Green,
  Blue
};

void f() {
  
}

// RUN: c-index-test -code-completion-at=%s:11:1 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: EnumConstantDecl:{ResultType enum <anonymous>}{TypedText Red}
