// The run lines are below, because this test is line- and
// column-number sensitive.
struct Foo {
  void babble() const volatile;
  void bar();
  void baz() const;
  void bingo() volatile;
  void theend() const volatile;
};

template<typename T>
struct smart_ptr {
  T *operator->();
  const T* operator->() const;
};

void text(Foo f, Foo *fp, const Foo &fc, const Foo *fcp,
          smart_ptr<Foo> sf, const smart_ptr<Foo> &sfc, Foo volatile *fvp) {
  f.bar();
  fp->bar();
  fc.baz();
  fcp->baz();
  sf->bar();
  sfc->baz();
  fvp->babble();
}

void Foo::bar() {
  
}

void Foo::baz() const {

}

void Foo::bingo() volatile {

}

// Check member access expressions.
// RUN: c-index-test -code-completion-at=%s:19:5 %s | FileCheck -check-prefix=CHECK-NOQUALS %s
// RUN: c-index-test -code-completion-at=%s:20:7 %s | FileCheck -check-prefix=CHECK-NOQUALS %s
// RUN: c-index-test -code-completion-at=%s:23:7 %s | FileCheck -check-prefix=CHECK-NOQUALS %s
// CHECK-NOQUALS: FunctionDecl:{ResultType void}{TypedText babble}{LeftParen (}{RightParen )}{Informative  const volatile} (20)
// CHECK-NOQUALS: FunctionDecl:{ResultType void}{TypedText bar}{LeftParen (}{RightParen )} (19)
// CHECK-NOQUALS: FunctionDecl:{ResultType void}{TypedText baz}{LeftParen (}{RightParen )}{Informative  const} (20)
// CHECK-NOQUALS: FunctionDecl:{ResultType void}{TypedText bingo}{LeftParen (}{RightParen )}{Informative  volatile} (20)
// RUN: c-index-test -code-completion-at=%s:21:6 %s | FileCheck -check-prefix=CHECK-CONST %s
// RUN: c-index-test -code-completion-at=%s:22:8 %s | FileCheck -check-prefix=CHECK-CONST %s
// RUN: c-index-test -code-completion-at=%s:24:8 %s | FileCheck -check-prefix=CHECK-CONST %s
// CHECK-CONST: FunctionDecl:{ResultType void}{TypedText babble}{LeftParen (}{RightParen )}{Informative  const volatile} (20)
// CHECK-CONST-NOT: bar
// CHECK-CONST: FunctionDecl:{ResultType void}{TypedText baz}{LeftParen (}{RightParen )}{Informative  const} (19)
// CHECK-CONST-NOT: bingo
// CHECK-CONST: theend
// RUN: c-index-test -code-completion-at=%s:25:8 %s | FileCheck -check-prefix=CHECK-VOLATILE %s
// CHECK-VOLATILE: FunctionDecl:{ResultType void}{TypedText babble}{LeftParen (}{RightParen )}{Informative  const volatile} (20)
// CHECK-VOLATILE-NOT: baz
// CHECK-VOLATILE: FunctionDecl:{ResultType void}{TypedText bingo}{LeftParen (}{RightParen )}{Informative  volatile} (19)

// Check implicit member access expressions.
// RUN: c-index-test -code-completion-at=%s:29:2 %s | FileCheck -check-prefix=CHECK-IMPLICIT-NOQUALS %s
// CHECK-IMPLICIT-NOQUALS: FunctionDecl:{ResultType void}{TypedText babble}{LeftParen (}{RightParen )}{Informative  const volatile} (15)
// CHECK-IMPLICIT-NOQUALS: FunctionDecl:{ResultType void}{TypedText bar}{LeftParen (}{RightParen )} (14)
// CHECK-IMPLICIT-NOQUALS: FunctionDecl:{ResultType void}{TypedText baz}{LeftParen (}{RightParen )}{Informative  const} (15)
// CHECK-IMPLICIT-NOQUALS: FunctionDecl:{ResultType void}{TypedText bingo}{LeftParen (}{RightParen )}{Informative  volatile} (15)

// RUN: c-index-test -code-completion-at=%s:33:1 %s | FileCheck -check-prefix=CHECK-IMPLICIT-CONST %s
// CHECK-IMPLICIT-CONST: FunctionDecl:{ResultType void}{TypedText babble}{LeftParen (}{RightParen )}{Informative  const volatile} (15)
// CHECK-IMPLICIT-CONST-NOT: bar
// CHECK-IMPLICIT-CONST: FunctionDecl:{ResultType void}{TypedText baz}{LeftParen (}{RightParen )}{Informative  const} (14)
// CHECK-IMPLICIT-CONST-NOT: bingo
// CHECK-IMPLICIT-CONST: theend

// RUN: c-index-test -code-completion-at=%s:37:1 %s | FileCheck -check-prefix=CHECK-IMPLICIT-VOLATILE %s
// CHECK-IMPLICIT-VOLATILE: FunctionDecl:{ResultType void}{TypedText babble}{LeftParen (}{RightParen )}{Informative  const volatile} (15)
// CHECK-IMPLICIT-VOLATILE-NOT: baz
// CHECK-IMPLICIT-VOLATILE: FunctionDecl:{ResultType void}{TypedText bingo}{LeftParen (}{RightParen )}{Informative  volatile} (14)
