// Note: the run lines follow their respective tests, since line/column
// matter in this test.


void f(int x) {
  if (x) {
  } 
}

// RUN: c-index-test -code-completion-at=%s:7:4 %s | FileCheck -check-prefix=CHECK-IF-ELSE %s
// CHECK-IF-ELSE: NotImplemented:{TypedText else}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Placeholder statements}{VerticalSpace  }{RightBrace }} (40)
// CHECK-IF-ELSE: NotImplemented:{TypedText else}{HorizontalSpace  }{Text if}{HorizontalSpace  }{LeftParen (}{Placeholder expression}{RightParen )}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Placeholder statements}{VerticalSpace  }{RightBrace }} (40)
