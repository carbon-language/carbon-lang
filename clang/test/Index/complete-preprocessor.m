// The line and column layout of this test is significant. Run lines
// are at the end.

#if 1
#endif

#define FOO(a, b) a##b
#define BAR
#ifdef FOO
#endif
#if defined(FOO)
#endif

// RUN: c-index-test -code-completion-at=%s:4:2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{TypedText define}{HorizontalSpace  }{Placeholder macro} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText define}{HorizontalSpace  }{Placeholder macro}{LeftParen (}{Placeholder args}{RightParen )} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText error}{HorizontalSpace  }{Placeholder message} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText if}{HorizontalSpace  }{Placeholder condition} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText ifdef}{HorizontalSpace  }{Placeholder macro} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText ifndef}{HorizontalSpace  }{Placeholder macro} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText import}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText import}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText include}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText include}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText include_next}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText include_next}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText line}{HorizontalSpace  }{Placeholder number} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText line}{HorizontalSpace  }{Placeholder number}{HorizontalSpace  }{Text "}{Placeholder filename}{Text "} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText pragma}{HorizontalSpace  }{Placeholder arguments} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText undef}{HorizontalSpace  }{Placeholder macro} (30)
// CHECK-CC1-NEXT: NotImplemented:{TypedText warning}{HorizontalSpace  }{Placeholder message} (30)
// RUN: c-index-test -code-completion-at=%s:5:2 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: NotImplemented:{TypedText define}{HorizontalSpace  }{Placeholder macro} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText define}{HorizontalSpace  }{Placeholder macro}{LeftParen (}{Placeholder args}{RightParen )} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText elif}{HorizontalSpace  }{Placeholder condition} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText else} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText endif} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText error}{HorizontalSpace  }{Placeholder message} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText if}{HorizontalSpace  }{Placeholder condition} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText ifdef}{HorizontalSpace  }{Placeholder macro} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText ifndef}{HorizontalSpace  }{Placeholder macro} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText import}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText import}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText include}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText include}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText include_next}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText include_next}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText line}{HorizontalSpace  }{Placeholder number} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText line}{HorizontalSpace  }{Placeholder number}{HorizontalSpace  }{Text "}{Placeholder filename}{Text "} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText pragma}{HorizontalSpace  }{Placeholder arguments} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText undef}{HorizontalSpace  }{Placeholder macro} (30)
// CHECK-CC2-NEXT: NotImplemented:{TypedText warning}{HorizontalSpace  }{Placeholder message} (30)
// RUN: c-index-test -code-completion-at=%s:9:8 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: NotImplemented:{TypedText BAR} (30)
// CHECK-CC3: NotImplemented:{TypedText FOO} (30)
// RUN: c-index-test -code-completion-at=%s:11:12 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: c-index-test -code-completion-at=%s:11:13 %s | FileCheck -check-prefix=CHECK-CC3 %s
