// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@interface Foo  {
  void *isa;
}
@property(copy) Foo *myprop;
@property(retain, nonatomic) id xx;
// RUN: c-index-test -code-completion-at=%s:7:11 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText assign}
// CHECK-CC1-NEXT: {TypedText copy}
// CHECK-CC1-NEXT: {TypedText getter}{Text  = }{Placeholder method}
// CHECK-CC1-NEXT: {TypedText nonatomic}
// CHECK-CC1-NEXT: {TypedText readonly}
// CHECK-CC1-NEXT: {TypedText readwrite}
// CHECK-CC1-NEXT: {TypedText retain}
// CHECK-CC1-NEXT: {TypedText setter}{Text  = }{Placeholder method}
// RUN: c-index-test -code-completion-at=%s:8:18 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText getter}{Text  = }{Placeholder method}
// CHECK-CC2-NEXT: {TypedText nonatomic}
// CHECK-CC2-NEXT: {TypedText readwrite}
// CHECK-CC2-NEXT: {TypedText setter}{Text  = }{Placeholder method}
@end
