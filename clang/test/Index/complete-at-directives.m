/* Run lines are at the end, since line/column matter in this test. */
@interface MyClass { }
@end

@implementation MyClass
@end

// RUN: c-index-test -code-completion-at=%s:2:2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText class}{HorizontalSpace  }{Placeholder identifier}{Text ;}
// CHECK-CC1: {TypedText compatibility_alias}{HorizontalSpace  }{Placeholder alias}{HorizontalSpace  }{Placeholder class}
// CHECK-CC1: {TypedText implementation}{HorizontalSpace  }{Placeholder class}
// CHECK-CC1: {TypedText interface}{HorizontalSpace  }{Placeholder class}
// CHECK-CC1: {TypedText protocol}{HorizontalSpace  }{Placeholder protocol}

// RUN: c-index-test -code-completion-at=%s:3:2 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText end}
// CHECK-CC2: {TypedText optional}
// CHECK-CC2: {TypedText property}
// CHECK-CC2: {TypedText required}

// RUN: c-index-test -code-completion-at=%s:6:2 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: {TypedText dynamic}{HorizontalSpace  }{Placeholder property}
// CHECK-CC3: {TypedText end}
// CHECK-CC3: {TypedText synthesize}{HorizontalSpace  }{Placeholder property}
