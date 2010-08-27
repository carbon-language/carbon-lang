// Note: the run lines follow their respective tests, since line/column
// matter in this test.

typedef int Bool;

@interface A
- (void)add:(int)x to:(int)y;
+ (void)select:(Bool)condition first:(int)x second:(int)y;
- (void)last;
+ (void)last;
@end

@interface B : A
- (void)add:(int)x to:(int)y;
+ (void)select:(Bool)condition first:(int)x second:(int)y;
@end

@implementation B
- (void)add:(int)a to:(int)b {
  [super add:a to:b];
}

+ (void)select:(Bool)condition first:(int)a second:(int)b {
  [super selector:condition first:a second:b];
}
@end

// Check "super" completion as a message receiver.
// RUN: c-index-test -code-completion-at=%s:20:4 %s | FileCheck -check-prefix=CHECK-ADD-RECEIVER %s
// CHECK-ADD-RECEIVER: ObjCInstanceMethodDecl:{ResultType void}{TypedText super}{HorizontalSpace  }{Text add:}{Placeholder a}{HorizontalSpace  }{Text to:}{Placeholder b} (8)

// RUN: c-index-test -code-completion-at=%s:24:4 %s | FileCheck -check-prefix=CHECK-SELECT-RECEIVER %s
// CHECK-SELECT-RECEIVER: ObjCClassMethodDecl:{ResultType void}{TypedText super}{HorizontalSpace  }{Text select:}{Placeholder condition}{HorizontalSpace  }{Text first:}{Placeholder a}{HorizontalSpace  }{Text second:}{Placeholder b} (8)

// Check "super" completion at the first identifier
// RUN: c-index-test -code-completion-at=%s:20:10 %s | FileCheck -check-prefix=CHECK-ADD-ADD %s
// CHECK-ADD-ADD: ObjCInstanceMethodDecl:{ResultType void}{TypedText add:}{Placeholder a}{HorizontalSpace  }{Text to:}{Placeholder b} (8)
// CHECK-ADD-ADD-NOT: add
// CHECK-ADD-ADD: ObjCInstanceMethodDecl:{ResultType void}{TypedText last} (20)

// RUN: c-index-test -code-completion-at=%s:24:10 %s | FileCheck -check-prefix=CHECK-SELECTOR-SELECTOR %s
// CHECK-SELECTOR-SELECTOR-NOT: x
// CHECK-SELECTOR-SELECTOR: ObjCClassMethodDecl:{ResultType void}{TypedText last} (20)
// CHECK-SELECTOR-SELECTOR: ObjCClassMethodDecl:{ResultType void}{TypedText select:}{Placeholder condition}{HorizontalSpace  }{Text first:}{Placeholder a}{HorizontalSpace  }{Text second:}{Placeholder b} (8)

// Check "super" completion at the second identifier
// RUN: c-index-test -code-completion-at=%s:20:16 %s | FileCheck -check-prefix=CHECK-ADD-TO %s
// CHECK-ADD-TO: ObjCInstanceMethodDecl:{ResultType void}{Informative add:}{TypedText to:}{Placeholder b} (8)

// RUN: c-index-test -code-completion-at=%s:24:28 %s | FileCheck -check-prefix=CHECK-SELECTOR-FIRST %s
// CHECK-SELECTOR-FIRST: ObjCClassMethodDecl:{ResultType void}{Informative select:}{TypedText first:}{Placeholder a}{HorizontalSpace  }{Text second:}{Placeholder b} (8)

// Check "super" completion at the third identifier
// RUN: c-index-test -code-completion-at=%s:24:37 %s | FileCheck -check-prefix=CHECK-SELECTOR-SECOND %s
// CHECK-SELECTOR-SECOND: ObjCClassMethodDecl:{ResultType void}{Informative select:}{Informative first:}{TypedText second:}{Placeholder b} (8)
