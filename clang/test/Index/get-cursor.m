// Test is line- and column-sensitive. Run lines are below.

@interface rdar9771715
@property (readonly) int foo1;
@property (readwrite) int foo2;
@end

// RUN: c-index-test -cursor-at=%s:4:28 %s | FileCheck -check-prefix=CHECK-PROP1 %s
// RUN: c-index-test -cursor-at=%s:5:28 %s | FileCheck -check-prefix=CHECK-PROP2 %s
// CHECK-PROP1: ObjCPropertyDecl=foo1:4:26
// CHECK-PROP2: ObjCPropertyDecl=foo2:5:27
