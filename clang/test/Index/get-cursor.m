// Test is line- and column-sensitive. Run lines are below.

@interface rdar9771715
@property (readonly) int foo1;
@property (readwrite) int foo2;
@end

@class Foo;

@interface rdar9535717 {
  __weak Foo *foo;
}
@end

@interface Test1 {
  id _name;
}
@end
@interface Test1 ()
- (id)name;
@end
@interface Test1 ()
@property (copy) id name;
@end
@implementation Test1
@synthesize name = _name;
@end

// RUN: c-index-test -cursor-at=%s:4:28 -cursor-at=%s:5:28 %s | FileCheck -check-prefix=CHECK-PROP %s
// CHECK-PROP: ObjCPropertyDecl=foo1:4:26
// CHECK-PROP: ObjCPropertyDecl=foo2:5:27

// RUN: c-index-test -cursor-at=%s:11:11 %s -ccc-host-triple x86_64-apple-macosx10.7.0 | FileCheck -check-prefix=CHECK-WITH-WEAK %s
// CHECK-WITH-WEAK: ObjCClassRef=Foo:8:8

// RUN: c-index-test -cursor-at=%s:20:10 %s | FileCheck -check-prefix=CHECK-METHOD %s
// CHECK-METHOD: ObjCInstanceMethodDecl=name:20:1
