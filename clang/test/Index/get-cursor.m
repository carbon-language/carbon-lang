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

@interface rdar10902015
@end

@implementation rdar10902015

struct S { int x; };

-(void)mm:(struct S*)s {
  rdar10902015 *i = 0;
  s->x = 0;
  Test1 *test1;
  test1.name = 0;
}
@end

// RUN: c-index-test -cursor-at=%s:4:28 -cursor-at=%s:5:28 %s | FileCheck -check-prefix=CHECK-PROP %s
// CHECK-PROP: ObjCPropertyDecl=foo1:4:26
// CHECK-PROP: ObjCPropertyDecl=foo2:5:27

// RUN: c-index-test -cursor-at=%s:11:11 %s -target x86_64-apple-macosx10.7.0 | FileCheck -check-prefix=CHECK-WITH-WEAK %s
// CHECK-WITH-WEAK: ObjCClassRef=Foo:8:8

// RUN: c-index-test -cursor-at=%s:20:10 %s | FileCheck -check-prefix=CHECK-METHOD %s
// CHECK-METHOD: 20:7 ObjCInstanceMethodDecl=name:20:7 Extent=[20:1 - 20:12]

// RUN: c-index-test -cursor-at=%s:37:17 %s | FileCheck -check-prefix=CHECK-IN-IMPL %s
// CHECK-IN-IMPL: VarDecl=i:37:17

// RUN: c-index-test -cursor-at=%s:38:6 -cursor-at=%s:40:11 %s | FileCheck -check-prefix=CHECK-MEMBERREF %s
// CHECK-MEMBERREF: 38:6 MemberRefExpr=x:34:16 SingleRefName=[38:6 - 38:7] RefName=[38:6 - 38:7] Extent=[38:3 - 38:7]
// CHECK-MEMBERREF: 40:9 MemberRefExpr=name:23:21 Extent=[40:3 - 40:13] Spelling=name
