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

@interface Test2
-(int)implicitProp;
-(void)setImplicitProp:(int)x;
@end

void foo1(Test2 *test2) {
  int x = test2.implicitProp;
  test2.implicitProp = x;
  ++test2.implicitProp;
}

@interface Test3
-(void)setFoo:(int)x withBar:(int)y;
@end

void foo3(Test3 *test3) {
  [test3 setFoo:2 withBar:4];
}

@interface Test4
@end
@interface Test4(Dido)
@end
@implementation Test4(Dido)
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

// RUN: c-index-test -cursor-at=%s:38:6 -cursor-at=%s:40:11 \
// RUN:   -cursor-at=%s:50:20 -cursor-at=%s:51:15 -cursor-at=%s:52:20 %s | FileCheck -check-prefix=CHECK-MEMBERREF %s
// CHECK-MEMBERREF: 38:6 MemberRefExpr=x:34:16 SingleRefName=[38:6 - 38:7] RefName=[38:6 - 38:7] Extent=[38:3 - 38:7]
// CHECK-MEMBERREF: 40:9 MemberRefExpr=name:23:21 Extent=[40:3 - 40:13] Spelling=name ([40:9 - 40:13])
// CHECK-MEMBERREF: 50:17 MemberRefExpr=implicitProp:45:7 Extent=[50:11 - 50:29] Spelling=implicitProp ([50:17 - 50:29])
// CHECK-MEMBERREF: 51:9 MemberRefExpr=setImplicitProp::46:8 Extent=[51:3 - 51:21] Spelling=setImplicitProp: ([51:9 - 51:21])
// CHECK-MEMBERREF: 52:11 MemberRefExpr=setImplicitProp::46:8 Extent=[52:5 - 52:23] Spelling=setImplicitProp: ([52:11 - 52:23])

// RUN: c-index-test -cursor-at=%s:56:24 -cursor-at=%s:60:14 \
// RUN:   -cursor-at=%s:65:20 -cursor-at=%s:67:25 \
// RUN:   %s | FileCheck -check-prefix=CHECK-SPELLRANGE %s
// CHECK-SPELLRANGE: 56:8 ObjCInstanceMethodDecl=setFoo:withBar::56:8 Extent=[56:1 - 56:37] Spelling=setFoo:withBar: ([56:8 - 56:14][56:22 - 56:29]) Selector index=1
// CHECK-SPELLRANGE: 60:3 ObjCMessageExpr=setFoo:withBar::56:8 Extent=[60:3 - 60:29] Spelling=setFoo:withBar: ([60:10 - 60:16][60:19 - 60:26]) Selector index=0
// CHECK-SPELLRANGE: 65:12 ObjCCategoryDecl=Dido:65:12 Extent=[65:1 - 66:5] Spelling=Dido ([65:18 - 65:22])
// CHECK-SPELLRANGE: 67:17 ObjCCategoryImplDecl=Dido:67:17 (Definition) Extent=[67:1 - 68:2] Spelling=Dido ([67:23 - 67:27])
