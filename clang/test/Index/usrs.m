// RUN: c-index-test -test-load-source-usrs all %s | FileCheck %s

static inline int my_helper(int x, int y) { return x + y; }

enum {
  ABA,
  CADABA
};

enum {
  FOO,
  BAR
};

typedef struct {
  int wa;
  int moo;
} MyStruct;

enum Pizza {
  CHEESE,
  MUSHROOMS
};

@interface Foo {
  id x;
  id y;
}
- (id) godzilla;
+ (id) kingkong;
@property int d1;
@end

@implementation Foo
- (id) godzilla {
  static int a = 0;
  extern int z;
  return 0;
}
+ (id) kingkong {
  int local_var;
  return 0;
}
@synthesize d1;
@end

int z;

static int local_func(int x) { return x; }

@interface CWithExt
- (id) meth1;
@end
@interface CWithExt ()
- (id) meth2;
@end
@interface CWithExt ()
- (id) meth3;
@end
@interface CWithExt (Bar)
- (id) meth4;
@end
@implementation CWithExt
- (id) meth1 { return 0; }
- (id) meth2 { return 0; }
- (id) meth3 { return 0; }
@end
@implementation CWithExt (Bar)
- (id) meth4 { return 0; }
@end

// CHECK: usrs.m c:usrs.m@85@F@my_helper Extent=[3:19 - 3:60]
// CHECK: usrs.m c:usrs.m@95@F@my_helper@x Extent=[3:29 - 3:34]
// CHECK: usrs.m c:usrs.m@102@F@my_helper@y Extent=[3:36 - 3:41]
// CHECK: usrs.m c:usrs.m@128@Ea Extent=[5:1 - 8:2]
// CHECK: usrs.m c:usrs.m@128@Ea@ABA Extent=[6:3 - 6:6]
// CHECK: usrs.m c:usrs.m@128@Ea@CADABA Extent=[7:3 - 7:9]
// CHECK: usrs.m c:usrs.m@155@Ea Extent=[10:1 - 13:2]
// CHECK: usrs.m c:usrs.m@155@Ea@FOO Extent=[11:3 - 11:6]
// CHECK: usrs.m c:usrs.m@155@Ea@BAR Extent=[12:3 - 12:6]
// CHECK: usrs.m c:@SA@MyStruct Extent=[15:9 - 18:2]
// CHECK: usrs.m c:@SA@MyStruct@FI@wa Extent=[16:7 - 16:9]
// CHECK: usrs.m c:@SA@MyStruct@FI@moo Extent=[17:7 - 17:10]
// CHECK: usrs.m c:usrs.m@219@T@MyStruct Extent=[18:3 - 18:11]
// CHECK: usrs.m c:@E@Pizza Extent=[20:1 - 23:2]
// CHECK: usrs.m c:@E@Pizza@CHEESE Extent=[21:3 - 21:9]
// CHECK: usrs.m c:@E@Pizza@MUSHROOMS Extent=[22:3 - 22:12]
// CHECK: usrs.m c:objc(cs)Foo Extent=[25:1 - 32:5]
// CHECK: usrs.m c:objc(cs)Foo@x Extent=[26:6 - 26:7]
// CHECK: usrs.m c:objc(cs)Foo@y Extent=[27:6 - 27:7]
// CHECK: usrs.m c:objc(cs)Foo(py)d1 Extent=[31:15 - 31:17]
// CHECK: usrs.m c:objc(cs)Foo(im)godzilla Extent=[29:1 - 29:17]
// CHECK: usrs.m c:objc(cs)Foo(cm)kingkong Extent=[30:1 - 30:17]
// CHECK: usrs.m c:objc(cs)Foo(im)d1 Extent=[31:15 - 31:17]
// CHECK: usrs.m c:objc(cs)Foo(im)setD1: Extent=[31:15 - 31:17]
// CHECK: usrs.m c:usrs.m@352objc(cs)Foo(im)setD1:@d1 Extent=[31:15 - 31:17]
// CHECK: usrs.m c:objc(cs)Foo Extent=[34:1 - 45:2]
// CHECK: usrs.m c:objc(cs)Foo(im)godzilla Extent=[35:1 - 39:2]
// CHECK: usrs.m c:usrs.m@409objc(cs)Foo(im)godzilla@a Extent=[36:10 - 36:19]
// CHECK: usrs.m c:objc(cs)Foo(im)godzilla@z Extent=[37:10 - 37:15]
// CHECK: usrs.m c:objc(cs)Foo(cm)kingkong Extent=[40:1 - 43:2]
// CHECK: usrs.m c:usrs.m@470objc(cs)Foo(cm)kingkong@local_var Extent=[41:3 - 41:16]
// CHECK: usrs.m c:objc(cs)Foo@d1 Extent=[44:13 - 44:15]
// CHECK: usrs.m c:objc(cs)Foo(py)d1 Extent=[44:1 - 44:15]
// CHECK: usrs.m c:@z Extent=[47:1 - 47:6]
// CHECK: usrs.m c:usrs.m@540@F@local_func Extent=[49:12 - 49:43]
// CHECK: usrs.m c:usrs.m@551@F@local_func@x Extent=[49:23 - 49:28]
// CHECK: usrs.m c:objc(cs)CWithExt Extent=[51:1 - 53:5]
// CHECK: usrs.m c:objc(cs)CWithExt(im)meth1 Extent=[52:1 - 52:14]
// CHECK: usrs.m c:objc(ext)CWithExt@usrs.m@612 Extent=[54:1 - 56:5]
// CHECK: usrs.m c:objc(cs)CWithExt(im)meth2 Extent=[55:1 - 55:14]
// CHECK: usrs.m c:objc(ext)CWithExt@usrs.m@654 Extent=[57:1 - 59:5]
// CHECK: usrs.m c:objc(cs)CWithExt(im)meth3 Extent=[58:1 - 58:14]
// CHECK: usrs.m c:objc(cy)CWithExt@Bar Extent=[60:1 - 62:5]
// CHECK: usrs.m c:objc(cy)CWithExt@Bar(im)meth4 Extent=[61:1 - 61:14]
// CHECK: usrs.m c:objc(cs)CWithExt Extent=[63:1 - 67:2]
// CHECK: usrs.m c:objc(cs)CWithExt(im)meth1 Extent=[64:1 - 64:27]
// CHECK: usrs.m c:objc(cs)CWithExt(im)meth2 Extent=[65:1 - 65:27]
// CHECK: usrs.m c:objc(cs)CWithExt(im)meth3 Extent=[66:1 - 66:27]
// CHECK: usrs.m c:objc(cy)CWithExt@Bar Extent=[68:1 - 70:2]
// CHECK: usrs.m c:objc(cy)CWithExt@Bar(im)meth4 Extent=[69:1 - 69:27]

