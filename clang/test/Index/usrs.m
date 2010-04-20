// RUN: c-index-test -test-load-source-usrs all %s | FileCheck %s

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
  return 0;
}
@synthesize d1;
@end

int z;

// CHECK: usrs.m c:@Ea@usrs.m@3:1 Extent=[3:1 - 6:2]
// CHECK: usrs.m c:@Ea@usrs.m@3:1@ABA Extent=[4:3 - 4:6]
// CHECK: usrs.m c:@Ea@usrs.m@3:1@CADABA Extent=[5:3 - 5:9]
// CHECK: usrs.m c:@Ea@usrs.m@8:1 Extent=[8:1 - 11:2]
// CHECK: usrs.m c:@Ea@usrs.m@8:1@FOO Extent=[9:3 - 9:6]
// CHECK: usrs.m c:@Ea@usrs.m@8:1@BAR Extent=[10:3 - 10:6]
// CHECK: usrs.m c:@SA@MyStruct Extent=[13:9 - 16:2]
// CHECK: usrs.m c:@SA@MyStruct@FI@wa Extent=[14:7 - 14:9]
// CHECK: usrs.m c:@SA@MyStruct@FI@moo Extent=[15:7 - 15:10]
// CHECK: usrs.m c:@T@usrs.m@16:3@MyStruct Extent=[16:3 - 16:11]
// CHECK: usrs.m c:@E@Pizza Extent=[18:1 - 21:2]
// CHECK: usrs.m c:@E@Pizza@CHEESE Extent=[19:3 - 19:9]
// CHECK: usrs.m c:@E@Pizza@MUSHROOMS Extent=[20:3 - 20:12]
// CHECK: usrs.m c:objc(cs)Foo Extent=[23:1 - 30:5]
// CHECK: usrs.m c:objc(cs)Foo@x Extent=[24:6 - 24:7]
// CHECK: usrs.m c:objc(cs)Foo@y Extent=[25:6 - 25:7]
// CHECK: usrs.m c:objc(cs)Foo(py)d1 Extent=[29:15 - 29:17]
// CHECK: usrs.m c:objc(cs)Foo(im)godzilla Extent=[27:1 - 27:17]
// CHECK: usrs.m c:objc(cs)Foo(cm)kingkong Extent=[28:1 - 28:17]
// CHECK: usrs.m c:objc(cs)Foo(im)d1 Extent=[29:15 - 29:17]
// CHECK: usrs.m c:objc(cs)Foo(im)setD1: Extent=[29:15 - 29:17]
// CHECK: usrs.m c:objc(cs)Foo Extent=[32:1 - 42:2]
// CHECK: usrs.m c:objc(cs)Foo(im)godzilla Extent=[33:1 - 37:2]
// CHECK: usrs.m c:@z Extent=[35:10 - 35:15]
// CHECK: usrs.m c:objc(cs)Foo(cm)kingkong Extent=[38:1 - 40:2]
// CHECK: usrs.m c:objc(cs)Foo@d1 Extent=[41:13 - 41:15]
// CHECK: usrs.m c:objc(cs)Foo(py)d1 Extent=[41:1 - 41:15]
// CHECK: usrs.m c:@z Extent=[44:1 - 44:6]

