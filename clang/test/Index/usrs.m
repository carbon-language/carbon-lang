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

// CHECK: usrs.m c:usrs.m@3:19@F@my_helper Extent=[3:19 - 3:60]
// CHECK: usrs.m c:usrs.m@3:29@x Extent=[3:29 - 3:34]
// CHECK: usrs.m c:usrs.m@3:36@y Extent=[3:36 - 3:41]
// CHECK: usrs.m c:@Ea@usrs.m@5:1 Extent=[5:1 - 8:2]
// CHECK: usrs.m c:@Ea@usrs.m@5:1@ABA Extent=[6:3 - 6:6]
// CHECK: usrs.m c:@Ea@usrs.m@5:1@CADABA Extent=[7:3 - 7:9]
// CHECK: usrs.m c:@Ea@usrs.m@10:1 Extent=[10:1 - 13:2]
// CHECK: usrs.m c:@Ea@usrs.m@10:1@FOO Extent=[11:3 - 11:6]
// CHECK: usrs.m c:@Ea@usrs.m@10:1@BAR Extent=[12:3 - 12:6]
// CHECK: usrs.m c:@SA@MyStruct Extent=[15:9 - 18:2]
// CHECK: usrs.m c:@SA@MyStruct@FI@wa Extent=[16:7 - 16:9]
// CHECK: usrs.m c:@SA@MyStruct@FI@moo Extent=[17:7 - 17:10]
// CHECK: usrs.m c:@T@usrs.m@18:3@MyStruct Extent=[18:3 - 18:11]
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
// CHECK: usrs.m c:usrs.m@31:15@d1 Extent=[31:15 - 31:17]
// CHECK: usrs.m c:objc(cs)Foo Extent=[34:1 - 45:2]
// CHECK: usrs.m c:objc(cs)Foo(im)godzilla Extent=[35:1 - 39:2]
// CHECK: usrs.m c:usrs.m@36:10@a Extent=[36:10 - 36:19]
// CHECK: usrs.m c:@z Extent=[37:10 - 37:15]
// CHECK: usrs.m c:objc(cs)Foo(cm)kingkong Extent=[40:1 - 43:2]
// CHECK: usrs.m c:usrs.m@41:3@local_var Extent=[41:3 - 41:16]
// CHECK: usrs.m c:objc(cs)Foo@d1 Extent=[44:13 - 44:15]
// CHECK: usrs.m c:objc(cs)Foo(py)d1 Extent=[44:1 - 44:15]
// CHECK: usrs.m c:@z Extent=[47:1 - 47:6]
// CHECK: usrs.m c:usrs.m@49:12@F@local_func Extent=[49:12 - 49:43]
// CHECK: usrs.m c:usrs.m@49:23@x Extent=[49:23 - 49:28]

