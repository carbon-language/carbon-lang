// RUN: mkdir -p %t
// RUN: %clang_cc1 -emit-module -o %t/diamond_top.pcm %s -D MODULE_TOP
// RUN: %clang_cc1 -fmodule-cache-path %t -fdisable-module-hash -emit-module -o %t/diamond_left.pcm %s -D MODULE_LEFT
// RUN: %clang_cc1 -fmodule-cache-path %t -fdisable-module-hash -emit-module -o %t/diamond_right.pcm %s -D MODULE_RIGHT
// RUN: %clang_cc1 -fmodule-cache-path %t -fdisable-module-hash -emit-module -o %t/diamond_bottom.pcm %s -D MODULE_BOTTOM
// RUN: %clang_cc1 -fmodule-cache-path %t -fdisable-module-hash %s -verify

/*============================================================================*/
#ifdef MODULE_TOP

@interface Foo
@end

@interface Foo(Top)
-(void)top;
@end

/*============================================================================*/
#elif defined(MODULE_LEFT)

__import_module__ diamond_top;

@interface Foo(Left)
-(void)left;
@end

@interface LeftFoo
-(void)left;
@end

@interface Foo(Duplicate) // expected-note {{previous definition}}
@end

@interface Foo(Duplicate)
@end

/*============================================================================*/
#elif defined(MODULE_RIGHT)

__import_module__ diamond_top;

@interface Foo(Right1)
-(void)right1;
@end

@interface Foo(Right2)
-(void)right2;
@end

@interface Foo(Duplicate) // expected-warning {{duplicate definition of category}}
@end

/*============================================================================*/
#elif defined(MODULE_BOTTOM)

__import_module__ diamond_left;

@interface Foo(Bottom)
-(void)bottom;
@end

__import_module__ diamond_right;

@interface LeftFoo(Bottom)
-(void)bottom;
@end

/*============================================================================*/
#else

__import_module__ diamond_bottom;

@interface Foo(Source)
-(void)source;
@end

void test(Foo *foo, LeftFoo *leftFoo) {
  [foo source];
  [foo bottom];
  [foo left];
  [foo right1];
  [foo right2];
  [foo top];

  [leftFoo left];
  [leftFoo bottom];
}

#endif
