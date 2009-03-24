// RUN: clang-cc -fblocks -fno-__block -fnext-runtime --emit-llvm -o %t %s -verify

@class Foo;
@protocol P;

void t1()
{
  __block int a;
  ^{ a = 10; }(); // expected-error {{cannot compile this __block variable in block literal yet}} expected-error {{cannot compile this block literal that requires copy/dispose yet}}

  void (^block)(void);
  ^{ (void)block; }(); // expected-error {{cannot compile this block pointer in block literal yet}} expected-error {{cannot compile this block literal that requires copy/dispose yet}}

  struct Foo *__attribute__ ((NSObject)) foo;
  ^{ (void)foo; }(); // expected-error {{cannot compile this __attribute__((NSObject)) variable in block literal yet}} expected-error {{cannot compile this block literal that requires copy/dispose yet}}

  typedef struct CGColor * __attribute__ ((NSObject)) CGColorRef;
  CGColorRef color;
  ^{ (void)color; }(); // expected-error {{cannot compile this __attribute__((NSObject)) variable in block literal yet}} expected-error {{cannot compile this block literal that requires copy/dispose yet}}

  id a1;
  ^{ (void)a1; }(); // expected-error {{cannot compile this Objective-C variable in block literal yet}} expected-error {{cannot compile this block literal that requires copy/dispose yet}}

  Foo *a2;
  ^{ (void)a2; }(); // expected-error {{cannot compile this Objective-C variable in block literal yet}} expected-error {{cannot compile this block literal that requires copy/dispose yet}}

  id<P> a3;
  ^{ (void)a3; }(); // expected-error {{cannot compile this Objective-C variable in block literal yet}} expected-error {{cannot compile this block literal that requires copy/dispose yet}}

  Foo<P> *a4;
  ^{ (void)a4; }(); // expected-error {{cannot compile this Objective-C variable in block literal yet}} expected-error {{cannot compile this block literal that requires copy/dispose yet}}
}
