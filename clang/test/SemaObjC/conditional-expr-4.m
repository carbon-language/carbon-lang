// RUN: clang-cc -fsyntax-only %s
// XFAIL
// <rdar://problem/6212771>

#define nil ((void*) 0)

@interface A 
@property int x;
@end

@interface B : A
@end

// Basic checks...
id f0(int cond, id a, void *b) {
  return cond ? a : b;
}
A *f0_a(int cond, A *a, void *b) {
  return cond ? a : b;
}

id f1(int cond, id a) {
  return cond ? a : nil;
}
A *f1_a(int cond, A *a) {
  return cond ? a : nil;
}

// Check interaction with qualified id

@protocol P0 @end

id f2(int cond, id<P0> a, void *b) {
  return cond ? a : b;
}

id f3(int cond, id<P0> a) {
  return cond ? a : nil;
}

// Check that result actually has correct type.

// Using properties is one way to find the compiler internal type of a
// conditional expression. Simple assignment doesn't work because if
// the type is id then it can be implicitly promoted.
@protocol P1
@property int x;
@end

int f5(int cond, id<P1> a, id<P1> b) {
  // This should result in something with id type, currently. This is
  // almost certainly wrong and should be fixed.
  return (cond ? a : b).x; // expected-error {{member reference base type ('id') is not a structure or union}}
}
int f5_a(int cond, A *a, A *b) {
  return (cond ? a : b).x;
}
int f5_b(int cond, A *a, B *b) {
  return (cond ? a : b).x;
}

int f6(int cond, id<P1> a, void *b) {
  // This should result in something with id type, currently.
  return (cond ? a : b).x; // expected-error {{member reference base type ('id') is not a structure or union}}
}

int f7(int cond, id<P1> a) {
  return (cond ? a : nil).x;
}

int f8(int cond, id<P1> a, A *b) {
  // GCC regards this as a warning (comparison of distinct Objective-C types lacks a cast)
  return a == b; // expected-error {{invalid operands to binary expression}}
}

int f9(int cond, id<P1> a, A *b) {
  return (cond ? a : b).x; // expected-error {{incompatible operand types}}
}
