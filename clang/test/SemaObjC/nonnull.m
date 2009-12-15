// RUN: %clang_cc1 -fblocks -fsyntax-only -verify %s

@class NSObject;

int f1(int x) __attribute__((nonnull)); // expected-warning{{'nonnull' attribute applied to function with no pointer arguments}}
int f2(int *x) __attribute__ ((nonnull (1)));
int f3(int *x) __attribute__ ((nonnull (0))); // expected-error {{'nonnull' attribute parameter 1 is out of bounds}}
int f4(int *x, int *y) __attribute__ ((nonnull (1,2)));
int f5(int *x, int *y) __attribute__ ((nonnull (2,1)));
int f6(NSObject *x) __attribute__ ((nonnull (1))); // no-warning
int f7(NSObject *x) __attribute__ ((nonnull)); // no-warning


extern void func1 (void (^block1)(), void (^block2)(), int) __attribute__((nonnull));

extern void func3 (void (^block1)(), int, void (^block2)(), int)
__attribute__((nonnull(1,3)));

extern void func4 (void (^block1)(), void (^block2)()) __attribute__((nonnull(1)))
__attribute__((nonnull(2)));

void
foo (int i1, int i2, int i3, void (^cp1)(), void (^cp2)(), void (^cp3)())
{
  func1(cp1, cp2, i1);
  
  func1(0, cp2, i1);  // expected-warning {{null passed to a callee which requires a non-null argument}}
  func1(cp1, 0, i1);  // expected-warning {{null passed to a callee which requires a non-null argument}}
  func1(cp1, cp2, 0);
  
  
  func3(0, i2, cp3, i3); // expected-warning {{null passed to a callee which requires a non-null argument}}
  func3(cp3, i2, 0, i3);  // expected-warning {{null passed to a callee which requires a non-null argument}}
  
  func4(0, cp1); // expected-warning {{null passed to a callee which requires a non-null argument}}
  func4(cp1, 0); // expected-warning {{null passed to a callee which requires a non-null argument}}
  
  // Shouldn't these emit warnings?  Clang doesn't, and neither does GCC.  It
  // seems that the checking should handle Objective-C pointers.
  func6((NSObject*) 0); // no-warning
  func7((NSObject*) 0); // no-warning
}
