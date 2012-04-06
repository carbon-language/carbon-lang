// RUN: %clang_cc1 -verify -Wunused -Wunused-parameter -fsyntax-only -Wno-objc-root-class %s

int printf(const char *, ...);

@interface Greeter
+ (void) hello;
@end

@implementation Greeter
+ (void) hello { printf("Hello, World!\n"); }
@end

int test1(void) {
  [Greeter hello];
  return 0;
}

@interface NSObject @end
@interface NSString : NSObject 
- (int)length;
@end

void test2() {
  @"pointless example call for test purposes".length; // expected-warning {{property access result unused - getters should not be used for side effects}}
}

@interface foo
- (int)meth: (int)x: (int)y: (int)z ;
@end

@implementation foo
- (int) meth: (int)x: 
(int)y: // expected-warning{{unused}} 
(int) __attribute__((unused))z { return x; }
@end

//===------------------------------------------------------------------------===
// The next test shows how clang accepted attribute((unused)) on ObjC
// instance variables, which GCC does not.
//===------------------------------------------------------------------------===

#if __has_feature(attribute_objc_ivar_unused)
#define UNUSED_IVAR __attribute__((unused))
#else
#error __attribute__((unused)) not supported on ivars
#endif

@interface TestUnusedIvar {
  id y __attribute__((unused)); // no-warning
  id x UNUSED_IVAR; // no-warning
}
@end

