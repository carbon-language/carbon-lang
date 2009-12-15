// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: @"_ZZ11+[A shared]E1a" = internal global
// CHECK: @"_ZZ11-[A(Foo) f]E1a" = internal global

@interface A
@end

@implementation A

+ (A *)shared {
  static A* a;
  
  return a;
}

@end

@interface A(Foo)
@end

@implementation A(Foo)
- (int)f {
  // FIXME: Add a member function to s and make sure that it's mangled correctly.
  struct s {
  };
  
  static s a;

  return 0;
}
@end
