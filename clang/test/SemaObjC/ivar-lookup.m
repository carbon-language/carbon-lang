// RUN: %clang_cc1 %s -fsyntax-only -verify

@interface Test {
   int x;
}

-(void) setX: (int) d;
@end

extern struct foo x;

@implementation Test

-(void) setX: (int) n {
   x = n;
}

@end

@interface Ivar
- (float*)method;
@end

@interface A {
  A *Ivar;
}
- (int*)method;
@end

@implementation A
- (int*)method {
  int *ip = [Ivar method]; // expected-warning{{warning: incompatible pointer types initializing 'int *' with an expression of type 'float *'}}
                           // Note that there is no warning in Objective-C++
  return 0;
}
@end

@interface TwoIvars {
  int a;
  int b;
}
@end

@implementation TwoIvars
+ (int)classMethod {
  return a + b; // expected-error{{instance variable 'a' accessed in class method}} \
  // expected-error{{instance variable 'b' accessed in class method}}
}
@end
