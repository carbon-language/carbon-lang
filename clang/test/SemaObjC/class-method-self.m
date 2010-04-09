// RUN: %clang_cc1 -verify %s 

typedef struct objc_class *Class;
@interface XX

- (void)addObserver:(XX*)o;

@end

@interface YY

+ (void)classMethod;

@end

@implementation YY

static XX *obj;

+ (void)classMethod {
  [obj addObserver:self];     // expected-warning {{incompatible pointer types sending 'Class' to parameter of type 'XX *'}}
  Class whatever;
  [obj addObserver:whatever]; // expected-warning {{incompatible pointer types sending 'Class' to parameter of type 'XX *'}}
}
@end

