// RUN: %clang_cc1 -verify -Wno-objc-root-class %s
// FIXME: expected-no-diagnostics
@interface XX

- (void)addObserver:(XX*)o; // FIXME -- should note 2{{passing argument to parameter 'o' here}}

@end

@interface YY

+ (void)classMethod;

@end

@implementation YY

static XX *obj;

+ (void)classMethod {
  [obj addObserver:self];     // FIXME -- should error {{cannot initialize a parameter of type 'XX *' with an lvalue of type 'Class'}}
  Class whatever;
  [obj addObserver:whatever]; // FIXME -- should error {{cannot initialize a parameter of type 'XX *' with an lvalue of type 'Class'}}
}
@end

