// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-ipa=dynamic-bifurcate -verify %s

#include "InlineObjCInstanceMethod.h"

@interface MyParent : NSObject
- (int)getZero;
@end
@implementation MyParent
- (int)getZero {
    return 0;
}
@end

@implementation PublicClass
- (int)getZeroPublic {
    return 0;
}
@end

@interface MyClassWithPublicParent : PublicClass
- (int)getZeroPublic;
@end
@implementation MyClassWithPublicParent
- (int)getZeroPublic {
    return 0;
}
@end

// Category overrides a public method.
@interface PublicSubClass (PrvateCat)
  - (int) getZeroPublic;
@end
@implementation PublicSubClass (PrvateCat)
- (int)getZeroPublic {
    return 0;
}
@end


@interface MyClass : MyParent
- (int)getZero;
@end

// Since class is private, we assume that it cannot be subclassed.
// False negative: this class is "privately subclassed". this is very rare 
// in practice.
@implementation MyClass
+ (int) testTypeFromParam:(MyParent*) p {
  int m = 0;
  int z = [p getZero];
  if (z)
    return 5/m; // false negative
  return 5/[p getZero];// expected-warning {{Division by zero}}
}

// Here only one definition is possible, since the declaration is not visible 
// from outside. 
+ (int) testTypeFromParamPrivateChild:(MyClass*) c {
  int m = 0;
  int z = [c getZero]; // MyClass overrides getZero to return '1'.
  if (z)
    return 5/m; // expected-warning {{Division by zero}}
  return 5/[c getZero];//no warning
}

- (int)getZero {
    return 1;
}
@end

// The class is prvate and is not subclassed.
int testCallToPublicAPIInParent(MyClassWithPublicParent *p) {
  int m = 0;
  int z = [p getZeroPublic];
  if (z)
    return 5/m; // no warning
  return 5/[p getZeroPublic];// expected-warning {{Division by zero}}  
}

// When the called method is public (due to it being defined outside of main file),
// split the path and analyze both branches.
// In this case, p can be either the object of type MyParent* or MyClass*:
// - If it's MyParent*, getZero returns 0.
// - If it's MyClass*, getZero returns 1 and 'return 5/m' is reachable.
// Declaration is provate, but p can be a subclass (MyClass*).
int testCallToPublicAPI(PublicClass *p) {
  int m = 0;
  int z = [p getZeroPublic];
  if (z)
    return 5/m; // expected-warning {{Division by zero}}
  return 5/[p getZeroPublic];// expected-warning {{Division by zero}}  
}

// Even though the method is privately declared in the category, the parent 
// declares the method as public. Assume the instance can be subclassed.
int testCallToPublicAPICat(PublicSubClass *p) {
  int m = 0;
  int z = [p getZeroPublic];
  if (z)
    return 5/m; // expected-warning {{Division by zero}}
  return 5/[p getZeroPublic];// expected-warning {{Division by zero}}  
}
