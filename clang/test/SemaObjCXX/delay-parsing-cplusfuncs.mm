// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -Werror -verify -Wno-objc-root-class %s
// rdar://10387088

@interface MyClass
- (void)someMethod;
@end

struct S {
  int bar(MyClass * myObject);

  int gorfbar(MyClass * myObject);

};

@implementation MyClass
- (void)someMethod {
    [self privateMethod];  // clang already does not warn here
}

int S::bar(MyClass * myObject) {
    [myObject privateMethod]; 
    return gorfbar(myObject);
}
- (void)privateMethod { }

int S::gorfbar(MyClass * myObject) {
    [myObject privateMethod]; 
    [myObject privateMethod1]; 
    return getMe + bar(myObject);
}

- (void)privateMethod1 {
  getMe = getMe+1;
}

static int getMe;

@end
