// RUN: %clang_cc1 -x objective-c++ -fcxx-exceptions -fsyntax-only -Werror -verify -Wno-objc-root-class %s
// rdar://10387088

@interface MyClass
- (void)someMethod;
@end

struct BadReturn {
  BadReturn(MyClass * myObject);
  int bar(MyClass * myObject);
  int i;
};

@implementation MyClass
- (void)someMethod {
    [self privateMethod];  // clang already does not warn here
}

int BadReturn::bar(MyClass * myObject) {
    [myObject privateMethod];
    return 0;
}

BadReturn::BadReturn(MyClass * myObject) try {
} catch(...) {
  try {
    [myObject privateMethod];
    [myObject privateMethod1];
    getMe = bar(myObject);
  } catch(int ei) {
    i = ei;
  } catch(...) {
    {
      i = 0;
    }
  }
}

- (void)privateMethod { }

- (void)privateMethod1 {
  getMe = getMe+1;
}

static int getMe;

@end
