// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface I
- (id) retain;
@end

int objc_lookUpClass(const char*);

void __raiseExc1(void) {
 [objc_lookUpClass("NSString") retain]; // expected-warning {{receiver type 'int' is not 'id'}}
}

typedef const struct __CFString * CFStringRef;

void func(void) {
  CFStringRef obj;

  [obj self]; // expected-warning {{receiver type 'CFStringRef' (aka 'const struct __CFString *') is not 'id'}} \\
                 expected-warning {{method '-self' not found}}
}
