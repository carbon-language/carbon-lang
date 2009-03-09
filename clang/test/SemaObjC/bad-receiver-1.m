// RUN: clang -fsyntax-only -verify %s

@interface I
- (id) retain;
@end

void __raiseExc1() {
 [objc_lookUpClass("NSString") retain]; // expected-warning {{ "bad receiver type 'int'" }} \
    expected-warning {{method '-retain' not found}}
}

typedef const struct __CFString * CFStringRef;

void func() {
  CFStringRef obj;

  [obj self]; // expected-warning {{bad receiver type 'CFStringRef' (aka 'struct __CFString const *')}} \\
                 expected-warning {{method '-self' not found}}
}
