// RUN: clang -cc1 -pedantic -fixit %s -o %t
// RUN: clang -cc1 -pedantic -x objective-c %t -verify

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

@protocol X;

void foo() {
  <X> *P;    // should be fixed to 'id<X>'.
}

@class A;
@class NSString;

@interface Test
- (void)test:(NSString *)string;

@property (copy) NSString *property;
@end

void g(NSString *a);
void h(id a);

void f(Test *t) {
  NSString *a = "Foo";
  id b = "Foo";
  A* c = "Foo"; // expected-warning {{incompatible pointer types initializing 'char [4]', expected 'A *'}}
  g("Foo");
  h("Foo");
  h(("Foo"));
  [t test:"Foo"];
  t.property = "Foo";
}
