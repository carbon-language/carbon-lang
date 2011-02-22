// RUN: %clang_cc1 %s -fsyntax-only -verify -fobjc-exceptions

// rdar://6124613
void test1() {
  void *p = @1; // expected-error {{unexpected '@' in program}}
}

// <rdar://problem/7495713>
// This previously triggered a crash because the class has not been defined.
@implementation RDar7495713 (rdar_7495713_cat)  // expected-error{{cannot find interface declaration for 'RDar7495713'}}
- (id) rdar_7495713 {
  __PRETTY_FUNCTION__; // expected-warning{{expression result unused}}
}
@end

// <rdar://problem/7881045>
// This previously triggered a crash because a ';' was expected after the @throw statement.
void foo() {
  @throw (id)0 // expected-error{{expected ';' after @throw}}
}

