// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Data
- (unsigned)length;
- (void)getData:(void*)buffer;
@end

void test(Data *d) {
  char buffer[[d length]]; // expected-error{{variable length arrays are not permitted in C++}}
  [d getData:buffer];
}

