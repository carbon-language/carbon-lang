// RUN: %clang_cc1 -rewrite-objc %s -o -

@interface Derived @end

int main(void) {
  Derived *v ;
  [v free];
  return 0;
}
