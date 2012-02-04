// RUN: %clang_cc1 -rewrite-objc -fobjc-fragile-abi  %s -o -

@interface Derived @end

int main(void) {
  Derived *v ;
  [v free];
  return 0;
}
