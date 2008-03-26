// RUN: clang -rewrite-test %s -o=-

@interface Derived @end

int main(void) {
  Derived *v ;
  [v free];
  return 0;
}
