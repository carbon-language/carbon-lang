// RUN: clang-cc -rewrite-objc %s -o=-

@interface Derived @end

int main(void) {
  Derived *v ;
  [v free];
  return 0;
}
