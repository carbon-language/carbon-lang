// RUN: clang -emit-llvm -fblocks -S -o - %s
// rdar://6676764

struct S {
  void (^F)(struct S*);
} P;


@interface T

  - (int)foo: (T (^)(T*)) x;
@end

void foo(T *P) {
 [P foo: 0];
}

