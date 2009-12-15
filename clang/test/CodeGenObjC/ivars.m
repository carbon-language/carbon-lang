// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-llvm -o - %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -emit-llvm -o - %s

// rdar://6800926
@interface ITF {
@public
  unsigned field :1 ;
  _Bool boolfield :1 ;
}
@end

void foo(ITF *P) {
  P->boolfield = 1;
}
