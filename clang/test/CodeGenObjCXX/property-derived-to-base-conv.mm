// RUN: %clang_cc1 -fobjc-gc -triple x86_64-apple-darwin10 -emit-llvm -o - %s
// rdar: // 7501812

struct A { int member; };
struct B : A { };

@interface BInt {
@private
  B *b;
}
- (B)value;
- (void)setValue : (B) arg;
@property B value;
@end

void g(BInt *bint) {
  bint.value.member = 17;
}

