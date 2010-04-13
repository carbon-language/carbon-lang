// RUN: %clang_cc1 -g -S -emit-llvm -o %t %s
// RUN: grep "i32 20, i32 3, metadata" %t | count 1
// Check there is a line number entry for line 20 where b1 is destructed.
class A { int a; };
class B {
public:
  B() { a = new A; }
  ~B() { delete a; }
private:
  A *a;
};

void fn(B b);

int i;
void foo() {
  if (i) {
    B b1;
    fn (b1);
  }
}
