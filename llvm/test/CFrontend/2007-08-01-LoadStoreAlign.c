// RUN: %llvmgcc -O3 -S -o - -emit-llvm %s | grep -c {align 1} | grep 2
// RUN: %llvmgcc -O3 -S -o - -emit-llvm %s | llvm-as | llc

struct p {
  char a;
  int b;
} __attribute__ ((packed));

struct p t = { 1, 10 };
struct p u;

int main () {
  int tmp = t.b;
  u.b = tmp;
  return tmp;

}
