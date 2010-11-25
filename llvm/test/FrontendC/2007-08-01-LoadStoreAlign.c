// RUN: %llvmgcc -O3 -S -o - %s | grep {align 1} | count 2
// RUN: %llvmgcc -O3 -S -o - %s | llc

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
