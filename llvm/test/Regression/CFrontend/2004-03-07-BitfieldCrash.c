// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/*
 * XFAIL: linux
 */
struct s {
  unsigned long long u33: 33;
  unsigned long long u40: 40;
};

struct s a = { 1, 2};

int foo() {
  return a.u40;
}
