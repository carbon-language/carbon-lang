// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null
// XFAIL: *
// See PR2425

struct {
  int *name;
} syms = { L"NUL" };
