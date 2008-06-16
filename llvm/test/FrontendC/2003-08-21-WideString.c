// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null
// XFAIL: *
// See PR2452

struct {
  int *name;
} syms = { L"NUL" };
