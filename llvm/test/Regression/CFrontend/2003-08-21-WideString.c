// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

struct {
  int *name;
} syms = { L"NUL" };
