// RUN: %llvmgxx -S %s -o - | llvm-as -o /dev/null

struct Gfx {
  void opMoveSetShowText();
};

struct Operator {
  void (Gfx::*func)();
};

Operator opTab[] = {
  {&Gfx::opMoveSetShowText},
};

