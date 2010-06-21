// RUN: %llvmgxx -g -Os -S %s -o - | llvm-as -disable-output
// Do not use function name to create named metadata used to hold
// local variable info. For example. llvm.dbg.lv.~A is an invalid name.
class A {
public:
  ~A() { int i = 0; i++; }
};

int foo(int i) {
  A a;
  return 0;
}

