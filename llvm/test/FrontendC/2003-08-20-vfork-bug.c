// RUN: %llvmgcc -S %s -o - | llvm-as -o /dev/null

extern int vfork(void);
test() {
  vfork();
}
