// RUN: %llvmgcc -Werror -S %s -o /dev/null
#pragma mark LLVM's world
#ifdef DO_ERROR
#error LLVM's world
#endif
int i;
