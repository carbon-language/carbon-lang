// RUN: %llvmgcc -S 2004-11-27-StaticFunctionRedeclare.c -o - | gccas | llvm-dis | not grep 'declare int.*func'

// There should not be an unresolved reference to func here.  Believe it or not,
// the "expected result" is a function named 'func' which is internal and 
// referenced by bar().

// This is PR244

static int func();
void bar() {
  int func();
  foo(func);
}
static int func(char** A, char ** B) {}
