; RUN: llvm-as < %s | opt -scalarrepl -mem2reg | llvm-dis | not grep alloca

int %test() {
  %X = alloca { int, float }
  %Y = getelementptr {int,float}* %X, long 0, ubyte 0
  store int 0, int* %Y

  %Z = load int* %Y
  ret int %Z
}
