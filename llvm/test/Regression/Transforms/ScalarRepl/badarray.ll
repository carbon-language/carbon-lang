; RUN: llvm-as < %s | opt -scalarrepl -mem2reg | llvm-dis | grep alloca

int %test() {
  %X = alloca [ 4 x int ]
  %Y = getelementptr [4x int]* %X, long 0, long 6 ; Off end of array!
  store int 0, int* %Y

  %Z = load int* %Y
  ret int %Z
}
