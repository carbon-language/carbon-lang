; RUN: as < %s | opt -scalarrepl

int %test() {
  %X = alloca { [ 4 x int] }
  %Y = getelementptr { [ 4 x int] }* %X, long 0, ubyte 0, long 2
  store int 4, int* %Y
  %Z = load int* %Y
  ret int %Z
}
