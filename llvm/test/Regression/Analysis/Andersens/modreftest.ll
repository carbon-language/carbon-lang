; RUN: llvm-as < %s | opt -anders-aa -load-vn -gcse -instcombine | llvm-dis | grep 'ret bool true'

%G = internal global int* null
declare int *%ext()
bool %bar() {
  %V1 = load int** %G
  %X2 = call int *%ext()
  %V2 = load int** %G
  store int* %X2, int** %G

  %C = seteq int* %V1, %V2
  ret bool %C
}
