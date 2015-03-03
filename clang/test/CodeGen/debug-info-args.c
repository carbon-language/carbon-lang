// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -g %s | FileCheck %s

int somefunc(char *x, int y, double z) {
  
  // CHECK: !MDSubroutineType(types: ![[NUM:[0-9]+]])
  // CHECK: ![[NUM]] = {{!{![^,]*, ![^,]*, ![^,]*, ![^,]*}}}
  
  return y;
}
