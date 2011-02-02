// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// rdar://8945175

struct X { 
  int array[0]; 
  int array1[0]; 
  int array2[0]; 
  X();
  ~X();
};

struct Y {
  int first;
  X padding;
  int second;
};

int main() {
// CHECK: store i32 0, i32* [[RETVAL:%.*]]
  return sizeof(Y) -8 ;
}
