// RUN: clang-cc -verify -emit-llvm -o %t %s

void t1() {
  extern int& a;
  int b = a; 
}

void t2(int& a) {
  int b = a;
}

int g;
int& gr = g;
void t3() {
  int b = gr;
}
