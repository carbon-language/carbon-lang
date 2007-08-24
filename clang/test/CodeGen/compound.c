// RUN: clang %s -emit-llvm
int A;
long long B;
int C;
int *P;
void foo() {
  C = (A /= B);

  P -= 4;

  C = P - (P+10);
}


