// PR1602
// RUN: %llvmgxx -S %s -o - -O3 | not grep ptrtoint
// RUN: %llvmgxx -S %s -o - -O3 | not grep getelementptr
// RUN: %llvmgxx -S %s -o - -O3 | grep bitcast


struct S { virtual void f(); };

typedef void (S::*P)(void);

const P p = &S::f; 

void g(S s) {
   (s.*p)();
 }
