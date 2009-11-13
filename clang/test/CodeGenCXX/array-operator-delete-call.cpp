// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: clang-cc -triple i386-apple-darwin -std=c++0x -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s

extern "C" int printf(...);

int count;

struct S {
  S() : iS (++count) { printf("S::S(%d)\n", iS); }
  ~S() { printf("S::~S(%d)\n", iS); }
  int iS;
};

struct COST
{
  S *cost;
  unsigned *cost_val;

  ~COST();
  COST();
};


COST::COST()
{
  cost = new S[3];
  cost_val = new unsigned[10];
}

COST::~COST()
{
  if (cost) {
   delete [] cost;
  }
  if (cost_val)
    delete [] cost_val;
}

COST c1;

int main()
{
  COST c3;
}
COST c2;

// CHECK-LP64: call     __ZdaPv

// CHECK-LP32: call     L__ZdaPv

