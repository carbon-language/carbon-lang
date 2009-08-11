// RUN: clang-cc -triple x86_64-apple-darwin -S %s -o %t-64.s &&
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s &&
// RUN: clang-cc -triple i386-apple-darwin -S %s -o %t-32.s &&
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s &&
// RUN: true

extern "C" int printf(...);

int init = 100;

struct M {
  int iM;
  M() : iM(init++) {}
};

struct N {
  int iN;
  N() : iN(200) {}
  N(N const & arg){this->iN = arg.iN; }
};

struct P {
  int iP;
  P() : iP(init++) {}
};


struct X  : M, N, P { // ...
	X() : f1(1.0), d1(2.0), i1(3), name("HELLO"), bf1(0xff), bf2(0xabcd),
              au_i1(1234), au1_4("MASKED") {}
        P p0;
	void pr() { printf("iM = %d iN = %d, m1.iM = %d\n", iM, iN, m1.iM); 
                    printf("im = %d p0.iP = %d, p1.iP = %d\n", iP, p0.iP, p1.iP); 
                    printf("f1 = %f  d1 = %f  i1 = %d name(%s) \n", f1, d1, i1, name);
		    printf("bf1 = %x  bf2 = %x\n", bf1, bf2);
		    printf("au_i2 = %d\n", au_i2); 
                    printf("au1_1 = %s\n", au1_1); 
                   }
	M m1;
        P p1;
	float f1;
	double d1;
	int i1;
	const char *name;
	unsigned bf1 : 8;
        unsigned bf2 : 16;

	union {
          int au_i1;
          int au_i2;
        };
        union {
          const char * au1_1;
          float au1_2;
          int au1_3;
          const char * au1_4;
        };
};

int main()
{
	X a;
	X b(a);
        b.pr();
	X x;
	X c(x);
        c.pr();
}
// CHECK-LP64: .globl  __ZN1XC1ERK1X
// CHECK-LP64: .weak_definition __ZN1XC1ERK1X
// CHECK-LP64: __ZN1XC1ERK1X:

// CHECK-LP32: .globl  __ZN1XC1ERK1X
// CHECK-LP32: .weak_definition __ZN1XC1ERK1X
// CHECK-LP32: __ZN1XC1ERK1X:
