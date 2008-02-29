// RUN: clang %s -verify -fsyntax-only
struct xx { int bitf:1; };

struct entry { struct xx *whatever; 
               int value; 
               int bitf:1; };
void add_one(int *p) { (*p)++; }

void test() {
 register struct entry *p;
 add_one(&p->value);
 struct entry pvalue;
 add_one(&p->bitf);  // expected-error {{address of bit-field requested}}
 add_one(&pvalue.bitf); // expected-error {{address of bit-field requested}}
 add_one(&p->whatever->bitf); // expected-error {{address of bit-field requested}}
}

void foo() {
  register int x[10];
  &x[10];              // expected-error {{address of register variable requested}}
    
  register int *y;
  
  int *x2 = &y; // expected-error {{address of register variable requested}}
  int *x3 = &y[10];
}

void testVectorComponentAccess() {
  typedef float v4sf __attribute__ ((vector_size (16)));
  static v4sf q;
  float* r = &q[0]; // expected-error {{address of vector requested}}
}

