// RUN: clang -fsyntax-only -pedantic -verify %s 
int* f(int);
float* f(float);
void f();

void test_f(int iv, float fv) {
  float* fp = f(fv);
  int* ip = f(iv);
}

int* g(int, float, int); // expected-note {{ candidate function }}
float* g(int, int, int); // expected-note {{ candidate function }}
double* g(int, float, float); // expected-note {{ candidate function }}
char* g(int, float, ...); // expected-note {{ candidate function }}
void g();

void test_g(int iv, float fv) {
  int* ip1 = g(iv, fv, 0);
  float* fp1 = g(iv, iv, 0);
  double* dp1 = g(iv, fv, fv);
  char* cp1 = g(0, 0);
  char* cp2 = g(0, 0, 0, iv, fv);

  double* dp2 = g(0, fv, 1.5); // expected-error {{ call to 'g' is ambiguous; candidates are: }}
}

double* h(double f);
int* h(int);

void test_h(float fv, unsigned char cv) {
  double* dp = h(fv);
  int* ip = h(cv);
}

int* i(int);
double* i(long);

void test_i(short sv, int iv, long lv, unsigned char ucv) {
  int* ip1 = i(sv);
  int* ip2 = i(iv);
  int* ip3 = i(ucv);
  double* dp1 = i(lv);
}

int* j(void*);
double* j(bool);

void test_j(int* ip) {
  int* ip1 = j(ip);
}

int* k(char*);
double* k(bool);

void test_k() {
  int* ip1 = k("foo");
  double* dp1 = k(L"foo");
}

int* l(wchar_t*);
double* l(bool);

void test_l() {
  int* ip1 = l(L"foo");
  double* dp1 = l("foo");
}

int* m(const char*);
double* m(char*);

void test_m() {
  int* ip = m("foo");
}

int* n(char*);
double* n(void*);

void test_n() {
  char ca[7];
  int* ip1 = n(ca);
  int* ip2 = n("foo");

  float fa[7];
  double* dp1 = n(fa);
}

enum PromotesToInt {
  PromotesToIntValue = 1
};

enum PromotesToUnsignedInt {
  PromotesToUnsignedIntValue = (unsigned int)-1
};

int* o(int);
double* o(unsigned int);
float* o(long);

void test_o() {
  int* ip1 = o(PromotesToIntValue);
  double* dp1 = o(PromotesToUnsignedIntValue);
}

int* p(int);
double* p(double);

void test_p() {
  int* ip = p((short)1);
  double* dp = p(1.0f);
}

struct Bits {
  signed short int_bitfield : 5;
  unsigned int uint_bitfield : 8;
};

int* bitfields(int, int);
float* bitfields(unsigned int, int);

void test_bitfield(Bits bits, int x) {
  int* ip = bitfields(bits.int_bitfield, 0);
  float* fp = bitfields(bits.uint_bitfield, 0u);
}

int* multiparm(long, int, long); // expected-note {{ candidate function }}
float* multiparm(int, int, int); // expected-note {{ candidate function }}
double* multiparm(int, int, short); // expected-note {{ candidate function }}

void test_multiparm(long lv, short sv, int iv) {
  int* ip1 = multiparm(lv, iv, lv);
  int* ip2 = multiparm(lv, sv, lv);
  float* fp1 = multiparm(iv, iv, iv);
  float* fp2 = multiparm(sv, iv, iv);
  double* dp1 = multiparm(sv, sv, sv);
  double* dp2 = multiparm(iv, sv, sv);
  multiparm(sv, sv, lv); // expected-error {{ call to 'multiparm' is ambiguous; candidates are: }}
}

// Test overloading based on qualification vs. no qualification
// conversion.
int* quals1(int const * p);
char* quals1(int * p);

int* quals2(int const * const * pp);
char* quals2(int * * pp);

int* quals3(int const * * const * ppp);
char* quals3(int *** ppp);

void test_quals(int * p, int * * pp, int * * * ppp) {
  char* q1 = quals1(p);
  char* q2 = quals2(pp);
  char* q3 = quals3(ppp);
}

// Test overloading based on qualification ranking (C++ 13.3.2)p3.
int* quals_rank1(int const * p);
float* quals_rank1(int const volatile *p);

int* quals_rank2(int const * const * pp);
float* quals_rank2(int * const * pp);

void test_quals_ranking(int * p, int volatile *pq, int * * pp, int * * * ppp) {
  //  int* q1 = quals_rank1(p);
  float* q2 = quals_rank1(pq); 
}
