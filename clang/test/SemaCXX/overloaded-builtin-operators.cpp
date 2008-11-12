// RUN: clang -fsyntax-only -verify %s 
struct yes;
struct no;

struct Short {
  operator short();
};

struct Long {
  operator long();
};

yes& islong(long);
no& islong(int);

void f(Short s, Long l) {
  // C++ [over.built]p12
  (void)static_cast<yes&>(islong(s + l));
  (void)static_cast<no&>(islong(s + s));

  // C++ [over.built]p17
  (void)static_cast<yes&>(islong(s % l));
  (void)static_cast<yes&>(islong(l << s));
  (void)static_cast<no&>(islong(s << l));
}

struct ShortRef {
  operator short&();
};

struct LongRef {
  operator volatile long&();
};

void g(ShortRef sr, LongRef lr) {
  // C++ [over.built]p18
  short& sr1 = (sr *= lr);
  volatile long& lr1 = (lr *= sr);

  // C++ [over.built]p22
  short& sr2 = (sr %= lr);
  volatile long& lr2 = (lr <<= sr);

  bool b1 = (sr && lr) || (sr || lr);
}

struct VolatileIntPtr {
  operator int volatile *();
};

struct ConstIntPtr {
  operator int const *();
};

void test_with_ptrs(VolatileIntPtr vip, ConstIntPtr cip, ShortRef sr) {
#if 0
  // FIXME: Enable these tests once we have operator overloading for
  // operator[].
  const int& cir1 = cip[sr];
  const int& cir2 = sr[cip];
  volatile int& vir1 = vip[sr];
  volatile int& vir2 = sr[vip];
#endif
  bool b1 = (vip == cip);
  long p1 = vip - cip;
}
