// RUN: %clang_cc1 -fsyntax-only -faccess-control -verify %s

class M {
  int iM;
};

class P {
  int iP; // expected-note {{declared private here}}
  int PPR(); // expected-note {{declared private here}}
};

class N : M,P {
  N() {}
  int PR() { return iP + PPR(); } // expected-error 2 {{access to private member of 'class P'}}
};
