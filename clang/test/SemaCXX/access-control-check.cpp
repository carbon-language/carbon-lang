// RUN: %clang_cc1 -fsyntax-only -faccess-control -verify %s

class M {
  int iM;
};

class P {
  int iP;
  int PPR();
};

class N : M,P {
  N() {}
  // FIXME. No access violation is reported in method call or member access.
  int PR() { return iP + PPR(); }
};
