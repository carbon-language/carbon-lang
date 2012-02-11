// RUN: %clang_cc1 -verify %s

template<typename T> struct atomic {
  _Atomic(T) value;
};

template<typename T> struct user {
  struct inner { char n[sizeof(T)]; };
  atomic<inner> i;
};

user<int> u;
