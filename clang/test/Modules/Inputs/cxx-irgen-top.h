template<typename T> struct S {
  __attribute__((always_inline)) static int f() { return 0; }
  __attribute__((always_inline, visibility("hidden"))) static int g() { return 0; }
};

extern template struct S<int>;
