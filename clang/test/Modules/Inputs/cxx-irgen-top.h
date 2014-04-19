template<typename T> struct S {
  __attribute__((always_inline)) static int f() { return 0; }
  __attribute__((always_inline, visibility("hidden"))) static int g() { return 0; }
};

extern template struct S<int>;

template<typename T> T min(T a, T b) { return a < b ? a : b; }

extern decltype(min(1, 2)) instantiate_min_decl;

template<typename T> struct CtorInit {
  static int f() { return 0; }
  int a;
  CtorInit() : a(f()) {}
};
