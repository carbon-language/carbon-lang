// Header for PCH test cxx-templates.cpp

template <typename T1, typename T2>
struct S {
  static void templ();
};

template <typename T>
struct S<int, T> {
    static void partial();
};

template <>
struct S<int, float> {
    static void explicit_special();
};

template <typename T>
T templ_f(T x) {
  return x;
}
