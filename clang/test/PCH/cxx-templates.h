// Header for PCH test cxx-templates.cpp

template <typename T>
struct S {
    T x;
};

template <typename T>
T templ_f(T x) {
  return x;
}
