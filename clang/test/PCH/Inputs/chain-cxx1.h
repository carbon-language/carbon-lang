// Primary header for C++ chained PCH test

void f();

// Name not appearing in dependent
void pf();

namespace ns {
  void g();

  void pg();
}

template <typename T>
struct S { typedef int G; };

// Partially specialize
template <typename T>
struct S<T *> { typedef int H; };
