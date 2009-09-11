// RUN: clang-cc -fsyntax-only %s

template<typename T>
class X0 {
public:
  void f(T t);
  
  struct Inner {
    void g(T t);
  };
};

template<typename T>
void X0<T>::f(T t) {
  t = 17;
}

extern template class X0<int>;

extern template class X0<int*>;

template<typename T>
void X0<T>::Inner::g(T t) {
  t = 17;
}

void test_intptr(X0<int*> xi, X0<int*>::Inner xii) {
  xi.f(0);
  xii.g(0);
}
