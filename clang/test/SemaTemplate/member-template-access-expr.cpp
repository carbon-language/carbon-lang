// RUN: clang-cc -fsyntax-only -verify %s

template<typename U, typename T>
U f0(T t) {
  return t.template get<U>();
}

template<typename U, typename T>
int &f1(T t) {
  // FIXME: When we pretty-print this, we lose the "template" keyword.
  return t.U::template get<int&>();
}

struct X {
  template<typename T> T get();
};

void test_f0(X x) {
  int i = f0<int>(x);
  int &ir = f0<int&>(x);
}

struct XDerived : public X {
};

void test_f1(XDerived xd) {
  // FIXME: Not quite functional yet.
//  int &ir = f1<X>(xd);
}

// PR5213
template <class T>
struct A {};

template<class T>
class B
{
  A<T> a_;
  
public:
  void destroy();
};

template<class T>
void
B<T>::destroy()
{
  a_.~A<T>();
}

void do_destroy_B(B<int> b) {
  b.destroy();
}
