// RUN: clang-cc -fsyntax-only %s

template<typename T>
class X0 {
public:
  void f(T t);
};

template<typename T>
void X0<T>::f(T t) {
  t = 17;
}

// FIXME: Later, we'll want to write an explicit template
// declaration (extern template) for X0<int*>, then try to
// call X0<int*>::f. The translation unit should succeed, 
// because we're not allowed to instantiate the out-of-line
// definition of X0<T>::f. For now, this is just a parsing
// test.
extern template class X0<int>;
