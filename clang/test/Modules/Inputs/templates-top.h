template<typename T> class Vector;

template<typename T> class List {
public:
  void push_back(T);

  struct node {};
  node *head;
  unsigned size;
};

extern List<double> *instantiateListDoubleDeclaration;

namespace A {
  class Y {
    template <typename T> friend class WhereAmI;
  };
}

template <typename T> class A::WhereAmI {
public:
  static void func() {}
};

template<typename T> struct Outer {
  struct Inner {};
};

template<bool, bool> struct ExplicitInstantiation {
  void f() {}
};

template<typename> struct DelayUpdates {};

template<typename T> struct OutOfLineInline {
  void f();
  void g();
  void h();
};
template<typename T> inline void OutOfLineInline<T>::f() {}
template<typename T> inline void OutOfLineInline<T>::g() {}
template<typename T> inline void OutOfLineInline<T>::h() {}
