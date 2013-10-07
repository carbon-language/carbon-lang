template<typename T> class Vector;

template<typename T> class List {
public:
  void push_back(T);

  struct node {};
  node *head;
  unsigned size;
};

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
