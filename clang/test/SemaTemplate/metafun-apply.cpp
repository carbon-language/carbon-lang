// RUN: clang-cc -fsyntax-only %s

struct add_pointer {
  template<typename T>
  struct apply {
    typedef T* type;
  };
};

struct add_reference {
  template<typename T>
  struct apply {
    typedef T& type;
  };
};

template<typename MetaFun, typename T>
struct apply1 {
  typedef typename MetaFun::template apply<T>::type type;
};

#if 0
// FIXME: The code below requires template instantiation for dependent
// template-names that occur within nested-name-specifiers.
int i;

apply1<add_pointer, int>::type ip = &i;
apply1<add_reference, int>::type ir = i;
#endif
