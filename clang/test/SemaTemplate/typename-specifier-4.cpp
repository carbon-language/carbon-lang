// RUN: clang-cc -fsyntax-only -verify %s
template<typename T, typename U> 
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

template<typename MetaFun, typename T1, typename T2>
struct metafun_apply2 {
  typedef typename MetaFun::template apply<T1, T2> inner;
  typedef typename inner::type type;
};

template<typename T, typename U> struct pair;

struct make_pair {
  template<typename T1, typename T2>
  struct apply {
    typedef pair<T1, T2> type;
  };
};

int a0[is_same<metafun_apply2<make_pair, int, float>::type, 
               pair<int, float> >::value? 1 : -1];
int a1[is_same<
         typename make_pair::template apply<int, float>,
         make_pair::apply<int, float>
       >::value? 1 : -1];

template<typename MetaFun>
struct swap_and_apply2 {
  template<typename T1, typename T2>
  struct apply {
    typedef typename MetaFun::template apply<T2, T1> new_metafun;
    typedef typename new_metafun::type type;
  };
};

int a2[is_same<swap_and_apply2<make_pair>::apply<int, float>::type, 
               pair<float, int> >::value? 1 : -1];

template<typename MetaFun>
struct swap_and_apply2b {
  template<typename T1, typename T2>
  struct apply {
    typedef typename MetaFun::template apply<T2, T1>::type type;
  };
};

int a3[is_same<swap_and_apply2b<make_pair>::apply<int, float>::type, 
               pair<float, int> >::value? 1 : -1];

template<typename T>
struct X0 {
  template<typename U, typename V>
  struct Inner;
  
  void f0(X0<T>::Inner<T*, T&>); // expected-note{{here}}
  void f0(typename X0<T>::Inner<T*, T&>); // expected-error{{redecl}}

  void f1(X0<T>::Inner<T*, T&>); // expected-note{{here}}
  void f1(typename X0<T>::template Inner<T*, T&>); // expected-error{{redecl}}

  void f2(typename X0<T>::Inner<T*, T&>::type); // expected-note{{here}}
  void f2(typename X0<T>::template Inner<T*, T&>::type); // expected-error{{redecl}}
};
