// RUN: %clang_cc1 -fsyntax-only -verify %s

// This test concerns the identity of dependent types within the
// canonical type system, specifically focusing on the difference
// between members of the current instantiation and membmers of an
// unknown specialization. This considers C++ [temp.type], which
// specifies type equivalence within a template, and C++0x
// [temp.dep.type], which defines what it means to be a member of the
// current instantiation.

template<typename T, typename U>
struct X0 {
  typedef T T_type;
  typedef U U_type;

  void f0(T&); // expected-note{{previous}}
  void f0(typename X0::U_type&);
  void f0(typename X0::T_type&); // expected-error{{redecl}}

  void f1(T&); // expected-note{{previous}}
  void f1(typename X0::U_type&);
  void f1(typename X0<T, U>::T_type&); // expected-error{{redecl}}

  void f2(T&); // expected-note{{previous}}
  void f2(typename X0::U_type&);
  void f2(typename X0<T_type, U_type>::T_type&); // expected-error{{redecl}}

  void f3(T&); // expected-note{{previous}}
  void f3(typename X0::U_type&);
  void f3(typename ::X0<T_type, U_type>::T_type&); // expected-error{{redecl}}

  struct X1 {
    typedef T my_T_type;

    void g0(T&); // expected-note{{previous}}
    void g0(typename X0::U_type&);
    void g0(typename X0::T_type&); // expected-error{{redecl}}

    void g1(T&); // expected-note{{previous}}
    void g1(typename X0::U_type&);
    void g1(typename X0<T, U>::T_type&); // expected-error{{redecl}}
    
    void g2(T&); // expected-note{{previous}}
    void g2(typename X0::U_type&);
    void g2(typename X0<T_type, U_type>::T_type&); // expected-error{{redecl}}
    
    void g3(T&); // expected-note{{previous}}
    void g3(typename X0::U_type&);
    void g3(typename ::X0<T_type, U_type>::T_type&); // expected-error{{redecl}}

    void g4(T&); // expected-note{{previous}}
    void g4(typename X0::U_type&);
    void g4(typename X1::my_T_type&); // expected-error{{redecl}}

    void g5(T&); // expected-note{{previous}}
    void g5(typename X0::U_type&);
    void g5(typename X0::X1::my_T_type&); // expected-error{{redecl}}

    void g6(T&); // expected-note{{previous}}
    void g6(typename X0::U_type&);
    void g6(typename X0<T, U>::X1::my_T_type&); // expected-error{{redecl}}

    void g7(T&); // expected-note{{previous}}
    void g7(typename X0::U_type&);
    void g7(typename ::X0<typename X1::my_T_type, U_type>::X1::my_T_type&); // expected-error{{redecl}}

    void g8(T&); // expected-note{{previous}}
    void g8(typename X0<U, T_type>::T_type&);
    void g8(typename ::X0<typename X0<T_type, U>::X1::my_T_type, U_type>::X1::my_T_type&); // expected-error{{redecl}}
  };
};


template<typename T, typename U>
struct X0<T*, U*> {
  typedef T T_type;
  typedef U U_type;
  typedef T* Tptr;
  typedef U* Uptr;
  
  void f0(T&); // expected-note{{previous}}
  void f0(typename X0::U_type&);
  void f0(typename X0::T_type&); // expected-error{{redecl}}
  
  void f1(T&); // expected-note{{previous}}
  void f1(typename X0::U_type&);
  void f1(typename X0<T*, U*>::T_type&); // expected-error{{redecl}}
  
  void f2(T&); // expected-note{{previous}}
  void f2(typename X0::U_type&);
  void f2(typename X0<T_type*, U_type*>::T_type&); // expected-error{{redecl}}
  
  void f3(T&); // expected-note{{previous}}
  void f3(typename X0::U_type&);
  void f3(typename ::X0<T_type*, U_type*>::T_type&); // expected-error{{redecl}}

  void f4(T&); // expected-note{{previous}}
  void f4(typename X0::U_type&);
  void f4(typename ::X0<Tptr, Uptr>::T_type&); // expected-error{{redecl}}
  
  void f5(X0*); // expected-note{{previous}}
  void f5(::X0<T, U>*);
  void f5(::X0<T*, U*>*); // expected-error{{redecl}}
  
  struct X2 {
    typedef T my_T_type;
    
    void g0(T&); // expected-note{{previous}}
    void g0(typename X0::U_type&);
    void g0(typename X0::T_type&); // expected-error{{redecl}}
    
    void g1(T&); // expected-note{{previous}}
    void g1(typename X0::U_type&);
    void g1(typename X0<T*, U*>::T_type&); // expected-error{{redecl}}
    
    void g2(T&); // expected-note{{previous}}
    void g2(typename X0::U_type&);
    void g2(typename X0<T_type*, U_type*>::T_type&); // expected-error{{redecl}}
    
    void g3(T&); // expected-note{{previous}}
    void g3(typename X0::U_type&);
    void g3(typename ::X0<T_type*, U_type*>::T_type&); // expected-error{{redecl}}
    
    void g4(T&); // expected-note{{previous}}
    void g4(typename X0::U_type&);
    void g4(typename X2::my_T_type&); // expected-error{{redecl}}
    
    void g5(T&); // expected-note{{previous}}
    void g5(typename X0::U_type&);
    void g5(typename X0::X2::my_T_type&); // expected-error{{redecl}}
    
    void g6(T&); // expected-note{{previous}}
    void g6(typename X0::U_type&);
    void g6(typename X0<T*, U*>::X2::my_T_type&); // expected-error{{redecl}}
    
    void g7(T&); // expected-note{{previous}}
    void g7(typename X0::U_type&);
    void g7(typename ::X0<typename X2::my_T_type*, U_type*>::X2::my_T_type&); // expected-error{{redecl}}
    
    void g8(T&); // expected-note{{previous}}
    void g8(typename X0<U, T_type>::T_type&);
    void g8(typename ::X0<typename X0<T_type*, U*>::X2::my_T_type*, U_type*>::X2::my_T_type&); // expected-error{{redecl}}
  };
};

template<typename T>
struct X1 {
  static int *a;
  void f(float *b) {
    X1<T>::a = b; // expected-error{{incompatible}}
    X1<T*>::a = b;
  }
};
