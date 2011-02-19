// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef __typeof__(((int*)0)-((int*)0)) ptrdiff_t;

namespace DontResolveTooEarly_WaitForOverloadResolution
{
  template <class T> T* f(int);	// #1 
  template <class T, class U> T& f(U); // #2 
  
  void g() {
    int *ip = f<int>(1);	// calls #1
  }

  template <class T> 
    T* f2(int); 
  template <class T, class U> 
    T& f2(U); 

  void g2() {
    int*ip = (f2<int>)(1); // ok
  }

} // End namespace

  template<typename T>
    void twoT() { }
  template<typename T, typename U>
    void twoT(T) { }
  

  void two() { }; //expected-note 5{{candidate}}
  void two(int) { }; //expected-note 5{{candidate}}
 


  void one() { }
  template<class T> 
    void oneT() { }

  template<class T>
  void cant_resolve() { } //expected-note 3{{candidate}}
 
  template<class T> void cant_resolve(T) { }//expected-note 3{{candidate}}
 

int main()
{
  { static_cast<void>(one); }
  { (void)(one); }
  { static_cast<void>(oneT<int>); }
  { (void)(oneT<int>); }

  { static_cast<void>(two); } // expected-error {{address of overloaded}}
  { (void)(two); } // expected-error {{address of overloaded}}
  { static_cast<void>(twoT<int>); } 
  { (void)(twoT<int>); } 


  { ptrdiff_t x = reinterpret_cast<ptrdiff_t>(oneT<int>); } 
  { (void) reinterpret_cast<int (*)(char, double)>(oneT<int>); } 
  { (void) reinterpret_cast<ptrdiff_t>(one); }
  { (void) reinterpret_cast<int (*)(char, double)>(one); }

  { ptrdiff_t x = reinterpret_cast<ptrdiff_t>(twoT<int>); }  
  { (void) reinterpret_cast<int (*)(char, double)>(twoT<int>); } 
  { (void) reinterpret_cast<void (*)(int)>(two); } //expected-error {{reinterpret_cast}}
  { (void) static_cast<void (*)(int)>(two); } //ok

  { (void) reinterpret_cast<int>(two); } //expected-error {{reinterpret_cast}}
  { (void) reinterpret_cast<int (*)(char, double)>(two); } //expected-error {{reinterpret_cast}}

  { bool b = (twoT<int>); } // ok
  { bool b = (twoT<int, int>); } //ok

  { bool b = &twoT<int>; //&foo<int>; }
    b = &(twoT<int>); }

  { ptrdiff_t x = (ptrdiff_t) &twoT<int>;
      x = (ptrdiff_t) &twoT<int>; }

  { ptrdiff_t x = (ptrdiff_t) twoT<int>;
      x = (ptrdiff_t) twoT<int>; }

  
  { ptrdiff_t x = (ptrdiff_t) &twoT<int,int>;
  x = (ptrdiff_t) &twoT<int>; }

  { oneT<int>;   &oneT<int>; } //expected-warning 2{{ expression result unused }}
  { static_cast<void>(cant_resolve<int>); } // expected-error {{address of overload}}
  { bool b = cant_resolve<int>; } // expected-error {{address of overload}}
  { (void) cant_resolve<int>; } // expected-error {{address of overload}}

}


