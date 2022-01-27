// RUN: %clang_cc1 -fsyntax-only -verify %s
void g();

void f(); // expected-note 11{{candidate function}}
void f(int); // expected-note 11{{candidate function}}

template <class T>
void t(T); // expected-note 3{{candidate function}} \
           // expected-note 3{{candidate template ignored: could not match 'void' against 'int'}}
template <class T>
void t(T *); // expected-note 3{{candidate function}} \
             // expected-note 3{{candidate template ignored: could not match 'void' against 'int'}}

template<class T> void u(T);

int main()
{
  { bool b = (void (&)(char))f; } // expected-error{{does not match required type}}
  { bool b = (void (*)(char))f; } // expected-error{{does not match required type}}
  
  { bool b = (void (&)(int))f; } //ok
  { bool b = (void (*)(int))f; } //ok
  
  { bool b = static_cast<void (&)(char)>(f); } // expected-error{{does not match}}
  { bool b = static_cast<void (*)(char)>(f); } // expected-error{{address of overloaded function}}
  
  { bool b = static_cast<void (&)(int)>(f); } //ok
  { bool b = static_cast<void (*)(int)>(f); } //ok
  
  
  { bool b = reinterpret_cast<void (&)(char)>(f); } // expected-error{{cannot resolve}}
  { bool b = reinterpret_cast<void (*)(char)>(f); } // expected-error{{cannot resolve}}
  
  { bool b = reinterpret_cast<void (*)(char)>(g); } //ok
  { bool b = static_cast<void (*)(char)>(g); } // expected-error{{not allowed}}
  
  { bool b = reinterpret_cast<void (&)(int)>(f); } // expected-error{{cannot resolve}}
  { bool b = reinterpret_cast<void (*)(int)>(f); } // expected-error{{cannot resolve}}

  { bool b = (int (&)(char))t; } // expected-error{{does not match}}
  { bool b = (int (*)(char))t; } // expected-error{{does not match}}
  
  { bool b = (void (&)(int))t; } //ok
  { bool b = (void (*)(int))t; } //ok
  
  { bool b = static_cast<void (&)(char)>(t); } //ok
  { bool b = static_cast<void (*)(char)>(t); } //ok
  
  { bool b = static_cast<void (&)(int)>(t); } //ok
  { bool b = static_cast<void (*)(int)>(t); } //ok
  
  
  { bool b = reinterpret_cast<void (&)(char)>(t); } // expected-error{{cannot resolve}}
  { bool b = reinterpret_cast<void (*)(char)>(t); } // expected-error{{cannot resolve}}
  
  { bool b = reinterpret_cast<int (*)(char)>(g); } //ok
  { bool b = static_cast<int (*)(char)>(t); } // expected-error{{cannot be static_cast}}
  { bool b = static_cast<int (&)(char)>(t); } // expected-error{{does not match required}}
  
  { bool b = static_cast<void (&)(char)>(f); } // expected-error{{does not match}}

  {
    // The error should be reported when casting overloaded function to the
    // compatible function type (not to be confused with function pointer or
    // function reference type.)
    typedef void (FnType)(int);
    FnType a = static_cast<FnType>(f); // expected-error{{address of overloaded function}}
    FnType b = (FnType)(f); // expected-error{{address of overloaded function}}
  }
}
