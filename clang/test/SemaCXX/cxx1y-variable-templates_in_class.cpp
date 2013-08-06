// RUN: %clang_cc1 -verify -fsyntax-only %s -Wno-c++11-extensions -Wno-c++1y-extensions -DPRECXX11
// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only -Wno-c++1y-extensions %s
// RUN: %clang_cc1 -std=c++1y -verify -fsyntax-only %s

class A {
  template<typename T> const T wrong;           // expected-error {{member 'wrong' declared as a template}}
  template<typename T> const T wrong_init = 5;      // expected-error {{member 'wrong_init' declared as a template}}
  template<typename T, typename T0> static const T right = T(100);
  template<typename T> static const T right<T,int> = 5;
  template<typename T> const int right<int,T>;  // expected-error {{member 'right' declared as a template}}
  template<typename T> const float right<float,T> = 5;  // expected-error {{member 'right' declared as a template}}
  template<> static const int right<int,int> = 7;       // expected-error {{explicit specialization of 'right' in class scope}}
  template<> static const float right<float,int>;       // expected-error {{explicit specialization of 'right' in class scope}}
  template static const int right<int,int>;     // expected-error {{template specialization requires 'template<>'}} \
                                                // expected-error {{explicit specialization of 'right' in class scope}}
};

namespace out_of_line {
  class B0 {
    template<typename T, typename T0> static const T right = T(100);
    template<typename T> static const T right<T,int> = T(5);
  };
  template<> const int B0::right<int,int> = 7;
  template const int B0::right<int,int>;
  template<> const int B0::right<int,float>;
  template const int B0::right<int,float>;

  class B1 {
    template<typename T, typename T0> static const T right;
    template<typename T> static const T right<T,int>;
  };
  template<typename T, typename T0> const T B1::right = T(100);
  template<typename T> const T B1::right<T,int> = T(5);

  class B2 {
    template<typename T, typename T0> static const T right = T(100);  // expected-note {{previous definition is here}}
    template<typename T> static const T right<T,int> = T(5);          // expected-note {{previous definition is here}}
  };
  template<typename T, typename T0> const T B2::right = T(100);   // expected-error {{redefinition of 'right'}}
  template<typename T> const T B2::right<T,int> = T(5);           // expected-error {{redefinition of 'right'}}

  class B3 {
    template<typename T, typename T0> static const T right = T(100);
    template<typename T> static const T right<T,int> = T(5);
  };
  template<typename T, typename T0> const T B3::right;  // expected-error {{forward declaration of variable template cannot have a nested name specifier}}
  template<typename T> const T B3::right<T,int>;        // expected-error {{forward declaration of variable template partial specialization cannot have a nested name specifier}}

  class B4 {
    template<typename T, typename T0> static const T right;
    template<typename T> static const T right<T,int>;
    template<typename T, typename T0> static const T right_def = T(100);
    template<typename T> static const T right_def<T,int>;   // expected-note {{explicit instantiation refers here}}
  };
  template<typename T, typename T0> const T B4::right;  // expected-error {{forward declaration of variable template cannot have a nested name specifier}}
  template<typename T> const T B4::right<T,int>;        // expected-error {{forward declaration of variable template partial specialization cannot have a nested name specifier}} \
                                                        // expected-note {{explicit instantiation refers here}}
  template const int B4::right<int,int>;  // expected-error {{explicit instantiation of undefined static data member template 'right' of class}}
  template const int B4::right_def<int,int>;  // expected-error {{explicit instantiation of undefined static data member template 'right_def' of class}}
}

namespace non_const_init {
  class A {
    template<typename T> static T wrong_inst = T(10); // expected-error {{non-const static data member must be initialized out of line}}
    template<typename T> static T wrong_inst_fixed;
  };
  template int A::wrong_inst<int>;  // expected-note {{in instantiation of static data member 'non_const_init::A::wrong_inst<int>' requested here}}
  template<typename T> T A::wrong_inst_fixed = T(10);
  template int A::wrong_inst_fixed<int>;
  
  class B {
    template<typename T> static T wrong_inst;
    template<typename T> static T wrong_inst<T*> = T(100);  // expected-error {{non-const static data member must be initialized out of line}}
    
    template<typename T> static T wrong_inst_fixed;
    template<typename T> static T wrong_inst_fixed<T*>;
  };
  template int B::wrong_inst<int*>;  // expected-note {{in instantiation of static data member 'non_const_init::B::wrong_inst<int *>' requested here}}
  template<typename T> T B::wrong_inst_fixed = T(100);
  template int B::wrong_inst_fixed<int>;
  
  class C {
    template<typename T> static const T right_inst = T(10);
    template<typename T> static const T right_inst<T*> = T(100);
  };
  template const int C::right_inst<int>;
  template const int C::right_inst<int*>;
  
  namespace pointers {
    
    struct C0 {
      template<typename U> static U Data;
      template<typename U> static const U Data<U*> = U();   // Okay
    };
    template const int C0::Data<int*>;
    
    struct C1a {
      template<typename U> static U Data;
      template<typename U> static U* Data<U>;   // Okay, with out-of-line definition
    };
    template<typename T> T* C1a::Data<T> = new T();
    template int* C1a::Data<int>;
    
    struct C1b {
      template<typename U> static U Data;
      template<typename U> static const U* Data<U>;   // Okay, with out-of-line definition
    };
    template<typename T> const T* C1b::Data<T> = (T*)(0);
    template const int* C1b::Data<int>;

    struct C2a {
      template<typename U> static U Data;
      template<typename U> static U* Data<U> = new U();   // expected-error {{non-const static data member must be initialized out of line}}
    };
    template int* C2a::Data<int>; // expected-note {{in instantiation of static data member 'non_const_init::pointers::C2a::Data<int>' requested here}}
    
    struct C2b {  // FIXME: ?!? Should this be an error? pointer-types are automatically non-const?
      template<typename U> static U Data;
      template<typename U> static const U* Data<U> = (U*)(0); // expected-error {{non-const static data member must be initialized out of line}}
    };
    template const int* C2b::Data<int>; // expected-note {{in instantiation of static data member 'non_const_init::pointers::C2b::Data<int>' requested here}}
  }
}

struct matrix_constants {
  // TODO: (?)
};

namespace in_class_template {
  // FIXME: member data templates of class templates are not well supported yet.

  template<typename T>
  class D0 {
    template<typename U> static U Data;
    template<typename U> static const U Data<U*> = U();
  };

  template<typename T>
  class D1 {
    template<typename U> static U Data;
    template<typename U> static U* Data<U*>;
  };
  template<typename T>
  template<typename U> U* D1<T>::Data<U*> = (U*)(0);
    
  namespace to_be_fixed {
    // FIXME: The following generate runtime exceptions!

    //template<>
    //template<typename U> U* D1<float>::Data<U*> = (U*)(0) + 1;
    //template const int D0<float>::Data<int*>;
    //template int* D1<float>::Data<int*>;
  }
}

namespace in_nested_classes {
  // TODO:
}

