// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// Member function declarations with the same name and the same
// parameter-type-list as well as mem- ber function template
// declarations with the same name, the same parameter-type-list, and
// the same template parameter lists cannot be overloaded if any of
// them, but not all, have a ref-qualifier (8.3.5).

class Y { 
  void h() &; 
  void h() const &; 
  void h() &&; 
  void i() &; 
  void i() const; // FIXME: expected an error here!

  template<typename T> void f(T*) &;
  template<typename T> void f(T*) &&;

  template<typename T> void g(T*) &;
  template<typename T> void g(T*); // FIXME: expected an error here
};
