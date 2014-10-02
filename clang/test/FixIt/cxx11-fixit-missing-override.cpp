// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -std=c++11 -Winconsistent-missing-override -fixit %t
// RUN: %clang_cc1 -x c++ -std=c++11 -Winconsistent-missing-override -Werror %t

struct A
{
    virtual void foo();
    virtual void bar(); // expected-note {{overridden virtual function is here}}
    virtual void gorf() {}
    virtual void g() = 0; // expected-note {{overridden virtual function is here}}
};
 
struct B : A
{
    void foo() override;
    void bar(); // expected-warning {{'bar' overrides a member function but is not marked 'override'}}
};

struct C : B 
{
    virtual void g() override = 0;  // expected-warning {{'g' overrides a member function but is not marked 'override'}}
    virtual void gorf() override {}
    void foo() {}
};

struct D : C {
  virtual void g()override ;
  virtual void foo(){
  }
  void bar() override;
};


