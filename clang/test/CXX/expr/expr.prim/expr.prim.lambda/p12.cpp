// RUN: %clang_cc1 -std=c++11 %s -Wunused -verify

void odr_used() {
  int i = 17;
  [i]{}();
}

struct ReachingThis {
  static void static_foo() {
    (void)[this](){}; // expected-error{{'this' cannot be captured in this context}}

    struct Local {
      int i;

      void bar() {
        (void)[this](){};
        (void)[&](){i = 7; };
      }
    };
  }

  void foo() {
    (void)[this](){};
    
    struct Local {
      int i;

      static void static_bar() {
        (void)[this](){}; // expected-error{{'this' cannot be captured in this context}}
        (void)[&](){i = 7; }; // expected-error{{invalid use of nonstatic data member 'i'}}
      }
    };
  }
};

void immediately_enclosing(int i) { // expected-note{{'i' declared here}}
  [i]() {
    [i] {}();
  }();

  [=]() {
    [i] {}();
  }();

  []() { // expected-note{{lambda expression begins here}}
    [i] {}(); // expected-error{{variable 'i' cannot be implicitly captured in a lambda with no capture-default specified}}
  }();
}
