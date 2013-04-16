// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -fblocks

void test_nest_lambda() {
  int x;
  int y;
  [&,y]() {
    int z;
    #pragma clang __debug captured
    {
      x = y; // OK
      y = z; // expected-error{{cannot assign to a variable captured by copy in a non-mutable lambda}}
      z = y; // OK
    }
  }();

  int a;
  #pragma clang __debug captured
  {
    int b;
    int c;
    [&,c]() {
      a = b; // OK
      b = c; // OK
      c = a; // expected-error{{cannot assign to a variable captured by copy in a non-mutable lambda}}
    }();
  }
}

class test_obj_capture {
  int a;
  void b();
  static void test() {
    test_obj_capture c;
    #pragma clang __debug captured
    { (void)c.a; }  // OK
    #pragma clang __debug captured
    { c.b(); }      // OK
  }
};

class test_this_capture {
  int a;
  void b();
  void test() {
    #pragma clang __debug captured
    { (void)this; } // OK
    #pragma clang __debug captured
    { (void)a; }    // OK
    #pragma clang __debug captured
    { b(); }        // OK
  }
};
