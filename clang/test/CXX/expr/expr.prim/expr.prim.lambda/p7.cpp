// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

// Check that analysis-based warnings work in lambda bodies.
void analysis_based_warnings() {
  (void)[]() -> int { }; // expected-warning{{control reaches end of non-void function}} \
  // expected-error{{lambda expressions are not supported yet}}
}

// Check that we get the right types of captured variables (the
// semantic-analysis part of p7).
int &check_const_int(int&);
float &check_const_int(const int&);

void test_capture_constness(int i, const int ic) {
  (void)[i,ic] ()->void { // expected-error{{lambda expressions are not supported yet}}
    float &fr1 = check_const_int(i);
    float &fr2 = check_const_int(ic);
  }; 

  (void)[=] ()->void { // expected-error{{lambda expressions are not supported yet}}
    float &fr1 = check_const_int(i);
    float &fr2 = check_const_int(ic);
  }; 

  (void)[i,ic] () mutable ->void { // expected-error{{lambda expressions are not supported yet}}
    int &ir = check_const_int(i);
    float &fr = check_const_int(ic);
  };

  (void)[=] () mutable ->void { // expected-error{{lambda expressions are not supported yet}}
    int &ir = check_const_int(i);
    float &fr = check_const_int(ic);
  };

  (void)[&i,&ic] ()->void { // expected-error{{lambda expressions are not supported yet}}
    int &ir = check_const_int(i);
    float &fr = check_const_int(ic);
  };

  (void)[&] ()->void { // expected-error{{lambda expressions are not supported yet}}
    int &ir = check_const_int(i);
    float &fr = check_const_int(ic);
  };
}


