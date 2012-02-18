// RUN: %clang_cc1 -std=c++11 -fblocks %s -verify

void block_capture_errors() {
  __block int var; // expected-note 2{{'var' declared here}}
  (void)[var] { }; // expected-error{{__block variable 'var' cannot be captured in a lambda}}

  (void)[=] { var = 17; }; // expected-error{{__block variable 'var' cannot be captured in a lambda}}
}

void conversion_to_block(int captured) {
  int (^b1)(int) = [=](int x) { return x + captured; };

  const auto lambda = [=](int x) { return x + captured; };
  int (^b2)(int) = lambda;
}

template<typename T>
class ConstCopyConstructorBoom {
public:
  ConstCopyConstructorBoom(ConstCopyConstructorBoom&);

  ConstCopyConstructorBoom(const ConstCopyConstructorBoom&) {
    T *ptr = 1; // expected-error{{cannot initialize a variable of type 'float *' with an rvalue of type 'int'}}
  }

  void foo() const;
};

void conversion_to_block_init(ConstCopyConstructorBoom<int> boom,
                              ConstCopyConstructorBoom<float> boom2) {
  const auto& lambda1([=] { boom.foo(); }); // okay

  const auto& lambda2([=] { boom2.foo(); }); // expected-note{{in instantiation of member function}}
  void (^block)(void) = lambda2;
}


void nesting() {
  int array[7]; // expected-note 2{{'array' declared here}}
  [=] () mutable {
    [&] {
      ^ {
        int i = array[2];
        i += array[3];
      }();
    }();
  }();

  [&] {
    [=] () mutable {
      ^ {
        int i = array[2]; // expected-error{{cannot refer to declaration with an array type inside block}}
        i += array[3]; // expected-error{{cannot refer to declaration with an array type inside block}}
      }();
    }();
  }();
}
