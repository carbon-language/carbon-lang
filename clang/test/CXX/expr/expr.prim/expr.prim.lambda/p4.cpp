// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

void missing_lambda_declarator() {
  [](){}();
}

template<typename T> T get();

void infer_void_return_type(int i) {
  if (i > 17)
    return []() { }();

  if (i > 11)
    return []() { return; }();

  return [](int x) {
    switch (x) {
    case 0: return get<void>();
    case 1: return;
    case 2: return { 1, 2.0 }; // expected-error{{cannot deduce lambda return type from initializer list}}
    }
  }(7);
}

struct X { };

X infer_X_return_type(X x) {
  return [&x](int y) { // expected-warning{{omitted result type}}
    if (y > 0)
      return X();
    else
      return x;
  }(5);
}

X infer_X_return_type_fail(X x) { 
  return [x](int y) { // expected-warning{{omitted result type}}
    if (y > 0)
      return X();
    else // FIXME: shouldn't mention blocks
      return x; // expected-error{{return type 'const X' must match previous return type 'X' when block literal has unspecified explicit return type}}
  }(5);
}
