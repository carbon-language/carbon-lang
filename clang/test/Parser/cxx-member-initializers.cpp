struct x {
  x() : a(4) ; // expected-error {{expected '{'}}
};

struct y {
  int a;
  y() : a(4) ; // expected-error {{expected '{'}}
};
