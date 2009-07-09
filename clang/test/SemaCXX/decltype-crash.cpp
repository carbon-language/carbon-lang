int& a();

void f() {
  decltype(a()) c;
}
