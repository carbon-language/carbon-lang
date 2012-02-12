// RUN: %clang_cc1 -std=c++11 %s -Wunused -verify

struct MoveOnly {
  MoveOnly(MoveOnly&&);
  MoveOnly(const MoveOnly&);
};

template<typename T> T &&move(T&);
void test_special_member_functions(MoveOnly mo, int i) {
  // FIXME: terrible note
  auto lambda1 = [i]() { }; // expected-note{{function has been explicitly marked deleted here}} \
  // expected-note{{the implicit copy assignment operator}} \
  // expected-note{{the implicit move assignment operator}} \

  // Default constructor
  decltype(lambda1) lambda2; // expected-error{{call to deleted constructor}}

  // Copy assignment operator
  lambda1 = lambda1; // expected-error{{overload resolution selected deleted operator '='}}

  // Move assignment operator
  lambda1 = move(lambda1);

  // Copy constructor
  decltype(lambda1) lambda3 = lambda1;
  decltype(lambda1) lambda4(lambda1);

  // Move constructor
  decltype(lambda1) lambda5 = move(lambda1);
  decltype(lambda1) lambda6(move(lambda1));
}
