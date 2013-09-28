// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify -Wno-c++1y-extensions
// RUN: %clang_cc1 -fsyntax-only -std=c++1y %s -verify

void print();

template<typename T, typename... Ts>
void print(T first, Ts... rest) {
  (void)first;
  print(rest...);
}

template<typename... Ts>
void unexpanded_capture(Ts ...values) {
  auto unexp = [values] {}; // expected-error{{initializer contains unexpanded parameter pack 'values'}}
}

template<typename... Ts>
void implicit_capture(Ts ...values) {
  auto implicit = [&] { print(values...); };
  implicit();
}

template<typename... Ts>
void do_print(Ts... values) {
  auto bycopy = [values...]() { print(values...); };
  bycopy();
  auto byref = [&values...]() { print(values...); };
  byref();

  auto bycopy2 = [=]() { print(values...); };
  bycopy2();
  auto byref2 = [&]() { print(values...); };
  byref2();
}

template void do_print(int, float, double);

template<typename T, int... Values>
void bogus_expansions(T x) {
  auto l1 = [x...] {}; // expected-error{{pack expansion does not contain any unexpanded parameter packs}}
  auto l2 = [Values...] {}; // expected-error{{'Values' in capture list does not name a variable}}
}

void g(int*, float*, double*);

template<class... Args>
void std_example(Args... args) {
  auto lm = [&, args...] { return g(args...); };
};

template void std_example(int*, float*, double*);

template<typename ...Args>
void variadic_lambda(Args... args) {
  auto lambda = [](Args... inner_args) { return g(inner_args...); };
  lambda(args...);
}

template void variadic_lambda(int*, float*, double*);

template<typename ...Args>
void init_capture_pack_err(Args ...args) {
  [as(args)...] {} (); // expected-error {{expected ','}}
  [as...(args)]{} (); // expected-error {{expected ','}}
}

template<typename ...Args>
void init_capture_pack_multi(Args ...args) {
  [as(args...)] {} (); // expected-error {{initializer missing for lambda capture 'as'}} expected-error {{multiple}}
}
template void init_capture_pack_multi(); // expected-note {{instantiation}}
template void init_capture_pack_multi(int);
template void init_capture_pack_multi(int, int); // expected-note {{instantiation}}

template<typename ...Args>
void init_capture_pack_outer(Args ...args) {
  print([as(args)] { return sizeof(as); } () ...);
}
template void init_capture_pack_outer();
template void init_capture_pack_outer(int);
template void init_capture_pack_outer(int, int);
