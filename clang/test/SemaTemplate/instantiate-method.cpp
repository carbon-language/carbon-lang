// RUN: clang-cc -fsyntax-only -verify %s
template<typename T>
class X {
public:
  void f(T x); // expected-error{{argument may not have 'void' type}}
  void g(T*);

  static int h(T, T); // expected-error 2{{argument may not have 'void' type}}
};

int identity(int x) { return x; }

void test(X<int> *xi, int *ip, X<int(int)> *xf) {
  xi->f(17);
  xi->g(ip);
  xf->f(&identity);
  xf->g(identity);
  X<int>::h(17, 25);
  X<int(int)>::h(identity, &identity);
}

void test_bad() {
  X<void> xv; // expected-note{{in instantiation of template class 'class X<void>' requested here}}
}

template<typename T, typename U>
class Overloading {
public:
  int& f(T, T); // expected-note{{previous declaration is here}}
  float& f(T, U); // expected-error{{functions that differ only in their return type cannot be overloaded}}
};

void test_ovl(Overloading<int, long> *oil, int i, long l) {
  int &ir = oil->f(i, i);
  float &fr = oil->f(i, l);
}

void test_ovl_bad() {
  Overloading<float, float> off; // expected-note{{in instantiation of template class 'class Overloading<float, float>' requested here}}
}

template<typename T>
class HasDestructor {
public:
  virtual ~HasDestructor() = 0;
};

int i = sizeof(HasDestructor<int>); // FIXME: forces instantiation, but 
                // the code below should probably instantiate by itself.
int abstract_destructor[__is_abstract(HasDestructor<int>)? 1 : -1];


template<typename T>
class Constructors {
public:
  Constructors(const T&);
  Constructors(const Constructors &other);
};

void test_constructors() {
  Constructors<int> ci1(17);
  Constructors<int> ci2 = ci1;
}


template<typename T>
struct ConvertsTo {
  operator T();
};

void test_converts_to(ConvertsTo<int> ci, ConvertsTo<int *> cip) {
  int i = ci;
  int *ip = cip;
}
