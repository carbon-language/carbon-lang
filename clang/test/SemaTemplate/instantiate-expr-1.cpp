// RUN: clang-cc -fsyntax-only -verify %s
template<int I, int J>
struct Bitfields {
  int simple : I; // expected-error{{bit-field 'simple' has zero width}}
  int parens : (J);
};

void test_Bitfields(Bitfields<0, 5> *b) {
  (void)sizeof(Bitfields<10, 5>);
  (void)sizeof(Bitfields<0, 1>); // expected-note{{in instantiation of template class 'struct Bitfields<0, 1>' requested here}}
}

template<int I, int J>
struct BitfieldPlus {
  int bitfield : I + J; // expected-error{{bit-field 'bitfield' has zero width}}
};

void test_BitfieldPlus() {
  (void)sizeof(BitfieldPlus<0, 1>);
  (void)sizeof(BitfieldPlus<-5, 5>); // expected-note{{in instantiation of template class 'struct BitfieldPlus<-5, 5>' requested here}}
}

template<int I, int J>
struct BitfieldMinus {
  int bitfield : I - J; // expected-error{{bit-field 'bitfield' has negative width (-1)}} \
  // expected-error{{bit-field 'bitfield' has zero width}}
};

void test_BitfieldMinus() {
  (void)sizeof(BitfieldMinus<5, 1>);
  (void)sizeof(BitfieldMinus<0, 1>); // expected-note{{in instantiation of template class 'struct BitfieldMinus<0, 1>' requested here}}
  (void)sizeof(BitfieldMinus<5, 5>); // expected-note{{in instantiation of template class 'struct BitfieldMinus<5, 5>' requested here}}
}

template<int I, int J>
struct BitfieldDivide {
  int bitfield : I / J; // expected-error{{expression is not an integer constant expression}} \
                        // expected-note{{division by zero}}
};

void test_BitfieldDivide() {
  (void)sizeof(BitfieldDivide<5, 1>);
  (void)sizeof(BitfieldDivide<5, 0>); // expected-note{{in instantiation of template class 'struct BitfieldDivide<5, 0>' requested here}}
}

template<typename T, T I, int J>
struct BitfieldDep {
  int bitfield : I + J;
};

void test_BitfieldDep() {
  (void)sizeof(BitfieldDep<int, 1, 5>);
}

template<int I>
struct BitfieldNeg {
  int bitfield : (-I); // expected-error{{bit-field 'bitfield' has negative width (-5)}}
};

template<typename T, T I>
struct BitfieldNeg2 {
  int bitfield : (-I); // expected-error{{bit-field 'bitfield' has negative width (-5)}}
};

void test_BitfieldNeg() {
  (void)sizeof(BitfieldNeg<-5>); // okay
  (void)sizeof(BitfieldNeg<5>); // expected-note{{in instantiation of template class 'struct BitfieldNeg<5>' requested here}}
  (void)sizeof(BitfieldNeg2<int, -5>); // okay
  (void)sizeof(BitfieldNeg2<int, 5>); // expected-note{{in instantiation of template class 'struct BitfieldNeg2<int, 5>' requested here}}
}

template<typename T>
void increment(T &x) {
  (void)++x;
}

struct Incrementable {
  Incrementable &operator++();
};

void test_increment(Incrementable inc) {
  increment(inc);
}

template<typename T>
void add(const T &x) {
  (void)(x + x);
}

struct Addable {
  Addable operator+(const Addable&) const;
};

void test_add(Addable &a) {
  add(a);
}

struct CallOperator {
  int &operator()(int);
  double &operator()(double);
};

template<typename Result, typename F, typename Arg1>
Result test_call_operator(F f, Arg1 arg1) {
  // PR5266: non-dependent invocations of a function call operator.
  CallOperator call_op;
  int &ir = call_op(17);
  return f(arg1);
}

void test_call_operator(CallOperator call_op, int i, double d) {
  int &ir = test_call_operator<int&>(call_op, i);
  double &dr = test_call_operator<double&>(call_op, d);
}
