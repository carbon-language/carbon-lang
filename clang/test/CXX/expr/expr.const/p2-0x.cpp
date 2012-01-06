// RUN: %clang_cc1 -fsyntax-only -std=c++11 -pedantic -verify -fcxx-exceptions %s -fconstexpr-depth 128

// A conditional-expression is a core constant expression unless it involves one
// of the following as a potentially evaluated subexpression [...]:

// - this (5.1.1 [expr.prim.general]) [Note: when evaluating a constant
//   expression, function invocation substitution (7.1.5 [dcl.constexpr])
//   replaces each occurrence of this in a constexpr member function with a
//   pointer to the class object. -end note];
struct This {
  int this1 : this1; // expected-error {{undeclared}}
  int this2 : this->this1; // expected-error {{invalid}}
  void this3() {
    int n1[this->this1]; // expected-warning {{variable length array}}
    int n2[this1]; // expected-warning {{variable length array}}
    (void)n1, (void)n2;
  }
};

// - an invocation of a function other than a constexpr constructor for a
//   literal class or a constexpr function [ Note: Overload resolution (13.3)
//   is applied as usual - end note ];
struct NonConstexpr1 {
  static int f() { return 1; } // expected-note {{here}}
  int n : f(); // expected-error {{constant expression}} expected-note {{non-constexpr function 'f' cannot be used in a constant expression}}
};
struct NonConstexpr2 {
  constexpr NonConstexpr2(); // expected-note {{here}}
  int n;
};
struct NonConstexpr3 {
  NonConstexpr3();
  int m : NonConstexpr2().n; // expected-error {{constant expression}} expected-note {{undefined constructor 'NonConstexpr2'}}
};
struct NonConstexpr4 {
  NonConstexpr4(); // expected-note {{declared here}}
  int n;
};
struct NonConstexpr5 {
  int n : NonConstexpr4().n; // expected-error {{constant expression}} expected-note {{non-constexpr constructor 'NonConstexpr4' cannot be used in a constant expression}}
};

// - an invocation of an undefined constexpr function or an undefined
//   constexpr constructor;
struct UndefinedConstexpr {
  constexpr UndefinedConstexpr();
  static constexpr int undefinedConstexpr1(); // expected-note {{here}}
  int undefinedConstexpr2 : undefinedConstexpr1(); // expected-error {{constant expression}} expected-note {{undefined function 'undefinedConstexpr1' cannot be used in a constant expression}}
};

// - an invocation of a constexpr function with arguments that, when substituted
//   by function invocation substitution (7.1.5), do not produce a constant
//   expression;
namespace NonConstExprReturn {
  static constexpr const int &id_ref(const int &n) {
    return n; // expected-note {{reference to temporary cannot be returned from a constexpr function}}
  }
  struct NonConstExprFunction {
    int n : id_ref( // expected-error {{constant expression}} expected-note {{in call to 'id_ref(16)'}}
        16 // expected-note {{temporary created here}}
        );
  };
  constexpr const int *address_of(const int &a) {
    return &a; // expected-note {{pointer to 'n' cannot be returned from a constexpr function}}
  }
  constexpr const int *return_param(int n) { // expected-note {{declared here}}
    return address_of(n); // expected-note {{in call to 'address_of(n)'}}
  }
  struct S {
    int n : *return_param(0); // expected-error {{constant expression}} expected-note {{in call to 'return_param(0)'}}
  };
}

// - an invocation of a constexpr constructor with arguments that, when
//   substituted by function invocation substitution (7.1.5), do not produce all
//   constant expressions for the constructor calls and full-expressions in the
//   mem-initializers (including conversions);
namespace NonConstExprCtor {
  struct T {
    constexpr T(const int &r) :
      r(r) { // expected-note 2{{reference to temporary cannot be used to initialize a member in a constant expression}}
    }
    const int &r;
  };
  constexpr int n = 0;
  constexpr T t1(n); // ok
  constexpr T t2(0); // expected-error {{must be initialized by a constant expression}} expected-note {{temporary created here}} expected-note {{in call to 'T(0)'}}

  struct S {
    int n : T(4).r; // expected-error {{constant expression}} expected-note {{temporary created here}} expected-note {{in call to 'T(4)'}}
  };
}

// - an invocation of a constexpr function or a constexpr constructor that would
//   exceed the implementation-defined recursion limits (see Annex B);
namespace RecursionLimits {
  constexpr int RecurseForever(int n) {
    return n + RecurseForever(n+1); // expected-note {{constexpr evaluation exceeded maximum depth of 128 calls}} expected-note 9{{in call to 'RecurseForever(}} expected-note {{skipping 118 calls}}
  }
  struct AlsoRecurseForever {
    constexpr AlsoRecurseForever(int n) :
      n(AlsoRecurseForever(n+1).n) // expected-note {{constexpr evaluation exceeded maximum depth of 128 calls}} expected-note 9{{in call to 'AlsoRecurseForever(}} expected-note {{skipping 118 calls}}
    {}
    int n;
  };
  struct S {
    int k : RecurseForever(0); // expected-error {{constant expression}} expected-note {{in call to}}
    int l : AlsoRecurseForever(0).n; // expected-error {{constant expression}} expected-note {{in call to}}
  };
}

// FIXME:
// - an operation that would have undefined behavior [Note: including, for
//   example, signed integer overflow (Clause 5 [expr]), certain pointer
//   arithmetic (5.7 [expr.add]), division by zero (5.6 [expr.mul]), or certain
//   shift operations (5.8 [expr.shift]) -end note];
namespace UndefinedBehavior {
  void f(int n) {
    switch (n) {
    case (int)4.4e9: // expected-error {{constant expression}} expected-note {{value 4.4E+9 is outside the range of representable values of type 'int'}}
    case (int)(unsigned)(long long)4.4e9: // ok
    case (float)1e300: // expected-error {{constant expression}} expected-note {{value 1.0E+300 is outside the range of representable values of type 'float'}}
    case (int)((float)1e37 / 1e30): // ok
    case (int)(__fp16)65536: // expected-error {{constant expression}} expected-note {{value 65536 is outside the range of representable values of type 'half'}}
      break;
    }
  }

  struct S {
    int m;
  };
  constexpr S s = { 5 }; // expected-note {{declared here}}
  constexpr const int *p = &s.m + 1;
  constexpr const int &f(const int *q) {
    return q[0]; // expected-note {{dereferenced pointer past the end of subobject of 's' is not a constant expression}}
  }
  struct T {
    int n : f(p); // expected-error {{not an integer constant expression}} expected-note {{in call to 'f(&s.m + 1)'}}
  };

  namespace Ptr {
    struct A {};
    struct B : A { int n; };
    B a[3][3];
    constexpr B *p = a[0] + 4; // expected-error {{constant expression}} expected-note {{element 4 of array of 3 elements}}
    B b = {};
    constexpr A *pa = &b + 1; // expected-error {{constant expression}} expected-note {{base class of pointer past the end}}
    constexpr B *pb = (B*)((A*)&b + 1); // expected-error {{constant expression}} expected-note {{derived class of pointer past the end}}
    constexpr const int *pn = &(&b + 1)->n; // expected-error {{constant expression}} expected-note {{field of pointer past the end}}
    constexpr B *parr = &a[3][0]; // expected-error {{constant expression}} expected-note {{array element of pointer past the end}}

    constexpr A *na = nullptr;
    constexpr B *nb = nullptr;
    constexpr A &ra = *nb; // expected-error {{constant expression}} expected-note {{cannot access base class of null pointer}}
    constexpr B &rb = (B&)*na; // expected-error {{constant expression}} expected-note {{cannot access derived class of null pointer}}
    static_assert((A*)nb == 0, "");
    static_assert((B*)na == 0, "");
    constexpr const int &nf = nb->n; // expected-error {{constant expression}} expected-note {{cannot access field of null pointer}}
    constexpr const int &np = (*(int(*)[4])nullptr)[2]; // expected-error {{constant expression}} expected-note {{cannot access array element of null pointer}}
  }
}

// - a lambda-expression (5.1.2);
struct Lambda {
  // FIXME: clang crashes when trying to parse this! Revisit this check once
  // lambdas are fully implemented.
  //int n : []{ return 1; }();
};

// - an lvalue-to-rvalue conversion (4.1) unless it is applied to
namespace LValueToRValue {
  // - a non-volatile glvalue of integral or enumeration type that refers to a
  //   non-volatile const object with a preceding initialization, initialized
  //   with a constant expression  [Note: a string literal (2.14.5 [lex.string])
  //   corresponds to an array of such objects. -end note], or
  volatile const int vi = 1; // expected-note {{here}}
  const int ci = 1;
  volatile const int &vrci = ci;
  static_assert(vi, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type 'const volatile int'}}
  static_assert(const_cast<int&>(vi), ""); // expected-error {{constant expression}} expected-note {{read of volatile object 'vi'}}
  static_assert(vrci, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}

  // - a non-volatile glvalue of literal type that refers to a non-volatile
  //   object defined with constexpr, or that refers to a sub-object of such an
  //   object, or
  struct S {
    constexpr S(int=0) : i(1), v(1) {}
    constexpr S(const S &s) : i(2), v(2) {}
    int i;
    volatile int v;
  };
  constexpr S s;
  constexpr volatile S vs; // expected-note {{here}}
  constexpr const volatile S &vrs = s;
  static_assert(s.i, "");
  static_assert(s.v, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}
  static_assert(vs.i, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}
  static_assert(const_cast<int&>(vs.i), ""); // expected-error {{constant expression}} expected-note {{read of volatile object 'vs'}}
  static_assert(vrs.i, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}

  // - a non-volatile glvalue of literal type that refers to a non-volatile
  //   temporary object whose lifetime has not ended, initialized with a
  //   constant expression;
  constexpr volatile S f() { return S(); }
  static_assert(f().i, ""); // ok! there's no lvalue-to-rvalue conversion here!
  static_assert(((volatile const S&&)(S)0).i, ""); // expected-error {{constant expression}} expected-note {{subexpression}}
}

// FIXME:
//
// DR1312: The proposed wording for this defect has issues, so we ignore this
// bullet and instead prohibit casts from pointers to cv void (see core-20842
// and core-20845).
//
// - an lvalue-to-rvalue conversion (4.1 [conv.lval]) that is applied to a
// glvalue of type cv1 T that refers to an object of type cv2 U, where T and U
// are neither the same type nor similar types (4.4 [conv.qual]);

// FIXME:
// - an lvalue-to-rvalue conversion (4.1) that is applied to a glvalue that
// refers to a non-active member of a union or a subobject thereof;

// - an id-expression that refers to a variable or data member of reference type
//   unless the reference has a preceding initialization, initialized with a
//   constant expression;
namespace References {
  const int a = 2;
  int &b = *const_cast<int*>(&a);
  int c = 10; // expected-note 2 {{here}}
  int &d = c;
  constexpr int e = 42;
  int &f = const_cast<int&>(e);
  extern int &g;
  constexpr int &h(); // expected-note 2{{here}}
  int &i = h(); // expected-note {{here}} expected-note {{undefined function 'h' cannot be used in a constant expression}}
  constexpr int &j() { return b; }
  int &k = j();

  struct S {
    int A : a;
    int B : b;
    int C : c; // expected-error {{constant expression}} expected-note {{read of non-const variable 'c'}}
    int D : d; // expected-error {{constant expression}} expected-note {{read of non-const variable 'c'}}
    int D2 : &d - &c + 1;
    int E : e / 2;
    int F : f - 11;
    int G : g; // expected-error {{constant expression}}
    int H : h(); // expected-error {{constant expression}} expected-note {{undefined function 'h'}}
    int I : i; // expected-error {{constant expression}} expected-note {{initializer of 'i' is not a constant expression}}
    int J : j();
    int K : k;
  };
}

// - a dynamic_cast (5.2.7);
namespace DynamicCast {
  struct S { int n; };
  constexpr S s { 16 };
  struct T {
    int n : dynamic_cast<const S*>(&s)->n; // expected-warning {{constant expression}} expected-note {{dynamic_cast}}
  };
}

// - a reinterpret_cast (5.2.10);
namespace ReinterpretCast {
  struct S { int n; };
  constexpr S s { 16 };
  struct T {
    int n : reinterpret_cast<const S*>(&s)->n; // expected-warning {{constant expression}} expected-note {{reinterpret_cast}}
  };
  struct U {
    int m : (long)(S*)6; // expected-warning {{constant expression}} expected-note {{reinterpret_cast}}
  };
}

// - a pseudo-destructor call (5.2.4);
namespace PseudoDtor {
  int k;
  typedef int I;
  struct T {
    int n : (k.~I(), 0); // expected-error {{constant expression}} expected-note{{subexpression}}
  };
}

// - increment or decrement operations (5.2.6, 5.3.2);
namespace IncDec {
  int k = 2;
  struct T {
    int n : ++k; // expected-error {{constant expression}}
    int m : --k; // expected-error {{constant expression}}
  };
}

// - a typeid expression (5.2.8) whose operand is of a polymorphic class type;
namespace std {
  struct type_info {
    virtual ~type_info();
    const char *name;
  };
}
namespace TypeId {
  struct S { virtual void f(); };
  constexpr S *p = 0;
  constexpr const std::type_info &ti1 = typeid(*p); // expected-error {{must be initialized by a constant expression}} expected-note {{typeid applied to expression of polymorphic type 'TypeId::S'}}

  struct T {} t;
  constexpr const std::type_info &ti2 = typeid(t);
}

// - a new-expression (5.3.4);
// - a delete-expression (5.3.5);
namespace NewDelete {
  int *p = 0;
  struct T {
    int n : *new int(4); // expected-error {{constant expression}} expected-note {{subexpression}}
    int m : (delete p, 2); // expected-error {{constant expression}} expected-note {{subexpression}}
  };
}

// - a relational (5.9) or equality (5.10) operator where the result is
//   unspecified;
namespace UnspecifiedRelations {
  int a, b;
  constexpr int *p = &a, *q = &b;
  // C++11 [expr.rel]p2: If two pointers p and q of the same type point to
  // different objects that are not members of the same array or to different
  // functions, or if only one of them is null, the results of p<q, p>q, p<=q,
  // and p>=q are unspecified.
  constexpr bool u1 = p < q; // expected-error {{constant expression}}
  constexpr bool u2 = p > q; // expected-error {{constant expression}}
  constexpr bool u3 = p <= q; // expected-error {{constant expression}}
  constexpr bool u4 = p >= q; // expected-error {{constant expression}}
  constexpr bool u5 = p < 0; // expected-error {{constant expression}}
  constexpr bool u6 = p <= 0; // expected-error {{constant expression}}
  constexpr bool u7 = p > 0; // expected-error {{constant expression}}
  constexpr bool u8 = p >= 0; // expected-error {{constant expression}}
  constexpr bool u9 = 0 < q; // expected-error {{constant expression}}
  constexpr bool u10 = 0 <= q; // expected-error {{constant expression}}
  constexpr bool u11 = 0 > q; // expected-error {{constant expression}}
  constexpr bool u12 = 0 >= q; // expected-error {{constant expression}}
  void f(), g();

  constexpr void (*pf)() = &f, (*pg)() = &g;
  constexpr bool u13 = pf < pg; // expected-error {{constant expression}}
  constexpr bool u14 = pf == pg;

  // FIXME:
  // If two pointers point to non-static data members of the same object with
  // different access control, the result is unspecified.

  // FIXME:
  // [expr.rel]p3: Pointers to void can be compared [...] if both pointers
  // represent the same address or are both the null pointer [...]; otherwise
  // the result is unspecified.

  // FIXME: Implement comparisons of pointers to members.
  // [expr.eq]p2: If either is a pointer to a virtual member function and
  // neither is null, the result is unspecified.
}

// - an assignment or a compound assignment (5.17); or
namespace Assignment {
  int k;
  struct T {
    int n : (k = 9); // expected-error {{constant expression}}
    int m : (k *= 2); // expected-error {{constant expression}}
  };

  struct Literal {
    constexpr Literal(const char *name) : name(name) {}
    const char *name;
  };
  struct Expr {
    constexpr Expr(Literal l) : IsLiteral(true), l(l) {}
    bool IsLiteral;
    union {
      Literal l;
      // ...
    };
  };
  struct MulEq {
    constexpr MulEq(Expr a, Expr b) : LHS(a), RHS(b) {}
    Expr LHS;
    Expr RHS;
  };
  constexpr MulEq operator*=(Expr a, Expr b) { return MulEq(a, b); }
  Literal a("a");
  Literal b("b");
  MulEq c = a *= b; // ok
}

// - a throw-expression (15.1)
namespace Throw {
  struct S {
    int n : (throw "hello", 10); // expected-error {{constant expression}} expected-note {{subexpression}}
  };
}

// PR9999
template<bool v>
class bitWidthHolding {
public:
  static const
  unsigned int width = (v == 0 ? 0 : bitWidthHolding<(v >> 1)>::width + 1);
};

static const int width=bitWidthHolding<255>::width;

template<bool b>
struct always_false {
  static const bool value = false;
};

template<bool b>
struct and_or {
  static const bool and_value = b && and_or<always_false<b>::value>::and_value;
  static const bool or_value = !b || and_or<always_false<b>::value>::or_value;
};

static const bool and_value = and_or<true>::and_value;
static const bool or_value = and_or<true>::or_value;
