// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-option -verify %s

// C++98 [basic.lookup.classref]p1:
//   In a class member access expression (5.2.5), if the . or -> token is
//   immediately followed by an identifier followed by a <, the identifier must
//   be looked up to determine whether the < is the beginning of a template
//   argument list (14.2) or a less-than operator. The identifier is first
//   looked up in the class of the object expression. If the identifier is not
//   found, it is then looked up in the context of the entire postfix-expression
//   and shall name a class or function template. If the lookup in the class of
//   the object expression finds a template, the name is also looked up in the
//   context of the entire postfix-expression and
//    -- if the name is not found, the name found in the class of the object
//       expression is used, otherwise
//    -- if the name is found in the context of the entire postfix-expression
//       and does not name a class template, the name found in the class of the
//       object expression is used, otherwise
//    -- if the name found is a class template, it must refer to the same
//       entity as the one found in the class of the object expression,
//       otherwise the program is ill-formed.

// From PR 7247
template<typename T>
struct set{};  // expected-note{{lookup from the current scope refers here}}
struct Value {
  template<typename T>
  void set(T value) {}  // expected-note{{lookup in the object type 'Value' refers here}}

  void resolves_to_same() {
    Value v;
    v.set<double>(3.2);
  }
};
void resolves_to_different() {
  {
    Value v;
    // The fact that the next line is a warning rather than an error is an
    // extension.
    v.set<double>(3.2);  // expected-warning{{lookup of 'set' in member access expression is ambiguous; using member of 'Value' [-Wambiguous-member-template]}}
  }
  {
    int set;  // Non-template.
    Value v;
    v.set<double>(3.2);
  }
}

namespace rdar9915664 {
  struct A {
    template<typename T> void a();
  };

  struct B : A { };

  struct C : A { };

  struct D : B, C {
    A &getA() { return static_cast<B&>(*this); }

    void test_a() {
      getA().a<int>();
    }
  };
}
