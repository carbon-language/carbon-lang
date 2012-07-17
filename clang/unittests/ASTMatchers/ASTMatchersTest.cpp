//===- unittest/Tooling/ASTMatchersTest.cpp - AST matcher unit tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ASTMatchersTest.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

#if GTEST_HAS_DEATH_TEST
TEST(HasNameDeathTest, DiesOnEmptyName) {
  ASSERT_DEBUG_DEATH({
    DeclarationMatcher HasEmptyName = record(hasName(""));
    EXPECT_TRUE(notMatches("class X {};", HasEmptyName));
  }, "");
}

TEST(HasNameDeathTest, DiesOnEmptyPattern) {
  ASSERT_DEBUG_DEATH({
      DeclarationMatcher HasEmptyName = record(matchesName(""));
      EXPECT_TRUE(notMatches("class X {};", HasEmptyName));
    }, "");
}

TEST(IsDerivedFromDeathTest, DiesOnEmptyBaseName) {
  ASSERT_DEBUG_DEATH({
    DeclarationMatcher IsDerivedFromEmpty = record(isDerivedFrom(""));
    EXPECT_TRUE(notMatches("class X {};", IsDerivedFromEmpty));
  }, "");
}
#endif

TEST(NameableDeclaration, MatchesVariousDecls) {
  DeclarationMatcher NamedX = nameableDeclaration(hasName("X"));
  EXPECT_TRUE(matches("typedef int X;", NamedX));
  EXPECT_TRUE(matches("int X;", NamedX));
  EXPECT_TRUE(matches("class foo { virtual void X(); };", NamedX));
  EXPECT_TRUE(matches("void foo() try { } catch(int X) { }", NamedX));
  EXPECT_TRUE(matches("void foo() { int X; }", NamedX));
  EXPECT_TRUE(matches("namespace X { }", NamedX));
  EXPECT_TRUE(matches("enum X { A, B, C };", NamedX));

  EXPECT_TRUE(notMatches("#define X 1", NamedX));
}

TEST(NameableDeclaration, REMatchesVariousDecls) {
  DeclarationMatcher NamedX = nameableDeclaration(matchesName("::X"));
  EXPECT_TRUE(matches("typedef int Xa;", NamedX));
  EXPECT_TRUE(matches("int Xb;", NamedX));
  EXPECT_TRUE(matches("class foo { virtual void Xc(); };", NamedX));
  EXPECT_TRUE(matches("void foo() try { } catch(int Xdef) { }", NamedX));
  EXPECT_TRUE(matches("void foo() { int Xgh; }", NamedX));
  EXPECT_TRUE(matches("namespace Xij { }", NamedX));
  EXPECT_TRUE(matches("enum X { A, B, C };", NamedX));

  EXPECT_TRUE(notMatches("#define Xkl 1", NamedX));

  DeclarationMatcher StartsWithNo = nameableDeclaration(matchesName("::no"));
  EXPECT_TRUE(matches("int no_foo;", StartsWithNo));
  EXPECT_TRUE(matches("class foo { virtual void nobody(); };", StartsWithNo));

  DeclarationMatcher Abc = nameableDeclaration(matchesName("a.*b.*c"));
  EXPECT_TRUE(matches("int abc;", Abc));
  EXPECT_TRUE(matches("int aFOObBARc;", Abc));
  EXPECT_TRUE(notMatches("int cab;", Abc));
  EXPECT_TRUE(matches("int cabc;", Abc));
}

TEST(DeclarationMatcher, MatchClass) {
  DeclarationMatcher ClassMatcher(record());
#if !defined(_MSC_VER)
  EXPECT_FALSE(matches("", ClassMatcher));
#else
  // Matches class type_info.
  EXPECT_TRUE(matches("", ClassMatcher));
#endif

  DeclarationMatcher ClassX = record(record(hasName("X")));
  EXPECT_TRUE(matches("class X;", ClassX));
  EXPECT_TRUE(matches("class X {};", ClassX));
  EXPECT_TRUE(matches("template<class T> class X {};", ClassX));
  EXPECT_TRUE(notMatches("", ClassX));
}

TEST(DeclarationMatcher, ClassIsDerived) {
  DeclarationMatcher IsDerivedFromX = record(isDerivedFrom("X"));

  EXPECT_TRUE(matches("class X {}; class Y : public X {};", IsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; class Y : public X {};", IsDerivedFromX));
  EXPECT_TRUE(matches("class X {};", IsDerivedFromX));
  EXPECT_TRUE(matches("class X;", IsDerivedFromX));
  EXPECT_TRUE(notMatches("class Y;", IsDerivedFromX));
  EXPECT_TRUE(notMatches("", IsDerivedFromX));

  DeclarationMatcher ZIsDerivedFromX =
      record(hasName("Z"), isDerivedFrom("X"));
  EXPECT_TRUE(
      matches("class X {}; class Y : public X {}; class Z : public Y {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class X {};"
              "template<class T> class Y : public X {};"
              "class Z : public Y<int> {};", ZIsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; template<class T> class Z : public X {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template<class T> class X {}; "
              "template<class T> class Z : public X<T> {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template<class T, class U=T> class X {}; "
              "template<class T> class Z : public X<T> {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<class X> class A { class Z : public X {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template<class X> class A { public: class Z : public X {}; }; "
              "class X{}; void y() { A<X>::Z z; }", ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template <class T> class X {}; "
              "template<class Y> class A { class Z : public X<Y> {}; };",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<template<class T> class X> class A { "
                 "  class Z : public X<int> {}; };", ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template<template<class T> class X> class A { "
              "  public: class Z : public X<int> {}; }; "
              "template<class T> class X {}; void y() { A<X>::Z z; }",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<class X> class A { class Z : public X::D {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template<class X> class A { public: "
              "  class Z : public X::D {}; }; "
              "class Y { public: class X {}; typedef X D; }; "
              "void y() { A<Y>::Z z; }", ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class X {}; typedef X Y; class Z : public Y {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template<class T> class Y { typedef typename T::U X; "
              "  class Z : public X {}; };", ZIsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; class Z : public ::X {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<class T> class X {}; "
                "template<class T> class A { class Z : public X<T>::D {}; };",
                ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template<class T> class X { public: typedef X<T> D; }; "
              "template<class T> class A { public: "
              "  class Z : public X<T>::D {}; }; void y() { A<int>::Z z; }",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<class X> class A { class Z : public X::D::E {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class X {}; typedef X V; typedef V W; class Z : public W {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class X {}; class Y : public X {}; "
              "typedef Y V; typedef V W; class Z : public W {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template<class T, class U> class X {}; "
              "template<class T> class A { class Z : public X<T, int> {}; };",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<class X> class D { typedef X A; typedef A B; "
                 "  typedef B C; class Z : public C {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class X {}; typedef X A; typedef A B; "
              "class Z : public B {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class X {}; typedef X A; typedef A B; typedef B C; "
              "class Z : public C {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class U {}; typedef U X; typedef X V; "
              "class Z : public V {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class Base {}; typedef Base X; "
              "class Z : public Base {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class Base {}; typedef Base Base2; typedef Base2 X; "
              "class Z : public Base {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("class Base {}; class Base2 {}; typedef Base2 X; "
                 "class Z : public Base {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class A {}; typedef A X; typedef A Y; "
              "class Z : public Y {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template <typename T> class Z;"
                 "template <> class Z<void> {};"
                 "template <typename T> class Z : public Z<void> {};",
                 IsDerivedFromX));
  EXPECT_TRUE(
      matches("template <typename T> class X;"
              "template <> class X<void> {};"
              "template <typename T> class X : public X<void> {};",
              IsDerivedFromX));
  EXPECT_TRUE(matches(
      "class X {};"
      "template <typename T> class Z;"
      "template <> class Z<void> {};"
      "template <typename T> class Z : public Z<void>, public X {};",
      ZIsDerivedFromX));

  // FIXME: Once we have better matchers for template type matching,
  // get rid of the Variable(...) matching and match the right template
  // declarations directly.
  const char *RecursiveTemplateOneParameter =
      "class Base1 {}; class Base2 {};"
      "template <typename T> class Z;"
      "template <> class Z<void> : public Base1 {};"
      "template <> class Z<int> : public Base2 {};"
      "template <> class Z<float> : public Z<void> {};"
      "template <> class Z<double> : public Z<int> {};"
      "template <typename T> class Z : public Z<float>, public Z<double> {};"
      "void f() { Z<float> z_float; Z<double> z_double; Z<char> z_char; }";
  EXPECT_TRUE(matches(
      RecursiveTemplateOneParameter,
      variable(hasName("z_float"),
               hasInitializer(hasType(record(isDerivedFrom("Base1")))))));
  EXPECT_TRUE(notMatches(
      RecursiveTemplateOneParameter,
      variable(
          hasName("z_float"),
          hasInitializer(hasType(record(isDerivedFrom("Base2")))))));
  EXPECT_TRUE(matches(
      RecursiveTemplateOneParameter,
      variable(
          hasName("z_char"),
          hasInitializer(hasType(record(isDerivedFrom("Base1"),
                                        isDerivedFrom("Base2")))))));

  const char *RecursiveTemplateTwoParameters =
      "class Base1 {}; class Base2 {};"
      "template <typename T1, typename T2> class Z;"
      "template <typename T> class Z<void, T> : public Base1 {};"
      "template <typename T> class Z<int, T> : public Base2 {};"
      "template <typename T> class Z<float, T> : public Z<void, T> {};"
      "template <typename T> class Z<double, T> : public Z<int, T> {};"
      "template <typename T1, typename T2> class Z : "
      "    public Z<float, T2>, public Z<double, T2> {};"
      "void f() { Z<float, void> z_float; Z<double, void> z_double; "
      "           Z<char, void> z_char; }";
  EXPECT_TRUE(matches(
      RecursiveTemplateTwoParameters,
      variable(
          hasName("z_float"),
          hasInitializer(hasType(record(isDerivedFrom("Base1")))))));
  EXPECT_TRUE(notMatches(
      RecursiveTemplateTwoParameters,
      variable(
          hasName("z_float"),
          hasInitializer(hasType(record(isDerivedFrom("Base2")))))));
  EXPECT_TRUE(matches(
      RecursiveTemplateTwoParameters,
      variable(
          hasName("z_char"),
          hasInitializer(hasType(record(isDerivedFrom("Base1"),
                                        isDerivedFrom("Base2")))))));
  EXPECT_TRUE(matches(
      "namespace ns { class X {}; class Y : public X {}; }",
      record(isDerivedFrom("::ns::X"))));
  EXPECT_TRUE(notMatches(
      "class X {}; class Y : public X {};",
      record(isDerivedFrom("::ns::X"))));

  EXPECT_TRUE(matches(
      "class X {}; class Y : public X {};",
      record(isDerivedFrom(id("test", record(hasName("X")))))));
}

TEST(AllOf, AllOverloadsWork) {
  const char Program[] =
      "struct T { }; int f(int, T*); void g(int x) { T t; f(x, &t); }";
  EXPECT_TRUE(matches(Program,
      call(allOf(callee(function(hasName("f"))),
                 hasArgument(0, declarationReference(to(variable())))))));
  EXPECT_TRUE(matches(Program,
      call(allOf(callee(function(hasName("f"))),
                 hasArgument(0, declarationReference(to(variable()))),
                 hasArgument(1, hasType(pointsTo(record(hasName("T")))))))));
}

TEST(DeclarationMatcher, MatchAnyOf) {
  DeclarationMatcher YOrZDerivedFromX =
      record(anyOf(hasName("Y"), allOf(isDerivedFrom("X"), hasName("Z"))));
  EXPECT_TRUE(
      matches("class X {}; class Z : public X {};", YOrZDerivedFromX));
  EXPECT_TRUE(matches("class Y {};", YOrZDerivedFromX));
  EXPECT_TRUE(
      notMatches("class X {}; class W : public X {};", YOrZDerivedFromX));
  EXPECT_TRUE(notMatches("class Z {};", YOrZDerivedFromX));

  DeclarationMatcher XOrYOrZOrU =
      record(anyOf(hasName("X"), hasName("Y"), hasName("Z"), hasName("U")));
  EXPECT_TRUE(matches("class X {};", XOrYOrZOrU));
  EXPECT_TRUE(notMatches("class V {};", XOrYOrZOrU));

  DeclarationMatcher XOrYOrZOrUOrV =
      record(anyOf(hasName("X"), hasName("Y"), hasName("Z"), hasName("U"),
                   hasName("V")));
  EXPECT_TRUE(matches("class X {};", XOrYOrZOrUOrV));
  EXPECT_TRUE(matches("class Y {};", XOrYOrZOrUOrV));
  EXPECT_TRUE(matches("class Z {};", XOrYOrZOrUOrV));
  EXPECT_TRUE(matches("class U {};", XOrYOrZOrUOrV));
  EXPECT_TRUE(matches("class V {};", XOrYOrZOrUOrV));
  EXPECT_TRUE(notMatches("class A {};", XOrYOrZOrUOrV));
}

TEST(DeclarationMatcher, MatchHas) {
  DeclarationMatcher HasClassX = record(has(record(hasName("X"))));

  EXPECT_TRUE(matches("class Y { class X {}; };", HasClassX));
  EXPECT_TRUE(matches("class X {};", HasClassX));

  DeclarationMatcher YHasClassX =
      record(hasName("Y"), has(record(hasName("X"))));
  EXPECT_TRUE(matches("class Y { class X {}; };", YHasClassX));
  EXPECT_TRUE(notMatches("class X {};", YHasClassX));
  EXPECT_TRUE(
      notMatches("class Y { class Z { class X {}; }; };", YHasClassX));
}

TEST(DeclarationMatcher, MatchHasRecursiveAllOf) {
  DeclarationMatcher Recursive =
    record(
      has(record(
        has(record(hasName("X"))),
        has(record(hasName("Y"))),
        hasName("Z"))),
      has(record(
        has(record(hasName("A"))),
        has(record(hasName("B"))),
        hasName("C"))),
      hasName("F"));

  EXPECT_TRUE(matches(
      "class F {"
      "  class Z {"
      "    class X {};"
      "    class Y {};"
      "  };"
      "  class C {"
      "    class A {};"
      "    class B {};"
      "  };"
      "};", Recursive));

  EXPECT_TRUE(matches(
      "class F {"
      "  class Z {"
      "    class A {};"
      "    class X {};"
      "    class Y {};"
      "  };"
      "  class C {"
      "    class X {};"
      "    class A {};"
      "    class B {};"
      "  };"
      "};", Recursive));

  EXPECT_TRUE(matches(
      "class O1 {"
      "  class O2 {"
      "    class F {"
      "      class Z {"
      "        class A {};"
      "        class X {};"
      "        class Y {};"
      "      };"
      "      class C {"
      "        class X {};"
      "        class A {};"
      "        class B {};"
      "      };"
      "    };"
      "  };"
      "};", Recursive));
}

TEST(DeclarationMatcher, MatchHasRecursiveAnyOf) {
  DeclarationMatcher Recursive =
      record(
          anyOf(
              has(record(
                  anyOf(
                      has(record(
                          hasName("X"))),
                      has(record(
                          hasName("Y"))),
                      hasName("Z")))),
              has(record(
                  anyOf(
                      hasName("C"),
                      has(record(
                          hasName("A"))),
                      has(record(
                          hasName("B")))))),
              hasName("F")));

  EXPECT_TRUE(matches("class F {};", Recursive));
  EXPECT_TRUE(matches("class Z {};", Recursive));
  EXPECT_TRUE(matches("class C {};", Recursive));
  EXPECT_TRUE(matches("class M { class N { class X {}; }; };", Recursive));
  EXPECT_TRUE(matches("class M { class N { class B {}; }; };", Recursive));
  EXPECT_TRUE(
      matches("class O1 { class O2 {"
              "  class M { class N { class B {}; }; }; "
              "}; };", Recursive));
}

TEST(DeclarationMatcher, MatchNot) {
  DeclarationMatcher NotClassX =
      record(
          isDerivedFrom("Y"),
          unless(hasName("Y")),
          unless(hasName("X")));
  EXPECT_TRUE(notMatches("", NotClassX));
  EXPECT_TRUE(notMatches("class Y {};", NotClassX));
  EXPECT_TRUE(matches("class Y {}; class Z : public Y {};", NotClassX));
  EXPECT_TRUE(notMatches("class Y {}; class X : public Y {};", NotClassX));
  EXPECT_TRUE(
      notMatches("class Y {}; class Z {}; class X : public Y {};",
                 NotClassX));

  DeclarationMatcher ClassXHasNotClassY =
      record(
          hasName("X"),
          has(record(hasName("Z"))),
          unless(
              has(record(hasName("Y")))));
  EXPECT_TRUE(matches("class X { class Z {}; };", ClassXHasNotClassY));
  EXPECT_TRUE(notMatches("class X { class Y {}; class Z {}; };",
                         ClassXHasNotClassY));
}

TEST(DeclarationMatcher, HasDescendant) {
  DeclarationMatcher ZDescendantClassX =
      record(
          hasDescendant(record(hasName("X"))),
          hasName("Z"));
  EXPECT_TRUE(matches("class Z { class X {}; };", ZDescendantClassX));
  EXPECT_TRUE(
      matches("class Z { class Y { class X {}; }; };", ZDescendantClassX));
  EXPECT_TRUE(
      matches("class Z { class A { class Y { class X {}; }; }; };",
              ZDescendantClassX));
  EXPECT_TRUE(
      matches("class Z { class A { class B { class Y { class X {}; }; }; }; };",
              ZDescendantClassX));
  EXPECT_TRUE(notMatches("class Z {};", ZDescendantClassX));

  DeclarationMatcher ZDescendantClassXHasClassY =
      record(
          hasDescendant(record(has(record(hasName("Y"))),
                              hasName("X"))),
          hasName("Z"));
  EXPECT_TRUE(matches("class Z { class X { class Y {}; }; };",
              ZDescendantClassXHasClassY));
  EXPECT_TRUE(
      matches("class Z { class A { class B { class X { class Y {}; }; }; }; };",
              ZDescendantClassXHasClassY));
  EXPECT_TRUE(notMatches(
      "class Z {"
      "  class A {"
      "    class B {"
      "      class X {"
      "        class C {"
      "          class Y {};"
      "        };"
      "      };"
      "    }; "
      "  };"
      "};", ZDescendantClassXHasClassY));

  DeclarationMatcher ZDescendantClassXDescendantClassY =
      record(
          hasDescendant(record(hasDescendant(record(hasName("Y"))),
                              hasName("X"))),
          hasName("Z"));
  EXPECT_TRUE(
      matches("class Z { class A { class X { class B { class Y {}; }; }; }; };",
              ZDescendantClassXDescendantClassY));
  EXPECT_TRUE(matches(
      "class Z {"
      "  class A {"
      "    class X {"
      "      class B {"
      "        class Y {};"
      "      };"
      "      class Y {};"
      "    };"
      "  };"
      "};", ZDescendantClassXDescendantClassY));
}

TEST(Enum, DoesNotMatchClasses) {
  EXPECT_TRUE(notMatches("class X {};", enumDecl(hasName("X"))));
}

TEST(Enum, MatchesEnums) {
  EXPECT_TRUE(matches("enum X {};", enumDecl(hasName("X"))));
}

TEST(EnumConstant, Matches) {
  DeclarationMatcher Matcher = enumConstant(hasName("A"));
  EXPECT_TRUE(matches("enum X{ A };", Matcher));
  EXPECT_TRUE(notMatches("enum X{ B };", Matcher));
  EXPECT_TRUE(notMatches("enum X {};", Matcher));
}

TEST(StatementMatcher, Has) {
  StatementMatcher HasVariableI =
      expression(
          hasType(pointsTo(record(hasName("X")))),
          has(declarationReference(to(variable(hasName("i"))))));

  EXPECT_TRUE(matches(
      "class X; X *x(int); void c() { int i; x(i); }", HasVariableI));
  EXPECT_TRUE(notMatches(
      "class X; X *x(int); void c() { int i; x(42); }", HasVariableI));
}

TEST(StatementMatcher, HasDescendant) {
  StatementMatcher HasDescendantVariableI =
      expression(
          hasType(pointsTo(record(hasName("X")))),
          hasDescendant(declarationReference(to(variable(hasName("i"))))));

  EXPECT_TRUE(matches(
      "class X; X *x(bool); bool b(int); void c() { int i; x(b(i)); }",
      HasDescendantVariableI));
  EXPECT_TRUE(notMatches(
      "class X; X *x(bool); bool b(int); void c() { int i; x(b(42)); }",
      HasDescendantVariableI));
}

TEST(TypeMatcher, MatchesClassType) {
  TypeMatcher TypeA = hasDeclaration(record(hasName("A")));

  EXPECT_TRUE(matches("class A { public: A *a; };", TypeA));
  EXPECT_TRUE(notMatches("class A {};", TypeA));

  TypeMatcher TypeDerivedFromA = hasDeclaration(record(isDerivedFrom("A")));

  EXPECT_TRUE(matches("class A {}; class B : public A { public: B *b; };",
              TypeDerivedFromA));
  EXPECT_TRUE(notMatches("class A {};", TypeA));

  TypeMatcher TypeAHasClassB = hasDeclaration(
      record(hasName("A"), has(record(hasName("B")))));

  EXPECT_TRUE(
      matches("class A { public: A *a; class B {}; };", TypeAHasClassB));
}

// Returns from Run whether 'bound_nodes' contain a Decl bound to 'Id', which
// can be dynamically casted to T.
// Optionally checks that the check succeeded a specific number of times.
template <typename T>
class VerifyIdIsBoundToDecl : public BoundNodesCallback {
public:
  // Create an object that checks that a node of type 'T' was bound to 'Id'.
  // Does not check for a certain number of matches.
  explicit VerifyIdIsBoundToDecl(const std::string& Id)
    : Id(Id), ExpectedCount(-1), Count(0) {}

  // Create an object that checks that a node of type 'T' was bound to 'Id'.
  // Checks that there were exactly 'ExpectedCount' matches.
  explicit VerifyIdIsBoundToDecl(const std::string& Id, int ExpectedCount)
    : Id(Id), ExpectedCount(ExpectedCount), Count(0) {}

  ~VerifyIdIsBoundToDecl() {
    if (ExpectedCount != -1) {
      EXPECT_EQ(ExpectedCount, Count);
    }
  }

  virtual bool run(const BoundNodes *Nodes) {
    if (Nodes->getDeclAs<T>(Id) != NULL) {
      ++Count;
      return true;
    }
    return false;
  }

private:
  const std::string Id;
  const int ExpectedCount;
  int Count;
};
template <typename T>
class VerifyIdIsBoundToStmt : public BoundNodesCallback {
public:
  explicit VerifyIdIsBoundToStmt(const std::string &Id) : Id(Id) {}
  virtual bool run(const BoundNodes *Nodes) {
    const T *Node = Nodes->getStmtAs<T>(Id);
    return Node != NULL;
  }
private:
  const std::string Id;
};

TEST(Matcher, BindMatchedNodes) {
  DeclarationMatcher ClassX = has(id("x", record(hasName("X"))));

  EXPECT_TRUE(matchAndVerifyResultTrue("class X {};",
      ClassX, new VerifyIdIsBoundToDecl<CXXRecordDecl>("x")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class X {};",
      ClassX, new VerifyIdIsBoundToDecl<CXXRecordDecl>("other-id")));

  TypeMatcher TypeAHasClassB = hasDeclaration(
      record(hasName("A"), has(id("b", record(hasName("B"))))));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { public: A *a; class B {}; };",
      TypeAHasClassB,
      new VerifyIdIsBoundToDecl<Decl>("b")));

  StatementMatcher MethodX = id("x", call(callee(method(hasName("x")))));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { void x() { x(); } };",
      MethodX,
      new VerifyIdIsBoundToStmt<CXXMemberCallExpr>("x")));
}

TEST(Matcher, BindTheSameNameInAlternatives) {
  StatementMatcher matcher = anyOf(
      binaryOperator(hasOperatorName("+"),
                     hasLHS(id("x", expression())),
                     hasRHS(integerLiteral(equals(0)))),
      binaryOperator(hasOperatorName("+"),
                     hasLHS(integerLiteral(equals(0))),
                     hasRHS(id("x", expression()))));

  EXPECT_TRUE(matchAndVerifyResultTrue(
      // The first branch of the matcher binds x to 0 but then fails.
      // The second branch binds x to f() and succeeds.
      "int f() { return 0 + f(); }",
      matcher,
      new VerifyIdIsBoundToStmt<CallExpr>("x")));
}

TEST(HasType, TakesQualTypeMatcherAndMatchesExpr) {
  TypeMatcher ClassX = hasDeclaration(record(hasName("X")));
  EXPECT_TRUE(
      matches("class X {}; void y(X &x) { x; }", expression(hasType(ClassX))));
  EXPECT_TRUE(
      notMatches("class X {}; void y(X *x) { x; }",
                 expression(hasType(ClassX))));
  EXPECT_TRUE(
      matches("class X {}; void y(X *x) { x; }",
              expression(hasType(pointsTo(ClassX)))));
}

TEST(HasType, TakesQualTypeMatcherAndMatchesValueDecl) {
  TypeMatcher ClassX = hasDeclaration(record(hasName("X")));
  EXPECT_TRUE(
      matches("class X {}; void y() { X x; }", variable(hasType(ClassX))));
  EXPECT_TRUE(
      notMatches("class X {}; void y() { X *x; }", variable(hasType(ClassX))));
  EXPECT_TRUE(
      matches("class X {}; void y() { X *x; }",
              variable(hasType(pointsTo(ClassX)))));
}

TEST(HasType, TakesDeclMatcherAndMatchesExpr) {
  DeclarationMatcher ClassX = record(hasName("X"));
  EXPECT_TRUE(
      matches("class X {}; void y(X &x) { x; }", expression(hasType(ClassX))));
  EXPECT_TRUE(
      notMatches("class X {}; void y(X *x) { x; }",
                 expression(hasType(ClassX))));
}

TEST(HasType, TakesDeclMatcherAndMatchesValueDecl) {
  DeclarationMatcher ClassX = record(hasName("X"));
  EXPECT_TRUE(
      matches("class X {}; void y() { X x; }", variable(hasType(ClassX))));
  EXPECT_TRUE(
      notMatches("class X {}; void y() { X *x; }", variable(hasType(ClassX))));
}

TEST(Matcher, Call) {
  // FIXME: Do we want to overload Call() to directly take
  // Matcher<Decl>, too?
  StatementMatcher MethodX = call(hasDeclaration(method(hasName("x"))));

  EXPECT_TRUE(matches("class Y { void x() { x(); } };", MethodX));
  EXPECT_TRUE(notMatches("class Y { void x() {} };", MethodX));

  StatementMatcher MethodOnY = call(on(hasType(record(hasName("Y")))));

  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      notMatches("class Y { public: void x(); }; void z(Y *&y) { y->x(); }",
                 MethodOnY));
  EXPECT_TRUE(
      notMatches("class Y { public: void x(); }; void z(Y y[]) { y->x(); }",
                 MethodOnY));
  EXPECT_TRUE(
      notMatches("class Y { public: void x(); }; void z() { Y *y; y->x(); }",
                 MethodOnY));

  StatementMatcher MethodOnYPointer =
      call(on(hasType(pointsTo(record(hasName("Y"))))));

  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z() { Y *y; y->x(); }",
              MethodOnYPointer));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y *&y) { y->x(); }",
              MethodOnYPointer));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y y[]) { y->x(); }",
              MethodOnYPointer));
  EXPECT_TRUE(
      notMatches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
                 MethodOnYPointer));
  EXPECT_TRUE(
      notMatches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
                 MethodOnYPointer));
}

TEST(HasType, MatchesAsString) {
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z() {Y* y; y->x(); }",
              call(on(hasType(asString("class Y *"))))));
  EXPECT_TRUE(matches("class X { void x(int x) {} };",
      method(hasParameter(0, hasType(asString("int"))))));
  EXPECT_TRUE(matches("namespace ns { struct A {}; }  struct B { ns::A a; };",
      field(hasType(asString("ns::A")))));
  EXPECT_TRUE(matches("namespace { struct A {}; }  struct B { A a; };",
      field(hasType(asString("struct <anonymous>::A")))));
}

TEST(Matcher, OverloadedOperatorCall) {
  StatementMatcher OpCall = overloadedOperatorCall();
  // Unary operator
  EXPECT_TRUE(matches("class Y { }; "
              "bool operator!(Y x) { return false; }; "
              "Y y; bool c = !y;", OpCall));
  // No match -- special operators like "new", "delete"
  // FIXME: operator new takes size_t, for which we need stddef.h, for which
  // we need to figure out include paths in the test.
  // EXPECT_TRUE(NotMatches("#include <stddef.h>\n"
  //             "class Y { }; "
  //             "void *operator new(size_t size) { return 0; } "
  //             "Y *y = new Y;", OpCall));
  EXPECT_TRUE(notMatches("class Y { }; "
              "void operator delete(void *p) { } "
              "void a() {Y *y = new Y; delete y;}", OpCall));
  // Binary operator
  EXPECT_TRUE(matches("class Y { }; "
              "bool operator&&(Y x, Y y) { return true; }; "
              "Y a; Y b; bool c = a && b;",
              OpCall));
  // No match -- normal operator, not an overloaded one.
  EXPECT_TRUE(notMatches("bool x = true, y = true; bool t = x && y;", OpCall));
  EXPECT_TRUE(notMatches("int t = 5 << 2;", OpCall));
}

TEST(Matcher, HasOperatorNameForOverloadedOperatorCall) {
  StatementMatcher OpCallAndAnd =
      overloadedOperatorCall(hasOverloadedOperatorName("&&"));
  EXPECT_TRUE(matches("class Y { }; "
              "bool operator&&(Y x, Y y) { return true; }; "
              "Y a; Y b; bool c = a && b;", OpCallAndAnd));
  StatementMatcher OpCallLessLess =
      overloadedOperatorCall(hasOverloadedOperatorName("<<"));
  EXPECT_TRUE(notMatches("class Y { }; "
              "bool operator&&(Y x, Y y) { return true; }; "
              "Y a; Y b; bool c = a && b;",
              OpCallLessLess));
}

TEST(Matcher, ThisPointerType) {
  StatementMatcher MethodOnY = call(thisPointerType(record(hasName("Y"))));

  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y *&y) { y->x(); }",
              MethodOnY));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y y[]) { y->x(); }",
              MethodOnY));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z() { Y *y; y->x(); }",
              MethodOnY));

  EXPECT_TRUE(matches(
      "class Y {"
      "  public: virtual void x();"
      "};"
      "class X : public Y {"
      "  public: virtual void x();"
      "};"
      "void z() { X *x; x->Y::x(); }", MethodOnY));
}

TEST(Matcher, VariableUsage) {
  StatementMatcher Reference =
      declarationReference(to(
          variable(hasInitializer(
              call(thisPointerType(record(hasName("Y"))))))));

  EXPECT_TRUE(matches(
      "class Y {"
      " public:"
      "  bool x() const;"
      "};"
      "void z(const Y &y) {"
      "  bool b = y.x();"
      "  if (b) {}"
      "}", Reference));

  EXPECT_TRUE(notMatches(
      "class Y {"
      " public:"
      "  bool x() const;"
      "};"
      "void z(const Y &y) {"
      "  bool b = y.x();"
      "}", Reference));
}

TEST(Matcher, CalledVariable) {
  StatementMatcher CallOnVariableY = expression(
      call(on(declarationReference(to(variable(hasName("y")))))));

  EXPECT_TRUE(matches(
      "class Y { public: void x() { Y y; y.x(); } };", CallOnVariableY));
  EXPECT_TRUE(matches(
      "class Y { public: void x() const { Y y; y.x(); } };", CallOnVariableY));
  EXPECT_TRUE(matches(
      "class Y { public: void x(); };"
      "class X : public Y { void z() { X y; y.x(); } };", CallOnVariableY));
  EXPECT_TRUE(matches(
      "class Y { public: void x(); };"
      "class X : public Y { void z() { X *y; y->x(); } };", CallOnVariableY));
  EXPECT_TRUE(notMatches(
      "class Y { public: void x(); };"
      "class X : public Y { void z() { unsigned long y; ((X*)y)->x(); } };",
      CallOnVariableY));
}

TEST(UnaryExprOrTypeTraitExpr, MatchesSizeOfAndAlignOf) {
  EXPECT_TRUE(matches("void x() { int a = sizeof(a); }",
                      unaryExprOrTypeTraitExpr()));
  EXPECT_TRUE(notMatches("void x() { int a = sizeof(a); }",
                         alignOfExpr(anything())));
  // FIXME: Uncomment once alignof is enabled.
  // EXPECT_TRUE(matches("void x() { int a = alignof(a); }",
  //                     unaryExprOrTypeTraitExpr()));
  // EXPECT_TRUE(notMatches("void x() { int a = alignof(a); }",
  //                        sizeOfExpr()));
}

TEST(UnaryExpressionOrTypeTraitExpression, MatchesCorrectType) {
  EXPECT_TRUE(matches("void x() { int a = sizeof(a); }", sizeOfExpr(
      hasArgumentOfType(asString("int")))));
  EXPECT_TRUE(notMatches("void x() { int a = sizeof(a); }", sizeOfExpr(
      hasArgumentOfType(asString("float")))));
  EXPECT_TRUE(matches(
      "struct A {}; void x() { A a; int b = sizeof(a); }",
      sizeOfExpr(hasArgumentOfType(hasDeclaration(record(hasName("A")))))));
  EXPECT_TRUE(notMatches("void x() { int a = sizeof(a); }", sizeOfExpr(
      hasArgumentOfType(hasDeclaration(record(hasName("string")))))));
}

TEST(MemberExpression, DoesNotMatchClasses) {
  EXPECT_TRUE(notMatches("class Y { void x() {} };", memberExpression()));
}

TEST(MemberExpression, MatchesMemberFunctionCall) {
  EXPECT_TRUE(matches("class Y { void x() { x(); } };", memberExpression()));
}

TEST(MemberExpression, MatchesVariable) {
  EXPECT_TRUE(
      matches("class Y { void x() { this->y; } int y; };", memberExpression()));
  EXPECT_TRUE(
      matches("class Y { void x() { y; } int y; };", memberExpression()));
  EXPECT_TRUE(
      matches("class Y { void x() { Y y; y.y; } int y; };",
              memberExpression()));
}

TEST(MemberExpression, MatchesStaticVariable) {
  EXPECT_TRUE(matches("class Y { void x() { this->y; } static int y; };",
              memberExpression()));
  EXPECT_TRUE(notMatches("class Y { void x() { y; } static int y; };",
              memberExpression()));
  EXPECT_TRUE(notMatches("class Y { void x() { Y::y; } static int y; };",
              memberExpression()));
}

TEST(IsInteger, MatchesIntegers) {
  EXPECT_TRUE(matches("int i = 0;", variable(hasType(isInteger()))));
  EXPECT_TRUE(matches("long long i = 0; void f(long long) { }; void g() {f(i);}",
                      call(hasArgument(0, declarationReference(
                          to(variable(hasType(isInteger()))))))));
}

TEST(IsInteger, ReportsNoFalsePositives) {
  EXPECT_TRUE(notMatches("int *i;", variable(hasType(isInteger()))));
  EXPECT_TRUE(notMatches("struct T {}; T t; void f(T *) { }; void g() {f(&t);}",
                      call(hasArgument(0, declarationReference(
                          to(variable(hasType(isInteger()))))))));
}

TEST(IsArrow, MatchesMemberVariablesViaArrow) {
  EXPECT_TRUE(matches("class Y { void x() { this->y; } int y; };",
              memberExpression(isArrow())));
  EXPECT_TRUE(matches("class Y { void x() { y; } int y; };",
              memberExpression(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { (*this).y; } int y; };",
              memberExpression(isArrow())));
}

TEST(IsArrow, MatchesStaticMemberVariablesViaArrow) {
  EXPECT_TRUE(matches("class Y { void x() { this->y; } static int y; };",
              memberExpression(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { y; } static int y; };",
              memberExpression(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { (*this).y; } static int y; };",
              memberExpression(isArrow())));
}

TEST(IsArrow, MatchesMemberCallsViaArrow) {
  EXPECT_TRUE(matches("class Y { void x() { this->x(); } };",
              memberExpression(isArrow())));
  EXPECT_TRUE(matches("class Y { void x() { x(); } };",
              memberExpression(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { Y y; y.x(); } };",
              memberExpression(isArrow())));
}

TEST(Callee, MatchesDeclarations) {
  StatementMatcher CallMethodX = call(callee(method(hasName("x"))));

  EXPECT_TRUE(matches("class Y { void x() { x(); } };", CallMethodX));
  EXPECT_TRUE(notMatches("class Y { void x() {} };", CallMethodX));
}

TEST(Callee, MatchesMemberExpressions) {
  EXPECT_TRUE(matches("class Y { void x() { this->x(); } };",
              call(callee(memberExpression()))));
  EXPECT_TRUE(
      notMatches("class Y { void x() { this->x(); } };", call(callee(call()))));
}

TEST(Function, MatchesFunctionDeclarations) {
  StatementMatcher CallFunctionF = call(callee(function(hasName("f"))));

  EXPECT_TRUE(matches("void f() { f(); }", CallFunctionF));
  EXPECT_TRUE(notMatches("void f() { }", CallFunctionF));

#if !defined(_MSC_VER)
  // FIXME: Make this work for MSVC.
  // Dependent contexts, but a non-dependent call.
  EXPECT_TRUE(matches("void f(); template <int N> void g() { f(); }",
                      CallFunctionF));
  EXPECT_TRUE(
      matches("void f(); template <int N> struct S { void g() { f(); } };",
              CallFunctionF));
#endif

  // Depedent calls don't match.
  EXPECT_TRUE(
      notMatches("void f(int); template <typename T> void g(T t) { f(t); }",
                 CallFunctionF));
  EXPECT_TRUE(
      notMatches("void f(int);"
                 "template <typename T> struct S { void g(T t) { f(t); } };",
                 CallFunctionF));
}

TEST(Matcher, Argument) {
  StatementMatcher CallArgumentY = expression(call(
      hasArgument(0, declarationReference(to(variable(hasName("y")))))));

  EXPECT_TRUE(matches("void x(int) { int y; x(y); }", CallArgumentY));
  EXPECT_TRUE(
      matches("class X { void x(int) { int y; x(y); } };", CallArgumentY));
  EXPECT_TRUE(notMatches("void x(int) { int z; x(z); }", CallArgumentY));

  StatementMatcher WrongIndex = expression(call(
      hasArgument(42, declarationReference(to(variable(hasName("y")))))));
  EXPECT_TRUE(notMatches("void x(int) { int y; x(y); }", WrongIndex));
}

TEST(Matcher, AnyArgument) {
  StatementMatcher CallArgumentY = expression(call(
      hasAnyArgument(declarationReference(to(variable(hasName("y")))))));
  EXPECT_TRUE(matches("void x(int, int) { int y; x(1, y); }", CallArgumentY));
  EXPECT_TRUE(matches("void x(int, int) { int y; x(y, 42); }", CallArgumentY));
  EXPECT_TRUE(notMatches("void x(int, int) { x(1, 2); }", CallArgumentY));
}

TEST(Matcher, ArgumentCount) {
  StatementMatcher Call1Arg = expression(call(argumentCountIs(1)));

  EXPECT_TRUE(matches("void x(int) { x(0); }", Call1Arg));
  EXPECT_TRUE(matches("class X { void x(int) { x(0); } };", Call1Arg));
  EXPECT_TRUE(notMatches("void x(int, int) { x(0, 0); }", Call1Arg));
}

TEST(Matcher, References) {
  DeclarationMatcher ReferenceClassX = variable(
      hasType(references(record(hasName("X")))));
  EXPECT_TRUE(matches("class X {}; void y(X y) { X &x = y; }",
                      ReferenceClassX));
  EXPECT_TRUE(
      matches("class X {}; void y(X y) { const X &x = y; }", ReferenceClassX));
  EXPECT_TRUE(
      notMatches("class X {}; void y(X y) { X x = y; }", ReferenceClassX));
  EXPECT_TRUE(
      notMatches("class X {}; void y(X *y) { X *&x = y; }", ReferenceClassX));
}

TEST(HasParameter, CallsInnerMatcher) {
  EXPECT_TRUE(matches("class X { void x(int) {} };",
      method(hasParameter(0, variable()))));
  EXPECT_TRUE(notMatches("class X { void x(int) {} };",
      method(hasParameter(0, hasName("x")))));
}

TEST(HasParameter, DoesNotMatchIfIndexOutOfBounds) {
  EXPECT_TRUE(notMatches("class X { void x(int) {} };",
      method(hasParameter(42, variable()))));
}

TEST(HasType, MatchesParameterVariableTypesStrictly) {
  EXPECT_TRUE(matches("class X { void x(X x) {} };",
      method(hasParameter(0, hasType(record(hasName("X")))))));
  EXPECT_TRUE(notMatches("class X { void x(const X &x) {} };",
      method(hasParameter(0, hasType(record(hasName("X")))))));
  EXPECT_TRUE(matches("class X { void x(const X *x) {} };",
      method(hasParameter(0, hasType(pointsTo(record(hasName("X"))))))));
  EXPECT_TRUE(matches("class X { void x(const X &x) {} };",
      method(hasParameter(0, hasType(references(record(hasName("X"))))))));
}

TEST(HasAnyParameter, MatchesIndependentlyOfPosition) {
  EXPECT_TRUE(matches("class Y {}; class X { void x(X x, Y y) {} };",
      method(hasAnyParameter(hasType(record(hasName("X")))))));
  EXPECT_TRUE(matches("class Y {}; class X { void x(Y y, X x) {} };",
      method(hasAnyParameter(hasType(record(hasName("X")))))));
}

TEST(Returns, MatchesReturnTypes) {
  EXPECT_TRUE(matches("class Y { int f() { return 1; } };",
                      function(returns(asString("int")))));
  EXPECT_TRUE(notMatches("class Y { int f() { return 1; } };",
                         function(returns(asString("float")))));
  EXPECT_TRUE(matches("class Y { Y getMe() { return *this; } };",
                      function(returns(hasDeclaration(record(hasName("Y")))))));
}

TEST(HasAnyParameter, DoesntMatchIfInnerMatcherDoesntMatch) {
  EXPECT_TRUE(notMatches("class Y {}; class X { void x(int) {} };",
      method(hasAnyParameter(hasType(record(hasName("X")))))));
}

TEST(HasAnyParameter, DoesNotMatchThisPointer) {
  EXPECT_TRUE(notMatches("class Y {}; class X { void x() {} };",
      method(hasAnyParameter(hasType(pointsTo(record(hasName("X"))))))));
}

TEST(HasName, MatchesParameterVariableDeclartions) {
  EXPECT_TRUE(matches("class Y {}; class X { void x(int x) {} };",
      method(hasAnyParameter(hasName("x")))));
  EXPECT_TRUE(notMatches("class Y {}; class X { void x(int) {} };",
      method(hasAnyParameter(hasName("x")))));
}

TEST(Matcher, ConstructorCall) {
  StatementMatcher Constructor = expression(constructorCall());

  EXPECT_TRUE(
      matches("class X { public: X(); }; void x() { X x; }", Constructor));
  EXPECT_TRUE(
      matches("class X { public: X(); }; void x() { X x = X(); }",
              Constructor));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { X x = 0; }",
              Constructor));
  EXPECT_TRUE(matches("class X {}; void x(int) { X x; }", Constructor));
}

TEST(Matcher, ConstructorArgument) {
  StatementMatcher Constructor = expression(constructorCall(
      hasArgument(0, declarationReference(to(variable(hasName("y")))))));

  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { int y; X x(y); }",
              Constructor));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { int y; X x = X(y); }",
              Constructor));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { int y; X x = y; }",
              Constructor));
  EXPECT_TRUE(
      notMatches("class X { public: X(int); }; void x() { int z; X x(z); }",
                 Constructor));

  StatementMatcher WrongIndex = expression(constructorCall(
      hasArgument(42, declarationReference(to(variable(hasName("y")))))));
  EXPECT_TRUE(
      notMatches("class X { public: X(int); }; void x() { int y; X x(y); }",
                 WrongIndex));
}

TEST(Matcher, ConstructorArgumentCount) {
  StatementMatcher Constructor1Arg =
      expression(constructorCall(argumentCountIs(1)));

  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { X x(0); }",
              Constructor1Arg));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { X x = X(0); }",
              Constructor1Arg));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { X x = 0; }",
              Constructor1Arg));
  EXPECT_TRUE(
      notMatches("class X { public: X(int, int); }; void x() { X x(0, 0); }",
                 Constructor1Arg));
}

TEST(Matcher, BindTemporaryExpression) {
  StatementMatcher TempExpression = expression(bindTemporaryExpression());

  std::string ClassString = "class string { public: string(); ~string(); }; ";

  EXPECT_TRUE(
      matches(ClassString +
              "string GetStringByValue();"
              "void FunctionTakesString(string s);"
              "void run() { FunctionTakesString(GetStringByValue()); }",
              TempExpression));

  EXPECT_TRUE(
      notMatches(ClassString +
                 "string* GetStringPointer(); "
                 "void FunctionTakesStringPtr(string* s);"
                 "void run() {"
                 "  string* s = GetStringPointer();"
                 "  FunctionTakesStringPtr(GetStringPointer());"
                 "  FunctionTakesStringPtr(s);"
                 "}",
                 TempExpression));

  EXPECT_TRUE(
      notMatches("class no_dtor {};"
                 "no_dtor GetObjByValue();"
                 "void ConsumeObj(no_dtor param);"
                 "void run() { ConsumeObj(GetObjByValue()); }",
                 TempExpression));
}

TEST(ConstructorDeclaration, SimpleCase) {
  EXPECT_TRUE(matches("class Foo { Foo(int i); };",
                      constructor(ofClass(hasName("Foo")))));
  EXPECT_TRUE(notMatches("class Foo { Foo(int i); };",
                         constructor(ofClass(hasName("Bar")))));
}

TEST(ConstructorDeclaration, IsImplicit) {
  // This one doesn't match because the constructor is not added by the
  // compiler (it is not needed).
  EXPECT_TRUE(notMatches("class Foo { };",
                         constructor(isImplicit())));
  // The compiler added the implicit default constructor.
  EXPECT_TRUE(matches("class Foo { }; Foo* f = new Foo();",
                      constructor(isImplicit())));
  EXPECT_TRUE(matches("class Foo { Foo(){} };",
                      constructor(unless(isImplicit()))));
}

TEST(DestructorDeclaration, MatchesVirtualDestructor) {
  EXPECT_TRUE(matches("class Foo { virtual ~Foo(); };",
                      destructor(ofClass(hasName("Foo")))));
}

TEST(DestructorDeclaration, DoesNotMatchImplicitDestructor) {
  EXPECT_TRUE(notMatches("class Foo {};", destructor(ofClass(hasName("Foo")))));
}

TEST(HasAnyConstructorInitializer, SimpleCase) {
  EXPECT_TRUE(notMatches(
      "class Foo { Foo() { } };",
      constructor(hasAnyConstructorInitializer(anything()))));
  EXPECT_TRUE(matches(
      "class Foo {"
      "  Foo() : foo_() { }"
      "  int foo_;"
      "};",
      constructor(hasAnyConstructorInitializer(anything()))));
}

TEST(HasAnyConstructorInitializer, ForField) {
  static const char Code[] =
      "class Baz { };"
      "class Foo {"
      "  Foo() : foo_() { }"
      "  Baz foo_;"
      "  Baz bar_;"
      "};";
  EXPECT_TRUE(matches(Code, constructor(hasAnyConstructorInitializer(
      forField(hasType(record(hasName("Baz"))))))));
  EXPECT_TRUE(matches(Code, constructor(hasAnyConstructorInitializer(
      forField(hasName("foo_"))))));
  EXPECT_TRUE(notMatches(Code, constructor(hasAnyConstructorInitializer(
      forField(hasType(record(hasName("Bar"))))))));
}

TEST(HasAnyConstructorInitializer, WithInitializer) {
  static const char Code[] =
      "class Foo {"
      "  Foo() : foo_(0) { }"
      "  int foo_;"
      "};";
  EXPECT_TRUE(matches(Code, constructor(hasAnyConstructorInitializer(
      withInitializer(integerLiteral(equals(0)))))));
  EXPECT_TRUE(notMatches(Code, constructor(hasAnyConstructorInitializer(
      withInitializer(integerLiteral(equals(1)))))));
}

TEST(HasAnyConstructorInitializer, IsWritten) {
  static const char Code[] =
      "struct Bar { Bar(){} };"
      "class Foo {"
      "  Foo() : foo_() { }"
      "  Bar foo_;"
      "  Bar bar_;"
      "};";
  EXPECT_TRUE(matches(Code, constructor(hasAnyConstructorInitializer(
      allOf(forField(hasName("foo_")), isWritten())))));
  EXPECT_TRUE(notMatches(Code, constructor(hasAnyConstructorInitializer(
      allOf(forField(hasName("bar_")), isWritten())))));
  EXPECT_TRUE(matches(Code, constructor(hasAnyConstructorInitializer(
      allOf(forField(hasName("bar_")), unless(isWritten()))))));
}

TEST(Matcher, NewExpression) {
  StatementMatcher New = expression(newExpression());

  EXPECT_TRUE(matches("class X { public: X(); }; void x() { new X; }", New));
  EXPECT_TRUE(
      matches("class X { public: X(); }; void x() { new X(); }", New));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { new X(0); }", New));
  EXPECT_TRUE(matches("class X {}; void x(int) { new X; }", New));
}

TEST(Matcher, NewExpressionArgument) {
  StatementMatcher New = expression(constructorCall(
      hasArgument(
          0, declarationReference(to(variable(hasName("y")))))));

  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { int y; new X(y); }",
              New));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { int y; new X(y); }",
              New));
  EXPECT_TRUE(
      notMatches("class X { public: X(int); }; void x() { int z; new X(z); }",
                 New));

  StatementMatcher WrongIndex = expression(constructorCall(
      hasArgument(
          42, declarationReference(to(variable(hasName("y")))))));
  EXPECT_TRUE(
      notMatches("class X { public: X(int); }; void x() { int y; new X(y); }",
                 WrongIndex));
}

TEST(Matcher, NewExpressionArgumentCount) {
  StatementMatcher New = constructorCall(argumentCountIs(1));

  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { new X(0); }", New));
  EXPECT_TRUE(
      notMatches("class X { public: X(int, int); }; void x() { new X(0, 0); }",
                 New));
}

TEST(Matcher, DeleteExpression) {
  EXPECT_TRUE(matches("struct A {}; void f(A* a) { delete a; }",
                      deleteExpression()));
}

TEST(Matcher, DefaultArgument) {
  StatementMatcher Arg = defaultArgument();

  EXPECT_TRUE(matches("void x(int, int = 0) { int y; x(y); }", Arg));
  EXPECT_TRUE(
      matches("class X { void x(int, int = 0) { int y; x(y); } };", Arg));
  EXPECT_TRUE(notMatches("void x(int, int = 0) { int y; x(y, 0); }", Arg));
}

TEST(Matcher, StringLiterals) {
  StatementMatcher Literal = expression(stringLiteral());
  EXPECT_TRUE(matches("const char *s = \"string\";", Literal));
  // wide string
  EXPECT_TRUE(matches("const wchar_t *s = L\"string\";", Literal));
  // with escaped characters
  EXPECT_TRUE(matches("const char *s = \"\x05five\";", Literal));
  // no matching -- though the data type is the same, there is no string literal
  EXPECT_TRUE(notMatches("const char s[1] = {'a'};", Literal));
}

TEST(Matcher, CharacterLiterals) {
  StatementMatcher CharLiteral = expression(characterLiteral());
  EXPECT_TRUE(matches("const char c = 'c';", CharLiteral));
  // wide character
  EXPECT_TRUE(matches("const char c = L'c';", CharLiteral));
  // wide character, Hex encoded, NOT MATCHED!
  EXPECT_TRUE(notMatches("const wchar_t c = 0x2126;", CharLiteral));
  EXPECT_TRUE(notMatches("const char c = 0x1;", CharLiteral));
}

TEST(Matcher, IntegerLiterals) {
  StatementMatcher HasIntLiteral = expression(integerLiteral());
  EXPECT_TRUE(matches("int i = 10;", HasIntLiteral));
  EXPECT_TRUE(matches("int i = 0x1AB;", HasIntLiteral));
  EXPECT_TRUE(matches("int i = 10L;", HasIntLiteral));
  EXPECT_TRUE(matches("int i = 10U;", HasIntLiteral));

  // Non-matching cases (character literals, float and double)
  EXPECT_TRUE(notMatches("int i = L'a';",
                HasIntLiteral));  // this is actually a character
                                  // literal cast to int
  EXPECT_TRUE(notMatches("int i = 'a';", HasIntLiteral));
  EXPECT_TRUE(notMatches("int i = 1e10;", HasIntLiteral));
  EXPECT_TRUE(notMatches("int i = 10.0;", HasIntLiteral));
}

TEST(Matcher, Conditions) {
  StatementMatcher Condition = ifStmt(hasCondition(boolLiteral(equals(true))));

  EXPECT_TRUE(matches("void x() { if (true) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { if (false) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { bool a = true; if (a) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { if (true || false) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { if (1) {} }", Condition));
}

TEST(MatchBinaryOperator, HasOperatorName) {
  StatementMatcher OperatorOr = binaryOperator(hasOperatorName("||"));

  EXPECT_TRUE(matches("void x() { true || false; }", OperatorOr));
  EXPECT_TRUE(notMatches("void x() { true && false; }", OperatorOr));
}

TEST(MatchBinaryOperator, HasLHSAndHasRHS) {
  StatementMatcher OperatorTrueFalse =
      binaryOperator(hasLHS(boolLiteral(equals(true))),
                     hasRHS(boolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { true || false; }", OperatorTrueFalse));
  EXPECT_TRUE(matches("void x() { true && false; }", OperatorTrueFalse));
  EXPECT_TRUE(notMatches("void x() { false || true; }", OperatorTrueFalse));
}

TEST(MatchBinaryOperator, HasEitherOperand) {
  StatementMatcher HasOperand =
      binaryOperator(hasEitherOperand(boolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { true || false; }", HasOperand));
  EXPECT_TRUE(matches("void x() { false && true; }", HasOperand));
  EXPECT_TRUE(notMatches("void x() { true || true; }", HasOperand));
}

TEST(Matcher, BinaryOperatorTypes) {
  // Integration test that verifies the AST provides all binary operators in
  // a way we expect.
  // FIXME: Operator ','
  EXPECT_TRUE(
      matches("void x() { 3, 4; }", binaryOperator(hasOperatorName(","))));
  EXPECT_TRUE(
      matches("bool b; bool c = (b = true);",
              binaryOperator(hasOperatorName("="))));
  EXPECT_TRUE(
      matches("bool b = 1 != 2;", binaryOperator(hasOperatorName("!="))));
  EXPECT_TRUE(
      matches("bool b = 1 == 2;", binaryOperator(hasOperatorName("=="))));
  EXPECT_TRUE(matches("bool b = 1 < 2;", binaryOperator(hasOperatorName("<"))));
  EXPECT_TRUE(
      matches("bool b = 1 <= 2;", binaryOperator(hasOperatorName("<="))));
  EXPECT_TRUE(
      matches("int i = 1 << 2;", binaryOperator(hasOperatorName("<<"))));
  EXPECT_TRUE(
      matches("int i = 1; int j = (i <<= 2);",
              binaryOperator(hasOperatorName("<<="))));
  EXPECT_TRUE(matches("bool b = 1 > 2;", binaryOperator(hasOperatorName(">"))));
  EXPECT_TRUE(
      matches("bool b = 1 >= 2;", binaryOperator(hasOperatorName(">="))));
  EXPECT_TRUE(
      matches("int i = 1 >> 2;", binaryOperator(hasOperatorName(">>"))));
  EXPECT_TRUE(
      matches("int i = 1; int j = (i >>= 2);",
              binaryOperator(hasOperatorName(">>="))));
  EXPECT_TRUE(
      matches("int i = 42 ^ 23;", binaryOperator(hasOperatorName("^"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i ^= 42);",
              binaryOperator(hasOperatorName("^="))));
  EXPECT_TRUE(
      matches("int i = 42 % 23;", binaryOperator(hasOperatorName("%"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i %= 42);",
              binaryOperator(hasOperatorName("%="))));
  EXPECT_TRUE(
      matches("bool b = 42  &23;", binaryOperator(hasOperatorName("&"))));
  EXPECT_TRUE(
      matches("bool b = true && false;",
              binaryOperator(hasOperatorName("&&"))));
  EXPECT_TRUE(
      matches("bool b = true; bool c = (b &= false);",
              binaryOperator(hasOperatorName("&="))));
  EXPECT_TRUE(
      matches("bool b = 42 | 23;", binaryOperator(hasOperatorName("|"))));
  EXPECT_TRUE(
      matches("bool b = true || false;",
              binaryOperator(hasOperatorName("||"))));
  EXPECT_TRUE(
      matches("bool b = true; bool c = (b |= false);",
              binaryOperator(hasOperatorName("|="))));
  EXPECT_TRUE(
      matches("int i = 42  *23;", binaryOperator(hasOperatorName("*"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i *= 23);",
              binaryOperator(hasOperatorName("*="))));
  EXPECT_TRUE(
      matches("int i = 42 / 23;", binaryOperator(hasOperatorName("/"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i /= 23);",
              binaryOperator(hasOperatorName("/="))));
  EXPECT_TRUE(
      matches("int i = 42 + 23;", binaryOperator(hasOperatorName("+"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i += 23);",
              binaryOperator(hasOperatorName("+="))));
  EXPECT_TRUE(
      matches("int i = 42 - 23;", binaryOperator(hasOperatorName("-"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i -= 23);",
              binaryOperator(hasOperatorName("-="))));
  EXPECT_TRUE(
      matches("struct A { void x() { void (A::*a)(); (this->*a)(); } };",
              binaryOperator(hasOperatorName("->*"))));
  EXPECT_TRUE(
      matches("struct A { void x() { void (A::*a)(); ((*this).*a)(); } };",
              binaryOperator(hasOperatorName(".*"))));

  // Member expressions as operators are not supported in matches.
  EXPECT_TRUE(
      notMatches("struct A { void x(A *a) { a->x(this); } };",
                 binaryOperator(hasOperatorName("->"))));

  // Initializer assignments are not represented as operator equals.
  EXPECT_TRUE(
      notMatches("bool b = true;", binaryOperator(hasOperatorName("="))));

  // Array indexing is not represented as operator.
  EXPECT_TRUE(notMatches("int a[42]; void x() { a[23]; }", unaryOperator()));

  // Overloaded operators do not match at all.
  EXPECT_TRUE(notMatches(
      "struct A { bool operator&&(const A &a) const { return false; } };"
      "void x() { A a, b; a && b; }",
      binaryOperator()));
}

TEST(MatchUnaryOperator, HasOperatorName) {
  StatementMatcher OperatorNot = unaryOperator(hasOperatorName("!"));

  EXPECT_TRUE(matches("void x() { !true; } ", OperatorNot));
  EXPECT_TRUE(notMatches("void x() { true; } ", OperatorNot));
}

TEST(MatchUnaryOperator, HasUnaryOperand) {
  StatementMatcher OperatorOnFalse =
      unaryOperator(hasUnaryOperand(boolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { !false; }", OperatorOnFalse));
  EXPECT_TRUE(notMatches("void x() { !true; }", OperatorOnFalse));
}

TEST(Matcher, UnaryOperatorTypes) {
  // Integration test that verifies the AST provides all unary operators in
  // a way we expect.
  EXPECT_TRUE(matches("bool b = !true;", unaryOperator(hasOperatorName("!"))));
  EXPECT_TRUE(
      matches("bool b; bool *p = &b;", unaryOperator(hasOperatorName("&"))));
  EXPECT_TRUE(matches("int i = ~ 1;", unaryOperator(hasOperatorName("~"))));
  EXPECT_TRUE(
      matches("bool *p; bool b = *p;", unaryOperator(hasOperatorName("*"))));
  EXPECT_TRUE(
      matches("int i; int j = +i;", unaryOperator(hasOperatorName("+"))));
  EXPECT_TRUE(
      matches("int i; int j = -i;", unaryOperator(hasOperatorName("-"))));
  EXPECT_TRUE(
      matches("int i; int j = ++i;", unaryOperator(hasOperatorName("++"))));
  EXPECT_TRUE(
      matches("int i; int j = i++;", unaryOperator(hasOperatorName("++"))));
  EXPECT_TRUE(
      matches("int i; int j = --i;", unaryOperator(hasOperatorName("--"))));
  EXPECT_TRUE(
      matches("int i; int j = i--;", unaryOperator(hasOperatorName("--"))));

  // We don't match conversion operators.
  EXPECT_TRUE(notMatches("int i; double d = (double)i;", unaryOperator()));

  // Function calls are not represented as operator.
  EXPECT_TRUE(notMatches("void f(); void x() { f(); }", unaryOperator()));

  // Overloaded operators do not match at all.
  // FIXME: We probably want to add that.
  EXPECT_TRUE(notMatches(
      "struct A { bool operator!() const { return false; } };"
      "void x() { A a; !a; }", unaryOperator(hasOperatorName("!"))));
}

TEST(Matcher, ConditionalOperator) {
  StatementMatcher Conditional = conditionalOperator(
      hasCondition(boolLiteral(equals(true))),
      hasTrueExpression(boolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { true ? false : true; }", Conditional));
  EXPECT_TRUE(notMatches("void x() { false ? false : true; }", Conditional));
  EXPECT_TRUE(notMatches("void x() { true ? true : false; }", Conditional));

  StatementMatcher ConditionalFalse = conditionalOperator(
      hasFalseExpression(boolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { true ? true : false; }", ConditionalFalse));
  EXPECT_TRUE(
      notMatches("void x() { true ? false : true; }", ConditionalFalse));
}

TEST(ArraySubscriptMatchers, ArraySubscripts) {
  EXPECT_TRUE(matches("int i[2]; void f() { i[1] = 1; }",
                      arraySubscriptExpr()));
  EXPECT_TRUE(notMatches("int i; void f() { i = 1; }",
                         arraySubscriptExpr()));
}

TEST(ArraySubscriptMatchers, ArrayIndex) {
  EXPECT_TRUE(matches(
      "int i[2]; void f() { i[1] = 1; }",
      arraySubscriptExpr(hasIndex(integerLiteral(equals(1))))));
  EXPECT_TRUE(matches(
      "int i[2]; void f() { 1[i] = 1; }",
      arraySubscriptExpr(hasIndex(integerLiteral(equals(1))))));
  EXPECT_TRUE(notMatches(
      "int i[2]; void f() { i[1] = 1; }",
      arraySubscriptExpr(hasIndex(integerLiteral(equals(0))))));
}

TEST(ArraySubscriptMatchers, MatchesArrayBase) {
  EXPECT_TRUE(matches(
      "int i[2]; void f() { i[1] = 2; }",
      arraySubscriptExpr(hasBase(implicitCast(
          hasSourceExpression(declarationReference()))))));
}

TEST(Matcher, HasNameSupportsNamespaces) {
  EXPECT_TRUE(matches("namespace a { namespace b { class C; } }",
              record(hasName("a::b::C"))));
  EXPECT_TRUE(matches("namespace a { namespace b { class C; } }",
              record(hasName("::a::b::C"))));
  EXPECT_TRUE(matches("namespace a { namespace b { class C; } }",
              record(hasName("b::C"))));
  EXPECT_TRUE(matches("namespace a { namespace b { class C; } }",
              record(hasName("C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
              record(hasName("c::b::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
              record(hasName("a::c::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
              record(hasName("a::b::A"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
              record(hasName("::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
              record(hasName("::b::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
              record(hasName("z::a::b::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
              record(hasName("a+b::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class AC; } }",
              record(hasName("C"))));
}

TEST(Matcher, HasNameSupportsOuterClasses) {
  EXPECT_TRUE(
      matches("class A { class B { class C; }; };", record(hasName("A::B::C"))));
  EXPECT_TRUE(
      matches("class A { class B { class C; }; };",
              record(hasName("::A::B::C"))));
  EXPECT_TRUE(
      matches("class A { class B { class C; }; };", record(hasName("B::C"))));
  EXPECT_TRUE(
      matches("class A { class B { class C; }; };", record(hasName("C"))));
  EXPECT_TRUE(
      notMatches("class A { class B { class C; }; };",
                 record(hasName("c::B::C"))));
  EXPECT_TRUE(
      notMatches("class A { class B { class C; }; };",
                 record(hasName("A::c::C"))));
  EXPECT_TRUE(
      notMatches("class A { class B { class C; }; };",
                 record(hasName("A::B::A"))));
  EXPECT_TRUE(
      notMatches("class A { class B { class C; }; };", record(hasName("::C"))));
  EXPECT_TRUE(
      notMatches("class A { class B { class C; }; };",
                 record(hasName("::B::C"))));
  EXPECT_TRUE(notMatches("class A { class B { class C; }; };",
              record(hasName("z::A::B::C"))));
  EXPECT_TRUE(
      notMatches("class A { class B { class C; }; };",
                 record(hasName("A+B::C"))));
}

TEST(Matcher, IsDefinition) {
  DeclarationMatcher DefinitionOfClassA =
      record(hasName("A"), isDefinition());
  EXPECT_TRUE(matches("class A {};", DefinitionOfClassA));
  EXPECT_TRUE(notMatches("class A;", DefinitionOfClassA));

  DeclarationMatcher DefinitionOfVariableA =
      variable(hasName("a"), isDefinition());
  EXPECT_TRUE(matches("int a;", DefinitionOfVariableA));
  EXPECT_TRUE(notMatches("extern int a;", DefinitionOfVariableA));

  DeclarationMatcher DefinitionOfMethodA =
      method(hasName("a"), isDefinition());
  EXPECT_TRUE(matches("class A { void a() {} };", DefinitionOfMethodA));
  EXPECT_TRUE(notMatches("class A { void a(); };", DefinitionOfMethodA));
}

TEST(Matcher, OfClass) {
  StatementMatcher Constructor = constructorCall(hasDeclaration(method(
      ofClass(hasName("X")))));

  EXPECT_TRUE(
      matches("class X { public: X(); }; void x(int) { X x; }", Constructor));
  EXPECT_TRUE(
      matches("class X { public: X(); }; void x(int) { X x = X(); }",
              Constructor));
  EXPECT_TRUE(
      notMatches("class Y { public: Y(); }; void x(int) { Y y; }",
                 Constructor));
}

TEST(Matcher, VisitsTemplateInstantiations) {
  EXPECT_TRUE(matches(
      "class A { public: void x(); };"
      "template <typename T> class B { public: void y() { T t; t.x(); } };"
      "void f() { B<A> b; b.y(); }", call(callee(method(hasName("x"))))));

  EXPECT_TRUE(matches(
      "class A { public: void x(); };"
      "class C {"
      " public:"
      "  template <typename T> class B { public: void y() { T t; t.x(); } };"
      "};"
      "void f() {"
      "  C::B<A> b; b.y();"
      "}", record(hasName("C"),
                 hasDescendant(call(callee(method(hasName("x"))))))));
}

TEST(Matcher, HandlesNullQualTypes) {
  // FIXME: Add a Type matcher so we can replace uses of this
  // variable with Type(True())
  const TypeMatcher AnyType = anything();

  // We don't really care whether this matcher succeeds; we're testing that
  // it completes without crashing.
  EXPECT_TRUE(matches(
      "struct A { };"
      "template <typename T>"
      "void f(T t) {"
      "  T local_t(t /* this becomes a null QualType in the AST */);"
      "}"
      "void g() {"
      "  f(0);"
      "}",
      expression(hasType(TypeMatcher(
          anyOf(
              TypeMatcher(hasDeclaration(anything())),
              pointsTo(AnyType),
              references(AnyType)
              // Other QualType matchers should go here.
                ))))));
}

// For testing AST_MATCHER_P().
AST_MATCHER_P(Decl, just, internal::Matcher<Decl>, AMatcher) {
  // Make sure all special variables are used: node, match_finder,
  // bound_nodes_builder, and the parameter named 'AMatcher'.
  return AMatcher.matches(Node, Finder, Builder);
}

TEST(AstMatcherPMacro, Works) {
  DeclarationMatcher HasClassB = just(has(id("b", record(hasName("B")))));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<Decl>("b")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<Decl>("a")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class C {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<Decl>("b")));
}

AST_POLYMORPHIC_MATCHER_P(
    polymorphicHas, internal::Matcher<Decl>, AMatcher) {
  TOOLING_COMPILE_ASSERT((llvm::is_same<NodeType, Decl>::value) ||
                         (llvm::is_same<NodeType, Stmt>::value),
                         assert_node_type_is_accessible);
  internal::TypedBaseMatcher<Decl> ChildMatcher(AMatcher);
  return Finder->matchesChildOf(
      Node, ChildMatcher, Builder,
      ASTMatchFinder::TK_IgnoreImplicitCastsAndParentheses,
      ASTMatchFinder::BK_First);
}

TEST(AstPolymorphicMatcherPMacro, Works) {
  DeclarationMatcher HasClassB = polymorphicHas(id("b", record(hasName("B"))));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<Decl>("b")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<Decl>("a")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class C {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<Decl>("b")));

  StatementMatcher StatementHasClassB =
      polymorphicHas(record(hasName("B")));

  EXPECT_TRUE(matches("void x() { class B {}; }", StatementHasClassB));
}

TEST(For, FindsForLoops) {
  EXPECT_TRUE(matches("void f() { for(;;); }", forStmt()));
  EXPECT_TRUE(matches("void f() { if(true) for(;;); }", forStmt()));
}

TEST(For, ForLoopInternals) {
  EXPECT_TRUE(matches("void f(){ int i; for (; i < 3 ; ); }",
                      forStmt(hasCondition(anything()))));
  EXPECT_TRUE(matches("void f() { for (int i = 0; ;); }",
                      forStmt(hasLoopInit(anything()))));
}

TEST(For, NegativeForLoopInternals) {
  EXPECT_TRUE(notMatches("void f(){ for (int i = 0; ; ++i); }",
                         forStmt(hasCondition(expression()))));
  EXPECT_TRUE(notMatches("void f() {int i; for (; i < 4; ++i) {} }",
                         forStmt(hasLoopInit(anything()))));
}

TEST(For, ReportsNoFalsePositives) {
  EXPECT_TRUE(notMatches("void f() { ; }", forStmt()));
  EXPECT_TRUE(notMatches("void f() { if(true); }", forStmt()));
}

TEST(CompoundStatement, HandlesSimpleCases) {
  EXPECT_TRUE(notMatches("void f();", compoundStatement()));
  EXPECT_TRUE(matches("void f() {}", compoundStatement()));
  EXPECT_TRUE(matches("void f() {{}}", compoundStatement()));
}

TEST(CompoundStatement, DoesNotMatchEmptyStruct) {
  // It's not a compound statement just because there's "{}" in the source
  // text. This is an AST search, not grep.
  EXPECT_TRUE(notMatches("namespace n { struct S {}; }",
              compoundStatement()));
  EXPECT_TRUE(matches("namespace n { struct S { void f() {{}} }; }",
              compoundStatement()));
}

TEST(HasBody, FindsBodyOfForWhileDoLoops) {
  EXPECT_TRUE(matches("void f() { for(;;) {} }",
              forStmt(hasBody(compoundStatement()))));
  EXPECT_TRUE(notMatches("void f() { for(;;); }",
              forStmt(hasBody(compoundStatement()))));
  EXPECT_TRUE(matches("void f() { while(true) {} }",
              whileStmt(hasBody(compoundStatement()))));
  EXPECT_TRUE(matches("void f() { do {} while(true); }",
              doStmt(hasBody(compoundStatement()))));
}

TEST(HasAnySubstatement, MatchesForTopLevelCompoundStatement) {
  // The simplest case: every compound statement is in a function
  // definition, and the function body itself must be a compound
  // statement.
  EXPECT_TRUE(matches("void f() { for (;;); }",
              compoundStatement(hasAnySubstatement(forStmt()))));
}

TEST(HasAnySubstatement, IsNotRecursive) {
  // It's really "has any immediate substatement".
  EXPECT_TRUE(notMatches("void f() { if (true) for (;;); }",
              compoundStatement(hasAnySubstatement(forStmt()))));
}

TEST(HasAnySubstatement, MatchesInNestedCompoundStatements) {
  EXPECT_TRUE(matches("void f() { if (true) { for (;;); } }",
              compoundStatement(hasAnySubstatement(forStmt()))));
}

TEST(HasAnySubstatement, FindsSubstatementBetweenOthers) {
  EXPECT_TRUE(matches("void f() { 1; 2; 3; for (;;); 4; 5; 6; }",
              compoundStatement(hasAnySubstatement(forStmt()))));
}

TEST(StatementCountIs, FindsNoStatementsInAnEmptyCompoundStatement) {
  EXPECT_TRUE(matches("void f() { }",
              compoundStatement(statementCountIs(0))));
  EXPECT_TRUE(notMatches("void f() {}",
              compoundStatement(statementCountIs(1))));
}

TEST(StatementCountIs, AppearsToMatchOnlyOneCount) {
  EXPECT_TRUE(matches("void f() { 1; }",
              compoundStatement(statementCountIs(1))));
  EXPECT_TRUE(notMatches("void f() { 1; }",
              compoundStatement(statementCountIs(0))));
  EXPECT_TRUE(notMatches("void f() { 1; }",
              compoundStatement(statementCountIs(2))));
}

TEST(StatementCountIs, WorksWithMultipleStatements) {
  EXPECT_TRUE(matches("void f() { 1; 2; 3; }",
              compoundStatement(statementCountIs(3))));
}

TEST(StatementCountIs, WorksWithNestedCompoundStatements) {
  EXPECT_TRUE(matches("void f() { { 1; } { 1; 2; 3; 4; } }",
              compoundStatement(statementCountIs(1))));
  EXPECT_TRUE(matches("void f() { { 1; } { 1; 2; 3; 4; } }",
              compoundStatement(statementCountIs(2))));
  EXPECT_TRUE(notMatches("void f() { { 1; } { 1; 2; 3; 4; } }",
              compoundStatement(statementCountIs(3))));
  EXPECT_TRUE(matches("void f() { { 1; } { 1; 2; 3; 4; } }",
              compoundStatement(statementCountIs(4))));
}

TEST(Member, WorksInSimplestCase) {
  EXPECT_TRUE(matches("struct { int first; } s; int i(s.first);",
                      memberExpression(member(hasName("first")))));
}

TEST(Member, DoesNotMatchTheBaseExpression) {
  // Don't pick out the wrong part of the member expression, this should
  // be checking the member (name) only.
  EXPECT_TRUE(notMatches("struct { int i; } first; int i(first.i);",
                         memberExpression(member(hasName("first")))));
}

TEST(Member, MatchesInMemberFunctionCall) {
  EXPECT_TRUE(matches("void f() {"
                      "  struct { void first() {}; } s;"
                      "  s.first();"
                      "};",
                      memberExpression(member(hasName("first")))));
}

TEST(HasObjectExpression, DoesNotMatchMember) {
  EXPECT_TRUE(notMatches(
      "class X {}; struct Z { X m; }; void f(Z z) { z.m; }",
      memberExpression(hasObjectExpression(hasType(record(hasName("X")))))));
}

TEST(HasObjectExpression, MatchesBaseOfVariable) {
  EXPECT_TRUE(matches(
      "struct X { int m; }; void f(X x) { x.m; }",
      memberExpression(hasObjectExpression(hasType(record(hasName("X")))))));
  EXPECT_TRUE(matches(
      "struct X { int m; }; void f(X* x) { x->m; }",
      memberExpression(hasObjectExpression(
          hasType(pointsTo(record(hasName("X"))))))));
}

TEST(HasObjectExpression,
     MatchesObjectExpressionOfImplicitlyFormedMemberExpression) {
  EXPECT_TRUE(matches(
      "class X {}; struct S { X m; void f() { this->m; } };",
      memberExpression(hasObjectExpression(
          hasType(pointsTo(record(hasName("S"))))))));
  EXPECT_TRUE(matches(
      "class X {}; struct S { X m; void f() { m; } };",
      memberExpression(hasObjectExpression(
          hasType(pointsTo(record(hasName("S"))))))));
}

TEST(Field, DoesNotMatchNonFieldMembers) {
  EXPECT_TRUE(notMatches("class X { void m(); };", field(hasName("m"))));
  EXPECT_TRUE(notMatches("class X { class m {}; };", field(hasName("m"))));
  EXPECT_TRUE(notMatches("class X { enum { m }; };", field(hasName("m"))));
  EXPECT_TRUE(notMatches("class X { enum m {}; };", field(hasName("m"))));
}

TEST(Field, MatchesField) {
  EXPECT_TRUE(matches("class X { int m; };", field(hasName("m"))));
}

TEST(IsConstQualified, MatchesConstInt) {
  EXPECT_TRUE(matches("const int i = 42;",
                      variable(hasType(isConstQualified()))));
}

TEST(IsConstQualified, MatchesConstPointer) {
  EXPECT_TRUE(matches("int i = 42; int* const p(&i);",
                      variable(hasType(isConstQualified()))));
}

TEST(IsConstQualified, MatchesThroughTypedef) {
  EXPECT_TRUE(matches("typedef const int const_int; const_int i = 42;",
                      variable(hasType(isConstQualified()))));
  EXPECT_TRUE(matches("typedef int* int_ptr; const int_ptr p(0);",
                      variable(hasType(isConstQualified()))));
}

TEST(IsConstQualified, DoesNotMatchInappropriately) {
  EXPECT_TRUE(notMatches("typedef int nonconst_int; nonconst_int i = 42;",
                         variable(hasType(isConstQualified()))));
  EXPECT_TRUE(notMatches("int const* p;",
                         variable(hasType(isConstQualified()))));
}

TEST(ReinterpretCast, MatchesSimpleCase) {
  EXPECT_TRUE(matches("char* p = reinterpret_cast<char*>(&p);",
                      expression(reinterpretCast())));
}

TEST(ReinterpretCast, DoesNotMatchOtherCasts) {
  EXPECT_TRUE(notMatches("char* p = (char*)(&p);",
                         expression(reinterpretCast())));
  EXPECT_TRUE(notMatches("char q, *p = const_cast<char*>(&q);",
                         expression(reinterpretCast())));
  EXPECT_TRUE(notMatches("void* p = static_cast<void*>(&p);",
                         expression(reinterpretCast())));
  EXPECT_TRUE(notMatches("struct B { virtual ~B() {} }; struct D : B {};"
                         "B b;"
                         "D* p = dynamic_cast<D*>(&b);",
                         expression(reinterpretCast())));
}

TEST(FunctionalCast, MatchesSimpleCase) {
  std::string foo_class = "class Foo { public: Foo(char*); };";
  EXPECT_TRUE(matches(foo_class + "void r() { Foo f = Foo(\"hello world\"); }",
                      expression(functionalCast())));
}

TEST(FunctionalCast, DoesNotMatchOtherCasts) {
  std::string FooClass = "class Foo { public: Foo(char*); };";
  EXPECT_TRUE(
      notMatches(FooClass + "void r() { Foo f = (Foo) \"hello world\"; }",
                 expression(functionalCast())));
  EXPECT_TRUE(
      notMatches(FooClass + "void r() { Foo f = \"hello world\"; }",
                 expression(functionalCast())));
}

TEST(DynamicCast, MatchesSimpleCase) {
  EXPECT_TRUE(matches("struct B { virtual ~B() {} }; struct D : B {};"
                      "B b;"
                      "D* p = dynamic_cast<D*>(&b);",
                      expression(dynamicCast())));
}

TEST(StaticCast, MatchesSimpleCase) {
  EXPECT_TRUE(matches("void* p(static_cast<void*>(&p));",
                      expression(staticCast())));
}

TEST(StaticCast, DoesNotMatchOtherCasts) {
  EXPECT_TRUE(notMatches("char* p = (char*)(&p);",
                         expression(staticCast())));
  EXPECT_TRUE(notMatches("char q, *p = const_cast<char*>(&q);",
                         expression(staticCast())));
  EXPECT_TRUE(notMatches("void* p = reinterpret_cast<char*>(&p);",
                         expression(staticCast())));
  EXPECT_TRUE(notMatches("struct B { virtual ~B() {} }; struct D : B {};"
                         "B b;"
                         "D* p = dynamic_cast<D*>(&b);",
                         expression(staticCast())));
}

TEST(HasDestinationType, MatchesSimpleCase) {
  EXPECT_TRUE(matches("char* p = static_cast<char*>(0);",
                      expression(
                          staticCast(hasDestinationType(
                              pointsTo(TypeMatcher(anything())))))));
}

TEST(HasSourceExpression, MatchesSimpleCase) {
  EXPECT_TRUE(matches("class string {}; class URL { public: URL(string s); };"
                      "void r() {string a_string; URL url = a_string; }",
                      expression(implicitCast(
                          hasSourceExpression(constructorCall())))));
}

TEST(Statement, DoesNotMatchDeclarations) {
  EXPECT_TRUE(notMatches("class X {};", statement()));
}

TEST(Statement, MatchesCompoundStatments) {
  EXPECT_TRUE(matches("void x() {}", statement()));
}

TEST(DeclarationStatement, DoesNotMatchCompoundStatements) {
  EXPECT_TRUE(notMatches("void x() {}", declarationStatement()));
}

TEST(DeclarationStatement, MatchesVariableDeclarationStatements) {
  EXPECT_TRUE(matches("void x() { int a; }", declarationStatement()));
}

TEST(InitListExpression, MatchesInitListExpression) {
  EXPECT_TRUE(matches("int a[] = { 1, 2 };",
                      initListExpr(hasType(asString("int [2]")))));
  EXPECT_TRUE(matches("struct B { int x, y; }; B b = { 5, 6 };",
                      initListExpr(hasType(record(hasName("B"))))));
}

TEST(UsingDeclaration, MatchesUsingDeclarations) {
  EXPECT_TRUE(matches("namespace X { int x; } using X::x;",
                      usingDecl()));
}

TEST(UsingDeclaration, MatchesShadowUsingDelcarations) {
  EXPECT_TRUE(matches("namespace f { int a; } using f::a;",
                      usingDecl(hasAnyUsingShadowDecl(hasName("a")))));
}

TEST(UsingDeclaration, MatchesSpecificTarget) {
  EXPECT_TRUE(matches("namespace f { int a; void b(); } using f::b;",
                      usingDecl(hasAnyUsingShadowDecl(
                          hasTargetDecl(function())))));
  EXPECT_TRUE(notMatches("namespace f { int a; void b(); } using f::a;",
                         usingDecl(hasAnyUsingShadowDecl(
                             hasTargetDecl(function())))));
}

TEST(UsingDeclaration, ThroughUsingDeclaration) {
  EXPECT_TRUE(matches(
      "namespace a { void f(); } using a::f; void g() { f(); }",
      declarationReference(throughUsingDecl(anything()))));
  EXPECT_TRUE(notMatches(
      "namespace a { void f(); } using a::f; void g() { a::f(); }",
      declarationReference(throughUsingDecl(anything()))));
}

TEST(While, MatchesWhileLoops) {
  EXPECT_TRUE(notMatches("void x() {}", whileStmt()));
  EXPECT_TRUE(matches("void x() { while(true); }", whileStmt()));
  EXPECT_TRUE(notMatches("void x() { do {} while(true); }", whileStmt()));
}

TEST(Do, MatchesDoLoops) {
  EXPECT_TRUE(matches("void x() { do {} while(true); }", doStmt()));
  EXPECT_TRUE(matches("void x() { do ; while(false); }", doStmt()));
}

TEST(Do, DoesNotMatchWhileLoops) {
  EXPECT_TRUE(notMatches("void x() { while(true) {} }", doStmt()));
}

TEST(SwitchCase, MatchesCase) {
  EXPECT_TRUE(matches("void x() { switch(42) { case 42:; } }", switchCase()));
  EXPECT_TRUE(matches("void x() { switch(42) { default:; } }", switchCase()));
  EXPECT_TRUE(matches("void x() { switch(42) default:; }", switchCase()));
  EXPECT_TRUE(notMatches("void x() { switch(42) {} }", switchCase()));
}

TEST(HasConditionVariableStatement, DoesNotMatchCondition) {
  EXPECT_TRUE(notMatches(
      "void x() { if(true) {} }",
      ifStmt(hasConditionVariableStatement(declarationStatement()))));
  EXPECT_TRUE(notMatches(
      "void x() { int x; if((x = 42)) {} }",
      ifStmt(hasConditionVariableStatement(declarationStatement()))));
}

TEST(HasConditionVariableStatement, MatchesConditionVariables) {
  EXPECT_TRUE(matches(
      "void x() { if(int* a = 0) {} }",
      ifStmt(hasConditionVariableStatement(declarationStatement()))));
}

TEST(ForEach, BindsOneNode) {
  EXPECT_TRUE(matchAndVerifyResultTrue("class C { int x; };",
      record(hasName("C"), forEach(id("x", field(hasName("x"))))),
      new VerifyIdIsBoundToDecl<FieldDecl>("x", 1)));
}

TEST(ForEach, BindsMultipleNodes) {
  EXPECT_TRUE(matchAndVerifyResultTrue("class C { int x; int y; int z; };",
      record(hasName("C"), forEach(id("f", field()))),
      new VerifyIdIsBoundToDecl<FieldDecl>("f", 3)));
}

TEST(ForEach, BindsRecursiveCombinations) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class C { class D { int x; int y; }; class E { int y; int z; }; };",
      record(hasName("C"), forEach(record(forEach(id("f", field()))))),
      new VerifyIdIsBoundToDecl<FieldDecl>("f", 4)));
}

TEST(ForEachDescendant, BindsOneNode) {
  EXPECT_TRUE(matchAndVerifyResultTrue("class C { class D { int x; }; };",
      record(hasName("C"), forEachDescendant(id("x", field(hasName("x"))))),
      new VerifyIdIsBoundToDecl<FieldDecl>("x", 1)));
}

TEST(ForEachDescendant, BindsMultipleNodes) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class C { class D { int x; int y; }; "
      "          class E { class F { int y; int z; }; }; };",
      record(hasName("C"), forEachDescendant(id("f", field()))),
      new VerifyIdIsBoundToDecl<FieldDecl>("f", 4)));
}

TEST(ForEachDescendant, BindsRecursiveCombinations) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class C { class D { "
      "          class E { class F { class G { int y; int z; }; }; }; }; };",
      record(hasName("C"), forEachDescendant(record(
          forEachDescendant(id("f", field()))))),
      new VerifyIdIsBoundToDecl<FieldDecl>("f", 8)));
}


TEST(IsTemplateInstantiation, MatchesImplicitClassTemplateInstantiation) {
  // Make sure that we can both match the class by name (::X) and by the type
  // the template was instantiated with (via a field).

  EXPECT_TRUE(matches(
      "template <typename T> class X {}; class A {}; X<A> x;",
      record(hasName("::X"), isTemplateInstantiation())));

  EXPECT_TRUE(matches(
      "template <typename T> class X { T t; }; class A {}; X<A> x;",
      record(isTemplateInstantiation(), hasDescendant(
          field(hasType(record(hasName("A"))))))));
}

TEST(IsTemplateInstantiation, MatchesImplicitFunctionTemplateInstantiation) {
  EXPECT_TRUE(matches(
      "template <typename T> void f(T t) {} class A {}; void g() { f(A()); }",
      function(hasParameter(0, hasType(record(hasName("A")))),
               isTemplateInstantiation())));
}

TEST(IsTemplateInstantiation, MatchesExplicitClassTemplateInstantiation) {
  EXPECT_TRUE(matches(
      "template <typename T> class X { T t; }; class A {};"
      "template class X<A>;",
      record(isTemplateInstantiation(), hasDescendant(
          field(hasType(record(hasName("A"))))))));
}

TEST(IsTemplateInstantiation,
     MatchesInstantiationOfPartiallySpecializedClassTemplate) {
  EXPECT_TRUE(matches(
      "template <typename T> class X {};"
      "template <typename T> class X<T*> {}; class A {}; X<A*> x;",
      record(hasName("::X"), isTemplateInstantiation())));
}

TEST(IsTemplateInstantiation,
     MatchesInstantiationOfClassTemplateNestedInNonTemplate) {
  EXPECT_TRUE(matches(
      "class A {};"
      "class X {"
      "  template <typename U> class Y { U u; };"
      "  Y<A> y;"
      "};",
      record(hasName("::X::Y"), isTemplateInstantiation())));
}

TEST(IsTemplateInstantiation, DoesNotMatchInstantiationsInsideOfInstantiation) {
  // FIXME: Figure out whether this makes sense. It doesn't affect the
  // normal use case as long as the uppermost instantiation always is marked
  // as template instantiation, but it might be confusing as a predicate.
  EXPECT_TRUE(matches(
      "class A {};"
      "template <typename T> class X {"
      "  template <typename U> class Y { U u; };"
      "  Y<T> y;"
      "}; X<A> x;",
      record(hasName("::X<A>::Y"), unless(isTemplateInstantiation()))));
}

TEST(IsTemplateInstantiation, DoesNotMatchExplicitClassTemplateSpecialization) {
  EXPECT_TRUE(notMatches(
      "template <typename T> class X {}; class A {};"
      "template <> class X<A> {}; X<A> x;",
      record(hasName("::X"), isTemplateInstantiation())));
}

TEST(IsTemplateInstantiation, DoesNotMatchNonTemplate) {
  EXPECT_TRUE(notMatches(
      "class A {}; class Y { A a; };",
      record(isTemplateInstantiation())));
}

} // end namespace ast_matchers
} // end namespace clang
