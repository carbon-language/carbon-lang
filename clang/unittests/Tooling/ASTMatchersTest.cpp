//===- unittest/Tooling/ASTMatchersTest.cpp - AST matcher unit tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ASTMatchers.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {

using match::AnyOf;
using match::AllOf;
using match::ArgumentCountIs;
using match::BinaryOperator;
using match::BoolLiteral;
using match::StringLiteral;
using match::IntegerLiteral;
using match::CharacterLiteral;
using match::Call;
using match::Callee;
using match::Class;
using match::CompoundStatement;
using match::ConstructorCall;
using match::DeclarationMatcher;
using match::DefaultArgument;
using match::ConditionalOperator;
using match::DeclarationReference;
using match::Expression;
using match::Equals;
using match::For;
using match::Has;
using match::HasAnyArgument;
using match::HasAnyParameter;
using match::HasAnySubstatement;
using match::HasArgument;
using match::HasBody;
using match::HasCondition;
using match::HasDeclaration;
using match::HasDescendant;
using match::HasEitherOperand;
using match::HasFalseExpression;
using match::HasInitializer;
using match::HasLHS;
using match::HasName;
using match::HasUnaryOperand;
using match::HasOperatorName;
using match::HasOverloadedOperatorName;
using match::HasParameter;
using match::HasRHS;
using match::HasTrueExpression;
using match::HasType;
using match::Id;
using match::If;
using match::IsArrow;
using match::IsDefinition;
using match::IsDerivedFrom;
using match::MemberExpression;
using match::Method;
using match::Not;
using match::OfClass;
using match::On;
using match::OverloadedOperatorCall;
using match::PointsTo;
using match::References;
using match::StatementCountIs;
using match::StatementMatcher;
using match::ThisPointerType;
using match::To;
using match::TypeMatcher;
using match::UnaryOperator;
using match::Variable;

class BoundNodesCallback {
 public:
  virtual ~BoundNodesCallback() {}
  virtual bool Run(const BoundNodes *BoundNodes) = 0;
};

// If 'FindResultVerifier' is not NULL, sets *Verified to the result of
// running 'FindResultVerifier' with the bound nodes as argument.
// If 'FindResultVerifier' is NULL, sets *Verified to true when Run is called.
class VerifyMatch : public MatchFinder::MatchCallback {
 public:
  VerifyMatch(BoundNodesCallback *FindResultVerifier, bool *Verified)
      : Verified(Verified), FindResultReviewer(FindResultVerifier) {}

  virtual void Run(const MatchFinder::MatchResult &Result) {
    if (FindResultReviewer != NULL) {
      *Verified = FindResultReviewer->Run(&Result.Nodes);
    } else {
      *Verified = true;
    }
  }

 private:
  bool *const Verified;
  BoundNodesCallback *const FindResultReviewer;
};

template <typename T>
testing::AssertionResult MatchesConditionally(
    const std::string &Code, const T &AMatcher, bool ExpectMatch) {
  bool Found = false;
  MatchFinder Finder;
  Finder.AddMatcher(AMatcher, new VerifyMatch(NULL, &Found));
  if (!Finder.FindAll(Code)) {
    return testing::AssertionFailure() << "Parsing error in \"" << Code << "\"";
  }
  if (!Found && ExpectMatch) {
    return testing::AssertionFailure()
        << "Could not find match in \"" << Code << "\"";
  } else if (Found && !ExpectMatch) {
    return testing::AssertionFailure()
        << "Found unexpected match in \"" << Code << "\"";
  }
  return testing::AssertionSuccess();
}

template <typename T>
testing::AssertionResult Matches(const std::string &Code, const T &AMatcher) {
  return MatchesConditionally(Code, AMatcher, true);
}

template <typename T>
testing::AssertionResult NotMatches(
    const std::string &Code, const T &AMatcher) {
  return MatchesConditionally(Code, AMatcher, false);
}

template <typename T>
testing::AssertionResult MatchAndVerifyResultConditionally(
    const std::string &Code, const T &AMatcher,
    BoundNodesCallback *FindResultVerifier, bool ExpectResult) {
  llvm::OwningPtr<BoundNodesCallback> ScopedVerifier(FindResultVerifier);
  bool VerifiedResult = false;
  MatchFinder Finder;
  Finder.AddMatcher(
      AMatcher, new VerifyMatch(FindResultVerifier, &VerifiedResult));
  if (!Finder.FindAll(Code)) {
    return testing::AssertionFailure() << "Parsing error in \"" << Code << "\"";
  }
  if (!VerifiedResult && ExpectResult) {
    return testing::AssertionFailure()
        << "Could not verify result in \"" << Code << "\"";
  } else if (VerifiedResult && !ExpectResult) {
    return testing::AssertionFailure()
        << "Verified unexpected result in \"" << Code << "\"";
  }
  return testing::AssertionSuccess();
}

// FIXME: Find better names for these functions (or document what they
// do more precisely).
template <typename T>
testing::AssertionResult MatchAndVerifyResultTrue(
    const std::string &Code, const T &AMatcher,
    BoundNodesCallback *FindResultVerifier) {
  return MatchAndVerifyResultConditionally(
      Code, AMatcher, FindResultVerifier, true);
}

template <typename T>
testing::AssertionResult MatchAndVerifyResultFalse(
    const std::string &Code, const T &AMatcher,
    BoundNodesCallback *FindResultVerifier) {
  return MatchAndVerifyResultConditionally(
      Code, AMatcher, FindResultVerifier, false);
}

TEST(HasNameDeathTest, DiesOnEmptyName) {
  ASSERT_DEBUG_DEATH({
      DeclarationMatcher HasEmptyName = Class(HasName(""));
      EXPECT_TRUE(NotMatches("class X {};", HasEmptyName));
    }, "");
}

TEST(IsDerivedFromDeathTest, DiesOnEmptyBaseName) {
  ASSERT_DEBUG_DEATH({
      DeclarationMatcher IsDerivedFromEmpty = Class(IsDerivedFrom(""));
      EXPECT_TRUE(NotMatches("class X {};", IsDerivedFromEmpty));
    }, "");
}

TEST(DeclarationMatcher, MatchClass) {
  DeclarationMatcher ClassMatcher(Class());
  // Even for an empty string there are classes in the AST.
  EXPECT_TRUE(Matches("", ClassMatcher));

  DeclarationMatcher ClassX = Class(Class(HasName("X")));
  EXPECT_TRUE(Matches("class X;", ClassX));
  EXPECT_TRUE(Matches("class X {};", ClassX));
  EXPECT_TRUE(Matches("template<class T> class X {};", ClassX));
  EXPECT_TRUE(NotMatches("", ClassX));
}

TEST(DeclarationMatcher, ClassIsDerived) {
  DeclarationMatcher IsDerivedFromX = Class(IsDerivedFrom("X"));

  EXPECT_TRUE(Matches("class X {}; class Y : public X {};", IsDerivedFromX));
  EXPECT_TRUE(Matches("class X {}; class Y : public X {};", IsDerivedFromX));
  EXPECT_TRUE(Matches("class X {};", IsDerivedFromX));
  EXPECT_TRUE(Matches("class X;", IsDerivedFromX));
  EXPECT_TRUE(NotMatches("class Y;", IsDerivedFromX));
  EXPECT_TRUE(NotMatches("", IsDerivedFromX));

  DeclarationMatcher ZIsDerivedFromX =
      Class(HasName("Z"), IsDerivedFrom("X"));
  EXPECT_TRUE(
      Matches("class X {}; class Y : public X {}; class Z : public Y {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("class X {};"
              "template<class T> class Y : public X {};"
              "class Z : public Y<int> {};", ZIsDerivedFromX));
  EXPECT_TRUE(Matches("class X {}; template<class T> class Z : public X {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("template<class T> class X {}; "
              "template<class T> class Z : public X<T> {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("template<class T, class U=T> class X {}; "
              "template<class T> class Z : public X<T> {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      NotMatches("template<class X> class A { class Z : public X {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("template<class X> class A { public: class Z : public X {}; }; "
              "class X{}; void y() { A<X>::Z z; }", ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("template <class T> class X {}; "
              "template<class Y> class A { class Z : public X<Y> {}; };",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      NotMatches("template<template<class T> class X> class A { "
                 "  class Z : public X<int> {}; };", ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("template<template<class T> class X> class A { "
              "  public: class Z : public X<int> {}; }; "
              "template<class T> class X {}; void y() { A<X>::Z z; }",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      NotMatches("template<class X> class A { class Z : public X::D {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("template<class X> class A { public: "
              "  class Z : public X::D {}; }; "
              "class Y { public: class X {}; typedef X D; }; "
              "void y() { A<Y>::Z z; }", ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("class X {}; typedef X Y; class Z : public Y {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("template<class T> class Y { typedef typename T::U X; "
              "  class Z : public X {}; };", ZIsDerivedFromX));
  EXPECT_TRUE(Matches("class X {}; class Z : public ::X {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(
      NotMatches("template<class T> class X {}; "
                "template<class T> class A { class Z : public X<T>::D {}; };",
                ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("template<class T> class X { public: typedef X<T> D; }; "
              "template<class T> class A { public: "
              "  class Z : public X<T>::D {}; }; void y() { A<int>::Z z; }",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      NotMatches("template<class X> class A { class Z : public X::D::E {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("class X {}; typedef X V; typedef V W; class Z : public W {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("class X {}; class Y : public X {}; "
              "typedef Y V; typedef V W; class Z : public W {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("template<class T, class U> class X {}; "
              "template<class T> class A { class Z : public X<T, int> {}; };",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      NotMatches("template<class X> class D { typedef X A; typedef A B; "
                 "  typedef B C; class Z : public C {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("class X {}; typedef X A; typedef A B; "
              "class Z : public B {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("class X {}; typedef X A; typedef A B; typedef B C; "
              "class Z : public C {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("class U {}; typedef U X; typedef X V; "
              "class Z : public V {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("class Base {}; typedef Base X; "
              "class Z : public Base {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("class Base {}; typedef Base Base2; typedef Base2 X; "
              "class Z : public Base {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      NotMatches("class Base {}; class Base2 {}; typedef Base2 X; "
                 "class Z : public Base {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      Matches("class A {}; typedef A X; typedef A Y; "
              "class Z : public Y {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      NotMatches("template <typename T> class Z;"
                 "template <> class Z<void> {};"
                 "template <typename T> class Z : public Z<void> {};",
                 IsDerivedFromX));
  EXPECT_TRUE(
      Matches("template <typename T> class X;"
              "template <> class X<void> {};"
              "template <typename T> class X : public X<void> {};",
              IsDerivedFromX));
  EXPECT_TRUE(Matches(
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
  EXPECT_TRUE(Matches(
      RecursiveTemplateOneParameter,
      Variable(HasName("z_float"),
               HasInitializer(HasType(Class(IsDerivedFrom("Base1")))))));
  EXPECT_TRUE(NotMatches(
      RecursiveTemplateOneParameter,
      Variable(
          HasName("z_float"),
          HasInitializer(HasType(Class(IsDerivedFrom("Base2")))))));
  EXPECT_TRUE(Matches(
      RecursiveTemplateOneParameter,
      Variable(
          HasName("z_char"),
          HasInitializer(HasType(Class(IsDerivedFrom("Base1"),
                                       IsDerivedFrom("Base2")))))));

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
  EXPECT_TRUE(Matches(
      RecursiveTemplateTwoParameters,
      Variable(
          HasName("z_float"),
          HasInitializer(HasType(Class(IsDerivedFrom("Base1")))))));
  EXPECT_TRUE(NotMatches(
      RecursiveTemplateTwoParameters,
      Variable(
          HasName("z_float"),
          HasInitializer(HasType(Class(IsDerivedFrom("Base2")))))));
  EXPECT_TRUE(Matches(
      RecursiveTemplateTwoParameters,
      Variable(
          HasName("z_char"),
          HasInitializer(HasType(Class(IsDerivedFrom("Base1"),
                                       IsDerivedFrom("Base2")))))));
}

TEST(DeclarationMatcher, MatchAnyOf) {
  DeclarationMatcher YOrZDerivedFromX =
      Class(AnyOf(HasName("Y"), AllOf(IsDerivedFrom("X"), HasName("Z"))));

  EXPECT_TRUE(
      Matches("class X {}; class Z : public X {};", YOrZDerivedFromX));
  EXPECT_TRUE(Matches("class Y {};", YOrZDerivedFromX));
  EXPECT_TRUE(
      NotMatches("class X {}; class W : public X {};", YOrZDerivedFromX));
  EXPECT_TRUE(NotMatches("class Z {};", YOrZDerivedFromX));

  DeclarationMatcher XOrYOrZOrU =
      Class(AnyOf(HasName("X"), HasName("Y"), HasName("Z"), HasName("U")));

  EXPECT_TRUE(Matches("class X {};", XOrYOrZOrU));
  EXPECT_TRUE(Matches("class Y {};", XOrYOrZOrU));
  EXPECT_TRUE(Matches("class Z {};", XOrYOrZOrU));
  EXPECT_TRUE(Matches("class U {};", XOrYOrZOrU));
  EXPECT_TRUE(NotMatches("class A {};", XOrYOrZOrU));
}

TEST(DeclarationMatcher, MatchHas) {
  DeclarationMatcher HasClassX = Class(Has(Class(HasName("X"))));

  EXPECT_TRUE(Matches("class Y { class X {}; };", HasClassX));
  EXPECT_TRUE(Matches("class X {};", HasClassX));

  DeclarationMatcher YHasClassX =
      Class(HasName("Y"), Has(Class(HasName("X"))));
  EXPECT_TRUE(Matches("class Y { class X {}; };", YHasClassX));
  EXPECT_TRUE(NotMatches("class X {};", YHasClassX));
  EXPECT_TRUE(
      NotMatches("class Y { class Z { class X {}; }; };", YHasClassX));
}

TEST(DeclarationMatcher, MatchHasRecursiveAllOf) {
  DeclarationMatcher Recursive =
      Class(
          Has(Class(
              Has(Class(
                  HasName("X"))),
              Has(Class(
                  HasName("Y"))),
              HasName("Z"))),
          Has(Class(
              Has(Class(
                  HasName("A"))),
              Has(Class(
                  HasName("B"))),
              HasName("C"))),
          HasName("F"));

  EXPECT_TRUE(Matches(
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

  EXPECT_TRUE(Matches(
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

  EXPECT_TRUE(Matches(
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
      Class(
          AnyOf(
              Has(Class(
                  AnyOf(
                      Has(Class(
                          HasName("X"))),
                      Has(Class(
                          HasName("Y"))),
                      HasName("Z")))),
              Has(Class(
                  AnyOf(
                      HasName("C"),
                      Has(Class(
                          HasName("A"))),
                      Has(Class(
                          HasName("B")))))),
              HasName("F")));

  EXPECT_TRUE(Matches("class F {};", Recursive));
  EXPECT_TRUE(Matches("class Z {};", Recursive));
  EXPECT_TRUE(Matches("class C {};", Recursive));
  EXPECT_TRUE(Matches("class M { class N { class X {}; }; };", Recursive));
  EXPECT_TRUE(Matches("class M { class N { class B {}; }; };", Recursive));
  EXPECT_TRUE(
      Matches("class O1 { class O2 {"
              "  class M { class N { class B {}; }; }; "
              "}; };", Recursive));
}

TEST(DeclarationMatcher, MatchNot) {
  DeclarationMatcher NotClassX =
      Class(
          IsDerivedFrom("Y"),
          Not(HasName("Y")),
          Not(HasName("X")));
  EXPECT_TRUE(NotMatches("", NotClassX));
  EXPECT_TRUE(NotMatches("class Y {};", NotClassX));
  EXPECT_TRUE(Matches("class Y {}; class Z : public Y {};", NotClassX));
  EXPECT_TRUE(NotMatches("class Y {}; class X : public Y {};", NotClassX));
  EXPECT_TRUE(
      NotMatches("class Y {}; class Z {}; class X : public Y {};",
                 NotClassX));

  DeclarationMatcher ClassXHasNotClassY =
      Class(
          HasName("X"),
          Has(Class(HasName("Z"))),
          Not(
              Has(Class(HasName("Y")))));
  EXPECT_TRUE(Matches("class X { class Z {}; };", ClassXHasNotClassY));
  EXPECT_TRUE(NotMatches("class X { class Y {}; class Z {}; };",
                         ClassXHasNotClassY));
}

TEST(DeclarationMatcher, HasDescendant) {
  DeclarationMatcher ZDescendantClassX =
      Class(
          HasDescendant(Class(HasName("X"))),
          HasName("Z"));
  EXPECT_TRUE(Matches("class Z { class X {}; };", ZDescendantClassX));
  EXPECT_TRUE(
      Matches("class Z { class Y { class X {}; }; };", ZDescendantClassX));
  EXPECT_TRUE(
      Matches("class Z { class A { class Y { class X {}; }; }; };",
              ZDescendantClassX));
  EXPECT_TRUE(
      Matches("class Z { class A { class B { class Y { class X {}; }; }; }; };",
              ZDescendantClassX));
  EXPECT_TRUE(NotMatches("class Z {};", ZDescendantClassX));

  DeclarationMatcher ZDescendantClassXHasClassY =
      Class(
          HasDescendant(Class(Has(Class(HasName("Y"))),
                              HasName("X"))),
          HasName("Z"));
  EXPECT_TRUE(Matches("class Z { class X { class Y {}; }; };",
              ZDescendantClassXHasClassY));
  EXPECT_TRUE(
      Matches("class Z { class A { class B { class X { class Y {}; }; }; }; };",
              ZDescendantClassXHasClassY));
  EXPECT_TRUE(NotMatches(
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
      Class(
          HasDescendant(Class(HasDescendant(Class(HasName("Y"))),
                              HasName("X"))),
          HasName("Z"));
  EXPECT_TRUE(
      Matches("class Z { class A { class X { class B { class Y {}; }; }; }; };",
              ZDescendantClassXDescendantClassY));
  EXPECT_TRUE(Matches(
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

TEST(StatementMatcher, Has) {
  StatementMatcher HasVariableI =
      Expression(
          HasType(PointsTo(Class(HasName("X")))),
          Has(DeclarationReference(To(Variable(HasName("i"))))));

  EXPECT_TRUE(Matches(
      "class X; X *x(int); void c() { int i; x(i); }", HasVariableI));
  EXPECT_TRUE(NotMatches(
      "class X; X *x(int); void c() { int i; x(42); }", HasVariableI));
}

TEST(StatementMatcher, HasDescendant) {
  StatementMatcher HasDescendantVariableI =
      Expression(
          HasType(PointsTo(Class(HasName("X")))),
          HasDescendant(DeclarationReference(To(Variable(HasName("i"))))));

  EXPECT_TRUE(Matches(
      "class X; X *x(bool); bool b(int); void c() { int i; x(b(i)); }",
      HasDescendantVariableI));
  EXPECT_TRUE(NotMatches(
      "class X; X *x(bool); bool b(int); void c() { int i; x(b(42)); }",
      HasDescendantVariableI));
}

TEST(TypeMatcher, MatchesClassType) {
  TypeMatcher TypeA = HasDeclaration(Class(HasName("A")));

  EXPECT_TRUE(Matches("class A { public: A *a; };", TypeA));
  EXPECT_TRUE(NotMatches("class A {};", TypeA));

  TypeMatcher TypeDerivedFromA = HasDeclaration(Class(IsDerivedFrom("A")));

  EXPECT_TRUE(Matches("class A {}; class B : public A { public: B *b; };",
              TypeDerivedFromA));
  EXPECT_TRUE(NotMatches("class A {};", TypeA));

  TypeMatcher TypeAHasClassB = HasDeclaration(
      Class(HasName("A"), Has(Class(HasName("B")))));

  EXPECT_TRUE(
      Matches("class A { public: A *a; class B {}; };", TypeAHasClassB));
}

// Returns whether 'bound_nodes' contain a Decl for bound to 'id', which
// can be dynamically casted to T.
template <typename T>
class VerifyIdIsBoundToDecl : public BoundNodesCallback {
 public:
  explicit VerifyIdIsBoundToDecl(const std::string &Id) : Id(Id) {}
  virtual bool Run(const BoundNodes *Nodes) {
    const T *Node = Nodes->GetDeclAs<T>(Id);
    return Node != NULL;
  }

 private:
  const std::string Id;
};
template <typename T>
class VerifyIdIsBoundToStmt : public BoundNodesCallback {
 public:
  explicit VerifyIdIsBoundToStmt(const std::string &Id) : Id(Id) {}
  virtual bool Run(const BoundNodes *Nodes) {
    const T *Node = Nodes->GetStmtAs<T>(Id);
    return Node != NULL;
  }
 private:
  const std::string Id;
};

TEST(Matcher, BindMatchedNodes) {
  DeclarationMatcher ClassX = Has(Id("x", Class(HasName("X"))));

  EXPECT_TRUE(MatchAndVerifyResultTrue("class X {};",
      ClassX, new VerifyIdIsBoundToDecl<clang::CXXRecordDecl>("x")));

  EXPECT_TRUE(MatchAndVerifyResultFalse("class X {};",
      ClassX, new VerifyIdIsBoundToDecl<clang::CXXRecordDecl>("other-id")));

  TypeMatcher TypeAHasClassB = HasDeclaration(
      Class(HasName("A"), Has(Id("b", Class(HasName("B"))))));

  EXPECT_TRUE(MatchAndVerifyResultTrue("class A { public: A *a; class B {}; };",
      TypeAHasClassB,
      new VerifyIdIsBoundToDecl<clang::Decl>("b")));

  StatementMatcher MethodX = Id("x", Call(Callee(Method(HasName("x")))));

  EXPECT_TRUE(MatchAndVerifyResultTrue("class A { void x() { x(); } };",
      MethodX,
      new VerifyIdIsBoundToStmt<clang::CXXMemberCallExpr>("x")));
}

TEST(HasType, TakesQualTypeMatcherAndMatchesExpr) {
  TypeMatcher ClassX = HasDeclaration(Class(HasName("X")));
  EXPECT_TRUE(
      Matches("class X {}; void y(X &x) { x; }", Expression(HasType(ClassX))));
  EXPECT_TRUE(
      NotMatches("class X {}; void y(X *x) { x; }",
                 Expression(HasType(ClassX))));
  EXPECT_TRUE(
      Matches("class X {}; void y(X *x) { x; }",
              Expression(HasType(PointsTo(ClassX)))));
}

TEST(HasType, TakesQualTypeMatcherAndMatchesValueDecl) {
  TypeMatcher ClassX = HasDeclaration(Class(HasName("X")));
  EXPECT_TRUE(
      Matches("class X {}; void y() { X x; }", Variable(HasType(ClassX))));
  EXPECT_TRUE(
      NotMatches("class X {}; void y() { X *x; }", Variable(HasType(ClassX))));
  EXPECT_TRUE(
      Matches("class X {}; void y() { X *x; }",
              Variable(HasType(PointsTo(ClassX)))));
}

TEST(HasType, TakesDeclMatcherAndMatchesExpr) {
  DeclarationMatcher ClassX = Class(HasName("X"));
  EXPECT_TRUE(
      Matches("class X {}; void y(X &x) { x; }", Expression(HasType(ClassX))));
  EXPECT_TRUE(
      NotMatches("class X {}; void y(X *x) { x; }",
                 Expression(HasType(ClassX))));
}

TEST(HasType, TakesDeclMatcherAndMatchesValueDecl) {
  DeclarationMatcher ClassX = Class(HasName("X"));
  EXPECT_TRUE(
      Matches("class X {}; void y() { X x; }", Variable(HasType(ClassX))));
  EXPECT_TRUE(
      NotMatches("class X {}; void y() { X *x; }", Variable(HasType(ClassX))));
}

TEST(Matcher, Call) {
  // FIXME: Do we want to overload Call() to directly take
  // Matcher<clang::Decl>, too?
  StatementMatcher MethodX = Call(HasDeclaration(Method(HasName("x"))));

  EXPECT_TRUE(Matches("class Y { void x() { x(); } };", MethodX));
  EXPECT_TRUE(NotMatches("class Y { void x() {} };", MethodX));

  StatementMatcher MethodOnY = Call(On(HasType(Class(HasName("Y")))));

  EXPECT_TRUE(
      Matches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      Matches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      NotMatches("class Y { public: void x(); }; void z(Y *&y) { y->x(); }",
                 MethodOnY));
  EXPECT_TRUE(
      NotMatches("class Y { public: void x(); }; void z(Y y[]) { y->x(); }",
                 MethodOnY));
  EXPECT_TRUE(
      NotMatches("class Y { public: void x(); }; void z() { Y *y; y->x(); }",
                 MethodOnY));

  StatementMatcher MethodOnYPointer =
      Call(On(HasType(PointsTo(Class(HasName("Y"))))));

  EXPECT_TRUE(
      Matches("class Y { public: void x(); }; void z() { Y *y; y->x(); }",
              MethodOnYPointer));
  EXPECT_TRUE(
      Matches("class Y { public: void x(); }; void z(Y *&y) { y->x(); }",
              MethodOnYPointer));
  EXPECT_TRUE(
      Matches("class Y { public: void x(); }; void z(Y y[]) { y->x(); }",
              MethodOnYPointer));
  EXPECT_TRUE(
      NotMatches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
                 MethodOnYPointer));
  EXPECT_TRUE(
      NotMatches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
                 MethodOnYPointer));
}

TEST(Matcher, OverloadedOperatorCall) {
  StatementMatcher OpCall = OverloadedOperatorCall();
  // Unary operator
  EXPECT_TRUE(Matches("class Y { }; "
              "bool operator!(Y x) { return false; }; "
              "Y y; bool c = !y;", OpCall));
  // No match -- special operators like "new", "delete"
  // FIXME: figure out why these does not match?
  EXPECT_TRUE(NotMatches("class Y { }; "
              "void *operator new(unsigned long size) { return 0; } "
              "Y *y = new Y;", OpCall));
  EXPECT_TRUE(NotMatches("class Y { }; "
              "void operator delete(void *p) { } "
              "void a() {Y *y = new Y; delete y;}", OpCall));
  // Binary operator
  EXPECT_TRUE(Matches("class Y { }; "
              "bool operator&&(Y x, Y y) { return true; }; "
              "Y a; Y b; bool c = a && b;",
              OpCall));
  // No match -- normal operator, not an overloaded one.
  EXPECT_TRUE(NotMatches("bool x = true, y = true; bool t = x && y;", OpCall));
  EXPECT_TRUE(NotMatches("int t = 5 << 2;", OpCall));
}

TEST(Matcher, HasOperatorNameForOverloadedOperatorCall) {
  StatementMatcher OpCallAndAnd =
      OverloadedOperatorCall(HasOverloadedOperatorName("&&"));
  EXPECT_TRUE(Matches("class Y { }; "
              "bool operator&&(Y x, Y y) { return true; }; "
              "Y a; Y b; bool c = a && b;", OpCallAndAnd));
  StatementMatcher OpCallLessLess =
      OverloadedOperatorCall(HasOverloadedOperatorName("<<"));
  EXPECT_TRUE(NotMatches("class Y { }; "
              "bool operator&&(Y x, Y y) { return true; }; "
              "Y a; Y b; bool c = a && b;",
              OpCallLessLess));
}

TEST(Matcher, ThisPointerType) {
  StatementMatcher MethodOnY = Call(ThisPointerType(Class(HasName("Y"))));

  EXPECT_TRUE(
      Matches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      Matches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      Matches("class Y { public: void x(); }; void z(Y *&y) { y->x(); }",
              MethodOnY));
  EXPECT_TRUE(
      Matches("class Y { public: void x(); }; void z(Y y[]) { y->x(); }",
              MethodOnY));
  EXPECT_TRUE(
      Matches("class Y { public: void x(); }; void z() { Y *y; y->x(); }",
              MethodOnY));

  EXPECT_TRUE(Matches(
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
      DeclarationReference(To(
          Variable(HasInitializer(
              Call(ThisPointerType(Class(HasName("Y"))))))));

  EXPECT_TRUE(Matches(
      "class Y {"
      " public:"
      "  bool x() const;"
      "};"
      "void z(const Y &y) {"
      "  bool b = y.x();"
      "  if (b) {}"
      "}", Reference));

  EXPECT_TRUE(NotMatches(
      "class Y {"
      " public:"
      "  bool x() const;"
      "};"
      "void z(const Y &y) {"
      "  bool b = y.x();"
      "}", Reference));
}

TEST(Matcher, CalledVariable) {
  StatementMatcher CallOnVariableY = Expression(
      Call(On(DeclarationReference(To(Variable(HasName("y")))))));

  EXPECT_TRUE(Matches(
      "class Y { public: void x() { Y y; y.x(); } };", CallOnVariableY));
  EXPECT_TRUE(Matches(
      "class Y { public: void x() const { Y y; y.x(); } };", CallOnVariableY));
  EXPECT_TRUE(Matches(
      "class Y { public: void x(); };"
      "class X : public Y { void z() { X y; y.x(); } };", CallOnVariableY));
  EXPECT_TRUE(Matches(
      "class Y { public: void x(); };"
      "class X : public Y { void z() { X *y; y->x(); } };", CallOnVariableY));
  EXPECT_TRUE(NotMatches(
      "class Y { public: void x(); };"
      "class X : public Y { void z() { unsigned long y; ((X*)y)->x(); } };",
      CallOnVariableY));
}

TEST(MemberExpression, DoesNotMatchClasses) {
  EXPECT_TRUE(NotMatches("class Y { void x() {} };", MemberExpression()));
}

TEST(MemberExpression, MatchesMemberFunctionCall) {
  EXPECT_TRUE(Matches("class Y { void x() { x(); } };", MemberExpression()));
}

TEST(MemberExpression, MatchesVariable) {
  EXPECT_TRUE(
      Matches("class Y { void x() { this->y; } int y; };", MemberExpression()));
  EXPECT_TRUE(
      Matches("class Y { void x() { y; } int y; };", MemberExpression()));
  EXPECT_TRUE(
      Matches("class Y { void x() { Y y; y.y; } int y; };",
              MemberExpression()));
}

TEST(MemberExpression, MatchesStaticVariable) {
  EXPECT_TRUE(Matches("class Y { void x() { this->y; } static int y; };",
              MemberExpression()));
  EXPECT_TRUE(NotMatches("class Y { void x() { y; } static int y; };",
              MemberExpression()));
  EXPECT_TRUE(NotMatches("class Y { void x() { Y::y; } static int y; };",
              MemberExpression()));
}

TEST(IsArrow, MatchesMemberVariablesViaArrow) {
  EXPECT_TRUE(Matches("class Y { void x() { this->y; } int y; };",
              MemberExpression(IsArrow())));
  EXPECT_TRUE(Matches("class Y { void x() { y; } int y; };",
              MemberExpression(IsArrow())));
  EXPECT_TRUE(NotMatches("class Y { void x() { (*this).y; } int y; };",
              MemberExpression(IsArrow())));
}

TEST(IsArrow, MatchesStaticMemberVariablesViaArrow) {
  EXPECT_TRUE(Matches("class Y { void x() { this->y; } static int y; };",
              MemberExpression(IsArrow())));
  EXPECT_TRUE(NotMatches("class Y { void x() { y; } static int y; };",
              MemberExpression(IsArrow())));
  EXPECT_TRUE(NotMatches("class Y { void x() { (*this).y; } static int y; };",
              MemberExpression(IsArrow())));
}

TEST(IsArrow, MatchesMemberCallsViaArrow) {
  EXPECT_TRUE(Matches("class Y { void x() { this->x(); } };",
              MemberExpression(IsArrow())));
  EXPECT_TRUE(Matches("class Y { void x() { x(); } };",
              MemberExpression(IsArrow())));
  EXPECT_TRUE(NotMatches("class Y { void x() { Y y; y.x(); } };",
              MemberExpression(IsArrow())));
}

TEST(Callee, MatchesDeclarations) {
  StatementMatcher CallMethodX = Call(Callee(Method(HasName("x"))));

  EXPECT_TRUE(Matches("class Y { void x() { x(); } };", CallMethodX));
  EXPECT_TRUE(NotMatches("class Y { void x() {} };", CallMethodX));
}

TEST(Callee, MatchesMemberExpressions) {
  EXPECT_TRUE(Matches("class Y { void x() { this->x(); } };",
              Call(Callee(MemberExpression()))));
  EXPECT_TRUE(
      NotMatches("class Y { void x() { this->x(); } };", Call(Callee(Call()))));
}

TEST(Matcher, Argument) {
  StatementMatcher CallArgumentY = Expression(Call(
      HasArgument(0, DeclarationReference(To(Variable(HasName("y")))))));

  EXPECT_TRUE(Matches("void x(int) { int y; x(y); }", CallArgumentY));
  EXPECT_TRUE(
      Matches("class X { void x(int) { int y; x(y); } };", CallArgumentY));
  EXPECT_TRUE(NotMatches("void x(int) { int z; x(z); }", CallArgumentY));

  StatementMatcher WrongIndex = Expression(Call(
      HasArgument(42, DeclarationReference(To(Variable(HasName("y")))))));
  EXPECT_TRUE(NotMatches("void x(int) { int y; x(y); }", WrongIndex));
}

TEST(Matcher, AnyArgument) {
  StatementMatcher CallArgumentY = Expression(Call(
      HasAnyArgument(DeclarationReference(To(Variable(HasName("y")))))));
  EXPECT_TRUE(Matches("void x(int, int) { int y; x(1, y); }", CallArgumentY));
  EXPECT_TRUE(Matches("void x(int, int) { int y; x(y, 42); }", CallArgumentY));
  EXPECT_TRUE(NotMatches("void x(int, int) { x(1, 2); }", CallArgumentY));
}

TEST(Matcher, ArgumentCount) {
  StatementMatcher Call1Arg = Expression(Call(ArgumentCountIs(1)));

  EXPECT_TRUE(Matches("void x(int) { x(0); }", Call1Arg));
  EXPECT_TRUE(Matches("class X { void x(int) { x(0); } };", Call1Arg));
  EXPECT_TRUE(NotMatches("void x(int, int) { x(0, 0); }", Call1Arg));
}

TEST(Matcher, References) {
  DeclarationMatcher ReferenceClassX = Variable(
      HasType(References(Class(HasName("X")))));
  EXPECT_TRUE(Matches("class X {}; void y(X y) { X &x = y; }", ReferenceClassX));
  EXPECT_TRUE(
      Matches("class X {}; void y(X y) { const X &x = y; }", ReferenceClassX));
  EXPECT_TRUE(
      NotMatches("class X {}; void y(X y) { X x = y; }", ReferenceClassX));
  EXPECT_TRUE(
      NotMatches("class X {}; void y(X *y) { X *&x = y; }", ReferenceClassX));
}

TEST(HasParameter, CallsInnerMatcher) {
  EXPECT_TRUE(Matches("class X { void x(int) {} };",
      Method(HasParameter(0, Variable()))));
  EXPECT_TRUE(NotMatches("class X { void x(int) {} };",
      Method(HasParameter(0, HasName("x")))));
}

TEST(HasParameter, DoesNotMatchIfIndexOutOfBounds) {
  EXPECT_TRUE(NotMatches("class X { void x(int) {} };",
      Method(HasParameter(42, Variable()))));
}

TEST(HasType, MatchesParameterVariableTypesStrictly) {
  EXPECT_TRUE(Matches("class X { void x(X x) {} };",
      Method(HasParameter(0, HasType(Class(HasName("X")))))));
  EXPECT_TRUE(NotMatches("class X { void x(const X &x) {} };",
      Method(HasParameter(0, HasType(Class(HasName("X")))))));
  EXPECT_TRUE(Matches("class X { void x(const X *x) {} };",
      Method(HasParameter(0, HasType(PointsTo(Class(HasName("X"))))))));
  EXPECT_TRUE(Matches("class X { void x(const X &x) {} };",
      Method(HasParameter(0, HasType(References(Class(HasName("X"))))))));
}

TEST(HasAnyParameter, MatchesIndependentlyOfPosition) {
  EXPECT_TRUE(Matches("class Y {}; class X { void x(X x, Y y) {} };",
      Method(HasAnyParameter(HasType(Class(HasName("X")))))));
  EXPECT_TRUE(Matches("class Y {}; class X { void x(Y y, X x) {} };",
      Method(HasAnyParameter(HasType(Class(HasName("X")))))));
}

TEST(HasAnyParameter, DoesntMatchIfInnerMatcherDoesntMatch) {
  EXPECT_TRUE(NotMatches("class Y {}; class X { void x(int) {} };",
      Method(HasAnyParameter(HasType(Class(HasName("X")))))));
}

TEST(HasAnyParameter, DoesNotMatchThisPointer) {
  EXPECT_TRUE(NotMatches("class Y {}; class X { void x() {} };",
      Method(HasAnyParameter(HasType(PointsTo(Class(HasName("X"))))))));
}

TEST(HasName, MatchesParameterVariableDeclartions) {
  EXPECT_TRUE(Matches("class Y {}; class X { void x(int x) {} };",
      Method(HasAnyParameter(HasName("x")))));
  EXPECT_TRUE(NotMatches("class Y {}; class X { void x(int) {} };",
      Method(HasAnyParameter(HasName("x")))));
}

TEST(Matcher, ConstructorCall) {
  StatementMatcher Constructor = Expression(ConstructorCall());

  EXPECT_TRUE(
      Matches("class X { public: X(); }; void x() { X x; }", Constructor));
  EXPECT_TRUE(
      Matches("class X { public: X(); }; void x() { X x = X(); }",
              Constructor));
  EXPECT_TRUE(
      Matches("class X { public: X(int); }; void x() { X x = 0; }",
              Constructor));
  EXPECT_TRUE(Matches("class X {}; void x(int) { X x; }", Constructor));
}

TEST(Matcher, ConstructorArgument) {
  StatementMatcher Constructor = Expression(ConstructorCall(
      HasArgument(0, DeclarationReference(To(Variable(HasName("y")))))));

  EXPECT_TRUE(
      Matches("class X { public: X(int); }; void x() { int y; X x(y); }",
              Constructor));
  EXPECT_TRUE(
      Matches("class X { public: X(int); }; void x() { int y; X x = X(y); }",
              Constructor));
  EXPECT_TRUE(
      Matches("class X { public: X(int); }; void x() { int y; X x = y; }",
              Constructor));
  EXPECT_TRUE(
      NotMatches("class X { public: X(int); }; void x() { int z; X x(z); }",
                 Constructor));

  StatementMatcher WrongIndex = Expression(ConstructorCall(
      HasArgument(42, DeclarationReference(To(Variable(HasName("y")))))));
  EXPECT_TRUE(
      NotMatches("class X { public: X(int); }; void x() { int y; X x(y); }",
                 WrongIndex));
}

TEST(Matcher, ConstructorArgumentCount) {
  StatementMatcher Constructor1Arg =
      Expression(ConstructorCall(ArgumentCountIs(1)));

  EXPECT_TRUE(
      Matches("class X { public: X(int); }; void x() { X x(0); }",
              Constructor1Arg));
  EXPECT_TRUE(
      Matches("class X { public: X(int); }; void x() { X x = X(0); }",
              Constructor1Arg));
  EXPECT_TRUE(
      Matches("class X { public: X(int); }; void x() { X x = 0; }",
              Constructor1Arg));
  EXPECT_TRUE(
      NotMatches("class X { public: X(int, int); }; void x() { X x(0, 0); }",
                 Constructor1Arg));
}

TEST(Matcher, DefaultArgument) {
  StatementMatcher Arg = DefaultArgument();

  EXPECT_TRUE(Matches("void x(int, int = 0) { int y; x(y); }", Arg));
  EXPECT_TRUE(
      Matches("class X { void x(int, int = 0) { int y; x(y); } };", Arg));
  EXPECT_TRUE(NotMatches("void x(int, int = 0) { int y; x(y, 0); }", Arg));
}

TEST(Matcher, StringLiterals) {
  StatementMatcher Literal = Expression(StringLiteral());
  EXPECT_TRUE(Matches("const char *s = \"string\";", Literal));
  // wide string
  EXPECT_TRUE(Matches("const wchar_t *s = L\"string\";", Literal));
  // with escaped characters
  EXPECT_TRUE(Matches("const char *s = \"\x05five\";", Literal));
  // no matching -- though the data type is the same, there is no string literal
  EXPECT_TRUE(NotMatches("const char s[1] = {'a'};", Literal));
}

TEST(Matcher, CharacterLiterals) {
  StatementMatcher CharLiteral = Expression(CharacterLiteral());
  EXPECT_TRUE(Matches("const char c = 'c';", CharLiteral));
  // wide character
  EXPECT_TRUE(Matches("const char c = L'c';", CharLiteral));
  // wide character, Hex encoded, NOT MATCHED!
  EXPECT_TRUE(NotMatches("const wchar_t c = 0x2126;", CharLiteral));
  EXPECT_TRUE(NotMatches("const char c = 0x1;", CharLiteral));
}

TEST(Matcher, IntegerLiterals) {
  StatementMatcher HasIntLiteral = Expression(IntegerLiteral());
  EXPECT_TRUE(Matches("int i = 10;", HasIntLiteral));
  EXPECT_TRUE(Matches("int i = 0x1AB;", HasIntLiteral));
  EXPECT_TRUE(Matches("int i = 10L;", HasIntLiteral));
  EXPECT_TRUE(Matches("int i = 10U;", HasIntLiteral));

  // Non-matching cases (character literals, float and double)
  EXPECT_TRUE(NotMatches("int i = L'a';",
                HasIntLiteral));  // this is actually a character
                                  // literal cast to int
  EXPECT_TRUE(NotMatches("int i = 'a';", HasIntLiteral));
  EXPECT_TRUE(NotMatches("int i = 1e10;", HasIntLiteral));
  EXPECT_TRUE(NotMatches("int i = 10.0;", HasIntLiteral));
}

TEST(Matcher, Conditions) {
  StatementMatcher Condition = If(HasCondition(BoolLiteral(Equals(true))));

  EXPECT_TRUE(Matches("void x() { if (true) {} }", Condition));
  EXPECT_TRUE(NotMatches("void x() { if (false) {} }", Condition));
  EXPECT_TRUE(NotMatches("void x() { bool a = true; if (a) {} }", Condition));
  EXPECT_TRUE(NotMatches("void x() { if (true || false) {} }", Condition));
  EXPECT_TRUE(NotMatches("void x() { if (1) {} }", Condition));
}

TEST(MatchBinaryOperator, HasOperatorName) {
  StatementMatcher OperatorOr = BinaryOperator(HasOperatorName("||"));

  EXPECT_TRUE(Matches("void x() { true || false; }", OperatorOr));
  EXPECT_TRUE(NotMatches("void x() { true && false; }", OperatorOr));
}

TEST(MatchBinaryOperator, HasLHSAndHasRHS) {
  StatementMatcher OperatorTrueFalse =
      BinaryOperator(HasLHS(BoolLiteral(Equals(true))),
                     HasRHS(BoolLiteral(Equals(false))));

  EXPECT_TRUE(Matches("void x() { true || false; }", OperatorTrueFalse));
  EXPECT_TRUE(Matches("void x() { true && false; }", OperatorTrueFalse));
  EXPECT_TRUE(NotMatches("void x() { false || true; }", OperatorTrueFalse));
}

TEST(MatchBinaryOperator, HasEitherOperand) {
  StatementMatcher HasOperand =
      BinaryOperator(HasEitherOperand(BoolLiteral(Equals(false))));

  EXPECT_TRUE(Matches("void x() { true || false; }", HasOperand));
  EXPECT_TRUE(Matches("void x() { false && true; }", HasOperand));
  EXPECT_TRUE(NotMatches("void x() { true || true; }", HasOperand));
}

TEST(Matcher, BinaryOperatorTypes) {
  // Integration test that verifies the AST provides all binary operators in
  // a way we expect.
  // FIXME: Operator ','
  EXPECT_TRUE(
      Matches("void x() { 3, 4; }", BinaryOperator(HasOperatorName(","))));
  EXPECT_TRUE(
      Matches("bool b; bool c = (b = true);",
              BinaryOperator(HasOperatorName("="))));
  EXPECT_TRUE(
      Matches("bool b = 1 != 2;", BinaryOperator(HasOperatorName("!="))));
  EXPECT_TRUE(
      Matches("bool b = 1 == 2;", BinaryOperator(HasOperatorName("=="))));
  EXPECT_TRUE(Matches("bool b = 1 < 2;", BinaryOperator(HasOperatorName("<"))));
  EXPECT_TRUE(
      Matches("bool b = 1 <= 2;", BinaryOperator(HasOperatorName("<="))));
  EXPECT_TRUE(
      Matches("int i = 1 << 2;", BinaryOperator(HasOperatorName("<<"))));
  EXPECT_TRUE(
      Matches("int i = 1; int j = (i <<= 2);",
              BinaryOperator(HasOperatorName("<<="))));
  EXPECT_TRUE(Matches("bool b = 1 > 2;", BinaryOperator(HasOperatorName(">"))));
  EXPECT_TRUE(
      Matches("bool b = 1 >= 2;", BinaryOperator(HasOperatorName(">="))));
  EXPECT_TRUE(
      Matches("int i = 1 >> 2;", BinaryOperator(HasOperatorName(">>"))));
  EXPECT_TRUE(
      Matches("int i = 1; int j = (i >>= 2);",
              BinaryOperator(HasOperatorName(">>="))));
  EXPECT_TRUE(
      Matches("int i = 42 ^ 23;", BinaryOperator(HasOperatorName("^"))));
  EXPECT_TRUE(
      Matches("int i = 42; int j = (i ^= 42);",
              BinaryOperator(HasOperatorName("^="))));
  EXPECT_TRUE(
      Matches("int i = 42 % 23;", BinaryOperator(HasOperatorName("%"))));
  EXPECT_TRUE(
      Matches("int i = 42; int j = (i %= 42);",
              BinaryOperator(HasOperatorName("%="))));
  EXPECT_TRUE(
      Matches("bool b = 42  &23;", BinaryOperator(HasOperatorName("&"))));
  EXPECT_TRUE(
      Matches("bool b = true && false;",
              BinaryOperator(HasOperatorName("&&"))));
  EXPECT_TRUE(
      Matches("bool b = true; bool c = (b &= false);",
              BinaryOperator(HasOperatorName("&="))));
  EXPECT_TRUE(
      Matches("bool b = 42 | 23;", BinaryOperator(HasOperatorName("|"))));
  EXPECT_TRUE(
      Matches("bool b = true || false;",
              BinaryOperator(HasOperatorName("||"))));
  EXPECT_TRUE(
      Matches("bool b = true; bool c = (b |= false);",
              BinaryOperator(HasOperatorName("|="))));
  EXPECT_TRUE(
      Matches("int i = 42  *23;", BinaryOperator(HasOperatorName("*"))));
  EXPECT_TRUE(
      Matches("int i = 42; int j = (i *= 23);",
              BinaryOperator(HasOperatorName("*="))));
  EXPECT_TRUE(
      Matches("int i = 42 / 23;", BinaryOperator(HasOperatorName("/"))));
  EXPECT_TRUE(
      Matches("int i = 42; int j = (i /= 23);",
              BinaryOperator(HasOperatorName("/="))));
  EXPECT_TRUE(
      Matches("int i = 42 + 23;", BinaryOperator(HasOperatorName("+"))));
  EXPECT_TRUE(
      Matches("int i = 42; int j = (i += 23);",
              BinaryOperator(HasOperatorName("+="))));
  EXPECT_TRUE(
      Matches("int i = 42 - 23;", BinaryOperator(HasOperatorName("-"))));
  EXPECT_TRUE(
      Matches("int i = 42; int j = (i -= 23);",
              BinaryOperator(HasOperatorName("-="))));
  EXPECT_TRUE(
      Matches("struct A { void x() { void (A::*a)(); (this->*a)(); } };",
              BinaryOperator(HasOperatorName("->*"))));
  EXPECT_TRUE(
      Matches("struct A { void x() { void (A::*a)(); ((*this).*a)(); } };",
              BinaryOperator(HasOperatorName(".*"))));

  // Member expressions as operators are not supported in matches.
  EXPECT_TRUE(
      NotMatches("struct A { void x(A *a) { a->x(this); } };",
                 BinaryOperator(HasOperatorName("->"))));

  // Initializer assignments are not represented as operator equals.
  EXPECT_TRUE(
      NotMatches("bool b = true;", BinaryOperator(HasOperatorName("="))));

  // Array indexing is not represented as operator.
  EXPECT_TRUE(NotMatches("int a[42]; void x() { a[23]; }", UnaryOperator()));

  // Overloaded operators do not match at all.
  EXPECT_TRUE(NotMatches(
      "struct A { bool operator&&(const A &a) const { return false; } };"
      "void x() { A a, b; a && b; }",
      BinaryOperator()));
}

TEST(MatchUnaryOperator, HasOperatorName) {
  StatementMatcher OperatorNot = UnaryOperator(HasOperatorName("!"));

  EXPECT_TRUE(Matches("void x() { !true; } ", OperatorNot));
  EXPECT_TRUE(NotMatches("void x() { true; } ", OperatorNot));
}

TEST(MatchUnaryOperator, HasUnaryOperand) {
  StatementMatcher OperatorOnFalse =
      UnaryOperator(HasUnaryOperand(BoolLiteral(Equals(false))));

  EXPECT_TRUE(Matches("void x() { !false; }", OperatorOnFalse));
  EXPECT_TRUE(NotMatches("void x() { !true; }", OperatorOnFalse));
}

TEST(Matcher, UnaryOperatorTypes) {
  // Integration test that verifies the AST provides all unary operators in
  // a way we expect.
  EXPECT_TRUE(Matches("bool b = !true;", UnaryOperator(HasOperatorName("!"))));
  EXPECT_TRUE(
      Matches("bool b; bool *p = &b;", UnaryOperator(HasOperatorName("&"))));
  EXPECT_TRUE(Matches("int i = ~ 1;", UnaryOperator(HasOperatorName("~"))));
  EXPECT_TRUE(
      Matches("bool *p; bool b = *p;", UnaryOperator(HasOperatorName("*"))));
  EXPECT_TRUE(
      Matches("int i; int j = +i;", UnaryOperator(HasOperatorName("+"))));
  EXPECT_TRUE(
      Matches("int i; int j = -i;", UnaryOperator(HasOperatorName("-"))));
  EXPECT_TRUE(
      Matches("int i; int j = ++i;", UnaryOperator(HasOperatorName("++"))));
  EXPECT_TRUE(
      Matches("int i; int j = i++;", UnaryOperator(HasOperatorName("++"))));
  EXPECT_TRUE(
      Matches("int i; int j = --i;", UnaryOperator(HasOperatorName("--"))));
  EXPECT_TRUE(
      Matches("int i; int j = i--;", UnaryOperator(HasOperatorName("--"))));

  // We don't match conversion operators.
  EXPECT_TRUE(NotMatches("int i; double d = (double)i;", UnaryOperator()));

  // Function calls are not represented as operator.
  EXPECT_TRUE(NotMatches("void f(); void x() { f(); }", UnaryOperator()));

  // Overloaded operators do not match at all.
  // FIXME: We probably want to add that.
  EXPECT_TRUE(NotMatches(
      "struct A { bool operator!() const { return false; } };"
      "void x() { A a; !a; }", UnaryOperator(HasOperatorName("!"))));
}

TEST(Matcher, ConditionalOperator) {
  StatementMatcher Conditional = ConditionalOperator(
      HasCondition(BoolLiteral(Equals(true))),
      HasTrueExpression(BoolLiteral(Equals(false))));

  EXPECT_TRUE(Matches("void x() { true ? false : true; }", Conditional));
  EXPECT_TRUE(NotMatches("void x() { false ? false : true; }", Conditional));
  EXPECT_TRUE(NotMatches("void x() { true ? true : false; }", Conditional));

  StatementMatcher ConditionalFalse = ConditionalOperator(
      HasFalseExpression(BoolLiteral(Equals(false))));

  EXPECT_TRUE(Matches("void x() { true ? true : false; }", ConditionalFalse));
  EXPECT_TRUE(
      NotMatches("void x() { true ? false : true; }", ConditionalFalse));
}

TEST(Matcher, HasNameSupportsNamespaces) {
  EXPECT_TRUE(Matches("namespace a { namespace b { class C; } }",
              Class(HasName("a::b::C"))));
  EXPECT_TRUE(Matches("namespace a { namespace b { class C; } }",
              Class(HasName("::a::b::C"))));
  EXPECT_TRUE(Matches("namespace a { namespace b { class C; } }",
              Class(HasName("b::C"))));
  EXPECT_TRUE(Matches("namespace a { namespace b { class C; } }",
              Class(HasName("C"))));
  EXPECT_TRUE(NotMatches("namespace a { namespace b { class C; } }",
              Class(HasName("c::b::C"))));
  EXPECT_TRUE(NotMatches("namespace a { namespace b { class C; } }",
              Class(HasName("a::c::C"))));
  EXPECT_TRUE(NotMatches("namespace a { namespace b { class C; } }",
              Class(HasName("a::b::A"))));
  EXPECT_TRUE(NotMatches("namespace a { namespace b { class C; } }",
              Class(HasName("::C"))));
  EXPECT_TRUE(NotMatches("namespace a { namespace b { class C; } }",
              Class(HasName("::b::C"))));
  EXPECT_TRUE(NotMatches("namespace a { namespace b { class C; } }",
              Class(HasName("z::a::b::C"))));
  EXPECT_TRUE(NotMatches("namespace a { namespace b { class C; } }",
              Class(HasName("a+b::C"))));
  EXPECT_TRUE(NotMatches("namespace a { namespace b { class AC; } }",
              Class(HasName("C"))));
}

TEST(Matcher, HasNameSupportsOuterClasses) {
  EXPECT_TRUE(
      Matches("class A { class B { class C; }; };", Class(HasName("A::B::C"))));
  EXPECT_TRUE(
      Matches("class A { class B { class C; }; };",
              Class(HasName("::A::B::C"))));
  EXPECT_TRUE(
      Matches("class A { class B { class C; }; };", Class(HasName("B::C"))));
  EXPECT_TRUE(
      Matches("class A { class B { class C; }; };", Class(HasName("C"))));
  EXPECT_TRUE(
      NotMatches("class A { class B { class C; }; };",
                 Class(HasName("c::B::C"))));
  EXPECT_TRUE(
      NotMatches("class A { class B { class C; }; };",
                 Class(HasName("A::c::C"))));
  EXPECT_TRUE(
      NotMatches("class A { class B { class C; }; };",
                 Class(HasName("A::B::A"))));
  EXPECT_TRUE(
      NotMatches("class A { class B { class C; }; };", Class(HasName("::C"))));
  EXPECT_TRUE(
      NotMatches("class A { class B { class C; }; };",
                 Class(HasName("::B::C"))));
  EXPECT_TRUE(NotMatches("class A { class B { class C; }; };",
              Class(HasName("z::A::B::C"))));
  EXPECT_TRUE(
      NotMatches("class A { class B { class C; }; };",
                 Class(HasName("A+B::C"))));
}

TEST(Matcher, IsDefinition) {
  DeclarationMatcher DefinitionOfClassA =
      Class(HasName("A"), IsDefinition());
  EXPECT_TRUE(Matches("class A {};", DefinitionOfClassA));
  EXPECT_TRUE(NotMatches("class A;", DefinitionOfClassA));

  DeclarationMatcher DefinitionOfVariableA =
      Variable(HasName("a"), IsDefinition());
  EXPECT_TRUE(Matches("int a;", DefinitionOfVariableA));
  EXPECT_TRUE(NotMatches("extern int a;", DefinitionOfVariableA));

  DeclarationMatcher DefinitionOfMethodA =
      Method(HasName("a"), IsDefinition());
  EXPECT_TRUE(Matches("class A { void a() {} };", DefinitionOfMethodA));
  EXPECT_TRUE(NotMatches("class A { void a(); };", DefinitionOfMethodA));
}

TEST(Matcher, OfClass) {
  StatementMatcher Constructor = ConstructorCall(HasDeclaration(Method(
      OfClass(HasName("X")))));

  EXPECT_TRUE(
      Matches("class X { public: X(); }; void x(int) { X x; }", Constructor));
  EXPECT_TRUE(
      Matches("class X { public: X(); }; void x(int) { X x = X(); }",
              Constructor));
  EXPECT_TRUE(
      NotMatches("class Y { public: Y(); }; void x(int) { Y y; }",
                 Constructor));
}

TEST(Matcher, VisitsTemplateInstantiations) {
  EXPECT_TRUE(Matches(
      "class A { public: void x(); };"
      "template <typename T> class B { public: void y() { T t; t.x(); } };"
      "void f() { B<A> b; b.y(); }", Call(Callee(Method(HasName("x"))))));

  EXPECT_TRUE(Matches(
      "class A { public: void x(); };"
      "class C {"
      " public:"
      "  template <typename T> class B { public: void y() { T t; t.x(); } };"
      "};"
      "void f() {"
      "  C::B<A> b; b.y();"
      "}", Class(HasName("C"),
                 HasDescendant(Call(Callee(Method(HasName("x"))))))));
}

// For testing AST_MATCHER_P().
AST_MATCHER_P(clang::Decl, Just, Matcher<clang::Decl>, AMatcher) {
  // Make sure all special variables are used: node, match_finder,
  // bound_nodes_builder, and the parameter named 'AMatcher'.
  return AMatcher.Matches(Node, Finder, Builder);
}

TEST(AstMatcherPMacro, Works) {
  DeclarationMatcher HasClassB = Just(Has(Id("b", Class(HasName("B")))));

  EXPECT_TRUE(MatchAndVerifyResultTrue("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<clang::Decl>("b")));

  EXPECT_TRUE(MatchAndVerifyResultFalse("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<clang::Decl>("a")));

  EXPECT_TRUE(MatchAndVerifyResultFalse("class A { class C {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<clang::Decl>("b")));
}

AST_POLYMORPHIC_MATCHER_P(
    PolymorphicHas, Matcher<clang::Decl>, AMatcher) {
  COMPILE_ASSERT((llvm::is_same<NodeType, clang::Decl>::value) ||
                 (llvm::is_same<NodeType, clang::Stmt>::value),
                 assert_node_type_is_accessible);
  TypedBaseMatcher<clang::Decl> ChildMatcher(AMatcher);
  return Finder->MatchesChildOf(
      Node, ChildMatcher, Builder,
      ASTMatchFinder::kIgnoreImplicitCastsAndParentheses);
}

TEST(AstPolymorphicMatcherPMacro, Works) {
  DeclarationMatcher HasClassB = PolymorphicHas(Id("b", Class(HasName("B"))));

  EXPECT_TRUE(MatchAndVerifyResultTrue("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<clang::Decl>("b")));

  EXPECT_TRUE(MatchAndVerifyResultFalse("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<clang::Decl>("a")));

  EXPECT_TRUE(MatchAndVerifyResultFalse("class A { class C {}; };",
      HasClassB, new VerifyIdIsBoundToDecl<clang::Decl>("b")));

  StatementMatcher StatementHasClassB =
      PolymorphicHas(Class(HasName("B")));

  EXPECT_TRUE(Matches("void x() { class B {}; }", StatementHasClassB));
}

TEST(For, FindsForLoops) {
  EXPECT_TRUE(Matches("void f() { for(;;); }", For()));
  EXPECT_TRUE(Matches("void f() { if(true) for(;;); }", For()));
}

TEST(For, ReportsNoFalsePositives) {
  EXPECT_TRUE(NotMatches("void f() { ; }", For()));
  EXPECT_TRUE(NotMatches("void f() { if(true); }", For()));
}

TEST(CompoundStatement, HandlesSimpleCases) {
  EXPECT_TRUE(NotMatches("void f();", CompoundStatement()));
  EXPECT_TRUE(Matches("void f() {}", CompoundStatement()));
  EXPECT_TRUE(Matches("void f() {{}}", CompoundStatement()));
}

TEST(CompoundStatement, DoesNotMatchEmptyStruct) {
  // It's not a compound statement just because there's "{}" in the source
  // text.  This is an AST search, not grep.
  EXPECT_TRUE(NotMatches("namespace n { struct S {}; }",
              CompoundStatement()));
  EXPECT_TRUE(Matches("namespace n { struct S { void f() {{}} }; }",
              CompoundStatement()));
}

TEST(HasBody, FindsBodyOfForLoop) {
  StatementMatcher HasCompoundStatementBody =
      For(HasBody(CompoundStatement()));
  EXPECT_TRUE(Matches("void f() { for(;;) {} }",
              HasCompoundStatementBody));
  EXPECT_TRUE(NotMatches("void f() { for(;;); }",
              HasCompoundStatementBody));
}

TEST(HasAnySubstatement, MatchesForTopLevelCompoundStatement) {
  // The simplest case: every compound statement is in a function
  // definition, and the function body itself must be a compound
  // statement.
  EXPECT_TRUE(Matches("void f() { for (;;); }",
              CompoundStatement(HasAnySubstatement(For()))));
}

TEST(HasAnySubstatement, IsNotRecursive) {
  // It's really "has any immediate substatement".
  EXPECT_TRUE(NotMatches("void f() { if (true) for (;;); }",
              CompoundStatement(HasAnySubstatement(For()))));
}

TEST(HasAnySubstatement, MatchesInNestedCompoundStatements) {
  EXPECT_TRUE(Matches("void f() { if (true) { for (;;); } }",
              CompoundStatement(HasAnySubstatement(For()))));
}

TEST(HasAnySubstatement, FindsSubstatementBetweenOthers) {
  EXPECT_TRUE(Matches("void f() { 1; 2; 3; for (;;); 4; 5; 6; }",
              CompoundStatement(HasAnySubstatement(For()))));
}

TEST(StatementCountIs, FindsNoStatementsInAnEmptyCompoundStatement) {
  EXPECT_TRUE(Matches("void f() { }",
              CompoundStatement(StatementCountIs(0))));
  EXPECT_TRUE(NotMatches("void f() {}",
              CompoundStatement(StatementCountIs(1))));
}

TEST(StatementCountIs, AppearsToMatchOnlyOneCount) {
  EXPECT_TRUE(Matches("void f() { 1; }",
              CompoundStatement(StatementCountIs(1))));
  EXPECT_TRUE(NotMatches("void f() { 1; }",
              CompoundStatement(StatementCountIs(0))));
  EXPECT_TRUE(NotMatches("void f() { 1; }",
              CompoundStatement(StatementCountIs(2))));
}

TEST(StatementCountIs, WorksWithMultipleStatements) {
  EXPECT_TRUE(Matches("void f() { 1; 2; 3; }",
              CompoundStatement(StatementCountIs(3))));
}

TEST(StatementCountIs, WorksWithNestedCompoundStatements) {
  EXPECT_TRUE(Matches("void f() { { 1; } { 1; 2; 3; 4; } }",
              CompoundStatement(StatementCountIs(1))));
  EXPECT_TRUE(Matches("void f() { { 1; } { 1; 2; 3; 4; } }",
              CompoundStatement(StatementCountIs(2))));
  EXPECT_TRUE(NotMatches("void f() { { 1; } { 1; 2; 3; 4; } }",
              CompoundStatement(StatementCountIs(3))));
  EXPECT_TRUE(Matches("void f() { { 1; } { 1; 2; 3; 4; } }",
              CompoundStatement(StatementCountIs(4))));
}

} // end namespace tooling
} // end namespace clang
