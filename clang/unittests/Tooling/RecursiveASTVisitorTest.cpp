//===- unittest/Tooling/RecursiveASTVisitorTest.cpp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {

/// \brief Base class for sipmle RecursiveASTVisitor based tests.
///
/// This is a drop-in replacement for RecursiveASTVisitor itself, with the
/// additional capability of running it over a snippet of code.
///
/// Visits template instantiations by default.
///
/// FIXME: Put into a common location.
template <typename T>
class TestVisitor : public clang::RecursiveASTVisitor<T> {
public:
  /// \brief Runs the current AST visitor over the given code.
  bool runOver(StringRef Code) {
    return tooling::runToolOnCode(new TestAction(this), Code);
  }

  bool shouldVisitTemplateInstantiations() const {
    return true;
  }

protected:
  clang::ASTContext *Context;

private:
  class FindConsumer : public clang::ASTConsumer {
  public:
    FindConsumer(TestVisitor *Visitor) : Visitor(Visitor) {}

    virtual void HandleTranslationUnit(clang::ASTContext &Context) {
      Visitor->TraverseDecl(Context.getTranslationUnitDecl());
    }

  private:
    TestVisitor *Visitor;
  };

  class TestAction : public clang::ASTFrontendAction {
  public:
    TestAction(TestVisitor *Visitor) : Visitor(Visitor) {}

    virtual clang::ASTConsumer* CreateASTConsumer(
        clang::CompilerInstance& compiler, llvm::StringRef dummy) {
      Visitor->Context = &compiler.getASTContext();
      /// TestConsumer will be deleted by the framework calling us.
      return new FindConsumer(Visitor);
    }

  private:
    TestVisitor *Visitor;
  };
};

/// \brief A RecursiveASTVisitor for testing the RecursiveASTVisitor itself.
///
/// Allows simple creation of test visitors running matches on only a small
/// subset of the Visit* methods.
template <typename T>
class ExpectedLocationVisitor : public TestVisitor<T> {
public:
  ExpectedLocationVisitor()
    : ExpectedLine(0), ExpectedColumn(0), Found(false) {}

  ~ExpectedLocationVisitor() {
    EXPECT_TRUE(Found)
      << "Expected \"" << ExpectedMatch << "\" at " << ExpectedLine
      << ":" << ExpectedColumn << PartialMatches;
  }

  /// \brief Expect 'Match' to occur at the given 'Line' and 'Column'.
  void ExpectMatch(Twine Match, unsigned Line, unsigned Column) {
    ExpectedMatch = Match.str();
    ExpectedLine = Line;
    ExpectedColumn = Column;
  }

protected:
  /// \brief Convenience method to simplify writing test visitors.
  ///
  /// Sets 'Found' to true if 'Name' and 'Location' match the expected
  /// values. If only a partial match is found, record the information
  /// to produce nice error output when a test fails.
  ///
  /// Implementations are required to call this with appropriate values
  /// for 'Name' during visitation.
  void Match(StringRef Name, SourceLocation Location) {
    FullSourceLoc FullLocation = this->Context->getFullLoc(Location);
    if (Name == ExpectedMatch &&
        FullLocation.isValid() &&
        FullLocation.getSpellingLineNumber() == ExpectedLine &&
        FullLocation.getSpellingColumnNumber() == ExpectedColumn) {
      EXPECT_TRUE(!Found);
      Found = true;
    } else if (Name == ExpectedMatch ||
               (FullLocation.isValid() &&
                FullLocation.getSpellingLineNumber() == ExpectedLine &&
                FullLocation.getSpellingColumnNumber() == ExpectedColumn)) {
      // If we did not match, record information about partial matches.
      llvm::raw_string_ostream Stream(PartialMatches);
      Stream << ", partial match: \"" << Name << "\" at ";
      Location.print(Stream, this->Context->getSourceManager());
    }
  }

  std::string ExpectedMatch;
  unsigned ExpectedLine;
  unsigned ExpectedColumn;
  std::string PartialMatches;
  bool Found;
};

class TypeLocVisitor : public ExpectedLocationVisitor<TypeLocVisitor> {
public:
  bool VisitTypeLoc(TypeLoc TypeLocation) {
    Match(TypeLocation.getType().getAsString(), TypeLocation.getBeginLoc());
    return true;
  }
};

class DeclRefExprVisitor : public ExpectedLocationVisitor<DeclRefExprVisitor> {
public:
  bool VisitDeclRefExpr(DeclRefExpr *Reference) {
    Match(Reference->getNameInfo().getAsString(), Reference->getLocation());
    return true;
  }
};

class CXXMemberCallVisitor
  : public ExpectedLocationVisitor<CXXMemberCallVisitor> {
public:
  bool VisitCXXMemberCallExpr(CXXMemberCallExpr *Call) {
    Match(Call->getMethodDecl()->getQualifiedNameAsString(),
          Call->getLocStart());
    return true;
  }
};

class NamedDeclVisitor
  : public ExpectedLocationVisitor<NamedDeclVisitor> {
public:
  bool VisitNamedDecl(NamedDecl *Decl) {
    std::string NameWithTemplateArgs;
    Decl->getNameForDiagnostic(NameWithTemplateArgs,
                               Decl->getASTContext().getPrintingPolicy(),
                               true);
    Match(NameWithTemplateArgs, Decl->getLocation());
    return true;
  }
};

class CXXOperatorCallExprTraverser
  : public ExpectedLocationVisitor<CXXOperatorCallExprTraverser> {
public:
  // Use Traverse, not Visit, to check that data recursion optimization isn't
  // bypassing the call of this function.
  bool TraverseCXXOperatorCallExpr(CXXOperatorCallExpr *CE) {
    Match(getOperatorSpelling(CE->getOperator()), CE->getExprLoc());
    return ExpectedLocationVisitor<CXXOperatorCallExprTraverser>::
        TraverseCXXOperatorCallExpr(CE);
  }
};

class ParenExprVisitor : public ExpectedLocationVisitor<ParenExprVisitor> {
public:
  bool VisitParenExpr(ParenExpr *Parens) {
    Match("", Parens->getExprLoc());
    return true;
  }
};

class TemplateArgumentLocTraverser
  : public ExpectedLocationVisitor<TemplateArgumentLocTraverser> {
public:
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &ArgLoc) {
    std::string ArgStr;
    llvm::raw_string_ostream Stream(ArgStr);
    const TemplateArgument &Arg = ArgLoc.getArgument();

    Arg.print(Context->getPrintingPolicy(), Stream);
    Match(Stream.str(), ArgLoc.getLocation());
    return ExpectedLocationVisitor<TemplateArgumentLocTraverser>::
      TraverseTemplateArgumentLoc(ArgLoc);
  }
};

class CXXBoolLiteralExprVisitor 
  : public ExpectedLocationVisitor<CXXBoolLiteralExprVisitor> {
public:
  bool VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *BE) {
    if (BE->getValue())
      Match("true", BE->getLocation());
    else
      Match("false", BE->getLocation());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsBaseClassDeclarations) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("class X", 1, 30);
  EXPECT_TRUE(Visitor.runOver("class X {}; class Y : public X {};"));
}

TEST(RecursiveASTVisitor, VisitsCXXBaseSpecifiersOfForwardDeclaredClass) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("class X", 3, 18);
  EXPECT_TRUE(Visitor.runOver(
    "class Y;\n"
    "class X {};\n"
    "class Y : public X {};"));
}

TEST(RecursiveASTVisitor, VisitsCXXBaseSpecifiersWithIncompleteInnerClass) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("class X", 2, 18);
  EXPECT_TRUE(Visitor.runOver(
    "class X {};\n"
    "class Y : public X { class Z; };"));
}

TEST(RecursiveASTVisitor, VisitsCXXBaseSpecifiersOfSelfReferentialType) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("X<class Y>", 2, 18);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> class X {};\n"
    "class Y : public X<Y> {};"));
}

TEST(RecursiveASTVisitor, VisitsBaseClassTemplateArguments) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 2, 3);
  EXPECT_TRUE(Visitor.runOver(
    "void x(); template <void (*T)()> class X {};\nX<x> y;"));
}

TEST(RecursiveASTVisitor, VisitsCallExpr) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 1, 22);
  EXPECT_TRUE(Visitor.runOver(
    "void x(); void y() { x(); }"));
}

TEST(RecursiveASTVisitor, VisitsCallInTemplateInstantiation) {
  CXXMemberCallVisitor Visitor;
  Visitor.ExpectMatch("Y::x", 3, 3);
  EXPECT_TRUE(Visitor.runOver(
    "struct Y { void x(); };\n"
    "template<typename T> void y(T t) {\n"
    "  t.x();\n"
    "}\n"
    "void foo() { y<Y>(Y()); }"));
}

TEST(RecursiveASTVisitor, VisitsCallInNestedFunctionTemplateInstantiation) {
  CXXMemberCallVisitor Visitor;
  Visitor.ExpectMatch("Y::x", 4, 5);
  EXPECT_TRUE(Visitor.runOver(
    "struct Y { void x(); };\n"
    "template<typename T> struct Z {\n"
    "  template<typename U> static void f() {\n"
    "    T().x();\n"
    "  }\n"
    "};\n"
    "void foo() { Z<Y>::f<int>(); }"));
}

TEST(RecursiveASTVisitor, VisitsCallInNestedClassTemplateInstantiation) {
  CXXMemberCallVisitor Visitor;
  Visitor.ExpectMatch("A::x", 5, 7);
  EXPECT_TRUE(Visitor.runOver(
    "template <typename T1> struct X {\n"
    "  template <typename T2> struct Y {\n"
    "    void f() {\n"
    "      T2 y;\n"
    "      y.x();\n"
    "    }\n"
    "  };\n"
    "};\n"
    "struct A { void x(); };\n"
    "int main() {\n"
    "  (new X<A>::Y<A>())->f();\n"
    "}"));
}

/* FIXME: According to Richard Smith this is a bug in the AST.
TEST(RecursiveASTVisitor, VisitsBaseClassTemplateArgumentsInInstantiation) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 3, 43);
  EXPECT_TRUE(Visitor.runOver(
    "template <typename T> void x();\n"
    "template <void (*T)()> class X {};\n"
    "template <typename T> class Y : public X< x<T> > {};\n"
    "Y<int> y;"));
}
*/

TEST(RecursiveASTVisitor, VisitsCallInPartialTemplateSpecialization) {
  CXXMemberCallVisitor Visitor;
  Visitor.ExpectMatch("A::x", 6, 20);
  EXPECT_TRUE(Visitor.runOver(
    "template <typename T1> struct X {\n"
    "  template <typename T2, bool B> struct Y { void g(); };\n"
    "};\n"
    "template <typename T1> template <typename T2>\n"
    "struct X<T1>::Y<T2, true> {\n"
    "  void f() { T2 y; y.x(); }\n"
    "};\n"
    "struct A { void x(); };\n"
    "int main() {\n"
    "  (new X<A>::Y<A, true>())->f();\n"
    "}\n"));
}

TEST(RecursiveASTVisitor, VisitsExplicitTemplateSpecialization) {
  CXXMemberCallVisitor Visitor;
  Visitor.ExpectMatch("A::f", 4, 5);
  EXPECT_TRUE(Visitor.runOver(
    "struct A {\n"
    "  void f() const {}\n"
    "  template<class T> void g(const T& t) const {\n"
    "    t.f();\n"
    "  }\n"
    "};\n"
    "template void A::g(const A& a) const;\n"));
}

TEST(RecursiveASTVisitor, VisitsPartialTemplateSpecialization) {
  // From cfe-commits/Week-of-Mon-20100830/033998.html
  // Contrary to the approach suggested in that email, we visit all
  // specializations when we visit the primary template.  Visiting them when we
  // visit the associated specialization is problematic for specializations of
  // template members of class templates.
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("A<bool>", 1, 26);
  Visitor.ExpectMatch("A<char *>", 2, 26);
  EXPECT_TRUE(Visitor.runOver(
    "template <class T> class A {};\n"
    "template <class T> class A<T*> {};\n"
    "A<bool> ab;\n"
    "A<char*> acp;\n"));
}

TEST(RecursiveASTVisitor, VisitsUndefinedClassTemplateSpecialization) {
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("A<int>", 1, 29);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> struct A;\n"
    "A<int> *p;\n"));
}

TEST(RecursiveASTVisitor, VisitsNestedUndefinedClassTemplateSpecialization) {
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("A<int>::B<char>", 2, 31);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> struct A {\n"
    "  template<typename U> struct B;\n"
    "};\n"
    "A<int>::B<char> *p;\n"));
}

TEST(RecursiveASTVisitor, VisitsUndefinedFunctionTemplateSpecialization) {
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("A<int>", 1, 26);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> int A();\n"
    "int k = A<int>();\n"));
}

TEST(RecursiveASTVisitor, VisitsNestedUndefinedFunctionTemplateSpecialization) {
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("A<int>::B<char>", 2, 35);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> struct A {\n"
    "  template<typename U> static int B();\n"
    "};\n"
    "int k = A<int>::B<char>();\n"));
}

TEST(RecursiveASTVisitor, NoRecursionInSelfFriend) {
  // From cfe-commits/Week-of-Mon-20100830/033977.html
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("vector_iterator<int>", 2, 7);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename Container>\n"
    "class vector_iterator {\n"
    "    template <typename C> friend class vector_iterator;\n"
    "};\n"
    "vector_iterator<int> it_int;\n"));
}

TEST(RecursiveASTVisitor, TraversesOverloadedOperator) {
  CXXOperatorCallExprTraverser Visitor;
  Visitor.ExpectMatch("()", 4, 9);
  EXPECT_TRUE(Visitor.runOver(
    "struct A {\n"
    "  int operator()();\n"
    "} a;\n"
    "int k = a();\n"));
}

TEST(RecursiveASTVisitor, VisitsParensDuringDataRecursion) {
  ParenExprVisitor Visitor;
  Visitor.ExpectMatch("", 1, 9);
  EXPECT_TRUE(Visitor.runOver("int k = (4) + 9;\n"));
}

TEST(RecursiveASTVisitor, VisitsClassTemplateNonTypeParmDefaultArgument) {
  CXXBoolLiteralExprVisitor Visitor;
  Visitor.ExpectMatch("true", 2, 19);
  EXPECT_TRUE(Visitor.runOver(
    "template<bool B> class X;\n"
    "template<bool B = true> class Y;\n"
    "template<bool B> class Y {};\n"));
}

TEST(RecursiveASTVisitor, VisitsClassTemplateTypeParmDefaultArgument) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("class X", 2, 23);
  EXPECT_TRUE(Visitor.runOver(
    "class X;\n"
    "template<typename T = X> class Y;\n"
    "template<typename T> class Y {};\n"));
}

TEST(RecursiveASTVisitor, VisitsClassTemplateTemplateParmDefaultArgument) {
  TemplateArgumentLocTraverser Visitor;
  Visitor.ExpectMatch("X", 2, 40);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> class X;\n"
    "template<template <typename> class T = X> class Y;\n"
    "template<template <typename> class T> class Y {};\n"));
}

// A visitor that visits implicit declarations and matches constructors.
class ImplicitCtorVisitor
    : public ExpectedLocationVisitor<ImplicitCtorVisitor> {
public:
  bool shouldVisitImplicitDeclarations() const { return true; }

  bool VisitCXXConstructorDecl(CXXConstructorDecl* Ctor) {
    if (Ctor->isImplicit()) {  // Was not written in source code
      if (const CXXRecordDecl* Class = Ctor->getParent()) {
        Match(Class->getName(), Ctor->getLocation());
      }
    }
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsImplicitCopyConstructors) {
  ImplicitCtorVisitor Visitor;
  Visitor.ExpectMatch("Simple", 2, 8);
  // Note: Clang lazily instantiates implicit declarations, so we need
  // to use them in order to force them to appear in the AST.
  EXPECT_TRUE(Visitor.runOver(
      "struct WithCtor { WithCtor(); }; \n"
      "struct Simple { Simple(); WithCtor w; }; \n"
      "int main() { Simple s; Simple t(s); }\n"));
}

} // end namespace clang
