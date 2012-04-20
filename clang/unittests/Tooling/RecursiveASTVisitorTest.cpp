//===- unittest/AST/RecursiveASTMatcherTest.cpp ---------------------------===//
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

TEST(RecursiveASTVisitor, VisitsBaseClassDeclarations) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("class X", 1, 30);
  EXPECT_TRUE(Visitor.runOver("class X {}; class Y : public X {};"));
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

/* FIXME:
TEST(RecursiveASTVisitor, VisitsCallInNestedTemplateInstantiation) {
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
*/

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

} // end namespace clang

