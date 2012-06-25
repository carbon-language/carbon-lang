//===--- TestVisitor.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a utility class for RecursiveASTVisitor related tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TEST_VISITOR_H
#define LLVM_CLANG_TEST_VISITOR_H

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {

/// \brief Base class for simple RecursiveASTVisitor based tests.
///
/// This is a drop-in replacement for RecursiveASTVisitor itself, with the
/// additional capability of running it over a snippet of code.
///
/// Visits template instantiations by default.
template <typename T>
class TestVisitor : public RecursiveASTVisitor<T> {
public:
  TestVisitor() { }

  virtual ~TestVisitor() { }

  /// \brief Runs the current AST visitor over the given code.
  bool runOver(StringRef Code) {
    return tooling::runToolOnCode(CreateTestAction(), Code);
  }

  bool shouldVisitTemplateInstantiations() const {
    return true;
  }

protected:
  virtual ASTFrontendAction* CreateTestAction() {
    return new TestAction(this);
  }

  class FindConsumer : public ASTConsumer {
  public:
    FindConsumer(TestVisitor *Visitor) : Visitor(Visitor) {}

    virtual void HandleTranslationUnit(clang::ASTContext &Context) {
      Visitor->TraverseDecl(Context.getTranslationUnitDecl());
    }

  private:
    TestVisitor *Visitor;
  };

  class TestAction : public ASTFrontendAction {
  public:
    TestAction(TestVisitor *Visitor) : Visitor(Visitor) {}

    virtual clang::ASTConsumer* CreateASTConsumer(
        CompilerInstance& compiler, llvm::StringRef dummy) {
      Visitor->Context = &compiler.getASTContext();
      /// TestConsumer will be deleted by the framework calling us.
      return new FindConsumer(Visitor);
    }

  protected:
    TestVisitor *Visitor;
  };

  ASTContext *Context;
};


/// \brief A RecursiveASTVisitor for testing the RecursiveASTVisitor itself.
///
/// Allows simple creation of test visitors running matches on only a small
/// subset of the Visit* methods.
template <typename T, template <typename> class Visitor = TestVisitor>
class ExpectedLocationVisitor : public Visitor<T> {
public:
  ExpectedLocationVisitor()
    : ExpectedLine(0), ExpectedColumn(0), Found(false) {}

  virtual ~ExpectedLocationVisitor() {
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
}

#endif /* LLVM_CLANG_TEST_VISITOR_H */
