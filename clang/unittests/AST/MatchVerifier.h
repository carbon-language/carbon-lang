//===- unittest/AST/MatchVerifier.h - AST unit test support ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Provides MatchVerifier, a base class to implement gtest matchers that
//  verify things that can be matched on the AST.
//
//  Also implements matchers based on MatchVerifier:
//  LocationVerifier and RangeVerifier to verify whether a matched node has
//  the expected source location or source range.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

enum Language { Lang_C, Lang_C89, Lang_CXX, Lang_CXX11, Lang_OpenCL };

/// \brief Base class for verifying some property of nodes found by a matcher.
template <typename NodeType>
class MatchVerifier : public MatchFinder::MatchCallback {
public:
  template <typename MatcherType>
  testing::AssertionResult match(const std::string &Code,
                                 const MatcherType &AMatcher) {
    return match(Code, AMatcher, Lang_CXX);
  }

  template <typename MatcherType>
  testing::AssertionResult match(const std::string &Code,
                                 const MatcherType &AMatcher, Language L);

protected:
  virtual void run(const MatchFinder::MatchResult &Result);
  virtual void verify(const MatchFinder::MatchResult &Result,
                      const NodeType &Node) {}

  void setFailure(const Twine &Result) {
    Verified = false;
    VerifyResult = Result.str();
  }

  void setSuccess() {
    Verified = true;
  }

private:
  bool Verified;
  std::string VerifyResult;
};

/// \brief Runs a matcher over some code, and returns the result of the
/// verifier for the matched node.
template <typename NodeType> template <typename MatcherType>
testing::AssertionResult MatchVerifier<NodeType>::match(
    const std::string &Code, const MatcherType &AMatcher, Language L) {
  MatchFinder Finder;
  Finder.addMatcher(AMatcher.bind(""), this);
  OwningPtr<tooling::FrontendActionFactory> Factory(
      tooling::newFrontendActionFactory(&Finder));

  std::vector<std::string> Args;
  StringRef FileName;
  switch (L) {
  case Lang_C:
    Args.push_back("-std=c99");
    FileName = "input.c";
    break;
  case Lang_C89:
    Args.push_back("-std=c89");
    FileName = "input.c";
    break;
  case Lang_CXX:
    Args.push_back("-std=c++98");
    FileName = "input.cc";
    break;
  case Lang_CXX11:
    Args.push_back("-std=c++11");
    FileName = "input.cc";
    break;
  case Lang_OpenCL:
    FileName = "input.cl";
  }

  // Default to failure in case callback is never called
  setFailure("Could not find match");
  if (!tooling::runToolOnCodeWithArgs(Factory->create(), Code, Args, FileName))
    return testing::AssertionFailure() << "Parsing error";
  if (!Verified)
    return testing::AssertionFailure() << VerifyResult;
  return testing::AssertionSuccess();
}

template <typename NodeType>
void MatchVerifier<NodeType>::run(const MatchFinder::MatchResult &Result) {
  const NodeType *Node = Result.Nodes.getNodeAs<NodeType>("");
  if (!Node) {
    setFailure("Matched node has wrong type");
  } else {
    // Callback has been called, default to success.
    setSuccess();
    verify(Result, *Node);
  }
}

/// \brief Verify whether a node has the correct source location.
///
/// By default, Node.getSourceLocation() is checked. This can be changed
/// by overriding getLocation().
template <typename NodeType>
class LocationVerifier : public MatchVerifier<NodeType> {
public:
  void expectLocation(unsigned Line, unsigned Column) {
    ExpectLine = Line;
    ExpectColumn = Column;
  }

protected:
  void verify(const MatchFinder::MatchResult &Result, const NodeType &Node) {
    SourceLocation Loc = getLocation(Node);
    unsigned Line = Result.SourceManager->getSpellingLineNumber(Loc);
    unsigned Column = Result.SourceManager->getSpellingColumnNumber(Loc);
    if (Line != ExpectLine || Column != ExpectColumn) {
      std::string MsgStr;
      llvm::raw_string_ostream Msg(MsgStr);
      Msg << "Expected location <" << ExpectLine << ":" << ExpectColumn
          << ">, found <";
      Loc.print(Msg, *Result.SourceManager);
      Msg << '>';
      this->setFailure(Msg.str());
    }
  }

  virtual SourceLocation getLocation(const NodeType &Node) {
    return Node.getLocation();
  }

private:
  unsigned ExpectLine, ExpectColumn;
};

/// \brief Verify whether a node has the correct source range.
///
/// By default, Node.getSourceRange() is checked. This can be changed
/// by overriding getRange().
template <typename NodeType>
class RangeVerifier : public MatchVerifier<NodeType> {
public:
  void expectRange(unsigned BeginLine, unsigned BeginColumn,
                   unsigned EndLine, unsigned EndColumn) {
    ExpectBeginLine = BeginLine;
    ExpectBeginColumn = BeginColumn;
    ExpectEndLine = EndLine;
    ExpectEndColumn = EndColumn;
  }

protected:
  void verify(const MatchFinder::MatchResult &Result, const NodeType &Node) {
    SourceRange R = getRange(Node);
    SourceLocation Begin = R.getBegin();
    SourceLocation End = R.getEnd();
    unsigned BeginLine = Result.SourceManager->getSpellingLineNumber(Begin);
    unsigned BeginColumn = Result.SourceManager->getSpellingColumnNumber(Begin);
    unsigned EndLine = Result.SourceManager->getSpellingLineNumber(End);
    unsigned EndColumn = Result.SourceManager->getSpellingColumnNumber(End);
    if (BeginLine != ExpectBeginLine || BeginColumn != ExpectBeginColumn ||
        EndLine != ExpectEndLine || EndColumn != ExpectEndColumn) {
      std::string MsgStr;
      llvm::raw_string_ostream Msg(MsgStr);
      Msg << "Expected range <" << ExpectBeginLine << ":" << ExpectBeginColumn
          << '-' << ExpectEndLine << ":" << ExpectEndColumn << ">, found <";
      Begin.print(Msg, *Result.SourceManager);
      Msg << '-';
      End.print(Msg, *Result.SourceManager);
      Msg << '>';
      this->setFailure(Msg.str());
    }
  }

  virtual SourceRange getRange(const NodeType &Node) {
    return Node.getSourceRange();
  }

private:
  unsigned ExpectBeginLine, ExpectBeginColumn, ExpectEndLine, ExpectEndColumn;
};

} // end namespace ast_matchers
} // end namespace clang
