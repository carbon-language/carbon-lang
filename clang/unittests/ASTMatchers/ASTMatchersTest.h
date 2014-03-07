//===- unittest/Tooling/ASTMatchersTest.h - Matcher tests helpers ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_AST_MATCHERS_AST_MATCHERS_TEST_H
#define LLVM_CLANG_UNITTESTS_AST_MATCHERS_AST_MATCHERS_TEST_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

using clang::tooling::buildASTFromCodeWithArgs;
using clang::tooling::newFrontendActionFactory;
using clang::tooling::runToolOnCodeWithArgs;
using clang::tooling::FrontendActionFactory;

class BoundNodesCallback {
public:
  virtual ~BoundNodesCallback() {}
  virtual bool run(const BoundNodes *BoundNodes) = 0;
  virtual bool run(const BoundNodes *BoundNodes, ASTContext *Context) = 0;
  virtual void onEndOfTranslationUnit() {}
};

// If 'FindResultVerifier' is not NULL, sets *Verified to the result of
// running 'FindResultVerifier' with the bound nodes as argument.
// If 'FindResultVerifier' is NULL, sets *Verified to true when Run is called.
class VerifyMatch : public MatchFinder::MatchCallback {
public:
  VerifyMatch(BoundNodesCallback *FindResultVerifier, bool *Verified)
      : Verified(Verified), FindResultReviewer(FindResultVerifier) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    if (FindResultReviewer != NULL) {
      *Verified |= FindResultReviewer->run(&Result.Nodes, Result.Context);
    } else {
      *Verified = true;
    }
  }

  void onEndOfTranslationUnit() override {
    if (FindResultReviewer)
      FindResultReviewer->onEndOfTranslationUnit();
  }

private:
  bool *const Verified;
  BoundNodesCallback *const FindResultReviewer;
};

template <typename T>
testing::AssertionResult matchesConditionally(const std::string &Code,
                                              const T &AMatcher,
                                              bool ExpectMatch,
                                              llvm::StringRef CompileArg) {
  bool Found = false, DynamicFound = false;
  MatchFinder Finder;
  Finder.addMatcher(AMatcher, new VerifyMatch(0, &Found));
  if (!Finder.addDynamicMatcher(AMatcher, new VerifyMatch(0, &DynamicFound)))
    return testing::AssertionFailure() << "Could not add dynamic matcher";
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  // Some tests use typeof, which is a gnu extension.
  std::vector<std::string> Args(1, CompileArg);
  if (!runToolOnCodeWithArgs(Factory->create(), Code, Args)) {
    return testing::AssertionFailure() << "Parsing error in \"" << Code << "\"";
  }
  if (Found != DynamicFound) {
    return testing::AssertionFailure() << "Dynamic match result ("
                                       << DynamicFound
                                       << ") does not match static result ("
                                       << Found << ")";
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
testing::AssertionResult matches(const std::string &Code, const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, true, "-std=c++11");
}

template <typename T>
testing::AssertionResult notMatches(const std::string &Code,
                                    const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, false, "-std=c++11");
}

template <typename T>
testing::AssertionResult
matchAndVerifyResultConditionally(const std::string &Code, const T &AMatcher,
                                  BoundNodesCallback *FindResultVerifier,
                                  bool ExpectResult) {
  std::unique_ptr<BoundNodesCallback> ScopedVerifier(FindResultVerifier);
  bool VerifiedResult = false;
  MatchFinder Finder;
  Finder.addMatcher(
      AMatcher, new VerifyMatch(FindResultVerifier, &VerifiedResult));
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  // Some tests use typeof, which is a gnu extension.
  std::vector<std::string> Args(1, "-std=gnu++98");
  if (!runToolOnCodeWithArgs(Factory->create(), Code, Args)) {
    return testing::AssertionFailure() << "Parsing error in \"" << Code << "\"";
  }
  if (!VerifiedResult && ExpectResult) {
    return testing::AssertionFailure()
      << "Could not verify result in \"" << Code << "\"";
  } else if (VerifiedResult && !ExpectResult) {
    return testing::AssertionFailure()
      << "Verified unexpected result in \"" << Code << "\"";
  }

  VerifiedResult = false;
  std::unique_ptr<ASTUnit> AST(buildASTFromCodeWithArgs(Code, Args));
  if (!AST.get())
    return testing::AssertionFailure() << "Parsing error in \"" << Code
                                       << "\" while building AST";
  Finder.matchAST(AST->getASTContext());
  if (!VerifiedResult && ExpectResult) {
    return testing::AssertionFailure()
      << "Could not verify result in \"" << Code << "\" with AST";
  } else if (VerifiedResult && !ExpectResult) {
    return testing::AssertionFailure()
      << "Verified unexpected result in \"" << Code << "\" with AST";
  }

  return testing::AssertionSuccess();
}

// FIXME: Find better names for these functions (or document what they
// do more precisely).
template <typename T>
testing::AssertionResult
matchAndVerifyResultTrue(const std::string &Code, const T &AMatcher,
                         BoundNodesCallback *FindResultVerifier) {
  return matchAndVerifyResultConditionally(
      Code, AMatcher, FindResultVerifier, true);
}

template <typename T>
testing::AssertionResult
matchAndVerifyResultFalse(const std::string &Code, const T &AMatcher,
                          BoundNodesCallback *FindResultVerifier) {
  return matchAndVerifyResultConditionally(
      Code, AMatcher, FindResultVerifier, false);
}

} // end namespace ast_matchers
} // end namespace clang

#endif  // LLVM_CLANG_UNITTESTS_AST_MATCHERS_AST_MATCHERS_TEST_H
