//===- examples/Tooling/RemoveCStrCalls.cpp - Redundant c_str call removal ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a tool that prints replacements that remove redundant
//  calls of c_str() on strings.
//
//  Usage:
//  remove-cstr-calls <cmake-output-dir> <file1> <file2> ...
//
//  Where <cmake-output-dir> is a CMake build directory in which a file named
//  compile_commands.json exists (enable -DCMAKE_EXPORT_COMPILE_COMMANDS in
//  CMake to get this output).
//
//  <file1> ... specify the paths of files in the CMake source tree. This path
//  is looked up in the compile command database. If the path of a file is
//  absolute, it needs to point into CMake's source tree. If the path is
//  relative, the current working directory needs to be in the CMake source
//  tree and the file must be in a subdirectory of the current working
//  directory. "./" prefixes in the relative files will be automatically
//  removed, but the rest of a relative path must be a suffix of a path in
//  the compile command line database.
//
//  For example, to use remove-cstr-calls on all files in a subtree of the
//  source tree, use:
//
//    /path/in/subtree $ find . -name '*.cpp'|
//        xargs remove-cstr-calls /path/to/source
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

using namespace clang::tooling::match;

// FIXME: Pull out helper methods in here into more fitting places.

// Returns the text that makes up 'node' in the source.
// Returns an empty string if the text cannot be found.
template <typename T>
std::string GetText(const clang::SourceManager &SourceManager, const T &Node) {
  clang::SourceLocation StartSpellingLocatino =
      SourceManager.getSpellingLoc(Node.getLocStart());
  clang::SourceLocation EndSpellingLocation =
      SourceManager.getSpellingLoc(Node.getLocEnd());
  if (!StartSpellingLocatino.isValid() || !EndSpellingLocation.isValid()) {
    return std::string();
  }
  bool Invalid = true;
  const char *Text =
      SourceManager.getCharacterData(StartSpellingLocatino, &Invalid);
  if (Invalid) {
    return std::string();
  }
  std::pair<clang::FileID, unsigned> Start =
      SourceManager.getDecomposedLoc(StartSpellingLocatino);
  std::pair<clang::FileID, unsigned> End =
      SourceManager.getDecomposedLoc(clang::Lexer::getLocForEndOfToken(
          EndSpellingLocation, 0, SourceManager, clang::LangOptions()));
  if (Start.first != End.first) {
    // Start and end are in different files.
    return std::string();
  }
  if (End.second < Start.second) {
    // Shuffling text with macros may cause this.
    return std::string();
  }
  return std::string(Text, End.second - Start.second);
}

// Returns the position of the spelling location of a node inside a file.
// The format is:
//     "<start_line>:<start_column>:<end_line>:<end_column>"
template <typename T1>
void PrintPosition(
    llvm::raw_ostream &OS,
    const clang::SourceManager &SourceManager, const T1 &Node) {
  clang::SourceLocation StartSpellingLocation =
      SourceManager.getSpellingLoc(Node.getLocStart());
  clang::SourceLocation EndSpellingLocation =
      SourceManager.getSpellingLoc(Node.getLocEnd());
  clang::PresumedLoc Start =
      SourceManager.getPresumedLoc(StartSpellingLocation);
  clang::SourceLocation EndToken = clang::Lexer::getLocForEndOfToken(
      EndSpellingLocation, 1, SourceManager, clang::LangOptions());
  clang::PresumedLoc End = SourceManager.getPresumedLoc(EndToken);
  OS << Start.getLine() << ":" << Start.getColumn() << ":"
     << End.getLine() << ":" << End.getColumn();
}

class ReportPosition : public clang::tooling::MatchFinder::MatchCallback {
 public:
  virtual void Run(const clang::tooling::MatchFinder::MatchResult &Result) {
    llvm::outs() << "Found!\n";
  }
};

// Return true if expr needs to be put in parens when it is an
// argument of a prefix unary operator, e.g. when it is a binary or
// ternary operator syntactically.
bool NeedParensAfterUnaryOperator(const clang::Expr &ExprNode) {
  if (llvm::dyn_cast<clang::BinaryOperator>(&ExprNode) ||
      llvm::dyn_cast<clang::ConditionalOperator>(&ExprNode)) {
    return true;
  }
  if (const clang::CXXOperatorCallExpr *op =
      llvm::dyn_cast<clang::CXXOperatorCallExpr>(&ExprNode)) {
    return op->getNumArgs() == 2 &&
        op->getOperator() != clang::OO_PlusPlus &&
        op->getOperator() != clang::OO_MinusMinus &&
        op->getOperator() != clang::OO_Call &&
        op->getOperator() != clang::OO_Subscript;
  }
  return false;
}

// Format a pointer to an expression: prefix with '*' but simplify
// when it already begins with '&'.  Return empty string on failure.
std::string FormatDereference(const clang::SourceManager &SourceManager,
                              const clang::Expr &ExprNode) {
  if (const clang::UnaryOperator *Op =
      llvm::dyn_cast<clang::UnaryOperator>(&ExprNode)) {
    if (Op->getOpcode() == clang::UO_AddrOf) {
      // Strip leading '&'.
      return GetText(SourceManager, *Op->getSubExpr()->IgnoreParens());
    }
  }
  const std::string Text = GetText(SourceManager, ExprNode);
  if (Text.empty()) return std::string();
  // Add leading '*'.
  if (NeedParensAfterUnaryOperator(ExprNode)) {
    return std::string("*(") + Text + ")";
  }
  return std::string("*") + Text;
}

class FixCStrCall : public clang::tooling::MatchFinder::MatchCallback {
 public:
  virtual void Run(const clang::tooling::MatchFinder::MatchResult &Result) {
    const clang::CallExpr *Call =
        Result.Nodes.GetStmtAs<clang::CallExpr>("call");
    const clang::Expr *Arg =
        Result.Nodes.GetStmtAs<clang::Expr>("arg");
    const bool Arrow =
        Result.Nodes.GetStmtAs<clang::MemberExpr>("member")->isArrow();
    // Replace the "call" node with the "arg" node, prefixed with '*'
    // if the call was using '->' rather than '.'.
    const std::string ArgText = Arrow ?
        FormatDereference(*Result.SourceManager, *Arg) :
        GetText(*Result.SourceManager, *Arg);
    if (ArgText.empty()) return;

    llvm::outs() <<
        Result.SourceManager->getBufferName(Call->getLocStart(), NULL) << ":";
    PrintPosition(llvm::outs(), *Result.SourceManager, *Call);
    llvm::outs() << ":" << ArgText << "\n";
  }
};

const char *StringConstructor =
    "::std::basic_string<char, std::char_traits<char>, std::allocator<char> >"
    "::basic_string";

const char *StringCStrMethod =
    "::std::basic_string<char, std::char_traits<char>, std::allocator<char> >"
    "::c_str";

int main(int argc, char **argv) {
  clang::tooling::ClangTool Tool(argc, argv);
  clang::tooling::MatchFinder finder;
  finder.AddMatcher(
      ConstructorCall(
          HasDeclaration(Method(HasName(StringConstructor))),
          ArgumentCountIs(2),
          // The first argument must have the form x.c_str() or p->c_str()
          // where the method is string::c_str().  We can use the copy
          // constructor of string instead (or the compiler might share
          // the string object).
          HasArgument(
              0,
              Id("call", Call(
                  Callee(Id("member", MemberExpression())),
                  Callee(Method(HasName(StringCStrMethod))),
                  On(Id("arg", Expression()))))),
          // The second argument is the alloc object which must not be
          // present explicitly.
          HasArgument(
              1,
              DefaultArgument())), new FixCStrCall);
  finder.AddMatcher(
      ConstructorCall(
          // Implicit constructors of these classes are overloaded
          // wrt. string types and they internally make a StringRef
          // referring to the argument.  Passing a string directly to
          // them is preferred to passing a char pointer.
          HasDeclaration(Method(AnyOf(
              HasName("::llvm::StringRef::StringRef"),
              HasName("::llvm::Twine::Twine")))),
          ArgumentCountIs(1),
          // The only argument must have the form x.c_str() or p->c_str()
          // where the method is string::c_str().  StringRef also has
          // a constructor from string which is more efficient (avoids
          // strlen), so we can construct StringRef from the string
          // directly.
          HasArgument(
              0,
              Id("call", Call(
                  Callee(Id("member", MemberExpression())),
                  Callee(Method(HasName(StringCStrMethod))),
                  On(Id("arg", Expression())))))),
      new FixCStrCall);
  return Tool.Run(finder.NewFrontendActionFactory());
}

