//===--- ContainerSizeEmpty.cpp - clang-tidy ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "ContainerSizeEmpty.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace {
bool isContainer(llvm::StringRef ClassName) {
  static const llvm::StringSet<> ContainerNames = [] {
    llvm::StringSet<> RetVal;
    RetVal.insert("std::vector");
    RetVal.insert("std::list");
    RetVal.insert("std::array");
    RetVal.insert("std::deque");
    RetVal.insert("std::forward_list");
    RetVal.insert("std::set");
    RetVal.insert("std::map");
    RetVal.insert("std::multiset");
    RetVal.insert("std::multimap");
    RetVal.insert("std::unordered_set");
    RetVal.insert("std::unordered_map");
    RetVal.insert("std::unordered_multiset");
    RetVal.insert("std::unordered_multimap");
    RetVal.insert("std::stack");
    RetVal.insert("std::queue");
    RetVal.insert("std::priority_queue");
    return RetVal;
  }();
  return ContainerNames.find(ClassName) != ContainerNames.end();
}
}

namespace clang {
namespace ast_matchers {
AST_MATCHER(QualType, isBoolType) { return Node->isBooleanType(); }

AST_MATCHER(NamedDecl, stlContainer) {
  return isContainer(Node.getQualifiedNameAsString());
}
}
}

namespace clang {
namespace tidy {
namespace readability {

ContainerSizeEmptyCheck::ContainerSizeEmptyCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void ContainerSizeEmptyCheck::registerMatchers(MatchFinder *Finder) {
  const auto WrongUse = anyOf(
      hasParent(
          binaryOperator(
              anyOf(has(integerLiteral(equals(0))),
                    allOf(anyOf(hasOperatorName("<"), hasOperatorName(">="),
                                hasOperatorName(">"), hasOperatorName("<=")),
                          hasEitherOperand(integerLiteral(equals(1))))))
              .bind("SizeBinaryOp")),
      hasParent(implicitCastExpr(
          hasImplicitDestinationType(isBoolType()),
          anyOf(
              hasParent(unaryOperator(hasOperatorName("!")).bind("NegOnSize")),
              anything()))),
      hasParent(explicitCastExpr(hasDestinationType(isBoolType()))));

  Finder->addMatcher(
      memberCallExpr(
          on(expr(anyOf(hasType(namedDecl(stlContainer())),
                        hasType(pointsTo(namedDecl(stlContainer()))),
                        hasType(references(namedDecl(stlContainer())))))
                 .bind("STLObject")),
          callee(methodDecl(hasName("size"))), WrongUse).bind("SizeCallExpr"),
      this);
}

void ContainerSizeEmptyCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemberCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("SizeCallExpr");
  const auto *BinaryOp = Result.Nodes.getNodeAs<BinaryOperator>("SizeBinaryOp");
  const auto *E = Result.Nodes.getNodeAs<Expr>("STLObject");
  FixItHint Hint;
  std::string ReplacementText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(E->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());
  if (E->getType()->isPointerType())
    ReplacementText += "->empty()";
  else
    ReplacementText += ".empty()";

  if (BinaryOp) { // Determine the correct transformation.
    bool Negation = false;
    const bool ContainerIsLHS = !llvm::isa<IntegerLiteral>(BinaryOp->getLHS());
    const auto OpCode = BinaryOp->getOpcode();
    uint64_t Value = 0;
    if (ContainerIsLHS) {
      if (const auto *Literal =
              llvm::dyn_cast<IntegerLiteral>(BinaryOp->getRHS()))
        Value = Literal->getValue().getLimitedValue();
      else
        return;
    } else {
      Value = llvm::dyn_cast<IntegerLiteral>(BinaryOp->getLHS())
                  ->getValue()
                  .getLimitedValue();
    }

    // Constant that is not handled.
    if (Value > 1)
      return;

    // Always true, no warnings for that.
    if ((OpCode == BinaryOperatorKind::BO_GE && Value == 0 && ContainerIsLHS) ||
        (OpCode == BinaryOperatorKind::BO_LE && Value == 0 && !ContainerIsLHS))
      return;

    if (OpCode == BinaryOperatorKind::BO_NE && Value == 0)
      Negation = true;
    if ((OpCode == BinaryOperatorKind::BO_GT ||
         OpCode == BinaryOperatorKind::BO_GE) &&
        ContainerIsLHS)
      Negation = true;
    if ((OpCode == BinaryOperatorKind::BO_LT ||
         OpCode == BinaryOperatorKind::BO_LE) &&
        !ContainerIsLHS)
      Negation = true;

    if (Negation)
      ReplacementText = "!" + ReplacementText;
    Hint = FixItHint::CreateReplacement(BinaryOp->getSourceRange(),
                                        ReplacementText);

  } else {
    // If there is a conversion above the size call to bool, it is safe to just
    // replace size with empty.
    if (const auto *UnaryOp =
            Result.Nodes.getNodeAs<UnaryOperator>("NegOnSize"))
      Hint = FixItHint::CreateReplacement(UnaryOp->getSourceRange(),
                                          ReplacementText);
    else
      Hint = FixItHint::CreateReplacement(MemberCall->getSourceRange(),
                                          "!" + ReplacementText);
  }
  diag(MemberCall->getLocStart(),
       "The 'empty' method should be used to check for emptiness instead "
       "of 'size'.")
      << Hint;
}

} // namespace readability
} // namespace tidy
} // namespace clang
