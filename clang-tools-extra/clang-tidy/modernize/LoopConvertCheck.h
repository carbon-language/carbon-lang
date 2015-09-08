//===--- LoopConvertCheck.h - clang-tidy-------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_LOOP_CONVERT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_LOOP_CONVERT_H

#include "../ClangTidy.h"
#include "LoopConvertUtils.h"

namespace clang {
namespace tidy {
namespace modernize {

class LoopConvertCheck : public ClangTidyCheck {
public:
  LoopConvertCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  struct RangeDescriptor {
    bool ContainerNeedsDereference;
    bool DerefByValue;
    bool IsTriviallyCopyable;
    bool DerefByConstRef;
  };

  void doConversion(ASTContext *Context, const VarDecl *IndexVar,
                    const VarDecl *MaybeContainer, StringRef ContainerString,
                    const UsageResult &Usages, const DeclStmt *AliasDecl,
                    bool AliasUseRequired, bool AliasFromForInit,
                    const ForStmt *TheLoop, RangeDescriptor Descriptor);

  StringRef checkRejections(ASTContext *Context, const Expr *ContainerExpr,
                            const ForStmt *TheLoop);

  void findAndVerifyUsages(ASTContext *Context, const VarDecl *LoopVar,
                           const VarDecl *EndVar, const Expr *ContainerExpr,
                           const Expr *BoundExpr, const ForStmt *TheLoop,
                           LoopFixerKind FixerKind, RangeDescriptor Descriptor);

  std::unique_ptr<TUTrackingInfo> TUInfo;
  Confidence::Level MinConfidence;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_LOOP_CONVERT_H
