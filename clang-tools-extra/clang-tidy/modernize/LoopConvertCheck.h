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
    RangeDescriptor();
    bool ContainerNeedsDereference;
    bool DerefByConstRef;
    bool DerefByValue;
    std::string ContainerString;
    QualType ElemType;
  };

  void getAliasRange(SourceManager &SM, SourceRange &DeclRange);

  void doConversion(ASTContext *Context, const VarDecl *IndexVar,
                    const VarDecl *MaybeContainer, const UsageResult &Usages,
                    const DeclStmt *AliasDecl, bool AliasUseRequired,
                    bool AliasFromForInit, const ForStmt *Loop,
                    RangeDescriptor Descriptor);

  StringRef getContainerString(ASTContext *Context, const ForStmt *Loop,
                               const Expr *ContainerExpr);

  void getArrayLoopQualifiers(ASTContext *Context,
                              const ast_matchers::BoundNodes &Nodes,
                              const Expr *ContainerExpr,
                              const UsageResult &Usages,
                              RangeDescriptor &Descriptor);

  void getIteratorLoopQualifiers(ASTContext *Context,
                                 const ast_matchers::BoundNodes &Nodes,
                                 RangeDescriptor &Descriptor);

  void determineRangeDescriptor(ASTContext *Context,
                                const ast_matchers::BoundNodes &Nodes,
                                const ForStmt *Loop, LoopFixerKind FixerKind,
                                const Expr *ContainerExpr,
                                const UsageResult &Usages,
                                RangeDescriptor &Descriptor);

  bool isConvertible(ASTContext *Context, const ast_matchers::BoundNodes &Nodes,
                     const ForStmt *Loop, LoopFixerKind FixerKind);

  std::unique_ptr<TUTrackingInfo> TUInfo;
  const Confidence::Level MinConfidence;
  const VariableNamer::NamingStyle NamingStyle;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_LOOP_CONVERT_H
