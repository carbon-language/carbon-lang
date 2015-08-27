//===--- TodoCommentCheck.h - clang-tidy ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_TODOCOMMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_TODOCOMMENTCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {
namespace readability {

/// Finds TODO comments without a username or bug number.
///
/// Corresponding cpplint.py check: 'readability/todo'
class TodoCommentCheck : public ClangTidyCheck {
public:
  TodoCommentCheck(StringRef Name, ClangTidyContext *Context);
  void registerPPCallbacks(CompilerInstance &Compiler) override;

private:
  class TodoCommentHandler;
  std::unique_ptr<TodoCommentHandler> Handler;
};

} // namespace readability
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_TODOCOMMENTCHECK_H
