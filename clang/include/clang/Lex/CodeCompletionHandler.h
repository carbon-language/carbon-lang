//===--- CodeCompletionHandler.h - Preprocessor code completion -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CodeCompletionHandler interface, which provides
//  code-completion callbacks for the preprocessor.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LEX_CODECOMPLETIONHANDLER_H
#define LLVM_CLANG_LEX_CODECOMPLETIONHANDLER_H

namespace clang {

/// \brief Callback handler that receives notifications when performing code 
/// completion within the preprocessor.
class CodeCompletionHandler {
public:
  virtual ~CodeCompletionHandler();
  
  /// \brief Callback invoked when performing code completion for a preprocessor
  /// directive.
  ///
  /// This callback will be invoked when the preprocessor processes a '#' at the
  /// start of a line, followed by the code-completion token.
  ///
  /// \param InConditional Whether we're inside a preprocessor conditional
  /// already.
  virtual void CodeCompleteDirective(bool InConditional) { }
  
  /// \brief Callback invoked when performing code completion within a block of
  /// code that was excluded due to preprocessor conditionals.
  virtual void CodeCompleteInConditionalExclusion() { }
};
  
}

#endif // LLVM_CLANG_LEX_CODECOMPLETIONHANDLER_H
