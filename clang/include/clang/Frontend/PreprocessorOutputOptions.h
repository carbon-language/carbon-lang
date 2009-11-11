//===--- PreprocessorOutputOptions.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_PREPROCESSOROUTPUTOPTIONS_H
#define LLVM_CLANG_FRONTEND_PREPROCESSOROUTPUTOPTIONS_H

namespace clang {

/// PreprocessorOutputOptions - Options for controlling the C preprocessor
/// output (e.g., -E).
class PreprocessorOutputOptions {
public:
  unsigned ShowCPP : 1;           ///< Print normal preprocessed output.
  unsigned ShowMacros : 1;        ///< Print macro definitions.
  unsigned ShowLineMarkers : 1;   ///< Show #line markers.
  unsigned ShowComments : 1;      ///< Show comments.
  unsigned ShowMacroComments : 1; ///< Show comments, even in macros.

public:
  PreprocessorOutputOptions() {
    ShowCPP = 1;
    ShowMacros = 0;
    ShowLineMarkers = 1;
    ShowComments = 0;
    ShowMacroComments = 0;
  }
};

}  // end namespace clang

#endif
