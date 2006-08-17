//===--- clang.h - C-Language Front-end -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This is the header file that pulls together the top-level driver.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CLANG_H
#define LLVM_CLANG_CLANG_H

namespace llvm {
namespace clang {
class Preprocessor;
class LangOptions;
class Action;

/// DoPrintPreprocessedInput - Implement -E mode.
void DoPrintPreprocessedInput(unsigned MainFileID, Preprocessor &PP,
                              LangOptions &Options);

/// CreatePrintParserActionsAction - Return the actions implementation that
/// implements the -parse-print-callbacks option.
Action *CreatePrintParserActionsAction();

}  // end namespace clang
}  // end namespace llvm

#endif
