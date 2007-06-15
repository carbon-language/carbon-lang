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

namespace clang {
class Preprocessor;
class LangOptions;
class MinimalAction;
class TargetInfo;
class Diagnostic;

/// DoPrintPreprocessedInput - Implement -E mode.
void DoPrintPreprocessedInput(unsigned MainFileID, Preprocessor &PP,
                              const LangOptions &Options);

/// CreatePrintParserActionsAction - Return the actions implementation that
/// implements the -parse-print-callbacks option.
MinimalAction *CreatePrintParserActionsAction();

/// CreateTargetInfo - Return the set of target info objects as specified by
/// the -arch command line option.
TargetInfo *CreateTargetInfo(Diagnostic &Diags);


void EmitLLVMFromASTs(Preprocessor &PP, unsigned MainFileID,
                      bool PrintStats);
  
}  // end namespace clang

#endif
