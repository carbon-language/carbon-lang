//===--- LLVMDiagChecker.h - Diagnostic Checking Functions ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Check diagnostic messages.
//
//===----------------------------------------------------------------------===//

#ifndef DRIVER_LLVM_DIAG_CHECKER_H_
#define DRIVER_LLVM_DIAG_CHECKER_H_

#include "TextDiagnosticBuffer.h"
#include <string>
#include <vector>

namespace clang {

class Preprocessor;
class SourceLocation;
class SourceManager;

void ProcessFileDiagnosticChecking(TextDiagnosticBuffer &DiagClient,
                                   Preprocessor &PP, const std::string &InFile,
                                   SourceManager &SourceMgr,
                                   unsigned MainFileID,
                                   TextDiagnosticBuffer::DiagList
                                     &ExpectedErrors,
                                   TextDiagnosticBuffer::DiagList
                                     &ExpectedWarnings);
bool ReportCheckingResults(TextDiagnosticBuffer &DiagClient,
                           const TextDiagnosticBuffer::DiagList
                             &ExpectedErrors,
                           const TextDiagnosticBuffer::DiagList
                             &ExpectedWarnings,
                           SourceManager &SourceMgr);

} // end clang namespace

#endif
