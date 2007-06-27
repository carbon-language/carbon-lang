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

#include <string>
#include <vector>

namespace clang {

class Preprocessor;
class SourceLocation;
class SourceManager;
class TextDiagnosticBuffer;

typedef std::vector<std::pair<SourceLocation, std::string> > DiagList;

void ProcessFileDiagnosticChecking(TextDiagnosticBuffer &DiagClient,
                                   Preprocessor &PP, const std::string &InFile,
                                   SourceManager &SourceMgr,
                                   unsigned MainFileID,
                                   DiagList &ExpectedErrors,
                                   DiagList &ExpectedWarnings);
bool ReportCheckingResults(TextDiagnosticBuffer &DiagClient,
                           const DiagList &ExpectedErrors,
                           const DiagList &ExpectedWarnings,
                           SourceManager &SourceMgr);

} // end clang namespace

#endif
