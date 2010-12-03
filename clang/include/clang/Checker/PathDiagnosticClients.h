//===--- PathDiagnosticClients.h - Path Diagnostic Clients ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface to create different path diagostic clients.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CHECKER_PATH_DIAGNOSTIC_CLIENTS_H
#define LLVM_CLANG_CHECKER_PATH_DIAGNOSTIC_CLiENTS_H

#include <string>

namespace clang {

class PathDiagnosticClient;
class Preprocessor;

PathDiagnosticClient*
createHTMLDiagnosticClient(const std::string& prefix, const Preprocessor &PP);

PathDiagnosticClient*
createPlistDiagnosticClient(const std::string& prefix, const Preprocessor &PP,
                            PathDiagnosticClient *SubPD = 0);

} // end clang namespace
#endif
