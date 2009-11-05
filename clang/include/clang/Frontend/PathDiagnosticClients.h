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

#ifndef LLVM_CLANG_FRONTEND_PATH_DIAGNOSTIC_CLIENTS_H
#define LLVM_CLANG_FRONTEND_PATH_DIAGNOSTIC_CLiENTS_H

#include <memory>
#include <string>
#include "llvm/ADT/SmallVector.h"

namespace clang {

class PathDiagnosticClient;
class Preprocessor;

PathDiagnosticClient*
CreateHTMLDiagnosticClient(const std::string& prefix, Preprocessor* PP = 0);

PathDiagnosticClient*
CreatePlistDiagnosticClient(const std::string& prefix, Preprocessor* PP,
                            PathDiagnosticClient *SubPD = 0);

} // end clang namespace
#endif
