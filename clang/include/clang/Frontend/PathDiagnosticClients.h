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

#include <string>

namespace clang {

class PathDiagnosticClient;
class Preprocessor;
class PreprocessorFactory;

PathDiagnosticClient* CreateHTMLDiagnosticClient(const std::string& prefix,
                                                 Preprocessor* PP = 0,
                                                 PreprocessorFactory* PPF = 0);
  
PathDiagnosticClient* CreatePlistDiagnosticClient(const std::string& prefix,
                                                  Preprocessor* PP,
                                                  PreprocessorFactory* PPF);
}

#endif
