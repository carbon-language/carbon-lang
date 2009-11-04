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

class PathDiagnosticClientFactory {
public:
  PathDiagnosticClientFactory() {}
  virtual ~PathDiagnosticClientFactory() {}

  virtual const char *getName() const = 0;

  virtual PathDiagnosticClient*
  createPathDiagnosticClient(llvm::SmallVectorImpl<std::string> *FilesMade) = 0;
};

PathDiagnosticClient*
CreateHTMLDiagnosticClient(const std::string& prefix, Preprocessor* PP = 0,
                           llvm::SmallVectorImpl<std::string>* FilesMade = 0);

PathDiagnosticClientFactory*
CreateHTMLDiagnosticClientFactory(const std::string& prefix,
                                  Preprocessor* PP = 0);

PathDiagnosticClient*
CreatePlistDiagnosticClient(const std::string& prefix, Preprocessor* PP,
                            PathDiagnosticClientFactory *PF = 0);

} // end clang namespace
#endif
