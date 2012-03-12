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

#ifndef LLVM_CLANG_GR_PATH_DIAGNOSTIC_CLIENTS_H
#define LLVM_CLANG_GR_PATH_DIAGNOSTIC_CLIENTS_H

#include <string>

namespace clang {

class Preprocessor;

namespace ento {

class PathDiagnosticConsumer;

PathDiagnosticConsumer*
createHTMLDiagnosticConsumer(const std::string& prefix, const Preprocessor &PP);

PathDiagnosticConsumer*
createPlistDiagnosticConsumer(const std::string& prefix, const Preprocessor &PP,
                              PathDiagnosticConsumer *SubPD = 0);

PathDiagnosticConsumer*
createPlistMultiFileDiagnosticConsumer(const std::string& prefix,
                                       const Preprocessor &PP);

PathDiagnosticConsumer*
createTextPathDiagnosticConsumer(const std::string& prefix,
                                 const Preprocessor &PP);

} // end GR namespace

} // end clang namespace

#endif
