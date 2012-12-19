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
#include <vector>

namespace clang {

class AnalyzerOptions;
class Preprocessor;

namespace ento {

class PathDiagnosticConsumer;
typedef std::vector<PathDiagnosticConsumer*> PathDiagnosticConsumers;

#define CREATE_CONSUMER(NAME)\
void create ## NAME ## DiagnosticConsumer(AnalyzerOptions &AnalyzerOpts,\
                                          PathDiagnosticConsumers &C,\
                                          const std::string& prefix,\
                                          const Preprocessor &PP);

CREATE_CONSUMER(HTML)
CREATE_CONSUMER(Plist)
CREATE_CONSUMER(PlistMultiFile)
CREATE_CONSUMER(TextPath)

#undef CREATE_CONSUMER

} // end 'ento' namespace
} // end 'clang' namespace

#endif
