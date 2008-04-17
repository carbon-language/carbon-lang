//===--- HTMLPathDiagnostic.h - HTML Diagnostics for Paths ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface to create a HTMLPathDiagnostic object.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PATH_HTML_DIAGNOSTIC_H
#define LLVM_CLANG_PATH_HTML_DIAGNOSTIC_H

#include <string>

namespace clang {

class PathDiagnosticClient;
class Preprocessor;
class PreprocessorFactory;

  
PathDiagnosticClient* CreateHTMLDiagnosticClient(const std::string& prefix,
                                                 Preprocessor* PP,
                                                 PreprocessorFactory* PPF);
}

#endif
