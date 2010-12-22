//===--- AnalysisConsumer.h - Front-end Analysis Engine Hooks ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header contains the functions necessary for a front-end to run various
// analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CHECKER_ANALYSISCONSUMER_H
#define LLVM_CLANG_CHECKER_ANALYSISCONSUMER_H

#include <string>

namespace clang {

class AnalyzerOptions;
class ASTConsumer;
class Preprocessor;

/// CreateAnalysisConsumer - Creates an ASTConsumer to run various code
/// analysis passes.  (The set of analyses run is controlled by command-line
/// options.)
ASTConsumer* CreateAnalysisConsumer(const Preprocessor &pp,
                                    const std::string &output,
                                    const AnalyzerOptions& Opts);

}

#endif
