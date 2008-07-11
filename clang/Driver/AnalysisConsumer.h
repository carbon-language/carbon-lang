//===--- AnalysisConsumer.cpp - ASTConsumer for running Analyses ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// "Meta" ASTConsumer for running different source analyses.
//
//===----------------------------------------------------------------------===//

#ifndef DRIVER_ANALYSISCONSUMER_H
#define DRIVER_ANALYSISCONSUMER_H

namespace clang {

enum Analyses {
  CFGDump,
  CFGView,
  WarnDeadStores,
  WarnUninitVals,
  DisplayLiveVariables,
  CheckerCFRef,
  CheckerSimple,
  CheckObjCMethSigs
};
  
ASTConsumer* CreateAnalysisConsumer(Analyses* Beg, Analyses* End,
                                    Diagnostic &diags, Preprocessor* pp,
                                    PreprocessorFactory* ppf,
                                    const LangOptions& lopts,
                                    const std::string& fname,
                                    const std::string& htmldir,
                                    bool visualize, bool trim,
                                    bool analyzeAll);
} // end clang namespace

#endif
