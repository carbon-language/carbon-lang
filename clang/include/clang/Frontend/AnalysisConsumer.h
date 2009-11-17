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

#ifndef LLVM_CLANG_FRONTEND_ANALYSISCONSUMER_H
#define LLVM_CLANG_FRONTEND_ANALYSISCONSUMER_H

#include <string>
#include <vector>

namespace clang {
class ASTConsumer;
class Diagnostic;
class Preprocessor;
class LangOptions;

/// Analysis - Set of available source code analyses.
enum Analyses {
#define ANALYSIS(NAME, CMDFLAG, DESC, SCOPE) NAME,
#include "clang/Frontend/Analyses.def"
NumAnalyses
};

/// AnalysisStores - Set of available analysis store models.
enum AnalysisStores {
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN) NAME##Model,
#include "clang/Frontend/Analyses.def"
NumStores
};

/// AnalysisConstraints - Set of available constraint models.
enum AnalysisConstraints {
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN) NAME##Model,
#include "clang/Frontend/Analyses.def"
NumConstraints
};

/// AnalysisDiagClients - Set of available diagnostic clients for rendering
///  analysis results.
enum AnalysisDiagClients {
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN, AUTOCREAT) PD_##NAME,
#include "clang/Frontend/Analyses.def"
NUM_ANALYSIS_DIAG_CLIENTS
};

class AnalyzerOptions {
public:
  std::vector<Analyses> AnalysisList;
  AnalysisStores AnalysisStoreOpt;
  AnalysisConstraints AnalysisConstraintsOpt;
  AnalysisDiagClients AnalysisDiagOpt;
  std::string AnalyzeSpecificFunction;
  unsigned AnalyzeAll : 1;
  unsigned AnalyzerDisplayProgress : 1;
  unsigned EagerlyAssume : 1;
  unsigned PurgeDead : 1;
  unsigned TrimGraph : 1;
  unsigned VisualizeEGDot : 1;
  unsigned VisualizeEGUbi : 1;
  unsigned EnableExperimentalChecks : 1;
  unsigned EnableExperimentalInternalChecks : 1;
public:
  AnalyzerOptions() {
    AnalysisStoreOpt = BasicStoreModel;
    AnalysisConstraintsOpt = RangeConstraintsModel;
    AnalysisDiagOpt = PD_HTML;
    AnalyzeAll = 0;
    AnalyzerDisplayProgress = 0;
    EagerlyAssume = 0;
    PurgeDead = 0;
    TrimGraph = 0;
    VisualizeEGDot = 0;
    VisualizeEGUbi = 0;
    EnableExperimentalChecks = 0;
    EnableExperimentalInternalChecks = 0;
  }
};

/// CreateAnalysisConsumer - Creates an ASTConsumer to run various code
/// analysis passes.  (The set of analyses run is controlled by command-line
/// options.)
ASTConsumer* CreateAnalysisConsumer(const Preprocessor &pp,
                                    const std::string &output,
                                    const AnalyzerOptions& Opts);

}

#endif
