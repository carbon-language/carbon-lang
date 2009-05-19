//===--- AnalysisConsumer.h - Front-end hooks for the analysis engine------===//
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

#include <string>
#include <vector>

namespace clang {
class ASTConsumer;
class Diagnostic;
class Preprocessor;
class PreprocessorFactory;
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

struct AnalyzerOptions {
  std::vector<Analyses> AnalysisList;
  AnalysisStores AnalysisStoreOpt;
  AnalysisConstraints AnalysisConstraintsOpt;
  AnalysisDiagClients AnalysisDiagOpt;
  bool VisualizeEGDot;
  bool VisualizeEGUbi;
  bool AnalyzeAll;
  bool AnalyzerDisplayProgress;
  bool PurgeDead;
  bool EagerlyAssume;
  std::string AnalyzeSpecificFunction;
  bool TrimGraph;
};

/// CreateAnalysisConsumer - Creates an ASTConsumer to run various code
/// analysis passes.  (The set of analyses run is controlled by command-line
/// options.)
ASTConsumer* CreateAnalysisConsumer(Diagnostic &diags, Preprocessor *pp,
                                    PreprocessorFactory *ppf,
                                    const LangOptions &lopts,
                                    const std::string &output,
                                    const AnalyzerOptions& Opts);

}
