//===--- AnalyzerOptions.h - Analysis Engine Options ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header contains the structures necessary for a front-end to specify
// various analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_ANALYZEROPTIONS_H
#define LLVM_CLANG_FRONTEND_ANALYZEROPTIONS_H

#include <string>
#include <vector>
#include "llvm/ADT/StringMap.h"

namespace clang {
class ASTConsumer;
class DiagnosticsEngine;
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

/// AnalysisPurgeModes - Set of available strategies for dead symbol removal.
enum AnalysisPurgeMode {
#define ANALYSIS_PURGE(NAME, CMDFLAG, DESC) NAME,
#include "clang/Frontend/Analyses.def"
NumPurgeModes
};

/// AnalysisIPAMode - Set of inter-procedural modes.
enum AnalysisIPAMode {
#define ANALYSIS_IPA(NAME, CMDFLAG, DESC) NAME,
#include "clang/Frontend/Analyses.def"
NumIPAModes
};

/// AnalysisInlineFunctionSelection - Set of inlining function selection heuristics.
enum AnalysisInliningMode {
#define ANALYSIS_INLINING_MODE(NAME, CMDFLAG, DESC) NAME,
#include "clang/Frontend/Analyses.def"
NumInliningModes
};

class AnalyzerOptions {
public:
  /// \brief Pair of checker name and enable/disable.
  std::vector<std::pair<std::string, bool> > CheckersControlList;
  llvm::StringMap<std::string> Config;
  AnalysisStores AnalysisStoreOpt;
  AnalysisConstraints AnalysisConstraintsOpt;
  AnalysisDiagClients AnalysisDiagOpt;
  AnalysisPurgeMode AnalysisPurgeOpt;
  AnalysisIPAMode IPAMode;
  std::string AnalyzeSpecificFunction;
  unsigned MaxNodes;
  unsigned MaxLoop;
  unsigned ShowCheckerHelp : 1;
  unsigned AnalyzeAll : 1;
  unsigned AnalyzerDisplayProgress : 1;
  unsigned AnalyzeNestedBlocks : 1;
  unsigned EagerlyAssume : 1;
  unsigned TrimGraph : 1;
  unsigned VisualizeEGDot : 1;
  unsigned VisualizeEGUbi : 1;
  unsigned UnoptimizedCFG : 1;
  unsigned CFGAddImplicitDtors : 1;
  unsigned EagerlyTrimEGraph : 1;
  unsigned PrintStats : 1;
  unsigned NoRetryExhausted : 1;
  unsigned InlineMaxStackDepth;
  unsigned InlineMaxFunctionSize;
  AnalysisInliningMode InliningMode;

public:
  AnalyzerOptions() {
    AnalysisStoreOpt = RegionStoreModel;
    AnalysisConstraintsOpt = RangeConstraintsModel;
    AnalysisDiagOpt = PD_HTML;
    AnalysisPurgeOpt = PurgeStmt;
    IPAMode = BasicInlining;
    ShowCheckerHelp = 0;
    AnalyzeAll = 0;
    AnalyzerDisplayProgress = 0;
    AnalyzeNestedBlocks = 0;
    EagerlyAssume = 0;
    TrimGraph = 0;
    VisualizeEGDot = 0;
    VisualizeEGUbi = 0;
    UnoptimizedCFG = 0;
    CFGAddImplicitDtors = 0;
    EagerlyTrimEGraph = 0;
    PrintStats = 0;
    NoRetryExhausted = 0;
    // Cap the stack depth at 4 calls (5 stack frames, base + 4 calls).
    InlineMaxStackDepth = 5;
    InlineMaxFunctionSize = 200;
    InliningMode = NoRedundancy;
  }
};

}

#endif
