//===--- CC1Options.cpp - Clang CC1 Options Table -----------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/CC1Options.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Option.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Host.h"

using namespace clang::driver;
using namespace clang::driver::options;
using namespace clang::driver::cc1options;

static OptTable::Info CC1InfoTable[] = {
#define OPTION(NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { NAME, HELPTEXT, METAVAR, Option::KIND##Class, FLAGS, PARAM, \
    OPT_##GROUP, OPT_##ALIAS },
#include "clang/Driver/CC1Options.inc"
};

namespace {

class CC1OptTable : public OptTable {
public:
  CC1OptTable()
    : OptTable(CC1InfoTable, sizeof(CC1InfoTable) / sizeof(CC1InfoTable[0])) {}
};

}

OptTable *clang::driver::createCC1OptTable() {
  return new CC1OptTable();
}

//

using namespace clang;

static llvm::StringRef getLastArgValue(ArgList &Args, cc1options::ID ID,
                                       llvm::StringRef Default = "") {
  if (Arg *A = Args.getLastArg(ID))
    return A->getValue(Args);
  return Default;
}

static int getLastArgIntValue(ArgList &Args, cc1options::ID ID,
                              int Default = 0) {
  Arg *A = Args.getLastArg(ID);
  if (!A)
    return Default;

  int Res = Default;
  // FIXME: What to do about argument parsing errors?
  if (llvm::StringRef(A->getValue(Args)).getAsInteger(10, Res))
    llvm::errs() << "error: invalid integral argument in '"
                 << A->getAsString(Args) << "'\n";

  return Res;
}

static std::vector<std::string>
getAllArgValues(ArgList &Args, cc1options::ID ID) {
  llvm::SmallVector<const char *, 16> Values;
  Args.AddAllArgValues(Values, ID);
  return std::vector<std::string>(Values.begin(), Values.end());
}

//

static void ParseAnalyzerArgs(AnalyzerOptions &Opts, ArgList &Args) {
  Opts.AnalysisList.clear();
#define ANALYSIS(NAME, CMDFLAG, DESC, SCOPE) \
  if (Args.hasArg(cc1options::OPT_analysis_##NAME)) \
    Opts.AnalysisList.push_back(NAME);
#include "clang/Frontend/Analyses.def"

  if (Arg *A = Args.getLastArg(cc1options::OPT_analyzer_store)) {
    llvm::StringRef Name = A->getValue(Args);
    AnalysisStores Value = llvm::StringSwitch<AnalysisStores>(Name)
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, NAME##Model)
#include "clang/Frontend/Analyses.def"
      .Default(NumStores);
    // FIXME: Error handling.
    if (Value == NumStores)
      llvm::errs() << "error: invalid analysis store '" << Name << "'\n";
    else
      Opts.AnalysisStoreOpt = Value;
  }

  if (Arg *A = Args.getLastArg(cc1options::OPT_analyzer_constraints)) {
    llvm::StringRef Name = A->getValue(Args);
    AnalysisConstraints Value = llvm::StringSwitch<AnalysisConstraints>(Name)
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, NAME##Model)
#include "clang/Frontend/Analyses.def"
      .Default(NumConstraints);
    // FIXME: Error handling.
    if (Value == NumConstraints)
      llvm::errs() << "error: invalid analysis constraints '" << Name << "'\n";
    else
      Opts.AnalysisConstraintsOpt = Value;
  }

  if (Arg *A = Args.getLastArg(cc1options::OPT_analyzer_output)) {
    llvm::StringRef Name = A->getValue(Args);
    AnalysisDiagClients Value = llvm::StringSwitch<AnalysisDiagClients>(Name)
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN, AUTOCREAT) \
      .Case(CMDFLAG, PD_##NAME)
#include "clang/Frontend/Analyses.def"
      .Default(NUM_ANALYSIS_DIAG_CLIENTS);
    // FIXME: Error handling.
    if (Value == NUM_ANALYSIS_DIAG_CLIENTS)
      llvm::errs() << "error: invalid analysis output '" << Name << "'\n";
    else
      Opts.AnalysisDiagOpt = Value;
  }

  Opts.VisualizeEGDot =
    Args.hasArg(cc1options::OPT_analyzer_viz_egraph_graphviz);
  Opts.VisualizeEGUbi =
    Args.hasArg(cc1options::OPT_analyzer_viz_egraph_ubigraph);
  Opts.AnalyzeAll = Args.hasArg(cc1options::OPT_analyzer_opt_analyze_headers);
  Opts.AnalyzerDisplayProgress =
    Args.hasArg(cc1options::OPT_analyzer_display_progress);
  Opts.PurgeDead = !Args.hasArg(cc1options::OPT_analyzer_no_purge_dead);
  Opts.EagerlyAssume = Args.hasArg(cc1options::OPT_analyzer_eagerly_assume);
  Opts.AnalyzeSpecificFunction =
    getLastArgValue(Args, cc1options::OPT_analyze_function);
  Opts.EnableExperimentalChecks =
    Args.hasArg(cc1options::OPT_analyzer_experimental_checks);
  Opts.EnableExperimentalInternalChecks =
    Args.hasArg(cc1options::OPT_analyzer_experimental_internal_checks);
  Opts.TrimGraph = Args.hasArg(cc1options::OPT_trim_egraph);
}

static void ParseCodeGenArgs(CodeGenOptions &Opts, ArgList &Args) {
  // -Os implies -O2
  if (Args.hasArg(cc1options::OPT_Os))
    Opts.OptimizationLevel = 2;
  else
    Opts.OptimizationLevel = getLastArgIntValue(Args, cc1options::OPT_O);

  // FIXME: What to do about argument parsing errors?
  if (Opts.OptimizationLevel > 3) {
    llvm::errs() << "error: invalid optimization level '"
                 << Opts.OptimizationLevel << "' (out of range)\n";
    Opts.OptimizationLevel = 3;
  }

  // We must always run at least the always inlining pass.
  Opts.Inlining = (Opts.OptimizationLevel > 1) ? CodeGenOptions::NormalInlining
    : CodeGenOptions::OnlyAlwaysInlining;

  Opts.DebugInfo = Args.hasArg(cc1options::OPT_g);
  Opts.DisableLLVMOpts = Args.hasArg(cc1options::OPT_disable_llvm_optzns);
  Opts.DisableRedZone = Args.hasArg(cc1options::OPT_disable_red_zone);
  Opts.MergeAllConstants = !Args.hasArg(cc1options::OPT_fno_merge_all_constants);
  Opts.NoCommon = Args.hasArg(cc1options::OPT_fno_common);
  Opts.NoImplicitFloat = Args.hasArg(cc1options::OPT_no_implicit_float);
  Opts.OptimizeSize = Args.hasArg(cc1options::OPT_Os);
  Opts.SimplifyLibCalls = 1;
  Opts.UnrollLoops = (Opts.OptimizationLevel > 1 && !Opts.OptimizeSize);

  // FIXME: Implement!
  // FIXME: Eliminate this dependency?
//   if (Lang.NoBuiltin)
//     Opts.SimplifyLibCalls = 0;
//   if (Lang.CPlusPlus)
//     Opts.NoCommon = 1;
//   Opts.TimePasses = TimePasses;

  // FIXME: Put elsewhere?
#ifdef NDEBUG
  Opts.VerifyModule = 0;
#endif
}

static void ParseDependencyOutputArgs(DependencyOutputOptions &Opts,
                                         ArgList &Args) {
  Opts.OutputFile = getLastArgValue(Args, cc1options::OPT_dependency_file);
  Opts.Targets = getAllArgValues(Args, cc1options::OPT_MT);
  Opts.IncludeSystemHeaders = Args.hasArg(cc1options::OPT_sys_header_deps);
  Opts.UsePhonyTargets = Args.hasArg(cc1options::OPT_MP);
}

static void ParseTargetArgs(TargetOptions &Opts, ArgList &Args) {
  Opts.ABI = getLastArgValue(Args, cc1options::OPT_target_abi);
  Opts.CPU = getLastArgValue(Args, cc1options::OPT_mcpu);
  Opts.Triple = getLastArgValue(Args, cc1options::OPT_triple);
  Opts.Features = getAllArgValues(Args, cc1options::OPT_target_feature);

  // Use the host triple if unspecified.
  if (Opts.Triple.empty())
    Opts.Triple = llvm::sys::getHostTriple();
}

//

void CompilerInvocation::CreateFromArgs(CompilerInvocation &Res,
                                        const char **ArgBegin,
                                        const char **ArgEnd) {
  // Parse the arguments.
  llvm::OwningPtr<OptTable> Opts(createCC1OptTable());
  unsigned MissingArgIndex, MissingArgCount;
  llvm::OwningPtr<InputArgList> InputArgs(
    Opts->ParseArgs(ArgBegin, ArgEnd,MissingArgIndex, MissingArgCount));

  // Check for missing argument error.
  if (MissingArgCount) {
    // FIXME: Use proper diagnostics!
    llvm::errs() << "error: argument to '"
                 << InputArgs->getArgString(MissingArgIndex)
                 << "' is missing (expected " << MissingArgCount
                 << " value )\n";
  }

  ParseAnalyzerArgs(Res.getAnalyzerOpts(), *InputArgs);
  ParseCodeGenArgs(Res.getCodeGenOpts(), *InputArgs);
  ParseDependencyOutputArgs(Res.getDependencyOutputOpts(), *InputArgs);
  ParseTargetArgs(Res.getTargetOpts(), *InputArgs);
}
