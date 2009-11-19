//===--- CompilerInvocation.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;

void CompilerInvocation::CreateFromArgs(CompilerInvocation &Res,
                           const llvm::SmallVectorImpl<llvm::StringRef> &Args) {
}

static const char *getAnalysisName(Analyses Kind) {
  switch (Kind) {
  default:
    llvm::llvm_unreachable("Unknown analysis store!");
#define ANALYSIS(NAME, CMDFLAG, DESC, SCOPE)\
  case NAME: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

static const char *getAnalysisStoreName(AnalysisStores Kind) {
  switch (Kind) {
  default:
    llvm::llvm_unreachable("Unknown analysis store!");
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN) \
  case NAME##Model: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

static const char *getAnalysisConstraintName(AnalysisConstraints Kind) {
  switch (Kind) {
  default:
    llvm::llvm_unreachable("Unknown analysis constraints!");
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN) \
  case NAME##Model: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

static const char *getAnalysisDiagClientName(AnalysisDiagClients Kind) {
  switch (Kind) {
  default:
    llvm::llvm_unreachable("Unknown analysis client!");
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN, AUTOCREATE) \
  case PD_##NAME: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

static void AnalyzerOptsToArgs(const AnalyzerOptions &Opts,
                               std::vector<std::string> &Res) {
  for (unsigned i = 0, e = Opts.AnalysisList.size(); i != e; ++i)
    Res.push_back(getAnalysisName(Opts.AnalysisList[i]));
  if (Opts.AnalysisStoreOpt != BasicStoreModel) {
    Res.push_back("-analyzer-store");
    Res.push_back(getAnalysisStoreName(Opts.AnalysisStoreOpt));
  }
  if (Opts.AnalysisConstraintsOpt != RangeConstraintsModel) {
    Res.push_back("-analyzer-constraints");
    Res.push_back(getAnalysisConstraintName(Opts.AnalysisConstraintsOpt));
  }
  if (Opts.AnalysisDiagOpt != PD_HTML) {
    Res.push_back("-analyzer-output");
    Res.push_back(getAnalysisDiagClientName(Opts.AnalysisDiagOpt));
  }
  if (!Opts.AnalyzeSpecificFunction.empty()) {
    Res.push_back("-analyze-function");
    Res.push_back(Opts.AnalyzeSpecificFunction);
  }
  if (Opts.AnalyzeAll)
    Res.push_back("-analyzer-opt-analyze-headers");
  if (Opts.AnalyzerDisplayProgress)
    Res.push_back("-analyzer-display-progress");
  if (Opts.EagerlyAssume)
    Res.push_back("-analyzer-eagerly-assume");
  if (!Opts.PurgeDead)
    Res.push_back("-analyzer-no-purge-dead");
  if (Opts.TrimGraph)
    Res.push_back("-trim-egraph");
  if (Opts.VisualizeEGDot)
    Res.push_back("-analyzer-viz-egraph-graphviz");
  if (Opts.VisualizeEGDot)
    Res.push_back("-analyzer-viz-egraph-ubigraph");
  if (Opts.EnableExperimentalChecks)
    Res.push_back("-analyzer-experimental-checks");
  if (Opts.EnableExperimentalInternalChecks)
    Res.push_back("-analyzer-experimental-internal-checls");
}

static void CodeGenOptsToArgs(const CodeGenOptions &Opts,
                              std::vector<std::string> &Res) {
  if (Opts.DebugInfo)
    Res.push_back("-g");
  if (Opts.DisableLLVMOpts)
    Res.push_back("-disable-llvm-optzns");
  if (Opts.DisableRedZone)
    Res.push_back("-disable-red-zone");
  if (!Opts.MergeAllConstants)
    Res.push_back("-fno-merge-all-constants");
  // NoCommon is only derived.
  if (Opts.NoImplicitFloat)
    Res.push_back("-no-implicit-float");
  if (Opts.OptimizeSize) {
    assert(Opts.OptimizationLevel == 2 && "Invalid options!");
    Res.push_back("-Os");
  } else if (Opts.OptimizationLevel == 0)
    Res.push_back("-O" + Opts.OptimizationLevel);
  // SimplifyLibCalls is only derived.
  // TimePasses is only derived.
  // UnitAtATime is unused.
  // UnrollLoops is only derived.
  // VerifyModule is only derived.
  // Inlining is only derived.
}

static void DependencyOutputOptsToArgs(const DependencyOutputOptions &Opts,
                                       std::vector<std::string> &Res) {
  if (Opts.IncludeSystemHeaders)
    Res.push_back("-sys-header-deps");
  if (Opts.UsePhonyTargets)
    Res.push_back("-MP");
  if (!Opts.OutputFile.empty()) {
    Res.push_back("-dependency-file");
    Res.push_back(Opts.OutputFile);
  }
  for (unsigned i = 0, e = Opts.Targets.size(); i != e; ++i) {
    Res.push_back("-MT");
    Res.push_back(Opts.Targets[i]);
  }
}

static void DiagnosticOptsToArgs(const DiagnosticOptions &Opts,
                                 std::vector<std::string> &Res) {
  if (Opts.IgnoreWarnings)
    Res.push_back("-w");
  if (Opts.NoRewriteMacros)
    Res.push_back("-Wno-rewrite-macros");
  if (Opts.Pedantic)
    Res.push_back("-pedantic");
  if (Opts.PedanticErrors)
    Res.push_back("-pedantic-errors");
  if (!Opts.ShowColumn)
    Res.push_back("-fno-show-column");
  if (!Opts.ShowLocation)
    Res.push_back("-fno-show-source-location");
  if (!Opts.ShowCarets)
    Res.push_back("-fno-caret-diagnostics");
  if (!Opts.ShowFixits)
    Res.push_back("-fno-diagnostics-fixit-info");
  if (Opts.ShowSourceRanges)
    Res.push_back("-fdiagnostics-print-source-range-info");
  if (Opts.ShowColors)
    Res.push_back("-fcolor-diagnostics");
  if (Opts.VerifyDiagnostics)
    Res.push_back("-verify");
  if (Opts.ShowOptionNames)
    Res.push_back("-fdiagnostics-show-option");
  if (Opts.MessageLength) {
    Res.push_back("-fmessage-length");
    Res.push_back(llvm::utostr(Opts.MessageLength));
  }
  if (!Opts.DumpBuildInformation.empty()) {
    Res.push_back("-dump-build-information");
    Res.push_back(Opts.DumpBuildInformation);
  }
  for (unsigned i = 0, e = Opts.Warnings.size(); i != e; ++i)
    Res.push_back("-W" + Opts.Warnings[i]);
}

static const char *getInputKindName(FrontendOptions::InputKind Kind) {
  switch (Kind) {
  case FrontendOptions::IK_None: break;
  case FrontendOptions::IK_AST: return "ast";
  case FrontendOptions::IK_Asm: return "assembler-with-cpp";
  case FrontendOptions::IK_C: return "c";
  case FrontendOptions::IK_CXX: return "c++";
  case FrontendOptions::IK_ObjC: return "objective-c";
  case FrontendOptions::IK_ObjCXX: return "objective-c++";
  case FrontendOptions::IK_OpenCL: return "cl";
  case FrontendOptions::IK_PreprocessedC: return "cpp-output";
  case FrontendOptions::IK_PreprocessedCXX: return "c++-cpp-output";
  case FrontendOptions::IK_PreprocessedObjC: return "objective-c-cpp-output";
  case FrontendOptions::IK_PreprocessedObjCXX: return "objective-c++-cpp-output";
  }

  llvm::llvm_unreachable("Unexpected language kind!");
  return 0;
}

static const char *getActionName(frontend::ActionKind Kind) {
  switch (Kind) {
  case frontend::PluginAction:
  case frontend::InheritanceView:
    llvm::llvm_unreachable("Invalid kind!");

  case frontend::ASTDump:                return "-ast-dump";
  case frontend::ASTPrint:               return "-ast-print";
  case frontend::ASTPrintXML:            return "-ast-print-xml";
  case frontend::ASTView:                return "-ast-view";
  case frontend::DumpRawTokens:          return "-dump-raw-tokens";
  case frontend::DumpRecordLayouts:      return "-dump-record-layouts";
  case frontend::DumpTokens:             return "-dump-tokens";
  case frontend::EmitAssembly:           return "-S";
  case frontend::EmitBC:                 return "-emit-llvm-bc";
  case frontend::EmitHTML:               return "-emit-html";
  case frontend::EmitLLVM:               return "-emit-llvm";
  case frontend::EmitLLVMOnly:           return "-emit-llvm-only";
  case frontend::FixIt:                  return "-fixit";
  case frontend::GeneratePCH:            return "-emit-pch";
  case frontend::GeneratePTH:            return "-emit-pth";
  case frontend::ParseNoop:              return "-parse-noop";
  case frontend::ParsePrintCallbacks:    return "-parse-print-callbacks";
  case frontend::ParseSyntaxOnly:        return "-fsyntax-only";
  case frontend::PrintDeclContext:       return "-print-decl-contexts";
  case frontend::PrintPreprocessedInput: return "-E";
  case frontend::RewriteBlocks:          return "-rewrite-blocks";
  case frontend::RewriteMacros:          return "-rewrite-macros";
  case frontend::RewriteObjC:            return "-rewrite-objc";
  case frontend::RewriteTest:            return "-rewrite-test";
  case frontend::RunAnalysis:            return "-analyze";
  case frontend::RunPreprocessorOnly:    return "-Eonly";
  }

  llvm::llvm_unreachable("Unexpected language kind!");
  return 0;
}

static void FrontendOptsToArgs(const FrontendOptions &Opts,
                               std::vector<std::string> &Res) {
  if (!Opts.DebugCodeCompletionPrinter)
    Res.push_back("-no-code-completion-debug-printer");
  if (Opts.DisableFree)
    Res.push_back("-disable-free");
  if (Opts.EmptyInputOnly)
    Res.push_back("-empty-input-only");
  if (Opts.RelocatablePCH)
    Res.push_back("-relocatable-pch");
  if (Opts.ShowMacrosInCodeCompletion)
    Res.push_back("-code-completion-macros");
  if (Opts.ShowStats)
    Res.push_back("-stats");
  if (Opts.ShowTimers)
    Res.push_back("-ftime-report");

  bool NeedLang = false;
  for (unsigned i = 0, e = Opts.Inputs.size(); i != e; ++i)
    if (FrontendOptions::getInputKindForExtension(Opts.Inputs[i].second) !=
        Opts.Inputs[i].first)
      NeedLang = true;
  if (NeedLang) {
    Res.push_back("-x");
    Res.push_back(getInputKindName(Opts.Inputs[0].first));
  }
  for (unsigned i = 0, e = Opts.Inputs.size(); i != e; ++i) {
    assert((!NeedLang || Opts.Inputs[i].first == Opts.Inputs[0].first) &&
           "Unable to represent this input vector!");
    Res.push_back(Opts.Inputs[i].second);
  }

  if (!Opts.OutputFile.empty()) {
    Res.push_back("-o");
    Res.push_back(Opts.OutputFile);
  }
  if (!Opts.ViewClassInheritance.empty()) {
    Res.push_back("-cxx-inheritance-view");
    Res.push_back(Opts.ViewClassInheritance);
  }
  for (unsigned i = 0, e = Opts.FixItLocations.size(); i != e; ++i) {
    Res.push_back("-fixit-at");
    Res.push_back(Opts.FixItLocations[i].FileName + ":" +
                  llvm::utostr(Opts.FixItLocations[i].Line) + ":" +
                  llvm::utostr(Opts.FixItLocations[i].Column));
  }
  if (!Opts.CodeCompletionAt.FileName.empty()) {
    Res.push_back("-code-completion-at");
    Res.push_back(Opts.CodeCompletionAt.FileName + ":" +
                  llvm::utostr(Opts.CodeCompletionAt.Line) + ":" +
                  llvm::utostr(Opts.CodeCompletionAt.Column));
  }
  if (Opts.ProgramAction != frontend::InheritanceView &&
      Opts.ProgramAction != frontend::PluginAction)
    Res.push_back(getActionName(Opts.ProgramAction));
  if (!Opts.ActionName.empty()) {
    Res.push_back("-plugin");
    Res.push_back(Opts.ActionName);
  }
}

static void HeaderSearchOptsToArgs(const HeaderSearchOptions &Opts,
                                   std::vector<std::string> &Res) {
  if (Opts.Sysroot != "/") {
    Res.push_back("-isysroot");
    Res.push_back(Opts.Sysroot);
  }

  /// User specified include entries.
  for (unsigned i = 0, e = Opts.UserEntries.size(); i != e; ++i) {
    const HeaderSearchOptions::Entry &E = Opts.UserEntries[i];
    if (E.IsFramework && (E.Group != frontend::Angled || E.IsUserSupplied))
      llvm::llvm_report_error("Invalid option set!");
    if (E.IsUserSupplied) {
      if (E.Group == frontend::After) {
        Res.push_back("-idirafter");
      } else if (E.Group == frontend::Quoted) {
        Res.push_back("-iquoted");
      } else if (E.Group == frontend::System) {
        Res.push_back("-isystem");
      } else {
        assert(E.Group == frontend::Angled && "Invalid group!");
        Res.push_back(E.IsFramework ? "-F" : "-I");
      }
    } else {
      if (E.Group != frontend::Angled && E.Group != frontend::System)
        llvm::llvm_report_error("Invalid option set!");
      Res.push_back(E.Group == frontend::Angled ? "-iwithprefixbefore" :
                    "-iwithprefix");
    }
    Res.push_back(E.Path);
  }

  if (!Opts.EnvIncPath.empty()) {
    // FIXME: Provide an option for this, and move env detection to driver.
    llvm::llvm_report_error("Not yet implemented!");
  }
  if (!Opts.CEnvIncPath.empty()) {
    // FIXME: Provide an option for this, and move env detection to driver.
    llvm::llvm_report_error("Not yet implemented!");
  }
  if (!Opts.ObjCEnvIncPath.empty()) {
    // FIXME: Provide an option for this, and move env detection to driver.
    llvm::llvm_report_error("Not yet implemented!");
  }
  if (!Opts.CXXEnvIncPath.empty()) {
    // FIXME: Provide an option for this, and move env detection to driver.
    llvm::llvm_report_error("Not yet implemented!");
  }
  if (!Opts.ObjCXXEnvIncPath.empty()) {
    // FIXME: Provide an option for this, and move env detection to driver.
    llvm::llvm_report_error("Not yet implemented!");
  }
  if (!Opts.BuiltinIncludePath.empty()) {
    // FIXME: Provide an option for this, and move to driver.
  }
  if (!Opts.UseStandardIncludes)
    Res.push_back("-nostdinc");
  if (Opts.Verbose)
    Res.push_back("-v");
}

static void LangOptsToArgs(const LangOptions &Opts,
                           std::vector<std::string> &Res) {
  LangOptions DefaultLangOpts;

  // FIXME: Need to set -std to get all the implicit options.

  // FIXME: We want to only pass options relative to the defaults, which
  // requires constructing a target. :(
  //
  // It would be better to push the all target specific choices into the driver,
  // so that everything below that was more uniform.

  if (Opts.Trigraphs)
    Res.push_back("-trigraphs");
  // Implicit based on the input kind:
  //   AsmPreprocessor, CPlusPlus, ObjC1, ObjC2, OpenCL
  // Implicit based on the input language standard:
  //   BCPLComment, C99, CPlusPlus0x, Digraphs, GNUInline, ImplicitInt, GNUMode
  if (Opts.DollarIdents)
    Res.push_back("-fdollars-in-identifiers");
  if (Opts.Microsoft)
    Res.push_back("-fms-extensions=1");
  if (Opts.ObjCNonFragileABI)
    Res.push_back("-fobjc-nonfragile-abi");
  // NoInline is implicit.
  if (!Opts.CXXOperatorNames)
    Res.push_back("-fno-operator-names");
  if (Opts.PascalStrings)
    Res.push_back("-fpascal-strings");
  if (Opts.WritableStrings)
    Res.push_back("-fwritable-strings");
  if (!Opts.LaxVectorConversions)
    Res.push_back("-fno-lax-vector-conversions");
  if (Opts.AltiVec)
    Res.push_back("-faltivec");
  Res.push_back("-fexceptions");
  Res.push_back(Opts.Exceptions ? "1" : "0");
  if (!Opts.Rtti)
    Res.push_back("-fno-rtti");
  if (!Opts.NeXTRuntime)
    Res.push_back("-fgnu-runtime");
  if (Opts.Freestanding)
    Res.push_back("-ffreestanding");
  if (Opts.NoBuiltin)
    Res.push_back("-fno-builtin");
  if (Opts.ThreadsafeStatics)
    llvm::llvm_report_error("FIXME: Not yet implemented!");
  if (Opts.POSIXThreads)
    Res.push_back("-pthread");
  if (Opts.Blocks)
    Res.push_back("-fblocks=1");
  if (Opts.EmitAllDecls)
    Res.push_back("-femit-all-decls");
  if (!Opts.MathErrno)
    Res.push_back("-fno-math-errno");
  if (Opts.OverflowChecking)
    Res.push_back("-ftrapv");
  if (Opts.HeinousExtensions)
    Res.push_back("-fheinous-gnu-extensions");
  // Optimize is implicit.
  // OptimizeSize is implicit.
  if (Opts.Static)
    Res.push_back("-static-define");
  if (Opts.PICLevel) {
    Res.push_back("-pic-level");
    Res.push_back(llvm::utostr(Opts.PICLevel));
  }
  if (Opts.ObjCGCBitmapPrint)
    Res.push_back("-print-ivar-layout");
  Res.push_back("-faccess-control");
  Res.push_back(Opts.AccessControl ? "1" : "0");
  Res.push_back("-fsigned-char");
  Res.push_back(Opts.CharIsSigned ? "1" : "0");
  Res.push_back("-fshort-wchar");
  Res.push_back(Opts.ShortWChar ? "1" : "0");
  if (!Opts.ElideConstructors)
    Res.push_back("-fno-elide-constructors");
  if (Opts.getGCMode() != LangOptions::NonGC) {
    if (Opts.getGCMode() == LangOptions::HybridGC) {
      Res.push_back("-fobjc-gc");
    } else {
      assert(Opts.getGCMode() == LangOptions::GCOnly && "Invalid GC mode!");
      Res.push_back("-fobjc-gc-only");
    }
  }
  if (Opts.getVisibilityMode() != LangOptions::Default) {
    Res.push_back("-fvisibility");
    if (Opts.getVisibilityMode() == LangOptions::Hidden) {
      Res.push_back("default");
    } else {
      assert(Opts.getVisibilityMode() == LangOptions::Protected &&
             "Invalid visibility!");
      Res.push_back("protected");
    }
  }
  if (Opts.getStackProtectorMode() != 0) {
    Res.push_back("-stack-protector");
    Res.push_back(llvm::utostr(Opts.getStackProtectorMode()));
  }
  if (Opts.getMainFileName()) {
    Res.push_back("-main-file-name");
    Res.push_back(Opts.getMainFileName());
  }
  if (Opts.InstantiationDepth != DefaultLangOpts.InstantiationDepth) {
    Res.push_back("-ftemplate-depth");
    Res.push_back(llvm::utostr(Opts.InstantiationDepth));
  }
  if (Opts.ObjCConstantStringClass) {
    Res.push_back("-fconstant-string-class");
    Res.push_back(Opts.ObjCConstantStringClass);
  }
}

static void PreprocessorOptsToArgs(const PreprocessorOptions &Opts,
                                   std::vector<std::string> &Res) {
  for (unsigned i = 0, e = Opts.Macros.size(); i != e; ++i)
    Res.push_back((Opts.Macros[i].second ? "-U" : "-D") + Opts.Macros[i].first);
  for (unsigned i = 0, e = Opts.Includes.size(); i != e; ++i) {
    Res.push_back("-include");
    Res.push_back(Opts.Includes[i]);
  }
  for (unsigned i = 0, e = Opts.MacroIncludes.size(); i != e; ++i) {
    Res.push_back("-imacros");
    Res.push_back(Opts.Includes[i]);
  }
  if (!Opts.UsePredefines)
    Res.push_back("-undef");
  if (!Opts.ImplicitPCHInclude.empty()) {
    Res.push_back("-implicit-pch-include");
    Res.push_back(Opts.ImplicitPCHInclude);
  }
  if (!Opts.ImplicitPTHInclude.empty()) {
    Res.push_back("-implicit-pth-include");
    Res.push_back(Opts.ImplicitPTHInclude);
  }
  if (!Opts.TokenCache.empty()) {
    Res.push_back("-token-cache");
    Res.push_back(Opts.TokenCache);
  }
}

static void PreprocessorOutputOptsToArgs(const PreprocessorOutputOptions &Opts,
                                         std::vector<std::string> &Res) {
  if (!Opts.ShowCPP && !Opts.ShowMacros)
    llvm::llvm_report_error("Invalid option combination!");

  if (Opts.ShowCPP && Opts.ShowMacros)
    Res.push_back("-dD");
  else if (!Opts.ShowCPP && Opts.ShowMacros)
    Res.push_back("-dM");

  if (!Opts.ShowLineMarkers)
    Res.push_back("-P");
  if (Opts.ShowComments)
    Res.push_back("-C");
  if (Opts.ShowMacroComments)
    Res.push_back("-CC");
}

static void TargetOptsToArgs(const TargetOptions &Opts,
                             std::vector<std::string> &Res) {
  Res.push_back("-triple");
  Res.push_back(Opts.Triple);
  if (!Opts.CPU.empty()) {
    Res.push_back("-target-cpu");
    Res.push_back(Opts.CPU);
  }
  if (!Opts.ABI.empty()) {
    Res.push_back("-target-abi");
    Res.push_back(Opts.ABI);
  }
  for (unsigned i = 0, e = Opts.Features.size(); i != e; ++i) {
    Res.push_back("-target-feature");
    Res.push_back(Opts.Features[i]);
  }
}

void CompilerInvocation::toArgs(std::vector<std::string> &Res) {
  AnalyzerOptsToArgs(getAnalyzerOpts(), Res);
  CodeGenOptsToArgs(getCodeGenOpts(), Res);
  DependencyOutputOptsToArgs(getDependencyOutputOpts(), Res);
  DiagnosticOptsToArgs(getDiagnosticOpts(), Res);
  FrontendOptsToArgs(getFrontendOpts(), Res);
  HeaderSearchOptsToArgs(getHeaderSearchOpts(), Res);
  LangOptsToArgs(getLangOpts(), Res);
  PreprocessorOptsToArgs(getPreprocessorOpts(), Res);
  PreprocessorOutputOptsToArgs(getPreprocessorOutputOpts(), Res);
  TargetOptsToArgs(getTargetOpts(), Res);
}
