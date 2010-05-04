//===--- CompilerInvocation.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/CC1Options.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Option.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/LangStandard.h"
#include "clang/Frontend/PCHReader.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
using namespace clang;

static const char *getAnalysisName(Analyses Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown analysis kind!");
#define ANALYSIS(NAME, CMDFLAG, DESC, SCOPE)\
  case NAME: return "-" CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

static const char *getAnalysisStoreName(AnalysisStores Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown analysis store!");
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN) \
  case NAME##Model: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

static const char *getAnalysisConstraintName(AnalysisConstraints Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown analysis constraints!");
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN) \
  case NAME##Model: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

static const char *getAnalysisDiagClientName(AnalysisDiagClients Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown analysis client!");
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN, AUTOCREATE) \
  case PD_##NAME: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

//===----------------------------------------------------------------------===//
// Serialization (to args)
//===----------------------------------------------------------------------===//

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
  if (Opts.AnalyzeNestedBlocks)
    Res.push_back("-analyzer-opt-analyze-nested-blocks");
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
    Res.push_back("-analyzer-experimental-internal-checks");
}

static void CodeGenOptsToArgs(const CodeGenOptions &Opts,
                              std::vector<std::string> &Res) {
  if (Opts.DebugInfo)
    Res.push_back("-g");
  if (Opts.DisableLLVMOpts)
    Res.push_back("-disable-llvm-optzns");
  if (Opts.DisableRedZone)
    Res.push_back("-disable-red-zone");
  if (!Opts.DwarfDebugFlags.empty()) {
    Res.push_back("-dwarf-debug-flags");
    Res.push_back(Opts.DwarfDebugFlags);
  }
  if (!Opts.MergeAllConstants)
    Res.push_back("-fno-merge-all-constants");
  if (Opts.NoCommon)
    Res.push_back("-fno-common");
  if (Opts.NoImplicitFloat)
    Res.push_back("-no-implicit-float");
  if (Opts.OptimizeSize) {
    assert(Opts.OptimizationLevel == 2 && "Invalid options!");
    Res.push_back("-Os");
  } else if (Opts.OptimizationLevel != 0)
    Res.push_back("-O" + llvm::utostr(Opts.OptimizationLevel));
  if (!Opts.MainFileName.empty()) {
    Res.push_back("-main-file-name");
    Res.push_back(Opts.MainFileName);
  }
  // SimplifyLibCalls is only derived.
  // TimePasses is only derived.
  // UnitAtATime is unused.
  // UnrollLoops is only derived.
  // Inlining is only derived.

  if (Opts.DataSections)
    Res.push_back("-fdata-sections");
  if (Opts.FunctionSections)
    Res.push_back("-ffunction-sections");
  if (Opts.AsmVerbose)
    Res.push_back("-masm-verbose");
  if (!Opts.CodeModel.empty()) {
    Res.push_back("-mcode-model");
    Res.push_back(Opts.CodeModel);
  }
  if (!Opts.CXAAtExit)
    Res.push_back("-fno-use-cxa-atexit");
  if (Opts.CXXCtorDtorAliases)
    Res.push_back("-mconstructor-aliases");
  if (!Opts.DebugPass.empty()) {
    Res.push_back("-mdebug-pass");
    Res.push_back(Opts.DebugPass);
  }
  if (Opts.DisableFPElim)
    Res.push_back("-mdisable-fp-elim");
  if (!Opts.FloatABI.empty()) {
    Res.push_back("-mfloat-abi");
    Res.push_back(Opts.FloatABI);
  }
  if (!Opts.LimitFloatPrecision.empty()) {
    Res.push_back("-mlimit-float-precision");
    Res.push_back(Opts.LimitFloatPrecision);
  }
  if (Opts.NoZeroInitializedInBSS)
    Res.push_back("-mno-zero-initialized-bss");
  switch (Opts.getObjCDispatchMethod()) {
  case CodeGenOptions::Legacy:
    break;
  case CodeGenOptions::Mixed:
    Res.push_back("-fobjc-dispatch-method=mixed");
    break;
  case CodeGenOptions::NonLegacy:
    Res.push_back("-fobjc-dispatch-method=non-legacy");
    break;
  }
  if (Opts.SoftFloat)
    Res.push_back("-msoft-float");
  if (Opts.UnwindTables)
    Res.push_back("-munwind-tables");
  if (Opts.RelocationModel != "pic") {
    Res.push_back("-mrelocation-model");
    Res.push_back(Opts.RelocationModel);
  }
  if (!Opts.VerifyModule)
    Res.push_back("-disable-llvm-verifier");
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
  if (Opts.BinaryOutput)
    Res.push_back("-fdiagnostics-binary");
  if (Opts.ShowOptionNames)
    Res.push_back("-fdiagnostics-show-option");
  if (Opts.ErrorLimit) {
    Res.push_back("-ferror-limit");
    Res.push_back(llvm::utostr(Opts.ErrorLimit));
  }
  if (Opts.MacroBacktraceLimit
                        != DiagnosticOptions::DefaultMacroBacktraceLimit) {
    Res.push_back("-fmacro-backtrace-limit");
    Res.push_back(llvm::utostr(Opts.MacroBacktraceLimit));
  }
  if (Opts.TemplateBacktraceLimit
                        != DiagnosticOptions::DefaultTemplateBacktraceLimit) {
    Res.push_back("-ftemplate-backtrace-limit");
    Res.push_back(llvm::utostr(Opts.TemplateBacktraceLimit));
  }

  if (Opts.TabStop != DiagnosticOptions::DefaultTabStop) {
    Res.push_back("-ftabstop");
    Res.push_back(llvm::utostr(Opts.TabStop));
  }
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
  case FrontendOptions::IK_None:              break;
  case FrontendOptions::IK_AST:               return "ast";
  case FrontendOptions::IK_Asm:               return "assembler-with-cpp";
  case FrontendOptions::IK_C:                 return "c";
  case FrontendOptions::IK_CXX:               return "c++";
  case FrontendOptions::IK_ObjC:              return "objective-c";
  case FrontendOptions::IK_ObjCXX:            return "objective-c++";
  case FrontendOptions::IK_OpenCL:            return "cl";
  case FrontendOptions::IK_PreprocessedC:     return "cpp-output";
  case FrontendOptions::IK_PreprocessedCXX:   return "c++-cpp-output";
  case FrontendOptions::IK_PreprocessedObjC:  return "objective-c-cpp-output";
  case FrontendOptions::IK_PreprocessedObjCXX:return "objective-c++-cpp-output";
  }

  llvm_unreachable("Unexpected language kind!");
  return 0;
}

static const char *getActionName(frontend::ActionKind Kind) {
  switch (Kind) {
  case frontend::PluginAction:
  case frontend::InheritanceView:
    llvm_unreachable("Invalid kind!");

  case frontend::ASTDump:                return "-ast-dump";
  case frontend::ASTPrint:               return "-ast-print";
  case frontend::ASTPrintXML:            return "-ast-print-xml";
  case frontend::ASTView:                return "-ast-view";
  case frontend::DumpRawTokens:          return "-dump-raw-tokens";
  case frontend::DumpTokens:             return "-dump-tokens";
  case frontend::EmitAssembly:           return "-S";
  case frontend::EmitBC:                 return "-emit-llvm-bc";
  case frontend::EmitHTML:               return "-emit-html";
  case frontend::EmitLLVM:               return "-emit-llvm";
  case frontend::EmitLLVMOnly:           return "-emit-llvm-only";
  case frontend::EmitObj:                return "-emit-obj";
  case frontend::FixIt:                  return "-fixit";
  case frontend::GeneratePCH:            return "-emit-pch";
  case frontend::GeneratePTH:            return "-emit-pth";
  case frontend::InitOnly:               return "-init-only";
  case frontend::ParseNoop:              return "-parse-noop";
  case frontend::ParsePrintCallbacks:    return "-parse-print-callbacks";
  case frontend::ParseSyntaxOnly:        return "-fsyntax-only";
  case frontend::PrintDeclContext:       return "-print-decl-contexts";
  case frontend::PrintPreprocessedInput: return "-E";
  case frontend::RewriteMacros:          return "-rewrite-macros";
  case frontend::RewriteObjC:            return "-rewrite-objc";
  case frontend::RewriteTest:            return "-rewrite-test";
  case frontend::RunAnalysis:            return "-analyze";
  case frontend::RunPreprocessorOnly:    return "-Eonly";
  }

  llvm_unreachable("Unexpected language kind!");
  return 0;
}

static void FrontendOptsToArgs(const FrontendOptions &Opts,
                               std::vector<std::string> &Res) {
  if (!Opts.DebugCodeCompletionPrinter)
    Res.push_back("-no-code-completion-debug-printer");
  if (Opts.DisableFree)
    Res.push_back("-disable-free");
  if (Opts.RelocatablePCH)
    Res.push_back("-relocatable-pch");
  if (Opts.ShowHelp)
    Res.push_back("-help");
  if (Opts.ShowMacrosInCodeCompletion)
    Res.push_back("-code-completion-macros");
  if (Opts.ShowStats)
    Res.push_back("-print-stats");
  if (Opts.ShowTimers)
    Res.push_back("-ftime-report");
  if (Opts.ShowVersion)
    Res.push_back("-version");

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
  for (unsigned i = 0, e = Opts.Plugins.size(); i != e; ++i) {
    Res.push_back("-load");
    Res.push_back(Opts.Plugins[i]);
  }
  for (unsigned i = 0, e = Opts.ASTMergeFiles.size(); i != e; ++i) {
    Res.push_back("-ast-merge");
    Res.push_back(Opts.ASTMergeFiles[i]);
  }
  for (unsigned i = 0, e = Opts.LLVMArgs.size(); i != e; ++i) {
    Res.push_back("-mllvm");
    Res.push_back(Opts.LLVMArgs[i]);
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
    if (E.IsFramework && (E.Group != frontend::Angled || !E.IsUserSupplied))
      llvm::report_fatal_error("Invalid option set!");
    if (E.IsUserSupplied) {
      if (E.Group == frontend::After) {
        Res.push_back("-idirafter");
      } else if (E.Group == frontend::Quoted) {
        Res.push_back("-iquote");
      } else if (E.Group == frontend::System) {
        Res.push_back("-isystem");
      } else {
        assert(E.Group == frontend::Angled && "Invalid group!");
        Res.push_back(E.IsFramework ? "-F" : "-I");
      }
    } else {
      if (E.Group != frontend::Angled && E.Group != frontend::System)
        llvm::report_fatal_error("Invalid option set!");
      Res.push_back(E.Group == frontend::Angled ? "-iwithprefixbefore" :
                    "-iwithprefix");
    }
    Res.push_back(E.Path);
  }

  if (!Opts.EnvIncPath.empty()) {
    // FIXME: Provide an option for this, and move env detection to driver.
    llvm::report_fatal_error("Not yet implemented!");
  }
  if (!Opts.CEnvIncPath.empty()) {
    // FIXME: Provide an option for this, and move env detection to driver.
    llvm::report_fatal_error("Not yet implemented!");
  }
  if (!Opts.ObjCEnvIncPath.empty()) {
    // FIXME: Provide an option for this, and move env detection to driver.
    llvm::report_fatal_error("Not yet implemented!");
  }
  if (!Opts.CXXEnvIncPath.empty()) {
    // FIXME: Provide an option for this, and move env detection to driver.
    llvm::report_fatal_error("Not yet implemented!");
  }
  if (!Opts.ObjCXXEnvIncPath.empty()) {
    // FIXME: Provide an option for this, and move env detection to driver.
    llvm::report_fatal_error("Not yet implemented!");
  }
  if (!Opts.ResourceDir.empty()) {
    Res.push_back("-resource-dir");
    Res.push_back(Opts.ResourceDir);
  }
  if (!Opts.UseStandardIncludes)
    Res.push_back("-nostdinc");
  if (!Opts.UseStandardCXXIncludes)
    Res.push_back("-nostdinc++");
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
  if (Opts.GNUMode && !Opts.GNUKeywords)
    Res.push_back("-fno-gnu-keywords");
  if (!Opts.GNUMode && Opts.GNUKeywords)
    Res.push_back("-fgnu-keywords");
  if (Opts.Microsoft)
    Res.push_back("-fms-extensions");
  if (Opts.ObjCNonFragileABI)
    Res.push_back("-fobjc-nonfragile-abi");
  if (Opts.ObjCNonFragileABI2)
    Res.push_back("-fobjc-nonfragile-abi2");
  // NoInline is implicit.
  if (!Opts.CXXOperatorNames)
    Res.push_back("-fno-operator-names");
  if (Opts.PascalStrings)
    Res.push_back("-fpascal-strings");
  if (Opts.CatchUndefined)
    Res.push_back("-fcatch-undefined-behavior");
  if (Opts.WritableStrings)
    Res.push_back("-fwritable-strings");
  if (Opts.ConstStrings)
    Res.push_back("-Wwrite-strings");
  if (!Opts.LaxVectorConversions)
    Res.push_back("-fno-lax-vector-conversions");
  if (Opts.AltiVec)
    Res.push_back("-faltivec");
  if (Opts.Exceptions)
    Res.push_back("-fexceptions");
  if (Opts.SjLjExceptions)
    Res.push_back("-fsjlj-exceptions");
  if (!Opts.RTTI)
    Res.push_back("-fno-rtti");
  if (!Opts.NeXTRuntime)
    Res.push_back("-fgnu-runtime");
  if (Opts.Freestanding)
    Res.push_back("-ffreestanding");
  if (Opts.NoBuiltin)
    Res.push_back("-fno-builtin");
  if (!Opts.AssumeSaneOperatorNew)
    Res.push_back("-fno-assume-sane-operator-new");
  if (!Opts.ThreadsafeStatics)
    Res.push_back("-fno-threadsafe-statics");
  if (Opts.POSIXThreads)
    Res.push_back("-pthread");
  if (Opts.Blocks)
    Res.push_back("-fblocks");
  if (Opts.EmitAllDecls)
    Res.push_back("-femit-all-decls");
  if (Opts.MathErrno)
    Res.push_back("-fmath-errno");
  if (Opts.OverflowChecking)
    Res.push_back("-ftrapv");
  if (Opts.HeinousExtensions)
    Res.push_back("-fheinous-gnu-extensions");
  // Optimize is implicit.
  // OptimizeSize is implicit.
  if (Opts.Static)
    Res.push_back("-static-define");
  if (Opts.DumpRecordLayouts)
    Res.push_back("-fdump-record-layouts");
  if (Opts.DumpVTableLayouts)
    Res.push_back("-fdump-vtable-layouts");
  if (Opts.NoBitFieldTypeAlign)
    Res.push_back("-fno-bitfield-type-alignment");
  if (Opts.SjLjExceptions)
    Res.push_back("-fsjlj-exceptions");
  if (Opts.PICLevel) {
    Res.push_back("-pic-level");
    Res.push_back(llvm::utostr(Opts.PICLevel));
  }
  if (Opts.ObjCGCBitmapPrint)
    Res.push_back("-print-ivar-layout");
  if (Opts.NoConstantCFStrings)
    Res.push_back("-fno-constant-cfstrings");
  if (!Opts.AccessControl)
    Res.push_back("-fno-access-control");
  if (!Opts.CharIsSigned)
    Res.push_back("-fno-signed-char");
  if (Opts.ShortWChar)
    Res.push_back("-fshort-wchar");
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
      Res.push_back("hidden");
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
  if (Opts.InstantiationDepth != DefaultLangOpts.InstantiationDepth) {
    Res.push_back("-ftemplate-depth");
    Res.push_back(llvm::utostr(Opts.InstantiationDepth));
  }
  if (!Opts.ObjCConstantStringClass.empty()) {
    Res.push_back("-fconstant-string-class");
    Res.push_back(Opts.ObjCConstantStringClass);
  }
}

static void PreprocessorOptsToArgs(const PreprocessorOptions &Opts,
                                   std::vector<std::string> &Res) {
  for (unsigned i = 0, e = Opts.Macros.size(); i != e; ++i)
    Res.push_back(std::string(Opts.Macros[i].second ? "-U" : "-D") +
                  Opts.Macros[i].first);
  for (unsigned i = 0, e = Opts.Includes.size(); i != e; ++i) {
    // FIXME: We need to avoid reincluding the implicit PCH and PTH includes.
    Res.push_back("-include");
    Res.push_back(Opts.Includes[i]);
  }
  for (unsigned i = 0, e = Opts.MacroIncludes.size(); i != e; ++i) {
    Res.push_back("-imacros");
    Res.push_back(Opts.MacroIncludes[i]);
  }
  if (!Opts.UsePredefines)
    Res.push_back("-undef");
  if (Opts.DetailedRecord)
    Res.push_back("-detailed-preprocessing-record");
  if (!Opts.ImplicitPCHInclude.empty()) {
    Res.push_back("-include-pch");
    Res.push_back(Opts.ImplicitPCHInclude);
  }
  if (!Opts.ImplicitPTHInclude.empty()) {
    Res.push_back("-include-pth");
    Res.push_back(Opts.ImplicitPTHInclude);
  }
  if (!Opts.TokenCache.empty()) {
    if (Opts.ImplicitPTHInclude.empty()) {
      Res.push_back("-token-cache");
      Res.push_back(Opts.TokenCache);
    } else
      assert(Opts.ImplicitPTHInclude == Opts.TokenCache &&
             "Unsupported option combination!");
  }
  for (unsigned i = 0, e = Opts.RemappedFiles.size(); i != e; ++i) {
    Res.push_back("-remap-file");
    Res.push_back(Opts.RemappedFiles[i].first + ";" +
                  Opts.RemappedFiles[i].second);
  }
}

static void PreprocessorOutputOptsToArgs(const PreprocessorOutputOptions &Opts,
                                         std::vector<std::string> &Res) {
  if (!Opts.ShowCPP && !Opts.ShowMacros)
    llvm::report_fatal_error("Invalid option combination!");

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

//===----------------------------------------------------------------------===//
// Deserialization (to args)
//===----------------------------------------------------------------------===//

using namespace clang::driver;
using namespace clang::driver::cc1options;

static llvm::StringRef getLastArgValue(ArgList &Args, cc1options::ID ID,
                                       llvm::StringRef Default = "") {
  if (Arg *A = Args.getLastArg(ID))
    return A->getValue(Args);
  return Default;
}

static int getLastArgIntValue(ArgList &Args, cc1options::ID ID,
                              int Default, Diagnostic &Diags) {
  Arg *A = Args.getLastArg(ID);
  if (!A)
    return Default;

  int Res = Default;
  if (llvm::StringRef(A->getValue(Args)).getAsInteger(10, Res))
    Diags.Report(diag::err_drv_invalid_int_value)
        << A->getAsString(Args) << A->getValue(Args);

  return Res;
}

static std::vector<std::string>
getAllArgValues(ArgList &Args, cc1options::ID ID) {
  llvm::SmallVector<const char *, 16> Values;
  Args.AddAllArgValues(Values, ID);
  return std::vector<std::string>(Values.begin(), Values.end());
}

//

static void ParseAnalyzerArgs(AnalyzerOptions &Opts, ArgList &Args,
                              Diagnostic &Diags) {
  using namespace cc1options;

  Opts.AnalysisList.clear();
#define ANALYSIS(NAME, CMDFLAG, DESC, SCOPE) \
  if (Args.hasArg(OPT_analysis_##NAME)) Opts.AnalysisList.push_back(NAME);
#include "clang/Frontend/Analyses.def"

  if (Arg *A = Args.getLastArg(OPT_analyzer_store)) {
    llvm::StringRef Name = A->getValue(Args);
    AnalysisStores Value = llvm::StringSwitch<AnalysisStores>(Name)
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, NAME##Model)
#include "clang/Frontend/Analyses.def"
      .Default(NumStores);
    // FIXME: Error handling.
    if (Value == NumStores)
      Diags.Report(diag::err_drv_invalid_value)
        << Args.getLastArg(OPT_O)->getAsString(Args) << Name;
    else
      Opts.AnalysisStoreOpt = Value;
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_constraints)) {
    llvm::StringRef Name = A->getValue(Args);
    AnalysisConstraints Value = llvm::StringSwitch<AnalysisConstraints>(Name)
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, NAME##Model)
#include "clang/Frontend/Analyses.def"
      .Default(NumConstraints);
    // FIXME: Error handling.
    if (Value == NumConstraints)
      Diags.Report(diag::err_drv_invalid_value)
        << Args.getLastArg(OPT_O)->getAsString(Args) << Name;
    else
      Opts.AnalysisConstraintsOpt = Value;
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_output)) {
    llvm::StringRef Name = A->getValue(Args);
    AnalysisDiagClients Value = llvm::StringSwitch<AnalysisDiagClients>(Name)
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN, AUTOCREAT) \
      .Case(CMDFLAG, PD_##NAME)
#include "clang/Frontend/Analyses.def"
      .Default(NUM_ANALYSIS_DIAG_CLIENTS);
    // FIXME: Error handling.
    if (Value == NUM_ANALYSIS_DIAG_CLIENTS)
      Diags.Report(diag::err_drv_invalid_value)
        << Args.getLastArg(OPT_O)->getAsString(Args) << Name;
    else
      Opts.AnalysisDiagOpt = Value;
  }

  Opts.VisualizeEGDot = Args.hasArg(OPT_analyzer_viz_egraph_graphviz);
  Opts.VisualizeEGUbi = Args.hasArg(OPT_analyzer_viz_egraph_ubigraph);
  Opts.AnalyzeAll = Args.hasArg(OPT_analyzer_opt_analyze_headers);
  Opts.AnalyzerDisplayProgress = Args.hasArg(OPT_analyzer_display_progress);
  Opts.AnalyzeNestedBlocks =
    Args.hasArg(OPT_analyzer_opt_analyze_nested_blocks);
  Opts.PurgeDead = !Args.hasArg(OPT_analyzer_no_purge_dead);
  Opts.EagerlyAssume = Args.hasArg(OPT_analyzer_eagerly_assume);
  Opts.AnalyzeSpecificFunction = getLastArgValue(Args, OPT_analyze_function);
  Opts.EnableExperimentalChecks = Args.hasArg(OPT_analyzer_experimental_checks);
  Opts.EnableExperimentalInternalChecks =
    Args.hasArg(OPT_analyzer_experimental_internal_checks);
  Opts.TrimGraph = Args.hasArg(OPT_trim_egraph);
  Opts.MaxNodes = getLastArgIntValue(Args, OPT_analyzer_max_nodes,150000,Diags);
}

static void ParseCodeGenArgs(CodeGenOptions &Opts, ArgList &Args,
                             Diagnostic &Diags) {
  using namespace cc1options;
  // -Os implies -O2
  if (Args.hasArg(OPT_Os))
    Opts.OptimizationLevel = 2;
  else {
    Opts.OptimizationLevel = getLastArgIntValue(Args, OPT_O, 0, Diags);
    if (Opts.OptimizationLevel > 3) {
      Diags.Report(diag::err_drv_invalid_value)
        << Args.getLastArg(OPT_O)->getAsString(Args) << Opts.OptimizationLevel;
      Opts.OptimizationLevel = 3;
    }
  }

  // We must always run at least the always inlining pass.
  Opts.Inlining = (Opts.OptimizationLevel > 1) ? CodeGenOptions::NormalInlining
    : CodeGenOptions::OnlyAlwaysInlining;

  Opts.DebugInfo = Args.hasArg(OPT_g);
  Opts.DisableLLVMOpts = Args.hasArg(OPT_disable_llvm_optzns);
  Opts.DisableRedZone = Args.hasArg(OPT_disable_red_zone);
  Opts.DwarfDebugFlags = getLastArgValue(Args, OPT_dwarf_debug_flags);
  Opts.MergeAllConstants = !Args.hasArg(OPT_fno_merge_all_constants);
  Opts.NoCommon = Args.hasArg(OPT_fno_common);
  Opts.NoImplicitFloat = Args.hasArg(OPT_no_implicit_float);
  Opts.OptimizeSize = Args.hasArg(OPT_Os);
  Opts.UnrollLoops = (Opts.OptimizationLevel > 1 && !Opts.OptimizeSize);

  Opts.AsmVerbose = Args.hasArg(OPT_masm_verbose);
  Opts.CXAAtExit = !Args.hasArg(OPT_fno_use_cxa_atexit);
  Opts.CXXCtorDtorAliases = Args.hasArg(OPT_mconstructor_aliases);
  Opts.CodeModel = getLastArgValue(Args, OPT_mcode_model);
  Opts.DebugPass = getLastArgValue(Args, OPT_mdebug_pass);
  Opts.DisableFPElim = Args.hasArg(OPT_mdisable_fp_elim);
  Opts.FloatABI = getLastArgValue(Args, OPT_mfloat_abi);
  Opts.LimitFloatPrecision = getLastArgValue(Args, OPT_mlimit_float_precision);
  Opts.NoZeroInitializedInBSS = Args.hasArg(OPT_mno_zero_initialized_in_bss);
  Opts.SoftFloat = Args.hasArg(OPT_msoft_float);
  Opts.UnwindTables = Args.hasArg(OPT_munwind_tables);
  Opts.RelocationModel = getLastArgValue(Args, OPT_mrelocation_model, "pic");

  Opts.FunctionSections = Args.hasArg(OPT_ffunction_sections);
  Opts.DataSections = Args.hasArg(OPT_fdata_sections);

  Opts.MainFileName = getLastArgValue(Args, OPT_main_file_name);
  Opts.VerifyModule = !Args.hasArg(OPT_disable_llvm_verifier);

  if (Arg *A = Args.getLastArg(OPT_fobjc_dispatch_method_EQ)) {
    llvm::StringRef Name = A->getValue(Args);
    unsigned Method = llvm::StringSwitch<unsigned>(Name)
      .Case("legacy", CodeGenOptions::Legacy)
      .Case("non-legacy", CodeGenOptions::NonLegacy)
      .Case("mixed", CodeGenOptions::Mixed)
      .Default(~0U);
    if (Method == ~0U)
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
    else
      Opts.ObjCDispatchMethod = Method;
  }
}

static void ParseDependencyOutputArgs(DependencyOutputOptions &Opts,
                                      ArgList &Args) {
  using namespace cc1options;
  Opts.OutputFile = getLastArgValue(Args, OPT_dependency_file);
  Opts.Targets = getAllArgValues(Args, OPT_MT);
  Opts.IncludeSystemHeaders = Args.hasArg(OPT_sys_header_deps);
  Opts.UsePhonyTargets = Args.hasArg(OPT_MP);
}

static void ParseDiagnosticArgs(DiagnosticOptions &Opts, ArgList &Args,
                                Diagnostic &Diags) {
  using namespace cc1options;
  Opts.IgnoreWarnings = Args.hasArg(OPT_w);
  Opts.NoRewriteMacros = Args.hasArg(OPT_Wno_rewrite_macros);
  Opts.Pedantic = Args.hasArg(OPT_pedantic);
  Opts.PedanticErrors = Args.hasArg(OPT_pedantic_errors);
  Opts.ShowCarets = !Args.hasArg(OPT_fno_caret_diagnostics);
  Opts.ShowColors = Args.hasArg(OPT_fcolor_diagnostics);
  Opts.ShowColumn = !Args.hasArg(OPT_fno_show_column);
  Opts.ShowFixits = !Args.hasArg(OPT_fno_diagnostics_fixit_info);
  Opts.ShowLocation = !Args.hasArg(OPT_fno_show_source_location);
  Opts.ShowOptionNames = Args.hasArg(OPT_fdiagnostics_show_option);
  Opts.ShowSourceRanges = Args.hasArg(OPT_fdiagnostics_print_source_range_info);
  Opts.VerifyDiagnostics = Args.hasArg(OPT_verify);
  Opts.BinaryOutput = Args.hasArg(OPT_fdiagnostics_binary);
  Opts.ErrorLimit = getLastArgIntValue(Args, OPT_ferror_limit, 0, Diags);
  Opts.MacroBacktraceLimit
    = getLastArgIntValue(Args, OPT_fmacro_backtrace_limit, 
                         DiagnosticOptions::DefaultMacroBacktraceLimit, Diags);
  Opts.TemplateBacktraceLimit
    = getLastArgIntValue(Args, OPT_ftemplate_backtrace_limit, 
                         DiagnosticOptions::DefaultTemplateBacktraceLimit, 
                         Diags);
  Opts.TabStop = getLastArgIntValue(Args, OPT_ftabstop,
                                    DiagnosticOptions::DefaultTabStop, Diags);
  if (Opts.TabStop == 0 || Opts.TabStop > DiagnosticOptions::MaxTabStop) {
    Diags.Report(diag::warn_ignoring_ftabstop_value)
      << Opts.TabStop << DiagnosticOptions::DefaultTabStop;
    Opts.TabStop = DiagnosticOptions::DefaultTabStop;
  }
  Opts.MessageLength = getLastArgIntValue(Args, OPT_fmessage_length, 0, Diags);
  Opts.DumpBuildInformation = getLastArgValue(Args, OPT_dump_build_information);
  Opts.Warnings = getAllArgValues(Args, OPT_W);
}

static FrontendOptions::InputKind
ParseFrontendArgs(FrontendOptions &Opts, ArgList &Args, Diagnostic &Diags) {
  using namespace cc1options;
  Opts.ProgramAction = frontend::ParseSyntaxOnly;
  if (const Arg *A = Args.getLastArg(OPT_Action_Group)) {
    switch (A->getOption().getID()) {
    default:
      assert(0 && "Invalid option in group!");
    case OPT_ast_dump:
      Opts.ProgramAction = frontend::ASTDump; break;
    case OPT_ast_print:
      Opts.ProgramAction = frontend::ASTPrint; break;
    case OPT_ast_print_xml:
      Opts.ProgramAction = frontend::ASTPrintXML; break;
    case OPT_ast_view:
      Opts.ProgramAction = frontend::ASTView; break;
    case OPT_dump_raw_tokens:
      Opts.ProgramAction = frontend::DumpRawTokens; break;
    case OPT_dump_tokens:
      Opts.ProgramAction = frontend::DumpTokens; break;
    case OPT_S:
      Opts.ProgramAction = frontend::EmitAssembly; break;
    case OPT_emit_llvm_bc:
      Opts.ProgramAction = frontend::EmitBC; break;
    case OPT_emit_html:
      Opts.ProgramAction = frontend::EmitHTML; break;
    case OPT_emit_llvm:
      Opts.ProgramAction = frontend::EmitLLVM; break;
    case OPT_emit_llvm_only:
      Opts.ProgramAction = frontend::EmitLLVMOnly; break;
    case OPT_emit_obj:
      Opts.ProgramAction = frontend::EmitObj; break;
    case OPT_fixit_EQ:
      Opts.FixItSuffix = A->getValue(Args);
      // fall-through!
    case OPT_fixit:
      Opts.ProgramAction = frontend::FixIt; break;
    case OPT_emit_pch:
      Opts.ProgramAction = frontend::GeneratePCH; break;
    case OPT_emit_pth:
      Opts.ProgramAction = frontend::GeneratePTH; break;
    case OPT_init_only:
      Opts.ProgramAction = frontend::InitOnly; break;
    case OPT_parse_noop:
      Opts.ProgramAction = frontend::ParseNoop; break;
    case OPT_parse_print_callbacks:
      Opts.ProgramAction = frontend::ParsePrintCallbacks; break;
    case OPT_fsyntax_only:
      Opts.ProgramAction = frontend::ParseSyntaxOnly; break;
    case OPT_print_decl_contexts:
      Opts.ProgramAction = frontend::PrintDeclContext; break;
    case OPT_E:
      Opts.ProgramAction = frontend::PrintPreprocessedInput; break;
    case OPT_rewrite_macros:
      Opts.ProgramAction = frontend::RewriteMacros; break;
    case OPT_rewrite_objc:
      Opts.ProgramAction = frontend::RewriteObjC; break;
    case OPT_rewrite_test:
      Opts.ProgramAction = frontend::RewriteTest; break;
    case OPT_analyze:
      Opts.ProgramAction = frontend::RunAnalysis; break;
    case OPT_Eonly:
      Opts.ProgramAction = frontend::RunPreprocessorOnly; break;
    }
  }
  if (const Arg *A = Args.getLastArg(OPT_plugin)) {
    Opts.ProgramAction = frontend::PluginAction;
    Opts.ActionName = A->getValue(Args);
  }

  if (const Arg *A = Args.getLastArg(OPT_code_completion_at)) {
    Opts.CodeCompletionAt =
      ParsedSourceLocation::FromString(A->getValue(Args));
    if (Opts.CodeCompletionAt.FileName.empty())
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue(Args);
  }
  Opts.DebugCodeCompletionPrinter =
    !Args.hasArg(OPT_no_code_completion_debug_printer);
  Opts.DisableFree = Args.hasArg(OPT_disable_free);

  Opts.OutputFile = getLastArgValue(Args, OPT_o);
  Opts.Plugins = getAllArgValues(Args, OPT_load);
  Opts.RelocatablePCH = Args.hasArg(OPT_relocatable_pch);
  Opts.ShowHelp = Args.hasArg(OPT_help);
  Opts.ShowMacrosInCodeCompletion = Args.hasArg(OPT_code_completion_macros);
  Opts.ShowStats = Args.hasArg(OPT_print_stats);
  Opts.ShowTimers = Args.hasArg(OPT_ftime_report);
  Opts.ShowVersion = Args.hasArg(OPT_version);
  Opts.ViewClassInheritance = getLastArgValue(Args, OPT_cxx_inheritance_view);
  Opts.ASTMergeFiles = getAllArgValues(Args, OPT_ast_merge);
  Opts.LLVMArgs = getAllArgValues(Args, OPT_mllvm);

  FrontendOptions::InputKind DashX = FrontendOptions::IK_None;
  if (const Arg *A = Args.getLastArg(OPT_x)) {
    DashX = llvm::StringSwitch<FrontendOptions::InputKind>(A->getValue(Args))
      .Case("c", FrontendOptions::IK_C)
      .Case("cl", FrontendOptions::IK_OpenCL)
      .Case("c", FrontendOptions::IK_C)
      .Case("cl", FrontendOptions::IK_OpenCL)
      .Case("c++", FrontendOptions::IK_CXX)
      .Case("objective-c", FrontendOptions::IK_ObjC)
      .Case("objective-c++", FrontendOptions::IK_ObjCXX)
      .Case("cpp-output", FrontendOptions::IK_PreprocessedC)
      .Case("assembler-with-cpp", FrontendOptions::IK_Asm)
      .Case("c++-cpp-output", FrontendOptions::IK_PreprocessedCXX)
      .Case("objective-c-cpp-output", FrontendOptions::IK_PreprocessedObjC)
      .Case("objective-c++-cpp-output", FrontendOptions::IK_PreprocessedObjCXX)
      .Case("c-header", FrontendOptions::IK_C)
      .Case("objective-c-header", FrontendOptions::IK_ObjC)
      .Case("c++-header", FrontendOptions::IK_CXX)
      .Case("objective-c++-header", FrontendOptions::IK_ObjCXX)
      .Case("ast", FrontendOptions::IK_AST)
      .Default(FrontendOptions::IK_None);
    if (DashX == FrontendOptions::IK_None)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue(Args);
  }

  // '-' is the default input if none is given.
  std::vector<std::string> Inputs = getAllArgValues(Args, OPT_INPUT);
  Opts.Inputs.clear();
  if (Inputs.empty())
    Inputs.push_back("-");
  for (unsigned i = 0, e = Inputs.size(); i != e; ++i) {
    FrontendOptions::InputKind IK = DashX;
    if (IK == FrontendOptions::IK_None) {
      IK = FrontendOptions::getInputKindForExtension(
        llvm::StringRef(Inputs[i]).rsplit('.').second);
      // FIXME: Remove this hack.
      if (i == 0)
        DashX = IK;
    }
    Opts.Inputs.push_back(std::make_pair(IK, Inputs[i]));
  }

  return DashX;
}

std::string CompilerInvocation::GetResourcesPath(const char *Argv0,
                                                 void *MainAddr) {
  llvm::sys::Path P = llvm::sys::Path::GetMainExecutable(Argv0, MainAddr);

  if (!P.isEmpty()) {
    P.eraseComponent();  // Remove /clang from foo/bin/clang
    P.eraseComponent();  // Remove /bin   from foo/bin

    // Get foo/lib/clang/<version>/include
    P.appendComponent("lib");
    P.appendComponent("clang");
    P.appendComponent(CLANG_VERSION_STRING);
  }

  return P.str();
}

static void ParseHeaderSearchArgs(HeaderSearchOptions &Opts, ArgList &Args) {
  using namespace cc1options;
  Opts.Sysroot = getLastArgValue(Args, OPT_isysroot, "/");
  Opts.Verbose = Args.hasArg(OPT_v);
  Opts.UseBuiltinIncludes = !Args.hasArg(OPT_nobuiltininc);
  Opts.UseStandardIncludes = !Args.hasArg(OPT_nostdinc);
  Opts.UseStandardCXXIncludes = !Args.hasArg(OPT_nostdincxx);
  Opts.ResourceDir = getLastArgValue(Args, OPT_resource_dir);

  // Add -I... and -F... options in order.
  for (arg_iterator it = Args.filtered_begin(OPT_I, OPT_F),
         ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath(it->getValue(Args), frontend::Angled, true,
                 /*IsFramework=*/ it->getOption().matches(OPT_F));

  // Add -iprefix/-iwith-prefix/-iwithprefixbefore options.
  llvm::StringRef Prefix = ""; // FIXME: This isn't the correct default prefix.
  for (arg_iterator it = Args.filtered_begin(OPT_iprefix, OPT_iwithprefix,
                                             OPT_iwithprefixbefore),
         ie = Args.filtered_end(); it != ie; ++it) {
    if (it->getOption().matches(OPT_iprefix))
      Prefix = it->getValue(Args);
    else if (it->getOption().matches(OPT_iwithprefix))
      Opts.AddPath(Prefix.str() + it->getValue(Args),
                   frontend::System, false, false);
    else
      Opts.AddPath(Prefix.str() + it->getValue(Args),
                   frontend::Angled, false, false);
  }

  for (arg_iterator it = Args.filtered_begin(OPT_idirafter),
         ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath(it->getValue(Args), frontend::After, true, false);
  for (arg_iterator it = Args.filtered_begin(OPT_iquote),
         ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath(it->getValue(Args), frontend::Quoted, true, false);
  for (arg_iterator it = Args.filtered_begin(OPT_isystem),
         ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath(it->getValue(Args), frontend::System, true, false);

  // FIXME: Need options for the various environment variables!
}

static void ParseLangArgs(LangOptions &Opts, ArgList &Args,
                          FrontendOptions::InputKind IK,
                          Diagnostic &Diags) {
  // FIXME: Cleanup per-file based stuff.

  // Set some properties which depend soley on the input kind; it would be nice
  // to move these to the language standard, and have the driver resolve the
  // input kind + language standard.
  if (IK == FrontendOptions::IK_Asm) {
    Opts.AsmPreprocessor = 1;
  } else if (IK == FrontendOptions::IK_ObjC ||
             IK == FrontendOptions::IK_ObjCXX ||
             IK == FrontendOptions::IK_PreprocessedObjC ||
             IK == FrontendOptions::IK_PreprocessedObjCXX) {
    Opts.ObjC1 = Opts.ObjC2 = 1;
  }

  LangStandard::Kind LangStd = LangStandard::lang_unspecified;
  if (const Arg *A = Args.getLastArg(OPT_std_EQ)) {
    LangStd = llvm::StringSwitch<LangStandard::Kind>(A->getValue(Args))
#define LANGSTANDARD(id, name, desc, features) \
      .Case(name, LangStandard::lang_##id)
#include "clang/Frontend/LangStandards.def"
      .Default(LangStandard::lang_unspecified);
    if (LangStd == LangStandard::lang_unspecified)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue(Args);
  }

  if (LangStd == LangStandard::lang_unspecified) {
    // Based on the base language, pick one.
    switch (IK) {
    case FrontendOptions::IK_None:
    case FrontendOptions::IK_AST:
      assert(0 && "Invalid input kind!");
    case FrontendOptions::IK_OpenCL:
      LangStd = LangStandard::lang_opencl;
      break;
    case FrontendOptions::IK_Asm:
    case FrontendOptions::IK_C:
    case FrontendOptions::IK_PreprocessedC:
    case FrontendOptions::IK_ObjC:
    case FrontendOptions::IK_PreprocessedObjC:
      LangStd = LangStandard::lang_gnu99;
      break;
    case FrontendOptions::IK_CXX:
    case FrontendOptions::IK_PreprocessedCXX:
    case FrontendOptions::IK_ObjCXX:
    case FrontendOptions::IK_PreprocessedObjCXX:
      LangStd = LangStandard::lang_gnucxx98;
      break;
    }
  }

  const LangStandard &Std = LangStandard::getLangStandardForKind(LangStd);
  Opts.BCPLComment = Std.hasBCPLComments();
  Opts.C99 = Std.isC99();
  Opts.CPlusPlus = Std.isCPlusPlus();
  Opts.CPlusPlus0x = Std.isCPlusPlus0x();
  Opts.Digraphs = Std.hasDigraphs();
  Opts.GNUMode = Std.isGNUMode();
  Opts.GNUInline = !Std.isC99();
  Opts.HexFloats = Std.hasHexFloats();
  Opts.ImplicitInt = Std.hasImplicitInt();

  // OpenCL has some additional defaults.
  if (LangStd == LangStandard::lang_opencl) {
    Opts.OpenCL = 1;
    Opts.AltiVec = 1;
    Opts.CXXOperatorNames = 1;
    Opts.LaxVectorConversions = 1;
  }

  // OpenCL and C++ both have bool, true, false keywords.
  Opts.Bool = Opts.OpenCL || Opts.CPlusPlus;

  // We abuse '-f[no-]gnu-keywords' to force overriding all GNU-extension
  // keywords. This behavior is provided by GCC's poorly named '-fasm' flag,
  // while a subset (the non-C++ GNU keywords) is provided by GCC's
  // '-fgnu-keywords'. Clang conflates the two for simplicity under the single
  // name, as it doesn't seem a useful distinction.
  Opts.GNUKeywords = Args.hasFlag(OPT_fgnu_keywords, OPT_fno_gnu_keywords,
                                  Opts.GNUMode);

  if (Opts.CPlusPlus)
    Opts.CXXOperatorNames = !Args.hasArg(OPT_fno_operator_names);

  if (Args.hasArg(OPT_fobjc_gc_only))
    Opts.setGCMode(LangOptions::GCOnly);
  else if (Args.hasArg(OPT_fobjc_gc))
    Opts.setGCMode(LangOptions::HybridGC);

  if (Args.hasArg(OPT_print_ivar_layout))
    Opts.ObjCGCBitmapPrint = 1;
  if (Args.hasArg(OPT_fno_constant_cfstrings))
    Opts.NoConstantCFStrings = 1;

  if (Args.hasArg(OPT_faltivec))
    Opts.AltiVec = 1;

  if (Args.hasArg(OPT_pthread))
    Opts.POSIXThreads = 1;

  llvm::StringRef Vis = getLastArgValue(Args, OPT_fvisibility,
                                        "default");
  if (Vis == "default")
    Opts.setVisibilityMode(LangOptions::Default);
  else if (Vis == "hidden")
    Opts.setVisibilityMode(LangOptions::Hidden);
  else if (Vis == "protected")
    Opts.setVisibilityMode(LangOptions::Protected);
  else
    Diags.Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fvisibility)->getAsString(Args) << Vis;

  Opts.OverflowChecking = Args.hasArg(OPT_ftrapv);

  // Mimicing gcc's behavior, trigraphs are only enabled if -trigraphs
  // is specified, or -std is set to a conforming mode.
  Opts.Trigraphs = !Opts.GNUMode;
  if (Args.hasArg(OPT_trigraphs))
    Opts.Trigraphs = 1;

  Opts.DollarIdents = Args.hasFlag(OPT_fdollars_in_identifiers,
                                   OPT_fno_dollars_in_identifiers,
                                   !Opts.AsmPreprocessor);
  Opts.PascalStrings = Args.hasArg(OPT_fpascal_strings);
  Opts.Microsoft = Args.hasArg(OPT_fms_extensions);
  Opts.WritableStrings = Args.hasArg(OPT_fwritable_strings);
  Opts.ConstStrings = Args.hasArg(OPT_Wwrite_strings);
  if (Args.hasArg(OPT_fno_lax_vector_conversions))
    Opts.LaxVectorConversions = 0;
  if (Args.hasArg(OPT_fno_threadsafe_statics))
    Opts.ThreadsafeStatics = 0;
  Opts.Exceptions = Args.hasArg(OPT_fexceptions);
  Opts.RTTI = !Args.hasArg(OPT_fno_rtti);
  Opts.Blocks = Args.hasArg(OPT_fblocks);
  Opts.CharIsSigned = !Args.hasArg(OPT_fno_signed_char);
  Opts.ShortWChar = Args.hasArg(OPT_fshort_wchar);
  Opts.Freestanding = Args.hasArg(OPT_ffreestanding);
  Opts.NoBuiltin = Args.hasArg(OPT_fno_builtin) || Opts.Freestanding;
  Opts.AssumeSaneOperatorNew = !Args.hasArg(OPT_fno_assume_sane_operator_new);
  Opts.HeinousExtensions = Args.hasArg(OPT_fheinous_gnu_extensions);
  Opts.AccessControl = !Args.hasArg(OPT_fno_access_control);
  Opts.ElideConstructors = !Args.hasArg(OPT_fno_elide_constructors);
  Opts.MathErrno = Args.hasArg(OPT_fmath_errno);
  Opts.InstantiationDepth = getLastArgIntValue(Args, OPT_ftemplate_depth, 1024,
                                               Diags);
  Opts.NeXTRuntime = !Args.hasArg(OPT_fgnu_runtime);
  Opts.ObjCConstantStringClass = getLastArgValue(Args,
                                                 OPT_fconstant_string_class);
  Opts.ObjCNonFragileABI = Args.hasArg(OPT_fobjc_nonfragile_abi);
  Opts.ObjCNonFragileABI2 = Args.hasArg(OPT_fobjc_nonfragile_abi2);
  if (Opts.ObjCNonFragileABI2)
    Opts.ObjCNonFragileABI = true;
  Opts.CatchUndefined = Args.hasArg(OPT_fcatch_undefined_behavior);
  Opts.EmitAllDecls = Args.hasArg(OPT_femit_all_decls);
  Opts.PICLevel = getLastArgIntValue(Args, OPT_pic_level, 0, Diags);
  Opts.SjLjExceptions = Args.hasArg(OPT_fsjlj_exceptions);
  Opts.Static = Args.hasArg(OPT_static_define);
  Opts.DumpRecordLayouts = Args.hasArg(OPT_fdump_record_layouts);
  Opts.DumpVTableLayouts = Args.hasArg(OPT_fdump_vtable_layouts);
  Opts.NoBitFieldTypeAlign = Args.hasArg(OPT_fno_bitfield_type_align);
  Opts.OptimizeSize = 0;

  // FIXME: Eliminate this dependency.
  unsigned Opt =
    Args.hasArg(OPT_Os) ? 2 : getLastArgIntValue(Args, OPT_O, 0, Diags);
  Opts.Optimize = Opt != 0;

  // This is the __NO_INLINE__ define, which just depends on things like the
  // optimization level and -fno-inline, not actually whether the backend has
  // inlining enabled.
  //
  // FIXME: This is affected by other options (-fno-inline).
  Opts.NoInline = !Opt;

  unsigned SSP = getLastArgIntValue(Args, OPT_stack_protector, 0, Diags);
  switch (SSP) {
  default:
    Diags.Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_stack_protector)->getAsString(Args) << SSP;
    break;
  case 0: Opts.setStackProtectorMode(LangOptions::SSPOff); break;
  case 1: Opts.setStackProtectorMode(LangOptions::SSPOn);  break;
  case 2: Opts.setStackProtectorMode(LangOptions::SSPReq); break;
  }
}

static void ParsePreprocessorArgs(PreprocessorOptions &Opts, ArgList &Args,
                                  Diagnostic &Diags) {
  using namespace cc1options;
  Opts.ImplicitPCHInclude = getLastArgValue(Args, OPT_include_pch);
  Opts.ImplicitPTHInclude = getLastArgValue(Args, OPT_include_pth);
  if (const Arg *A = Args.getLastArg(OPT_token_cache))
      Opts.TokenCache = A->getValue(Args);
  else
    Opts.TokenCache = Opts.ImplicitPTHInclude;
  Opts.UsePredefines = !Args.hasArg(OPT_undef);
  Opts.DetailedRecord = Args.hasArg(OPT_detailed_preprocessing_record);
  // Add macros from the command line.
  for (arg_iterator it = Args.filtered_begin(OPT_D, OPT_U),
         ie = Args.filtered_end(); it != ie; ++it) {
    if (it->getOption().matches(OPT_D))
      Opts.addMacroDef(it->getValue(Args));
    else
      Opts.addMacroUndef(it->getValue(Args));
  }

  Opts.MacroIncludes = getAllArgValues(Args, OPT_imacros);

  // Add the ordered list of -includes.
  for (arg_iterator it = Args.filtered_begin(OPT_include, OPT_include_pch,
                                             OPT_include_pth),
         ie = Args.filtered_end(); it != ie; ++it) {
    // PCH is handled specially, we need to extra the original include path.
    if (it->getOption().matches(OPT_include_pch)) {
      std::string OriginalFile =
        PCHReader::getOriginalSourceFile(it->getValue(Args), Diags);
      if (OriginalFile.empty())
        continue;

      Opts.Includes.push_back(OriginalFile);
    } else
      Opts.Includes.push_back(it->getValue(Args));
  }

  // Include 'altivec.h' if -faltivec option present
  if (Args.hasArg(OPT_faltivec))
    Opts.Includes.push_back("altivec.h");

  for (arg_iterator it = Args.filtered_begin(OPT_remap_file),
         ie = Args.filtered_end(); it != ie; ++it) {
    std::pair<llvm::StringRef,llvm::StringRef> Split =
      llvm::StringRef(it->getValue(Args)).split(';');

    if (Split.second.empty()) {
      Diags.Report(diag::err_drv_invalid_remap_file) << it->getAsString(Args);
      continue;
    }

    Opts.addRemappedFile(Split.first, Split.second);
  }
}

static void ParsePreprocessorOutputArgs(PreprocessorOutputOptions &Opts,
                                        ArgList &Args) {
  using namespace cc1options;
  Opts.ShowCPP = !Args.hasArg(OPT_dM);
  Opts.ShowMacros = Args.hasArg(OPT_dM) || Args.hasArg(OPT_dD);
  Opts.ShowLineMarkers = !Args.hasArg(OPT_P);
  Opts.ShowComments = Args.hasArg(OPT_C);
  Opts.ShowMacroComments = Args.hasArg(OPT_CC);
}

static void ParseTargetArgs(TargetOptions &Opts, ArgList &Args) {
  using namespace cc1options;
  Opts.ABI = getLastArgValue(Args, OPT_target_abi);
  Opts.CPU = getLastArgValue(Args, OPT_target_cpu);
  Opts.Triple = getLastArgValue(Args, OPT_triple);
  Opts.Features = getAllArgValues(Args, OPT_target_feature);

  // Use the host triple if unspecified.
  if (Opts.Triple.empty())
    Opts.Triple = llvm::sys::getHostTriple();
}

//

void CompilerInvocation::CreateFromArgs(CompilerInvocation &Res,
                                        const char **ArgBegin,
                                        const char **ArgEnd,
                                        Diagnostic &Diags) {
  // Parse the arguments.
  llvm::OwningPtr<OptTable> Opts(createCC1OptTable());
  unsigned MissingArgIndex, MissingArgCount;
  llvm::OwningPtr<InputArgList> Args(
    Opts->ParseArgs(ArgBegin, ArgEnd,MissingArgIndex, MissingArgCount));

  // Check for missing argument error.
  if (MissingArgCount)
    Diags.Report(diag::err_drv_missing_argument)
      << Args->getArgString(MissingArgIndex) << MissingArgCount;

  // Issue errors on unknown arguments.
  for (arg_iterator it = Args->filtered_begin(OPT_UNKNOWN),
         ie = Args->filtered_end(); it != ie; ++it)
    Diags.Report(diag::err_drv_unknown_argument) << it->getAsString(*Args);

  ParseAnalyzerArgs(Res.getAnalyzerOpts(), *Args, Diags);
  ParseCodeGenArgs(Res.getCodeGenOpts(), *Args, Diags);
  ParseDependencyOutputArgs(Res.getDependencyOutputOpts(), *Args);
  ParseDiagnosticArgs(Res.getDiagnosticOpts(), *Args, Diags);
  FrontendOptions::InputKind DashX =
    ParseFrontendArgs(Res.getFrontendOpts(), *Args, Diags);
  ParseHeaderSearchArgs(Res.getHeaderSearchOpts(), *Args);
  if (DashX != FrontendOptions::IK_AST)
    ParseLangArgs(Res.getLangOpts(), *Args, DashX, Diags);
  ParsePreprocessorArgs(Res.getPreprocessorOpts(), *Args, Diags);
  ParsePreprocessorOutputArgs(Res.getPreprocessorOutputOpts(), *Args);
  ParseTargetArgs(Res.getTargetOpts(), *Args);
}
