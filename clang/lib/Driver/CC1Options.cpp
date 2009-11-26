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
#include "clang/Frontend/PCHReader.h"
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
      llvm::errs() << "error: invalid analysis store '" << Name << "'\n";
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
      llvm::errs() << "error: invalid analysis constraints '" << Name << "'\n";
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
      llvm::errs() << "error: invalid analysis output '" << Name << "'\n";
    else
      Opts.AnalysisDiagOpt = Value;
  }

  Opts.VisualizeEGDot = Args.hasArg(OPT_analyzer_viz_egraph_graphviz);
  Opts.VisualizeEGUbi = Args.hasArg(OPT_analyzer_viz_egraph_ubigraph);
  Opts.AnalyzeAll = Args.hasArg(OPT_analyzer_opt_analyze_headers);
  Opts.AnalyzerDisplayProgress = Args.hasArg(OPT_analyzer_display_progress);
  Opts.PurgeDead = !Args.hasArg(OPT_analyzer_no_purge_dead);
  Opts.EagerlyAssume = Args.hasArg(OPT_analyzer_eagerly_assume);
  Opts.AnalyzeSpecificFunction = getLastArgValue(Args, OPT_analyze_function);
  Opts.EnableExperimentalChecks = Args.hasArg(OPT_analyzer_experimental_checks);
  Opts.EnableExperimentalInternalChecks =
    Args.hasArg(OPT_analyzer_experimental_internal_checks);
  Opts.TrimGraph = Args.hasArg(OPT_trim_egraph);
}

static void ParseCodeGenArgs(CodeGenOptions &Opts, ArgList &Args) {
  using namespace cc1options;
  // -Os implies -O2
  if (Args.hasArg(OPT_Os))
    Opts.OptimizationLevel = 2;
  else
    Opts.OptimizationLevel = getLastArgIntValue(Args, OPT_O);

  // FIXME: What to do about argument parsing errors?
  if (Opts.OptimizationLevel > 3) {
    llvm::errs() << "error: invalid optimization level '"
                 << Opts.OptimizationLevel << "' (out of range)\n";
    Opts.OptimizationLevel = 3;
  }

  // We must always run at least the always inlining pass.
  Opts.Inlining = (Opts.OptimizationLevel > 1) ? CodeGenOptions::NormalInlining
    : CodeGenOptions::OnlyAlwaysInlining;

  Opts.DebugInfo = Args.hasArg(OPT_g);
  Opts.DisableLLVMOpts = Args.hasArg(OPT_disable_llvm_optzns);
  Opts.DisableRedZone = Args.hasArg(OPT_disable_red_zone);
  Opts.MergeAllConstants = !Args.hasArg(OPT_fno_merge_all_constants);
  Opts.NoCommon = Args.hasArg(OPT_fno_common);
  Opts.NoImplicitFloat = Args.hasArg(OPT_no_implicit_float);
  Opts.OptimizeSize = Args.hasArg(OPT_Os);
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
#else
  Opts.VerifyModule = 1;
#endif
}

static void ParseDependencyOutputArgs(DependencyOutputOptions &Opts,
                                      ArgList &Args) {
  using namespace cc1options;
  Opts.OutputFile = getLastArgValue(Args, OPT_dependency_file);
  Opts.Targets = getAllArgValues(Args, OPT_MT);
  Opts.IncludeSystemHeaders = Args.hasArg(OPT_sys_header_deps);
  Opts.UsePhonyTargets = Args.hasArg(OPT_MP);
}

static void ParseDiagnosticArgs(DiagnosticOptions &Opts, ArgList &Args) {
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
  Opts.MessageLength = getLastArgIntValue(Args, OPT_fmessage_length);
  Opts.DumpBuildInformation = getLastArgValue(Args, OPT_dump_build_information);
  Opts.Warnings = getAllArgValues(Args, OPT_W);
}

static void ParseFrontendArgs(FrontendOptions &Opts, ArgList &Args) {
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
    case OPT_dump_record_layouts:
      Opts.ProgramAction = frontend::DumpRecordLayouts; break;
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
    case OPT_fixit:
      Opts.ProgramAction = frontend::FixIt; break;
    case OPT_emit_pch:
      Opts.ProgramAction = frontend::GeneratePCH; break;
    case OPT_emit_pth:
      Opts.ProgramAction = frontend::GeneratePTH; break;
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
    case OPT_rewrite_blocks:
      Opts.ProgramAction = frontend::RewriteBlocks; break;
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
      llvm::errs() << "error: invalid source location '"
                   << A->getAsString(Args) << "'\n";
  }
  Opts.DebugCodeCompletionPrinter =
    !Args.hasArg(OPT_no_code_completion_debug_printer);
  Opts.DisableFree = Args.hasArg(OPT_disable_free);
  Opts.EmptyInputOnly = Args.hasArg(OPT_empty_input_only);

  std::vector<std::string> Fixits = getAllArgValues(Args, OPT_fixit_at);
  Opts.FixItLocations.clear();
  for (unsigned i = 0, e = Fixits.size(); i != e; ++i) {
    ParsedSourceLocation PSL = ParsedSourceLocation::FromString(Fixits[i]);

    if (PSL.FileName.empty()) {
      llvm::errs() << "error: invalid source location '" << Fixits[i] << "'\n";
      continue;
    }

    Opts.FixItLocations.push_back(PSL);
  }

  Opts.OutputFile = getLastArgValue(Args, OPT_o);
  Opts.RelocatablePCH = Args.hasArg(OPT_relocatable_pch);
  Opts.ShowMacrosInCodeCompletion = Args.hasArg(OPT_code_completion_macros);
  Opts.ShowStats = Args.hasArg(OPT_print_stats);
  Opts.ShowTimers = Args.hasArg(OPT_ftime_report);
  Opts.ViewClassInheritance = getLastArgValue(Args, OPT_cxx_inheritance_view);

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
      llvm::errs() << "error: invalid argument '" << A->getValue(Args)
                   << "' to '-x'\n";
  }

  // '-' is the default input if none is given.
  std::vector<std::string> Inputs = getAllArgValues(Args, OPT_INPUT);
  Opts.Inputs.clear();
  if (Inputs.empty())
    Inputs.push_back("-");
  for (unsigned i = 0, e = Inputs.size(); i != e; ++i) {
    FrontendOptions::InputKind IK = DashX;
    if (IK == FrontendOptions::IK_None)
      IK = FrontendOptions::getInputKindForExtension(
        llvm::StringRef(Inputs[i]).rsplit('.').second);
    Opts.Inputs.push_back(std::make_pair(IK, Inputs[i]));
  }
}

static void ParseHeaderSearchArgs(HeaderSearchOptions &Opts, ArgList &Args) {
  using namespace cc1options;
  Opts.Sysroot = getLastArgValue(Args, OPT_isysroot, "/");
  Opts.Verbose = Args.hasArg(OPT_v);
  Opts.UseStandardIncludes = !Args.hasArg(OPT_nostdinc);
  Opts.BuiltinIncludePath = "";
  if (!Args.hasArg(OPT_fno_builtin))
      Opts.BuiltinIncludePath = "FIXME"; // FIXME: Get builtin include path!

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

static void ParseLangArgs(LangOptions &Opts, ArgList &Args) {
}

static void ParsePreprocessorArgs(PreprocessorOptions &Opts, ArgList &Args) {
  using namespace cc1options;
  Opts.ImplicitPCHInclude = getLastArgValue(Args, OPT_include_pch);
  Opts.ImplicitPTHInclude = getLastArgValue(Args, OPT_include_pth);
  Opts.TokenCache = getLastArgValue(Args, OPT_token_cache);
  Opts.UsePredefines = !Args.hasArg(OPT_undef);

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
      // FIXME: Disabled for now, I don't want to incur the cost of linking in
      // Sema and all until we are actually going to use it. Alternatively this
      // could be factored out somehow.
      //        PCHReader::getOriginalSourceFile(it->getValue(Args));
      std::string OriginalFile = "FIXME";

      // FIXME: Don't fail like this.
      if (OriginalFile.empty())
        exit(1);

      Opts.Includes.push_back(OriginalFile);
    } else
      Opts.Includes.push_back(it->getValue(Args));
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
  Opts.CPU = getLastArgValue(Args, OPT_mcpu);
  Opts.Triple = getLastArgValue(Args, OPT_triple);
  Opts.Features = getAllArgValues(Args, OPT_target_feature);

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
  ParseDiagnosticArgs(Res.getDiagnosticOpts(), *InputArgs);
  ParseFrontendArgs(Res.getFrontendOpts(), *InputArgs);
  ParseHeaderSearchArgs(Res.getHeaderSearchOpts(), *InputArgs);
  ParseLangArgs(Res.getLangOpts(), *InputArgs);
  ParsePreprocessorArgs(Res.getPreprocessorOpts(), *InputArgs);
  ParsePreprocessorOutputArgs(Res.getPreprocessorOutputOpts(), *InputArgs);
  ParseTargetArgs(Res.getTargetOpts(), *InputArgs);
}
