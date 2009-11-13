//===--- Options.cpp - clang-cc Option Handling ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This file contains "pure" option handling, it is only responsible for turning
// the options into internal *Option classes, but shouldn't have any other
// logic.

#include "Options.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/AnalysisConsumer.h"
#include "clang/CodeGen/CodeGenOptions.h"
#include "clang/Frontend/DependencyOutputOptions.h"
#include "clang/Frontend/DiagnosticOptions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/HeaderSearchOptions.h"
#include "clang/Frontend/PCHReader.h"
#include "clang/Frontend/PreprocessorOptions.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include <stdio.h>

using namespace clang;

//===----------------------------------------------------------------------===//
// Analyzer Options
//===----------------------------------------------------------------------===//

namespace analyzeroptions {

static llvm::cl::list<Analyses>
AnalysisList(llvm::cl::desc("Source Code Analysis - Checks and Analyses"),
llvm::cl::values(
#define ANALYSIS(NAME, CMDFLAG, DESC, SCOPE)\
clEnumValN(NAME, CMDFLAG, DESC),
#include "clang/Frontend/Analyses.def"
clEnumValEnd));

static llvm::cl::opt<AnalysisStores>
AnalysisStoreOpt("analyzer-store",
  llvm::cl::desc("Source Code Analysis - Abstract Memory Store Models"),
  llvm::cl::init(BasicStoreModel),
  llvm::cl::values(
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN)\
clEnumValN(NAME##Model, CMDFLAG, DESC),
#include "clang/Frontend/Analyses.def"
clEnumValEnd));

static llvm::cl::opt<AnalysisConstraints>
AnalysisConstraintsOpt("analyzer-constraints",
  llvm::cl::desc("Source Code Analysis - Symbolic Constraint Engines"),
  llvm::cl::init(RangeConstraintsModel),
  llvm::cl::values(
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN)\
clEnumValN(NAME##Model, CMDFLAG, DESC),
#include "clang/Frontend/Analyses.def"
clEnumValEnd));

static llvm::cl::opt<AnalysisDiagClients>
AnalysisDiagOpt("analyzer-output",
                llvm::cl::desc("Source Code Analysis - Output Options"),
                llvm::cl::init(PD_HTML),
                llvm::cl::values(
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN, AUTOCREATE)\
clEnumValN(PD_##NAME, CMDFLAG, DESC),
#include "clang/Frontend/Analyses.def"
clEnumValEnd));

static llvm::cl::opt<bool>
AnalyzeAll("analyzer-opt-analyze-headers",
    llvm::cl::desc("Force the static analyzer to analyze "
                   "functions defined in header files"));
  
static llvm::cl::opt<bool>
AnalyzerDisplayProgress("analyzer-display-progress",
          llvm::cl::desc("Emit verbose output about the analyzer's progress"));

static llvm::cl::opt<bool>
AnalyzerExperimentalChecks("analyzer-experimental-checks",
          llvm::cl::desc("Use experimental path-sensitive checks"));
  
static llvm::cl::opt<std::string>
AnalyzeSpecificFunction("analyze-function",
               llvm::cl::desc("Run analysis on specific function"));

static llvm::cl::opt<bool>
EagerlyAssume("analyzer-eagerly-assume",
          llvm::cl::init(false),
              llvm::cl::desc("Eagerly assume the truth/falseness of some "
                             "symbolic constraints"));

static llvm::cl::opt<bool>
PurgeDead("analyzer-purge-dead",
          llvm::cl::init(true),
          llvm::cl::desc("Remove dead symbols, bindings, and constraints before"
                         " processing a statement"));

static llvm::cl::opt<bool>
TrimGraph("trim-egraph",
     llvm::cl::desc("Only show error-related paths in the analysis graph"));

static llvm::cl::opt<bool>
VisualizeEGDot("analyzer-viz-egraph-graphviz",
               llvm::cl::desc("Display exploded graph using GraphViz"));

static llvm::cl::opt<bool>
VisualizeEGUbi("analyzer-viz-egraph-ubigraph",
               llvm::cl::desc("Display exploded graph using Ubigraph"));

}

void clang::InitializeAnalyzerOptions(AnalyzerOptions &Opts) {
  using namespace analyzeroptions;
  Opts.AnalysisList = AnalysisList;
  Opts.AnalysisStoreOpt = AnalysisStoreOpt;
  Opts.AnalysisConstraintsOpt = AnalysisConstraintsOpt;
  Opts.AnalysisDiagOpt = AnalysisDiagOpt;
  Opts.VisualizeEGDot = VisualizeEGDot;
  Opts.VisualizeEGUbi = VisualizeEGUbi;
  Opts.AnalyzeAll = AnalyzeAll;
  Opts.AnalyzerDisplayProgress = AnalyzerDisplayProgress;
  Opts.PurgeDead = PurgeDead;
  Opts.EagerlyAssume = EagerlyAssume;
  Opts.AnalyzeSpecificFunction = AnalyzeSpecificFunction;
  Opts.EnableExperimentalChecks = AnalyzerExperimentalChecks;
  Opts.TrimGraph = TrimGraph;
}


//===----------------------------------------------------------------------===//
// Code Generation Options
//===----------------------------------------------------------------------===//

namespace codegenoptions {

static llvm::cl::opt<bool>
DisableLLVMOptimizations("disable-llvm-optzns",
                         llvm::cl::desc("Don't run LLVM optimization passes"));

static llvm::cl::opt<bool>
DisableRedZone("disable-red-zone",
               llvm::cl::desc("Do not emit code that uses the red zone."),
               llvm::cl::init(false));

static llvm::cl::opt<bool>
GenerateDebugInfo("g",
                  llvm::cl::desc("Generate source level debug information"));

static llvm::cl::opt<bool>
NoCommon("fno-common",
         llvm::cl::desc("Compile common globals like normal definitions"),
         llvm::cl::ValueDisallowed);

static llvm::cl::opt<bool>
NoImplicitFloat("no-implicit-float",
  llvm::cl::desc("Don't generate implicit floating point instructions (x86-only)"),
  llvm::cl::init(false));

static llvm::cl::opt<bool>
NoMergeConstants("fno-merge-all-constants",
                       llvm::cl::desc("Disallow merging of constants."));

// It might be nice to add bounds to the CommandLine library directly.
struct OptLevelParser : public llvm::cl::parser<unsigned> {
  bool parse(llvm::cl::Option &O, llvm::StringRef ArgName,
             llvm::StringRef Arg, unsigned &Val) {
    if (llvm::cl::parser<unsigned>::parse(O, ArgName, Arg, Val))
      return true;
    if (Val > 3)
      return O.error("'" + Arg + "' invalid optimization level!");
    return false;
  }
};
static llvm::cl::opt<unsigned, false, OptLevelParser>
OptLevel("O", llvm::cl::Prefix,
         llvm::cl::desc("Optimization level"),
         llvm::cl::init(0));

static llvm::cl::opt<bool>
OptSize("Os", llvm::cl::desc("Optimize for size"));

static llvm::cl::opt<std::string>
TargetCPU("mcpu",
         llvm::cl::desc("Target a specific cpu type (-mcpu=help for details)"));

static llvm::cl::list<std::string>
TargetFeatures("target-feature", llvm::cl::desc("Target specific attributes"));

}

//===----------------------------------------------------------------------===//
// Dependency Output Options
//===----------------------------------------------------------------------===//

namespace dependencyoutputoptions {

static llvm::cl::opt<std::string>
DependencyFile("dependency-file",
               llvm::cl::desc("Filename (or -) to write dependency output to"));

static llvm::cl::opt<bool>
DependenciesIncludeSystemHeaders("sys-header-deps",
                 llvm::cl::desc("Include system headers in dependency output"));

static llvm::cl::list<std::string>
DependencyTargets("MT",
         llvm::cl::desc("Specify target for dependency"));

static llvm::cl::opt<bool>
PhonyDependencyTarget("MP",
            llvm::cl::desc("Create phony target for each dependency "
                           "(other than main file)"));

}

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

namespace diagnosticoptions {

static llvm::cl::opt<std::string>
DumpBuildInformation("dump-build-information",
                     llvm::cl::value_desc("filename"),
          llvm::cl::desc("output a dump of some build information to a file"));

static llvm::cl::opt<bool>
NoShowColumn("fno-show-column",
             llvm::cl::desc("Do not include column number on diagnostics"));

static llvm::cl::opt<bool>
NoShowLocation("fno-show-source-location",
               llvm::cl::desc("Do not include source location information with"
                              " diagnostics"));

static llvm::cl::opt<bool>
NoCaretDiagnostics("fno-caret-diagnostics",
                   llvm::cl::desc("Do not include source line and caret with"
                                  " diagnostics"));

static llvm::cl::opt<bool>
NoDiagnosticsFixIt("fno-diagnostics-fixit-info",
                   llvm::cl::desc("Do not include fixit information in"
                                  " diagnostics"));

static llvm::cl::opt<bool> OptNoWarnings("w");

static llvm::cl::opt<bool> OptPedantic("pedantic");

static llvm::cl::opt<bool> OptPedanticErrors("pedantic-errors");

// This gets all -W options, including -Werror, -W[no-]system-headers, etc.  The
// driver has stripped off -Wa,foo etc.  The driver has also translated -W to
// -Wextra, so we don't need to worry about it.
static llvm::cl::list<std::string>
OptWarnings("W", llvm::cl::Prefix, llvm::cl::ValueOptional);

static llvm::cl::opt<bool>
PrintSourceRangeInfo("fdiagnostics-print-source-range-info",
                     llvm::cl::desc("Print source range spans in numeric form"));

static llvm::cl::opt<bool>
PrintDiagnosticOption("fdiagnostics-show-option",
             llvm::cl::desc("Print diagnostic name with mappable diagnostics"));

static llvm::cl::opt<unsigned>
MessageLength("fmessage-length",
              llvm::cl::desc("Format message diagnostics so that they fit "
                             "within N columns or fewer, when possible."),
              llvm::cl::value_desc("N"));

static llvm::cl::opt<bool>
PrintColorDiagnostic("fcolor-diagnostics",
                     llvm::cl::desc("Use colors in diagnostics"));

static llvm::cl::opt<bool>
SilenceRewriteMacroWarning("Wno-rewrite-macros", llvm::cl::init(false),
                           llvm::cl::desc("Silence ObjC rewriting warnings"));

static llvm::cl::opt<bool>
VerifyDiagnostics("verify",
                  llvm::cl::desc("Verify emitted diagnostics and warnings"));

}


//===----------------------------------------------------------------------===//
// Frontend Options
//===----------------------------------------------------------------------===//

namespace frontendoptions {

static llvm::cl::opt<ParsedSourceLocation>
CodeCompletionAt("code-completion-at",
                 llvm::cl::value_desc("file:line:column"),
              llvm::cl::desc("Dump code-completion information at a location"));

static llvm::cl::opt<bool>
CodeCompletionDebugPrinter("code-completion-debug-printer",
                      llvm::cl::desc("Use the \"debug\" code-completion print"),
                           llvm::cl::init(true));

static llvm::cl::opt<bool>
CodeCompletionWantsMacros("code-completion-macros",
                 llvm::cl::desc("Include macros in code-completion results"));

static llvm::cl::opt<bool>
DisableFree("disable-free",
           llvm::cl::desc("Disable freeing of memory on exit"),
           llvm::cl::init(false));

static llvm::cl::opt<bool>
EmptyInputOnly("empty-input-only",
      llvm::cl::desc("Force running on an empty input file"));

static llvm::cl::opt<FrontendOptions::InputKind>
InputType("x", llvm::cl::desc("Input language type"),
         llvm::cl::init(FrontendOptions::IK_None),
   llvm::cl::values(clEnumValN(FrontendOptions::IK_C,     "c", "C"),
                    clEnumValN(FrontendOptions::IK_OpenCL,   "cl", "OpenCL C"),
                    clEnumValN(FrontendOptions::IK_CXX,   "c++", "C++"),
                    clEnumValN(FrontendOptions::IK_ObjC,  "objective-c",
                               "Objective C"),
                    clEnumValN(FrontendOptions::IK_ObjCXX, "objective-c++",
                               "Objective C++"),
                    clEnumValN(FrontendOptions::IK_PreprocessedC,
                               "cpp-output",
                               "Preprocessed C"),
                    clEnumValN(FrontendOptions::IK_Asm,
                               "assembler-with-cpp",
                               "Assembly Source Codde"),
                    clEnumValN(FrontendOptions::IK_PreprocessedCXX,
                               "c++-cpp-output",
                               "Preprocessed C++"),
                    clEnumValN(FrontendOptions::IK_PreprocessedObjC,
                               "objective-c-cpp-output",
                               "Preprocessed Objective C"),
                    clEnumValN(FrontendOptions::IK_PreprocessedObjCXX,
                               "objective-c++-cpp-output",
                               "Preprocessed Objective C++"),
                    clEnumValN(FrontendOptions::IK_C, "c-header",
                               "C header"),
                    clEnumValN(FrontendOptions::IK_ObjC, "objective-c-header",
                               "Objective-C header"),
                    clEnumValN(FrontendOptions::IK_CXX, "c++-header",
                               "C++ header"),
                    clEnumValN(FrontendOptions::IK_ObjCXX,
                               "objective-c++-header",
                               "Objective-C++ header"),
                    clEnumValN(FrontendOptions::IK_AST, "ast",
                               "Clang AST"),
                    clEnumValEnd));

static llvm::cl::list<std::string>
InputFilenames(llvm::cl::Positional, llvm::cl::desc("<input files>"));

static llvm::cl::opt<std::string>
InheritanceViewCls("cxx-inheritance-view",
                   llvm::cl::value_desc("class name"),
                  llvm::cl::desc("View C++ inheritance for a specified class"));

static llvm::cl::opt<bool>
FixItAll("fixit", llvm::cl::desc("Apply fix-it advice to the input source"));

static llvm::cl::list<ParsedSourceLocation>
FixItAtLocations("fixit-at", llvm::cl::value_desc("source-location"),
   llvm::cl::desc("Perform Fix-It modifications at the given source location"));

static llvm::cl::opt<std::string>
OutputFile("o",
 llvm::cl::value_desc("path"),
 llvm::cl::desc("Specify output file"));

static llvm::cl::opt<bool>
RelocatablePCH("relocatable-pch",
               llvm::cl::desc("Whether to build a relocatable precompiled "
                              "header"));
static llvm::cl::opt<bool>
Stats("print-stats",
      llvm::cl::desc("Print performance metrics and statistics"));

static llvm::cl::opt<std::string>
TargetABI("target-abi",
          llvm::cl::desc("Target a particular ABI type"));

static llvm::cl::opt<std::string>
TargetTriple("triple",
  llvm::cl::desc("Specify target triple (e.g. i686-apple-darwin9)"));

static llvm::cl::opt<bool>
TimeReport("ftime-report",
           llvm::cl::desc("Print the amount of time each "
                          "phase of compilation takes"));

}

//===----------------------------------------------------------------------===//
// Language Options
//===----------------------------------------------------------------------===//

namespace langoptions {

static llvm::cl::opt<bool>
AllowBuiltins("fbuiltin", llvm::cl::init(true),
             llvm::cl::desc("Disable implicit builtin knowledge of functions"));

static llvm::cl::opt<bool>
AltiVec("faltivec", llvm::cl::desc("Enable AltiVec vector initializer syntax"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool>
AccessControl("faccess-control",
              llvm::cl::desc("Enable C++ access control"));

static llvm::cl::opt<bool>
CharIsSigned("fsigned-char",
    llvm::cl::desc("Force char to be a signed/unsigned type"));

static llvm::cl::opt<bool>
DollarsInIdents("fdollars-in-identifiers",
                llvm::cl::desc("Allow '$' in identifiers"));

static llvm::cl::opt<bool>
EmitAllDecls("femit-all-decls",
              llvm::cl::desc("Emit all declarations, even if unused"));

static llvm::cl::opt<bool>
EnableBlocks("fblocks", llvm::cl::desc("enable the 'blocks' language feature"));

static llvm::cl::opt<bool>
EnableHeinousExtensions("fheinous-gnu-extensions",
   llvm::cl::desc("enable GNU extensions that you really really shouldn't use"),
                        llvm::cl::ValueDisallowed, llvm::cl::Hidden);

static llvm::cl::opt<bool>
Exceptions("fexceptions",
           llvm::cl::desc("Enable support for exception handling"));

static llvm::cl::opt<bool>
Freestanding("ffreestanding",
             llvm::cl::desc("Assert that the compilation takes place in a "
                            "freestanding environment"));

static llvm::cl::opt<bool>
GNURuntime("fgnu-runtime",
            llvm::cl::desc("Generate output compatible with the standard GNU "
                           "Objective-C runtime"));

/// LangStds - Language standards we support.
enum LangStds {
  lang_unspecified,
  lang_c89, lang_c94, lang_c99,
  lang_gnu89, lang_gnu99,
  lang_cxx98, lang_gnucxx98,
  lang_cxx0x, lang_gnucxx0x
};
static llvm::cl::opt<LangStds>
LangStd("std", llvm::cl::desc("Language standard to compile for"),
        llvm::cl::init(lang_unspecified),
  llvm::cl::values(clEnumValN(lang_c89,      "c89",            "ISO C 1990"),
                   clEnumValN(lang_c89,      "c90",            "ISO C 1990"),
                   clEnumValN(lang_c89,      "iso9899:1990",   "ISO C 1990"),
                   clEnumValN(lang_c94,      "iso9899:199409",
                              "ISO C 1990 with amendment 1"),
                   clEnumValN(lang_c99,      "c99",            "ISO C 1999"),
                   clEnumValN(lang_c99,      "c9x",            "ISO C 1999"),
                   clEnumValN(lang_c99,      "iso9899:1999",   "ISO C 1999"),
                   clEnumValN(lang_c99,      "iso9899:199x",   "ISO C 1999"),
                   clEnumValN(lang_gnu89,    "gnu89",
                              "ISO C 1990 with GNU extensions"),
                   clEnumValN(lang_gnu99,    "gnu99",
                              "ISO C 1999 with GNU extensions (default for C)"),
                   clEnumValN(lang_gnu99,    "gnu9x",
                              "ISO C 1999 with GNU extensions"),
                   clEnumValN(lang_cxx98,    "c++98",
                              "ISO C++ 1998 with amendments"),
                   clEnumValN(lang_gnucxx98, "gnu++98",
                              "ISO C++ 1998 with amendments and GNU "
                              "extensions (default for C++)"),
                   clEnumValN(lang_cxx0x,    "c++0x",
                              "Upcoming ISO C++ 200x with amendments"),
                   clEnumValN(lang_gnucxx0x, "gnu++0x",
                              "Upcoming ISO C++ 200x with amendments and GNU "
                              "extensions"),
                   clEnumValEnd));

static llvm::cl::opt<bool>
MSExtensions("fms-extensions",
             llvm::cl::desc("Accept some non-standard constructs used in "
                            "Microsoft header files "));

static llvm::cl::opt<std::string>
MainFileName("main-file-name",
             llvm::cl::desc("Main file name to use for debug info"));

static llvm::cl::opt<bool>
MathErrno("fmath-errno", llvm::cl::init(true),
          llvm::cl::desc("Require math functions to respect errno"));

static llvm::cl::opt<bool>
NeXTRuntime("fnext-runtime",
            llvm::cl::desc("Generate output compatible with the NeXT "
                           "runtime"));

static llvm::cl::opt<bool>
NoElideConstructors("fno-elide-constructors",
                    llvm::cl::desc("Disable C++ copy constructor elision"));

static llvm::cl::opt<bool>
NoLaxVectorConversions("fno-lax-vector-conversions",
                       llvm::cl::desc("Disallow implicit conversions between "
                                      "vectors with a different number of "
                                      "elements or different element types"));


static llvm::cl::opt<bool>
NoOperatorNames("fno-operator-names",
                llvm::cl::desc("Do not treat C++ operator name keywords as "
                               "synonyms for operators"));

static llvm::cl::opt<std::string>
ObjCConstantStringClass("fconstant-string-class",
                llvm::cl::value_desc("class name"),
                llvm::cl::desc("Specify the class to use for constant "
                               "Objective-C string objects."));

static llvm::cl::opt<bool>
ObjCEnableGC("fobjc-gc",
             llvm::cl::desc("Enable Objective-C garbage collection"));

static llvm::cl::opt<bool>
ObjCExclusiveGC("fobjc-gc-only",
                llvm::cl::desc("Use GC exclusively for Objective-C related "
                               "memory management"));

static llvm::cl::opt<bool>
ObjCEnableGCBitmapPrint("print-ivar-layout",
             llvm::cl::desc("Enable Objective-C Ivar layout bitmap print trace"));

static llvm::cl::opt<bool>
ObjCNonFragileABI("fobjc-nonfragile-abi",
                  llvm::cl::desc("enable objective-c's nonfragile abi"));

static llvm::cl::opt<bool>
OverflowChecking("ftrapv",
                 llvm::cl::desc("Trap on integer overflow"),
                 llvm::cl::init(false));

static llvm::cl::opt<unsigned>
PICLevel("pic-level", llvm::cl::desc("Value for __PIC__"));

static llvm::cl::opt<bool>
PThread("pthread", llvm::cl::desc("Support POSIX threads in generated code"),
         llvm::cl::init(false));

static llvm::cl::opt<bool>
PascalStrings("fpascal-strings",
              llvm::cl::desc("Recognize and construct Pascal-style "
                             "string literals"));

static llvm::cl::opt<bool>
Rtti("frtti", llvm::cl::init(true),
     llvm::cl::desc("Enable generation of rtti information"));

static llvm::cl::opt<bool>
ShortWChar("fshort-wchar",
    llvm::cl::desc("Force wchar_t to be a short unsigned int"));

static llvm::cl::opt<bool>
StaticDefine("static-define", llvm::cl::desc("Should __STATIC__ be defined"));

static llvm::cl::opt<int>
StackProtector("stack-protector",
               llvm::cl::desc("Enable stack protectors"),
               llvm::cl::init(-1));

static llvm::cl::opt<LangOptions::VisibilityMode>
SymbolVisibility("fvisibility",
                 llvm::cl::desc("Set the default symbol visibility:"),
                 llvm::cl::init(LangOptions::Default),
                 llvm::cl::values(clEnumValN(LangOptions::Default, "default",
                                             "Use default symbol visibility"),
                                  clEnumValN(LangOptions::Hidden, "hidden",
                                             "Use hidden symbol visibility"),
                                  clEnumValN(LangOptions::Protected,"protected",
                                             "Use protected symbol visibility"),
                                  clEnumValEnd));

static llvm::cl::opt<unsigned>
TemplateDepth("ftemplate-depth", llvm::cl::init(99),
              llvm::cl::desc("Maximum depth of recursive template "
                             "instantiation"));

static llvm::cl::opt<bool>
Trigraphs("trigraphs", llvm::cl::desc("Process trigraph sequences"));

static llvm::cl::opt<bool>
WritableStrings("fwritable-strings",
              llvm::cl::desc("Store string literals as writable data"));

}

//===----------------------------------------------------------------------===//
// General Preprocessor Options
//===----------------------------------------------------------------------===//

namespace preprocessoroptions {

static llvm::cl::list<std::string>
D_macros("D", llvm::cl::value_desc("macro"), llvm::cl::Prefix,
       llvm::cl::desc("Predefine the specified macro"));

static llvm::cl::list<std::string>
ImplicitIncludes("include", llvm::cl::value_desc("file"),
                 llvm::cl::desc("Include file before parsing"));
static llvm::cl::list<std::string>
ImplicitMacroIncludes("imacros", llvm::cl::value_desc("file"),
                      llvm::cl::desc("Include macros from file before parsing"));

static llvm::cl::opt<std::string>
ImplicitIncludePCH("include-pch", llvm::cl::value_desc("file"),
                   llvm::cl::desc("Include precompiled header file"));

static llvm::cl::opt<std::string>
ImplicitIncludePTH("include-pth", llvm::cl::value_desc("file"),
                   llvm::cl::desc("Include file before parsing"));

static llvm::cl::opt<std::string>
TokenCache("token-cache", llvm::cl::value_desc("path"),
           llvm::cl::desc("Use specified token cache file"));

static llvm::cl::list<std::string>
U_macros("U", llvm::cl::value_desc("macro"), llvm::cl::Prefix,
         llvm::cl::desc("Undefine the specified macro"));

static llvm::cl::opt<bool>
UndefMacros("undef", llvm::cl::value_desc("macro"),
            llvm::cl::desc("undef all system defines"));

}

//===----------------------------------------------------------------------===//
// Header Search Options
//===----------------------------------------------------------------------===//

namespace headersearchoptions {

static llvm::cl::opt<bool>
nostdinc("nostdinc", llvm::cl::desc("Disable standard #include directories"));

static llvm::cl::opt<bool>
nobuiltininc("nobuiltininc",
             llvm::cl::desc("Disable builtin #include directories"));

// Various command line options.  These four add directories to each chain.
static llvm::cl::list<std::string>
F_dirs("F", llvm::cl::value_desc("directory"), llvm::cl::Prefix,
       llvm::cl::desc("Add directory to framework include search path"));

static llvm::cl::list<std::string>
I_dirs("I", llvm::cl::value_desc("directory"), llvm::cl::Prefix,
       llvm::cl::desc("Add directory to include search path"));

static llvm::cl::list<std::string>
idirafter_dirs("idirafter", llvm::cl::value_desc("directory"), llvm::cl::Prefix,
               llvm::cl::desc("Add directory to AFTER include search path"));

static llvm::cl::list<std::string>
iquote_dirs("iquote", llvm::cl::value_desc("directory"), llvm::cl::Prefix,
               llvm::cl::desc("Add directory to QUOTE include search path"));

static llvm::cl::list<std::string>
isystem_dirs("isystem", llvm::cl::value_desc("directory"), llvm::cl::Prefix,
            llvm::cl::desc("Add directory to SYSTEM include search path"));

// These handle -iprefix/-iwithprefix/-iwithprefixbefore.
static llvm::cl::list<std::string>
iprefix_vals("iprefix", llvm::cl::value_desc("prefix"), llvm::cl::Prefix,
             llvm::cl::desc("Set the -iwithprefix/-iwithprefixbefore prefix"));
static llvm::cl::list<std::string>
iwithprefix_vals("iwithprefix", llvm::cl::value_desc("dir"), llvm::cl::Prefix,
     llvm::cl::desc("Set directory to SYSTEM include search path with prefix"));
static llvm::cl::list<std::string>
iwithprefixbefore_vals("iwithprefixbefore", llvm::cl::value_desc("dir"),
                       llvm::cl::Prefix,
            llvm::cl::desc("Set directory to include search path with prefix"));

static llvm::cl::opt<std::string>
isysroot("isysroot", llvm::cl::value_desc("dir"), llvm::cl::init("/"),
         llvm::cl::desc("Set the system root directory (usually /)"));

static llvm::cl::opt<bool>
Verbose("v", llvm::cl::desc("Enable verbose output"));

}

//===----------------------------------------------------------------------===//
// Preprocessed Output Options
//===----------------------------------------------------------------------===//

namespace preprocessoroutputoptions {

static llvm::cl::opt<bool>
DisableLineMarkers("P", llvm::cl::desc("Disable linemarker output in -E mode"));

static llvm::cl::opt<bool>
EnableCommentOutput("C", llvm::cl::desc("Enable comment output in -E mode"));

static llvm::cl::opt<bool>
EnableMacroCommentOutput("CC",
                         llvm::cl::desc("Enable comment output in -E mode, "
                                        "even from macro expansions"));
static llvm::cl::opt<bool>
DumpMacros("dM", llvm::cl::desc("Print macro definitions in -E mode instead of"
                                " normal output"));
static llvm::cl::opt<bool>
DumpDefines("dD", llvm::cl::desc("Print macro definitions in -E mode in "
                                "addition to normal output"));

}

//===----------------------------------------------------------------------===//
// Option Object Construction
//===----------------------------------------------------------------------===//

void clang::InitializeCodeGenOptions(CodeGenOptions &Opts,
                                     const TargetInfo &Target) {
  using namespace codegenoptions;

  // Compute the target features, we need the target to handle this because
  // features may have dependencies on one another.
  llvm::StringMap<bool> Features;
  Target.getDefaultFeatures(TargetCPU, Features);

  // Apply the user specified deltas.
  for (llvm::cl::list<std::string>::iterator it = TargetFeatures.begin(),
         ie = TargetFeatures.end(); it != ie; ++it) {
    const char *Name = it->c_str();

    // FIXME: Don't handle errors like this.
    if (Name[0] != '-' && Name[0] != '+') {
      fprintf(stderr, "error: clang-cc: invalid target feature string: %s\n",
              Name);
      exit(1);
    }

    // Apply the feature via the target.
    if (!Target.setFeatureEnabled(Features, Name + 1, (Name[0] == '+'))) {
      fprintf(stderr, "error: clang-cc: invalid target feature name: %s\n",
              Name + 1);
      exit(1);
    }
  }

  // Add the features to the compile options.
  //
  // FIXME: If we are completely confident that we have the right set, we only
  // need to pass the minuses.
  for (llvm::StringMap<bool>::const_iterator it = Features.begin(),
         ie = Features.end(); it != ie; ++it)
    Opts.Features.push_back(std::string(it->second ? "+" : "-") + it->first());

  // -Os implies -O2
  Opts.OptimizationLevel = OptSize ? 2 : OptLevel;

  // We must always run at least the always inlining pass.
  Opts.Inlining = (Opts.OptimizationLevel > 1) ? CodeGenOptions::NormalInlining
    : CodeGenOptions::OnlyAlwaysInlining;

  Opts.CPU = TargetCPU;
  Opts.DebugInfo = GenerateDebugInfo;
  Opts.DisableLLVMOpts = DisableLLVMOptimizations;
  Opts.DisableRedZone = DisableRedZone;
  Opts.MergeAllConstants = !NoMergeConstants;
  Opts.NoCommon = NoCommon;
  Opts.NoImplicitFloat = NoImplicitFloat;
  Opts.OptimizeSize = OptSize;
  Opts.SimplifyLibCalls = 1;
  Opts.UnrollLoops = (Opts.OptimizationLevel > 1 && !OptSize);

#ifdef NDEBUG
  Opts.VerifyModule = 0;
#endif
}

void clang::InitializeDependencyOutputOptions(DependencyOutputOptions &Opts) {
  using namespace dependencyoutputoptions;

  Opts.OutputFile = DependencyFile;
  Opts.Targets = DependencyTargets;
  Opts.IncludeSystemHeaders = DependenciesIncludeSystemHeaders;
  Opts.UsePhonyTargets = PhonyDependencyTarget;
}

void clang::InitializeDiagnosticOptions(DiagnosticOptions &Opts) {
  using namespace diagnosticoptions;

  Opts.Warnings = OptWarnings;
  Opts.DumpBuildInformation = DumpBuildInformation;
  Opts.IgnoreWarnings = OptNoWarnings;
  Opts.MessageLength = MessageLength;
  Opts.NoRewriteMacros = SilenceRewriteMacroWarning;
  Opts.Pedantic = OptPedantic;
  Opts.PedanticErrors = OptPedanticErrors;
  Opts.ShowCarets = !NoCaretDiagnostics;
  Opts.ShowColors = PrintColorDiagnostic;
  Opts.ShowColumn = !NoShowColumn;
  Opts.ShowFixits = !NoDiagnosticsFixIt;
  Opts.ShowLocation = !NoShowLocation;
  Opts.ShowOptionNames = PrintDiagnosticOption;
  Opts.ShowSourceRanges = PrintSourceRangeInfo;
  Opts.VerifyDiagnostics = VerifyDiagnostics;
}

void clang::InitializeFrontendOptions(FrontendOptions &Opts) {
  using namespace frontendoptions;

  Opts.CodeCompletionAt = CodeCompletionAt;
  Opts.DebugCodeCompletionPrinter = CodeCompletionDebugPrinter;
  Opts.DisableFree = DisableFree;
  Opts.EmptyInputOnly = EmptyInputOnly;
  Opts.FixItAll = FixItAll;
  Opts.FixItLocations = FixItAtLocations;
  Opts.OutputFile = OutputFile;
  Opts.RelocatablePCH = RelocatablePCH;
  Opts.ShowMacrosInCodeCompletion = CodeCompletionWantsMacros;
  Opts.ShowStats = Stats;
  Opts.ShowTimers = TimeReport;
  Opts.TargetABI = TargetABI;
  Opts.TargetTriple = TargetTriple;
  Opts.ViewClassInheritance = InheritanceViewCls;

  // '-' is the default input if none is given.
  if (InputFilenames.empty()) {
    FrontendOptions::InputKind IK = InputType;
    if (IK == FrontendOptions::IK_None) IK = FrontendOptions::IK_C;
    Opts.Inputs.push_back(std::make_pair(IK, "-"));
  } else {
    for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
      FrontendOptions::InputKind IK = InputType;
      llvm::StringRef Ext =
        llvm::StringRef(InputFilenames[i]).rsplit('.').second;
      if (IK == FrontendOptions::IK_None)
        IK = FrontendOptions::getInputKindForExtension(Ext);
      Opts.Inputs.push_back(std::make_pair(IK, InputFilenames[i]));
                                           
    }
  }
}

void clang::InitializeHeaderSearchOptions(HeaderSearchOptions &Opts,
                                          llvm::StringRef BuiltinIncludePath,
                                          const LangOptions &Lang) {
  using namespace headersearchoptions;

  Opts.Sysroot = isysroot;
  Opts.Verbose = Verbose;

  // Handle -I... and -F... options, walking the lists in parallel.
  unsigned Iidx = 0, Fidx = 0;
  while (Iidx < I_dirs.size() && Fidx < F_dirs.size()) {
    if (I_dirs.getPosition(Iidx) < F_dirs.getPosition(Fidx)) {
      Opts.AddPath(I_dirs[Iidx], frontend::Angled, false, true, false);
      ++Iidx;
    } else {
      Opts.AddPath(F_dirs[Fidx], frontend::Angled, false, true, true);
      ++Fidx;
    }
  }

  // Consume what's left from whatever list was longer.
  for (; Iidx != I_dirs.size(); ++Iidx)
    Opts.AddPath(I_dirs[Iidx], frontend::Angled, false, true, false);
  for (; Fidx != F_dirs.size(); ++Fidx)
    Opts.AddPath(F_dirs[Fidx], frontend::Angled, false, true, true);

  // Handle -idirafter... options.
  for (unsigned i = 0, e = idirafter_dirs.size(); i != e; ++i)
    Opts.AddPath(idirafter_dirs[i], frontend::After,
        false, true, false);

  // Handle -iquote... options.
  for (unsigned i = 0, e = iquote_dirs.size(); i != e; ++i)
    Opts.AddPath(iquote_dirs[i], frontend::Quoted, false, true, false);

  // Handle -isystem... options.
  for (unsigned i = 0, e = isystem_dirs.size(); i != e; ++i)
    Opts.AddPath(isystem_dirs[i], frontend::System, false, true, false);

  // Walk the -iprefix/-iwithprefix/-iwithprefixbefore argument lists in
  // parallel, processing the values in order of occurance to get the right
  // prefixes.
  {
    std::string Prefix = "";  // FIXME: this isn't the correct default prefix.
    unsigned iprefix_idx = 0;
    unsigned iwithprefix_idx = 0;
    unsigned iwithprefixbefore_idx = 0;
    bool iprefix_done           = iprefix_vals.empty();
    bool iwithprefix_done       = iwithprefix_vals.empty();
    bool iwithprefixbefore_done = iwithprefixbefore_vals.empty();
    while (!iprefix_done || !iwithprefix_done || !iwithprefixbefore_done) {
      if (!iprefix_done &&
          (iwithprefix_done ||
           iprefix_vals.getPosition(iprefix_idx) <
           iwithprefix_vals.getPosition(iwithprefix_idx)) &&
          (iwithprefixbefore_done ||
           iprefix_vals.getPosition(iprefix_idx) <
           iwithprefixbefore_vals.getPosition(iwithprefixbefore_idx))) {
        Prefix = iprefix_vals[iprefix_idx];
        ++iprefix_idx;
        iprefix_done = iprefix_idx == iprefix_vals.size();
      } else if (!iwithprefix_done &&
                 (iwithprefixbefore_done ||
                  iwithprefix_vals.getPosition(iwithprefix_idx) <
                  iwithprefixbefore_vals.getPosition(iwithprefixbefore_idx))) {
        Opts.AddPath(Prefix+iwithprefix_vals[iwithprefix_idx],
                     frontend::System, false, false, false);
        ++iwithprefix_idx;
        iwithprefix_done = iwithprefix_idx == iwithprefix_vals.size();
      } else {
        Opts.AddPath(Prefix+iwithprefixbefore_vals[iwithprefixbefore_idx],
                     frontend::Angled, false, false, false);
        ++iwithprefixbefore_idx;
        iwithprefixbefore_done =
          iwithprefixbefore_idx == iwithprefixbefore_vals.size();
      }
    }
  }

  // Add CPATH environment paths.
  if (const char *Env = getenv("CPATH"))
    Opts.EnvIncPath = Env;

  // Add language specific environment paths.
  if (Lang.CPlusPlus && Lang.ObjC1) {
    if (const char *Env = getenv("OBJCPLUS_INCLUDE_PATH"))
      Opts.LangEnvIncPath = Env;
  } else if (Lang.CPlusPlus) {
    if (const char *Env = getenv("CPLUS_INCLUDE_PATH"))
      Opts.LangEnvIncPath = Env;
  } else if (Lang.ObjC1) {
    if (const char *Env = getenv("OBJC_INCLUDE_PATH"))
      Opts.LangEnvIncPath = Env;
  } else {
    if (const char *Env = getenv("C_INCLUDE_PATH"))
      Opts.LangEnvIncPath = Env;
  }

  if (!nobuiltininc)
    Opts.BuiltinIncludePath = BuiltinIncludePath;

  Opts.UseStandardIncludes = !nostdinc;
}

void clang::InitializePreprocessorOptions(PreprocessorOptions &Opts) {
  using namespace preprocessoroptions;

  Opts.setImplicitPCHInclude(ImplicitIncludePCH);
  Opts.setImplicitPTHInclude(ImplicitIncludePTH);

  // Select the token cache file, we don't support more than one currently so we
  // can't have both an implicit-pth and a token cache file.
  if (TokenCache.getPosition() && ImplicitIncludePTH.getPosition()) {
    // FIXME: Don't fail like this.
    fprintf(stderr, "error: cannot use both -token-cache and -include-pth "
            "options\n");
    exit(1);
  }
  if (TokenCache.getPosition())
    Opts.setTokenCache(TokenCache);
  else
    Opts.setTokenCache(ImplicitIncludePTH);

  // Use predefines?
  Opts.setUsePredefines(!UndefMacros);

  // Add macros from the command line.
  unsigned d = 0, D = D_macros.size();
  unsigned u = 0, U = U_macros.size();
  while (d < D || u < U) {
    if (u == U || (d < D && D_macros.getPosition(d) < U_macros.getPosition(u)))
      Opts.addMacroDef(D_macros[d++]);
    else
      Opts.addMacroUndef(U_macros[u++]);
  }

  // If -imacros are specified, include them now.  These are processed before
  // any -include directives.
  for (unsigned i = 0, e = ImplicitMacroIncludes.size(); i != e; ++i)
    Opts.addMacroInclude(ImplicitMacroIncludes[i]);

  // Add the ordered list of -includes, sorting in the implicit include options
  // at the appropriate location.
  llvm::SmallVector<std::pair<unsigned, std::string*>, 8> OrderedPaths;
  std::string OriginalFile;

  if (!ImplicitIncludePTH.empty())
    OrderedPaths.push_back(std::make_pair(ImplicitIncludePTH.getPosition(),
                                          &ImplicitIncludePTH));
  if (!ImplicitIncludePCH.empty()) {
    OriginalFile = PCHReader::getOriginalSourceFile(ImplicitIncludePCH);
    // FIXME: Don't fail like this.
    if (OriginalFile.empty())
      exit(1);
    OrderedPaths.push_back(std::make_pair(ImplicitIncludePCH.getPosition(),
                                          &OriginalFile));
  }
  for (unsigned i = 0, e = ImplicitIncludes.size(); i != e; ++i)
    OrderedPaths.push_back(std::make_pair(ImplicitIncludes.getPosition(i),
                                          &ImplicitIncludes[i]));
  llvm::array_pod_sort(OrderedPaths.begin(), OrderedPaths.end());

  for (unsigned i = 0, e = OrderedPaths.size(); i != e; ++i)
    Opts.addInclude(*OrderedPaths[i].second);
}

void clang::InitializeLangOptions(LangOptions &Options,
                                  FrontendOptions::InputKind IK,
                                  TargetInfo &Target,
                                  const CodeGenOptions &CodeGenOpts) {
  using namespace langoptions;

  bool NoPreprocess = false;

  switch (IK) {
  case FrontendOptions::IK_None:
  case FrontendOptions::IK_AST:
    assert(0 && "Invalid input kind!");
  case FrontendOptions::IK_Asm:
    Options.AsmPreprocessor = 1;
    // FALLTHROUGH
  case FrontendOptions::IK_PreprocessedC:
    NoPreprocess = true;
    // FALLTHROUGH
  case FrontendOptions::IK_C:
    // Do nothing.
    break;
  case FrontendOptions::IK_PreprocessedCXX:
    NoPreprocess = true;
    // FALLTHROUGH
  case FrontendOptions::IK_CXX:
    Options.CPlusPlus = 1;
    break;
  case FrontendOptions::IK_PreprocessedObjC:
    NoPreprocess = true;
    // FALLTHROUGH
  case FrontendOptions::IK_ObjC:
    Options.ObjC1 = Options.ObjC2 = 1;
    break;
  case FrontendOptions::IK_PreprocessedObjCXX:
    NoPreprocess = true;
    // FALLTHROUGH
  case FrontendOptions::IK_ObjCXX:
    Options.ObjC1 = Options.ObjC2 = 1;
    Options.CPlusPlus = 1;
    break;
  case FrontendOptions::IK_OpenCL:
    Options.OpenCL = 1;
    Options.AltiVec = 1;
    Options.CXXOperatorNames = 1;
    Options.LaxVectorConversions = 1;
    break;
  }

  if (ObjCExclusiveGC)
    Options.setGCMode(LangOptions::GCOnly);
  else if (ObjCEnableGC)
    Options.setGCMode(LangOptions::HybridGC);

  if (ObjCEnableGCBitmapPrint)
    Options.ObjCGCBitmapPrint = 1;

  if (AltiVec)
    Options.AltiVec = 1;

  if (PThread)
    Options.POSIXThreads = 1;

  Options.setVisibilityMode(SymbolVisibility);
  Options.OverflowChecking = OverflowChecking;


  // Allow the target to set the default the language options as it sees fit.
  Target.getDefaultLangOptions(Options);

  // Pass the map of target features to the target for validation and
  // processing.
  Target.HandleTargetFeatures(CodeGenOpts.Features);

  if (LangStd == lang_unspecified) {
    // Based on the base language, pick one.
    switch (IK) {
    case FrontendOptions::IK_None:
    case FrontendOptions::IK_AST:
      assert(0 && "Invalid input kind!");
    case FrontendOptions::IK_OpenCL:
      LangStd = lang_c99;
      break;
    case FrontendOptions::IK_Asm:
    case FrontendOptions::IK_C:
    case FrontendOptions::IK_PreprocessedC:
    case FrontendOptions::IK_ObjC:
    case FrontendOptions::IK_PreprocessedObjC:
      LangStd = lang_gnu99;
      break;
    case FrontendOptions::IK_CXX:
    case FrontendOptions::IK_PreprocessedCXX:
    case FrontendOptions::IK_ObjCXX:
    case FrontendOptions::IK_PreprocessedObjCXX:
      LangStd = lang_gnucxx98;
      break;
    }
  }

  switch (LangStd) {
  default: assert(0 && "Unknown language standard!");

  // Fall through from newer standards to older ones.  This isn't really right.
  // FIXME: Enable specifically the right features based on the language stds.
  case lang_gnucxx0x:
  case lang_cxx0x:
    Options.CPlusPlus0x = 1;
    // FALL THROUGH
  case lang_gnucxx98:
  case lang_cxx98:
    Options.CPlusPlus = 1;
    Options.CXXOperatorNames = !NoOperatorNames;
    // FALL THROUGH.
  case lang_gnu99:
  case lang_c99:
    Options.C99 = 1;
    Options.HexFloats = 1;
    // FALL THROUGH.
  case lang_gnu89:
    Options.BCPLComment = 1;  // Only for C99/C++.
    // FALL THROUGH.
  case lang_c94:
    Options.Digraphs = 1;     // C94, C99, C++.
    // FALL THROUGH.
  case lang_c89:
    break;
  }

  // GNUMode - Set if we're in gnu99, gnu89, gnucxx98, etc.
  switch (LangStd) {
  default: assert(0 && "Unknown language standard!");
  case lang_gnucxx0x:
  case lang_gnucxx98:
  case lang_gnu99:
  case lang_gnu89:
    Options.GNUMode = 1;
    break;
  case lang_cxx0x:
  case lang_cxx98:
  case lang_c99:
  case lang_c94:
  case lang_c89:
    Options.GNUMode = 0;
    break;
  }

  if (Options.CPlusPlus) {
    Options.C99 = 0;
    Options.HexFloats = 0;
  }

  if (LangStd == lang_c89 || LangStd == lang_c94 || LangStd == lang_gnu89)
    Options.ImplicitInt = 1;
  else
    Options.ImplicitInt = 0;

  // Mimicing gcc's behavior, trigraphs are only enabled if -trigraphs
  // is specified, or -std is set to a conforming mode.
  Options.Trigraphs = !Options.GNUMode;
  if (Trigraphs.getPosition())
    Options.Trigraphs = Trigraphs;  // Command line option wins if specified.

  // If in a conformant language mode (e.g. -std=c99) Blocks defaults to off
  // even if they are normally on for the target.  In GNU modes (e.g.
  // -std=gnu99) the default for blocks depends on the target settings.
  // However, blocks are not turned off when compiling Obj-C or Obj-C++ code.
  if (!Options.ObjC1 && !Options.GNUMode)
    Options.Blocks = 0;

  // Default to not accepting '$' in identifiers when preprocessing assembler,
  // but do accept when preprocessing C.  FIXME: these defaults are right for
  // darwin, are they right everywhere?
  Options.DollarIdents = IK != FrontendOptions::IK_Asm;
  if (DollarsInIdents.getPosition())  // Explicit setting overrides default.
    Options.DollarIdents = DollarsInIdents;

  if (PascalStrings.getPosition())
    Options.PascalStrings = PascalStrings;
  if (MSExtensions.getPosition())
    Options.Microsoft = MSExtensions;
  Options.WritableStrings = WritableStrings;
  if (NoLaxVectorConversions.getPosition())
      Options.LaxVectorConversions = 0;
  Options.Exceptions = Exceptions;
  Options.Rtti = Rtti;
  if (EnableBlocks.getPosition())
    Options.Blocks = EnableBlocks;
  if (CharIsSigned.getPosition())
    Options.CharIsSigned = CharIsSigned;
  if (ShortWChar.getPosition())
    Options.ShortWChar = ShortWChar;

  if (!AllowBuiltins)
    Options.NoBuiltin = 1;
  if (Freestanding)
    Options.Freestanding = Options.NoBuiltin = 1;

  if (EnableHeinousExtensions)
    Options.HeinousExtensions = 1;

  if (AccessControl)
    Options.AccessControl = 1;

  Options.ElideConstructors = !NoElideConstructors;

  // OpenCL and C++ both have bool, true, false keywords.
  Options.Bool = Options.OpenCL | Options.CPlusPlus;

  Options.MathErrno = MathErrno;

  Options.InstantiationDepth = TemplateDepth;

  // Override the default runtime if the user requested it.
  if (NeXTRuntime)
    Options.NeXTRuntime = 1;
  else if (GNURuntime)
    Options.NeXTRuntime = 0;

  if (!ObjCConstantStringClass.empty())
    Options.ObjCConstantStringClass = ObjCConstantStringClass.c_str();

  if (ObjCNonFragileABI)
    Options.ObjCNonFragileABI = 1;

  if (EmitAllDecls)
    Options.EmitAllDecls = 1;

  // The __OPTIMIZE_SIZE__ define is tied to -Oz, which we don't support.
  Options.OptimizeSize = 0;
  Options.Optimize = !!CodeGenOpts.OptimizationLevel;

  assert(PICLevel <= 2 && "Invalid value for -pic-level");
  Options.PICLevel = PICLevel;

  Options.GNUInline = !Options.C99;
  // FIXME: This is affected by other options (-fno-inline).

  // This is the __NO_INLINE__ define, which just depends on things like the
  // optimization level and -fno-inline, not actually whether the backend has
  // inlining enabled.
  Options.NoInline = !CodeGenOpts.OptimizationLevel;

  Options.Static = StaticDefine;

  switch (StackProtector) {
  default:
    assert(StackProtector <= 2 && "Invalid value for -stack-protector");
  case -1: break;
  case 0: Options.setStackProtectorMode(LangOptions::SSPOff); break;
  case 1: Options.setStackProtectorMode(LangOptions::SSPOn);  break;
  case 2: Options.setStackProtectorMode(LangOptions::SSPReq); break;
  }

  if (MainFileName.getPosition())
    Options.setMainFileName(MainFileName.c_str());

  Target.setForcedLangOptions(Options);
}

void
clang::InitializePreprocessorOutputOptions(PreprocessorOutputOptions &Opts) {
  using namespace preprocessoroutputoptions;

  Opts.ShowCPP = !DumpMacros;
  Opts.ShowMacros = DumpMacros || DumpDefines;
  Opts.ShowLineMarkers = !DisableLineMarkers;
  Opts.ShowComments = EnableCommentOutput;
  Opts.ShowMacroComments = EnableMacroCommentOutput;
}
