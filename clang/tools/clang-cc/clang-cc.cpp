//===--- clang.cpp - C-Language Front-end ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This utility may be invoked in the following manner:
//   clang --help                - Output help info.
//   clang [options]             - Read from stdin.
//   clang [options] file        - Read from "file".
//   clang [options] file1 file2 - Read these files.
//
//===----------------------------------------------------------------------===//
//
// TODO: Options to support:
//
//   -Wfatal-errors
//   -ftabstop=width
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/AnalysisConsumer.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/CompileOptions.h"
#include "clang/Frontend/DiagnosticOptions.h"
#include "clang/Frontend/FixItRewriter.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/HeaderSearchOptions.h"
#include "clang/Frontend/InitHeaderSearch.h"
#include "clang/Frontend/PCHReader.h"
#include "clang/Frontend/PathDiagnosticClients.h"
#include "clang/Frontend/PreprocessorOptions.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/Frontend/Utils.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/ParseAST.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Parse/Parser.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/TargetSelect.h"
#include <cstdlib>
#if HAVE_SYS_TYPES_H
#  include <sys/types.h>
#endif

using namespace clang;

//===----------------------------------------------------------------------===//
// Source Location Parser
//===----------------------------------------------------------------------===//

static bool ResolveParsedLocation(ParsedSourceLocation &ParsedLoc,
                                  FileManager &FileMgr,
                                  RequestedSourceLocation &Result) {
  const FileEntry *File = FileMgr.getFile(ParsedLoc.FileName);
  if (!File)
    return true;

  Result.File = File;
  Result.Line = ParsedLoc.Line;
  Result.Column = ParsedLoc.Column;
  return false;
}

//===----------------------------------------------------------------------===//
// Global options.
//===----------------------------------------------------------------------===//

/// ClangFrontendTimer - The front-end activities should charge time to it with
/// TimeRegion.  The -ftime-report option controls whether this will do
/// anything.
llvm::Timer *ClangFrontendTimer = 0;

static llvm::cl::opt<bool>
Verbose("v", llvm::cl::desc("Enable verbose output"));
static llvm::cl::opt<bool>
Stats("print-stats",
      llvm::cl::desc("Print performance metrics and statistics"));
static llvm::cl::opt<bool>
DisableFree("disable-free",
           llvm::cl::desc("Disable freeing of memory on exit"),
           llvm::cl::init(false));
static llvm::cl::opt<bool>
EmptyInputOnly("empty-input-only",
      llvm::cl::desc("Force running on an empty input file"));

enum ProgActions {
  RewriteObjC,                  // ObjC->C Rewriter.
  RewriteBlocks,                // ObjC->C Rewriter for Blocks.
  RewriteMacros,                // Expand macros but not #includes.
  RewriteTest,                  // Rewriter playground
  FixIt,                        // Fix-It Rewriter
  HTMLTest,                     // HTML displayer testing stuff.
  EmitAssembly,                 // Emit a .s file.
  EmitLLVM,                     // Emit a .ll file.
  EmitBC,                       // Emit a .bc file.
  EmitLLVMOnly,                 // Generate LLVM IR, but do not
  EmitHTML,                     // Translate input source into HTML.
  ASTPrint,                     // Parse ASTs and print them.
  ASTPrintXML,                  // Parse ASTs and print them in XML.
  ASTDump,                      // Parse ASTs and dump them.
  ASTView,                      // Parse ASTs and view them in Graphviz.
  PrintDeclContext,             // Print DeclContext and their Decls.
  DumpRecordLayouts,            // Dump record layout information.
  ParsePrintCallbacks,          // Parse and print each callback.
  ParseSyntaxOnly,              // Parse and perform semantic analysis.
  ParseNoop,                    // Parse with noop callbacks.
  RunPreprocessorOnly,          // Just lex, no output.
  PrintPreprocessedInput,       // -E mode.
  DumpTokens,                   // Dump out preprocessed tokens.
  DumpRawTokens,                // Dump out raw tokens.
  RunAnalysis,                  // Run one or more source code analyses.
  GeneratePTH,                  // Generate pre-tokenized header.
  GeneratePCH,                  // Generate pre-compiled header.
  InheritanceView               // View C++ inheritance for a specified class.
};

static llvm::cl::opt<ProgActions>
ProgAction(llvm::cl::desc("Choose output type:"), llvm::cl::ZeroOrMore,
           llvm::cl::init(ParseSyntaxOnly),
           llvm::cl::values(
             clEnumValN(RunPreprocessorOnly, "Eonly",
                        "Just run preprocessor, no output (for timings)"),
             clEnumValN(PrintPreprocessedInput, "E",
                        "Run preprocessor, emit preprocessed file"),
             clEnumValN(DumpRawTokens, "dump-raw-tokens",
                        "Lex file in raw mode and dump raw tokens"),
             clEnumValN(RunAnalysis, "analyze",
                        "Run static analysis engine"),
             clEnumValN(DumpTokens, "dump-tokens",
                        "Run preprocessor, dump internal rep of tokens"),
             clEnumValN(ParseNoop, "parse-noop",
                        "Run parser with noop callbacks (for timings)"),
             clEnumValN(ParseSyntaxOnly, "fsyntax-only",
                        "Run parser and perform semantic analysis"),
             clEnumValN(ParsePrintCallbacks, "parse-print-callbacks",
                        "Run parser and print each callback invoked"),
             clEnumValN(EmitHTML, "emit-html",
                        "Output input source as HTML"),
             clEnumValN(ASTPrint, "ast-print",
                        "Build ASTs and then pretty-print them"),
             clEnumValN(ASTPrintXML, "ast-print-xml",
                        "Build ASTs and then print them in XML format"),
             clEnumValN(ASTDump, "ast-dump",
                        "Build ASTs and then debug dump them"),
             clEnumValN(ASTView, "ast-view",
                        "Build ASTs and view them with GraphViz"),
             clEnumValN(PrintDeclContext, "print-decl-contexts",
                        "Print DeclContexts and their Decls"),
             clEnumValN(DumpRecordLayouts, "dump-record-layouts",
                        "Dump record layout information"),
             clEnumValN(GeneratePTH, "emit-pth",
                        "Generate pre-tokenized header file"),
             clEnumValN(GeneratePCH, "emit-pch",
                        "Generate pre-compiled header file"),
             clEnumValN(EmitAssembly, "S",
                        "Emit native assembly code"),
             clEnumValN(EmitLLVM, "emit-llvm",
                        "Build ASTs then convert to LLVM, emit .ll file"),
             clEnumValN(EmitBC, "emit-llvm-bc",
                        "Build ASTs then convert to LLVM, emit .bc file"),
             clEnumValN(EmitLLVMOnly, "emit-llvm-only",
                        "Build ASTs and convert to LLVM, discarding output"),
             clEnumValN(RewriteTest, "rewrite-test",
                        "Rewriter playground"),
             clEnumValN(RewriteObjC, "rewrite-objc",
                        "Rewrite ObjC into C (code rewriter example)"),
             clEnumValN(RewriteMacros, "rewrite-macros",
                        "Expand macros without full preprocessing"),
             clEnumValN(RewriteBlocks, "rewrite-blocks",
                        "Rewrite Blocks to C"),
             clEnumValN(FixIt, "fixit",
                        "Apply fix-it advice to the input source"),
             clEnumValEnd));


static llvm::cl::opt<std::string>
OutputFile("o",
 llvm::cl::value_desc("path"),
 llvm::cl::desc("Specify output file"));


enum CodeCompletionPrinter {
  CCP_Debug,
  CCP_CIndex
};

static llvm::cl::opt<ParsedSourceLocation>
CodeCompletionAt("code-completion-at",
                 llvm::cl::value_desc("file:line:column"),
              llvm::cl::desc("Dump code-completion information at a location"));

static llvm::cl::opt<CodeCompletionPrinter>
CodeCompletionPrinter("code-completion-printer",
                      llvm::cl::desc("Choose output type:"),
                      llvm::cl::init(CCP_Debug),
                      llvm::cl::values(
                        clEnumValN(CCP_Debug, "debug",
                          "Debug code-completion results"),
                        clEnumValN(CCP_CIndex, "cindex",
                          "Code-completion results for the CIndex library"),
                        clEnumValEnd));

static llvm::cl::opt<bool>
CodeCompletionWantsMacros("code-completion-macros",
                 llvm::cl::desc("Include macros in code-completion results"));

/// \brief Buld a new code-completion consumer that prints the results of
/// code completion to standard output.
static CodeCompleteConsumer *BuildPrintingCodeCompleter(Sema &S, void *) {
  switch (CodeCompletionPrinter.getValue()) {
  case CCP_Debug:
    return new PrintingCodeCompleteConsumer(S, CodeCompletionWantsMacros, 
                                            llvm::outs());
      
  case CCP_CIndex:
    return new CIndexCodeCompleteConsumer(S, CodeCompletionWantsMacros,
                                          llvm::outs());
  };
  
  return 0;
}

//===----------------------------------------------------------------------===//
// PTH.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
TokenCache("token-cache", llvm::cl::value_desc("path"),
           llvm::cl::desc("Use specified token cache file"));

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool>
VerifyDiagnostics("verify",
                  llvm::cl::desc("Verify emitted diagnostics and warnings"));

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

//===----------------------------------------------------------------------===//
// C++ Visualization.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
InheritanceViewCls("cxx-inheritance-view",
                   llvm::cl::value_desc("class name"),
                  llvm::cl::desc("View C++ inheritance for a specified class"));

//===----------------------------------------------------------------------===//
// Builtin Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool>
TimeReport("ftime-report",
           llvm::cl::desc("Print the amount of time each "
                          "phase of compilation takes"));

static llvm::cl::opt<bool>
Freestanding("ffreestanding",
             llvm::cl::desc("Assert that the compilation takes place in a "
                            "freestanding environment"));

static llvm::cl::opt<bool>
AllowBuiltins("fbuiltin", llvm::cl::init(true),
             llvm::cl::desc("Disable implicit builtin knowledge of functions"));


static llvm::cl::opt<bool>
MathErrno("fmath-errno", llvm::cl::init(true),
          llvm::cl::desc("Require math functions to respect errno"));

//===----------------------------------------------------------------------===//
// Language Options
//===----------------------------------------------------------------------===//

enum LangKind {
  langkind_unspecified,
  langkind_c,
  langkind_c_cpp,
  langkind_asm_cpp,
  langkind_cxx,
  langkind_cxx_cpp,
  langkind_objc,
  langkind_objc_cpp,
  langkind_objcxx,
  langkind_objcxx_cpp,
  langkind_ocl,
  langkind_ast
};

static llvm::cl::opt<LangKind>
BaseLang("x", llvm::cl::desc("Base language to compile"),
         llvm::cl::init(langkind_unspecified),
   llvm::cl::values(clEnumValN(langkind_c,     "c",            "C"),
                    clEnumValN(langkind_ocl,   "cl",           "OpenCL C"),
                    clEnumValN(langkind_cxx,   "c++",          "C++"),
                    clEnumValN(langkind_objc,  "objective-c",  "Objective C"),
                    clEnumValN(langkind_objcxx,"objective-c++","Objective C++"),
                    clEnumValN(langkind_c_cpp,     "cpp-output",
                               "Preprocessed C"),
                    clEnumValN(langkind_asm_cpp,     "assembler-with-cpp",
                               "Preprocessed asm"),
                    clEnumValN(langkind_cxx_cpp,   "c++-cpp-output",
                               "Preprocessed C++"),
                    clEnumValN(langkind_objc_cpp,  "objective-c-cpp-output",
                               "Preprocessed Objective C"),
                    clEnumValN(langkind_objcxx_cpp, "objective-c++-cpp-output",
                               "Preprocessed Objective C++"),
                    clEnumValN(langkind_c, "c-header",
                               "C header"),
                    clEnumValN(langkind_objc, "objective-c-header",
                               "Objective-C header"),
                    clEnumValN(langkind_cxx, "c++-header",
                               "C++ header"),
                    clEnumValN(langkind_objcxx, "objective-c++-header",
                               "Objective-C++ header"),
                    clEnumValN(langkind_ast, "ast",
                               "Clang AST"),
                    clEnumValEnd));

static llvm::cl::opt<bool>
ObjCExclusiveGC("fobjc-gc-only",
                llvm::cl::desc("Use GC exclusively for Objective-C related "
                               "memory management"));

static llvm::cl::opt<std::string>
ObjCConstantStringClass("fconstant-string-class",
                llvm::cl::value_desc("class name"),
                llvm::cl::desc("Specify the class to use for constant "
                               "Objective-C string objects."));

static llvm::cl::opt<bool>
ObjCEnableGC("fobjc-gc",
             llvm::cl::desc("Enable Objective-C garbage collection"));

static llvm::cl::opt<bool>
ObjCEnableGCBitmapPrint("print-ivar-layout",
             llvm::cl::desc("Enable Objective-C Ivar layout bitmap print trace"));

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

static llvm::cl::opt<bool>
OverflowChecking("ftrapv",
                 llvm::cl::desc("Trap on integer overflow"),
                 llvm::cl::init(false));

static llvm::cl::opt<bool>
AltiVec("faltivec", llvm::cl::desc("Enable AltiVec vector initializer syntax"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool>
PThread("pthread", llvm::cl::desc("Support POSIX threads in generated code"),
         llvm::cl::init(false));

static void InitializeCOptions(LangOptions &Options) {
    // Do nothing.
}

static void InitializeObjCOptions(LangOptions &Options) {
  Options.ObjC1 = Options.ObjC2 = 1;
}


static void InitializeLangOptions(LangOptions &Options, LangKind LK){
  // FIXME: implement -fpreprocessed mode.
  bool NoPreprocess = false;

  switch (LK) {
  default: assert(0 && "Unknown language kind!");
  case langkind_asm_cpp:
    Options.AsmPreprocessor = 1;
    // FALLTHROUGH
  case langkind_c_cpp:
    NoPreprocess = true;
    // FALLTHROUGH
  case langkind_c:
    InitializeCOptions(Options);
    break;
  case langkind_cxx_cpp:
    NoPreprocess = true;
    // FALLTHROUGH
  case langkind_cxx:
    Options.CPlusPlus = 1;
    break;
  case langkind_objc_cpp:
    NoPreprocess = true;
    // FALLTHROUGH
  case langkind_objc:
    InitializeObjCOptions(Options);
    break;
  case langkind_objcxx_cpp:
    NoPreprocess = true;
    // FALLTHROUGH
  case langkind_objcxx:
    Options.ObjC1 = Options.ObjC2 = 1;
    Options.CPlusPlus = 1;
    break;
  case langkind_ocl:
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
}

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
NoOperatorNames("fno-operator-names",
                llvm::cl::desc("Do not treat C++ operator name keywords as "
                               "synonyms for operators"));

static llvm::cl::opt<bool>
PascalStrings("fpascal-strings",
              llvm::cl::desc("Recognize and construct Pascal-style "
                             "string literals"));

static llvm::cl::opt<bool>
MSExtensions("fms-extensions",
             llvm::cl::desc("Accept some non-standard constructs used in "
                            "Microsoft header files "));

static llvm::cl::opt<bool>
WritableStrings("fwritable-strings",
              llvm::cl::desc("Store string literals as writable data"));

static llvm::cl::opt<bool>
NoLaxVectorConversions("fno-lax-vector-conversions",
                       llvm::cl::desc("Disallow implicit conversions between "
                                      "vectors with a different number of "
                                      "elements or different element types"));

static llvm::cl::opt<bool>
EnableBlocks("fblocks", llvm::cl::desc("enable the 'blocks' language feature"));

static llvm::cl::opt<bool>
EnableHeinousExtensions("fheinous-gnu-extensions",
   llvm::cl::desc("enable GNU extensions that you really really shouldn't use"),
                        llvm::cl::ValueDisallowed, llvm::cl::Hidden);

static llvm::cl::opt<bool>
ObjCNonFragileABI("fobjc-nonfragile-abi",
                  llvm::cl::desc("enable objective-c's nonfragile abi"));


static llvm::cl::opt<bool>
EmitAllDecls("femit-all-decls",
              llvm::cl::desc("Emit all declarations, even if unused"));

static llvm::cl::opt<bool>
Exceptions("fexceptions",
           llvm::cl::desc("Enable support for exception handling"));

static llvm::cl::opt<bool>
Rtti("frtti", llvm::cl::init(true),
     llvm::cl::desc("Enable generation of rtti information"));

static llvm::cl::opt<bool>
GNURuntime("fgnu-runtime",
            llvm::cl::desc("Generate output compatible with the standard GNU "
                           "Objective-C runtime"));

static llvm::cl::opt<bool>
NeXTRuntime("fnext-runtime",
            llvm::cl::desc("Generate output compatible with the NeXT "
                           "runtime"));

static llvm::cl::opt<bool>
CharIsSigned("fsigned-char",
    llvm::cl::desc("Force char to be a signed/unsigned type"));

static llvm::cl::opt<bool>
ShortWChar("fshort-wchar",
    llvm::cl::desc("Force wchar_t to be a short unsigned int"));


static llvm::cl::opt<bool>
Trigraphs("trigraphs", llvm::cl::desc("Process trigraph sequences"));

static llvm::cl::opt<unsigned>
TemplateDepth("ftemplate-depth", llvm::cl::init(99),
              llvm::cl::desc("Maximum depth of recursive template "
                             "instantiation"));
static llvm::cl::opt<bool>
DollarsInIdents("fdollars-in-identifiers",
                llvm::cl::desc("Allow '$' in identifiers"));


static llvm::cl::opt<bool>
OptSize("Os", llvm::cl::desc("Optimize for size"));

static llvm::cl::opt<bool>
DisableLLVMOptimizations("disable-llvm-optzns",
                         llvm::cl::desc("Don't run LLVM optimization passes"));

static llvm::cl::opt<bool>
NoCommon("fno-common",
         llvm::cl::desc("Compile common globals like normal definitions"),
         llvm::cl::ValueDisallowed);

static llvm::cl::opt<std::string>
MainFileName("main-file-name",
             llvm::cl::desc("Main file name to use for debug info"));

// FIXME: Also add an "-fno-access-control" option.
static llvm::cl::opt<bool>
AccessControl("faccess-control",
              llvm::cl::desc("Enable C++ access control"));

static llvm::cl::opt<bool>
NoElideConstructors("fno-elide-constructors",
                    llvm::cl::desc("Disable C++ copy constructor elision"));

static llvm::cl::opt<bool>
NoMergeConstants("fno-merge-all-constants",
                       llvm::cl::desc("Disallow merging of constants."));

static llvm::cl::opt<std::string>
TargetABI("target-abi",
          llvm::cl::desc("Target a particular ABI type"));

static llvm::cl::opt<std::string>
TargetTriple("triple",
  llvm::cl::desc("Specify target triple (e.g. i686-apple-darwin9)"));


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

static llvm::cl::opt<unsigned>
PICLevel("pic-level", llvm::cl::desc("Value for __PIC__"));

static llvm::cl::opt<bool>
StaticDefine("static-define", llvm::cl::desc("Should __STATIC__ be defined"));

static llvm::cl::opt<int>
StackProtector("stack-protector",
               llvm::cl::desc("Enable stack protectors"),
               llvm::cl::init(-1));

static void InitializeLanguageStandard(LangOptions &Options, LangKind LK,
                                       TargetInfo &Target,
                                       const llvm::StringMap<bool> &Features) {
  // Allow the target to set the default the language options as it sees fit.
  Target.getDefaultLangOptions(Options);

  // Pass the map of target features to the target for validation and
  // processing.
  Target.HandleTargetFeatures(Features);

  if (LangStd == lang_unspecified) {
    // Based on the base language, pick one.
    switch (LK) {
    case langkind_ast: assert(0 && "Invalid call for AST inputs");
    case lang_unspecified: assert(0 && "Unknown base language");
    case langkind_ocl:
      LangStd = lang_c99;
      break;
    case langkind_c:
    case langkind_asm_cpp:
    case langkind_c_cpp:
    case langkind_objc:
    case langkind_objc_cpp:
      LangStd = lang_gnu99;
      break;
    case langkind_cxx:
    case langkind_cxx_cpp:
    case langkind_objcxx:
    case langkind_objcxx_cpp:
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
  Options.DollarIdents = LK != langkind_asm_cpp;
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

  // The __OPTIMIZE_SIZE__ define is tied to -Oz, which we don't
  // support.
  Options.OptimizeSize = 0;

  // -Os implies -O2
  if (OptSize || OptLevel)
    Options.Optimize = 1;

  assert(PICLevel <= 2 && "Invalid value for -pic-level");
  Options.PICLevel = PICLevel;

  Options.GNUInline = !Options.C99;
  // FIXME: This is affected by other options (-fno-inline).
  Options.NoInline = !OptSize && !OptLevel;

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

//===----------------------------------------------------------------------===//
// SourceManager initialization.
//===----------------------------------------------------------------------===//

static bool InitializeSourceManager(Preprocessor &PP,
                                    const std::string &InFile) {
  // Figure out where to get and map in the main file.
  SourceManager &SourceMgr = PP.getSourceManager();
  FileManager &FileMgr = PP.getFileManager();

  if (EmptyInputOnly) {
    const char *EmptyStr = "";
    llvm::MemoryBuffer *SB =
      llvm::MemoryBuffer::getMemBuffer(EmptyStr, EmptyStr, "<empty input>");
    SourceMgr.createMainFileIDForMemBuffer(SB);
  } else if (InFile != "-") {
    const FileEntry *File = FileMgr.getFile(InFile);
    if (File) SourceMgr.createMainFileID(File, SourceLocation());
    if (SourceMgr.getMainFileID().isInvalid()) {
      PP.getDiagnostics().Report(FullSourceLoc(), diag::err_fe_error_reading)
        << InFile.c_str();
      return true;
    }
  } else {
    llvm::MemoryBuffer *SB = llvm::MemoryBuffer::getSTDIN();

    // If stdin was empty, SB is null.  Cons up an empty memory
    // buffer now.
    if (!SB) {
      const char *EmptyStr = "";
      SB = llvm::MemoryBuffer::getMemBuffer(EmptyStr, EmptyStr, "<stdin>");
    }

    SourceMgr.createMainFileIDForMemBuffer(SB);
    if (SourceMgr.getMainFileID().isInvalid()) {
      PP.getDiagnostics().Report(FullSourceLoc(),
                                 diag::err_fe_error_reading_stdin);
      return true;
    }
  }

  return false;
}


//===----------------------------------------------------------------------===//
// Preprocessor Initialization
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool>
UndefMacros("undef", llvm::cl::value_desc("macro"),
            llvm::cl::desc("undef all system defines"));

static llvm::cl::list<std::string>
D_macros("D", llvm::cl::value_desc("macro"), llvm::cl::Prefix,
       llvm::cl::desc("Predefine the specified macro"));
static llvm::cl::list<std::string>
U_macros("U", llvm::cl::value_desc("macro"), llvm::cl::Prefix,
         llvm::cl::desc("Undefine the specified macro"));

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

static llvm::cl::opt<bool>
RelocatablePCH("relocatable-pch",
               llvm::cl::desc("Whether to build a relocatable precompiled "
                              "header"));

//===----------------------------------------------------------------------===//
// Preprocessor include path information.
//===----------------------------------------------------------------------===//

// This tool exports a large number of command line options to control how the
// preprocessor searches for header files.  At root, however, the Preprocessor
// object takes a very simple interface: a list of directories to search for
//
// FIXME: -nostdinc++
// FIXME: -imultilib
//

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

// Finally, implement the code that groks the options above.

// Add the clang headers, which are relative to the clang binary.
std::string GetBuiltinIncludePath(const char *Argv0) {
  llvm::sys::Path P =
    llvm::sys::Path::GetMainExecutable(Argv0,
                                       (void*)(intptr_t) GetBuiltinIncludePath);

  if (!P.isEmpty()) {
    P.eraseComponent();  // Remove /clang from foo/bin/clang
    P.eraseComponent();  // Remove /bin   from foo/bin

    // Get foo/lib/clang/<version>/include
    P.appendComponent("lib");
    P.appendComponent("clang");
    P.appendComponent(CLANG_VERSION_STRING);
    P.appendComponent("include");
  }

  return P.str();
}

/// InitializeIncludePaths - Process the -I options and set them in the
/// HeaderSearch object.
void InitializeIncludePaths(const char *Argv0, HeaderSearch &Headers,
                            FileManager &FM, const LangOptions &Lang,
                            llvm::Triple &triple) {
  HeaderSearchOptions Opts(isysroot);

  Opts.Verbose = Verbose;

  // Handle -I... and -F... options, walking the lists in parallel.
  unsigned Iidx = 0, Fidx = 0;
  while (Iidx < I_dirs.size() && Fidx < F_dirs.size()) {
    if (I_dirs.getPosition(Iidx) < F_dirs.getPosition(Fidx)) {
      Opts.AddPath(I_dirs[Iidx], InitHeaderSearch::Angled, false, true, false);
      ++Iidx;
    } else {
      Opts.AddPath(F_dirs[Fidx], InitHeaderSearch::Angled, false, true, true);
      ++Fidx;
    }
  }

  // Consume what's left from whatever list was longer.
  for (; Iidx != I_dirs.size(); ++Iidx)
    Opts.AddPath(I_dirs[Iidx], InitHeaderSearch::Angled, false, true, false);
  for (; Fidx != F_dirs.size(); ++Fidx)
    Opts.AddPath(F_dirs[Fidx], InitHeaderSearch::Angled, false, true, true);

  // Handle -idirafter... options.
  for (unsigned i = 0, e = idirafter_dirs.size(); i != e; ++i)
    Opts.AddPath(idirafter_dirs[i], InitHeaderSearch::After,
        false, true, false);

  // Handle -iquote... options.
  for (unsigned i = 0, e = iquote_dirs.size(); i != e; ++i)
    Opts.AddPath(iquote_dirs[i], InitHeaderSearch::Quoted, false, true, false);

  // Handle -isystem... options.
  for (unsigned i = 0, e = isystem_dirs.size(); i != e; ++i)
    Opts.AddPath(isystem_dirs[i], InitHeaderSearch::System, false, true, false);

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
                InitHeaderSearch::System, false, false, false);
        ++iwithprefix_idx;
        iwithprefix_done = iwithprefix_idx == iwithprefix_vals.size();
      } else {
        Opts.AddPath(Prefix+iwithprefixbefore_vals[iwithprefixbefore_idx],
                InitHeaderSearch::Angled, false, false, false);
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
    Opts.BuiltinIncludePath = GetBuiltinIncludePath(Argv0);

  Opts.UseStandardIncludes = !nostdinc;

  // Apply all the options to the header search object.
  ApplyHeaderSearchOptions(Opts, Headers, Lang, triple);
}

void InitializePreprocessorOptions(PreprocessorOptions &InitOpts) {
  // Use predefines?
  InitOpts.setUsePredefines(!UndefMacros);

  // Add macros from the command line.
  unsigned d = 0, D = D_macros.size();
  unsigned u = 0, U = U_macros.size();
  while (d < D || u < U) {
    if (u == U || (d < D && D_macros.getPosition(d) < U_macros.getPosition(u)))
      InitOpts.addMacroDef(D_macros[d++]);
    else
      InitOpts.addMacroUndef(U_macros[u++]);
  }

  // If -imacros are specified, include them now.  These are processed before
  // any -include directives.
  for (unsigned i = 0, e = ImplicitMacroIncludes.size(); i != e; ++i)
    InitOpts.addMacroInclude(ImplicitMacroIncludes[i]);

  if (!ImplicitIncludePTH.empty() || !ImplicitIncludes.empty() ||
      (!ImplicitIncludePCH.empty() && ProgAction == PrintPreprocessedInput)) {
    // We want to add these paths to the predefines buffer in order, make a
    // temporary vector to sort by their occurrence.
    llvm::SmallVector<std::pair<unsigned, std::string*>, 8> OrderedPaths;

    if (!ImplicitIncludePTH.empty())
      OrderedPaths.push_back(std::make_pair(ImplicitIncludePTH.getPosition(),
                                            &ImplicitIncludePTH));
    if (!ImplicitIncludePCH.empty() && ProgAction == PrintPreprocessedInput)
      OrderedPaths.push_back(std::make_pair(ImplicitIncludePCH.getPosition(),
                                            &ImplicitIncludePCH));
    for (unsigned i = 0, e = ImplicitIncludes.size(); i != e; ++i)
      OrderedPaths.push_back(std::make_pair(ImplicitIncludes.getPosition(i),
                                            &ImplicitIncludes[i]));
    llvm::array_pod_sort(OrderedPaths.begin(), OrderedPaths.end());


    // Now that they are ordered by position, add to the predefines buffer.
    for (unsigned i = 0, e = OrderedPaths.size(); i != e; ++i) {
      std::string *Ptr = OrderedPaths[i].second;
      if (!ImplicitIncludes.empty() &&
          Ptr >= &ImplicitIncludes[0] &&
          Ptr <= &ImplicitIncludes[ImplicitIncludes.size()-1]) {
        InitOpts.addInclude(*Ptr, false);
      } else if (Ptr == &ImplicitIncludePTH) {
        InitOpts.addInclude(*Ptr, true);
      } else {
        // We end up here when we're producing preprocessed output and
        // we loaded a PCH file. In this case, just include the header
        // file that was used to build the precompiled header.
        assert(Ptr == &ImplicitIncludePCH);
        std::string OriginalFile = PCHReader::getOriginalSourceFile(*Ptr);
        if (!OriginalFile.empty()) {
          InitOpts.addInclude(OriginalFile, false);
          ImplicitIncludePCH.clear();
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Preprocessor construction
//===----------------------------------------------------------------------===//

static Preprocessor *
CreatePreprocessor(Diagnostic &Diags,const LangOptions &LangInfo,
                   TargetInfo &Target, SourceManager &SourceMgr,
                   HeaderSearch &HeaderInfo) {
  PTHManager *PTHMgr = 0;
  if (!TokenCache.empty() && !ImplicitIncludePTH.empty()) {
    fprintf(stderr, "error: cannot use both -token-cache and -include-pth "
            "options\n");
    exit(1);
  }

  // Use PTH?
  if (!TokenCache.empty() || !ImplicitIncludePTH.empty()) {
    const std::string& x = TokenCache.empty() ? ImplicitIncludePTH:TokenCache;
    PTHMgr = PTHManager::Create(x, &Diags,
                                TokenCache.empty() ? Diagnostic::Error
                                : Diagnostic::Warning);
  }

  if (Diags.hasErrorOccurred())
    exit(1);

  // Create the Preprocessor.
  Preprocessor *PP = new Preprocessor(Diags, LangInfo, Target,
                                      SourceMgr, HeaderInfo, PTHMgr);

  // Note that this is different then passing PTHMgr to Preprocessor's ctor.
  // That argument is used as the IdentifierInfoLookup argument to
  // IdentifierTable's ctor.
  if (PTHMgr) {
    PTHMgr->setPreprocessor(PP);
    PP->setPTHManager(PTHMgr);
  }

  PreprocessorOptions InitOpts;
  InitializePreprocessorOptions(InitOpts);
  InitializePreprocessor(*PP, InitOpts);

  return PP;
}

//===----------------------------------------------------------------------===//
// Basic Parser driver
//===----------------------------------------------------------------------===//

static void ParseFile(Preprocessor &PP, MinimalAction *PA) {
  Parser P(PP, *PA);
  PP.EnterMainSourceFile();

  // Parsing the specified input file.
  P.ParseTranslationUnit();
  delete PA;
}

//===----------------------------------------------------------------------===//
// Code generation options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool>
GenerateDebugInfo("g",
                  llvm::cl::desc("Generate source level debug information"));

static llvm::cl::opt<std::string>
TargetCPU("mcpu",
         llvm::cl::desc("Target a specific cpu type (-mcpu=help for details)"));

static llvm::cl::list<std::string>
TargetFeatures("target-feature", llvm::cl::desc("Target specific attributes"));


static llvm::cl::opt<bool>
DisableRedZone("disable-red-zone",
               llvm::cl::desc("Do not emit code that uses the red zone."),
               llvm::cl::init(false));

static llvm::cl::opt<bool>
NoImplicitFloat("no-implicit-float",
  llvm::cl::desc("Don't generate implicit floating point instructions (x86-only)"),
  llvm::cl::init(false));

/// ComputeTargetFeatures - Recompute the target feature list to only
/// be the list of things that are enabled, based on the target cpu
/// and feature list.
static void ComputeFeatureMap(const TargetInfo &Target,
                              llvm::StringMap<bool> &Features) {
  assert(Features.empty() && "invalid map");

  // Initialize the feature map based on the target.
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
    if (!Target.setFeatureEnabled(Features, Name + 1, (Name[0] == '+'))) {
      fprintf(stderr, "error: clang-cc: invalid target feature name: %s\n",
              Name + 1);
      exit(1);
    }
  }
}

static void InitializeCompileOptions(CompileOptions &Opts,
                                     const LangOptions &LangOpts,
                                     const llvm::StringMap<bool> &Features) {
  Opts.OptimizeSize = OptSize;
  Opts.DebugInfo = GenerateDebugInfo;

  if (DisableLLVMOptimizations) {
    Opts.OptimizationLevel = 0;
    Opts.Inlining = CompileOptions::NoInlining;
  } else {
    if (OptSize) {
      // -Os implies -O2
      Opts.OptimizationLevel = 2;
    } else {
      Opts.OptimizationLevel = OptLevel;
    }

    // We must always run at least the always inlining pass.
    if (Opts.OptimizationLevel > 1)
      Opts.Inlining = CompileOptions::NormalInlining;
    else
      Opts.Inlining = CompileOptions::OnlyAlwaysInlining;
  }

  // FIXME: There are llvm-gcc options to control these selectively.
  Opts.UnrollLoops = (Opts.OptimizationLevel > 1 && !OptSize);
  Opts.SimplifyLibCalls = !LangOpts.NoBuiltin;

#ifdef NDEBUG
  Opts.VerifyModule = 0;
#endif

  Opts.CPU = TargetCPU;
  Opts.Features.clear();
  for (llvm::StringMap<bool>::const_iterator it = Features.begin(),
         ie = Features.end(); it != ie; ++it) {
    // FIXME: If we are completely confident that we have the right
    // set, we only need to pass the minuses.
    std::string Name(it->second ? "+" : "-");
    Name += it->first();
    Opts.Features.push_back(Name);
  }

  Opts.NoCommon = NoCommon | LangOpts.CPlusPlus;

  // Handle -ftime-report.
  Opts.TimePasses = TimeReport;

  Opts.DisableRedZone = DisableRedZone;
  Opts.NoImplicitFloat = NoImplicitFloat;
  
  Opts.MergeAllConstants = !NoMergeConstants;
}

//===----------------------------------------------------------------------===//
// Fix-It Options
//===----------------------------------------------------------------------===//
static llvm::cl::list<ParsedSourceLocation>
FixItAtLocations("fixit-at", llvm::cl::value_desc("source-location"),
   llvm::cl::desc("Perform Fix-It modifications at the given source location"));

//===----------------------------------------------------------------------===//
// ObjC Rewriter Options
//===----------------------------------------------------------------------===//
static llvm::cl::opt<bool>
SilenceRewriteMacroWarning("Wno-rewrite-macros", llvm::cl::init(false),
                           llvm::cl::desc("Silence ObjC rewriting warnings"));

//===----------------------------------------------------------------------===//
// Warning Options
//===----------------------------------------------------------------------===//

// This gets all -W options, including -Werror, -W[no-]system-headers, etc.  The
// driver has stripped off -Wa,foo etc.  The driver has also translated -W to
// -Wextra, so we don't need to worry about it.
static llvm::cl::list<std::string>
OptWarnings("W", llvm::cl::Prefix, llvm::cl::ValueOptional);

static llvm::cl::opt<bool> OptPedantic("pedantic");
static llvm::cl::opt<bool> OptPedanticErrors("pedantic-errors");
static llvm::cl::opt<bool> OptNoWarnings("w");

//===----------------------------------------------------------------------===//
// Preprocessing (-E mode) Options
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// Dependency file options
//===----------------------------------------------------------------------===//
static llvm::cl::opt<std::string>
DependencyFile("dependency-file",
               llvm::cl::desc("Filename (or -) to write dependency output to"));

static llvm::cl::opt<bool>
DependenciesIncludeSystemHeaders("sys-header-deps",
                 llvm::cl::desc("Include system headers in dependency output"));

static llvm::cl::list<std::string>
DependencyTargets("MT",
         llvm::cl::desc("Specify target for dependency"));

// FIXME: Implement feature
static llvm::cl::opt<bool>
PhonyDependencyTarget("MP",
            llvm::cl::desc("Create phony target for each dependency "
                           "(other than main file)"));

//===----------------------------------------------------------------------===//
// Analysis options
//===----------------------------------------------------------------------===//

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
VisualizeEGDot("analyzer-viz-egraph-graphviz",
               llvm::cl::desc("Display exploded graph using GraphViz"));

static llvm::cl::opt<bool>
VisualizeEGUbi("analyzer-viz-egraph-ubigraph",
               llvm::cl::desc("Display exploded graph using Ubigraph"));

static llvm::cl::opt<bool>
AnalyzeAll("analyzer-opt-analyze-headers",
    llvm::cl::desc("Force the static analyzer to analyze "
                   "functions defined in header files"));

static llvm::cl::opt<bool>
AnalyzerDisplayProgress("analyzer-display-progress",
          llvm::cl::desc("Emit verbose output about the analyzer's progress."));

static llvm::cl::opt<bool>
PurgeDead("analyzer-purge-dead",
          llvm::cl::init(true),
          llvm::cl::desc("Remove dead symbols, bindings, and constraints before"
                         " processing a statement."));

static llvm::cl::opt<bool>
EagerlyAssume("analyzer-eagerly-assume",
          llvm::cl::init(false),
              llvm::cl::desc("Eagerly assume the truth/falseness of some "
                             "symbolic constraints."));

static llvm::cl::opt<std::string>
AnalyzeSpecificFunction("analyze-function",
               llvm::cl::desc("Run analysis on specific function"));

static llvm::cl::opt<bool>
TrimGraph("trim-egraph",
     llvm::cl::desc("Only show error-related paths in the analysis graph"));

static AnalyzerOptions ReadAnalyzerOptions() {
  AnalyzerOptions Opts;
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
  Opts.TrimGraph = TrimGraph;
  return Opts;
}

//===----------------------------------------------------------------------===//
// -dump-build-information Stuff
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
DumpBuildInformation("dump-build-information",
                     llvm::cl::value_desc("filename"),
          llvm::cl::desc("output a dump of some build information to a file"));

static llvm::raw_ostream *BuildLogFile = 0;

/// LoggingDiagnosticClient - This is a simple diagnostic client that forwards
/// all diagnostics to both BuildLogFile and a chained DiagnosticClient.
namespace {
class LoggingDiagnosticClient : public DiagnosticClient {
  llvm::OwningPtr<DiagnosticClient> Chain1;
  llvm::OwningPtr<DiagnosticClient> Chain2;
public:

  LoggingDiagnosticClient(const DiagnosticOptions &DiagOpts,
                          DiagnosticClient *Normal) {
    // Output diags both where requested...
    Chain1.reset(Normal);
    // .. and to our log file.
    Chain2.reset(new TextDiagnosticPrinter(*BuildLogFile, DiagOpts));
  }

  virtual void BeginSourceFile(const LangOptions &LO) {
    Chain1->BeginSourceFile(LO);
    Chain2->BeginSourceFile(LO);
  }

  virtual void EndSourceFile() {
    Chain1->EndSourceFile();
    Chain2->EndSourceFile();
  }

  virtual bool IncludeInDiagnosticCounts() const {
    return Chain1->IncludeInDiagnosticCounts();
  }

  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                const DiagnosticInfo &Info) {
    Chain1->HandleDiagnostic(DiagLevel, Info);
    Chain2->HandleDiagnostic(DiagLevel, Info);
  }
};
} // end anonymous namespace.

static void SetUpBuildDumpLog(const DiagnosticOptions &DiagOpts,
                              unsigned argc, char **argv,
                              llvm::OwningPtr<DiagnosticClient> &DiagClient) {

  std::string ErrorInfo;
  BuildLogFile = new llvm::raw_fd_ostream(DumpBuildInformation.c_str(),
                                          ErrorInfo);

  if (!ErrorInfo.empty()) {
    llvm::errs() << "error opening -dump-build-information file '"
                 << DumpBuildInformation << "', option ignored!\n";
    delete BuildLogFile;
    BuildLogFile = 0;
    DumpBuildInformation = "";
    return;
  }

  (*BuildLogFile) << "clang-cc command line arguments: ";
  for (unsigned i = 0; i != argc; ++i)
    (*BuildLogFile) << argv[i] << ' ';
  (*BuildLogFile) << '\n';

  // LoggingDiagnosticClient - Insert a new logging diagnostic client in between
  // the diagnostic producers and the normal receiver.
  DiagClient.reset(new LoggingDiagnosticClient(DiagOpts, DiagClient.take()));
}



//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

static llvm::raw_ostream *ComputeOutFile(const CompilerInvocation &CompOpts,
                                         const std::string &InFile,
                                         const char *Extension,
                                         bool Binary,
                                         llvm::sys::Path& OutPath) {
  llvm::raw_ostream *Ret;
  std::string OutFile;
  if (!CompOpts.getOutputFile().empty())
    OutFile = CompOpts.getOutputFile();
  else if (InFile == "-") {
    OutFile = "-";
  } else if (Extension) {
    llvm::sys::Path Path(InFile);
    Path.eraseSuffix();
    Path.appendSuffix(Extension);
    OutFile = Path.str();
  } else {
    OutFile = "-";
  }

  std::string Error;
  Ret = new llvm::raw_fd_ostream(OutFile.c_str(), Error,
                                 (Binary ? llvm::raw_fd_ostream::F_Binary : 0));
  if (!Error.empty()) {
    // FIXME: Don't fail this way.
    llvm::errs() << "ERROR: " << Error << "\n";
    ::exit(1);
  }

  if (OutFile != "-")
    OutPath = OutFile;

  return Ret;
}

static ASTConsumer *CreateConsumerAction(const CompilerInvocation &CompOpts,
                                         Preprocessor &PP,
                                         const std::string &InFile,
                                         ProgActions PA,
                                         llvm::OwningPtr<llvm::raw_ostream> &OS,
                                         llvm::sys::Path &OutPath,
                                         const llvm::StringMap<bool> &Features,
                                         llvm::LLVMContext& Context) {
  switch (PA) {
  default:
    return 0;

  case ASTPrint:
    OS.reset(ComputeOutFile(CompOpts, InFile, 0, false, OutPath));
    return CreateASTPrinter(OS.get());

  case ASTPrintXML:
    OS.reset(ComputeOutFile(CompOpts, InFile, "xml", false, OutPath));
    return CreateASTPrinterXML(OS.get());

  case ASTDump:
    return CreateASTDumper();

  case ASTView:
    return CreateASTViewer();

  case PrintDeclContext:
    return CreateDeclContextPrinter();

  case DumpRecordLayouts:
    return CreateRecordLayoutDumper();

  case InheritanceView:
    return CreateInheritanceViewer(InheritanceViewCls);

  case EmitAssembly:
  case EmitLLVM:
  case EmitBC:
  case EmitLLVMOnly: {
    BackendAction Act;
    if (ProgAction == EmitAssembly) {
      Act = Backend_EmitAssembly;
      OS.reset(ComputeOutFile(CompOpts, InFile, "s", true, OutPath));
    } else if (ProgAction == EmitLLVM) {
      Act = Backend_EmitLL;
      OS.reset(ComputeOutFile(CompOpts, InFile, "ll", true, OutPath));
    } else if (ProgAction == EmitLLVMOnly) {
      Act = Backend_EmitNothing;
    } else {
      Act = Backend_EmitBC;
      OS.reset(ComputeOutFile(CompOpts, InFile, "bc", true, OutPath));
    }

    CompileOptions Opts;
    InitializeCompileOptions(Opts, PP.getLangOptions(), Features);
    return CreateBackendConsumer(Act, PP.getDiagnostics(), PP.getLangOptions(),
                                 Opts, InFile, OS.get(), Context);
  }

  case RewriteObjC:
    OS.reset(ComputeOutFile(CompOpts, InFile, "cpp", true, OutPath));
    return CreateObjCRewriter(InFile, OS.get(), PP.getDiagnostics(),
                              PP.getLangOptions(), SilenceRewriteMacroWarning);

  case RewriteBlocks:
    return CreateBlockRewriter(InFile, PP.getDiagnostics(),
                               PP.getLangOptions());
  }
}

/// ProcessInputFile - Process a single input file with the specified state.
///
static void ProcessInputFile(const CompilerInvocation &CompOpts,
                             Preprocessor &PP, const std::string &InFile,
                             ProgActions PA,
                             const llvm::StringMap<bool> &Features,
                             llvm::LLVMContext& Context) {
  llvm::OwningPtr<llvm::raw_ostream> OS;
  llvm::OwningPtr<ASTConsumer> Consumer;
  bool ClearSourceMgr = false;
  FixItRewriter *FixItRewrite = 0;
  bool CompleteTranslationUnit = true;
  llvm::sys::Path OutPath;

  switch (PA) {
  default:
    Consumer.reset(CreateConsumerAction(CompOpts, PP, InFile, PA, OS, OutPath,
                                        Features, Context));

    if (!Consumer.get()) {
      PP.getDiagnostics().Report(FullSourceLoc(),
                                 diag::err_fe_invalid_ast_action);
      return;
    }

    break;;

  case EmitHTML:
    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    Consumer.reset(CreateHTMLPrinter(OS.get(), PP));
    break;

  case RunAnalysis:
    Consumer.reset(CreateAnalysisConsumer(PP, CompOpts.getOutputFile(),
                                          ReadAnalyzerOptions()));
    break;

  case GeneratePCH:
    if (RelocatablePCH.getValue() && !isysroot.getNumOccurrences()) {
      PP.Diag(SourceLocation(), diag::err_relocatable_without_without_isysroot);
      RelocatablePCH.setValue(false);
    }

    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    if (RelocatablePCH.getValue())
      Consumer.reset(CreatePCHGenerator(PP, OS.get(), isysroot.c_str()));
    else
      Consumer.reset(CreatePCHGenerator(PP, OS.get()));
    CompleteTranslationUnit = false;
    break;

  case DumpRawTokens: {
    llvm::TimeRegion Timer(ClangFrontendTimer);
    SourceManager &SM = PP.getSourceManager();
    // Start lexing the specified input file.
    Lexer RawLex(SM.getMainFileID(), SM, PP.getLangOptions());
    RawLex.SetKeepWhitespaceMode(true);

    Token RawTok;
    RawLex.LexFromRawLexer(RawTok);
    while (RawTok.isNot(tok::eof)) {
      PP.DumpToken(RawTok, true);
      fprintf(stderr, "\n");
      RawLex.LexFromRawLexer(RawTok);
    }
    ClearSourceMgr = true;
    break;
  }
  case DumpTokens: {                 // Token dump mode.
    llvm::TimeRegion Timer(ClangFrontendTimer);
    Token Tok;
    // Start preprocessing the specified input file.
    PP.EnterMainSourceFile();
    do {
      PP.Lex(Tok);
      PP.DumpToken(Tok, true);
      fprintf(stderr, "\n");
    } while (Tok.isNot(tok::eof));
    ClearSourceMgr = true;
    break;
  }
  case RunPreprocessorOnly:
    break;

  case GeneratePTH: {
    llvm::TimeRegion Timer(ClangFrontendTimer);
    if (CompOpts.getOutputFile().empty() || CompOpts.getOutputFile() == "-") {
      // FIXME: Don't fail this way.
      // FIXME: Verify that we can actually seek in the given file.
      llvm::errs() << "ERROR: PTH requires an seekable file for output!\n";
      ::exit(1);
    }
    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    CacheTokens(PP, static_cast<llvm::raw_fd_ostream*>(OS.get()));
    ClearSourceMgr = true;
    break;
  }

  case PrintPreprocessedInput:
    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    break;

  case ParseNoop:
    break;

  case ParsePrintCallbacks: {
    llvm::TimeRegion Timer(ClangFrontendTimer);
    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    ParseFile(PP, CreatePrintParserActionsAction(PP, OS.get()));
    ClearSourceMgr = true;
    break;
  }

  case ParseSyntaxOnly: {             // -fsyntax-only
    llvm::TimeRegion Timer(ClangFrontendTimer);
    Consumer.reset(new ASTConsumer());
    break;
  }

  case RewriteMacros:
    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    RewriteMacrosInInput(PP, OS.get());
    ClearSourceMgr = true;
    break;

  case RewriteTest:
    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    DoRewriteTest(PP, OS.get());
    ClearSourceMgr = true;
    break;

  case FixIt:
    llvm::TimeRegion Timer(ClangFrontendTimer);
    Consumer.reset(new ASTConsumer());
    FixItRewrite = new FixItRewriter(PP.getDiagnostics(),
                                     PP.getSourceManager(),
                                     PP.getLangOptions());
    break;
  }

  if (FixItAtLocations.size() > 0) {
    // Even without the "-fixit" flag, with may have some specific
    // locations where the user has requested fixes. Process those
    // locations now.
    if (!FixItRewrite)
      FixItRewrite = new FixItRewriter(PP.getDiagnostics(),
                                       PP.getSourceManager(),
                                       PP.getLangOptions());

    bool AddedFixitLocation = false;
    for (unsigned Idx = 0, Last = FixItAtLocations.size();
         Idx != Last; ++Idx) {
      RequestedSourceLocation Requested;
      if (ResolveParsedLocation(FixItAtLocations[Idx],
                                PP.getFileManager(), Requested)) {
        fprintf(stderr, "FIX-IT could not find file \"%s\"\n",
                FixItAtLocations[Idx].FileName.c_str());
      } else {
        FixItRewrite->addFixItLocation(Requested);
        AddedFixitLocation = true;
      }
    }

    if (!AddedFixitLocation) {
      // All of the fix-it locations were bad. Don't fix anything.
      delete FixItRewrite;
      FixItRewrite = 0;
    }
  }

  llvm::OwningPtr<ASTContext> ContextOwner;
  if (Consumer)
    ContextOwner.reset(new ASTContext(PP.getLangOptions(),
                                      PP.getSourceManager(),
                                      PP.getTargetInfo(),
                                      PP.getIdentifierTable(),
                                      PP.getSelectorTable(),
                                      PP.getBuiltinInfo(),
                                      /* FreeMemory = */ !DisableFree,
                                      /* size_reserve = */0));

  llvm::OwningPtr<PCHReader> Reader;
  llvm::OwningPtr<ExternalASTSource> Source;

  if (!ImplicitIncludePCH.empty()) {
    // If the user specified -isysroot, it will be used for relocatable PCH
    // files.
    const char *isysrootPCH = 0;
    if (isysroot.getNumOccurrences() != 0)
      isysrootPCH = isysroot.c_str();

    Reader.reset(new PCHReader(PP, ContextOwner.get(), isysrootPCH));

    // The user has asked us to include a precompiled header. Load
    // the precompiled header into the AST context.
    switch (Reader->ReadPCH(ImplicitIncludePCH)) {
    case PCHReader::Success: {
      // Set the predefines buffer as suggested by the PCH
      // reader. Typically, the predefines buffer will be empty.
      PP.setPredefines(Reader->getSuggestedPredefines());

      // Attach the PCH reader to the AST context as an external AST
      // source, so that declarations will be deserialized from the
      // PCH file as needed.
      if (ContextOwner) {
        Source.reset(Reader.take());
        ContextOwner->setExternalSource(Source);
      }
      break;
    }

    case PCHReader::Failure:
      // Unrecoverable failure: don't even try to process the input
      // file.
      return;

    case PCHReader::IgnorePCH:
      // No suitable PCH file could be found. Return an error.
      return;

#if 0
      // FIXME: We can recover from failed attempts to load PCH
      // files. This code will do so, if we ever want to enable it.

      // We delayed the initialization of builtins in the hope of
      // loading the PCH file. Since the PCH file could not be
      // loaded, initialize builtins now.
      if (ContextOwner)
        ContextOwner->InitializeBuiltins(PP.getIdentifierTable());
#endif
    }

    // Finish preprocessor initialization. We do this now (rather
    // than earlier) because this initialization creates new source
    // location entries in the source manager, which must come after
    // the source location entries for the PCH file.
    if (InitializeSourceManager(PP, InFile))
      return;
  }

  // If we have an ASTConsumer, run the parser with it.
  if (Consumer) {
    CodeCompleteConsumer *(*CreateCodeCompleter)(Sema &, void *) = 0;
    void *CreateCodeCompleterData = 0;

    if (!CodeCompletionAt.FileName.empty()) {
      // Tell the source manager to chop off the given file at a specific
      // line and column.
      if (const FileEntry *Entry
            = PP.getFileManager().getFile(CodeCompletionAt.FileName)) {
        // Truncate the named file at the given line/column.
        PP.getSourceManager().truncateFileAt(Entry, CodeCompletionAt.Line,
                                             CodeCompletionAt.Column);

        // Set up the creation routine for code-completion.
        CreateCodeCompleter = BuildPrintingCodeCompleter;
      } else {
        PP.getDiagnostics().Report(FullSourceLoc(),
                                   diag::err_fe_invalid_code_complete_file)
          << CodeCompletionAt.FileName;
      }
    }

    ParseAST(PP, Consumer.get(), *ContextOwner.get(), Stats,
             CompleteTranslationUnit,
             CreateCodeCompleter, CreateCodeCompleterData);
  }

  // Perform post processing actions and actions which don't use a consumer.
  switch (PA) {
  default: break;

  case RunPreprocessorOnly: {    // Just lex as fast as we can, no output.
    llvm::TimeRegion Timer(ClangFrontendTimer);
    Token Tok;
    // Start parsing the specified input file.
    PP.EnterMainSourceFile();
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::eof));
    ClearSourceMgr = true;
    break;
  }

  case ParseNoop: {
    llvm::TimeRegion Timer(ClangFrontendTimer);
    ParseFile(PP, new MinimalAction(PP));
    ClearSourceMgr = true;
    break;
  }

  case PrintPreprocessedInput: {
    llvm::TimeRegion Timer(ClangFrontendTimer);
    if (DumpMacros)
      DoPrintMacros(PP, OS.get());
    else
      DoPrintPreprocessedInput(PP, OS.get(), EnableCommentOutput,
                               EnableMacroCommentOutput,
                               DisableLineMarkers, DumpDefines);
    ClearSourceMgr = true;
  }

  }

  if (FixItRewrite)
    FixItRewrite->WriteFixedFile(InFile, CompOpts.getOutputFile());

  // Disable the consumer prior to the context, the consumer may perform actions
  // in its destructor which require the context.
  if (DisableFree)
    Consumer.take();
  else
    Consumer.reset();

  // If in -disable-free mode, don't deallocate ASTContext.
  if (DisableFree)
    ContextOwner.take();
  else
    ContextOwner.reset(); // Delete ASTContext

  if (VerifyDiagnostics)
    if (CheckDiagnostics(PP))
      exit(1);

  if (Stats) {
    fprintf(stderr, "\nSTATISTICS FOR '%s':\n", InFile.c_str());
    PP.PrintStats();
    PP.getIdentifierTable().PrintStats();
    PP.getHeaderSearchInfo().PrintStats();
    PP.getSourceManager().PrintStats();
    fprintf(stderr, "\n");
  }

  // For a multi-file compilation, some things are ok with nuking the source
  // manager tables, other require stable fileid/macroid's across multiple
  // files.
  if (ClearSourceMgr)
    PP.getSourceManager().clearIDTables();

  // Always delete the output stream because we don't want to leak file
  // handles.  Also, we don't want to try to erase an open file.
  OS.reset();

  // If we had errors, try to erase the output file.
  if (PP.getDiagnostics().getNumErrors() && !OutPath.isEmpty())
    OutPath.eraseFromDisk();
}

/// ProcessInputFile - Process a single AST input file with the specified state.
///
static void ProcessASTInputFile(const CompilerInvocation &CompOpts,
                                const std::string &InFile, ProgActions PA,
                                const llvm::StringMap<bool> &Features,
                                Diagnostic &Diags, FileManager &FileMgr,
                                llvm::LLVMContext& Context) {
  std::string Error;
  llvm::OwningPtr<ASTUnit> AST(ASTUnit::LoadFromPCHFile(InFile, &Error));
  if (!AST) {
    Diags.Report(FullSourceLoc(), diag::err_fe_invalid_ast_file) << Error;
    return;
  }

  Preprocessor &PP = AST->getPreprocessor();

  llvm::OwningPtr<llvm::raw_ostream> OS;
  llvm::sys::Path OutPath;
  llvm::OwningPtr<ASTConsumer> Consumer(CreateConsumerAction(CompOpts, PP, 
                                                             InFile, PA, OS,
                                                             OutPath, Features,
                                                             Context));

  if (!Consumer.get()) {
    Diags.Report(FullSourceLoc(), diag::err_fe_invalid_ast_action);
    return;
  }

  // Set the main file ID to an empty file.
  //
  // FIXME: We probably shouldn't need this, but for now this is the simplest
  // way to reuse the logic in ParseAST.
  const char *EmptyStr = "";
  llvm::MemoryBuffer *SB =
    llvm::MemoryBuffer::getMemBuffer(EmptyStr, EmptyStr, "<dummy input>");
  AST->getSourceManager().createMainFileIDForMemBuffer(SB);

  // Stream the input AST to the consumer.
  Diags.getClient()->BeginSourceFile(PP.getLangOptions());
  ParseAST(PP, Consumer.get(), AST->getASTContext(), Stats);
  Diags.getClient()->EndSourceFile();

  // Release the consumer and the AST, in that order since the consumer may
  // perform actions in its destructor which require the context.
  if (DisableFree) {
    Consumer.take();
    AST.take();
  } else {
    Consumer.reset();
    AST.reset();
  }

  // Always delete the output stream because we don't want to leak file
  // handles.  Also, we don't want to try to erase an open file.
  OS.reset();

  // If we had errors, try to erase the output file.
  if (PP.getDiagnostics().getNumErrors() && !OutPath.isEmpty())
    OutPath.eraseFromDisk();
}

static llvm::cl::list<std::string>
InputFilenames(llvm::cl::Positional, llvm::cl::desc("<input files>"));

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
  Diagnostic &Diags = *static_cast<Diagnostic*>(UserData);

  Diags.Report(FullSourceLoc(), diag::err_fe_error_backend) << Message;

  // We cannot recover from llvm errors.
  exit(1);
}

static LangKind GetLanguage() {
  // If -x was given, that's the language.
  if (BaseLang != langkind_unspecified)
    return BaseLang;

  // Otherwise guess it from the input filenames;
  LangKind LK = langkind_unspecified;
  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    llvm::StringRef Name(InputFilenames[i]);
    LangKind ThisKind =  llvm::StringSwitch<LangKind>(Name.rsplit('.').second)
      .Case("ast", langkind_ast)
      .Case("c", langkind_c)
      .Cases("S", "s", langkind_asm_cpp)
      .Case("i", langkind_c_cpp)
      .Case("ii", langkind_cxx_cpp)
      .Case("m", langkind_objc)
      .Case("mi", langkind_objc_cpp)
      .Cases("mm", "M", langkind_objcxx)
      .Case("mii", langkind_objcxx_cpp)
      .Case("C", langkind_cxx)
      .Cases("C", "cc", "cp", langkind_cxx)
      .Cases("cpp", "CPP", "c++", "cxx", langkind_cxx)
      .Case("cl", langkind_ocl)
      .Default(langkind_c);

    if (LK != langkind_unspecified && ThisKind != LK) {
      llvm::errs() << "error: cannot have multiple input files of distinct "
                   << "language kinds without -x\n";
      exit(1);
    }

    LK = ThisKind;
  }

  return LK;
}

static void ConstructDiagnosticOptions(DiagnosticOptions &Opts) {
  // Initialize the diagnostic options.
  Opts.ShowColumn = !NoShowColumn;
  Opts.ShowLocation = !NoShowLocation;
  Opts.ShowCarets = !NoCaretDiagnostics;
  Opts.ShowFixits = !NoDiagnosticsFixIt;
  Opts.ShowSourceRanges = PrintSourceRangeInfo;
  Opts.ShowOptionNames = PrintDiagnosticOption;
  Opts.ShowColors = PrintColorDiagnostic;
  Opts.MessageLength = MessageLength;
}

static void ConstructCompilerInvocation(CompilerInvocation &Opts,
                                        const DiagnosticOptions &DiagOpts,
                                        const TargetInfo &Target) {
  Opts.getDiagnosticOpts() = DiagOpts;

  Opts.getOutputFile() = OutputFile;
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::LLVMContext &Context = llvm::getGlobalContext();

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                              "LLVM 'Clang' Compiler: http://clang.llvm.org\n");

  if (TimeReport)
    ClangFrontendTimer = new llvm::Timer("Clang front-end time");

  if (Verbose)
    llvm::errs() << "clang-cc version " CLANG_VERSION_STRING
                 << " based upon " << PACKAGE_STRING
                 << " hosted on " << llvm::sys::getHostTriple() << "\n";

  // If no input was specified, read from stdin.
  if (InputFilenames.empty())
    InputFilenames.push_back("-");

  // Construct the diagnostic options first, which cannot fail, so that we can
  // build a diagnostic client to use for any errors during option handling.
  DiagnosticOptions DiagOpts;
  ConstructDiagnosticOptions(DiagOpts);

  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  llvm::OwningPtr<DiagnosticClient> DiagClient;
  if (VerifyDiagnostics) {
    // When checking diagnostics, just buffer them up.
    DiagClient.reset(new TextDiagnosticBuffer());
    if (InputFilenames.size() != 1) {
      fprintf(stderr, "-verify only works on single input files.\n");
      return 1;
    }
  } else {
    DiagClient.reset(new TextDiagnosticPrinter(llvm::errs(), DiagOpts));
  }

  if (!DumpBuildInformation.empty())
    SetUpBuildDumpLog(DiagOpts, argc, argv, DiagClient);

  // Configure our handling of diagnostics.
  Diagnostic Diags(DiagClient.get());
  if (ProcessWarningOptions(Diags, OptWarnings, OptPedantic, OptPedanticErrors,
                            OptNoWarnings))
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::llvm_install_error_handler(LLVMErrorHandler,
                                   static_cast<void*>(&Diags));

  // Initialize base triple.  If a -triple option has been specified, use
  // that triple.  Otherwise, default to the host triple.
  llvm::Triple Triple(TargetTriple);
  if (Triple.getTriple().empty())
    Triple = llvm::Triple(llvm::sys::getHostTriple());

  // Get information about the target being compiled for.
  llvm::OwningPtr<TargetInfo>
  Target(TargetInfo::CreateTargetInfo(Triple.getTriple()));

  if (Target == 0) {
    Diags.Report(FullSourceLoc(), diag::err_fe_unknown_triple)
      << Triple.getTriple().c_str();
    return 1;
  }

  // Set the target ABI if specified.
  if (!TargetABI.empty()) {
    if (!Target->setABI(TargetABI)) {
      Diags.Report(FullSourceLoc(), diag::err_fe_unknown_target_abi)
        << TargetABI;
      return 1;
    }
  }

  if (!InheritanceViewCls.empty())  // C++ visualization?
    ProgAction = InheritanceView;

  // Infer the input language.
  //
  // FIXME: We should move .ast inputs to taking a separate path, they are
  // really quite different.
  LangKind LK = GetLanguage();

  // Now that we have initialized the diagnostics engine and the target, finish
  // setting up the compiler invocation.
  CompilerInvocation CompOpts;
  ConstructCompilerInvocation(CompOpts, DiagOpts, *Target);

  // Create the source manager.
  SourceManager SourceMgr;

  // Create a file manager object to provide access to and cache the filesystem.
  FileManager FileMgr;

  // Compute the feature set, unfortunately this effects the language!
  llvm::StringMap<bool> Features;
  ComputeFeatureMap(*Target, Features);

  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    const std::string &InFile = InputFilenames[i];

    // AST inputs are handled specially.
    if (LK == langkind_ast) {
      ProcessASTInputFile(CompOpts, InFile, ProgAction, Features,
                          Diags, FileMgr, Context);
      continue;
    }

    // Reset the ID tables if we are reusing the SourceManager.
    if (i)
      SourceMgr.clearIDTables();

    // Initialize language options, inferring file types from input filenames.
    LangOptions LangInfo;
    InitializeLangOptions(LangInfo, LK);
    InitializeLanguageStandard(LangInfo, LK, *Target, Features);

    // Process the -I options and set them in the HeaderInfo.
    HeaderSearch HeaderInfo(FileMgr);


    InitializeIncludePaths(argv[0], HeaderInfo, FileMgr, LangInfo, Triple);

    // Set up the preprocessor with these options.
    llvm::OwningPtr<Preprocessor> PP(CreatePreprocessor(Diags, LangInfo,
                                                        *Target, SourceMgr,
                                                        HeaderInfo));

    // Handle generating dependencies, if requested.
    if (!DependencyFile.empty()) {
      if (DependencyTargets.empty()) {
        Diags.Report(FullSourceLoc(), diag::err_fe_dependency_file_requires_MT);
        continue;
      }
      std::string ErrStr;
      llvm::raw_ostream *DependencyOS =
          new llvm::raw_fd_ostream(DependencyFile.c_str(), ErrStr);
      if (!ErrStr.empty()) {
        Diags.Report(FullSourceLoc(), diag::err_fe_error_opening)
          << DependencyFile << ErrStr;
        continue;
      }

      AttachDependencyFileGen(PP.get(), DependencyOS, DependencyTargets,
                              DependenciesIncludeSystemHeaders,
                              PhonyDependencyTarget);
    }

    if (ImplicitIncludePCH.empty()) {
      if (InitializeSourceManager(*PP.get(), InFile))
        continue;

      // Initialize builtin info.
      PP->getBuiltinInfo().InitializeBuiltins(PP->getIdentifierTable(),
                                              PP->getLangOptions().NoBuiltin);
    }

    // Process the source file.
    Diags.getClient()->BeginSourceFile(LangInfo);
    ProcessInputFile(CompOpts, *PP, InFile, ProgAction, Features, Context);
    Diags.getClient()->EndSourceFile();

    HeaderInfo.ClearFileInfo();
  }

  if (!NoCaretDiagnostics)
    if (unsigned NumDiagnostics = Diags.getNumDiagnostics())
      fprintf(stderr, "%d diagnostic%s generated.\n", NumDiagnostics,
              (NumDiagnostics == 1 ? "" : "s"));

  if (Stats) {
    FileMgr.PrintStats();
    fprintf(stderr, "\n");
  }

  delete ClangFrontendTimer;
  delete BuildLogFile;

  // If verifying diagnostics and we reached here, all is well.
  if (VerifyDiagnostics)
    return 0;

  // Managed static deconstruction. Useful for making things like
  // -time-passes usable.
  llvm::llvm_shutdown();

  return (Diags.getNumErrors() != 0);
}
