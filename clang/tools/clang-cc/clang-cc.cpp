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

#include "clang-cc.h"
#include "ASTConsumers.h"
#include "clang/Frontend/CompileOptions.h"
#include "clang/Frontend/FixItRewriter.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/InitHeaderSearch.h"
#include "clang/Frontend/PathDiagnosticClients.h"
#include "clang/Frontend/PCHReader.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/CodeGen/ModuleBuilder.h"
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
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Timer.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
#include "llvm/System/Signals.h"
#include <cstdlib>

using namespace clang;

//===----------------------------------------------------------------------===//
// Source Location Parser
//===----------------------------------------------------------------------===//

/// \brief A source location that has been parsed on the command line.
struct ParsedSourceLocation {
  std::string FileName;
  unsigned Line;
  unsigned Column;

  /// \brief Try to resolve the file name of a parsed source location.
  ///
  /// \returns true if there was an error, false otherwise.
  bool ResolveLocation(FileManager &FileMgr, RequestedSourceLocation &Result);
};

bool
ParsedSourceLocation::ResolveLocation(FileManager &FileMgr, 
                                      RequestedSourceLocation &Result) {
  const FileEntry *File = FileMgr.getFile(FileName);
  if (!File)
    return true;

  Result.File = File;
  Result.Line = Line;
  Result.Column = Column;
  return false;
}

namespace llvm {
  namespace cl {
    /// \brief Command-line option parser that parses source locations.
    ///
    /// Source locations are of the form filename:line:column.
    template<>
    class parser<ParsedSourceLocation> 
      : public basic_parser<ParsedSourceLocation> {
    public:
      bool parse(Option &O, const char *ArgName, 
                 const std::string &ArgValue,
                 ParsedSourceLocation &Val);
    };

    bool 
    parser<ParsedSourceLocation>::
    parse(Option &O, const char *ArgName, const std::string &ArgValue, 
          ParsedSourceLocation &Val) {
      using namespace clang;

      const char *ExpectedFormat 
        = "source location must be of the form filename:line:column";
      std::string::size_type SecondColon = ArgValue.rfind(':');
      if (SecondColon == std::string::npos) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }
      char *EndPtr;
      long Column 
        = std::strtol(ArgValue.c_str() + SecondColon + 1, &EndPtr, 10);
      if (EndPtr != ArgValue.c_str() + ArgValue.size()) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }

      std::string::size_type FirstColon = ArgValue.rfind(':', SecondColon-1);
      if (SecondColon == std::string::npos) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }
      long Line = std::strtol(ArgValue.c_str() + FirstColon + 1, &EndPtr, 10);
      if (EndPtr != ArgValue.c_str() + SecondColon) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }
      
      Val.FileName = ArgValue.substr(0, FirstColon);
      Val.Line = Line;
      Val.Column = Column;
      return false;
    }
  }
}

//===----------------------------------------------------------------------===//
// Global options.
//===----------------------------------------------------------------------===//

/// ClangFrontendTimer - The front-end activities should charge time to it with
/// TimeRegion.  The -ftime-report option controls whether this will do
/// anything.
llvm::Timer *ClangFrontendTimer = 0;

static bool HadErrors = false;

static llvm::cl::opt<bool>
Verbose("v", llvm::cl::desc("Enable verbose output"));
static llvm::cl::opt<bool>
Stats("print-stats", 
      llvm::cl::desc("Print performance metrics and statistics"));
static llvm::cl::opt<bool>
DisableFree("disable-free",
           llvm::cl::desc("Disable freeing of memory on exit"),
           llvm::cl::init(false));

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
  SerializeAST,                 // Emit a .ast file.
  EmitHTML,                     // Translate input source into HTML.
  ASTPrint,                     // Parse ASTs and print them.
  ASTDump,                      // Parse ASTs and dump them.
  ASTView,                      // Parse ASTs and view them in Graphviz.
  PrintDeclContext,             // Print DeclContext and their Decls.
  TestSerialization,            // Run experimental serialization code.
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
             clEnumValN(ASTDump, "ast-dump",
                        "Build ASTs and then debug dump them"),
             clEnumValN(ASTView, "ast-view",
                        "Build ASTs and view them with GraphViz"),
             clEnumValN(PrintDeclContext, "print-decl-contexts",
                        "Print DeclContexts and their Decls"),
             clEnumValN(GeneratePTH, "emit-pth",
                        "Generate pre-tokenized header file"),
             clEnumValN(GeneratePCH, "emit-pch",
                        "Generate pre-compiled header file"),
             clEnumValN(TestSerialization, "test-pickling",
                        "Run prototype serialization code"),
             clEnumValN(EmitAssembly, "S",
                        "Emit native assembly code"),
             clEnumValN(EmitLLVM, "emit-llvm",
                        "Build ASTs then convert to LLVM, emit .ll file"),
             clEnumValN(EmitBC, "emit-llvm-bc",
                        "Build ASTs then convert to LLVM, emit .bc file"),
             clEnumValN(EmitLLVMOnly, "emit-llvm-only",
                        "Build ASTs and convert to LLVM, discarding output"),
             clEnumValN(SerializeAST, "serialize",
                        "Build ASTs and emit .ast file"),
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
 llvm::cl::desc("Specify output file (for --serialize, this is a directory)"));


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

static llvm::cl::opt<std::string>
HTMLDiag("html-diags",
         llvm::cl::desc("Generate HTML to report diagnostics"),
         llvm::cl::value_desc("HTML directory"));

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
PrintSourceRangeInfo("fprint-source-range-info",
                    llvm::cl::desc("Print source range spans in numeric form"));

static llvm::cl::opt<bool>
PrintDiagnosticOption("fdiagnostics-show-option",
             llvm::cl::desc("Print diagnostic name with mappable diagnostics"));

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
  langkind_objcxx_cpp
};

static llvm::cl::opt<LangKind>
BaseLang("x", llvm::cl::desc("Base language to compile"),
         llvm::cl::init(langkind_unspecified),
   llvm::cl::values(clEnumValN(langkind_c,     "c",            "C"),
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
                    clEnumValEnd));

static llvm::cl::opt<bool>
LangObjC("ObjC", llvm::cl::desc("Set base language to Objective-C"),
         llvm::cl::Hidden);
static llvm::cl::opt<bool>
LangObjCXX("ObjC++", llvm::cl::desc("Set base language to Objective-C++"),
           llvm::cl::Hidden);

static llvm::cl::opt<bool>
ObjCExclusiveGC("fobjc-gc-only",
                llvm::cl::desc("Use GC exclusively for Objective-C related "
                               "memory management"));

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


/// InitializeBaseLanguage - Handle the -x foo options.
static void InitializeBaseLanguage() {
  if (LangObjC)
    BaseLang = langkind_objc;
  else if (LangObjCXX)
    BaseLang = langkind_objcxx;
}

static LangKind GetLanguage(const std::string &Filename) {
  if (BaseLang != langkind_unspecified)
    return BaseLang;
  
  std::string::size_type DotPos = Filename.rfind('.');

  if (DotPos == std::string::npos) {
    BaseLang = langkind_c;  // Default to C if no extension.
    return langkind_c;
  }
  
  std::string Ext = std::string(Filename.begin()+DotPos+1, Filename.end());
  // C header: .h
  // C++ header: .hh or .H;
  // assembler no preprocessing: .s
  // assembler: .S
  if (Ext == "c")
    return langkind_c;
  else if (Ext == "S" ||
           // If the compiler is run on a .s file, preprocess it as .S
           Ext == "s")
    return langkind_asm_cpp;
  else if (Ext == "i")
    return langkind_c_cpp;
  else if (Ext == "ii")
    return langkind_cxx_cpp;
  else if (Ext == "m")
    return langkind_objc;
  else if (Ext == "mi")
    return langkind_objc_cpp;
  else if (Ext == "mm" || Ext == "M")
    return langkind_objcxx;
  else if (Ext == "mii")
    return langkind_objcxx_cpp;
  else if (Ext == "C" || Ext == "cc" || Ext == "cpp" || Ext == "CPP" ||
           Ext == "c++" || Ext == "cp" || Ext == "cxx")
    return langkind_cxx;
  else
    return langkind_c;
}


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
  }
  
  if (ObjCExclusiveGC)
    Options.setGCMode(LangOptions::GCOnly);
  else if (ObjCEnableGC)
    Options.setGCMode(LangOptions::HybridGC);
  
  if (ObjCEnableGCBitmapPrint)
    Options.ObjCGCBitmapPrint = 1;
  
  Options.setVisibilityMode(SymbolVisibility);
  Options.OverflowChecking = OverflowChecking;
}

/// LangStds - Language standards we support.
enum LangStds {
  lang_unspecified,  
  lang_c89, lang_c94, lang_c99,
  lang_gnu_START,
  lang_gnu89 = lang_gnu_START, lang_gnu99,
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

// FIXME: This (and all GCC -f options) really come in -f... and
// -fno-... forms, and additionally support automagic behavior when
// they are not defined. For example, -fexceptions defaults to on or
// off depending on the language. We should support this behavior in
// some form (perhaps just add a facility for distinguishing when an
// has its default value from when it has been set to its default
// value).
static llvm::cl::opt<bool>
Exceptions("fexceptions",
           llvm::cl::desc("Enable support for exception handling"));

static llvm::cl::opt<bool>
GNURuntime("fgnu-runtime",
            llvm::cl::desc("Generate output compatible with the standard GNU "
                           "Objective-C runtime"));

static llvm::cl::opt<bool>
NeXTRuntime("fnext-runtime",
            llvm::cl::desc("Generate output compatible with the NeXT "
                           "runtime"));



static llvm::cl::opt<bool>
Trigraphs("trigraphs", llvm::cl::desc("Process trigraph sequences"));

static llvm::cl::list<std::string>
TargetFeatures("mattr", llvm::cl::CommaSeparated,
        llvm::cl::desc("Target specific attributes (-mattr=help for details)"));

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
NoCommon("fno-common",
         llvm::cl::desc("Compile common globals like normal definitions"),
         llvm::cl::ValueDisallowed);

static llvm::cl::opt<std::string>
MainFileName("main-file-name",
             llvm::cl::desc("Main file name to use for debug info"));

// It might be nice to add bounds to the CommandLine library directly.
struct OptLevelParser : public llvm::cl::parser<unsigned> {
  bool parse(llvm::cl::Option &O, const char *ArgName,
             const std::string &Arg, unsigned &Val) {
    if (llvm::cl::parser<unsigned>::parse(O, ArgName, Arg, Val))
      return true;
    if (Val > 3)
      return O.error(": '" + Arg + "' invalid optimization level!");
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

static void InitializeLanguageStandard(LangOptions &Options, LangKind LK,
                                       TargetInfo *Target) {
  // Allow the target to set the default the langauge options as it sees fit.
  Target->getDefaultLangOptions(Options);
  
  // If there are any -mattr options, pass them to the target for validation and
  // processing.  The driver should have already consolidated all the
  // target-feature settings and passed them to us in the -mattr list.  The
  // -mattr list is treated by the code generator as a diff against the -mcpu
  // setting, but the driver should pass all enabled options as "+" settings.
  // This means that the target should only look at + settings.
  if (!TargetFeatures.empty()) {
    std::string ErrorStr;
    int Opt = Target->HandleTargetFeatures(&TargetFeatures[0],
                                           TargetFeatures.size(), ErrorStr);
    if (Opt != -1) {
      if (ErrorStr.empty())
        fprintf(stderr, "invalid feature '%s'\n",
                TargetFeatures[Opt].c_str());
      else
        fprintf(stderr, "feature '%s': %s\n",
                TargetFeatures[Opt].c_str(), ErrorStr.c_str());
      exit(1);
    }
  }
  
  if (LangStd == lang_unspecified) {
    // Based on the base language, pick one.
    switch (LK) {
    case lang_unspecified: assert(0 && "Unknown base language");
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
    Options.Boolean = 1;
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
  Options.GNUMode = LangStd >= lang_gnu_START;
  
  if (Options.CPlusPlus) {
    Options.C99 = 0;
    Options.HexFloats = Options.GNUMode;
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
  
  // Never accept '$' in identifiers when preprocessing assembler.
  if (LK != langkind_asm_cpp)
    Options.DollarIdents = true;  // FIXME: target property?
  else
    Options.DollarIdents = DollarsInIdents;
  
  if (PascalStrings.getPosition())
    Options.PascalStrings = PascalStrings;
  Options.Microsoft = MSExtensions;
  Options.WritableStrings = WritableStrings;
  if (NoLaxVectorConversions.getPosition())
      Options.LaxVectorConversions = 0;
  Options.Exceptions = Exceptions;
  if (EnableBlocks.getPosition())
    Options.Blocks = EnableBlocks;

  if (!AllowBuiltins)
    Options.NoBuiltin = 1;
  if (Freestanding)
    Options.Freestanding = Options.NoBuiltin = 1;
  
  if (EnableHeinousExtensions)
    Options.HeinousExtensions = 1;

  Options.MathErrno = MathErrno;

  Options.InstantiationDepth = TemplateDepth;

  // Override the default runtime if the user requested it.
  if (NeXTRuntime)
    Options.NeXTRuntime = 1;
  else if (GNURuntime)
    Options.NeXTRuntime = 0;

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

  if (MainFileName.getPosition())
    Options.setMainFileName(MainFileName.c_str());
}

//===----------------------------------------------------------------------===//
// Target Triple Processing.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
TargetTriple("triple",
  llvm::cl::desc("Specify target triple (e.g. i686-apple-darwin9)"));

static llvm::cl::opt<std::string>
Arch("arch", llvm::cl::desc("Specify target architecture (e.g. i686)"));

static llvm::cl::opt<std::string>
MacOSVersionMin("mmacosx-version-min", 
                llvm::cl::desc("Specify target Mac OS X version (e.g. 10.5)"));

// If -mmacosx-version-min=10.3.9 is specified, change the triple from being
// something like powerpc-apple-darwin9 to powerpc-apple-darwin7

// FIXME: We should have the driver do this instead.
static void HandleMacOSVersionMin(std::string &Triple) {
  std::string::size_type DarwinDashIdx = Triple.find("-darwin");
  if (DarwinDashIdx == std::string::npos) {
    fprintf(stderr, 
            "-mmacosx-version-min only valid for darwin (Mac OS X) targets\n");
    exit(1);
  }
  unsigned DarwinNumIdx = DarwinDashIdx + strlen("-darwin");
  
  // Remove the number.
  Triple.resize(DarwinNumIdx);

  // Validate that MacOSVersionMin is a 'version number', starting with 10.[3-9]
  bool MacOSVersionMinIsInvalid = false;
  int VersionNum = 0;
  if (MacOSVersionMin.size() < 4 ||
      MacOSVersionMin.substr(0, 3) != "10." ||
      !isdigit(MacOSVersionMin[3])) {
    MacOSVersionMinIsInvalid = true;
  } else {
    const char *Start = MacOSVersionMin.c_str()+3;
    char *End = 0;
    VersionNum = (int)strtol(Start, &End, 10);

    // The version number must be in the range 0-9.
    MacOSVersionMinIsInvalid = (unsigned)VersionNum > 9;
    
    // Turn MacOSVersionMin into a darwin number: e.g. 10.3.9 is 3 -> 7.
    Triple += llvm::itostr(VersionNum+4);
    
    if (End[0] == '.' && isdigit(End[1]) && End[2] == '\0') {   // 10.4.7 is ok.
      // Add the period piece (.7) to the end of the triple.  This gives us
      // something like ...-darwin8.7
      Triple += End;
    } else if (End[0] != '\0') { // "10.4" is ok.  10.4x is not.
      MacOSVersionMinIsInvalid = true;
    }
  }
  
  if (MacOSVersionMinIsInvalid) {
    fprintf(stderr, 
        "-mmacosx-version-min=%s is invalid, expected something like '10.4'.\n",
            MacOSVersionMin.c_str());
    exit(1);
  }
  else if (VersionNum <= 4 && 
           !strncmp(Triple.c_str(), "x86_64", strlen("x86_64"))) {
    fprintf(stderr, 
        "-mmacosx-version-min=%s is invalid with -arch x86_64.\n",
            MacOSVersionMin.c_str());
    exit(1);
  }

}

static llvm::cl::opt<std::string>
IPhoneOSVersionMin("miphoneos-version-min", 
                llvm::cl::desc("Specify target iPhone OS version (e.g. 2.0)"));

// If -miphoneos-version-min=2.2 is specified, change the triple from being
// something like armv6-apple-darwin10 to armv6-apple-darwin9.2.2. We use
// 9 as the default major Darwin number, and encode the iPhone OS version
// number in the minor version and revision.

// FIXME: We should have the driver do this instead.
static void HandleIPhoneOSVersionMin(std::string &Triple) {
  std::string::size_type DarwinDashIdx = Triple.find("-darwin");
  if (DarwinDashIdx == std::string::npos) {
    fprintf(stderr, 
            "-miphoneos-version-min only valid for darwin (Mac OS X) targets\n");
    exit(1);
  }
  unsigned DarwinNumIdx = DarwinDashIdx + strlen("-darwin");
  
  // Remove the number.
  Triple.resize(DarwinNumIdx);
  
  // Validate that IPhoneOSVersionMin is a 'version number', starting with [2-9].[0-9]
  bool IPhoneOSVersionMinIsInvalid = false;
  int VersionNum = 0;
  if (IPhoneOSVersionMin.size() < 3 ||
      !isdigit(IPhoneOSVersionMin[0])) {
    IPhoneOSVersionMinIsInvalid = true;
  } else {
    const char *Start = IPhoneOSVersionMin.c_str();
    char *End = 0;
    VersionNum = (int)strtol(Start, &End, 10);
    
    // The version number must be in the range 0-9.
    IPhoneOSVersionMinIsInvalid = (unsigned)VersionNum > 9;
    
    // Turn IPhoneOSVersionMin into a darwin number: e.g. 2.0 is 2 -> 9.2.
    Triple += "9." + llvm::itostr(VersionNum);
    
    if (End[0] == '.' && isdigit(End[1]) && End[2] == '\0') {   // 2.2 is ok.
      // Add the period piece (.2) to the end of the triple.  This gives us
      // something like ...-darwin9.2.2
      Triple += End;
    } else if (End[0] != '\0') { // "2.2" is ok.  2x is not.
      IPhoneOSVersionMinIsInvalid = true;
    }
  }
  
  if (IPhoneOSVersionMinIsInvalid) {
    fprintf(stderr, 
            "-miphoneos-version-min=%s is invalid, expected something like '2.0'.\n",
            IPhoneOSVersionMin.c_str());
    exit(1);
  }
}

/// CreateTargetTriple - Process the various options that affect the target
/// triple and build a final aggregate triple that we are compiling for.
static std::string CreateTargetTriple() {
  // Initialize base triple.  If a -triple option has been specified, use
  // that triple.  Otherwise, default to the host triple.
  std::string Triple = TargetTriple;
  if (Triple.empty())
    Triple = llvm::sys::getHostTriple();
  
  // If -arch foo was specified, remove the architecture from the triple we have
  // so far and replace it with the specified one.

  // FIXME: -arch should be removed, the driver should handle this.
  if (!Arch.empty()) {
    // Decompose the base triple into "arch" and suffix.
    std::string::size_type FirstDashIdx = Triple.find('-');
    
    if (FirstDashIdx == std::string::npos) {
      fprintf(stderr, 
              "Malformed target triple: \"%s\" ('-' could not be found).\n",
              Triple.c_str());
      exit(1);
    }
    
    // Canonicalize -arch ppc to add "powerpc" to the triple, not ppc.
    if (Arch == "ppc")
      Arch = "powerpc";
    else if (Arch == "ppc64")
      Arch = "powerpc64";
  
    Triple = Arch + std::string(Triple.begin()+FirstDashIdx, Triple.end());
  }

  // If -mmacosx-version-min=10.3.9 is specified, change the triple from being
  // something like powerpc-apple-darwin9 to powerpc-apple-darwin7
  if (!MacOSVersionMin.empty())
    HandleMacOSVersionMin(Triple);
  else if (!IPhoneOSVersionMin.empty())
    HandleIPhoneOSVersionMin(Triple);;
  
  return Triple;
}

//===----------------------------------------------------------------------===//
// Preprocessor Initialization
//===----------------------------------------------------------------------===//

// FIXME: Preprocessor builtins to support.
//   -A...    - Play with #assertions
//   -undef   - Undefine all predefined macros

// FIXME: -imacros

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
ImplicitIncludePTH("include-pth", llvm::cl::value_desc("file"),
                   llvm::cl::desc("Include file before parsing"));

static llvm::cl::opt<std::string>
ImplicitIncludePCH("include-pch", llvm::cl::value_desc("file"),
                   llvm::cl::desc("Include precompiled header file"));

// Append a #define line to Buf for Macro.  Macro should be of the form XXX,
// in which case we emit "#define XXX 1" or "XXX=Y z W" in which case we emit
// "#define XXX Y z W".  To get a #define with no value, use "XXX=".
static void DefineBuiltinMacro(std::vector<char> &Buf, const char *Macro,
                               const char *Command = "#define ") {
  Buf.insert(Buf.end(), Command, Command+strlen(Command));
  if (const char *Equal = strchr(Macro, '=')) {
    // Turn the = into ' '.
    Buf.insert(Buf.end(), Macro, Equal);
    Buf.push_back(' ');
    
    // Per GCC -D semantics, the macro ends at \n if it exists.
    const char *End = strpbrk(Equal, "\n\r");
    if (End) {
      fprintf(stderr, "warning: macro '%s' contains embedded newline, text "
              "after the newline is ignored.\n",
              std::string(Macro, Equal).c_str());
    } else {
      End = Equal+strlen(Equal);
    }
    
    Buf.insert(Buf.end(), Equal+1, End);
  } else {
    // Push "macroname 1".
    Buf.insert(Buf.end(), Macro, Macro+strlen(Macro));
    Buf.push_back(' ');
    Buf.push_back('1');
  }
  Buf.push_back('\n');
}

/// Add the quoted name of an implicit include file.
static void AddQuotedIncludePath(std::vector<char> &Buf, 
                                 const std::string &File) {
  // Implicit include paths are relative to the current working
  // directory; resolve them now instead of using the normal machinery
  // (which would look relative to the input file).
  llvm::sys::Path Path(File);
  Path.makeAbsolute();
    
  // Escape double quotes etc.
  Buf.push_back('"');
  std::string EscapedFile = Lexer::Stringify(Path.toString());
  Buf.insert(Buf.end(), EscapedFile.begin(), EscapedFile.end());
  Buf.push_back('"');
}

/// AddImplicitInclude - Add an implicit #include of the specified file to the
/// predefines buffer.
static void AddImplicitInclude(std::vector<char> &Buf, 
                               const std::string &File) {
  const char *Inc = "#include ";
  Buf.insert(Buf.end(), Inc, Inc+strlen(Inc));
  AddQuotedIncludePath(Buf, File);
  Buf.push_back('\n');
}

static void AddImplicitIncludeMacros(std::vector<char> &Buf,
                                     const std::string &File) {
  const char *Inc = "#__include_macros ";
  Buf.insert(Buf.end(), Inc, Inc+strlen(Inc));
  AddQuotedIncludePath(Buf, File);
  Buf.push_back('\n');
  // Marker token to stop the __include_macros fetch loop.
  const char *Marker = "##\n"; // ##?
  Buf.insert(Buf.end(), Marker, Marker+strlen(Marker));
}

/// AddImplicitIncludePTH - Add an implicit #include using the original file
///  used to generate a PTH cache.
static void AddImplicitIncludePTH(std::vector<char> &Buf, Preprocessor &PP) {
  PTHManager *P = PP.getPTHManager();
  assert(P && "No PTHManager.");
  const char *OriginalFile = P->getOriginalSourceFile();
  
  if (!OriginalFile) {
    assert(!ImplicitIncludePTH.empty());
    fprintf(stderr, "error: PTH file '%s' does not designate an original "
            "source header file for -include-pth\n",
            ImplicitIncludePTH.c_str());
    exit (1);
  }
  
  AddImplicitInclude(Buf, OriginalFile);
}

/// PickFP - This is used to pick a value based on the FP semantics of the
/// specified FP model.
template <typename T>
static T PickFP(const llvm::fltSemantics *Sem, T IEEESingleVal,
                T IEEEDoubleVal, T X87DoubleExtendedVal, T PPCDoubleDoubleVal) {
  if (Sem == &llvm::APFloat::IEEEsingle)
    return IEEESingleVal;
  if (Sem == &llvm::APFloat::IEEEdouble)
    return IEEEDoubleVal;
  if (Sem == &llvm::APFloat::x87DoubleExtended)
    return X87DoubleExtendedVal;
  assert(Sem == &llvm::APFloat::PPCDoubleDouble);
  return PPCDoubleDoubleVal;
}

static void DefineFloatMacros(std::vector<char> &Buf, const char *Prefix,
                              const llvm::fltSemantics *Sem) {
  const char *DenormMin, *Epsilon, *Max, *Min;
  DenormMin = PickFP(Sem, "1.40129846e-45F", "4.9406564584124654e-324", 
                     "3.64519953188247460253e-4951L",
                     "4.94065645841246544176568792868221e-324L");
  int Digits = PickFP(Sem, 6, 15, 18, 31);
  Epsilon = PickFP(Sem, "1.19209290e-7F", "2.2204460492503131e-16",
                   "1.08420217248550443401e-19L",
                   "4.94065645841246544176568792868221e-324L");
  int HasInifinity = 1, HasQuietNaN = 1;
  int MantissaDigits = PickFP(Sem, 24, 53, 64, 106);
  int Min10Exp = PickFP(Sem, -37, -307, -4931, -291);
  int Max10Exp = PickFP(Sem, 38, 308, 4932, 308);
  int MinExp = PickFP(Sem, -125, -1021, -16381, -968);
  int MaxExp = PickFP(Sem, 128, 1024, 16384, 1024);
  Min = PickFP(Sem, "1.17549435e-38F", "2.2250738585072014e-308",
               "3.36210314311209350626e-4932L",
               "2.00416836000897277799610805135016e-292L");
  Max = PickFP(Sem, "3.40282347e+38F", "1.7976931348623157e+308",
               "1.18973149535723176502e+4932L",
               "1.79769313486231580793728971405301e+308L");
  
  char MacroBuf[60];
  sprintf(MacroBuf, "__%s_DENORM_MIN__=%s", Prefix, DenormMin);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_DIG__=%d", Prefix, Digits);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_EPSILON__=%s", Prefix, Epsilon);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_HAS_INFINITY__=%d", Prefix, HasInifinity);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_HAS_QUIET_NAN__=%d", Prefix, HasQuietNaN);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MANT_DIG__=%d", Prefix, MantissaDigits);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MAX_10_EXP__=%d", Prefix, Max10Exp);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MAX_EXP__=%d", Prefix, MaxExp);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MAX__=%s", Prefix, Max);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MIN_10_EXP__=(%d)", Prefix, Min10Exp);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MIN_EXP__=(%d)", Prefix, MinExp);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MIN__=%s", Prefix, Min);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_HAS_DENORM__=1", Prefix);
  DefineBuiltinMacro(Buf, MacroBuf);
}


/// DefineTypeSize - Emit a macro to the predefines buffer that declares a macro
/// named MacroName with the max value for a type with width 'TypeWidth' a
/// signedness of 'isSigned' and with a value suffix of 'ValSuffix' (e.g. LL).
static void DefineTypeSize(const char *MacroName, unsigned TypeWidth,
                           const char *ValSuffix, bool isSigned,
                           std::vector<char> &Buf) {
  char MacroBuf[60];
  long long MaxVal;
  if (isSigned)
    MaxVal = (1LL << (TypeWidth - 1)) - 1;
  else
    MaxVal = ~0LL >> (64-TypeWidth);
  
  sprintf(MacroBuf, "%s=%llu%s", MacroName, MaxVal, ValSuffix);
  DefineBuiltinMacro(Buf, MacroBuf);
}

static void DefineType(const char *MacroName, TargetInfo::IntType Ty,
                       std::vector<char> &Buf) {
  char MacroBuf[60];
  sprintf(MacroBuf, "%s=%s", MacroName, TargetInfo::getTypeName(Ty));
  DefineBuiltinMacro(Buf, MacroBuf);
}


static void InitializePredefinedMacros(const TargetInfo &TI,
                                       const LangOptions &LangOpts,
                                       std::vector<char> &Buf) {
  char MacroBuf[60];
  // Compiler version introspection macros.
  DefineBuiltinMacro(Buf, "__llvm__=1");   // LLVM Backend
  DefineBuiltinMacro(Buf, "__clang__=1");  // Clang Frontend
  
  // Currently claim to be compatible with GCC 4.2.1-5621.
  DefineBuiltinMacro(Buf, "__APPLE_CC__=5621");
  DefineBuiltinMacro(Buf, "__GNUC_MINOR__=2");
  DefineBuiltinMacro(Buf, "__GNUC_PATCHLEVEL__=1");
  DefineBuiltinMacro(Buf, "__GNUC__=4");
  DefineBuiltinMacro(Buf, "__GXX_ABI_VERSION=1002");
  DefineBuiltinMacro(Buf, "__VERSION__=\"4.2.1 Compatible Clang Compiler\"");
  
  
  // Initialize language-specific preprocessor defines.
  
  // These should all be defined in the preprocessor according to the
  // current language configuration.
  if (!LangOpts.Microsoft)
    DefineBuiltinMacro(Buf, "__STDC__=1");
  if (LangOpts.AsmPreprocessor)
    DefineBuiltinMacro(Buf, "__ASSEMBLER__=1");
  if (LangOpts.C99 && !LangOpts.CPlusPlus)
    DefineBuiltinMacro(Buf, "__STDC_VERSION__=199901L");
  else if (0) // STDC94 ?
    DefineBuiltinMacro(Buf, "__STDC_VERSION__=199409L");

  // Standard conforming mode?
  if (!LangOpts.GNUMode)
    DefineBuiltinMacro(Buf, "__STRICT_ANSI__=1");
  
  if (LangOpts.CPlusPlus0x)
    DefineBuiltinMacro(Buf, "__GXX_EXPERIMENTAL_CXX0X__");

  if (LangOpts.Freestanding)
    DefineBuiltinMacro(Buf, "__STDC_HOSTED__=0");
  else
    DefineBuiltinMacro(Buf, "__STDC_HOSTED__=1");
  
  if (LangOpts.ObjC1) {
    DefineBuiltinMacro(Buf, "__OBJC__=1");
    if (LangOpts.ObjCNonFragileABI) {
      DefineBuiltinMacro(Buf, "__OBJC2__=1");
      DefineBuiltinMacro(Buf, "OBJC_ZEROCOST_EXCEPTIONS=1");
      DefineBuiltinMacro(Buf, "__EXCEPTIONS=1");
    }

    if (LangOpts.getGCMode() != LangOptions::NonGC)
      DefineBuiltinMacro(Buf, "__OBJC_GC__=1");
    
    if (LangOpts.NeXTRuntime)
      DefineBuiltinMacro(Buf, "__NEXT_RUNTIME__=1");
  }
  
  // darwin_constant_cfstrings controls this. This is also dependent
  // on other things like the runtime I believe.  This is set even for C code.
  DefineBuiltinMacro(Buf, "__CONSTANT_CFSTRINGS__=1");
  
  if (LangOpts.ObjC2)
    DefineBuiltinMacro(Buf, "OBJC_NEW_PROPERTIES");

  if (LangOpts.PascalStrings)
    DefineBuiltinMacro(Buf, "__PASCAL_STRINGS__");

  if (LangOpts.Blocks) {
    DefineBuiltinMacro(Buf, "__block=__attribute__((__blocks__(byref)))");
    DefineBuiltinMacro(Buf, "__BLOCKS__=1");
  }
  
  if (LangOpts.CPlusPlus) {
    DefineBuiltinMacro(Buf, "__DEPRECATED=1");
    DefineBuiltinMacro(Buf, "__EXCEPTIONS=1");
    DefineBuiltinMacro(Buf, "__GNUG__=4");
    DefineBuiltinMacro(Buf, "__GXX_WEAK__=1");
    DefineBuiltinMacro(Buf, "__cplusplus=1");
    DefineBuiltinMacro(Buf, "__private_extern__=extern");
  }
  
  // Filter out some microsoft extensions when trying to parse in ms-compat
  // mode. 
  if (LangOpts.Microsoft) {
    DefineBuiltinMacro(Buf, "_cdecl=__cdecl");
    DefineBuiltinMacro(Buf, "__int8=__INT8_TYPE__");
    DefineBuiltinMacro(Buf, "__int16=__INT16_TYPE__");
    DefineBuiltinMacro(Buf, "__int32=__INT32_TYPE__");
    DefineBuiltinMacro(Buf, "__int64=__INT64_TYPE__");
  }
  
  if (LangOpts.Optimize)
    DefineBuiltinMacro(Buf, "__OPTIMIZE__=1");
  if (LangOpts.OptimizeSize)
    DefineBuiltinMacro(Buf, "__OPTIMIZE_SIZE__=1");
    
  // Initialize target-specific preprocessor defines.
  
  // Define type sizing macros based on the target properties.
  assert(TI.getCharWidth() == 8 && "Only support 8-bit char so far");
  DefineBuiltinMacro(Buf, "__CHAR_BIT__=8");

  unsigned IntMaxWidth;
  const char *IntMaxSuffix;
  if (TI.getIntMaxType() == TargetInfo::SignedLongLong) {
    IntMaxWidth = TI.getLongLongWidth();
    IntMaxSuffix = "LL";
  } else if (TI.getIntMaxType() == TargetInfo::SignedLong) {
    IntMaxWidth = TI.getLongWidth();
    IntMaxSuffix = "L";
  } else {
    assert(TI.getIntMaxType() == TargetInfo::SignedInt);
    IntMaxWidth = TI.getIntWidth();
    IntMaxSuffix = "";
  }
  
  DefineTypeSize("__SCHAR_MAX__", TI.getCharWidth(), "", true, Buf);
  DefineTypeSize("__SHRT_MAX__", TI.getShortWidth(), "", true, Buf);
  DefineTypeSize("__INT_MAX__", TI.getIntWidth(), "", true, Buf);
  DefineTypeSize("__LONG_MAX__", TI.getLongWidth(), "L", true, Buf);
  DefineTypeSize("__LONG_LONG_MAX__", TI.getLongLongWidth(), "LL", true, Buf);
  DefineTypeSize("__WCHAR_MAX__", TI.getWCharWidth(), "", true, Buf);
  DefineTypeSize("__INTMAX_MAX__", IntMaxWidth, IntMaxSuffix, true, Buf);

  DefineType("__INTMAX_TYPE__", TI.getIntMaxType(), Buf);
  DefineType("__UINTMAX_TYPE__", TI.getUIntMaxType(), Buf);
  DefineType("__PTRDIFF_TYPE__", TI.getPtrDiffType(0), Buf);
  DefineType("__INTPTR_TYPE__", TI.getIntPtrType(), Buf);
  DefineType("__SIZE_TYPE__", TI.getSizeType(), Buf);
  DefineType("__WCHAR_TYPE__", TI.getWCharType(), Buf);
  // FIXME: TargetInfo hookize __WINT_TYPE__.
  DefineBuiltinMacro(Buf, "__WINT_TYPE__=int");
  
  DefineFloatMacros(Buf, "FLT", &TI.getFloatFormat());
  DefineFloatMacros(Buf, "DBL", &TI.getDoubleFormat());
  DefineFloatMacros(Buf, "LDBL", &TI.getLongDoubleFormat());

  // Define a __POINTER_WIDTH__ macro for stdint.h.
  sprintf(MacroBuf, "__POINTER_WIDTH__=%d", (int)TI.getPointerWidth(0));
  DefineBuiltinMacro(Buf, MacroBuf);
  
  if (!TI.isCharSigned())
    DefineBuiltinMacro(Buf, "__CHAR_UNSIGNED__");  

  // Define fixed-sized integer types for stdint.h
  assert(TI.getCharWidth() == 8 && "unsupported target types");
  assert(TI.getShortWidth() == 16 && "unsupported target types");
  DefineBuiltinMacro(Buf, "__INT8_TYPE__=char");
  DefineBuiltinMacro(Buf, "__INT16_TYPE__=short");
  
  if (TI.getIntWidth() == 32)
    DefineBuiltinMacro(Buf, "__INT32_TYPE__=int");
  else {
    assert(TI.getLongLongWidth() == 32 && "unsupported target types");
    DefineBuiltinMacro(Buf, "__INT32_TYPE__=long long");
  }
  
  // 16-bit targets doesn't necessarily have a 64-bit type.
  if (TI.getLongLongWidth() == 64)
    DefineBuiltinMacro(Buf, "__INT64_TYPE__=long long");
  
  // Add __builtin_va_list typedef.
  {
    const char *VAList = TI.getVAListDeclaration();
    Buf.insert(Buf.end(), VAList, VAList+strlen(VAList));
    Buf.push_back('\n');
  }
  
  if (const char *Prefix = TI.getUserLabelPrefix()) {
    sprintf(MacroBuf, "__USER_LABEL_PREFIX__=%s", Prefix);
    DefineBuiltinMacro(Buf, MacroBuf);
  }
  
  // Build configuration options.  FIXME: these should be controlled by
  // command line options or something.
  DefineBuiltinMacro(Buf, "__FINITE_MATH_ONLY__=0");

  if (LangOpts.Static)
    DefineBuiltinMacro(Buf, "__STATIC__=1");
  else
    DefineBuiltinMacro(Buf, "__DYNAMIC__=1");

  if (LangOpts.GNUInline)
    DefineBuiltinMacro(Buf, "__GNUC_GNU_INLINE__=1");
  else
    DefineBuiltinMacro(Buf, "__GNUC_STDC_INLINE__=1");

  if (LangOpts.NoInline)
    DefineBuiltinMacro(Buf, "__NO_INLINE__=1");

  if (unsigned PICLevel = LangOpts.PICLevel) {
    sprintf(MacroBuf, "__PIC__=%d", PICLevel);
    DefineBuiltinMacro(Buf, MacroBuf);

    sprintf(MacroBuf, "__pic__=%d", PICLevel);
    DefineBuiltinMacro(Buf, MacroBuf);
  }

  // Macros to control C99 numerics and <float.h>
  DefineBuiltinMacro(Buf, "__FLT_EVAL_METHOD__=0");
  DefineBuiltinMacro(Buf, "__FLT_RADIX__=2");
  sprintf(MacroBuf, "__DECIMAL_DIG__=%d",
          PickFP(&TI.getLongDoubleFormat(), -1/*FIXME*/, 17, 21, 33));
  DefineBuiltinMacro(Buf, MacroBuf);
  
  // Get other target #defines.
  TI.getTargetDefines(LangOpts, Buf);
}

static bool InitializeSourceManager(Preprocessor &PP,
                                    const std::string &InFile) {
  // Figure out where to get and map in the main file.
  SourceManager &SourceMgr = PP.getSourceManager();
  FileManager &FileMgr = PP.getFileManager();
  
  if (InFile != "-") {
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

/// InitializePreprocessor - Initialize the preprocessor getting it and the
/// environment ready to process a single file. This returns true on error.
///
static bool InitializePreprocessor(Preprocessor &PP,
                                   const std::string &InFile) {
  std::vector<char> PredefineBuffer;
  
  // Install things like __POWERPC__, __GNUC__, etc into the macro table.
  InitializePredefinedMacros(PP.getTargetInfo(), PP.getLangOptions(),
                             PredefineBuffer);
  
  // Add on the predefines from the driver.  Wrap in a #line directive to report
  // that they come from the command line.
  const char *LineDirective = "# 1 \"<command line>\" 1\n";
  PredefineBuffer.insert(PredefineBuffer.end(),
                         LineDirective, LineDirective+strlen(LineDirective));
  

  // Add macros from the command line.
  unsigned d = 0, D = D_macros.size();
  unsigned u = 0, U = U_macros.size();
  while (d < D || u < U) {
    if (u == U || (d < D && D_macros.getPosition(d) < U_macros.getPosition(u)))
      DefineBuiltinMacro(PredefineBuffer, D_macros[d++].c_str());
    else
      DefineBuiltinMacro(PredefineBuffer, U_macros[u++].c_str(), "#undef ");
  }

  // If -imacros are specified, include them now.  These are processed before
  // any -include directives.

  for (unsigned i = 0, e = ImplicitMacroIncludes.size(); i != e; ++i)
    AddImplicitIncludeMacros(PredefineBuffer, ImplicitMacroIncludes[i]);
  
  if (!ImplicitIncludePTH.empty() || !ImplicitIncludes.empty()) {
    // We want to add these paths to the predefines buffer in order, make a
    // temporary vector to sort by their occurrence.
    llvm::SmallVector<std::pair<unsigned, std::string*>, 8> OrderedPaths;

    if (!ImplicitIncludePTH.empty())
      OrderedPaths.push_back(std::make_pair(ImplicitIncludePTH.getPosition(),
                                            &ImplicitIncludePTH));
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
        AddImplicitInclude(PredefineBuffer, *Ptr);
      } else {
        assert(Ptr == &ImplicitIncludePTH);
        AddImplicitIncludePTH(PredefineBuffer, PP);
      }
    }
  }
  
  LineDirective = "# 2 \"<built-in>\" 2\n";
  PredefineBuffer.insert(PredefineBuffer.end(),
                         LineDirective, LineDirective+strlen(LineDirective));
  
  // Null terminate PredefinedBuffer and add it.
  PredefineBuffer.push_back(0);
  PP.setPredefines(&PredefineBuffer[0]);
  
  // Once we've read this, we're done.
  return false;
}

//===----------------------------------------------------------------------===//
// Preprocessor include path information.
//===----------------------------------------------------------------------===//

// This tool exports a large number of command line options to control how the
// preprocessor searches for header files.  At root, however, the Preprocessor
// object takes a very simple interface: a list of directories to search for
// 
// FIXME: -nostdinc,-nostdinc++
// FIXME: -imultilib
//

static llvm::cl::opt<bool>
nostdinc("nostdinc", llvm::cl::desc("Disable standard #include directories"));

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

/// InitializeIncludePaths - Process the -I options and set them in the
/// HeaderSearch object.
void InitializeIncludePaths(const char *Argv0, HeaderSearch &Headers,
                            FileManager &FM, const LangOptions &Lang) {
  InitHeaderSearch Init(Headers, Verbose, isysroot);

  // Handle -I... and -F... options, walking the lists in parallel.
  unsigned Iidx = 0, Fidx = 0;
  while (Iidx < I_dirs.size() && Fidx < F_dirs.size()) {
    if (I_dirs.getPosition(Iidx) < F_dirs.getPosition(Fidx)) {
      Init.AddPath(I_dirs[Iidx], InitHeaderSearch::Angled, false, true, false);
      ++Iidx;
    } else {
      Init.AddPath(F_dirs[Fidx], InitHeaderSearch::Angled, false, true, true);
      ++Fidx;
    }
  }
  
  // Consume what's left from whatever list was longer.
  for (; Iidx != I_dirs.size(); ++Iidx)
    Init.AddPath(I_dirs[Iidx], InitHeaderSearch::Angled, false, true, false);
  for (; Fidx != F_dirs.size(); ++Fidx)
    Init.AddPath(F_dirs[Fidx], InitHeaderSearch::Angled, false, true, true);
  
  // Handle -idirafter... options.
  for (unsigned i = 0, e = idirafter_dirs.size(); i != e; ++i)
    Init.AddPath(idirafter_dirs[i], InitHeaderSearch::After,
        false, true, false);
  
  // Handle -iquote... options.
  for (unsigned i = 0, e = iquote_dirs.size(); i != e; ++i)
    Init.AddPath(iquote_dirs[i], InitHeaderSearch::Quoted, false, true, false);
  
  // Handle -isystem... options.
  for (unsigned i = 0, e = isystem_dirs.size(); i != e; ++i)
    Init.AddPath(isystem_dirs[i], InitHeaderSearch::System, false, true, false);

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
        Init.AddPath(Prefix+iwithprefix_vals[iwithprefix_idx], 
                InitHeaderSearch::System, false, false, false);
        ++iwithprefix_idx;
        iwithprefix_done = iwithprefix_idx == iwithprefix_vals.size();
      } else {
        Init.AddPath(Prefix+iwithprefixbefore_vals[iwithprefixbefore_idx], 
                InitHeaderSearch::Angled, false, false, false);
        ++iwithprefixbefore_idx;
        iwithprefixbefore_done = 
          iwithprefixbefore_idx == iwithprefixbefore_vals.size();
      }
    }
  }

  Init.AddDefaultEnvVarPaths(Lang);

  // Add the clang headers, which are relative to the clang binary.
  llvm::sys::Path MainExecutablePath = 
     llvm::sys::Path::GetMainExecutable(Argv0,
                                    (void*)(intptr_t)InitializeIncludePaths);
  if (!MainExecutablePath.isEmpty()) {
    MainExecutablePath.eraseComponent();  // Remove /clang from foo/bin/clang
    MainExecutablePath.eraseComponent();  // Remove /bin   from foo/bin

    // Get foo/lib/clang/1.0/include    
    // 
    // FIXME: Don't embed version here.
    MainExecutablePath.appendComponent("lib");
    MainExecutablePath.appendComponent("clang");
    MainExecutablePath.appendComponent("1.0");
    MainExecutablePath.appendComponent("include");
    
    // We pass true to ignore sysroot so that we *always* look for clang headers
    // relative to our executable, never relative to -isysroot.
    Init.AddPath(MainExecutablePath.c_str(), InitHeaderSearch::System,
                 false, false, false, true /*ignore sysroot*/);
  }
  
  if (!nostdinc) 
    Init.AddDefaultSystemIncludePaths(Lang);

  // Now that we have collected all of the include paths, merge them all
  // together and tell the preprocessor about them.
  
  Init.Realize();
}

//===----------------------------------------------------------------------===//
// Driver PreprocessorFactory - For lazily generating preprocessors ...
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN DriverPreprocessorFactory : public PreprocessorFactory {
  const std::string &InFile;
  Diagnostic        &Diags;
  const LangOptions &LangInfo;
  TargetInfo        &Target;
  SourceManager     &SourceMgr;
  HeaderSearch      &HeaderInfo;
  
public:
  DriverPreprocessorFactory(const std::string &infile,
                            Diagnostic &diags, const LangOptions &opts,
                            TargetInfo &target, SourceManager &SM,
                            HeaderSearch &Headers)  
  : InFile(infile), Diags(diags), LangInfo(opts), Target(target),
    SourceMgr(SM), HeaderInfo(Headers) {}
  
  
  virtual ~DriverPreprocessorFactory() {}
  
  virtual Preprocessor* CreatePreprocessor() {
    llvm::OwningPtr<PTHManager> PTHMgr;

    if (!TokenCache.empty() && !ImplicitIncludePTH.empty()) {
      fprintf(stderr, "error: cannot use both -token-cache and -include-pth "
                      "options\n");
      exit(1);
    }
    
    // Use PTH?
    if (!TokenCache.empty() || !ImplicitIncludePTH.empty()) {
      const std::string& x = TokenCache.empty() ? ImplicitIncludePTH:TokenCache;
      PTHMgr.reset(PTHManager::Create(x, &Diags, 
                                      TokenCache.empty() ? Diagnostic::Error
                                                        : Diagnostic::Warning));
    }
    
    if (Diags.hasErrorOccurred())
      exit(1);
    
    // Create the Preprocessor.
    llvm::OwningPtr<Preprocessor> PP(new Preprocessor(Diags, LangInfo, Target,
                                                      SourceMgr, HeaderInfo,
                                                      PTHMgr.get()));
    
    // Note that this is different then passing PTHMgr to Preprocessor's ctor.
    // That argument is used as the IdentifierInfoLookup argument to
    // IdentifierTable's ctor.
    if (PTHMgr) {
      PTHMgr->setPreprocessor(PP.get());
      PP->setPTHManager(PTHMgr.take());
    }

    if (InitializePreprocessor(*PP, InFile))
      return 0;
    
    /// FIXME: PP can only handle one callback
    if (ProgAction != PrintPreprocessedInput) {
      std::string ErrStr;
      bool DFG = CreateDependencyFileGen(PP.get(), ErrStr);
      if (!DFG && !ErrStr.empty()) {
        fprintf(stderr, "%s", ErrStr.c_str());
        return 0;
      }
    }

    return PP.take();
  }
};
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

static void InitializeCompileOptions(CompileOptions &Opts,
                                     const LangOptions &LangOpts) {
  Opts.OptimizeSize = OptSize;
  Opts.DebugInfo = GenerateDebugInfo;
  if (OptSize) {
    // -Os implies -O2
    // FIXME: Diagnose conflicting options.
    Opts.OptimizationLevel = 2;
  } else {
    Opts.OptimizationLevel = OptLevel;
  }

  // FIXME: There are llvm-gcc options to control these selectively.
  Opts.InlineFunctions = (Opts.OptimizationLevel > 1);
  Opts.UnrollLoops = (Opts.OptimizationLevel > 1 && !OptSize);
  Opts.SimplifyLibCalls = !LangOpts.NoBuiltin;

#ifdef NDEBUG
  Opts.VerifyModule = 0;
#endif

  Opts.CPU = TargetCPU;
  Opts.Features.insert(Opts.Features.end(),
                       TargetFeatures.begin(), TargetFeatures.end());
  
  Opts.NoCommon = NoCommon | LangOpts.CPlusPlus;
  
  // Handle -ftime-report.
  Opts.TimePasses = TimeReport;
}

//===----------------------------------------------------------------------===//
// Fix-It Options
//===----------------------------------------------------------------------===//
static llvm::cl::list<ParsedSourceLocation>
FixItAtLocations("fixit-at", llvm::cl::value_desc("source-location"),
   llvm::cl::desc("Perform Fix-It modifications at the given source location"));

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
  
  LoggingDiagnosticClient(DiagnosticClient *Normal) {
    // Output diags both where requested...
    Chain1.reset(Normal);
    // .. and to our log file.
    Chain2.reset(new TextDiagnosticPrinter(*BuildLogFile,
                                           !NoShowColumn,
                                           !NoCaretDiagnostics,
                                           !NoShowLocation,
                                           PrintSourceRangeInfo,
                                           PrintDiagnosticOption));
  }
  
  virtual void setLangOptions(const LangOptions *LO) {
    Chain1->setLangOptions(LO);
    Chain2->setLangOptions(LO);
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

static void SetUpBuildDumpLog(unsigned argc, char **argv,
                              llvm::OwningPtr<DiagnosticClient> &DiagClient) {
  
  std::string ErrorInfo;
  BuildLogFile = new llvm::raw_fd_ostream(DumpBuildInformation.c_str(), false,
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
  DiagClient.reset(new LoggingDiagnosticClient(DiagClient.take()));
}



//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

/// CreateASTConsumer - Create the ASTConsumer for the corresponding program
/// action.  These consumers can operate on both ASTs that are freshly
/// parsed from source files as well as those deserialized from Bitcode.
/// Note that PP and PPF may be null here.
static ASTConsumer *CreateASTConsumer(const std::string& InFile,
                                      Diagnostic& Diag, FileManager& FileMgr, 
                                      const LangOptions& LangOpts,
                                      Preprocessor *PP,
                                      PreprocessorFactory *PPF) {
  switch (ProgAction) {
  default:
    return NULL;
    
  case ASTPrint:
    return CreateASTPrinter();
    
  case ASTDump:
    return CreateASTDumper();
    
  case ASTView:
    return CreateASTViewer();   

  case PrintDeclContext:
    return CreateDeclContextPrinter();
    
  case EmitHTML:
    return CreateHTMLPrinter(OutputFile, Diag, PP, PPF);

  case InheritanceView:
    return CreateInheritanceViewer(InheritanceViewCls);
    
  case TestSerialization:
    return CreateSerializationTest(Diag, FileMgr);
    
  case EmitAssembly:
  case EmitLLVM:
  case EmitBC: 
  case EmitLLVMOnly: {
    BackendAction Act;
    if (ProgAction == EmitAssembly)
      Act = Backend_EmitAssembly;
    else if (ProgAction == EmitLLVM)
      Act = Backend_EmitLL;
    else if (ProgAction == EmitLLVMOnly)
      Act = Backend_EmitNothing;
    else
      Act = Backend_EmitBC;
    
    CompileOptions Opts;
    InitializeCompileOptions(Opts, LangOpts);
    return CreateBackendConsumer(Act, Diag, LangOpts, Opts, 
                                 InFile, OutputFile);
  }

  case SerializeAST:
    // FIXME: Allow user to tailor where the file is written.
    return CreateASTSerializer(InFile, OutputFile, Diag);
    
  case GeneratePCH:
    assert(PP && "Generate PCH doesn't work from serialized file yet");
    return CreatePCHGenerator(*PP, OutputFile);    

  case RewriteObjC:
    return CreateCodeRewriterTest(InFile, OutputFile, Diag, LangOpts);

  case RewriteBlocks:
    return CreateBlockRewriter(InFile, OutputFile, Diag, LangOpts);
    
  case RunAnalysis:
    return CreateAnalysisConsumer(Diag, PP, PPF, LangOpts, OutputFile);
  }
}

/// ProcessInputFile - Process a single input file with the specified state.
///
static void ProcessInputFile(Preprocessor &PP, PreprocessorFactory &PPF,
                             const std::string &InFile, ProgActions PA) {
  llvm::OwningPtr<ASTConsumer> Consumer;
  bool ClearSourceMgr = false;
  FixItRewriter *FixItRewrite = 0;
  bool CompleteTranslationUnit = true;

  switch (PA) {
  default:
    Consumer.reset(CreateASTConsumer(InFile, PP.getDiagnostics(),
                                     PP.getFileManager(), PP.getLangOptions(),
                                     &PP, &PPF));
    
    if (!Consumer) {      
      fprintf(stderr, "Unexpected program action!\n");
      HadErrors = true;
      return;
    }

    if (ProgAction == GeneratePCH)
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
  case RunPreprocessorOnly: {        // Just lex as fast as we can, no output.
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
      
  case GeneratePTH: {
    llvm::TimeRegion Timer(ClangFrontendTimer);
    CacheTokens(PP, OutputFile);
    ClearSourceMgr = true;
    break;
  }      

  case PrintPreprocessedInput: {      // -E mode.
    llvm::TimeRegion Timer(ClangFrontendTimer);
    DoPrintPreprocessedInput(PP, OutputFile);
    ClearSourceMgr = true;
    break;
  }
      
  case ParseNoop: {                  // -parse-noop
    llvm::TimeRegion Timer(ClangFrontendTimer);
    ParseFile(PP, new MinimalAction(PP));
    ClearSourceMgr = true;
    break;
  }
    
  case ParsePrintCallbacks: {
    llvm::TimeRegion Timer(ClangFrontendTimer);
    ParseFile(PP, CreatePrintParserActionsAction(PP));
    ClearSourceMgr = true;
    break;
  }

  case ParseSyntaxOnly: {             // -fsyntax-only
    llvm::TimeRegion Timer(ClangFrontendTimer);
    Consumer.reset(new ASTConsumer());
    break;
  }
      
  case RewriteMacros:
    RewriteMacrosInInput(PP, InFile, OutputFile);
    ClearSourceMgr = true;
    break;
      
  case RewriteTest: {
    DoRewriteTest(PP, InFile, OutputFile);
    ClearSourceMgr = true;
    break;
  }

  case FixIt:
    llvm::TimeRegion Timer(ClangFrontendTimer);
    Consumer.reset(new ASTConsumer());
    FixItRewrite = new FixItRewriter(PP.getDiagnostics(),
                                     PP.getSourceManager(),
                                     PP.getLangOptions());
    break;
  }

  if (Consumer) {
    llvm::OwningPtr<ASTContext> ContextOwner;

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
        if (FixItAtLocations[Idx].ResolveLocation(PP.getFileManager(), 
                                                  Requested)) {
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

    ContextOwner.reset(new ASTContext(PP.getLangOptions(),
                                      PP.getSourceManager(),
                                      PP.getTargetInfo(),
                                      PP.getIdentifierTable(),
                                      PP.getSelectorTable(),
                                      /* FreeMemory = */ !DisableFree));
    
    if (!ImplicitIncludePCH.empty()) {
      // The user has asked us to include a precompiled header. Load
      // the precompiled header into the AST context.
      llvm::OwningPtr<PCHReader> Reader(new PCHReader(PP, *ContextOwner.get()));
      switch (Reader->ReadPCH(ImplicitIncludePCH)) {
      case PCHReader::Success: {
        // Attach the PCH reader to the AST context as an external AST
        // source, so that declarations will be deserialized from the
        // PCH file as needed.
        llvm::OwningPtr<ExternalASTSource> Source(Reader.take());
        ContextOwner->setExternalSource(Source);

        // Clear out the predefines buffer, because all of the
        // predefines are already in the PCH file.
        PP.setPredefines("");
        break;
      }

      case PCHReader::Failure:
        // Unrecoverable failure: don't even try to process the input
        // file.
        return;

      case PCHReader::IgnorePCH:
        // No suitable PCH file could be found. Just ignore the
        // -include-pch option entirely.
        break;
      }

      // Finish preprocessor initialization. We do this now (rather
      // than earlier) because this initialization creates new source
      // location entries in the source manager, which must come after
      // the source location entries for the PCH file.
      if (InitializeSourceManager(PP, InFile))
        return;
    }

    ParseAST(PP, Consumer.get(), *ContextOwner.get(), Stats, 
             CompleteTranslationUnit);
    
    if (FixItRewrite)
      FixItRewrite->WriteFixedFile(InFile, OutputFile);

    // If in -disable-free mode, don't deallocate these when they go out of
    // scope.
    if (DisableFree)
      ContextOwner.take();
  }

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

  if (DisableFree)
    Consumer.take();
}

static void ProcessSerializedFile(const std::string& InFile, Diagnostic& Diag,
                                  FileManager& FileMgr) {
  
  if (VerifyDiagnostics) {
    fprintf(stderr, "-verify does not yet work with serialized ASTs.\n");
    exit (1);
  }
  
  llvm::sys::Path Filename(InFile);
  
  if (!Filename.isValid()) {
    fprintf(stderr, "serialized file '%s' not available.\n",InFile.c_str());
    exit (1);
  }
  
  llvm::OwningPtr<ASTContext> Ctx;
  
  // Create the memory buffer that contains the contents of the file.  
  llvm::OwningPtr<llvm::MemoryBuffer> 
    MBuffer(llvm::MemoryBuffer::getFile(Filename.c_str()));
  
  if (MBuffer)
    Ctx.reset(ASTContext::ReadASTBitcodeBuffer(*MBuffer, FileMgr));
  
  if (!Ctx) {
    fprintf(stderr, "error: file '%s' could not be deserialized\n", 
            InFile.c_str());
    exit (1);
  }
  
  // Observe that we use the source file name stored in the deserialized
  // translation unit, rather than InFile.
  llvm::OwningPtr<ASTConsumer>
    Consumer(CreateASTConsumer(InFile, Diag, FileMgr, Ctx->getLangOptions(),
                               0, 0));

  if (!Consumer) {      
    fprintf(stderr, "Unsupported program action with serialized ASTs!\n");
    exit (1);
  }

  Consumer->Initialize(*Ctx);

  // FIXME: We need to inform Consumer about completed TagDecls as well.
  TranslationUnitDecl *TUD = Ctx->getTranslationUnitDecl();
  for (DeclContext::decl_iterator I = TUD->decls_begin(*Ctx), 
                                  E = TUD->decls_end(*Ctx);
       I != E; ++I)
    Consumer->HandleTopLevelDecl(DeclGroupRef(*I));
}


static llvm::cl::list<std::string>
InputFilenames(llvm::cl::Positional, llvm::cl::desc("<input files>"));

static bool isSerializedFile(const std::string& InFile) {
  if (InFile.size() < 4)
    return false;
  
  const char* s = InFile.c_str()+InFile.size()-4;
  return s[0] == '.' && s[1] == 'a' && s[2] == 's' && s[3] == 't';    
}


int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                              "LLVM 'Clang' Compiler: http://clang.llvm.org\n");
  
  if (TimeReport)
    ClangFrontendTimer = new llvm::Timer("Clang front-end time");
  
  // If no input was specified, read from stdin.
  if (InputFilenames.empty())
    InputFilenames.push_back("-");
  
  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  llvm::OwningPtr<DiagnosticClient> DiagClient;
  if (VerifyDiagnostics) {
    // When checking diagnostics, just buffer them up.
    DiagClient.reset(new TextDiagnosticBuffer());
    if (InputFilenames.size() != 1) {
      fprintf(stderr, "-verify only works on single input files for now.\n");
      return 1;
    }
    if (!HTMLDiag.empty()) {
      fprintf(stderr, "-verify and -html-diags don't work together\n");
      return 1;
    }
  } else if (HTMLDiag.empty()) {
    // Print diagnostics to stderr by default.
    DiagClient.reset(new TextDiagnosticPrinter(llvm::errs(),
                                               !NoShowColumn,
                                               !NoCaretDiagnostics,
                                               !NoShowLocation,
                                               PrintSourceRangeInfo,
                                               PrintDiagnosticOption));
  } else {
    DiagClient.reset(CreateHTMLDiagnosticClient(HTMLDiag));
  }
  
  if (!DumpBuildInformation.empty()) {
    if (!HTMLDiag.empty()) {
      fprintf(stderr,
              "-dump-build-information and -html-diags don't work together\n");
      return 1;
    }
    
    SetUpBuildDumpLog(argc, argv, DiagClient);
  }
  

  // Configure our handling of diagnostics.
  Diagnostic Diags(DiagClient.get());
  if (ProcessWarningOptions(Diags))
    return 1;

  // -I- is a deprecated GCC feature, scan for it and reject it.
  for (unsigned i = 0, e = I_dirs.size(); i != e; ++i) {
    if (I_dirs[i] == "-") {
      Diags.Report(FullSourceLoc(), diag::err_pp_I_dash_not_supported);      
      I_dirs.erase(I_dirs.begin()+i);
      --i;
    }
  }

  // Get information about the target being compiled for.
  std::string Triple = CreateTargetTriple();
  llvm::OwningPtr<TargetInfo> Target(TargetInfo::CreateTargetInfo(Triple));
  
  if (Target == 0) {
    Diags.Report(FullSourceLoc(), diag::err_fe_unknown_triple) 
      << Triple.c_str();
    return 1;
  }
  
  if (!InheritanceViewCls.empty())  // C++ visualization?
    ProgAction = InheritanceView;
    
  llvm::OwningPtr<SourceManager> SourceMgr;
  
  // Create a file manager object to provide access to and cache the filesystem.
  FileManager FileMgr;
  
  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    const std::string &InFile = InputFilenames[i];
    
    if (isSerializedFile(InFile)) {
      ProcessSerializedFile(InFile,Diags,FileMgr);
      continue;
    }
    
    /// Create a SourceManager object.  This tracks and owns all the file
    /// buffers allocated to a translation unit.
    if (!SourceMgr)
      SourceMgr.reset(new SourceManager());
    else
      SourceMgr->clearIDTables();
    
    // Initialize language options, inferring file types from input filenames.
    LangOptions LangInfo;
    DiagClient->setLangOptions(&LangInfo);
    
    InitializeBaseLanguage();
    LangKind LK = GetLanguage(InFile);
    InitializeLangOptions(LangInfo, LK);
    InitializeLanguageStandard(LangInfo, LK, Target.get());
          
    // Process the -I options and set them in the HeaderInfo.
    HeaderSearch HeaderInfo(FileMgr);
    
    
    InitializeIncludePaths(argv[0], HeaderInfo, FileMgr, LangInfo);
    
    // Set up the preprocessor with these options.
    DriverPreprocessorFactory PPFactory(InFile, Diags, LangInfo, *Target,
                                        *SourceMgr.get(), HeaderInfo);
    
    llvm::OwningPtr<Preprocessor> PP(PPFactory.CreatePreprocessor());
          
    if (!PP)
      continue;

    if (ImplicitIncludePCH.empty() && 
        InitializeSourceManager(*PP.get(), InFile))
      continue;

    if (!HTMLDiag.empty())
      ((PathDiagnosticClient*)DiagClient.get())->SetPreprocessor(PP.get());

    // Process the source file.
    ProcessInputFile(*PP, PPFactory, InFile, ProgAction);
    
    HeaderInfo.ClearFileInfo();
    DiagClient->setLangOptions(0);
  }

  if (Verbose)
    fprintf(stderr, "clang version 1.0 based upon " PACKAGE_STRING
            " hosted on " LLVM_HOSTTRIPLE "\n");

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

  return HadErrors || (Diags.getNumErrors() != 0);
}
