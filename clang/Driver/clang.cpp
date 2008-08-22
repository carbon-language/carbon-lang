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
//   -ffatal-errors
//   -ftabstop=width
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "ASTConsumers.h"
#include "HTMLDiagnostics.h"
#include "clang/Driver/InitHeaderSearch.h"
#include "clang/Driver/TextDiagnosticBuffer.h"
#include "clang/Driver/TextDiagnosticPrinter.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/AST/Decl.h"
#include "clang/AST/TranslationUnit.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Sema/ParseAST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Parse/Parser.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Signals.h"
#include "llvm/Config/config.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/System/Path.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Global options.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool>
Verbose("v", llvm::cl::desc("Enable verbose output"));
static llvm::cl::opt<bool>
Stats("print-stats", 
      llvm::cl::desc("Print performance metrics and statistics"));

enum ProgActions {
  RewriteObjC,                  // ObjC->C Rewriter.
  RewriteMacros,                // Expand macros but not #includes.
  HTMLTest,                     // HTML displayer testing stuff.
  EmitLLVM,                     // Emit a .ll file.
  EmitBC,                       // Emit a .bc file.
  SerializeAST,                 // Emit a .ast file.
  EmitHTML,                     // Translate input source into HTML.
  ASTPrint,                     // Parse ASTs and print them.
  ASTDump,                      // Parse ASTs and dump them.
  ASTView,                      // Parse ASTs and view them in Graphviz.
  TestSerialization,            // Run experimental serialization code.
  ParsePrintCallbacks,          // Parse and print each callback.
  ParseSyntaxOnly,              // Parse and perform semantic analysis.
  ParseNoop,                    // Parse with noop callbacks.
  RunPreprocessorOnly,          // Just lex, no output.
  PrintPreprocessedInput,       // -E mode.
  DumpTokens,                   // Token dump mode.
  RunAnalysis                   // Run one or more source code analyses. 
};

static llvm::cl::opt<ProgActions> 
ProgAction(llvm::cl::desc("Choose output type:"), llvm::cl::ZeroOrMore,
           llvm::cl::init(ParseSyntaxOnly),
           llvm::cl::values(
             clEnumValN(RunPreprocessorOnly, "Eonly",
                        "Just run preprocessor, no output (for timings)"),
             clEnumValN(PrintPreprocessedInput, "E",
                        "Run preprocessor, emit preprocessed file"),
             clEnumValN(DumpTokens, "dumptokens",
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
             clEnumValN(TestSerialization, "test-pickling",
                        "Run prototype serialization code"),
             clEnumValN(EmitLLVM, "emit-llvm",
                        "Build ASTs then convert to LLVM, emit .ll file"),
             clEnumValN(EmitBC, "emit-llvm-bc",
                        "Build ASTs then convert to LLVM, emit .bc file"),
             clEnumValN(SerializeAST, "serialize",
                        "Build ASTs and emit .ast file"),
             clEnumValN(RewriteObjC, "rewrite-objc",
                        "Rewrite ObjC into C (code rewriter example)"),
             clEnumValN(RewriteMacros, "rewrite-macros",
                        "Expand macros without full preprocessing"),
             clEnumValEnd));


static llvm::cl::opt<std::string>
OutputFile("o",
 llvm::cl::value_desc("path"),
 llvm::cl::desc("Specify output file (for --serialize, this is a directory)"));

//===----------------------------------------------------------------------===//
// Code Generator Options
//===----------------------------------------------------------------------===//
static llvm::cl::opt<bool>
GenerateDebugInfo("g",
                  llvm::cl::desc("Generate source level debug information"));

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
NoCaretDiagnostics("fno-caret-diagnostics",
                   llvm::cl::desc("Do not include source line and caret with"
                                  " diagnostics"));


//===----------------------------------------------------------------------===//
// Analyzer Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool>
VisualizeEG("visualize-egraph",
            llvm::cl::desc("Display static analysis Exploded Graph"));

static llvm::cl::opt<bool>
AnalyzeAll("checker-opt-analyze-headers",
    llvm::cl::desc("Force the static analyzer to analyze "
                   "functions defined in header files"));

static llvm::cl::list<Analyses>
AnalysisList(llvm::cl::desc("Available Source Code Analyses:"),
llvm::cl::values(
#define ANALYSIS(NAME, CMDFLAG, DESC, SCOPE)\
clEnumValN(NAME, CMDFLAG, DESC),
#include "Analyses.def"
clEnumValEnd));

//===----------------------------------------------------------------------===//
// Language Options
//===----------------------------------------------------------------------===//

enum LangKind {
  langkind_unspecified,
  langkind_c,
  langkind_c_cpp,
  langkind_cxx,
  langkind_cxx_cpp,
  langkind_objc,
  langkind_objc_cpp,
  langkind_objcxx,
  langkind_objcxx_cpp
};

/* TODO: GCC also accepts:
   c-header c++-header objective-c-header objective-c++-header
   assembler  assembler-with-cpp
   ada, f77*, ratfor (!), f95, java, treelang
 */
static llvm::cl::opt<LangKind>
BaseLang("x", llvm::cl::desc("Base language to compile"),
         llvm::cl::init(langkind_unspecified),
   llvm::cl::values(clEnumValN(langkind_c,     "c",            "C"),
                    clEnumValN(langkind_cxx,   "c++",          "C++"),
                    clEnumValN(langkind_objc,  "objective-c",  "Objective C"),
                    clEnumValN(langkind_objcxx,"objective-c++","Objective C++"),
                    clEnumValN(langkind_c_cpp,     "c-cpp-output",
                               "Preprocessed C"),
                    clEnumValN(langkind_cxx_cpp,   "c++-cpp-output",
                               "Preprocessed C++"),
                    clEnumValN(langkind_objc_cpp,  "objective-c-cpp-output",
                               "Preprocessed Objective C"),
                    clEnumValN(langkind_objcxx_cpp,"objective-c++-cpp-output",
                               "Preprocessed Objective C++"),
                    clEnumValEnd));

static llvm::cl::opt<bool>
LangObjC("ObjC", llvm::cl::desc("Set base language to Objective-C"),
         llvm::cl::Hidden);
static llvm::cl::opt<bool>
LangObjCXX("ObjC++", llvm::cl::desc("Set base language to Objective-C++"),
           llvm::cl::Hidden);

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


static void InitializeLangOptions(LangOptions &Options, LangKind LK) {
  // FIXME: implement -fpreprocessed mode.
  bool NoPreprocess = false;
  
  switch (LK) {
  default: assert(0 && "Unknown language kind!");
  case langkind_c_cpp:
    NoPreprocess = true;
    // FALLTHROUGH
  case langkind_c:
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
    Options.ObjC1 = Options.ObjC2 = 1;      
    break;
  case langkind_objcxx_cpp:
    NoPreprocess = true;
    // FALLTHROUGH
  case langkind_objcxx:
    Options.ObjC1 = Options.ObjC2 = 1;
    Options.CPlusPlus = 1;
    break;
  }
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
//                 clEnumValN(lang_c99,      "c9x",            "ISO C 1999"),
                   clEnumValN(lang_c99,      "iso9899:1999",   "ISO C 1999"),
//                 clEnumValN(lang_c99,      "iso9899:199x",   "ISO C 1999"),
                   clEnumValN(lang_gnu89,    "gnu89",
                              "ISO C 1990 with GNU extensions (default for C)"),
                   clEnumValN(lang_gnu99,    "gnu99",
                              "ISO C 1999 with GNU extensions"),
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
                              "extensions (default for C++)"),
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
LaxVectorConversions("flax-vector-conversions",
                     llvm::cl::desc("Allow implicit conversions between vectors"
                                    " with a different number of elements or "
                                    "different element types"));

// FIXME: This (and all GCC -f options) really come in -f... and
// -fno-... forms, and additionally support automagic behavior when
// they are not defined. For example, -fexceptions defaults to on or
// off depending on the language. We should support this behavior in
// some form (perhaps just add a facility for distinguishing when an
// has its default value from when it has been set to its default
// value).
static llvm::cl::opt<bool>
Exceptions("fexceptions",
           llvm::cl::desc("Enable support for exception handling."));

static llvm::cl::opt<bool>
GNURuntime("fgnu-runtime",
            llvm::cl::desc("Generate output compatible with the standard GNU Objective-C runtime."));

static llvm::cl::opt<bool>
NeXTRuntime("fnext-runtime",
            llvm::cl::desc("Generate output compatible with the NeXT runtime."));

// FIXME: add:
//   -ansi
//   -trigraphs
//   -fdollars-in-identifiers
//   -fpascal-strings
static void InitializeLanguageStandard(LangOptions &Options, LangKind LK) {
  if (LangStd == lang_unspecified) {
    // Based on the base language, pick one.
    switch (LK) {
    default: assert(0 && "Unknown base language");
    case langkind_c:
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
  
  if (LangStd == lang_c89 || LangStd == lang_c94 || LangStd == lang_gnu89)
    Options.ImplicitInt = 1;
  else
    Options.ImplicitInt = 0;
  Options.Trigraphs = 1; // -trigraphs or -ansi
  Options.DollarIdents = 1;  // FIXME: Really a target property.
  Options.PascalStrings = PascalStrings;
  Options.Microsoft = MSExtensions;
  Options.WritableStrings = WritableStrings;
  Options.LaxVectorConversions = LaxVectorConversions;
  Options.Exceptions = Exceptions;

  if (NeXTRuntime) {
    Options.NeXTRuntime = 1;
  } else if (GNURuntime) {
    Options.NeXTRuntime = 0;
  } else {
    // FIXME: Should autoselect based on platform.
    Options.NeXTRuntime = 0;
  }
}

static llvm::cl::opt<bool>
ObjCExclusiveGC("fobjc-gc-only",
                llvm::cl::desc("Use GC exclusively for Objective-C related "
                               "memory management"));

static llvm::cl::opt<bool>
ObjCEnableGC("fobjc-gc",
             llvm::cl::desc("Enable Objective-C garbage collection"));

void InitializeGCMode(LangOptions &Options) {
  if (ObjCExclusiveGC)
    Options.setGCMode(LangOptions::GCOnly);
  else if (ObjCEnableGC)
    Options.setGCMode(LangOptions::HybridGC);
}


//===----------------------------------------------------------------------===//
// Our DiagnosticClient implementation
//===----------------------------------------------------------------------===//

// FIXME: Werror should take a list of things, -Werror=foo,bar
static llvm::cl::opt<bool>
WarningsAsErrors("Werror", llvm::cl::desc("Treat all warnings as errors"));

static llvm::cl::opt<bool>
SilenceWarnings("w", llvm::cl::desc("Do not emit any warnings"));

static llvm::cl::opt<bool>
WarnOnExtensions("pedantic", llvm::cl::init(false),
                 llvm::cl::desc("Issue a warning on uses of GCC extensions"));

static llvm::cl::opt<bool>
ErrorOnExtensions("pedantic-errors",
                  llvm::cl::desc("Issue an error on uses of GCC extensions"));

static llvm::cl::opt<bool>
WarnUnusedMacros("Wunused_macros",
         llvm::cl::desc("Warn for unused macros in the main translation unit"));

static llvm::cl::opt<bool>
WarnFloatEqual("Wfloat-equal",
   llvm::cl::desc("Warn about equality comparisons of floating point values"));

static llvm::cl::opt<bool>
WarnNoFormatNonLiteral("Wno-format-nonliteral",
   llvm::cl::desc("Do not warn about non-literal format strings"));

static llvm::cl::opt<bool>
WarnUndefMacros("Wundef",
   llvm::cl::desc("Warn on use of undefined macros in #if's"));

static llvm::cl::opt<bool>
WarnImplicitFunctionDeclaration("Wimplicit-function-declaration",
   llvm::cl::desc("Warn about uses of implicitly defined functions"));

/// InitializeDiagnostics - Initialize the diagnostic object, based on the
/// current command line option settings.
static void InitializeDiagnostics(Diagnostic &Diags) {
  Diags.setIgnoreAllWarnings(SilenceWarnings);
  Diags.setWarningsAsErrors(WarningsAsErrors);
  Diags.setWarnOnExtensions(WarnOnExtensions);
  Diags.setErrorOnExtensions(ErrorOnExtensions);

  // Silence the "macro is not used" warning unless requested.
  if (!WarnUnusedMacros)
    Diags.setDiagnosticMapping(diag::pp_macro_not_used, diag::MAP_IGNORE);
               
  // Silence "floating point comparison" warnings unless requested.
  if (!WarnFloatEqual)
    Diags.setDiagnosticMapping(diag::warn_floatingpoint_eq, diag::MAP_IGNORE);

  // Silence "format string is not a string literal" warnings if requested
  if (WarnNoFormatNonLiteral)
    Diags.setDiagnosticMapping(diag::warn_printf_not_string_constant,
                               diag::MAP_IGNORE);
  if (!WarnUndefMacros)
    Diags.setDiagnosticMapping(diag::warn_pp_undef_identifier,diag::MAP_IGNORE);
    
  if (!WarnImplicitFunctionDeclaration)
    Diags.setDiagnosticMapping(diag::warn_implicit_function_decl,
                               diag::MAP_IGNORE);
  
  if (MSExtensions) // MS allows unnamed struct/union fields.
    Diags.setDiagnosticMapping(diag::w_no_declarators, diag::MAP_IGNORE);

  // If -pedantic-errors is set, turn extensions that warn by default into
  // errors. 
  if (ErrorOnExtensions) {
    Diags.setDiagnosticMapping(diag::warn_hex_escape_too_large,
                               diag::MAP_ERROR);
    Diags.setDiagnosticMapping(diag::warn_octal_escape_too_large,
                               diag::MAP_ERROR);
  }
}

//===----------------------------------------------------------------------===//
// Analysis-specific options.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
AnalyzeSpecificFunction("analyze-function",
                llvm::cl::desc("Run analysis on specific function"));

static llvm::cl::opt<bool>
TrimGraph("trim-egraph",
      llvm::cl::desc("Only show error-related paths in the analysis graph"));

//===----------------------------------------------------------------------===//
// Target Triple Processing.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
TargetTriple("triple",
  llvm::cl::desc("Specify target triple (e.g. i686-apple-darwin9)"));

static llvm::cl::opt<std::string>
Arch("arch", llvm::cl::desc("Specify target architecture (e.g. i686)"));

static std::string CreateTargetTriple() {
  // Initialize base triple.  If a -triple option has been specified, use
  // that triple.  Otherwise, default to the host triple.
  std::string Triple = TargetTriple;
  if (Triple.empty()) Triple = LLVM_HOSTTRIPLE;
  
  // If -arch foo was specified, remove the architecture from the triple we have
  // so far and replace it with the specified one.
  if (Arch.empty())
    return Triple;
    
  // Decompose the base triple into "arch" and suffix.
  std::string::size_type FirstDashIdx = Triple.find("-");
  
  if (FirstDashIdx == std::string::npos) {
    fprintf(stderr, 
            "Malformed target triple: \"%s\" ('-' could not be found).\n",
            Triple.c_str());
    exit(1);
  }
  
  return Arch + std::string(Triple.begin()+FirstDashIdx, Triple.end());
}

//===----------------------------------------------------------------------===//
// Preprocessor Initialization
//===----------------------------------------------------------------------===//

// FIXME: Preprocessor builtins to support.
//   -A...    - Play with #assertions
//   -undef   - Undefine all predefined macros

static llvm::cl::list<std::string>
D_macros("D", llvm::cl::value_desc("macro"), llvm::cl::Prefix,
       llvm::cl::desc("Predefine the specified macro"));
static llvm::cl::list<std::string>
U_macros("U", llvm::cl::value_desc("macro"), llvm::cl::Prefix,
         llvm::cl::desc("Undefine the specified macro"));

static llvm::cl::list<std::string>
ImplicitIncludes("include", llvm::cl::value_desc("file"),
                 llvm::cl::desc("Include file before parsing"));


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
    Buf.insert(Buf.end(), Equal+1, Equal+strlen(Equal));
  } else {
    // Push "macroname 1".
    Buf.insert(Buf.end(), Macro, Macro+strlen(Macro));
    Buf.push_back(' ');
    Buf.push_back('1');
  }
  Buf.push_back('\n');
}

/// AddImplicitInclude - Add an implicit #include of the specified file to the
/// predefines buffer.
static void AddImplicitInclude(std::vector<char> &Buf, const std::string &File){
  const char *Inc = "#include \"";
  Buf.insert(Buf.end(), Inc, Inc+strlen(Inc));
  Buf.insert(Buf.end(), File.begin(), File.end());
  Buf.push_back('"');
  Buf.push_back('\n');
}


/// InitializePreprocessor - Initialize the preprocessor getting it and the
/// environment ready to process a single file. This returns true on error.
///
static bool InitializePreprocessor(Preprocessor &PP,
                                   bool InitializeSourceMgr, 
                                   const std::string &InFile) {
  FileManager &FileMgr = PP.getFileManager();
  
  // Figure out where to get and map in the main file.
  SourceManager &SourceMgr = PP.getSourceManager();

  if (InitializeSourceMgr) {
    if (InFile != "-") {
      const FileEntry *File = FileMgr.getFile(InFile);
      if (File) SourceMgr.createMainFileID(File, SourceLocation());
      if (SourceMgr.getMainFileID() == 0) {
        fprintf(stderr, "Error reading '%s'!\n",InFile.c_str());
        return true;
      }
    } else {
      llvm::MemoryBuffer *SB = llvm::MemoryBuffer::getSTDIN();
      if (SB) SourceMgr.createMainFileIDForMemBuffer(SB);
      if (SourceMgr.getMainFileID() == 0) {
        fprintf(stderr, "Error reading standard input!  Empty?\n");
        return true;
      }
    }
  }

  std::vector<char> PredefineBuffer;

  // Add macros from the command line.
  unsigned d = 0, D = D_macros.size();
  unsigned u = 0, U = U_macros.size();
  while (d < D || u < U) {
    if (u == U || (d < D && D_macros.getPosition(d) < U_macros.getPosition(u)))
      DefineBuiltinMacro(PredefineBuffer, D_macros[d++].c_str());
    else
      DefineBuiltinMacro(PredefineBuffer, U_macros[u++].c_str(), "#undef ");
  }

  // FIXME: Read any files specified by -imacros.
  
  // Add implicit #includes from -include.
  for (unsigned i = 0, e = ImplicitIncludes.size(); i != e; ++i)
    AddImplicitInclude(PredefineBuffer, ImplicitIncludes[i]);
  
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
// FIXME: -imacros

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

  // Add the clang headers, which are relative to the clang driver.
  llvm::sys::Path MainExecutablePath = 
     llvm::sys::Path::GetMainExecutable(Argv0,
                                    (void*)(intptr_t)InitializeIncludePaths);
  if (!MainExecutablePath.isEmpty()) {
    MainExecutablePath.eraseComponent();  // Remove /clang from foo/bin/clang
    MainExecutablePath.eraseComponent();  // Remove /bin   from foo/bin
    MainExecutablePath.appendComponent("Headers"); // Get foo/Headers
    Init.AddPath(MainExecutablePath.c_str(), InitHeaderSearch::System,
        false, false, false);
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
  bool              InitializeSourceMgr;
  
public:
  DriverPreprocessorFactory(const std::string &infile,
                            Diagnostic &diags, const LangOptions &opts,
                            TargetInfo &target, SourceManager &SM,
                            HeaderSearch &Headers)  
  : InFile(infile), Diags(diags), LangInfo(opts), Target(target),
    SourceMgr(SM), HeaderInfo(Headers), InitializeSourceMgr(true) {}
  
  
  virtual ~DriverPreprocessorFactory() {}
  
  virtual Preprocessor* CreatePreprocessor() {
    Preprocessor* PP = new Preprocessor(Diags, LangInfo, Target,
                                        SourceMgr, HeaderInfo);
    
    if (InitializePreprocessor(*PP, InitializeSourceMgr, InFile)) {
      delete PP;
      return NULL;
    }
    
    InitializeSourceMgr = false;
    return PP;
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

  if (VerifyDiagnostics)
    exit(CheckDiagnostics(PP));
}

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

/// CreateASTConsumer - Create the ASTConsumer for the corresponding program
///  action.  These consumers can operate on both ASTs that are freshly
///  parsed from source files as well as those deserialized from Bitcode.
static ASTConsumer* CreateASTConsumer(const std::string& InFile,
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
      
    case EmitHTML:
      return CreateHTMLPrinter(OutputFile, Diag, PP, PPF);

    case TestSerialization:
      return CreateSerializationTest(Diag, FileMgr);
      
    case EmitLLVM:
    case EmitBC:
      return CreateLLVMCodeGenWriter(ProgAction == EmitBC, Diag, LangOpts,
                                     InFile, OutputFile, GenerateDebugInfo);

    case SerializeAST:
      // FIXME: Allow user to tailor where the file is written.
      return CreateASTSerializer(InFile, OutputFile, Diag);
      
    case RewriteObjC:
      return CreateCodeRewriterTest(InFile, OutputFile, Diag, LangOpts);
      
    case RunAnalysis:
      assert (!AnalysisList.empty());
      return CreateAnalysisConsumer(&AnalysisList[0],
                                    &AnalysisList[0]+AnalysisList.size(),
                                    Diag, PP, PPF, LangOpts,
                                    AnalyzeSpecificFunction,
                                    OutputFile, VisualizeEG, TrimGraph,
                                    AnalyzeAll);
  }
}

/// ProcessInputFile - Process a single input file with the specified state.
///
static void ProcessInputFile(Preprocessor &PP, PreprocessorFactory &PPF,
                             const std::string &InFile) {

  llvm::OwningPtr<ASTConsumer> Consumer;
  bool ClearSourceMgr = false;
  
  switch (ProgAction) {
  default:
    Consumer.reset(CreateASTConsumer(InFile, PP.getDiagnostics(),
                                     PP.getFileManager(), PP.getLangOptions(),
                                     &PP, &PPF));
    
    if (!Consumer) {      
      fprintf(stderr, "Unexpected program action!\n");
      return;
    }

    break;
      
  case DumpTokens: {                 // Token dump mode.
    Token Tok;
    // Start parsing the specified input file.
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
    Token Tok;
    // Start parsing the specified input file.
    PP.EnterMainSourceFile();
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::eof));
    ClearSourceMgr = true;
    break;
  }
    
  case PrintPreprocessedInput:       // -E mode.
    DoPrintPreprocessedInput(PP, OutputFile);
    ClearSourceMgr = true;
    break;
      
  case ParseNoop:                    // -parse-noop
    ParseFile(PP, new MinimalAction(PP.getIdentifierTable()));
    ClearSourceMgr = true;
    break;
    
  case ParsePrintCallbacks:
    ParseFile(PP, CreatePrintParserActionsAction(PP.getIdentifierTable()));
    ClearSourceMgr = true;
    break;
      
  case ParseSyntaxOnly:              // -fsyntax-only
    Consumer.reset(new ASTConsumer());
    break;
      
  case RewriteMacros:
    RewriteMacrosInInput(PP, InFile, OutputFile);
    ClearSourceMgr = true;
    break;
  }
  
  if (Consumer) {
    if (VerifyDiagnostics)
      exit(CheckASTConsumer(PP, Consumer.get()));
    
    ParseAST(PP, Consumer.get(), Stats);
  }

  if (Stats) {
    fprintf(stderr, "\nSTATISTICS FOR '%s':\n", InFile.c_str());
    PP.PrintStats();
    PP.getIdentifierTable().PrintStats();
    PP.getHeaderSearchInfo().PrintStats();
    if (ClearSourceMgr)
      PP.getSourceManager().PrintStats();
    fprintf(stderr, "\n");
  }

  // For a multi-file compilation, some things are ok with nuking the source 
  // manager tables, other require stable fileid/macroid's across multiple
  // files.
  if (ClearSourceMgr)
    PP.getSourceManager().clearIDTables();
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
  
  llvm::OwningPtr<TranslationUnit> TU(ReadASTBitcodeFile(Filename, FileMgr));
  
  if (!TU) {
    fprintf(stderr, "error: file '%s' could not be deserialized\n", 
            InFile.c_str());
    exit (1);
  }
  
  // Observe that we use the source file name stored in the deserialized
  // translation unit, rather than InFile.
  llvm::OwningPtr<ASTConsumer>
    Consumer(CreateASTConsumer(InFile, Diag, FileMgr, TU->getLangOptions(),
                               0, 0));

  if (!Consumer) {      
    fprintf(stderr, "Unsupported program action with serialized ASTs!\n");
    exit (1);
  }

  Consumer->Initialize(TU->getContext());

  // FIXME: We need to inform Consumer about completed TagDecls as well.
  for (TranslationUnit::iterator I=TU->begin(), E=TU->end(); I!=E; ++I)
    Consumer->HandleTopLevelDecl(*I);
}


static llvm::cl::list<std::string>
InputFilenames(llvm::cl::Positional, llvm::cl::desc("<input files>"));

static bool isSerializedFile(const std::string& InFile) {
  if (InFile.size() < 4)
    return false;
  
  const char* s = InFile.c_str()+InFile.size()-4;
  
  return s[0] == '.' &&
         s[1] == 'a' &&
         s[2] == 's' &&
         s[3] == 't';    
}


int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, " llvm clang cfe\n");
  llvm::sys::PrintStackTraceOnErrorSignal();
  
  // If no input was specified, read from stdin.
  if (InputFilenames.empty())
    InputFilenames.push_back("-");
    
  // Create a file manager object to provide access to and cache the filesystem.
  FileManager FileMgr;
  
  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  DiagnosticClient* TextDiagClient = 0;
  
  if (!VerifyDiagnostics) {
    // Print diagnostics to stderr by default.
    TextDiagClient = new TextDiagnosticPrinter(!NoShowColumn,
                                               !NoCaretDiagnostics);
  } else {
    // When checking diagnostics, just buffer them up.
    TextDiagClient = new TextDiagnosticBuffer();
   
    if (InputFilenames.size() != 1) {
      fprintf(stderr,
              "-verify only works on single input files for now.\n");
      return 1;
    }
  }

  // Configure our handling of diagnostics.
  llvm::OwningPtr<DiagnosticClient> DiagClient(TextDiagClient);
  Diagnostic Diags(DiagClient.get());
  InitializeDiagnostics(Diags);  

  // -I- is a deprecated GCC feature, scan for it and reject it.
  for (unsigned i = 0, e = I_dirs.size(); i != e; ++i) {
    if (I_dirs[i] == "-") {
      Diags.Report(diag::err_pp_I_dash_not_supported);      
      I_dirs.erase(I_dirs.begin()+i);
      --i;
    }
  }

  // Get information about the target being compiled for.
  std::string Triple = CreateTargetTriple();
  llvm::OwningPtr<TargetInfo> Target(TargetInfo::CreateTargetInfo(Triple));
  
  if (Target == 0) {
    fprintf(stderr, "Sorry, I don't know what target this is: %s\n",
            Triple.c_str());
    fprintf(stderr, "Please use -triple or -arch.\n");
    exit(1);
  }
  
  // Are we invoking one or more source analyses?
  if (!AnalysisList.empty() && ProgAction == ParseSyntaxOnly)
    ProgAction = RunAnalysis;  
    
  llvm::OwningPtr<SourceManager> SourceMgr;
  
  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    const std::string &InFile = InputFilenames[i];
    
    if (isSerializedFile(InFile)) {
      Diags.setClient(TextDiagClient);
      ProcessSerializedFile(InFile,Diags,FileMgr);
    }
    else {            
      /// Create a SourceManager object.  This tracks and owns all the file
      /// buffers allocated to a translation unit.
      if (!SourceMgr)
        SourceMgr.reset(new SourceManager());
      else
        SourceMgr->clearIDTables();
      
      // Initialize language options, inferring file types from input filenames.
      LangOptions LangInfo;
      InitializeBaseLanguage();
      LangKind LK = GetLanguage(InFile);
      InitializeLangOptions(LangInfo, LK);
      InitializeLanguageStandard(LangInfo, LK);
      InitializeGCMode(LangInfo);
      
      // Process the -I options and set them in the HeaderInfo.
      HeaderSearch HeaderInfo(FileMgr);
      
      InitializeIncludePaths(argv[0], HeaderInfo, FileMgr, LangInfo);
      
      // Set up the preprocessor with these options.
      DriverPreprocessorFactory PPFactory(InFile, Diags, LangInfo, *Target,
                                          *SourceMgr.get(), HeaderInfo);
      
      llvm::OwningPtr<Preprocessor> PP(PPFactory.CreatePreprocessor());
            
      if (!PP)
        continue;

      // Create the HTMLDiagnosticsClient if we are using one.  Otherwise,
      // always reset to using TextDiagClient.
      llvm::OwningPtr<DiagnosticClient> TmpClient;
      
      if (!HTMLDiag.empty()) {
        TmpClient.reset(CreateHTMLDiagnosticClient(HTMLDiag, PP.get(),
                                                   &PPFactory));
        Diags.setClient(TmpClient.get());
      }
      else
        Diags.setClient(TextDiagClient);

      // Process the source file.
      ProcessInputFile(*PP, PPFactory, InFile);
      HeaderInfo.ClearFileInfo();      
      
      if (Stats)
        SourceMgr->PrintStats();
    }
  }

  if (unsigned NumDiagnostics = Diags.getNumDiagnostics())
    fprintf(stderr, "%d diagnostic%s generated.\n", NumDiagnostics,
            (NumDiagnostics == 1 ? "" : "s"));
  
  if (Stats) {
    FileMgr.PrintStats();
    fprintf(stderr, "\n");
  }
  
  return Diags.getNumErrors() != 0;
}
