//===--- clang.cpp - C-Language Front-end ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include "TextDiagnosticBuffer.h"
#include "TextDiagnosticPrinter.h"
#include "clang/AST/TranslationUnit.h"
#include "clang/Sema/ASTStreamer.h"
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
#include "llvm/ADT/scoped_ptr.h"
#include <memory>
using namespace clang;

//===----------------------------------------------------------------------===//
// Global options.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool>
Verbose("v", llvm::cl::desc("Enable verbose output"));
static llvm::cl::opt<bool>
Stats("stats", llvm::cl::desc("Print performance metrics and statistics"));

enum ProgActions {
  RewriteTest,                  // Rewriter testing stuff.
  EmitLLVM,                     // Emit a .ll file.
  SerializeAST,                 // Emit a .ast file.
  ASTPrint,                     // Parse ASTs and print them.
  ASTDump,                      // Parse ASTs and dump them.
  ASTView,                      // Parse ASTs and view them in Graphviz.
  ParseCFGDump,                 // Parse ASTS. Build CFGs. Print CFGs.
  ParseCFGView,                 // Parse ASTS. Build CFGs. View CFGs.
  AnalysisLiveVariables,        // Print results of live-variable analysis.
  WarnDeadStores,               // Run DeadStores checker on parsed ASTs.
  WarnDeadStoresCheck,          // Check diagnostics for "DeadStores".
  WarnUninitVals,               // Run UnitializedVariables checker.
  TestSerialization,            // Run experimental serialization code.
  ParsePrintCallbacks,          // Parse and print each callback.
  ParseSyntaxOnly,              // Parse and perform semantic analysis.
  ParseNoop,                    // Parse with noop callbacks.
  RunPreprocessorOnly,          // Just lex, no output.
  PrintPreprocessedInput,       // -E mode.
  DumpTokens                    // Token dump mode.
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
             clEnumValN(ASTPrint, "ast-print",
                        "Build ASTs and then pretty-print them"),
             clEnumValN(ASTDump, "ast-dump",
                        "Build ASTs and then debug dump them"),
             clEnumValN(ASTView, "ast-view",
                        "Build ASTs and view them with GraphViz."),
             clEnumValN(ParseCFGDump, "dump-cfg",
                        "Run parser, then build and print CFGs."),
             clEnumValN(ParseCFGView, "view-cfg",
                        "Run parser, then build and view CFGs with Graphviz."),
             clEnumValN(AnalysisLiveVariables, "dump-live-variables",
                        "Print results of live variable analysis."),
             clEnumValN(WarnDeadStores, "warn-dead-stores",
                        "Flag warnings of stores to dead variables."),
             clEnumValN(WarnUninitVals, "warn-uninit-values",
                        "Flag warnings of uses of unitialized variables."),
             clEnumValN(TestSerialization, "test-pickling",
                        "Run prototype serializtion code."),
             clEnumValN(EmitLLVM, "emit-llvm",
                        "Build ASTs then convert to LLVM, emit .ll file"),
             clEnumValN(SerializeAST, "serialize",
                        "Build ASTs and emit .ast file"),
             clEnumValN(RewriteTest, "rewrite-test",
                        "Playground for the code rewriter"),
             clEnumValEnd));


static llvm::cl::opt<std::string>
OutputFile("o",
 llvm::cl::value_desc("path"),
 llvm::cl::desc("Specify output file (for --serialize, this is a directory)"));
                          
static llvm::cl::opt<bool>
VerifyDiagnostics("verify",
                  llvm::cl::desc("Verify emitted diagnostics and warnings."));

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
WritableStrings("fwritable-strings",
              llvm::cl::desc("Store string literals as writable data."));

static llvm::cl::opt<bool>
LaxVectorConversions("flax-vector-conversions",
                     llvm::cl::desc("Allow implicit conversions between vectors"
                                    " with a different number of elements or "
                                    "different element types."));
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
    Options.Digraphs = 1;
    Options.C99 = 1;
    Options.HexFloats = 1;
    // FALL THROUGH.
  case lang_gnu89:
    Options.BCPLComment = 1;  // Only for C99/C++.
    // FALL THROUGH.
  case lang_c94:
  case lang_c89:
    break;
  }
  
  Options.Trigraphs = 1; // -trigraphs or -ansi
  Options.DollarIdents = 1;  // FIXME: Really a target property.
  Options.PascalStrings = PascalStrings;
  Options.WritableStrings = WritableStrings;
  Options.LaxVectorConversions = LaxVectorConversions;
}

//===----------------------------------------------------------------------===//
// Our DiagnosticClient implementation
//===----------------------------------------------------------------------===//

// FIXME: Werror should take a list of things, -Werror=foo,bar
static llvm::cl::opt<bool>
WarningsAsErrors("Werror", llvm::cl::desc("Treat all warnings as errors"));

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
   llvm::cl::desc("Warn about equality comparisons of floating point values."));

static llvm::cl::opt<bool>
WarnNoFormatNonLiteral("Wno-format-nonliteral",
   llvm::cl::desc("Do not warn about non-literal format strings."));

/// InitializeDiagnostics - Initialize the diagnostic object, based on the
/// current command line option settings.
static void InitializeDiagnostics(Diagnostic &Diags) {
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

}

//===----------------------------------------------------------------------===//
// Target Triple Processing.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
TargetTriple("triple",
  llvm::cl::desc("Specify target triple (e.g. i686-apple-darwin9)."));

static llvm::cl::list<std::string>
Archs("arch",
  llvm::cl::desc("Specify target architecture (e.g. i686)."));

namespace {
  class TripleProcessor {
    llvm::StringMap<char> TriplesProcessed;
    std::vector<std::string>& triples;
  public:
    TripleProcessor(std::vector<std::string>& t) :  triples(t) {}
    
    void addTriple(const std::string& t) {
      if (TriplesProcessed.find(t.c_str(),t.c_str()+t.size()) ==
          TriplesProcessed.end()) {
        triples.push_back(t);
        TriplesProcessed.GetOrCreateValue(t.c_str(),t.c_str()+t.size());
      }
    }
  };
}

static void CreateTargetTriples(std::vector<std::string>& triples) {
  // Initialize base triple.  If a -triple option has been specified, use
  // that triple.  Otherwise, default to the host triple.
  std::string Triple = TargetTriple;
  if (Triple.empty()) Triple = LLVM_HOSTTRIPLE;
  
  // Decompose the base triple into "arch" and suffix.
  std::string::size_type firstDash = Triple.find("-");
  
  if (firstDash == std::string::npos) {
    fprintf(stderr, 
            "Malformed target triple: \"%s\" ('-' could not be found).\n",
            Triple.c_str());
    exit(1);
  }
  
  std::string suffix(Triple, firstDash+1);
  
  if (suffix.empty()) {
    fprintf(stderr, "Malformed target triple: \"%s\" (no vendor or OS).\n",
            Triple.c_str());
    exit(1);
  }

  // Create triple cacher.
  TripleProcessor tp(triples);

  // Add the primary triple to our set of triples if we are using the
  // host-triple with no archs or using a specified target triple.
  if (!TargetTriple.getValue().empty() || Archs.empty())
    tp.addTriple(Triple);
           
  for (unsigned i = 0, e = Archs.size(); i !=e; ++i)
    tp.addTriple(Archs[i] + "-" + suffix);
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


/// InitializePreprocessor - Initialize the preprocessor getting it and the
/// environment ready to process a single file. This returns the file ID for the
/// input file. If a failure happens, it returns 0.
///
static unsigned InitializePreprocessor(Preprocessor &PP,
                                       const std::string &InFile,
                                       std::vector<char> &PredefineBuffer) {
  
  FileManager &FileMgr = PP.getFileManager();
  
  // Figure out where to get and map in the main file.
  unsigned MainFileID = 0;
  SourceManager &SourceMgr = PP.getSourceManager();
  if (InFile != "-") {
    const FileEntry *File = FileMgr.getFile(InFile);
    if (File) MainFileID = SourceMgr.createFileID(File, SourceLocation());
    if (MainFileID == 0) {
      fprintf(stderr, "Error reading '%s'!\n",InFile.c_str());
      return 0;
    }
  } else {
    llvm::MemoryBuffer *SB = llvm::MemoryBuffer::getSTDIN();
    if (SB) MainFileID = SourceMgr.createFileIDForMemBuffer(SB);
    if (MainFileID == 0) {
      fprintf(stderr, "Error reading standard input!  Empty?\n");
      return 0;
    }
  }
  
  // Add macros from the command line.
  // FIXME: Should traverse the #define/#undef lists in parallel.
  for (unsigned i = 0, e = D_macros.size(); i != e; ++i)
    DefineBuiltinMacro(PredefineBuffer, D_macros[i].c_str());
  for (unsigned i = 0, e = U_macros.size(); i != e; ++i)
    DefineBuiltinMacro(PredefineBuffer, U_macros[i].c_str(), "#undef ");
  
  // FIXME: Read any files specified by -imacros or -include.
  
  // Null terminate PredefinedBuffer and add it.
  PredefineBuffer.push_back(0);
  PP.setPredefines(&PredefineBuffer[0]);
  
  // Once we've read this, we're done.
  return MainFileID;
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
// FIXME: -include,-imacros

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
enum IncludeDirGroup {
  Quoted = 0,
  Angled,
  System,
  After
};

static std::vector<DirectoryLookup> IncludeGroup[4];

/// AddPath - Add the specified path to the specified group list.
///
static void AddPath(const std::string &Path, IncludeDirGroup Group,
                    bool isCXXAware, bool isUserSupplied,
                    bool isFramework, HeaderSearch &HS) {
  assert(!Path.empty() && "can't handle empty path here");
  FileManager &FM = HS.getFileMgr();
  
  // Compute the actual path, taking into consideration -isysroot.
  llvm::SmallString<256> MappedPath;
  
  // Handle isysroot.
  if (Group == System) {
    // FIXME: Portability.  This should be a sys::Path interface, this doesn't
    // handle things like C:\ right, nor win32 \\network\device\blah.
    if (isysroot.size() != 1 || isysroot[0] != '/') // Add isysroot if present.
      MappedPath.append(isysroot.begin(), isysroot.end());
    if (Path[0] != '/')  // If in the system group, add a /.
      MappedPath.push_back('/');
  }
  
  MappedPath.append(Path.begin(), Path.end());

  // Compute the DirectoryLookup type.
  DirectoryLookup::DirType Type;
  if (Group == Quoted || Group == Angled)
    Type = DirectoryLookup::NormalHeaderDir;
  else if (isCXXAware)
    Type = DirectoryLookup::SystemHeaderDir;
  else
    Type = DirectoryLookup::ExternCSystemHeaderDir;
  
  
  // If the directory exists, add it.
  if (const DirectoryEntry *DE = FM.getDirectory(&MappedPath[0], 
                                                 &MappedPath[0]+
                                                 MappedPath.size())) {
    IncludeGroup[Group].push_back(DirectoryLookup(DE, Type, isUserSupplied,
                                                  isFramework));
    return;
  }
  
  // Check to see if this is an apple-style headermap (which are not allowed to
  // be frameworks).
  if (!isFramework) {
    if (const FileEntry *FE = FM.getFile(&MappedPath[0], 
                                         &MappedPath[0]+MappedPath.size())) {
      if (const HeaderMap *HM = HS.CreateHeaderMap(FE)) {
        // It is a headermap, add it to the search path.
        IncludeGroup[Group].push_back(DirectoryLookup(HM, Type,isUserSupplied));
        return;
      }
    }
  }
  
  if (Verbose)
    fprintf(stderr, "ignoring nonexistent directory \"%s\"\n", Path.c_str());
}

/// RemoveDuplicates - If there are duplicate directory entries in the specified
/// search list, remove the later (dead) ones.
static void RemoveDuplicates(std::vector<DirectoryLookup> &SearchList) {
  llvm::SmallPtrSet<const DirectoryEntry *, 8> SeenDirs;
  llvm::SmallPtrSet<const DirectoryEntry *, 8> SeenFrameworkDirs;
  llvm::SmallPtrSet<const HeaderMap *, 8> SeenHeaderMaps;
  for (unsigned i = 0; i != SearchList.size(); ++i) {
    if (SearchList[i].isNormalDir()) {
      // If this isn't the first time we've seen this dir, remove it.
      if (SeenDirs.insert(SearchList[i].getDir()))
        continue;
      
      if (Verbose)
        fprintf(stderr, "ignoring duplicate directory \"%s\"\n",
                SearchList[i].getDir()->getName());
    } else if (SearchList[i].isFramework()) {
      // If this isn't the first time we've seen this framework dir, remove it.
      if (SeenFrameworkDirs.insert(SearchList[i].getFrameworkDir()))
        continue;
      
      if (Verbose)
        fprintf(stderr, "ignoring duplicate framework \"%s\"\n",
                SearchList[i].getFrameworkDir()->getName());
      
    } else {
      assert(SearchList[i].isHeaderMap() && "Not a headermap or normal dir?");
      // If this isn't the first time we've seen this headermap, remove it.
      if (SeenHeaderMaps.insert(SearchList[i].getHeaderMap()))
        continue;
      
      if (Verbose)
        fprintf(stderr, "ignoring duplicate directory \"%s\"\n",
                SearchList[i].getDir()->getName());
    }
    
    // This is reached if the current entry is a duplicate.
    SearchList.erase(SearchList.begin()+i);
    --i;
  }
}

/// InitializeIncludePaths - Process the -I options and set them in the
/// HeaderSearch object.
static void InitializeIncludePaths(HeaderSearch &Headers, FileManager &FM,
                                   const LangOptions &Lang) {
  // Handle -F... options.
  for (unsigned i = 0, e = F_dirs.size(); i != e; ++i)
    AddPath(F_dirs[i], Angled, false, true, true, Headers);
  
  // Handle -I... options.
  for (unsigned i = 0, e = I_dirs.size(); i != e; ++i)
    AddPath(I_dirs[i], Angled, false, true, false, Headers);
  
  // Handle -idirafter... options.
  for (unsigned i = 0, e = idirafter_dirs.size(); i != e; ++i)
    AddPath(idirafter_dirs[i], After, false, true, false, Headers);
  
  // Handle -iquote... options.
  for (unsigned i = 0, e = iquote_dirs.size(); i != e; ++i)
    AddPath(iquote_dirs[i], Quoted, false, true, false, Headers);
  
  // Handle -isystem... options.
  for (unsigned i = 0, e = isystem_dirs.size(); i != e; ++i)
    AddPath(isystem_dirs[i], System, false, true, false, Headers);

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
        AddPath(Prefix+iwithprefix_vals[iwithprefix_idx], 
                System, false, false, false, Headers);
        ++iwithprefix_idx;
        iwithprefix_done = iwithprefix_idx == iwithprefix_vals.size();
      } else {
        AddPath(Prefix+iwithprefixbefore_vals[iwithprefixbefore_idx], 
                Angled, false, false, false, Headers);
        ++iwithprefixbefore_idx;
        iwithprefixbefore_done = 
          iwithprefixbefore_idx == iwithprefixbefore_vals.size();
      }
    }
  }
  
  // FIXME: Add contents of the CPATH, C_INCLUDE_PATH, CPLUS_INCLUDE_PATH,
  // OBJC_INCLUDE_PATH, OBJCPLUS_INCLUDE_PATH environment variables.
  
  // FIXME: temporary hack: hard-coded paths.
  // FIXME: get these from the target?
  if (!nostdinc) {
    if (Lang.CPlusPlus) {
      AddPath("/usr/include/c++/4.0.0", System, true, false, false, Headers);
      AddPath("/usr/include/c++/4.0.0/i686-apple-darwin8", System, true, false,
              false, Headers);
      AddPath("/usr/include/c++/4.0.0/backward", System, true, false, false,
              Headers);
    }
    
    AddPath("/usr/local/include", System, false, false, false, Headers);
    // leopard
    AddPath("/usr/lib/gcc/i686-apple-darwin9/4.0.1/include", System, 
            false, false, false, Headers);
    AddPath("/usr/lib/gcc/powerpc-apple-darwin9/4.0.1/include", 
            System, false, false, false, Headers);
    AddPath("/usr/lib/gcc/powerpc-apple-darwin9/"
            "4.0.1/../../../../powerpc-apple-darwin0/include", 
            System, false, false, false, Headers);

    // tiger
    AddPath("/usr/lib/gcc/i686-apple-darwin8/4.0.1/include", System, 
            false, false, false, Headers);
    AddPath("/usr/lib/gcc/powerpc-apple-darwin8/4.0.1/include", 
            System, false, false, false, Headers);
    AddPath("/usr/lib/gcc/powerpc-apple-darwin8/"
            "4.0.1/../../../../powerpc-apple-darwin8/include", 
            System, false, false, false, Headers);

    AddPath("/usr/include", System, false, false, false, Headers);
    AddPath("/System/Library/Frameworks", System, true, false, true, Headers);
    AddPath("/Library/Frameworks", System, true, false, true, Headers);
  }

  // Now that we have collected all of the include paths, merge them all
  // together and tell the preprocessor about them.
  
  // Concatenate ANGLE+SYSTEM+AFTER chains together into SearchList.
  std::vector<DirectoryLookup> SearchList;
  SearchList = IncludeGroup[Angled];
  SearchList.insert(SearchList.end(), IncludeGroup[System].begin(),
                    IncludeGroup[System].end());
  SearchList.insert(SearchList.end(), IncludeGroup[After].begin(),
                    IncludeGroup[After].end());
  RemoveDuplicates(SearchList);
  RemoveDuplicates(IncludeGroup[Quoted]);
  
  // Prepend QUOTED list on the search list.
  SearchList.insert(SearchList.begin(), IncludeGroup[Quoted].begin(), 
                    IncludeGroup[Quoted].end());
  

  bool DontSearchCurDir = false;  // TODO: set to true if -I- is set?
  Headers.SetSearchPaths(SearchList, IncludeGroup[Quoted].size(),
                         DontSearchCurDir);

  // If verbose, print the list of directories that will be searched.
  if (Verbose) {
    fprintf(stderr, "#include \"...\" search starts here:\n");
    unsigned QuotedIdx = IncludeGroup[Quoted].size();
    for (unsigned i = 0, e = SearchList.size(); i != e; ++i) {
      if (i == QuotedIdx)
        fprintf(stderr, "#include <...> search starts here:\n");
      const char *Name = SearchList[i].getName();
      const char *Suffix;
      if (SearchList[i].isNormalDir())
        Suffix = "";
      else if (SearchList[i].isFramework())
        Suffix = " (framework directory)";
      else {
        assert(SearchList[i].isHeaderMap() && "Unknown DirectoryLookup");
        Suffix = " (headermap)";
      }
      fprintf(stderr, " %s%s\n", Name, Suffix);
    }
    fprintf(stderr, "End of search list.\n");
  }
}


//===----------------------------------------------------------------------===//
// Basic Parser driver
//===----------------------------------------------------------------------===//

static void ParseFile(Preprocessor &PP, MinimalAction *PA, unsigned MainFileID){
  Parser P(PP, *PA);
  PP.EnterMainSourceFile(MainFileID);
  
  // Parsing the specified input file.
  P.ParseTranslationUnit();
  delete PA;
}

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

/// CreateASTConsumer - Create the ASTConsumer for the corresponding program
///  action.  These consumers can operate on both ASTs that are freshly
///  parsed from source files as well as those deserialized from Bitcode.
static ASTConsumer* CreateASTConsumer(const std::string& InFile,
                                      Diagnostic& Diag, FileManager& FileMgr, 
                                      const LangOptions& LangOpts) {
  switch (ProgAction) {
    default:
      return NULL;
      
    case ASTPrint:
      return CreateASTPrinter();
      
    case ASTDump:
      return CreateASTDumper();
      
    case ASTView:
      return CreateASTViewer();      
      
    case ParseCFGDump:
    case ParseCFGView:
      return CreateCFGDumper(ProgAction == ParseCFGView);
      
    case AnalysisLiveVariables:
      return CreateLiveVarAnalyzer();
      
    case WarnDeadStores:    
      return CreateDeadStoreChecker(Diag);
      
    case WarnUninitVals:
      return CreateUnitValsChecker(Diag);
      
    case TestSerialization:
      return CreateSerializationTest(Diag, FileMgr, LangOpts);
      
    case EmitLLVM:
      return CreateLLVMEmitter(Diag, LangOpts);
      
    case SerializeAST:
      // FIXME: Allow user to tailor where the file is written.
      return CreateASTSerializer(InFile, Diag, LangOpts);
      
    case RewriteTest:
      return CreateCodeRewriterTest(Diag);
  }
}

/// ProcessInputFile - Process a single input file with the specified state.
///
static void ProcessInputFile(Preprocessor &PP, unsigned MainFileID,
                             const std::string &InFile,
                             TextDiagnostics &OurDiagnosticClient) {

  ASTConsumer* Consumer = NULL;
  bool ClearSourceMgr = false;
  
  switch (ProgAction) {
  default:
    Consumer = CreateASTConsumer(InFile,
                                 PP.getDiagnostics(),
                                 PP.getFileManager(),
                                 PP.getLangOptions());
    
    if (!Consumer) {      
      fprintf(stderr, "Unexpected program action!\n");
      return;
    }

    break;
      
  case DumpTokens: {                 // Token dump mode.
    Token Tok;
    // Start parsing the specified input file.
    PP.EnterMainSourceFile(MainFileID);
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
    PP.EnterMainSourceFile(MainFileID);
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::eof));
    ClearSourceMgr = true;
    break;
  }
    
  case PrintPreprocessedInput:       // -E mode.
    DoPrintPreprocessedInput(MainFileID, PP);
    ClearSourceMgr = true;
    break;
    
  case ParseNoop:                    // -parse-noop
    ParseFile(PP, new MinimalAction(PP.getIdentifierTable()), MainFileID);
    ClearSourceMgr = true;
    break;
    
  case ParsePrintCallbacks:
    ParseFile(PP, CreatePrintParserActionsAction(PP.getIdentifierTable()), 
              MainFileID);
    ClearSourceMgr = true;
    break;
      
  case ParseSyntaxOnly:              // -fsyntax-only
    Consumer = new ASTConsumer();
    break;
  }
  
  if (Consumer) {
    if (VerifyDiagnostics)
      exit(CheckASTConsumer(PP, MainFileID, Consumer));
    
    // This deletes Consumer.
    ParseAST(PP, MainFileID, Consumer, Stats);
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
  
  llvm::scoped_ptr<TranslationUnit> TU(ReadASTBitcodeFile(Filename,FileMgr));
  
  if (!TU) {
    fprintf(stderr, "error: file '%s' could not be deserialized\n", 
            InFile.c_str());
    exit (1);
  }
  
  // Observe that we use the source file name stored in the deserialized
  // translation unit, rather than InFile.
  llvm::scoped_ptr<ASTConsumer>
    Consumer(CreateASTConsumer(InFile, Diag, FileMgr, TU->getLangOpts()));
  
  if (!Consumer) {      
    fprintf(stderr, "Unsupported program action with serialized ASTs!\n");
    exit (1);
  }
  
  // FIXME: only work on consumers that do not require MainFileID.
  Consumer->Initialize(*TU->getContext(), 0);
  
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
  llvm::cl::ParseCommandLineOptions(argc, argv, " llvm cfe\n");
  llvm::sys::PrintStackTraceOnErrorSignal();
  
  // If no input was specified, read from stdin.
  if (InputFilenames.empty())
    InputFilenames.push_back("-");
    
  // Create a file manager object to provide access to and cache the filesystem.
  FileManager FileMgr;
  
  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  std::auto_ptr<TextDiagnostics> DiagClient;
  if (!VerifyDiagnostics) {
    // Print diagnostics to stderr by default.
    DiagClient.reset(new TextDiagnosticPrinter());
  } else {
    // When checking diagnostics, just buffer them up.
    DiagClient.reset(new TextDiagnosticBuffer());
   
    if (InputFilenames.size() != 1) {
      fprintf(stderr,
              "-verify only works on single input files for now.\n");
      return 1;
    }
  }
  
  // Configure our handling of diagnostics.
  Diagnostic Diags(*DiagClient);
  InitializeDiagnostics(Diags);  

  // -I- is a deprecated GCC feature, scan for it and reject it.
  for (unsigned i = 0, e = I_dirs.size(); i != e; ++i) {
    if (I_dirs[i] == "-") {
      Diags.Report(diag::err_pp_I_dash_not_supported);      
      I_dirs.erase(I_dirs.begin()+i);
      --i;
    }
  }
  
  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    const std::string &InFile = InputFilenames[i];
    
    if (isSerializedFile(InFile))
      ProcessSerializedFile(InFile,Diags,FileMgr);
    else {            
      /// Create a SourceManager object.  This tracks and owns all the file
      /// buffers allocated to a translation unit.
      SourceManager SourceMgr;
      
      // Initialize language options, inferring file types from input filenames.
      LangOptions LangInfo;
      InitializeBaseLanguage();
      LangKind LK = GetLanguage(InFile);
      InitializeLangOptions(LangInfo, LK);
      InitializeLanguageStandard(LangInfo, LK);
      
      // Process the -I options and set them in the HeaderInfo.
      HeaderSearch HeaderInfo(FileMgr);
      DiagClient->setHeaderSearch(HeaderInfo);
      InitializeIncludePaths(HeaderInfo, FileMgr, LangInfo);
      
      // Get information about the targets being compiled for.  Note that this
      // pointer and the TargetInfoImpl objects are never deleted by this toy
      // driver.
      TargetInfo *Target;
      
      // Create triples, and create the TargetInfo.
      std::vector<std::string> triples;
      CreateTargetTriples(triples);
      Target = TargetInfo::CreateTargetInfo(&triples[0],
                                            &triples[0]+triples.size(),
                                            &Diags);
        
      if (Target == 0) {
        fprintf(stderr, "Sorry, I don't know what target this is: %s\n",
                triples[0].c_str());
        fprintf(stderr, "Please use -triple or -arch.\n");
        exit(1);
      }
      
      // Set up the preprocessor with these options.
      Preprocessor PP(Diags, LangInfo, *Target, SourceMgr, HeaderInfo);
      
      std::vector<char> PredefineBuffer;
      unsigned MainFileID = InitializePreprocessor(PP, InFile, PredefineBuffer);
      
      if (!MainFileID)
        continue;
      
      SourceMgr.setMainFileID(MainFileID);

      ProcessInputFile(PP, MainFileID, InFile, *DiagClient);
      
      HeaderInfo.ClearFileInfo();
      
      if (Stats)
        SourceMgr.PrintStats();
    }
  }
  
  unsigned NumDiagnostics = Diags.getNumDiagnostics();
  
  if (NumDiagnostics)
    fprintf(stderr, "%d diagnostic%s generated.\n", NumDiagnostics,
            (NumDiagnostics == 1 ? "" : "s"));
  
  if (Stats) {
    FileMgr.PrintStats();
    fprintf(stderr, "\n");
  }
  
  return Diags.getNumErrors() != 0;
}
