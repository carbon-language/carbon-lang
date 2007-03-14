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
#include "clang/AST/AST.h"
#include "clang/Parse/Parser.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceBuffer.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/System/MappedFile.h"
#include "llvm/System/Signals.h"
#include <iostream>
using namespace llvm;
using namespace clang;

static unsigned NumDiagnostics = 0;
static unsigned NumErrors = 0;

//===----------------------------------------------------------------------===//
// Global options.
//===----------------------------------------------------------------------===//

static cl::opt<bool>
Verbose("v", cl::desc("Enable verbose output"));
static cl::opt<bool>
Stats("stats", cl::desc("Print performance metrics and statistics"));

enum ProgActions {
  ParseASTPrint,                // Parse ASTs and print them.
  ParseAST,                     // Parse ASTs.
  ParsePrintCallbacks,          // Parse and print each callback.
  ParseSyntaxOnly,              // Parse and perform semantic analysis.
  ParseNoop,                    // Parse with noop callbacks.
  RunPreprocessorOnly,          // Just lex, no output.
  PrintPreprocessedInput,       // -E mode.
  DumpTokens                    // Token dump mode.
};

static cl::opt<ProgActions> 
ProgAction(cl::desc("Choose output type:"), cl::ZeroOrMore,
           cl::init(ParseSyntaxOnly),
           cl::values(
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
             clEnumValN(ParseAST, "parse-ast",
                        "Run parser and build ASTs"),
             clEnumValN(ParseASTPrint, "parse-ast-print",
                        "Run parser, build ASTs, then print ASTs"),
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
static cl::opt<LangKind>
BaseLang("x", cl::desc("Base language to compile"),
         cl::init(langkind_unspecified),
         cl::values(clEnumValN(langkind_c,     "c",            "C"),
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

static cl::opt<bool>
LangObjC("ObjC", cl::desc("Set base language to Objective-C"),
         cl::Hidden);
static cl::opt<bool>
LangObjCXX("ObjC++", cl::desc("Set base language to Objective-C++"),
           cl::Hidden);

/// InitializeBaseLanguage - Handle the -x foo options or infer a base language
/// from the input filename.
static void InitializeBaseLanguage(LangOptions &Options,
                                   const std::string &Filename) {
  if (BaseLang == langkind_unspecified) {
    std::string::size_type DotPos = Filename.rfind('.');
    if (LangObjC) {
      BaseLang = langkind_objc;
    } else if (LangObjCXX) {
      BaseLang = langkind_objcxx;
    } else if (DotPos == std::string::npos) {
      BaseLang = langkind_c;  // Default to C if no extension.
    } else {
      std::string Ext = std::string(Filename.begin()+DotPos+1, Filename.end());
      // C header: .h
      // C++ header: .hh or .H;
      // assembler no preprocessing: .s
      // assembler: .S
      if (Ext == "c")
        BaseLang = langkind_c;
      else if (Ext == "i")
        BaseLang = langkind_c_cpp;
      else if (Ext == "ii")
        BaseLang = langkind_cxx_cpp;
      else if (Ext == "m")
        BaseLang = langkind_objc;
      else if (Ext == "mi")
        BaseLang = langkind_objc_cpp;
      else if (Ext == "mm" || Ext == "M")
        BaseLang = langkind_objcxx;
      else if (Ext == "mii")
        BaseLang = langkind_objcxx_cpp;
      else if (Ext == "C" || Ext == "cc" || Ext == "cpp" || Ext == "CPP" ||
               Ext == "c++" || Ext == "cp" || Ext == "cxx")
        BaseLang = langkind_cxx;
      else
        BaseLang = langkind_c;
    }
  }
  
  // FIXME: implement -fpreprocessed mode.
  bool NoPreprocess = false;
  
  switch (BaseLang) {
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
  lang_cxx98, lang_gnucxx98
};

static cl::opt<LangStds>
LangStd("std", cl::desc("Language standard to compile for"),
        cl::init(lang_unspecified),
        cl::values(clEnumValN(lang_c89,      "c89",            "ISO C 1990"),
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
                   clEnumValEnd));

static cl::opt<bool>
NoOperatorNames("fno-operator-names",
                cl::desc("Do not treat C++ operator name keywords as "
                         "synonyms for operators"));

// FIXME: add:
//   -ansi
//   -trigraphs
//   -fdollars-in-identifiers
static void InitializeLanguageStandard(LangOptions &Options) {
  if (LangStd == lang_unspecified) {
    // Based on the base language, pick one.
    switch (BaseLang) {
    default: assert(0 && "Unknown base language");
    case langkind_c:
    case langkind_c_cpp:
    case langkind_objc:
    case langkind_objc_cpp:
      LangStd = lang_gnu89;
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
  case lang_gnucxx98:
  case lang_cxx98:
    Options.CPlusPlus = 1;
    Options.CXXOperatorNames = !NoOperatorNames;
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
}

//===----------------------------------------------------------------------===//
// Our DiagnosticClient implementation
//===----------------------------------------------------------------------===//

// FIXME: Werror should take a list of things, -Werror=foo,bar
static cl::opt<bool>
WarningsAsErrors("Werror", cl::desc("Treat all warnings as errors"));

static cl::opt<bool>
WarnOnExtensions("pedantic",
                 cl::desc("Issue a warning on uses of GCC extensions"));

static cl::opt<bool>
ErrorOnExtensions("pedantic-errors",
                  cl::desc("Issue an error on uses of GCC extensions"));

static cl::opt<bool>
WarnUnusedMacros("Wunused_macros",
               cl::desc("Warn for unused macros in the main translation unit"));


/// InitializeDiagnostics - Initialize the diagnostic object, based on the
/// current command line option settings.
static void InitializeDiagnostics(Diagnostic &Diags) {
  Diags.setWarningsAsErrors(WarningsAsErrors);
  Diags.setWarnOnExtensions(WarnOnExtensions);
  Diags.setErrorOnExtensions(ErrorOnExtensions);

  // Silence the "macro is not used" warning unless requested.
  if (!WarnUnusedMacros)
    Diags.setDiagnosticMapping(diag::pp_macro_not_used, diag::MAP_IGNORE);
}

static cl::opt<bool>
NoShowColumn("fno-show-column",
             cl::desc("Do not include column number on diagnostics"));
static cl::opt<bool>
NoCaretDiagnostics("fno-caret-diagnostics",
                   cl::desc("Do not include source line and caret with"
                            " diagnostics"));

/// DiagnosticPrinterSTDERR - This is a concrete diagnostic client, which prints
/// the diagnostics to standard error.
class DiagnosticPrinterSTDERR : public DiagnosticClient {
  SourceManager &SourceMgr;
  SourceLocation LastWarningLoc;
public:
  DiagnosticPrinterSTDERR(SourceManager &sourceMgr)
    : SourceMgr(sourceMgr) {}
  
  void PrintIncludeStack(SourceLocation Pos);

  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                SourceLocation Pos,
                                diag::kind ID, const std::string &Msg);
};

void DiagnosticPrinterSTDERR::
PrintIncludeStack(SourceLocation Pos) {
  unsigned FileID = Pos.getFileID();
  if (FileID == 0) return;
  
  // Print out the other include frames first.
  PrintIncludeStack(SourceMgr.getIncludeLoc(FileID));
  
  unsigned LineNo = SourceMgr.getLineNumber(Pos);
  
  const SourceBuffer *Buffer = SourceMgr.getBuffer(FileID);
  std::cerr << "In file included from " << Buffer->getBufferIdentifier()
            << ":" << LineNo << ":\n";
}


void DiagnosticPrinterSTDERR::HandleDiagnostic(Diagnostic::Level Level, 
                                               SourceLocation Pos,
                                               diag::kind ID, 
                                               const std::string &Extra) {
  ++NumDiagnostics;
  unsigned LineNo = 0, FilePos = 0, FileID = 0, ColNo = 0;
  unsigned LineStart = 0, LineEnd = 0;
  const SourceBuffer *Buffer = 0;
  
  if (Pos.isValid()) {
    LineNo = SourceMgr.getLineNumber(Pos);
    FileID  = SourceMgr.getLogicalLoc(Pos).getFileID();
    
    // First, if this diagnostic is not in the main file, print out the
    // "included from" lines.
    if (LastWarningLoc != SourceMgr.getIncludeLoc(FileID)) {
      LastWarningLoc = SourceMgr.getIncludeLoc(FileID);
      PrintIncludeStack(LastWarningLoc);
    }
  
    // Compute the column number.  Rewind from the current position to the start
    // of the line.
    ColNo = SourceMgr.getColumnNumber(Pos);
    FilePos = SourceMgr.getSourceFilePos(Pos);
    LineStart = FilePos-ColNo+1;  // Column # is 1-based
  
    // Compute the line end.  Scan forward from the error position to the end of
    // the line.
    Buffer = SourceMgr.getBuffer(FileID);
    const char *Buf = Buffer->getBufferStart();
    const char *BufEnd = Buffer->getBufferEnd();
    LineEnd = FilePos;
    while (Buf+LineEnd != BufEnd && 
           Buf[LineEnd] != '\n' && Buf[LineEnd] != '\r')
      ++LineEnd;
  
    std::cerr << Buffer->getBufferIdentifier() 
              << ":" << LineNo << ":";
    if (ColNo && !NoShowColumn) 
      std::cerr << ColNo << ":";
    std::cerr << " ";
  }
  
  switch (Level) {
  default: assert(0 && "Unknown diagnostic type!");
  case Diagnostic::Note:                 std::cerr << "note: "; break;
  case Diagnostic::Warning:              std::cerr << "warning: "; break;
  case Diagnostic::Error:   ++NumErrors; std::cerr << "error: "; break;
  case Diagnostic::Fatal:   ++NumErrors; std::cerr << "fatal error: "; break;
  case Diagnostic::Sorry:   ++NumErrors; std::cerr << "sorry, unimplemented: ";
    break;
  }
  
  std::string Msg = Diagnostic::getDescription(ID);
  
  // Replace all instances of %s in Msg with 'Extra'.
  if (Msg.size() > 1) {
    for (unsigned i = 0; i < Msg.size()-1; ++i) {
      if (Msg[i] == '%' && Msg[i+1] == 's') {
        Msg = std::string(Msg.begin(), Msg.begin()+i) +
              Extra +
              std::string(Msg.begin()+i+2, Msg.end());
      }
    }
  }
  std::cerr << Msg << "\n";
  
  if (!NoCaretDiagnostics && Pos.isValid()) {
    // Print out a line of the source file.
    const char *Buf = Buffer->getBufferStart();
    std::cerr << std::string(Buf+LineStart, Buf+LineEnd) << "\n";
    
    // If the source line contained any tab characters between the start of the
    // line and the diagnostic, replace the space we inserted with a tab, so
    // that the carat will be indented exactly like the source line.
    std::string Indent(ColNo-1, ' ');
    for (unsigned i = LineStart; i != FilePos; ++i)
      if (Buf[i] == '\t')
        Indent[i-LineStart] = '\t';
    
    // Print out the caret itself.
    std::cerr << Indent << "^\n";
  }
}


//===----------------------------------------------------------------------===//
// Preprocessor Initialization
//===----------------------------------------------------------------------===//

// FIXME: Preprocessor builtins to support.
//   -A...    - Play with #assertions
//   -undef   - Undefine all predefined macros

static cl::list<std::string>
D_macros("D", cl::value_desc("macro"), cl::Prefix,
       cl::desc("Predefine the specified macro"));
static cl::list<std::string>
U_macros("U", cl::value_desc("macro"), cl::Prefix,
         cl::desc("Undefine the specified macro"));

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

static void InitializePredefinedMacros(Preprocessor &PP, 
                                       std::vector<char> &Buf) {
  // FIXME: Implement magic like cpp_init_builtins for things like __STDC__
  // and __DATE__ etc.
#if 0
  /* __STDC__ has the value 1 under normal circumstances.
  However, if (a) we are in a system header, (b) the option
  stdc_0_in_system_headers is true (set by target config), and
  (c) we are not in strictly conforming mode, then it has the
  value 0.  (b) and (c) are already checked in cpp_init_builtins.  */
  //case BT_STDC:
    if (cpp_in_system_header (pfile))
      number = 0;
    else
      number = 1;
    break;
#endif    
  // These should all be defined in the preprocessor according to the
  // current language configuration.
  DefineBuiltinMacro(Buf, "__STDC__=1");
  //DefineBuiltinMacro(Buf, "__ASSEMBLER__=1");
  if (PP.getLangOptions().C99)
    DefineBuiltinMacro(Buf, "__STDC_VERSION__=199901L");
  else
    DefineBuiltinMacro(Buf, "__STDC_VERSION__=199409L");
  
  DefineBuiltinMacro(Buf, "__STDC_HOSTED__=1");
  if (PP.getLangOptions().ObjC1)
    DefineBuiltinMacro(Buf, "__OBJC__=1");
  if (PP.getLangOptions().ObjC2)
    DefineBuiltinMacro(Buf, "__OBJC2__=1");

  // Get the target #defines.
  PP.getTargetInfo().getTargetDefines(Buf);
  
  // Compiler set macros.
  DefineBuiltinMacro(Buf, "__APPLE_CC__=5250");
  DefineBuiltinMacro(Buf, "__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__=1030");
  DefineBuiltinMacro(Buf, "__GNUC_MINOR__=0");
  DefineBuiltinMacro(Buf, "__GNUC_PATCHLEVEL__=1");
  DefineBuiltinMacro(Buf, "__GNUC__=4");
  DefineBuiltinMacro(Buf, "__GXX_ABI_VERSION=1002");
  DefineBuiltinMacro(Buf, "__VERSION__=\"4.0.1 (Apple Computer, Inc. "
                     "build 5250)\"");

  // Build configuration options.
  DefineBuiltinMacro(Buf, "__DYNAMIC__=1");
  DefineBuiltinMacro(Buf, "__FINITE_MATH_ONLY__=0");
  DefineBuiltinMacro(Buf, "__NO_INLINE__=1");
  DefineBuiltinMacro(Buf, "__PIC__=1");

  
  if (PP.getLangOptions().CPlusPlus) {
    DefineBuiltinMacro(Buf, "__DEPRECATED=1");
    DefineBuiltinMacro(Buf, "__EXCEPTIONS=1");
    DefineBuiltinMacro(Buf, "__GNUG__=4");
    DefineBuiltinMacro(Buf, "__GXX_WEAK__=1");
    DefineBuiltinMacro(Buf, "__cplusplus=1");
    DefineBuiltinMacro(Buf, "__private_extern__=extern");
  }
  
  // FIXME: Should emit a #line directive here.

  // Add macros from the command line.
  // FIXME: Should traverse the #define/#undef lists in parallel.
  for (unsigned i = 0, e = D_macros.size(); i != e; ++i)
    DefineBuiltinMacro(Buf, D_macros[i].c_str());
  for (unsigned i = 0, e = U_macros.size(); i != e; ++i)
    DefineBuiltinMacro(Buf, U_macros[i].c_str(), "#undef ");
}

//===----------------------------------------------------------------------===//
// Preprocessor include path information.
//===----------------------------------------------------------------------===//

// This tool exports a large number of command line options to control how the
// preprocessor searches for header files.  At root, however, the Preprocessor
// object takes a very simple interface: a list of directories to search for
// 
// FIXME: -nostdinc,-nostdinc++
// FIXME: -isysroot,-imultilib
//
// FIXME: -include,-imacros

static cl::opt<bool>
nostdinc("nostdinc", cl::desc("Disable standard #include directories"));

// Various command line options.  These four add directories to each chain.
static cl::list<std::string>
F_dirs("F", cl::value_desc("directory"), cl::Prefix,
       cl::desc("Add directory to framework include search path"));
static cl::list<std::string>
I_dirs("I", cl::value_desc("directory"), cl::Prefix,
       cl::desc("Add directory to include search path"));
static cl::list<std::string>
idirafter_dirs("idirafter", cl::value_desc("directory"), cl::Prefix,
               cl::desc("Add directory to AFTER include search path"));
static cl::list<std::string>
iquote_dirs("iquote", cl::value_desc("directory"), cl::Prefix,
               cl::desc("Add directory to QUOTE include search path"));
static cl::list<std::string>
isystem_dirs("isystem", cl::value_desc("directory"), cl::Prefix,
            cl::desc("Add directory to SYSTEM include search path"));

// These handle -iprefix/-iwithprefix/-iwithprefixbefore.
static cl::list<std::string>
iprefix_vals("iprefix", cl::value_desc("prefix"), cl::Prefix,
             cl::desc("Set the -iwithprefix/-iwithprefixbefore prefix"));
static cl::list<std::string>
iwithprefix_vals("iwithprefix", cl::value_desc("dir"), cl::Prefix,
          cl::desc("Set directory to SYSTEM include search path with prefix"));
static cl::list<std::string>
iwithprefixbefore_vals("iwithprefixbefore", cl::value_desc("dir"), cl::Prefix,
                 cl::desc("Set directory to include search path with prefix"));

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
                    bool isFramework, FileManager &FM) {
  const DirectoryEntry *DE = FM.getDirectory(Path);
  if (DE == 0) {
    if (Verbose)
      std::cerr << "ignoring nonexistent directory \"" << Path << "\"\n";
    return;
  }
  
  DirectoryLookup::DirType Type;
  if (Group == Quoted || Group == Angled)
    Type = DirectoryLookup::NormalHeaderDir;
  else if (isCXXAware)
    Type = DirectoryLookup::SystemHeaderDir;
  else
    Type = DirectoryLookup::ExternCSystemHeaderDir;
  
  IncludeGroup[Group].push_back(DirectoryLookup(DE, Type, isUserSupplied,
                                                isFramework));
}

/// RemoveDuplicates - If there are duplicate directory entries in the specified
/// search list, remove the later (dead) ones.
static void RemoveDuplicates(std::vector<DirectoryLookup> &SearchList) {
  std::set<const DirectoryEntry *> SeenDirs;
  for (unsigned i = 0; i != SearchList.size(); ++i) {
    // If this isn't the first time we've seen this dir, remove it.
    if (!SeenDirs.insert(SearchList[i].getDir()).second) {
      if (Verbose)
        std::cerr << "ignoring duplicate directory \""
                  << SearchList[i].getDir()->getName() << "\"\n";
      SearchList.erase(SearchList.begin()+i);
      --i;
    }
  }
}

/// InitializeIncludePaths - Process the -I options and set them in the
/// HeaderSearch object.
static void InitializeIncludePaths(HeaderSearch &Headers, FileManager &FM,
                                   Diagnostic &Diags, const LangOptions &Lang) {
  // Handle -F... options.
  for (unsigned i = 0, e = F_dirs.size(); i != e; ++i)
    AddPath(F_dirs[i], Angled, false, true, true, FM);
  
  // Handle -I... options.
  for (unsigned i = 0, e = I_dirs.size(); i != e; ++i) {
    if (I_dirs[i] == "-") {
      // -I- is a deprecated GCC feature.
      Diags.Report(SourceLocation(), diag::err_pp_I_dash_not_supported);
    } else {
      AddPath(I_dirs[i], Angled, false, true, false, FM);
    }
  }
  
  // Handle -idirafter... options.
  for (unsigned i = 0, e = idirafter_dirs.size(); i != e; ++i)
    AddPath(idirafter_dirs[i], After, false, true, false, FM);
  
  // Handle -iquote... options.
  for (unsigned i = 0, e = iquote_dirs.size(); i != e; ++i)
    AddPath(iquote_dirs[i], Quoted, false, true, false, FM);
  
  // Handle -isystem... options.
  for (unsigned i = 0, e = isystem_dirs.size(); i != e; ++i)
    AddPath(isystem_dirs[i], System, false, true, false, FM);

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
                System, false, false, false, FM);
        ++iwithprefix_idx;
        iwithprefix_done = iwithprefix_idx == iwithprefix_vals.size();
      } else {
        AddPath(Prefix+iwithprefixbefore_vals[iwithprefixbefore_idx], 
                Angled, false, false, false, FM);
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
      AddPath("/usr/include/c++/4.0.0", System, true, false, false, FM);
      AddPath("/usr/include/c++/4.0.0/i686-apple-darwin8", System, true, false,
              false, FM);
      AddPath("/usr/include/c++/4.0.0/backward", System, true, false, false,FM);
    }
    
    AddPath("/usr/local/include", System, false, false, false, FM);
    AddPath("/usr/lib/gcc/i686-apple-darwin9/4.0.1/include", System, 
            false, false, false, FM);
    AddPath("/usr/lib/gcc/powerpc-apple-darwin9/4.0.1/include", 
            System, false, false, false, FM);
    AddPath("/usr/lib/gcc/powerpc-apple-darwin9/"
            "4.0.1/../../../../powerpc-apple-darwin8/include", 
            System, false, false, false, FM);
    AddPath("/usr/include", System, false, false, false, FM);
    AddPath("/System/Library/Frameworks", System, true, false, true, FM);
    AddPath("/Library/Frameworks", System, true, false, true, FM);
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
    std::cerr << "#include \"...\" search starts here:\n";
    unsigned QuotedIdx = IncludeGroup[Quoted].size();
    for (unsigned i = 0, e = SearchList.size(); i != e; ++i) {
      if (i == QuotedIdx)
        std::cerr << "#include <...> search starts here:\n";
      std::cerr << " " << SearchList[i].getDir()->getName() << "\n";
    }
  }
}


// Read any files specified by -imacros or -include.
static void ReadPrologFiles(Preprocessor &PP, std::vector<char> &Buf) {
  // FIXME: IMPLEMENT
}

//===----------------------------------------------------------------------===//
// Basic Parser driver
//===----------------------------------------------------------------------===//

static void ParseFile(Preprocessor &PP, MinimalAction *PA, unsigned MainFileID){
  Parser P(PP, *PA);
  PP.EnterSourceFile(MainFileID, 0, true);
  
  // Parsing the specified input file.
  P.ParseTranslationUnit();
  delete PA;
}

//===----------------------------------------------------------------------===//
// ASTStreamer drivers
//===----------------------------------------------------------------------===//

static void BuildASTs(Preprocessor &PP, unsigned MainFileID) {
  ASTContext Context(PP.getTargetInfo(), PP.getIdentifierTable());
  ASTStreamerTy *Streamer = ASTStreamer_Init(PP, Context, MainFileID);
  while (ASTStreamer_ReadTopLevelDecl(Streamer))
    /* keep reading */;

  if (Stats) {
    std::cerr << "\nSTATISTICS:\n";
    ASTStreamer_PrintStats(Streamer);
    Context.PrintStats();
  }
  
  ASTStreamer_Terminate(Streamer);
}

static void PrintFunctionDecl(FunctionDecl *FD) {
  bool HasBody = FD->getBody();
  
  std::string Proto = FD->getName();
  FunctionType *AFT = cast<FunctionType>(FD->getType());
  if (FunctionTypeProto *FT = dyn_cast<FunctionTypeProto>(AFT)) {
    Proto += "(";
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
      if (i) Proto += ", ";
      std::string ParamStr;
      if (HasBody) ParamStr = FD->getParamDecl(i)->getName();
      
      FT->getArgType(i).getAsString(ParamStr);
      Proto += ParamStr;
    }
    
    if (FT->isVariadic()) {
      if (FD->getNumParams()) Proto += ", ";
      Proto += "...";
    }
    Proto += ")";
  } else {
    assert(isa<FunctionTypeNoProto>(AFT));
    Proto += "()";
  }
  AFT->getResultType().getAsString(Proto);

  std::cerr << "\n" << Proto;
  
  if (FD->getBody()) {
    std::cerr << " ";
    FD->getBody()->dump();
    std::cerr << "\n";
  } else {
    std::cerr << ";\n";
  }
}

static void PrintTypeDefDecl(TypedefDecl *TD) {
  std::string S = TD->getName();
  TD->getUnderlyingType().getAsString(S);
  std::cerr << "typedef " << S << ";\n";
}

static void PrintASTs(Preprocessor &PP, unsigned MainFileID) {
  ASTContext Context(PP.getTargetInfo(), PP.getIdentifierTable());
  ASTStreamerTy *Streamer = ASTStreamer_Init(PP, Context, MainFileID);
  
  while (Decl *D = ASTStreamer_ReadTopLevelDecl(Streamer)) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      PrintFunctionDecl(FD);
    } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
      PrintTypeDefDecl(TD);
    } else {
      std::cerr << "Read top-level variable decl: '" << D->getName() << "'\n";
    }
  }
  
  if (Stats) {
    std::cerr << "\nSTATISTICS:\n";
    ASTStreamer_PrintStats(Streamer);
    Context.PrintStats();
  }
  
  ASTStreamer_Terminate(Streamer);
}

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

/// ProcessInputFile - Process a single input file with the specified state.
///
static void ProcessInputFile(const std::string &InFile, 
                             SourceManager &SourceMgr, Diagnostic &Diags,
                             HeaderSearch &HeaderInfo, TargetInfo &Target,
                             const LangOptions &LangInfo) {
  FileManager &FileMgr = HeaderInfo.getFileMgr();
  
  // Set up the preprocessor with these options.
  Preprocessor PP(Diags, LangInfo, Target, SourceMgr, HeaderInfo);
  
  // Install things like __POWERPC__, __GNUC__, etc into the macro table.
  std::vector<char> PrologMacros;
  InitializePredefinedMacros(PP, PrologMacros);
  
  // Read any files specified by -imacros or -include.
  ReadPrologFiles(PP, PrologMacros);
  
  // Figure out where to get and map in the main file.
  unsigned MainFileID = 0;
  if (InFile != "-") {
    const FileEntry *File = FileMgr.getFile(InFile);
    if (File) MainFileID = SourceMgr.createFileID(File, SourceLocation());
    if (MainFileID == 0) {
      std::cerr << "Error reading '" << InFile << "'!\n";
      ++NumErrors;
      return;
    }
  } else {
    SourceBuffer *SB = SourceBuffer::getSTDIN();
    if (SB) MainFileID = SourceMgr.createFileIDForMemBuffer(SB);
    if (MainFileID == 0) {
      std::cerr << "Error reading standard input!  Empty?\n";
      ++NumErrors;
      return;
    }
  }
  
  // Now that we have emitted the predefined macros, #includes, etc into
  // PrologMacros, preprocess it to populate the initial preprocessor state.
  {
    // Memory buffer must end with a null byte!
    PrologMacros.push_back(0);
    
    SourceBuffer *SB = SourceBuffer::getMemBuffer(&PrologMacros.front(),
                                                  &PrologMacros.back(),
                                                  "<predefines>");
    assert(SB && "Cannot fail to create predefined source buffer");
    unsigned FileID = SourceMgr.createFileIDForMemBuffer(SB);
    assert(FileID && "Could not create FileID for predefines?");
    
    // Start parsing the predefines.
    PP.EnterSourceFile(FileID, 0);
    
    // Lex the file, which will read all the macros.
    LexerToken Tok;
    PP.Lex(Tok);
    assert(Tok.getKind() == tok::eof && "Didn't read entire file!");
    
    // Once we've read this, we're done.
  }
  
  switch (ProgAction) {
  case DumpTokens: {                 // Token dump mode.
    LexerToken Tok;
    // Start parsing the specified input file.
    PP.EnterSourceFile(MainFileID, 0, true);
    do {
      PP.Lex(Tok);
      PP.DumpToken(Tok, true);
      std::cerr << "\n";
    } while (Tok.getKind() != tok::eof);
    break;
  }
  case RunPreprocessorOnly: {        // Just lex as fast as we can, no output.
    LexerToken Tok;
    // Start parsing the specified input file.
    PP.EnterSourceFile(MainFileID, 0, true);
    do {
      PP.Lex(Tok);
    } while (Tok.getKind() != tok::eof);
    break;
  }
    
  case PrintPreprocessedInput:       // -E mode.
    DoPrintPreprocessedInput(MainFileID, PP, LangInfo);
    break;
    
  case ParseNoop:                    // -parse-noop
    ParseFile(PP, new MinimalAction(), MainFileID);
    break;
    
  case ParsePrintCallbacks:
    ParseFile(PP, CreatePrintParserActionsAction(), MainFileID);
    break;
  case ParseSyntaxOnly:              // -fsyntax-only
  case ParseAST:
    BuildASTs(PP, MainFileID);
    break;
  case ParseASTPrint:
    PrintASTs(PP, MainFileID);
    break;
  }
  
  if (Stats) {
    std::cerr << "\nSTATISTICS FOR '" << InFile << "':\n";
    PP.PrintStats();
    PP.getIdentifierTable().PrintStats();
    HeaderInfo.PrintStats();
    std::cerr << "\n";
  }
  
  HeaderInfo.ClearFileInfo();
}

static cl::list<std::string>
InputFilenames(cl::Positional, cl::desc("<input files>"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm cfe\n");
  sys::PrintStackTraceOnErrorSignal();
  
  // If no input was specified, read from stdin.
  if (InputFilenames.empty())
    InputFilenames.push_back("-");
  
  /// Create a SourceManager object.  This tracks and owns all the file buffers
  /// allocated to the program.
  SourceManager SourceMgr;
  
  // Print diagnostics to stderr.
  DiagnosticPrinterSTDERR OurDiagnosticClient(SourceMgr);
  
  // Configure our handling of diagnostics.
  Diagnostic Diags(OurDiagnosticClient);
  InitializeDiagnostics(Diags);
  
  // Get information about the targets being compiled for.  Note that this
  // pointer and the TargetInfoImpl objects are never deleted by this toy
  // driver.
  TargetInfo *Target = CreateTargetInfo(Diags);
  if (Target == 0) {
    std::cerr << "Sorry, don't know what target this is, please use -arch.\n";
    return 1;
  }
  
  // Create a file manager object to provide access to and cache the filesystem.
  FileManager FileMgr;
  
  // Initialize language options, inferring file types from input filenames.
  // FIXME: This infers info from the first file, we should clump by language
  // to handle 'x.c y.c a.cpp b.cpp'.
  LangOptions LangInfo;
  InitializeBaseLanguage(LangInfo, InputFilenames[0]);
  InitializeLanguageStandard(LangInfo);
  
  // Process the -I options and set them in the HeaderInfo.
  HeaderSearch HeaderInfo(FileMgr);
  InitializeIncludePaths(HeaderInfo, FileMgr, Diags, LangInfo);
  
  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i)
    ProcessInputFile(InputFilenames[i], SourceMgr, Diags,
                     HeaderInfo, *Target, LangInfo);
  
  if (NumDiagnostics)
    std::cerr << NumDiagnostics << " diagnostics generated.\n";
  
  if (Stats) {
    // Printed from high-to-low level.
    SourceMgr.PrintStats();
    FileMgr.PrintStats();
    std::cerr << "\n";
  }
  
  return NumErrors != 0;
}
