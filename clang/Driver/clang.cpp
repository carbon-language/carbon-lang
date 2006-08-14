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
//   clang --help         - Output information about command line switches
//   clang [options]      - Read from stdin.
//   clang [options] file - Read from "file".
//
//===----------------------------------------------------------------------===//
//
// TODO: Options to support:
//
//   -ffatal-errors
//   -ftabstop=width
//   -fdollars-in-identifiers
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceBuffer.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/System/MappedFile.h"
#include "llvm/System/Signals.h"
#include <iostream>
using namespace llvm;
using namespace clang;

static unsigned NumDiagnostics = 0;

//===----------------------------------------------------------------------===//
// Global options.
//===----------------------------------------------------------------------===//

static cl::opt<bool>
Verbose("v", cl::desc("Enable verbose output"));
static cl::opt<bool>
Stats("stats", cl::desc("Print performance metrics and statistics"));

enum ProgActions {
  ParsePrintCallbacks,          // Parse and print each callback.
  ParseNoop,                    // Parse with noop callbacks.
  ParseSyntaxOnly,              // Parse and perform semantic analysis.
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
             clEnumValN(ParseSyntaxOnly, "fsyntax-only",
                        "Run parser and perform semantic analysis"),
             clEnumValN(ParsePrintCallbacks, "parse-print-callbacks",
                        "Run parser and print each callback invoked"),
             clEnumValN(ParseNoop, "parse-noop",
                        "Run parser with noop callbacks (for timings)"),
             // TODO: NULL PARSER.
             clEnumValEnd));


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
  case Diagnostic::Note: std::cerr << "note: "; break;
  case Diagnostic::Warning: std::cerr << "warning: "; break;
  case Diagnostic::Error: std::cerr << "error: "; break;
  case Diagnostic::Fatal: std::cerr << "fatal error: "; break;
  case Diagnostic::Sorry: std::cerr << "sorry, unimplemented: "; break;
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
  // FIXME: These should all be defined in the preprocessor according to the
  // current language configuration.
  DefineBuiltinMacro(Buf, "__STDC__=1");
  //DefineBuiltinMacro(Buf, "__cplusplus=1");
  //DefineBuiltinMacro(Buf, "__ASSEMBLER__=1");
  //DefineBuiltinMacro(Buf, "__STDC_VERSION__=199409L");
  DefineBuiltinMacro(Buf, "__STDC_VERSION__=199901L");
  DefineBuiltinMacro(Buf, "__STDC_HOSTED__=1");
  //DefineBuiltinMacro(Buf, "__OBJC__=1");
  
  // FIXME: This is obviously silly.  It should be more like gcc/c-cppbuiltin.c.
  // Macros predefined by GCC 4.0.1.
  DefineBuiltinMacro(Buf, "_ARCH_PPC=1");
  DefineBuiltinMacro(Buf, "_BIG_ENDIAN=1");
  DefineBuiltinMacro(Buf, "__APPLE_CC__=5250");
  DefineBuiltinMacro(Buf, "__APPLE__=1");
  DefineBuiltinMacro(Buf, "__BIG_ENDIAN__=1");
  DefineBuiltinMacro(Buf, "__CHAR_BIT__=8");
  DefineBuiltinMacro(Buf, "__CONSTANT_CFSTRINGS__=1");
  DefineBuiltinMacro(Buf, "__DBL_DENORM_MIN__=4.9406564584124654e-324");
  DefineBuiltinMacro(Buf, "__DBL_DIG__=15");
  DefineBuiltinMacro(Buf, "__DBL_EPSILON__=2.2204460492503131e-16");
  DefineBuiltinMacro(Buf, "__DBL_HAS_INFINITY__=1");
  DefineBuiltinMacro(Buf, "__DBL_HAS_QUIET_NAN__=1");
  DefineBuiltinMacro(Buf, "__DBL_MANT_DIG__=53");
  DefineBuiltinMacro(Buf, "__DBL_MAX_10_EXP__=308");
  DefineBuiltinMacro(Buf, "__DBL_MAX_EXP__=1024");
  DefineBuiltinMacro(Buf, "__DBL_MAX__=1.7976931348623157e+308");
  DefineBuiltinMacro(Buf, "__DBL_MIN_10_EXP__=(-307)");
  DefineBuiltinMacro(Buf, "__DBL_MIN_EXP__=(-1021)");
  DefineBuiltinMacro(Buf, "__DBL_MIN__=2.2250738585072014e-308");
  DefineBuiltinMacro(Buf, "__DECIMAL_DIG__=33");
  DefineBuiltinMacro(Buf, "__DYNAMIC__=1");
  DefineBuiltinMacro(Buf, "__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__=1030");
  DefineBuiltinMacro(Buf, "__FINITE_MATH_ONLY__=0");
  DefineBuiltinMacro(Buf, "__FLT_DENORM_MIN__=1.40129846e-45F");
  DefineBuiltinMacro(Buf, "__FLT_DIG__=6");
  DefineBuiltinMacro(Buf, "__FLT_EPSILON__=1.19209290e-7F");
  DefineBuiltinMacro(Buf, "__FLT_EVAL_METHOD__=0");
  DefineBuiltinMacro(Buf, "__FLT_HAS_INFINITY__=1");
  DefineBuiltinMacro(Buf, "__FLT_HAS_QUIET_NAN__=1");
  DefineBuiltinMacro(Buf, "__FLT_MANT_DIG__=24");
  DefineBuiltinMacro(Buf, "__FLT_MAX_10_EXP__=38");
  DefineBuiltinMacro(Buf, "__FLT_MAX_EXP__=128");
  DefineBuiltinMacro(Buf, "__FLT_MAX__=3.40282347e+38F");
  DefineBuiltinMacro(Buf, "__FLT_MIN_10_EXP__=(-37)");
  DefineBuiltinMacro(Buf, "__FLT_MIN_EXP__=(-125)");
  DefineBuiltinMacro(Buf, "__FLT_MIN__=1.17549435e-38F");
  DefineBuiltinMacro(Buf, "__FLT_RADIX__=2");
  DefineBuiltinMacro(Buf, "__GNUC_MINOR__=0");
  DefineBuiltinMacro(Buf, "__GNUC_PATCHLEVEL__=1");
  DefineBuiltinMacro(Buf, "__GNUC__=4");
  DefineBuiltinMacro(Buf, "__GXX_ABI_VERSION=1002");
  DefineBuiltinMacro(Buf, "__INTMAX_MAX__=9223372036854775807LL");
  DefineBuiltinMacro(Buf, "__INTMAX_TYPE__=long long int");
  DefineBuiltinMacro(Buf, "__INT_MAX__=2147483647");
  DefineBuiltinMacro(Buf, "__LDBL_DENORM_MIN__=4.940656458412465441765687"
                        "92868221e-324L");
  DefineBuiltinMacro(Buf, "__LDBL_DIG__=31");
  DefineBuiltinMacro(Buf, "__LDBL_EPSILON__=4.9406564584124654417656879286822"
                        "1e-324L");
  DefineBuiltinMacro(Buf, "__LDBL_HAS_INFINITY__=1");
  DefineBuiltinMacro(Buf, "__LDBL_HAS_QUIET_NAN__=1");
  DefineBuiltinMacro(Buf, "__LDBL_MANT_DIG__=106");
  DefineBuiltinMacro(Buf, "__LDBL_MAX_10_EXP__=308");
  DefineBuiltinMacro(Buf, "__LDBL_MAX_EXP__=1024");
  DefineBuiltinMacro(Buf, "__LDBL_MAX__=1.7976931348623158079372897140"
                        "5301e+308L");
  DefineBuiltinMacro(Buf, "__LDBL_MIN_10_EXP__=(-291)");
  DefineBuiltinMacro(Buf, "__LDBL_MIN_EXP__=(-968)");
  DefineBuiltinMacro(Buf, "__LDBL_MIN__=2.004168360008972777996108051350"
                        "16e-292L");
  DefineBuiltinMacro(Buf, "__LONG_DOUBLE_128__=1");
  DefineBuiltinMacro(Buf, "__LONG_LONG_MAX__=9223372036854775807LL");
  DefineBuiltinMacro(Buf, "__LONG_MAX__=2147483647L");
  DefineBuiltinMacro(Buf, "__MACH__=1");
  DefineBuiltinMacro(Buf, "__NATURAL_ALIGNMENT__=1");
  DefineBuiltinMacro(Buf, "__NO_INLINE__=1");
  DefineBuiltinMacro(Buf, "__PIC__=1");
  DefineBuiltinMacro(Buf, "__POWERPC__=1");
  DefineBuiltinMacro(Buf, "__PTRDIFF_TYPE__=int");
  DefineBuiltinMacro(Buf, "__REGISTER_PREFIX__");
  DefineBuiltinMacro(Buf, "__SCHAR_MAX__=127");
  DefineBuiltinMacro(Buf, "__SHRT_MAX__=32767");
  DefineBuiltinMacro(Buf, "__SIZE_TYPE__=long unsigned int");
  DefineBuiltinMacro(Buf, "__UINTMAX_TYPE__=long long unsigned int");
  DefineBuiltinMacro(Buf, "__USER_LABEL_PREFIX__=_");
  DefineBuiltinMacro(Buf, "__VERSION__=\"4.0.1 (Apple Computer, Inc. "
                        "build 5250)\"");
  DefineBuiltinMacro(Buf, "__WCHAR_MAX__=2147483647");
  DefineBuiltinMacro(Buf, "__WCHAR_TYPE__=int");
  DefineBuiltinMacro(Buf, "__WINT_TYPE__=int");
  DefineBuiltinMacro(Buf, "__ppc__=1");
  DefineBuiltinMacro(Buf, "__strong");
  DefineBuiltinMacro(Buf, "__weak");
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
                    FileManager &FM) {
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
  
  IncludeGroup[Group].push_back(DirectoryLookup(DE, Type, isUserSupplied));
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

// Process the -I options and set them in the preprocessor.
static void InitializeIncludePaths(Preprocessor &PP) {
  FileManager &FM = PP.getFileManager();

  // Handle -I... options.
  for (unsigned i = 0, e = I_dirs.size(); i != e; ++i) {
    if (I_dirs[i] == "-") {
      // -I- is a deprecated GCC feature.
      PP.getDiagnostics().Report(SourceLocation(),
                                 diag::err_pp_I_dash_not_supported);
    } else {
      AddPath(I_dirs[i], Angled, false, true, FM);
    }
  }
  
  // Handle -idirafter... options.
  for (unsigned i = 0, e = idirafter_dirs.size(); i != e; ++i)
    AddPath(idirafter_dirs[i], After, false, true, FM);
  
  // Handle -iquote... options.
  for (unsigned i = 0, e = iquote_dirs.size(); i != e; ++i)
    AddPath(iquote_dirs[i], Quoted, false, true, FM);
  
  // Handle -isystem... options.
  for (unsigned i = 0, e = isystem_dirs.size(); i != e; ++i)
    AddPath(isystem_dirs[i], System, false, true, FM);

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
                System, false, false, FM);
        ++iwithprefix_idx;
        iwithprefix_done = iwithprefix_idx == iwithprefix_vals.size();
      } else {
        AddPath(Prefix+iwithprefixbefore_vals[iwithprefixbefore_idx], 
                Angled, false, false, FM);
        ++iwithprefixbefore_idx;
        iwithprefixbefore_done = 
          iwithprefixbefore_idx == iwithprefixbefore_vals.size();
      }
    }
  }
  
  // FIXME: Add contents of the CPATH, C_INCLUDE_PATH, CPLUS_INCLUDE_PATH,
  // OBJC_INCLUDE_PATH, OBJCPLUS_INCLUDE_PATH environment variables.
  
  // FIXME: temporary hack: hard-coded paths.
  if (!nostdinc) {
    AddPath("/usr/local/include", System, false, false, FM);
    AddPath("/usr/lib/gcc/powerpc-apple-darwin8/4.0.1/include", 
            System, false, false, FM);
    AddPath("/usr/lib/gcc/powerpc-apple-darwin8/"
            "4.0.1/../../../../powerpc-apple-darwin8/include", 
            System, false, false, FM);
    AddPath("/usr/include", System, false, false, FM);
    AddPath("/System/Library/Frameworks", System, false, false, FM);
    AddPath("/Library/Frameworks", System, false, false, FM);
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
  PP.SetSearchPaths(SearchList, IncludeGroup[Quoted].size(),
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
// Parser driver
//===----------------------------------------------------------------------===//

static void ParseFile(Preprocessor &PP, Action *PA, unsigned MainFileID) {
  Parser P(PP, *PA);

  PP.EnterSourceFile(MainFileID, 0, true);
  
  P.ParseTranslationUnit();

  // Start parsing the specified input file.

  
  delete PA;
}

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm cfe\n");
  sys::PrintStackTraceOnErrorSignal();
  
  /// Create a SourceManager object.  This tracks and owns all the file buffers
  /// allocated to the program.
  SourceManager SourceMgr;
  
  // Print diagnostics to stderr.
  DiagnosticPrinterSTDERR OurDiagnosticClient(SourceMgr);
  
  // Configure our handling of diagnostics.
  Diagnostic OurDiagnostics(OurDiagnosticClient);
  InitializeDiagnostics(OurDiagnostics);
  
  // Turn all options on.
  // FIXME: add -ansi and -std= options.
  LangOptions Options;
  Options.Trigraphs = 1;
  Options.BCPLComment = 1;  // Only for C99/C++.
  Options.C99 = 1;
  Options.DollarIdents = Options.Digraphs = 1;
  Options.ObjC1 = Options.ObjC2 = 1;

  // Create a file manager object to provide access to and cache the filesystem.
  FileManager FileMgr;
  
  // Set up the preprocessor with these options.
  Preprocessor PP(OurDiagnostics, Options, FileMgr, SourceMgr);
  
  // Install things like __POWERPC__, __GNUC__, etc into the macro table.
  std::vector<char> PrologMacros;
  InitializePredefinedMacros(PP, PrologMacros);
  
  // Process the -I options and set them in the preprocessor.
  InitializeIncludePaths(PP);

  // Read any files specified by -imacros or -include.
  ReadPrologFiles(PP, PrologMacros);
  
  // Set up keywords.
  PP.AddKeywords();
  
  // Figure out where to get and map in the main file.
  unsigned MainFileID = 0;
  if (InputFilename != "-") {
    const FileEntry *File = FileMgr.getFile(InputFilename);
    if (File) MainFileID = SourceMgr.createFileID(File, SourceLocation());
    if (MainFileID == 0) {
      std::cerr << "Error reading '" << InputFilename << "'!\n";
      return 1;
    }
  } else {
    SourceBuffer *SB = SourceBuffer::getSTDIN();
    if (SB) MainFileID = SourceMgr.createFileIDForMemBuffer(SB);
    if (MainFileID == 0) {
      std::cerr << "Error reading standard input!  Empty?\n";
      return 1;
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
    DoPrintPreprocessedInput(MainFileID, PP, Options);
    break;

  case ParseNoop:                    // -parse-noop
  case ParsePrintCallbacks:
    //ParseFile(PP, new ParserPrintActions(PP), MainFileID);
    break;
  case ParseSyntaxOnly:              // -fsyntax-only
    ParseFile(PP, new Action(), MainFileID);
    break;
  }
  
  if (NumDiagnostics)
    std::cerr << NumDiagnostics << " diagnostics generated.\n";
  
  if (Stats) {
    // Printed from low-to-high level.
    PP.getFileManager().PrintStats();
    PP.getSourceManager().PrintStats();
    PP.getIdentifierTable().PrintStats();
    PP.PrintStats();
    std::cerr << "\n";
  }
}
