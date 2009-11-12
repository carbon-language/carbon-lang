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

#include "Options.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/AnalysisConsumer.h"
#include "clang/Frontend/ChainedDiagnosticClient.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/DependencyOutputOptions.h"
#include "clang/Frontend/FixItRewriter.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/PCHReader.h"
#include "clang/Frontend/PathDiagnosticClients.h"
#include "clang/Frontend/PreprocessorOptions.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/ParseAST.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
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
// Code Completion Options
//===----------------------------------------------------------------------===//

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
// Frontend Actions
//===----------------------------------------------------------------------===//

enum ProgActions {
  RewriteObjC,                  // ObjC->C Rewriter.
  RewriteBlocks,                // ObjC->C Rewriter for Blocks.
  RewriteMacros,                // Expand macros but not #includes.
  RewriteTest,                  // Rewriter playground
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
             clEnumValEnd));

//===----------------------------------------------------------------------===//
// Frontend Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool>
DisableFree("disable-free",
           llvm::cl::desc("Disable freeing of memory on exit"),
           llvm::cl::init(false));

static llvm::cl::opt<bool>
EmptyInputOnly("empty-input-only",
      llvm::cl::desc("Force running on an empty input file"));

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

static llvm::cl::opt<bool>
TimeReport("ftime-report",
           llvm::cl::desc("Print the amount of time each "
                          "phase of compilation takes"));

static llvm::cl::opt<bool>
Verbose("v", llvm::cl::desc("Enable verbose output"));

static llvm::cl::opt<bool>
VerifyDiagnostics("verify",
                  llvm::cl::desc("Verify emitted diagnostics and warnings"));

//===----------------------------------------------------------------------===//
// Language Options
//===----------------------------------------------------------------------===//

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

static llvm::cl::opt<std::string>
TargetTriple("triple",
  llvm::cl::desc("Specify target triple (e.g. i686-apple-darwin9)"));

static llvm::cl::opt<std::string>
TargetABI("target-abi",
          llvm::cl::desc("Target a particular ABI type"));

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
      PP.getDiagnostics().Report(diag::err_fe_error_reading) << InFile.c_str();
      return true;
    }
  } else {
    llvm::MemoryBuffer *SB = llvm::MemoryBuffer::getSTDIN();
    SourceMgr.createMainFileIDForMemBuffer(SB);
    if (SourceMgr.getMainFileID().isInvalid()) {
      PP.getDiagnostics().Report(diag::err_fe_error_reading_stdin);
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Preprocessor construction
//===----------------------------------------------------------------------===//

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

static Preprocessor *
CreatePreprocessor(Diagnostic &Diags, const LangOptions &LangInfo,
                   const PreprocessorOptions &PPOpts,
                   const HeaderSearchOptions &HSOpts,
                   const DependencyOutputOptions &DepOpts,
                   TargetInfo &Target, SourceManager &SourceMgr,
                   FileManager &FileMgr) {
  // Create a PTH manager if we are using some form of a token cache.
  PTHManager *PTHMgr = 0;
  if (!PPOpts.getTokenCache().empty())
    PTHMgr = PTHManager::Create(PPOpts.getTokenCache(), Diags);

  // FIXME: Don't fail like this.
  if (Diags.hasErrorOccurred())
    exit(1);

  // Create the Preprocessor.
  HeaderSearch *HeaderInfo = new HeaderSearch(FileMgr);
  Preprocessor *PP = new Preprocessor(Diags, LangInfo, Target,
                                      SourceMgr, *HeaderInfo, PTHMgr,
                                      /*OwnsHeaderSearch=*/true);

  // Note that this is different then passing PTHMgr to Preprocessor's ctor.
  // That argument is used as the IdentifierInfoLookup argument to
  // IdentifierTable's ctor.
  if (PTHMgr) {
    PTHMgr->setPreprocessor(PP);
    PP->setPTHManager(PTHMgr);
  }

  InitializePreprocessor(*PP, PPOpts, HSOpts);

  // Handle generating dependencies, if requested.
  if (!DepOpts.OutputFile.empty())
    AttachDependencyFileGen(*PP, DepOpts);

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
// Dump Build Information
//===----------------------------------------------------------------------===//

static void SetUpBuildDumpLog(const DiagnosticOptions &DiagOpts,
                              unsigned argc, char **argv,
                              llvm::OwningPtr<DiagnosticClient> &DiagClient) {
  std::string ErrorInfo;
  llvm::raw_ostream *OS =
    new llvm::raw_fd_ostream(DiagOpts.DumpBuildInformation.c_str(), ErrorInfo);
  if (!ErrorInfo.empty()) {
    llvm::errs() << "error opening -dump-build-information file '"
                 << DiagOpts.DumpBuildInformation << "', option ignored!\n";
    delete OS;
    return;
  }

  (*OS) << "clang-cc command line arguments: ";
  for (unsigned i = 0; i != argc; ++i)
    (*OS) << argv[i] << ' ';
  (*OS) << '\n';

  // Chain in a diagnostic client which will log the diagnostics.
  DiagnosticClient *Logger =
    new TextDiagnosticPrinter(*OS, DiagOpts, /*OwnsOutputStream=*/true);
  DiagClient.reset(new ChainedDiagnosticClient(DiagClient.take(), Logger));
}

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

/// ClangFrontendTimer - The front-end activities should charge time to it with
/// TimeRegion.  The -ftime-report option controls whether this will do
/// anything.
llvm::Timer *ClangFrontendTimer = 0;

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

/// AddFixItLocations - Add any individual user specified "fix-it" locations,
/// and return true on success (if any were added).
static bool AddFixItLocations(FixItRewriter *FixItRewrite,
                              FileManager &FileMgr) {
  bool AddedFixItLocation = false;

  for (unsigned i = 0, e = FixItAtLocations.size(); i != e; ++i) {
    if (const FileEntry *File = FileMgr.getFile(FixItAtLocations[i].FileName)) {
      RequestedSourceLocation Requested;
      Requested.File = File;
      Requested.Line = FixItAtLocations[i].Line;
      Requested.Column = FixItAtLocations[i].Column;
      FixItRewrite->addFixItLocation(Requested);
      AddedFixItLocation = true;
    } else {
      llvm::errs() << "FIX-IT could not find file \""
                   << FixItAtLocations[i].FileName << "\"\n";
    }
  }

  return AddedFixItLocation;
}

static ASTConsumer *CreateConsumerAction(const CompilerInvocation &CompOpts,
                                         Preprocessor &PP,
                                         const std::string &InFile,
                                         ProgActions PA,
                                         llvm::OwningPtr<llvm::raw_ostream> &OS,
                                         llvm::sys::Path &OutPath,
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

    // Fix-its can change semantics, disallow with any IRgen action.
    if (FixItAll || !FixItAtLocations.empty()) {
      PP.getDiagnostics().Report(diag::err_fe_no_fixit_and_codegen);
      return 0;
    }

    return CreateBackendConsumer(Act, PP.getDiagnostics(), PP.getLangOptions(),
                                 CompOpts.getCodeGenOpts(), InFile, OS.get(),
                                 Context);
  }

  case RewriteObjC:
    OS.reset(ComputeOutFile(CompOpts, InFile, "cpp", true, OutPath));
    return CreateObjCRewriter(InFile, OS.get(), PP.getDiagnostics(),
                              PP.getLangOptions(),
                              CompOpts.getDiagnosticOpts().NoRewriteMacros);

  case RewriteBlocks:
    return CreateBlockRewriter(InFile, PP.getDiagnostics(),
                               PP.getLangOptions());

  case ParseSyntaxOnly:
    return new ASTConsumer();

  case PrintDeclContext:
    return CreateDeclContextPrinter();
  }
}

/// ReadPCHFile - Load a PCH file from disk, and initialize the preprocessor for
/// reading from the PCH file.
///
/// \return The AST source, or null on failure.
static ExternalASTSource *ReadPCHFile(llvm::StringRef Path,
                                      const CompilerInvocation &CompOpts,
                                      Preprocessor &PP,
                                      ASTContext &Context) {
  // If the user specified -isysroot, it will be used for relocatable PCH files.
  const char *isysrootPCH = CompOpts.getHeaderSearchOpts().Sysroot.c_str();
  if (isysrootPCH[0] == '\0')
    isysrootPCH = 0;

  llvm::OwningPtr<PCHReader> Reader;
  Reader.reset(new PCHReader(PP, &Context, isysrootPCH));

  switch (Reader->ReadPCH(Path)) {
  case PCHReader::Success:
    // Set the predefines buffer as suggested by the PCH reader. Typically, the
    // predefines buffer will be empty.
    PP.setPredefines(Reader->getSuggestedPredefines());
    return Reader.take();

  case PCHReader::Failure:
    // Unrecoverable failure: don't even try to process the input file.
    break;

  case PCHReader::IgnorePCH:
    // No suitable PCH file could be found. Return an error.
    break;
  }

  return 0;
}

/// ProcessInputFile - Process a single input file with the specified state.
///
static void ProcessInputFile(const CompilerInvocation &CompOpts,
                             Preprocessor &PP, const std::string &InFile,
                             ProgActions PA, llvm::LLVMContext& Context) {
  llvm::OwningPtr<llvm::raw_ostream> OS;
  llvm::OwningPtr<ASTConsumer> Consumer;
  FixItRewriter *FixItRewrite = 0;
  bool CompleteTranslationUnit = true;
  llvm::sys::Path OutPath;

  switch (PA) {
  default:
    Consumer.reset(CreateConsumerAction(CompOpts, PP, InFile, PA, OS, OutPath,
                                        Context));
    if (!Consumer.get()) {
      PP.getDiagnostics().Report(diag::err_fe_invalid_ast_action);
      return;
    }
    break;

  case EmitHTML:
    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    Consumer.reset(CreateHTMLPrinter(OS.get(), PP));
    break;

  case RunAnalysis:
    Consumer.reset(CreateAnalysisConsumer(PP, CompOpts.getOutputFile(),
                                          CompOpts.getAnalyzerOpts()));
    break;

  case GeneratePCH: {
    const std::string &Sysroot = CompOpts.getHeaderSearchOpts().Sysroot;
    if (RelocatablePCH.getValue() && Sysroot.empty()) {
      PP.Diag(SourceLocation(), diag::err_relocatable_without_without_isysroot);
      RelocatablePCH.setValue(false);
    }

    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    if (RelocatablePCH.getValue())
      Consumer.reset(CreatePCHGenerator(PP, OS.get(), Sysroot.c_str()));
    else
      Consumer.reset(CreatePCHGenerator(PP, OS.get()));
    CompleteTranslationUnit = false;
    break;
  }

    // Do any necessary set up for non-consumer actions.
  case DumpRawTokens:
  case DumpTokens:
  case RunPreprocessorOnly:
  case ParseNoop:
    break; // No setup.

  case GeneratePTH:
    if (CompOpts.getOutputFile().empty() || CompOpts.getOutputFile() == "-") {
      // FIXME: Don't fail this way.
      // FIXME: Verify that we can actually seek in the given file.
      llvm::errs() << "ERROR: PTH requires an seekable file for output!\n";
      ::exit(1);
    }
    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    break;

  case PrintPreprocessedInput:
  case ParsePrintCallbacks:
  case RewriteMacros:
  case RewriteTest:
    OS.reset(ComputeOutFile(CompOpts, InFile, 0, true, OutPath));
    break;
  }

  // Check if we want a fix-it rewriter.
  if (FixItAll || !FixItAtLocations.empty()) {
    FixItRewrite = new FixItRewriter(PP.getDiagnostics(),
                                     PP.getSourceManager(),
                                     PP.getLangOptions());
    if (!FixItAtLocations.empty() &&
        !AddFixItLocations(FixItRewrite, PP.getFileManager())) {
      // All of the fix-it locations were bad. Don't fix anything.
      delete FixItRewrite;
      FixItRewrite = 0;
    }
  }

  llvm::OwningPtr<ASTContext> ContextOwner;
  llvm::OwningPtr<ExternalASTSource> Source;
  const std::string &ImplicitPCHInclude =
    CompOpts.getPreprocessorOpts().getImplicitPCHInclude();
  if (Consumer) {
    ContextOwner.reset(new ASTContext(PP.getLangOptions(),
                                      PP.getSourceManager(),
                                      PP.getTargetInfo(),
                                      PP.getIdentifierTable(),
                                      PP.getSelectorTable(),
                                      PP.getBuiltinInfo(),
                                      /* FreeMemory = */ !DisableFree,
                                      /* size_reserve = */0));

    if (!ImplicitPCHInclude.empty()) {
      Source.reset(ReadPCHFile(ImplicitPCHInclude, CompOpts, PP,
                               *ContextOwner));
      if (!Source)
        return;

      // Attach the PCH reader to the AST context as an external AST source, so
      // that declarations will be deserialized from the PCH file as needed.
      ContextOwner->setExternalSource(Source);
    } else {
      // Initialize builtin info when not using PCH.
      PP.getBuiltinInfo().InitializeBuiltins(PP.getIdentifierTable(),
                                             PP.getLangOptions().NoBuiltin);
    }

    // Initialize the main file entry. This needs to be delayed until after PCH
    // has loaded.
    if (InitializeSourceManager(PP, InFile))
      return;

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
        PP.getDiagnostics().Report(diag::err_fe_invalid_code_complete_file)
          << CodeCompletionAt.FileName;
      }
    }

    // Run the AST consumer action.
    ParseAST(PP, Consumer.get(), *ContextOwner.get(), Stats,
             CompleteTranslationUnit,
             CreateCodeCompleter, CreateCodeCompleterData);
  } else {
    // Initialize builtin info.
    PP.getBuiltinInfo().InitializeBuiltins(PP.getIdentifierTable(),
                                           PP.getLangOptions().NoBuiltin);

    // Initialize the main file entry. This needs to be delayed until after PCH
    // has loaded.
    if (InitializeSourceManager(PP, InFile))
      return;

    // Run the preprocessor actions.
    llvm::TimeRegion Timer(ClangFrontendTimer);
    switch (PA) {
    default:
      assert(0 && "unexpected program action");

    case DumpRawTokens: {
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
      break;
    }

    case DumpTokens: {
      Token Tok;
      // Start preprocessing the specified input file.
      PP.EnterMainSourceFile();
      do {
        PP.Lex(Tok);
        PP.DumpToken(Tok, true);
        fprintf(stderr, "\n");
      } while (Tok.isNot(tok::eof));
      break;
    }

    case GeneratePTH:
      CacheTokens(PP, static_cast<llvm::raw_fd_ostream*>(OS.get()));
      break;

    case ParseNoop:
      ParseFile(PP, new MinimalAction(PP));
      break;

    case ParsePrintCallbacks:
      ParseFile(PP, CreatePrintParserActionsAction(PP, OS.get()));
      break;

    case PrintPreprocessedInput:
      DoPrintPreprocessedInput(PP, OS.get(),
                               CompOpts.getPreprocessorOutputOpts());
      break;

    case RewriteMacros:
      RewriteMacrosInInput(PP, OS.get());
      break;

    case RewriteTest:
      DoRewriteTest(PP, OS.get());
      break;

    case RunPreprocessorOnly: {    // Just lex as fast as we can, no output.
      Token Tok;
      // Start parsing the specified input file.
      PP.EnterMainSourceFile();
      do {
        PP.Lex(Tok);
      } while (Tok.isNot(tok::eof));
      break;
    }
    }
  }

  if (FixItRewrite)
    FixItRewrite->WriteFixedFile(InFile, CompOpts.getOutputFile());

  // Release the consumer and the AST, in that order since the consumer may
  // perform actions in its destructor which require the context.
  if (DisableFree) {
    Consumer.take();
    ContextOwner.take();
  } else {
    Consumer.reset();
    ContextOwner.reset();
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
                                Diagnostic &Diags, FileManager &FileMgr,
                                llvm::LLVMContext& Context) {
  std::string Error;
  llvm::OwningPtr<ASTUnit> AST(ASTUnit::LoadFromPCHFile(InFile, &Error));
  if (!AST) {
    Diags.Report(diag::err_fe_invalid_ast_file) << Error;
    return;
  }

  Preprocessor &PP = AST->getPreprocessor();

  llvm::OwningPtr<llvm::raw_ostream> OS;
  llvm::sys::Path OutPath;
  llvm::OwningPtr<ASTConsumer> Consumer(CreateConsumerAction(CompOpts, PP,
                                                             InFile, PA, OS,
                                                             OutPath, Context));
  if (!Consumer.get()) {
    Diags.Report(diag::err_fe_invalid_ast_action);
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

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
  Diagnostic &Diags = *static_cast<Diagnostic*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

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

static void FinalizeCodeGenOptions(CodeGenOptions &Opts,
                                   const LangOptions &Lang) {
  if (Lang.NoBuiltin)
    Opts.SimplifyLibCalls = 0;
  if (Lang.CPlusPlus)
    Opts.NoCommon = 1;

  // Handle -ftime-report.
  Opts.TimePasses = TimeReport;
}

static void ConstructCompilerInvocation(CompilerInvocation &Opts,
                                        const char *Argv0,
                                        const DiagnosticOptions &DiagOpts,
                                        TargetInfo &Target,
                                        LangKind LK) {
  Opts.getDiagnosticOpts() = DiagOpts;

  Opts.getOutputFile() = OutputFile;

  // Initialize backend options, which may also be used to key some language
  // options.
  InitializeCodeGenOptions(Opts.getCodeGenOpts(), Target);

  // Initialize language options.
  //
  // FIXME: These aren't used during operations on ASTs. Split onto a separate
  // code path to make this obvious.
  if (LK != langkind_ast)
    InitializeLangOptions(Opts.getLangOpts(), LK, Target,
                          Opts.getCodeGenOpts());

  // Initialize the static analyzer options.
  InitializeAnalyzerOptions(Opts.getAnalyzerOpts());

  // Initialize the dependency output options (-M...).
  InitializeDependencyOutputOptions(Opts.getDependencyOutputOpts());

  // Initialize the header search options.
  InitializeHeaderSearchOptions(Opts.getHeaderSearchOpts(),
                                GetBuiltinIncludePath(Argv0),
                                Verbose,
                                Opts.getLangOpts());

  // Initialize the other preprocessor options.
  InitializePreprocessorOptions(Opts.getPreprocessorOpts());

  // Initialize the preprocessed output options.
  InitializePreprocessorOutputOptions(Opts.getPreprocessorOutputOpts());

  // Finalize some code generation options.
  FinalizeCodeGenOptions(Opts.getCodeGenOpts(), Opts.getLangOpts());
}

static Diagnostic *CreateDiagnosticEngine(const DiagnosticOptions &Opts,
                                          int argc, char **argv) {
  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  llvm::OwningPtr<DiagnosticClient> DiagClient;
  if (VerifyDiagnostics) {
    // When checking diagnostics, just buffer them up.
    DiagClient.reset(new TextDiagnosticBuffer());
  } else {
    DiagClient.reset(new TextDiagnosticPrinter(llvm::errs(), Opts));
  }

  if (!Opts.DumpBuildInformation.empty())
    SetUpBuildDumpLog(Opts, argc, argv, DiagClient);

  // Configure our handling of diagnostics.
  Diagnostic *Diags = new Diagnostic(DiagClient.take());
  if (ProcessWarningOptions(*Diags, Opts))
    return 0;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::llvm_install_error_handler(LLVMErrorHandler,
                                   static_cast<void*>(Diags));

  return Diags;
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

  if (VerifyDiagnostics && InputFilenames.size() > 1) {
    fprintf(stderr, "-verify only works on single input files.\n");
    return 1;
  }

  if (TimeReport)
    ClangFrontendTimer = new llvm::Timer("Clang front-end time");

  if (Verbose)
    llvm::errs() << "clang-cc version " CLANG_VERSION_STRING
                 << " based upon " << PACKAGE_STRING
                 << " hosted on " << llvm::sys::getHostTriple() << "\n";

  // If no input was specified, read from stdin.
  if (InputFilenames.empty())
    InputFilenames.push_back("-");

  // Construct the diagnostic engine first, so that we can build a diagnostic
  // client to use for any errors during option handling.
  DiagnosticOptions DiagOpts;
  InitializeDiagnosticOptions(DiagOpts);
  llvm::OwningPtr<Diagnostic>
    Diags(CreateDiagnosticEngine(DiagOpts, argc, argv));
  if (!Diags)
    return 1;

  // FIXME: Hack to make sure we release the diagnostic client, the engine
  // should (optionally?) take ownership of it.
  llvm::OwningPtr<DiagnosticClient> DiagClient(Diags->getClient());

  // Initialize base triple.  If a -triple option has been specified, use
  // that triple.  Otherwise, default to the host triple.
  llvm::Triple Triple(TargetTriple);
  if (Triple.getTriple().empty())
    Triple = llvm::Triple(llvm::sys::getHostTriple());

  // Get information about the target being compiled for.
  llvm::OwningPtr<TargetInfo>
  Target(TargetInfo::CreateTargetInfo(Triple.getTriple()));
  if (Target == 0) {
    Diags->Report(diag::err_fe_unknown_triple) << Triple.getTriple().c_str();
    return 1;
  }

  // Set the target ABI if specified.
  if (!TargetABI.empty() &&!Target->setABI(TargetABI)) {
    Diags->Report(diag::err_fe_unknown_target_abi) << TargetABI;
    return 1;
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
  ConstructCompilerInvocation(CompOpts, argv[0], DiagOpts, *Target, LK);

  // Create the source manager.
  SourceManager SourceMgr;

  // Create a file manager object to provide access to and cache the filesystem.
  FileManager FileMgr;

  for (unsigned i = 0, e = InputFilenames.size(); i != e; ++i) {
    const std::string &InFile = InputFilenames[i];

    // AST inputs are handled specially.
    if (LK == langkind_ast) {
      ProcessASTInputFile(CompOpts, InFile, ProgAction, *Diags, FileMgr,
                          Context);
      continue;
    }

    // Reset the ID tables if we are reusing the SourceManager.
    if (i)
      SourceMgr.clearIDTables();

    // Set up the preprocessor with these options.
    llvm::OwningPtr<Preprocessor>
      PP(CreatePreprocessor(*Diags, CompOpts.getLangOpts(),
                            CompOpts.getPreprocessorOpts(),
                            CompOpts.getHeaderSearchOpts(),
                            CompOpts.getDependencyOutputOpts(),
                            *Target, SourceMgr, FileMgr));

    // Process the source file.
    Diags->getClient()->BeginSourceFile(CompOpts.getLangOpts());
    ProcessInputFile(CompOpts, *PP, InFile, ProgAction, Context);
    Diags->getClient()->EndSourceFile();
  }

  if (CompOpts.getDiagnosticOpts().ShowCarets)
    if (unsigned NumDiagnostics = Diags->getNumDiagnostics())
      fprintf(stderr, "%d diagnostic%s generated.\n", NumDiagnostics,
              (NumDiagnostics == 1 ? "" : "s"));

  if (Stats) {
    FileMgr.PrintStats();
    fprintf(stderr, "\n");
  }

  delete ClangFrontendTimer;

  // If verifying diagnostics and we reached here, all is well.
  if (VerifyDiagnostics)
    return 0;

  // Managed static deconstruction. Useful for making things like
  // -time-passes usable.
  llvm::llvm_shutdown();

  return (Diags->getNumErrors() != 0);
}
