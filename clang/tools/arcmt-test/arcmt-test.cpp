//===-- arcmt-test.cpp - ARC Migration Tool testbed -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/ARCMigrate/ARCMT.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticsClient.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"

using namespace clang;
using namespace arcmt;

static llvm::cl::opt<bool>
CheckOnly("check-only",
      llvm::cl::desc("Just check for issues that need to be handled manually"));

//static llvm::cl::opt<bool>
//TestResultForARC("test-result",
//llvm::cl::desc("Test the result of transformations by parsing it in ARC mode"));

static llvm::cl::opt<bool>
OutputTransformations("output-transformations",
                      llvm::cl::desc("Print the source transformations"));

static llvm::cl::opt<bool>
VerifyDiags("verify",llvm::cl::desc("Verify emitted diagnostics and warnings"));

static llvm::cl::opt<bool>
VerboseOpt("v", llvm::cl::desc("Enable verbose output"));

static llvm::cl::extrahelp extraHelp(
  "\nusage with compiler args: arcmt-test [options] --args [compiler flags]\n");

// This function isn't referenced outside its translation unit, but it
// can't use the "static" keyword because its address is used for
// GetMainExecutable (since some platforms don't support taking the
// address of main, and some platforms can't implement GetMainExecutable
// without being given the address of a function in the main executable).
llvm::sys::Path GetExecutablePath(const char *Argv0) {
  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *MainAddr = (void*) (intptr_t) GetExecutablePath;
  return llvm::sys::Path::GetMainExecutable(Argv0, MainAddr);
}

static void printSourceLocation(SourceLocation loc, ASTContext &Ctx,
                                llvm::raw_ostream &OS);
static void printSourceRange(CharSourceRange range, ASTContext &Ctx,
                             llvm::raw_ostream &OS);

namespace {

class PrintTransforms : public MigrationProcess::RewriteListener {
  ASTContext *Ctx;
  llvm::raw_ostream &OS;

public:
  PrintTransforms(llvm::raw_ostream &OS)
    : Ctx(0), OS(OS) { }

  virtual void start(ASTContext &ctx) { Ctx = &ctx; }
  virtual void finish() { Ctx = 0; }

  virtual void insert(SourceLocation loc, llvm::StringRef text) {
    assert(Ctx);
    OS << "Insert: ";
    printSourceLocation(loc, *Ctx, OS);
    OS << " \"" << text << "\"\n";
  }

  virtual void remove(CharSourceRange range) {
    assert(Ctx);
    OS << "Remove: ";
    printSourceRange(range, *Ctx, OS);
    OS << '\n';
  }
};

} // anonymous namespace

static bool checkForMigration(llvm::StringRef resourcesPath,
                              llvm::ArrayRef<const char *> Args) {
  DiagnosticClient *DiagClient =
    new TextDiagnosticPrinter(llvm::errs(), DiagnosticOptions());
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags(new Diagnostic(DiagID, DiagClient));
  // Chain in -verify checker, if requested.
  VerifyDiagnosticsClient *verifyDiag = 0;
  if (VerifyDiags) {
    verifyDiag = new VerifyDiagnosticsClient(*Diags, Diags->takeClient());
    Diags->setClient(verifyDiag);
  }

  CompilerInvocation CI;
  CompilerInvocation::CreateFromArgs(CI, Args.begin(), Args.end(), *Diags);

  if (CI.getFrontendOpts().Inputs.empty()) {
    llvm::errs() << "error: no input files\n";
    return true;
  }

  if (!CI.getLangOpts().ObjC1)
    return false;

  return arcmt::checkForManualIssues(CI,
                                     CI.getFrontendOpts().Inputs[0].second,
                                     CI.getFrontendOpts().Inputs[0].first,
                                     Diags->getClient());
}

static void printResult(FileRemapper &remapper, llvm::raw_ostream &OS) {
  CompilerInvocation CI;
  remapper.applyMappings(CI);
  PreprocessorOptions &PPOpts = CI.getPreprocessorOpts();
  // The changed files will be in memory buffers, print them.
  for (unsigned i = 0, e = PPOpts.RemappedFileBuffers.size(); i != e; ++i) {
    const llvm::MemoryBuffer *mem = PPOpts.RemappedFileBuffers[i].second;
    OS << mem->getBuffer();
  }
}

static bool performTransformations(llvm::StringRef resourcesPath,
                                   llvm::ArrayRef<const char *> Args) {
  // Check first.
  if (checkForMigration(resourcesPath, Args))
    return true;

  DiagnosticClient *DiagClient =
    new TextDiagnosticPrinter(llvm::errs(), DiagnosticOptions());
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<Diagnostic> TopDiags(new Diagnostic(DiagID, DiagClient));

  CompilerInvocation origCI;
  CompilerInvocation::CreateFromArgs(origCI, Args.begin(), Args.end(),
                                     *TopDiags);

  if (origCI.getFrontendOpts().Inputs.empty()) {
    llvm::errs() << "error: no input files\n";
    return true;
  }

  if (!origCI.getLangOpts().ObjC1)
    return false;

  MigrationProcess migration(origCI, DiagClient);

  std::vector<TransformFn> transforms = arcmt::getAllTransformations();
  assert(!transforms.empty());

  llvm::OwningPtr<PrintTransforms> transformPrinter;
  if (OutputTransformations)
    transformPrinter.reset(new PrintTransforms(llvm::outs()));

  for (unsigned i=0, e = transforms.size(); i != e; ++i) {
    bool err = migration.applyTransform(transforms[i], transformPrinter.get());
    if (err) return true;

    if (VerboseOpt) {
      if (i == e-1)
        llvm::errs() << "\n##### FINAL RESULT #####\n";
      else
        llvm::errs() << "\n##### OUTPUT AFTER "<< i+1 <<". TRANSFORMATION #####\n";
      printResult(migration.getRemapper(), llvm::errs());
      llvm::errs() << "\n##########################\n\n";
    }
  }

  if (!OutputTransformations)
    printResult(migration.getRemapper(), llvm::outs());

  // FIXME: TestResultForARC

  return false;
}

//===----------------------------------------------------------------------===//
// Misc. functions.
//===----------------------------------------------------------------------===//

static void printSourceLocation(SourceLocation loc, ASTContext &Ctx,
                                llvm::raw_ostream &OS) {
  SourceManager &SM = Ctx.getSourceManager();
  PresumedLoc PL = SM.getPresumedLoc(loc);

  OS << llvm::sys::path::filename(PL.getFilename());
  OS << ":" << PL.getLine() << ":"
            << PL.getColumn();
}

static void printSourceRange(CharSourceRange range, ASTContext &Ctx,
                             llvm::raw_ostream &OS) {
  SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &langOpts = Ctx.getLangOptions();

  PresumedLoc PL = SM.getPresumedLoc(range.getBegin());

  OS << llvm::sys::path::filename(PL.getFilename());
  OS << " [" << PL.getLine() << ":"
             << PL.getColumn();
  OS << " - ";

  SourceLocation end = range.getEnd();
  PL = SM.getPresumedLoc(end);

  unsigned endCol = PL.getColumn() - 1;
  if (!range.isTokenRange())
    endCol += Lexer::MeasureTokenLength(end, SM, langOpts);
  OS << PL.getLine() << ":" << endCol << "]";
}

//===----------------------------------------------------------------------===//
// Command line processing.
//===----------------------------------------------------------------------===//

int main(int argc, const char **argv) {
  using llvm::StringRef;
  void *MainAddr = (void*) (intptr_t) GetExecutablePath;
  llvm::sys::PrintStackTraceOnErrorSignal();

  std::string
    resourcesPath = CompilerInvocation::GetResourcesPath(argv[0], MainAddr);

  int optargc = 0;
  for (; optargc != argc; ++optargc) {
    if (StringRef(argv[optargc]) == "--args")
      break;
  }
  llvm::cl::ParseCommandLineOptions(optargc, const_cast<char **>(argv), "arcmt-test");
  
  if (optargc == argc) {
    llvm::cl::PrintHelpMessage();
    return 1;
  }

  llvm::ArrayRef<const char*> Args(argv+optargc+1, argc-optargc-1);

  if (CheckOnly)
    return checkForMigration(resourcesPath, Args);

  return performTransformations(resourcesPath, Args);
}
