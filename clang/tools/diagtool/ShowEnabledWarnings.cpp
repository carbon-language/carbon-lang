//===- ShowEnabledWarnings - diagtool tool for printing enabled flags -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DiagTool.h"
#include "DiagnosticNames.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/Support/TargetSelect.h"

DEF_DIAGTOOL("show-enabled",
             "Show which warnings are enabled for a given command line",
             ShowEnabledWarnings)

using namespace clang;

namespace {
  struct PrettyDiag {
    StringRef Name;
    StringRef Flag;
    DiagnosticsEngine::Level Level;

    PrettyDiag(StringRef name, StringRef flag, DiagnosticsEngine::Level level)
    : Name(name), Flag(flag), Level(level) {}

    bool operator<(const PrettyDiag &x) const { return Name < x.Name; }
  };
}

static char getFlagForLevel(DiagnosticsEngine::Level Level) {
  switch (Level) {
  case DiagnosticsEngine::Ignored: return ' ';
  case DiagnosticsEngine::Note:    return '-';
  case DiagnosticsEngine::Warning: return 'W';
  case DiagnosticsEngine::Error:   return 'E';
  case DiagnosticsEngine::Fatal:   return 'F';
  }

  llvm_unreachable("Unknown diagnostic level");
}

static CompilerInstance *createCompiler(unsigned int argc, char **argv) {
  // First, build a compiler instance!
  OwningPtr<CompilerInstance> Clang(new CompilerInstance());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagIDs(new DiagnosticIDs());
  
  // Initialize targets first. This may be necessary if any warnings are
  // selectively enabled.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  
  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagIDs, DiagsBuffer);
  
  bool Success;
  Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(),
                                               argv, argv+argc, Diags);
  
  // We are skipping cc1's usual search for a resource directory.
  // This is not likely to affect warning output.
  
  // Create the actual diagnostics engine.
  Clang->createDiagnostics(argc, argv);
  if (!Clang->hasDiagnostics())
    return NULL;
  
  // Flush any errors created when initializing everything. This could happen
  // for invalid command lines, which will probably give non-sensical results.
  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success)
    return NULL;

  return Clang.take();
}

int ShowEnabledWarnings::run(unsigned int argc, char **argv, raw_ostream &Out) {
  // First check our one flag (--flags).
  bool ShouldShowFlags = true;
  if (argc > 0) {
    StringRef FirstArg(*argv);
    if (FirstArg.equals("--no-flags")) {
      ShouldShowFlags = false;
      --argc;
      ++argv;
    } else if (FirstArg.equals("--flags")) {
      ShouldShowFlags = true;
      --argc;
      ++argv;
    }
  }

  // Create the compiler instance.
  OwningPtr<CompilerInstance> Clang(createCompiler(argc, argv));
  if (!Clang)
    return EXIT_FAILURE;

  DiagnosticsEngine &Diags = Clang->getDiagnostics();
  
  // Now we have our diagnostics. Iterate through EVERY diagnostic and see
  // which ones are turned on.
  // FIXME: It would be very nice to print which flags are turning on which
  // diagnostics, but this can be done with a diff.
  std::vector<PrettyDiag> Active;

  for (const diagtool::DiagnosticRecord *I = diagtool::BuiltinDiagnostics,
       *E = I + diagtool::BuiltinDiagnosticsCount; I != E; ++I) {
    unsigned DiagID = I->DiagID;
    
    if (DiagnosticIDs::isBuiltinNote(DiagID))
      continue;
    
    if (!DiagnosticIDs::isBuiltinWarningOrExtension(DiagID))
      continue;

    DiagnosticsEngine::Level DiagLevel =
      Diags.getDiagnosticLevel(DiagID, SourceLocation());
    if (DiagLevel == DiagnosticsEngine::Ignored)
      continue;

    StringRef WarningOpt = DiagnosticIDs::getWarningOptionForDiag(DiagID);
    Active.push_back(PrettyDiag(I->getName(), WarningOpt, DiagLevel));
  }

  std::sort(Active.begin(), Active.end());

  // Print them all out.
  for (std::vector<PrettyDiag>::const_iterator I = Active.begin(),
       E = Active.end(); I != E; ++I) {
    if (ShouldShowFlags)
      Out << getFlagForLevel(I->Level) << "  ";
    Out << I->Name;
    if (!I->Flag.empty())
      Out << " [-W" << I->Flag << "]";
    Out << '\n';
  }

  return EXIT_SUCCESS;
}
