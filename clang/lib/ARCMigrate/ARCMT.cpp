//===--- ARCMT.cpp - Migration to ARC mode --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Internals.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Basic/DiagnosticCategories.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/Triple.h"
using namespace clang;
using namespace arcmt;

bool CapturedDiagList::clearDiagnostic(ArrayRef<unsigned> IDs,
                                       SourceRange range) {
  if (range.isInvalid())
    return false;

  bool cleared = false;
  ListTy::iterator I = List.begin();
  while (I != List.end()) {
    FullSourceLoc diagLoc = I->getLocation();
    if ((IDs.empty() || // empty means clear all diagnostics in the range.
         std::find(IDs.begin(), IDs.end(), I->getID()) != IDs.end()) &&
        !diagLoc.isBeforeInTranslationUnitThan(range.getBegin()) &&
        (diagLoc == range.getEnd() ||
           diagLoc.isBeforeInTranslationUnitThan(range.getEnd()))) {
      cleared = true;
      ListTy::iterator eraseS = I++;
      while (I != List.end() && I->getLevel() == Diagnostic::Note)
        ++I;
      // Clear the diagnostic and any notes following it.
      List.erase(eraseS, I);
      continue;
    }

    ++I;
  }

  return cleared;
}

bool CapturedDiagList::hasDiagnostic(ArrayRef<unsigned> IDs,
                                     SourceRange range) const {
  if (range.isInvalid())
    return false;

  ListTy::const_iterator I = List.begin();
  while (I != List.end()) {
    FullSourceLoc diagLoc = I->getLocation();
    if ((IDs.empty() || // empty means any diagnostic in the range.
         std::find(IDs.begin(), IDs.end(), I->getID()) != IDs.end()) &&
        !diagLoc.isBeforeInTranslationUnitThan(range.getBegin()) &&
        (diagLoc == range.getEnd() ||
           diagLoc.isBeforeInTranslationUnitThan(range.getEnd()))) {
      return true;
    }

    ++I;
  }

  return false;
}

void CapturedDiagList::reportDiagnostics(Diagnostic &Diags) const {
  for (ListTy::const_iterator I = List.begin(), E = List.end(); I != E; ++I)
    Diags.Report(*I);
}

bool CapturedDiagList::hasErrors() const {
  for (ListTy::const_iterator I = List.begin(), E = List.end(); I != E; ++I)
    if (I->getLevel() >= Diagnostic::Error)
      return true;

  return false;
}

namespace {

class CaptureDiagnosticClient : public DiagnosticClient {
  Diagnostic &Diags;
  CapturedDiagList &CapturedDiags;
public:
  CaptureDiagnosticClient(Diagnostic &diags,
                          CapturedDiagList &capturedDiags)
    : Diags(diags), CapturedDiags(capturedDiags) { }

  virtual void HandleDiagnostic(Diagnostic::Level level,
                                const DiagnosticInfo &Info) {
    if (arcmt::isARCDiagnostic(Info.getID(), Diags) ||
        level >= Diagnostic::Error || level == Diagnostic::Note) {
      CapturedDiags.push_back(StoredDiagnostic(level, Info));
      return;
    }

    // Non-ARC warnings are ignored.
    Diags.setLastDiagnosticIgnored();
  }
};

} // end anonymous namespace

static inline StringRef SimulatorVersionDefineName() {
  return "__IPHONE_OS_VERSION_MIN_REQUIRED=";
}

/// \brief Parse the simulator version define:
/// __IPHONE_OS_VERSION_MIN_REQUIRED=([0-9])([0-9][0-9])([0-9][0-9])
// and return the grouped values as integers, e.g:
//   __IPHONE_OS_VERSION_MIN_REQUIRED=40201
// will return Major=4, Minor=2, Micro=1.
static bool GetVersionFromSimulatorDefine(StringRef define,
                                          unsigned &Major, unsigned &Minor,
                                          unsigned &Micro) {
  assert(define.startswith(SimulatorVersionDefineName()));
  StringRef name, version;
  llvm::tie(name, version) = define.split('=');
  if (version.empty())
    return false;
  std::string verstr = version.str();
  char *end;
  unsigned num = (unsigned) strtol(verstr.c_str(), &end, 10);
  if (*end != '\0')
    return false;
  Major = num / 10000;
  num = num % 10000;
  Minor = num / 100;
  Micro = num % 100;
  return true;
}

static bool HasARCRuntime(CompilerInvocation &origCI) {
  // This duplicates some functionality from Darwin::AddDeploymentTarget
  // but this function is well defined, so keep it decoupled from the driver
  // and avoid unrelated complications.

  for (unsigned i = 0, e = origCI.getPreprocessorOpts().Macros.size();
         i != e; ++i) {
    StringRef define = origCI.getPreprocessorOpts().Macros[i].first;
    bool isUndef = origCI.getPreprocessorOpts().Macros[i].second;
    if (isUndef)
      continue;
    if (!define.startswith(SimulatorVersionDefineName()))
      continue;
    unsigned Major, Minor, Micro;
    if (GetVersionFromSimulatorDefine(define, Major, Minor, Micro) &&
        Major < 10 && Minor < 100 && Micro < 100)
      return Major >= 5;
  }

  llvm::Triple triple(origCI.getTargetOpts().Triple);

  if (triple.getOS() == llvm::Triple::IOS)
    return triple.getOSMajorVersion() >= 5;

  if (triple.getOS() == llvm::Triple::Darwin)
    return triple.getOSMajorVersion() >= 11;

  if (triple.getOS() == llvm::Triple::MacOSX) {
    unsigned Major, Minor, Micro;
    triple.getOSVersion(Major, Minor, Micro);
    return Major > 10 || (Major == 10 && Minor >= 7);
  }

  return false;
}

CompilerInvocation *createInvocationForMigration(CompilerInvocation &origCI) {
  llvm::OwningPtr<CompilerInvocation> CInvok;
  CInvok.reset(new CompilerInvocation(origCI));
  CInvok->getPreprocessorOpts().ImplicitPCHInclude = std::string();
  CInvok->getPreprocessorOpts().ImplicitPTHInclude = std::string();
  std::string define = getARCMTMacroName();
  define += '=';
  CInvok->getPreprocessorOpts().addMacroDef(define);
  CInvok->getLangOpts().ObjCAutoRefCount = true;
  CInvok->getDiagnosticOpts().ErrorLimit = 0;
  CInvok->getDiagnosticOpts().Warnings.push_back(
                                            "error=arc-unsafe-retained-assign");
  CInvok->getLangOpts().ObjCRuntimeHasWeak = HasARCRuntime(origCI);

  return CInvok.take();
}

void emitPremigrationErrors(const CapturedDiagList &arcDiags,
                            const DiagnosticOptions &diagOpts,
                            Preprocessor &PP) {
  TextDiagnosticPrinter printer(llvm::errs(), diagOpts);
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags(
                   new Diagnostic(DiagID, &printer, /*ShouldOwnClient=*/false));
  Diags->setSourceManager(&PP.getSourceManager());
  
  printer.BeginSourceFile(PP.getLangOptions(), &PP);
  arcDiags.reportDiagnostics(*Diags);
  printer.EndSourceFile();
}

//===----------------------------------------------------------------------===//
// checkForManualIssues.
//===----------------------------------------------------------------------===//

bool arcmt::checkForManualIssues(CompilerInvocation &origCI,
                                 StringRef Filename, InputKind Kind,
                                 DiagnosticClient *DiagClient,
                                 bool emitPremigrationARCErrors,
                                 StringRef plistOut) {
  if (!origCI.getLangOpts().ObjC1)
    return false;

  std::vector<TransformFn> transforms = arcmt::getAllTransformations();
  assert(!transforms.empty());

  llvm::OwningPtr<CompilerInvocation> CInvok;
  CInvok.reset(createInvocationForMigration(origCI));
  CInvok->getFrontendOpts().Inputs.clear();
  CInvok->getFrontendOpts().Inputs.push_back(std::make_pair(Kind, Filename));

  CapturedDiagList capturedDiags;

  assert(DiagClient);
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags(
                 new Diagnostic(DiagID, DiagClient, /*ShouldOwnClient=*/false));

  // Filter of all diagnostics.
  CaptureDiagnosticClient errRec(*Diags, capturedDiags);
  Diags->setClient(&errRec, /*ShouldOwnClient=*/false);

  llvm::OwningPtr<ASTUnit> Unit(
      ASTUnit::LoadFromCompilerInvocationAction(CInvok.take(), Diags));
  if (!Unit)
    return true;

  // Don't filter diagnostics anymore.
  Diags->setClient(DiagClient, /*ShouldOwnClient=*/false);

  ASTContext &Ctx = Unit->getASTContext();

  if (Diags->hasFatalErrorOccurred()) {
    Diags->Reset();
    DiagClient->BeginSourceFile(Ctx.getLangOptions(), &Unit->getPreprocessor());
    capturedDiags.reportDiagnostics(*Diags);
    DiagClient->EndSourceFile();
    return true;
  }

  if (emitPremigrationARCErrors)
    emitPremigrationErrors(capturedDiags, origCI.getDiagnosticOpts(),
                           Unit->getPreprocessor());
  if (!plistOut.empty()) {
    SmallVector<StoredDiagnostic, 8> arcDiags;
    for (CapturedDiagList::iterator
           I = capturedDiags.begin(), E = capturedDiags.end(); I != E; ++I)
      arcDiags.push_back(*I);
    writeARCDiagsToPlist(plistOut, arcDiags,
                         Ctx.getSourceManager(), Ctx.getLangOptions());
  }

  // After parsing of source files ended, we want to reuse the
  // diagnostics objects to emit further diagnostics.
  // We call BeginSourceFile because DiagnosticClient requires that 
  // diagnostics with source range information are emitted only in between
  // BeginSourceFile() and EndSourceFile().
  DiagClient->BeginSourceFile(Ctx.getLangOptions(), &Unit->getPreprocessor());

  // No macros will be added since we are just checking and we won't modify
  // source code.
  std::vector<SourceLocation> ARCMTMacroLocs;

  TransformActions testAct(*Diags, capturedDiags, Ctx, Unit->getPreprocessor());
  MigrationPass pass(Ctx, Unit->getSema(), testAct, ARCMTMacroLocs);

  for (unsigned i=0, e = transforms.size(); i != e; ++i)
    transforms[i](pass);

  capturedDiags.reportDiagnostics(*Diags);

  DiagClient->EndSourceFile();

  // If we are migrating code that gets the '-fobjc-arc' flag, make sure
  // to remove it so that we don't get errors from normal compilation.
  origCI.getLangOpts().ObjCAutoRefCount = false;

  return capturedDiags.hasErrors() || testAct.hasReportedErrors();
}

//===----------------------------------------------------------------------===//
// applyTransformations.
//===----------------------------------------------------------------------===//

static bool applyTransforms(CompilerInvocation &origCI,
                            StringRef Filename, InputKind Kind,
                            DiagnosticClient *DiagClient,
                            StringRef outputDir,
                            bool emitPremigrationARCErrors,
                            StringRef plistOut) {
  if (!origCI.getLangOpts().ObjC1)
    return false;

  // Make sure checking is successful first.
  CompilerInvocation CInvokForCheck(origCI);
  if (arcmt::checkForManualIssues(CInvokForCheck, Filename, Kind, DiagClient,
                                  emitPremigrationARCErrors, plistOut))
    return true;

  CompilerInvocation CInvok(origCI);
  CInvok.getFrontendOpts().Inputs.clear();
  CInvok.getFrontendOpts().Inputs.push_back(std::make_pair(Kind, Filename));
  
  MigrationProcess migration(CInvok, DiagClient, outputDir);

  std::vector<TransformFn> transforms = arcmt::getAllTransformations();
  assert(!transforms.empty());

  for (unsigned i=0, e = transforms.size(); i != e; ++i) {
    bool err = migration.applyTransform(transforms[i]);
    if (err) return true;
  }

  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags(
                 new Diagnostic(DiagID, DiagClient, /*ShouldOwnClient=*/false));

  if (outputDir.empty()) {
    origCI.getLangOpts().ObjCAutoRefCount = true;
    return migration.getRemapper().overwriteOriginal(*Diags);
  } else {
    // If we are migrating code that gets the '-fobjc-arc' flag, make sure
    // to remove it so that we don't get errors from normal compilation.
    origCI.getLangOpts().ObjCAutoRefCount = false;
    return migration.getRemapper().flushToDisk(outputDir, *Diags);
  }
}

bool arcmt::applyTransformations(CompilerInvocation &origCI,
                                 StringRef Filename, InputKind Kind,
                                 DiagnosticClient *DiagClient) {
  return applyTransforms(origCI, Filename, Kind, DiagClient,
                         StringRef(), false, StringRef());
}

bool arcmt::migrateWithTemporaryFiles(CompilerInvocation &origCI,
                                      StringRef Filename, InputKind Kind,
                                      DiagnosticClient *DiagClient,
                                      StringRef outputDir,
                                      bool emitPremigrationARCErrors,
                                      StringRef plistOut) {
  assert(!outputDir.empty() && "Expected output directory path");
  return applyTransforms(origCI, Filename, Kind, DiagClient,
                         outputDir, emitPremigrationARCErrors, plistOut);
}

bool arcmt::getFileRemappings(std::vector<std::pair<std::string,std::string> > &
                                  remap,
                              StringRef outputDir,
                              DiagnosticClient *DiagClient) {
  assert(!outputDir.empty());

  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags(
                 new Diagnostic(DiagID, DiagClient, /*ShouldOwnClient=*/false));

  FileRemapper remapper;
  bool err = remapper.initFromDisk(outputDir, *Diags,
                                   /*ignoreIfFilesChanged=*/true);
  if (err)
    return true;

  CompilerInvocation CI;
  remapper.applyMappings(CI);
  remap = CI.getPreprocessorOpts().RemappedFiles;

  return false;
}

//===----------------------------------------------------------------------===//
// CollectTransformActions.
//===----------------------------------------------------------------------===//

namespace {

class ARCMTMacroTrackerPPCallbacks : public PPCallbacks {
  std::vector<SourceLocation> &ARCMTMacroLocs;

public:
  ARCMTMacroTrackerPPCallbacks(std::vector<SourceLocation> &ARCMTMacroLocs)
    : ARCMTMacroLocs(ARCMTMacroLocs) { }

  virtual void MacroExpands(const Token &MacroNameTok, const MacroInfo *MI) {
    if (MacroNameTok.getIdentifierInfo()->getName() == getARCMTMacroName())
      ARCMTMacroLocs.push_back(MacroNameTok.getLocation());
  }
};

class ARCMTMacroTrackerAction : public ASTFrontendAction {
  std::vector<SourceLocation> &ARCMTMacroLocs;

public:
  ARCMTMacroTrackerAction(std::vector<SourceLocation> &ARCMTMacroLocs)
    : ARCMTMacroLocs(ARCMTMacroLocs) { }

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile) {
    CI.getPreprocessor().addPPCallbacks(
                              new ARCMTMacroTrackerPPCallbacks(ARCMTMacroLocs));
    return new ASTConsumer();
  }
};

class RewritesApplicator : public TransformActions::RewriteReceiver {
  Rewriter &rewriter;
  ASTContext &Ctx;
  MigrationProcess::RewriteListener *Listener;

public:
  RewritesApplicator(Rewriter &rewriter, ASTContext &ctx,
                     MigrationProcess::RewriteListener *listener)
    : rewriter(rewriter), Ctx(ctx), Listener(listener) {
    if (Listener)
      Listener->start(ctx);
  }
  ~RewritesApplicator() {
    if (Listener)
      Listener->finish();
  }

  virtual void insert(SourceLocation loc, StringRef text) {
    bool err = rewriter.InsertText(loc, text, /*InsertAfter=*/true,
                                   /*indentNewLines=*/true);
    if (!err && Listener)
      Listener->insert(loc, text);
  }

  virtual void remove(CharSourceRange range) {
    Rewriter::RewriteOptions removeOpts;
    removeOpts.IncludeInsertsAtBeginOfRange = false;
    removeOpts.IncludeInsertsAtEndOfRange = false;
    removeOpts.RemoveLineIfEmpty = true;

    bool err = rewriter.RemoveText(range, removeOpts);
    if (!err && Listener)
      Listener->remove(range);
  }

  virtual void increaseIndentation(CharSourceRange range,
                                    SourceLocation parentIndent) {
    rewriter.IncreaseIndentation(range, parentIndent);
  }
};

} // end anonymous namespace.

/// \brief Anchor for VTable.
MigrationProcess::RewriteListener::~RewriteListener() { }

MigrationProcess::MigrationProcess(const CompilerInvocation &CI,
                                   DiagnosticClient *diagClient,
                                   StringRef outputDir)
  : OrigCI(CI), DiagClient(diagClient) {
  if (!outputDir.empty()) {
    llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
    llvm::IntrusiveRefCntPtr<Diagnostic> Diags(
                 new Diagnostic(DiagID, DiagClient, /*ShouldOwnClient=*/false));
    Remapper.initFromDisk(outputDir, *Diags, /*ignoreIfFilesChanges=*/true);
  }
}

bool MigrationProcess::applyTransform(TransformFn trans,
                                      RewriteListener *listener) {
  llvm::OwningPtr<CompilerInvocation> CInvok;
  CInvok.reset(createInvocationForMigration(OrigCI));
  CInvok->getDiagnosticOpts().IgnoreWarnings = true;

  Remapper.applyMappings(*CInvok);

  CapturedDiagList capturedDiags;
  std::vector<SourceLocation> ARCMTMacroLocs;

  assert(DiagClient);
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags(
               new Diagnostic(DiagID, DiagClient, /*ShouldOwnClient=*/false));

  // Filter of all diagnostics.
  CaptureDiagnosticClient errRec(*Diags, capturedDiags);
  Diags->setClient(&errRec, /*ShouldOwnClient=*/false);

  llvm::OwningPtr<ARCMTMacroTrackerAction> ASTAction;
  ASTAction.reset(new ARCMTMacroTrackerAction(ARCMTMacroLocs));

  llvm::OwningPtr<ASTUnit> Unit(
      ASTUnit::LoadFromCompilerInvocationAction(CInvok.take(), Diags,
                                                ASTAction.get()));
  if (!Unit)
    return true;
  Unit->setOwnsRemappedFileBuffers(false); // FileRemapper manages that.

  // Don't filter diagnostics anymore.
  Diags->setClient(DiagClient, /*ShouldOwnClient=*/false);

  ASTContext &Ctx = Unit->getASTContext();

  if (Diags->hasFatalErrorOccurred()) {
    Diags->Reset();
    DiagClient->BeginSourceFile(Ctx.getLangOptions(), &Unit->getPreprocessor());
    capturedDiags.reportDiagnostics(*Diags);
    DiagClient->EndSourceFile();
    return true;
  }

  // After parsing of source files ended, we want to reuse the
  // diagnostics objects to emit further diagnostics.
  // We call BeginSourceFile because DiagnosticClient requires that 
  // diagnostics with source range information are emitted only in between
  // BeginSourceFile() and EndSourceFile().
  DiagClient->BeginSourceFile(Ctx.getLangOptions(), &Unit->getPreprocessor());

  Rewriter rewriter(Ctx.getSourceManager(), Ctx.getLangOptions());
  TransformActions TA(*Diags, capturedDiags, Ctx, Unit->getPreprocessor());
  MigrationPass pass(Ctx, Unit->getSema(), TA, ARCMTMacroLocs);

  trans(pass);

  {
    RewritesApplicator applicator(rewriter, Ctx, listener);
    TA.applyRewrites(applicator);
  }

  DiagClient->EndSourceFile();

  if (DiagClient->getNumErrors())
    return true;

  for (Rewriter::buffer_iterator
        I = rewriter.buffer_begin(), E = rewriter.buffer_end(); I != E; ++I) {
    FileID FID = I->first;
    RewriteBuffer &buf = I->second;
    const FileEntry *file = Ctx.getSourceManager().getFileEntryForID(FID);
    assert(file);
    std::string newFname = file->getName();
    newFname += "-trans";
    llvm::SmallString<512> newText;
    llvm::raw_svector_ostream vecOS(newText);
    buf.write(vecOS);
    vecOS.flush();
    llvm::MemoryBuffer *memBuf = llvm::MemoryBuffer::getMemBufferCopy(
                   StringRef(newText.data(), newText.size()), newFname);
    llvm::SmallString<64> filePath(file->getName());
    Unit->getFileManager().FixupRelativePath(filePath);
    Remapper.remap(filePath.str(), memBuf);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// isARCDiagnostic.
//===----------------------------------------------------------------------===//

bool arcmt::isARCDiagnostic(unsigned diagID, Diagnostic &Diag) {
  return Diag.getDiagnosticIDs()->getCategoryNumberForDiag(diagID) ==
           diag::DiagCat_Automatic_Reference_Counting_Issue;
}
