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
using llvm::StringRef;

bool CapturedDiagList::clearDiagnostic(llvm::ArrayRef<unsigned> IDs,
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

bool CapturedDiagList::hasDiagnostic(llvm::ArrayRef<unsigned> IDs,
                                     SourceRange range) {
  if (range.isInvalid())
    return false;

  ListTy::iterator I = List.begin();
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

void CapturedDiagList::reportDiagnostics(Diagnostic &Diags) {
  for (ListTy::iterator I = List.begin(), E = List.end(); I != E; ++I)
    Diags.Report(*I);
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
  
  // FIXME: Hackety hack! Try to find out if there is an ARC runtime.
  bool hasARCRuntime = false;
  llvm::SmallVector<std::string, 16> args;
  args.push_back("-x");
  args.push_back("objective-c");
  args.push_back("-fobjc-arc");

  llvm::Triple triple(CInvok->getTargetOpts().Triple);
  if (triple.getOS() == llvm::Triple::IOS ||
      triple.getOS() == llvm::Triple::MacOSX) {
    args.push_back("-ccc-host-triple");
    std::string forcedTriple = triple.getArchName();
    forcedTriple += "-apple-darwin10";
    args.push_back(forcedTriple);

    unsigned Major, Minor, Micro;
    triple.getOSVersion(Major, Minor, Micro);
    llvm::SmallString<100> flag;
    if (triple.getOS() == llvm::Triple::IOS)
      flag += "-miphoneos-version-min=";
    else
      flag += "-mmacosx-version-min=";
    llvm::raw_svector_ostream(flag) << Major << '.' << Minor << '.' << Micro;
    args.push_back(flag.str());
  }

  args.push_back(origCI.getFrontendOpts().Inputs[0].second.c_str());
  // Also push all defines to deal with the iOS simulator hack.
  for (unsigned i = 0, e = origCI.getPreprocessorOpts().Macros.size();
         i != e; ++i) {
    std::string &def = origCI.getPreprocessorOpts().Macros[i].first;
    bool isUndef = origCI.getPreprocessorOpts().Macros[i].second;
    if (!isUndef) {
      std::string newdef = "-D";
      newdef += def;
      args.push_back(newdef);
    }
  }

  llvm::SmallVector<const char *, 16> cargs;
  for (unsigned i = 0, e = args.size(); i != e; ++i)
    cargs.push_back(args[i].c_str());

  llvm::OwningPtr<CompilerInvocation> checkCI;
  checkCI.reset(clang::createInvocationFromCommandLine(cargs));
  if (checkCI)
    hasARCRuntime = !checkCI->getLangOpts().ObjCNoAutoRefCountRuntime;

  CInvok->getLangOpts().ObjCNoAutoRefCountRuntime = !hasARCRuntime;

  return CInvok.take();
}

//===----------------------------------------------------------------------===//
// checkForManualIssues.
//===----------------------------------------------------------------------===//

bool arcmt::checkForManualIssues(CompilerInvocation &origCI,
                                 llvm::StringRef Filename, InputKind Kind,
                                 DiagnosticClient *DiagClient) {
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

  return Diags->getClient()->getNumErrors() > 0;
}

//===----------------------------------------------------------------------===//
// applyTransformations.
//===----------------------------------------------------------------------===//

bool arcmt::applyTransformations(CompilerInvocation &origCI,
                                 llvm::StringRef Filename, InputKind Kind,
                                 DiagnosticClient *DiagClient) {
  if (!origCI.getLangOpts().ObjC1)
    return false;

  // Make sure checking is successful first.
  CompilerInvocation CInvokForCheck(origCI);
  if (arcmt::checkForManualIssues(CInvokForCheck, Filename, Kind, DiagClient))
    return true;

  CompilerInvocation CInvok(origCI);
  CInvok.getFrontendOpts().Inputs.clear();
  CInvok.getFrontendOpts().Inputs.push_back(std::make_pair(Kind, Filename));
  
  MigrationProcess migration(CInvok, DiagClient);

  std::vector<TransformFn> transforms = arcmt::getAllTransformations();
  assert(!transforms.empty());

  for (unsigned i=0, e = transforms.size(); i != e; ++i) {
    bool err = migration.applyTransform(transforms[i]);
    if (err) return true;
  }

  origCI.getLangOpts().ObjCAutoRefCount = true;

  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags(
                 new Diagnostic(DiagID, DiagClient, /*ShouldOwnClient=*/false));
  return migration.getRemapper().overwriteOriginal(*Diags);
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
                                         llvm::StringRef InFile) {
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

  virtual void insert(SourceLocation loc, llvm::StringRef text) {
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
                   llvm::StringRef(newText.data(), newText.size()), newFname);
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
