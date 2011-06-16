//===--- FrontendAction.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendAction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "clang/Serialization/ChainedIncludesSource.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace {

/// \brief Dumps deserialized declarations.
class DeserializedDeclsDumper : public ASTDeserializationListener {
  ASTDeserializationListener *Previous;

public:
  DeserializedDeclsDumper(ASTDeserializationListener *Previous)
    : Previous(Previous) { }

  virtual void DeclRead(serialization::DeclID ID, const Decl *D) {
    llvm::outs() << "PCH DECL: " << D->getDeclKindName();
    if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
      llvm::outs() << " - " << ND->getNameAsString();
    llvm::outs() << "\n";

    if (Previous)
      Previous->DeclRead(ID, D);
  }
};

  /// \brief Checks deserialized declarations and emits error if a name
  /// matches one given in command-line using -error-on-deserialized-decl.
  class DeserializedDeclsChecker : public ASTDeserializationListener {
    ASTContext &Ctx;
    std::set<std::string> NamesToCheck;
    ASTDeserializationListener *Previous;

  public:
    DeserializedDeclsChecker(ASTContext &Ctx,
                             const std::set<std::string> &NamesToCheck, 
                             ASTDeserializationListener *Previous)
      : Ctx(Ctx), NamesToCheck(NamesToCheck), Previous(Previous) { }

    virtual void DeclRead(serialization::DeclID ID, const Decl *D) {
      if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
        if (NamesToCheck.find(ND->getNameAsString()) != NamesToCheck.end()) {
          unsigned DiagID
            = Ctx.getDiagnostics().getCustomDiagID(Diagnostic::Error,
                                                   "%0 was deserialized");
          Ctx.getDiagnostics().Report(Ctx.getFullLoc(D->getLocation()), DiagID)
              << ND->getNameAsString();
        }

      if (Previous)
        Previous->DeclRead(ID, D);
    }
};

} // end anonymous namespace

FrontendAction::FrontendAction() : Instance(0) {}

FrontendAction::~FrontendAction() {}

void FrontendAction::setCurrentFile(llvm::StringRef Value, InputKind Kind,
                                    ASTUnit *AST) {
  CurrentFile = Value;
  CurrentFileKind = Kind;
  CurrentASTUnit.reset(AST);
}

ASTConsumer* FrontendAction::CreateWrappedASTConsumer(CompilerInstance &CI,
                                                      llvm::StringRef InFile) {
  ASTConsumer* Consumer = CreateASTConsumer(CI, InFile);
  if (!Consumer)
    return 0;

  if (CI.getFrontendOpts().AddPluginActions.size() == 0)
    return Consumer;

  // Make sure the non-plugin consumer is first, so that plugins can't
  // modifiy the AST.
  std::vector<ASTConsumer*> Consumers(1, Consumer);

  for (size_t i = 0, e = CI.getFrontendOpts().AddPluginActions.size();
       i != e; ++i) { 
    // This is O(|plugins| * |add_plugins|), but since both numbers are
    // way below 50 in practice, that's ok.
    for (FrontendPluginRegistry::iterator
        it = FrontendPluginRegistry::begin(),
        ie = FrontendPluginRegistry::end();
        it != ie; ++it) {
      if (it->getName() == CI.getFrontendOpts().AddPluginActions[i]) {
        llvm::OwningPtr<PluginASTAction> P(it->instantiate());
        FrontendAction* c = P.get();
        if (P->ParseArgs(CI, CI.getFrontendOpts().AddPluginArgs[i]))
          Consumers.push_back(c->CreateASTConsumer(CI, InFile));
      }
    }
  }

  return new MultiplexConsumer(Consumers);
}

bool FrontendAction::BeginSourceFile(CompilerInstance &CI,
                                     llvm::StringRef Filename,
                                     InputKind InputKind) {
  assert(!Instance && "Already processing a source file!");
  assert(!Filename.empty() && "Unexpected empty filename!");
  setCurrentFile(Filename, InputKind);
  setCompilerInstance(&CI);

  // AST files follow a very different path, since they share objects via the
  // AST unit.
  if (InputKind == IK_AST) {
    assert(!usesPreprocessorOnly() &&
           "Attempt to pass AST file to preprocessor only action!");
    assert(hasASTFileSupport() &&
           "This action does not have AST file support!");

    llvm::IntrusiveRefCntPtr<Diagnostic> Diags(&CI.getDiagnostics());
    std::string Error;
    ASTUnit *AST = ASTUnit::LoadFromASTFile(Filename, Diags,
                                            CI.getFileSystemOpts());
    if (!AST)
      goto failure;

    setCurrentFile(Filename, InputKind, AST);

    // Set the shared objects, these are reset when we finish processing the
    // file, otherwise the CompilerInstance will happily destroy them.
    CI.setFileManager(&AST->getFileManager());
    CI.setSourceManager(&AST->getSourceManager());
    CI.setPreprocessor(&AST->getPreprocessor());
    CI.setASTContext(&AST->getASTContext());

    // Initialize the action.
    if (!BeginSourceFileAction(CI, Filename))
      goto failure;

    /// Create the AST consumer.
    CI.setASTConsumer(CreateWrappedASTConsumer(CI, Filename));
    if (!CI.hasASTConsumer())
      goto failure;

    return true;
  }

  // Set up the file and source managers, if needed.
  if (!CI.hasFileManager())
    CI.createFileManager();
  if (!CI.hasSourceManager())
    CI.createSourceManager(CI.getFileManager());

  // IR files bypass the rest of initialization.
  if (InputKind == IK_LLVM_IR) {
    assert(hasIRSupport() &&
           "This action does not have IR file support!");

    // Inform the diagnostic client we are processing a source file.
    CI.getDiagnosticClient().BeginSourceFile(CI.getLangOpts(), 0);

    // Initialize the action.
    if (!BeginSourceFileAction(CI, Filename))
      goto failure;

    return true;
  }

  // Set up the preprocessor.
  CI.createPreprocessor();

  // Inform the diagnostic client we are processing a source file.
  CI.getDiagnosticClient().BeginSourceFile(CI.getLangOpts(),
                                           &CI.getPreprocessor());

  // Initialize the action.
  if (!BeginSourceFileAction(CI, Filename))
    goto failure;

  /// Create the AST context and consumer unless this is a preprocessor only
  /// action.
  if (!usesPreprocessorOnly()) {
    CI.createASTContext();

    llvm::OwningPtr<ASTConsumer> Consumer(
        CreateWrappedASTConsumer(CI, Filename));
    if (!Consumer)
      goto failure;

    CI.getASTContext().setASTMutationListener(Consumer->GetASTMutationListener());

    if (!CI.getPreprocessorOpts().ChainedIncludes.empty()) {
      // Convert headers to PCH and chain them.
      llvm::OwningPtr<ExternalASTSource> source;
      source.reset(ChainedIncludesSource::create(CI));
      if (!source)
        goto failure;
      CI.getASTContext().setExternalSource(source);

    } else if (!CI.getPreprocessorOpts().ImplicitPCHInclude.empty()) {
      // Use PCH.
      assert(hasPCHSupport() && "This action does not have PCH support!");
      ASTDeserializationListener *DeserialListener
          = CI.getInvocation().getFrontendOpts().ChainedPCH ?
                  Consumer->GetASTDeserializationListener() : 0;
      if (CI.getPreprocessorOpts().DumpDeserializedPCHDecls)
        DeserialListener = new DeserializedDeclsDumper(DeserialListener);
      if (!CI.getPreprocessorOpts().DeserializedPCHDeclsToErrorOn.empty())
        DeserialListener = new DeserializedDeclsChecker(CI.getASTContext(),
                         CI.getPreprocessorOpts().DeserializedPCHDeclsToErrorOn,
                                                        DeserialListener);
      CI.createPCHExternalASTSource(
                                CI.getPreprocessorOpts().ImplicitPCHInclude,
                                CI.getPreprocessorOpts().DisablePCHValidation,
                                CI.getPreprocessorOpts().DisableStatCache,
                                DeserialListener);
      if (!CI.getASTContext().getExternalSource())
        goto failure;
    }

    CI.setASTConsumer(Consumer.take());
    if (!CI.hasASTConsumer())
      goto failure;
  }

  // Initialize builtin info as long as we aren't using an external AST
  // source.
  if (!CI.hasASTContext() || !CI.getASTContext().getExternalSource()) {
    Preprocessor &PP = CI.getPreprocessor();
    PP.getBuiltinInfo().InitializeBuiltins(PP.getIdentifierTable(),
                                           PP.getLangOptions());
  }

  return true;

  // If we failed, reset state since the client will not end up calling the
  // matching EndSourceFile().
  failure:
  if (isCurrentFileAST()) {
    CI.setASTContext(0);
    CI.setPreprocessor(0);
    CI.setSourceManager(0);
    CI.setFileManager(0);
  }

  CI.getDiagnosticClient().EndSourceFile();
  setCurrentFile("", IK_None);
  setCompilerInstance(0);
  return false;
}

void FrontendAction::Execute() {
  CompilerInstance &CI = getCompilerInstance();

  // Initialize the main file entry. This needs to be delayed until after PCH
  // has loaded.
  if (isCurrentFileAST()) {
    // Set the main file ID to an empty file.
    //
    // FIXME: We probably shouldn't need this, but for now this is the
    // simplest way to reuse the logic in ParseAST.
    const char *EmptyStr = "";
    llvm::MemoryBuffer *SB =
      llvm::MemoryBuffer::getMemBuffer(EmptyStr, "<dummy input>");
    CI.getSourceManager().createMainFileIDForMemBuffer(SB);
  } else {
    if (!CI.InitializeSourceManager(getCurrentFile()))
      return;
  }

  if (CI.hasFrontendTimer()) {
    llvm::TimeRegion Timer(CI.getFrontendTimer());
    ExecuteAction();
  }
  else ExecuteAction();
}

void FrontendAction::EndSourceFile() {
  CompilerInstance &CI = getCompilerInstance();

  // Inform the diagnostic client we are done with this source file.
  CI.getDiagnosticClient().EndSourceFile();

  // Finalize the action.
  EndSourceFileAction();

  // Release the consumer and the AST, in that order since the consumer may
  // perform actions in its destructor which require the context.
  //
  // FIXME: There is more per-file stuff we could just drop here?
  if (CI.getFrontendOpts().DisableFree) {
    CI.takeASTConsumer();
    if (!isCurrentFileAST()) {
      CI.takeSema();
      CI.resetAndLeakASTContext();
    }
  } else {
    if (!isCurrentFileAST()) {
      CI.setSema(0);
      CI.setASTContext(0);
    }
    CI.setASTConsumer(0);
  }

  // Inform the preprocessor we are done.
  if (CI.hasPreprocessor())
    CI.getPreprocessor().EndSourceFile();

  if (CI.getFrontendOpts().ShowStats) {
    llvm::errs() << "\nSTATISTICS FOR '" << getCurrentFile() << "':\n";
    CI.getPreprocessor().PrintStats();
    CI.getPreprocessor().getIdentifierTable().PrintStats();
    CI.getPreprocessor().getHeaderSearchInfo().PrintStats();
    CI.getSourceManager().PrintStats();
    llvm::errs() << "\n";
  }

  // Cleanup the output streams, and erase the output files if we encountered
  // an error.
  CI.clearOutputFiles(/*EraseFiles=*/CI.getDiagnostics().hasErrorOccurred());

  if (isCurrentFileAST()) {
    CI.takeSema();
    CI.resetAndLeakASTContext();
    CI.resetAndLeakPreprocessor();
    CI.resetAndLeakSourceManager();
    CI.resetAndLeakFileManager();
  }

  setCompilerInstance(0);
  setCurrentFile("", IK_None);
}

//===----------------------------------------------------------------------===//
// Utility Actions
//===----------------------------------------------------------------------===//

void ASTFrontendAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();

  // FIXME: Move the truncation aspect of this into Sema, we delayed this till
  // here so the source manager would be initialized.
  if (hasCodeCompletionSupport() &&
      !CI.getFrontendOpts().CodeCompletionAt.FileName.empty())
    CI.createCodeCompletionConsumer();

  // Use a code completion consumer?
  CodeCompleteConsumer *CompletionConsumer = 0;
  if (CI.hasCodeCompletionConsumer())
    CompletionConsumer = &CI.getCodeCompletionConsumer();

  if (!CI.hasSema())
    CI.createSema(usesCompleteTranslationUnit(), CompletionConsumer);

  ParseAST(CI.getSema(), CI.getFrontendOpts().ShowStats);
}

ASTConsumer *
PreprocessorFrontendAction::CreateASTConsumer(CompilerInstance &CI,
                                              llvm::StringRef InFile) {
  llvm_unreachable("Invalid CreateASTConsumer on preprocessor action!");
}

ASTConsumer *WrapperFrontendAction::CreateASTConsumer(CompilerInstance &CI,
                                                      llvm::StringRef InFile) {
  return WrappedAction->CreateASTConsumer(CI, InFile);
}
bool WrapperFrontendAction::BeginSourceFileAction(CompilerInstance &CI,
                                                  llvm::StringRef Filename) {
  return WrappedAction->BeginSourceFileAction(CI, Filename);
}
void WrapperFrontendAction::ExecuteAction() {
  WrappedAction->ExecuteAction();
}
void WrapperFrontendAction::EndSourceFileAction() {
  WrappedAction->EndSourceFileAction();
}

bool WrapperFrontendAction::usesPreprocessorOnly() const {
  return WrappedAction->usesPreprocessorOnly();
}
bool WrapperFrontendAction::usesCompleteTranslationUnit() {
  return WrappedAction->usesCompleteTranslationUnit();
}
bool WrapperFrontendAction::hasPCHSupport() const {
  return WrappedAction->hasPCHSupport();
}
bool WrapperFrontendAction::hasASTFileSupport() const {
  return WrappedAction->hasASTFileSupport();
}
bool WrapperFrontendAction::hasIRSupport() const {
  return WrappedAction->hasIRSupport();
}
bool WrapperFrontendAction::hasCodeCompletionSupport() const {
  return WrappedAction->hasCodeCompletionSupport();
}

WrapperFrontendAction::WrapperFrontendAction(FrontendAction *WrappedAction)
  : WrappedAction(WrappedAction) {}

