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
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Parse/ParseAST.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

FrontendAction::FrontendAction() : Instance(0) {}

FrontendAction::~FrontendAction() {}

void FrontendAction::setCurrentFile(llvm::StringRef Value, InputKind Kind,
                                    ASTUnit *AST) {
  CurrentFile = Value;
  CurrentFileKind = Kind;
  CurrentASTUnit.reset(AST);
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
    ASTUnit *AST = ASTUnit::LoadFromASTFile(Filename, Diags);
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
    CI.setASTConsumer(CreateASTConsumer(CI, Filename));
    if (!CI.hasASTConsumer())
      goto failure;

    return true;
  }

  // Set up the file and source managers, if needed.
  if (!CI.hasFileManager())
    CI.createFileManager();
  if (!CI.hasSourceManager())
    CI.createSourceManager();

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

    llvm::OwningPtr<ASTConsumer> Consumer(CreateASTConsumer(CI, Filename));

    /// Use PCH?
    if (!CI.getPreprocessorOpts().ImplicitPCHInclude.empty()) {
      assert(hasPCHSupport() && "This action does not have PCH support!");
      CI.createPCHExternalASTSource(
                                CI.getPreprocessorOpts().ImplicitPCHInclude,
                                CI.getPreprocessorOpts().DisablePCHValidation,
                                CI.getInvocation().getFrontendOpts().ChainedPCH?
                                 Consumer->GetASTDeserializationListener() : 0);
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
                                           PP.getLangOptions().NoBuiltin);
  }

  return true;

  // If we failed, reset state since the client will not end up calling the
  // matching EndSourceFile().
  failure:
  if (isCurrentFileAST()) {
    CI.takeASTContext();
    CI.takePreprocessor();
    CI.takeSourceManager();
    CI.takeFileManager();
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
      CI.takeASTContext();
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
  CI.clearOutputFiles(/*EraseFiles=*/CI.getDiagnostics().getNumErrors());

  // Inform the diagnostic client we are done with this source file.
  CI.getDiagnosticClient().EndSourceFile();

  if (isCurrentFileAST()) {
    CI.takeSema();
    CI.takeASTContext();
    CI.takePreprocessor();
    CI.takeSourceManager();
    CI.takeFileManager();
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
