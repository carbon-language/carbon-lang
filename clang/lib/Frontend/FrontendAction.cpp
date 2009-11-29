//===--- FrontendAction.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendAction.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Sema/ParseAST.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

FrontendAction::FrontendAction() : Instance(0) {}

FrontendAction::~FrontendAction() {}

void FrontendAction::setCurrentFile(llvm::StringRef Value, ASTUnit *AST) {
  CurrentFile = Value;
  CurrentASTUnit.reset(AST);
}

bool FrontendAction::BeginSourceFile(CompilerInstance &CI,
                                     llvm::StringRef Filename,
                                     bool IsAST) {
  assert(!Instance && "Already processing a source file!");
  assert(!Filename.empty() && "Unexpected empty filename!");
  setCurrentFile(Filename);
  setCompilerInstance(&CI);

  // AST files follow a very different path, since they share objects via the
  // AST unit.
  if (IsAST) {
    assert(!usesPreprocessorOnly() &&
           "Attempt to pass AST file to preprocessor only action!");
    assert(hasASTSupport() && "This action does not have AST support!");

    std::string Error;
    ASTUnit *AST = ASTUnit::LoadFromPCHFile(Filename, &Error);
    if (!AST) {
      CI.getDiagnostics().Report(diag::err_fe_invalid_ast_file) << Error;
      goto failure;
    }

    setCurrentFile(Filename, AST);

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
    CI.setASTConsumer(CreateASTConsumer(CI, Filename));
    if (!CI.hasASTConsumer())
      goto failure;

    /// Use PCH?
    if (!CI.getPreprocessorOpts().ImplicitPCHInclude.empty()) {
      assert(hasPCHSupport() && "This action does not have PCH support!");
      CI.createPCHExternalASTSource(
        CI.getPreprocessorOpts().ImplicitPCHInclude);
      if (!CI.getASTContext().getExternalSource())
        goto failure;
    }
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
  setCurrentFile("");
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
      llvm::MemoryBuffer::getMemBuffer(EmptyStr, EmptyStr, "<dummy input>");
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
    if (!isCurrentFileAST())
      CI.takeASTContext();
  } else {
    CI.setASTConsumer(0);
    if (!isCurrentFileAST())
      CI.setASTContext(0);
  }

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
  CI.ClearOutputFiles(/*EraseFiles=*/CI.getDiagnostics().getNumErrors());

  // Inform the diagnostic client we are done with this source file.
  CI.getDiagnosticClient().EndSourceFile();

  if (isCurrentFileAST()) {
    CI.takeASTContext();
    CI.takePreprocessor();
    CI.takeSourceManager();
    CI.takeFileManager();
  }

  setCompilerInstance(0);
  setCurrentFile("");
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

  ParseAST(CI.getPreprocessor(), &CI.getASTConsumer(), CI.getASTContext(),
           CI.getFrontendOpts().ShowStats,
           usesCompleteTranslationUnit(), CompletionConsumer);
}

ASTConsumer *
PreprocessorFrontendAction::CreateASTConsumer(CompilerInstance &CI,
                                              llvm::StringRef InFile) {
  llvm::llvm_unreachable("Invalid CreateASTConsumer on preprocessor action!");
}
