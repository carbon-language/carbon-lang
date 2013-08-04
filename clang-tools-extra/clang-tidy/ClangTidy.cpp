//===--- tools/extra/clang-tidy/ClangTidy.cpp - Clang tidy tool -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file This file implements a clang-tidy tool.
///
///  This tool uses the Clang Tooling infrastructure, see
///    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

#include "ClangTidy.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyModuleRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include <vector>

using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

namespace clang {
namespace tidy {
namespace {

class ClangTidyPPAction : public PreprocessOnlyAction {
public:
  ClangTidyPPAction(SmallVectorImpl<ClangTidyCheck *> &Checks,
                    ClangTidyContext &Context)
      : Checks(Checks), Context(Context) {}

private:
  virtual bool BeginSourceFileAction(CompilerInstance &Compiler,
                                     llvm::StringRef file_name) {
    Context.setSourceManager(&Compiler.getSourceManager());
    for (SmallVectorImpl<ClangTidyCheck *>::iterator I = Checks.begin(),
                                                     E = Checks.end();
         I != E; ++I)
      (*I)->registerPPCallbacks(Compiler);
    return true;
  }

  SmallVectorImpl<ClangTidyCheck *> &Checks;
  ClangTidyContext &Context;
};

class ClangTidyPPActionFactory : public FrontendActionFactory {
public:
  ClangTidyPPActionFactory(SmallVectorImpl<ClangTidyCheck *> &Checks,
                           ClangTidyContext &Context)
      : Checks(Checks), Context(Context) {}

  virtual FrontendAction *create() {
    return new ClangTidyPPAction(Checks, Context);
  }

private:
  SmallVectorImpl<ClangTidyCheck *> &Checks;
  ClangTidyContext &Context;
};

} // namespace

ClangTidyError::ClangTidyError(const SourceManager &Sources, SourceLocation Loc,
                               StringRef Message,
                               const tooling::Replacements &Fix)
    : Message(Message), Fix(Fix) {
  FilePath = Sources.getFilename(Loc);
  FileOffset = Sources.getFileOffset(Loc);
}

DiagnosticBuilder ClangTidyContext::Diag(SourceLocation Loc,
                                         StringRef Message) {
  return DiagEngine->Report(
      Loc, DiagEngine->getCustomDiagID(DiagnosticsEngine::Warning, Message));
}

void ClangTidyContext::setDiagnosticsEngine(DiagnosticsEngine *Engine) {
  DiagEngine = Engine;
}

void ClangTidyContext::setSourceManager(SourceManager *SourceMgr) {
  DiagEngine->setSourceManager(SourceMgr);
}

/// \brief Store a \c ClangTidyError.
void ClangTidyContext::storeError(const ClangTidyError &Error) {
  Errors->push_back(Error);
}

void ClangTidyCheck::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  Context->setSourceManager(Result.SourceManager);
  check(Result);
}

void runClangTidy(StringRef CheckRegexString,
                  const tooling::CompilationDatabase &Compilations,
                  ArrayRef<std::string> Ranges,
                  SmallVectorImpl<ClangTidyError> *Errors) {
  // FIXME: Ranges are currently full files. Support selecting specific
  // (line-)ranges.
  ClangTool Tool(Compilations, Ranges);
  clang::tidy::ClangTidyContext Context(Errors);
  ClangTidyDiagnosticConsumer DiagConsumer(Context);
  ClangTidyCheckFactories CheckFactories;
  for (ClangTidyModuleRegistry::iterator I = ClangTidyModuleRegistry::begin(),
                                         E = ClangTidyModuleRegistry::end();
       I != E; ++I) {
    OwningPtr<ClangTidyModule> Module(I->instantiate());
    Module->addCheckFactories(CheckFactories);
  }

  SmallVector<ClangTidyCheck *, 16> Checks;
  CheckFactories.createChecks(CheckRegexString, Checks);

  MatchFinder Finder;
  for (SmallVectorImpl<ClangTidyCheck *>::iterator I = Checks.begin(),
                                                   E = Checks.end();
       I != E; ++I) {
    (*I)->setContext(&Context);
    (*I)->registerMatchers(&Finder);
  }

  Tool.run(new ClangTidyPPActionFactory(Checks, Context));
  Tool.run(newFrontendActionFactory(&Finder));
}

void handleErrors(SmallVectorImpl<ClangTidyError> &Errors, bool Fix) {
  FileManager Files((FileSystemOptions()));
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticConsumer *DiagPrinter =
      new TextDiagnosticPrinter(llvm::outs(), &*DiagOpts);
  DiagnosticsEngine Diags(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
                          &*DiagOpts, DiagPrinter);
  DiagPrinter->BeginSourceFile(LangOptions());
  SourceManager SourceMgr(Diags, Files);
  Rewriter Rewrite(SourceMgr, LangOptions());
  for (SmallVectorImpl<ClangTidyError>::iterator I = Errors.begin(),
                                                 E = Errors.end();
       I != E; ++I) {
    const FileEntry *File = Files.getFile(I->FilePath);
    FileID ID = SourceMgr.createFileID(File, SourceLocation(), SrcMgr::C_User);
    SourceLocation Loc = SourceMgr.getLocForStartOfFile(ID);
    Diags.Report(Loc.getLocWithOffset(I->FileOffset),
                 Diags.getCustomDiagID(DiagnosticsEngine::Warning, I->Message));
    tooling::applyAllReplacements(I->Fix, Rewrite);
  }
  // FIXME: Run clang-format on changes.
  if (Fix)
    Rewrite.overwriteChangedFiles();
}

} // namespace tidy
} // namespace clang
