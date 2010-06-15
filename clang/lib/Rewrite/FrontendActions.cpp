//===--- FrontendActions.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/FrontendActions.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/Utils.h"
#include "clang/Rewrite/ASTConsumers.h"
#include "clang/Rewrite/FixItRewriter.h"
#include "clang/Rewrite/Rewriters.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// AST Consumer Actions
//===----------------------------------------------------------------------===//

ASTConsumer *HTMLPrintAction::CreateASTConsumer(CompilerInstance &CI,
                                                llvm::StringRef InFile) {
  if (llvm::raw_ostream *OS = CI.createDefaultOutputFile(false, InFile))
    return CreateHTMLPrinter(OS, CI.getPreprocessor());
  return 0;
}

FixItAction::FixItAction() {}
FixItAction::~FixItAction() {}

ASTConsumer *FixItAction::CreateASTConsumer(CompilerInstance &CI,
                                            llvm::StringRef InFile) {
  return new ASTConsumer();
}

class FixItActionSuffixInserter : public FixItPathRewriter {
  std::string NewSuffix;

public:
  explicit FixItActionSuffixInserter(std::string NewSuffix)
    : NewSuffix(NewSuffix) {}

  std::string RewriteFilename(const std::string &Filename) {
    llvm::sys::Path Path(Filename);
    std::string Suffix = Path.getSuffix();
    Path.eraseSuffix();
    Path.appendSuffix(NewSuffix + "." + Suffix);
    return Path.c_str();
  }
};

bool FixItAction::BeginSourceFileAction(CompilerInstance &CI,
                                        llvm::StringRef Filename) {
  const FrontendOptions &FEOpts = getCompilerInstance().getFrontendOpts();
  if (!FEOpts.FixItSuffix.empty()) {
    PathRewriter.reset(new FixItActionSuffixInserter(FEOpts.FixItSuffix));
  } else {
    PathRewriter.reset();
  }
  Rewriter.reset(new FixItRewriter(CI.getDiagnostics(), CI.getSourceManager(),
                                   CI.getLangOpts(), PathRewriter.get()));
  return true;
}

void FixItAction::EndSourceFileAction() {
  // Otherwise rewrite all files.
  Rewriter->WriteFixedFiles();
}

//===----------------------------------------------------------------------===//
// Preprocessor Actions
//===----------------------------------------------------------------------===//

ASTConsumer *RewriteObjCAction::CreateASTConsumer(CompilerInstance &CI,
                                                  llvm::StringRef InFile) {
  if (llvm::raw_ostream *OS = CI.createDefaultOutputFile(false, InFile, "cpp"))
    return CreateObjCRewriter(InFile, OS,
                              CI.getDiagnostics(), CI.getLangOpts(),
                              CI.getDiagnosticOpts().NoRewriteMacros);
  return 0;
}

void RewriteMacrosAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  llvm::raw_ostream *OS = CI.createDefaultOutputFile(true, getCurrentFile());
  if (!OS) return;

  RewriteMacrosInInput(CI.getPreprocessor(), OS);
}

void RewriteTestAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  llvm::raw_ostream *OS = CI.createDefaultOutputFile(false, getCurrentFile());
  if (!OS) return;

  DoRewriteTest(CI.getPreprocessor(), OS);
}
