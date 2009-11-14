//===--- FrontendActions.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendActions.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/AnalysisConsumer.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FixItRewriter.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/Utils.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

ASTConsumer *AnalysisAction::CreateASTConsumer(CompilerInstance &CI,
                                               llvm::StringRef InFile) {
  return CreateAnalysisConsumer(CI.getPreprocessor(),
                                CI.getFrontendOpts().OutputFile,
                                CI.getAnalyzerOpts());
}

ASTConsumer *ASTPrintAction::CreateASTConsumer(CompilerInstance &CI,
                                               llvm::StringRef InFile) {
  return CreateASTPrinter(CI.createDefaultOutputFile(false, InFile));
}

ASTConsumer *ASTPrintXMLAction::CreateASTConsumer(CompilerInstance &CI,
                                                  llvm::StringRef InFile) {
  return CreateASTPrinterXML(CI.createDefaultOutputFile(false, InFile,
                                                        "xml"));
}

ASTConsumer *ASTDumpAction::CreateASTConsumer(CompilerInstance &CI,
                                              llvm::StringRef InFile) {
  return CreateASTDumper();
}

ASTConsumer *ASTViewAction::CreateASTConsumer(CompilerInstance &CI,
                                              llvm::StringRef InFile) {
  return CreateASTViewer();
}

ASTConsumer *DeclContextPrintAction::CreateASTConsumer(CompilerInstance &CI,
                                                       llvm::StringRef InFile) {
  return CreateDeclContextPrinter();
}

ASTConsumer *DumpRecordAction::CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef InFile) {
  return CreateRecordLayoutDumper();
}

ASTConsumer *GeneratePCHAction::CreateASTConsumer(CompilerInstance &CI,
                                                  llvm::StringRef InFile) {
  const std::string &Sysroot = CI.getHeaderSearchOpts().Sysroot;
  if (CI.getFrontendOpts().RelocatablePCH &&
      Sysroot.empty()) {
    CI.getDiagnostics().Report(diag::err_relocatable_without_without_isysroot);
    return 0;
  }

  llvm::raw_ostream *OS = CI.createDefaultOutputFile(true, InFile);
  if (CI.getFrontendOpts().RelocatablePCH)
    return CreatePCHGenerator(CI.getPreprocessor(), OS, Sysroot.c_str());

  return CreatePCHGenerator(CI.getPreprocessor(), OS);
}

ASTConsumer *HTMLPrintAction::CreateASTConsumer(CompilerInstance &CI,
                                                llvm::StringRef InFile) {
  return CreateHTMLPrinter(CI.createDefaultOutputFile(false, InFile),
                           CI.getPreprocessor());
}

ASTConsumer *InheritanceViewAction::CreateASTConsumer(CompilerInstance &CI,
                                                      llvm::StringRef InFile) {
  return CreateInheritanceViewer(CI.getFrontendOpts().ViewClassInheritance);
}

FixItAction::FixItAction() {}
FixItAction::~FixItAction() {}

ASTConsumer *FixItAction::CreateASTConsumer(CompilerInstance &CI,
                                                    llvm::StringRef InFile) {
  return new ASTConsumer();
}

/// AddFixItLocations - Add any individual user specified "fix-it" locations,
/// and return true on success.
static bool AddFixItLocations(CompilerInstance &CI,
                              FixItRewriter &FixItRewrite) {
  const std::vector<ParsedSourceLocation> &Locs =
    CI.getFrontendOpts().FixItLocations;
  for (unsigned i = 0, e = Locs.size(); i != e; ++i) {
    const FileEntry *File = CI.getFileManager().getFile(Locs[i].FileName);
    if (!File) {
      CI.getDiagnostics().Report(diag::err_fe_unable_to_find_fixit_file)
        << Locs[i].FileName;
      return false;
    }

    RequestedSourceLocation Requested;
    Requested.File = File;
    Requested.Line = Locs[i].Line;
    Requested.Column = Locs[i].Column;
    FixItRewrite.addFixItLocation(Requested);
  }

  return true;
}

bool FixItAction::BeginSourceFileAction(CompilerInstance &CI,
                                        llvm::StringRef Filename) {
  Rewriter.reset(new FixItRewriter(CI.getDiagnostics(), CI.getSourceManager(),
                                   CI.getLangOpts()));
  if (!AddFixItLocations(CI, *Rewriter))
    return false;

  return true;
}

void FixItAction::EndSourceFileAction() {
  const FrontendOptions &FEOpts = getCompilerInstance().getFrontendOpts();
  Rewriter->WriteFixedFile(getCurrentFile(), FEOpts.OutputFile);
}

ASTConsumer *RewriteObjCAction::CreateASTConsumer(CompilerInstance &CI,
                                                  llvm::StringRef InFile) {
  return CreateObjCRewriter(InFile,
                            CI.createDefaultOutputFile(true, InFile, "cpp"),
                            CI.getDiagnostics(), CI.getLangOpts(),
                            CI.getDiagnosticOpts().NoRewriteMacros);
}

ASTConsumer *RewriteBlocksAction::CreateASTConsumer(CompilerInstance &CI,
                                                    llvm::StringRef InFile) {
  return CreateBlockRewriter(InFile, CI.getDiagnostics(), CI.getLangOpts());
}

ASTConsumer *SyntaxOnlyAction::CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef InFile) {
  return new ASTConsumer();
}

CodeGenAction::CodeGenAction(unsigned _Act) : Act(_Act) {}

ASTConsumer *CodeGenAction::CreateASTConsumer(CompilerInstance &CI,
                                              llvm::StringRef InFile) {
  BackendAction BA = static_cast<BackendAction>(Act);
  llvm::OwningPtr<llvm::raw_ostream> OS;
  if (BA == Backend_EmitAssembly)
    OS.reset(CI.createDefaultOutputFile(false, InFile, "s"));
  else if (BA == Backend_EmitLL)
    OS.reset(CI.createDefaultOutputFile(false, InFile, "ll"));
  else if (BA == Backend_EmitBC)
    OS.reset(CI.createDefaultOutputFile(true, InFile, "bc"));

  return CreateBackendConsumer(BA, CI.getDiagnostics(), CI.getLangOpts(),
                               CI.getCodeGenOpts(), InFile, OS.take(),
                               CI.getLLVMContext());
}

EmitAssemblyAction::EmitAssemblyAction()
  : CodeGenAction(Backend_EmitAssembly) {}

EmitBCAction::EmitBCAction() : CodeGenAction(Backend_EmitBC) {}

EmitLLVMAction::EmitLLVMAction() : CodeGenAction(Backend_EmitLL) {}

EmitLLVMOnlyAction::EmitLLVMOnlyAction() : CodeGenAction(Backend_EmitNothing) {}
