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
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
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

//===----------------------------------------------------------------------===//
// AST Consumer Actions
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Preprocessor Actions
//===----------------------------------------------------------------------===//

void DumpRawTokensAction::ExecuteAction() {
  Preprocessor &PP = getCompilerInstance().getPreprocessor();
  SourceManager &SM = PP.getSourceManager();

  // Start lexing the specified input file.
  Lexer RawLex(SM.getMainFileID(), SM, PP.getLangOptions());
  RawLex.SetKeepWhitespaceMode(true);

  Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  while (RawTok.isNot(tok::eof)) {
    PP.DumpToken(RawTok, true);
    fprintf(stderr, "\n");
    RawLex.LexFromRawLexer(RawTok);
  }
}

void DumpTokensAction::ExecuteAction() {
  Preprocessor &PP = getCompilerInstance().getPreprocessor();
  // Start preprocessing the specified input file.
  Token Tok;
  PP.EnterMainSourceFile();
  do {
    PP.Lex(Tok);
    PP.DumpToken(Tok, true);
    fprintf(stderr, "\n");
  } while (Tok.isNot(tok::eof));
}

void GeneratePTHAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  if (CI.getFrontendOpts().OutputFile.empty() ||
      CI.getFrontendOpts().OutputFile == "-") {
    // FIXME: Don't fail this way.
    // FIXME: Verify that we can actually seek in the given file.
    llvm::errs() << "ERROR: PTH requires an seekable file for output!\n";
    ::exit(1);
  }
  llvm::raw_fd_ostream *OS =
    CI.createDefaultOutputFile(true, getCurrentFile());
  CacheTokens(CI.getPreprocessor(), OS);
}

void ParseOnlyAction::ExecuteAction() {
  Preprocessor &PP = getCompilerInstance().getPreprocessor();
  llvm::OwningPtr<Action> PA(new MinimalAction(PP));

  Parser P(PP, *PA);
  PP.EnterMainSourceFile();
  P.ParseTranslationUnit();
}

void PreprocessOnlyAction::ExecuteAction() {
  Preprocessor &PP = getCompilerInstance().getPreprocessor();

  Token Tok;
  // Start parsing the specified input file.
  PP.EnterMainSourceFile();
  do {
    PP.Lex(Tok);
  } while (Tok.isNot(tok::eof));
}

void PrintParseAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  Preprocessor &PP = getCompilerInstance().getPreprocessor();
  llvm::raw_ostream *OS = CI.createDefaultOutputFile(false, getCurrentFile());
  llvm::OwningPtr<Action> PA(CreatePrintParserActionsAction(PP, OS));

  Parser P(PP, *PA);
  PP.EnterMainSourceFile();
  P.ParseTranslationUnit();
}

void PrintPreprocessedAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  llvm::raw_ostream *OS = CI.createDefaultOutputFile(false, getCurrentFile());
  DoPrintPreprocessedInput(CI.getPreprocessor(), OS,
                           CI.getPreprocessorOutputOpts());
}

void RewriteMacrosAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  llvm::raw_ostream *OS = CI.createDefaultOutputFile(true, getCurrentFile());
  RewriteMacrosInInput(CI.getPreprocessor(), OS);
}

void RewriteTestAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  llvm::raw_ostream *OS = CI.createDefaultOutputFile(false, getCurrentFile());
  DoRewriteTest(CI.getPreprocessor(), OS);
}
