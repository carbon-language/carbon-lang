//===--- HTMLPrint.cpp - Source code -> HTML pretty-printing --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pretty-printing of source code to HTML.
//
//===----------------------------------------------------------------------===//

#include "ASTConsumers.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/Rewrite/HTMLRewrite.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/ASTContext.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Functional HTML pretty-printing.
//===----------------------------------------------------------------------===//  

namespace {
  class HTMLPrinter : public ASTConsumer {
    Rewriter R;
    std::string OutFilename;
    Diagnostic &Diags;
    Preprocessor *PP;
    PreprocessorFactory *PPF;
  public:
    HTMLPrinter(const std::string &OutFile, Diagnostic &D, Preprocessor *pp,
                PreprocessorFactory* ppf)
      : OutFilename(OutFile), Diags(D), PP(pp), PPF(ppf) {}
    virtual ~HTMLPrinter();
    
    void Initialize(ASTContext &context);
  };
}

ASTConsumer* clang::CreateHTMLPrinter(const std::string &OutFile, 
                                      Diagnostic &D, Preprocessor *PP,
                                      PreprocessorFactory* PPF) {
  
  return new HTMLPrinter(OutFile, D, PP, PPF);
}

void HTMLPrinter::Initialize(ASTContext &context) {
  R.setSourceMgr(context.getSourceManager(), context.getLangOptions());
}

HTMLPrinter::~HTMLPrinter() {
  if (Diags.hasErrorOccurred())
    return;

  // Format the file.
  FileID FID = R.getSourceMgr().getMainFileID();
  const FileEntry* Entry = R.getSourceMgr().getFileEntryForID(FID);
  
  html::AddLineNumbers(R, FID);
  html::AddHeaderFooterInternalBuiltinCSS(R, FID, Entry->getName());

  // If we have a preprocessor, relex the file and syntax highlight.
  // We might not have a preprocessor if we come from a deserialized AST file,
  // for example.
  
  if (PP) html::SyntaxHighlight(R, FID, *PP);
  if (PPF) html::HighlightMacros(R, FID, *PP);
  html::EscapeText(R, FID, false, true);
  
  // Open the output.
  FILE *OutputFILE;
  if (OutFilename.empty() || OutFilename == "-")
    OutputFILE = stdout;
  else {
    OutputFILE = fopen(OutFilename.c_str(), "w+");
    if (OutputFILE == 0) {
      fprintf(stderr, "Error opening output file '%s'.\n", OutFilename.c_str());
      exit(1);
    }
  }
  
  // Emit the HTML.
  const RewriteBuffer &RewriteBuf = R.getEditBuffer(FID);
  char *Buffer = (char*)malloc(RewriteBuf.size());
  std::copy(RewriteBuf.begin(), RewriteBuf.end(), Buffer);
  fwrite(Buffer, 1, RewriteBuf.size(), OutputFILE);
  free(Buffer);
  
  if (OutputFILE != stdout) fclose(OutputFILE);
}
