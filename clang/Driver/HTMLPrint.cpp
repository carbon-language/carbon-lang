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
#include "clang/Rewrite/Rewriter.h"
#include "clang/Rewrite/HTMLRewrite.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
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
  public:
    HTMLPrinter(const std::string &OutFile, Diagnostic &D)
      : OutFilename(OutFile), Diags(D) {}
    virtual ~HTMLPrinter();
    
    void Initialize(ASTContext &context);
  };
}

ASTConsumer* clang::CreateHTMLPrinter(const std::string &OutFile, 
                                      Diagnostic &D) {
  return new HTMLPrinter(OutFile, D);
}

void HTMLPrinter::Initialize(ASTContext &context) {
  R.setSourceMgr(context.getSourceManager());
}

HTMLPrinter::~HTMLPrinter() {
  if (Diags.hasErrorOccurred())
    return;

  // Format the file.
  unsigned FileID = R.getSourceMgr().getMainFileID();
  html::EscapeText(R, FileID, false, true);
  html::AddLineNumbers(R, FileID);
  html::AddHeaderFooterInternalBuiltinCSS(R, FileID);

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
  const RewriteBuffer &RewriteBuf = R.getEditBuffer(FileID);
  char *Buffer = (char*)malloc(RewriteBuf.size());
  std::copy(RewriteBuf.begin(), RewriteBuf.end(), Buffer);
  fwrite(Buffer, 1, RewriteBuf.size(), OutputFILE);
  free(Buffer);
  
  if (OutputFILE != stdout) fclose(OutputFILE);
}
