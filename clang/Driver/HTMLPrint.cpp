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
    Preprocessor *PP;
  public:
    HTMLPrinter(const std::string &OutFile, Diagnostic &D, Preprocessor *pp)
      : OutFilename(OutFile), Diags(D), PP(pp) {}
    virtual ~HTMLPrinter();
    
    void Initialize(ASTContext &context);
  };
}

ASTConsumer* clang::CreateHTMLPrinter(const std::string &OutFile, 
                                      Diagnostic &D, Preprocessor *PP) {
  return new HTMLPrinter(OutFile, D, PP);
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

  // If we have a preprocessor, relex the file and syntax hilight.  We might not
  // have a preprocessor if we come from a deserialized AST file, for example.
  if (PP) {
    html::SyntaxHighlight(R, FileID, *PP);
    html::HighlightMacros(R, FileID, *PP);
  }
  
  
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
