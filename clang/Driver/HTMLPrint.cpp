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
#include "clang/Basic/SourceManager.h"
#include "clang/AST/ASTContext.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Functional HTML pretty-printing.
//===----------------------------------------------------------------------===//  

namespace {
  class HTMLPrinter : public ASTConsumer {
    Rewriter R;
  public:
    HTMLPrinter() {}
    virtual ~HTMLPrinter();
    
    void Initialize(ASTContext &context);
  };
}

ASTConsumer* clang::CreateHTMLPrinter() { return new HTMLPrinter(); }

void HTMLPrinter::Initialize(ASTContext &context) {
  R.setSourceMgr(context.getSourceManager());
}

HTMLPrinter::~HTMLPrinter() {
  
  unsigned FileID = R.getSourceMgr().getMainFileID();
  html::EscapeText(R, FileID, false, true);
  html::AddLineNumbers(R, FileID);
  html::AddHeaderFooterInternalBuiltinCSS(R, FileID);
  
  // Emit the HTML.
  
  if (const RewriteBuffer *RewriteBuf = R.getRewriteBufferFor(FileID)) {
    char *Buffer = (char*)malloc(RewriteBuf->size());
    std::copy(RewriteBuf->begin(), RewriteBuf->end(), Buffer);
    fwrite(Buffer, 1, RewriteBuf->size(), stdout);
    free(Buffer);
  }
}
