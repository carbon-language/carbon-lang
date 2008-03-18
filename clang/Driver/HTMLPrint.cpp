//===--- RewriteTest.cpp - Playground for the code rewriter ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hacks and fun related to the code rewriter.
//
//===----------------------------------------------------------------------===//

#include "ASTConsumers.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/Rewrite/HTMLRewrite.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "clang/AST/ASTContext.h"
#include <sstream>

using namespace clang;

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

  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* FileStart = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();
  SourceLocation StartLoc = SourceLocation::getFileLoc(FileID, 0);
  SourceLocation EndLoc = SourceLocation::getFileLoc(FileID, FileEnd-FileStart);
  
  html::EscapeText(R, FileID);
  html::AddLineNumbers(R, FileID);
  html::InsertOuterTag(R, html::PRE, StartLoc, EndLoc, 0, 0, true);
  html::InsertOuterTag(R, html::BODY, StartLoc, EndLoc, NULL, "\n", true);
  
  // Generate CSS.
  
  std::ostringstream css;
  css << "\n <style type=\"text/css\">\n";
  css << "  .nums, .lines { vertical-align:top }\n";
  css << "  .nums { padding-right:.5em; width:2.5em }\n";
  css << "  </style>\n";

  
  // Add <head> and <html> tags.
  
  html::InsertTagBefore(R, html::HEAD, StartLoc, StartLoc, 0,css.str().c_str());
  html::InsertOuterTag(R, html::HTML, StartLoc, EndLoc, 0, "\n");
  
  // Emit the HTML.
  
  if (const RewriteBuffer *RewriteBuf = R.getRewriteBufferFor(FileID)) {
    std::string S(RewriteBuf->begin(), RewriteBuf->end());
    printf("%s\n", S.c_str());
  }
}
