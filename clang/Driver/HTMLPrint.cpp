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
  
  // Generate header

  {
    std::ostringstream os;
  
    os << "<html>\n<head>\n"
       << " <style type=\"text/css\">\n"    
       << "  .codeblock { width:100% }\n"
       << "  .codeline { font-family: \"Monaco\", fixed; font-size:11pt }\n"
       << "  .codeline { height:1.5em; line-height:1.5em }\n"
       << "  .nums, .lines { float:left; height:100% }\n"
       << "  .nums { background-color: #eeeeee }\n"
       << "  .nums { font-family: \"Andale Mono\", fixed; font-size:smaller }\n"
       << "  .nums { width:2.5em; padding-right:2ex; text-align:right }\n"
       << "  .lines { padding-left: 1ex; border-left: 3px solid #ccc }\n"
       << "  .lines { white-space: pre }\n"
       << " </style>\n"
       << "</head>\n"
       << "<body>";

    R.InsertStrBefore(StartLoc, os.str());
  }
  
  // Generate footer
  
  {
    std::ostringstream os;
    
    os << "</body></html>\n";
    R.InsertStrAfter(EndLoc, os.str());
  }
    
  
  // Emit the HTML.
  
  if (const RewriteBuffer *RewriteBuf = R.getRewriteBufferFor(FileID)) {
    std::string S(RewriteBuf->begin(), RewriteBuf->end());
    printf("%s\n", S.c_str());
  }
}
