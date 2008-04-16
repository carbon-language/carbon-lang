//===--- HTMLPrint.cpp - Playground for the HTML code rewriter ------------===//
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
#include "clang/Basic/Diagnostic.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/AST/CFG.h"
#include <sstream>

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

//===----------------------------------------------------------------------===//
// Other HTML pretty-printing code used to test new features.
//===----------------------------------------------------------------------===//  

namespace {
  class HTMLTest : public ASTConsumer {
    Rewriter R;
    ASTContext* Ctx;
  public:
    HTMLTest() : Ctx(NULL) {}
    virtual ~HTMLTest();
    virtual void HandleTopLevelDecl(Decl* D);
    
    void Initialize(ASTContext &context);
    void ProcessBody(Stmt* S);
  };
}

ASTConsumer* clang::CreateHTMLTest() { return new HTMLTest(); }

void HTMLTest::Initialize(ASTContext &context) {
  Ctx = &context;
  R.setSourceMgr(context.getSourceManager());
}

void HTMLTest::HandleTopLevelDecl(Decl* D) {  
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    if (Stmt* B = FD->getBody()) {
      SourceLocation L = B->getLocStart();

      if (L.isFileID() && L.getFileID() == R.getSourceMgr().getMainFileID())
        ProcessBody(B);
    }
}

HTMLTest::~HTMLTest() {

  unsigned FileID = R.getSourceMgr().getMainFileID();
  html::EscapeText(R, FileID);
  html::AddLineNumbers(R, FileID);
  html::AddHeaderFooterInternalBuiltinCSS(R, FileID);
  
  // Emit the HTML.
  
  if (const RewriteBuffer *RewriteBuf = R.getRewriteBufferFor(FileID)) {
    std::string S(RewriteBuf->begin(), RewriteBuf->end());
    printf("%s\n", S.c_str());
  }
}

namespace {
  class HTMLDiagnostic : public DiagnosticClient {
    Rewriter& R;
  public:
    HTMLDiagnostic(Rewriter& r) : R(r) {}
    virtual void HandleDiagnostic(Diagnostic &Diags, 
                                  Diagnostic::Level DiagLevel,
                                  FullSourceLoc Pos,
                                  diag::kind ID,
                                  const std::string *Strs,
                                  unsigned NumStrs,
                                  const SourceRange *Ranges, 
                                  unsigned NumRanges);
  };  
}

void HTMLTest::ProcessBody(Stmt* S) {
  CFG* cfg = CFG::buildCFG(S);

  if (!cfg)
    return;
  
  HTMLDiagnostic HD(R);
  Diagnostic D(HD);
  
  CheckDeadStores(*cfg, *Ctx, D);
}

void HTMLDiagnostic::HandleDiagnostic(Diagnostic &Diags, 
                                      Diagnostic::Level DiagLevel,
                                      FullSourceLoc Pos,
                                      diag::kind ID,
                                      const std::string *Strs,
                                      unsigned NumStrs,
                                      const SourceRange *Ranges, 
                                      unsigned NumRanges) {
  
  // For now, just draw a box above the line in question, and emit the
  // warning.
  
  if (!Pos.isValid())
    return;  
  
  SourceManager& SM = R.getSourceMgr();
  
  FullSourceLoc LPos = Pos.getLogicalLoc();
  unsigned FileID = SM.getCanonicalFileID(LPos.getLocation());
  
  assert (&LPos.getManager() == &SM && "SourceManagers are different!");
  
  if (!SM.isFromMainFile(LPos.getLocation()))
    return;
  
  // Compute the column number.  Rewind from the current position to the start
  // of the line.

  unsigned ColNo = LPos.getColumnNumber();
  const char *TokLogicalPtr = LPos.getCharacterData();
  const char *LineStart = TokLogicalPtr-ColNo;
  
  // Ripped from TextDiagnostics::FormatDiagnostic:
  
  std::string Msg = Diags.getDescription(ID);
  
  for (unsigned i = 0; i < Msg.size() - 1; ++i) {
    if (Msg[i] == '%' && isdigit(Msg[i + 1])) {
      unsigned StrNo = Msg[i + 1] - '0';
      Msg = std::string(Msg.begin(), Msg.begin() + i) +
      (StrNo < NumStrs ? Strs[StrNo] : "<<<INTERNAL ERROR>>>") +
      std::string(Msg.begin() + i + 2, Msg.end());
    }
  }  
  
  // Create the html for the message.
  
  std::ostringstream os;

  os << "\n<tr><td class=\"num\"></td><td class=\"line\">"
     << "<div class=\"msg\" style=\"margin-left:"
     << ColNo << "ex\">";
  
  switch (DiagLevel) {
    default: assert(0 && "Unknown diagnostic type!");
    case Diagnostic::Note:    os << "note: "; break;
    case Diagnostic::Warning: os << "warning: "; break;
    case Diagnostic::Error:   os << "error: "; break;
    case Diagnostic::Fatal:   os << "fatal error: "; break;
      break;
  }
  
  os << Msg << "</div></td></tr>";
  
  // Insert the new html.
  
  const llvm::MemoryBuffer *Buf = SM.getBuffer(FileID);
  const char* FileStart = Buf->getBufferStart();
  
  R.InsertStrBefore(SourceLocation::getFileLoc(FileID, LineStart - FileStart),
                    os.str());
  
  // Now highlight the ranges.
  
  for (unsigned i = 0; i < NumRanges; ++i) {
    
    SourceLocation B = SM.getLogicalLoc(Ranges->getBegin());
    SourceLocation E = SM.getLogicalLoc(Ranges->getEnd());
    
    // We do this because the position seems to point to the beginning of
    // the last character.  FIXME: Is this what is suppose to happen?
    std::pair<unsigned,unsigned> X = SM.getDecomposedFileLoc(E);
    E = SourceLocation::getFileLoc(X.first, X.second+1);

    ++Ranges;
    
    if (!SM.isFromMainFile(B) || !SM.isFromMainFile(E))
      continue;
    
    // Highlight the range.  Make the span tag the outermost tag for the
    // selected range.
    R.InsertCStrBefore(B, "<span class=\"mrange\">");
    R.InsertCStrAfter(E, "</span>");
  }  
}
