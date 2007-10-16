//===--- RewriteTest.cpp - Playground for the code rewriter ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hacks and fun related to the code rewriter.
//
//===----------------------------------------------------------------------===//

#include "ASTConsumers.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/SourceManager.h"
using namespace clang;


namespace {
  class RewriteTest : public ASTConsumer {
    Rewriter Rewrite;
    SourceManager *SM;
    unsigned MainFileID;
    SourceLocation LastIncLoc;
  public:
    void Initialize(ASTContext &Context, unsigned mainFileID) {
      SM = &Context.SourceMgr;
      MainFileID = mainFileID;
      Rewrite.setSourceMgr(Context.SourceMgr);
    }
    
    virtual void HandleTopLevelDecl(Decl *D);

    void HandleDeclInMainFile(Decl *D);
    void RewriteInclude(SourceLocation Loc);
    
    void RewriteFunctionBody(Stmt *S);
    void RewriteAtEncode(ObjCEncodeExpr *Exp);

    ~RewriteTest();
  };
}

ASTConsumer *clang::CreateCodeRewriterTest() { return new RewriteTest(); }

void RewriteTest::HandleTopLevelDecl(Decl *D) {
  // Two cases: either the decl could be in the main file, or it could be in a
  // #included file.  If the former, rewrite it now.  If the later, check to see
  // if we rewrote the #include/#import.
  SourceLocation Loc = D->getLocation();
  Loc = SM->getLogicalLoc(Loc);
  
  // If this is for a builtin, ignore it.
  if (Loc.isInvalid()) return;

  if (SM->getDecomposedFileLoc(Loc).first == MainFileID)
    return HandleDeclInMainFile(D);

  RewriteInclude(Loc);
}

void RewriteTest::RewriteInclude(SourceLocation Loc) {
  // Rip up the #include stack to the main file.
  SourceLocation IncLoc = Loc, NextLoc = Loc;
  do {
    IncLoc = Loc;
    Loc = SM->getLogicalLoc(NextLoc);
    NextLoc = SM->getIncludeLoc(Loc);
  } while (!NextLoc.isInvalid());

  // Loc is now the location of the #include filename "foo" or <foo/bar.h>.
  // IncLoc indicates the header that was included if it is useful.
  IncLoc = SM->getLogicalLoc(IncLoc);
  if (SM->getDecomposedFileLoc(Loc).first != MainFileID ||
      Loc == LastIncLoc)
    return;
  LastIncLoc = Loc;
  
  unsigned IncCol = SM->getColumnNumber(Loc);
  SourceLocation LineStartLoc = Loc.getFileLocWithOffset(-IncCol+1);

  // Replace the #import with #include.
  Rewrite.ReplaceText(LineStartLoc, IncCol-1, "#include ", strlen("#include "));
}

/// HandleDeclInMainFile - This is called for each top-level decl defined in the
/// main file of the input.
void RewriteTest::HandleDeclInMainFile(Decl *D) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    if (Stmt *Body = FD->getBody())
      RewriteFunctionBody(Body);
  // Nothing yet.
}


void RewriteTest::RewriteFunctionBody(Stmt *S) {
  // Handle specific things.
  if (ObjCEncodeExpr *AtEncode = dyn_cast<ObjCEncodeExpr>(S))
    return RewriteAtEncode(AtEncode);
  
  // Otherwise, just rewrite all children.
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end();
       CI != E; ++CI)
    RewriteFunctionBody(*CI);
}

void RewriteTest::RewriteAtEncode(ObjCEncodeExpr *Exp) {
#if 0
  int Size = Rewrite.getRangeSize(Exp->getSourceRange());
  if (Size == -1) {
    printf("BLAH!");
  }
  
  Rewrite.RemoveText(Exp->getAtLoc(), Size);
#endif
}


RewriteTest::~RewriteTest() {
  // Get the top-level buffer that this corresponds to.
  std::pair<const char*, const char*> MainBuf = SM->getBufferData(MainFileID);
  const char *MainBufStart = MainBuf.first;
  const char *MainBufEnd = MainBuf.second;
  
  // Loop over the whole file, looking for tabs.
  for (const char *BufPtr = MainBufStart; BufPtr != MainBufEnd; ++BufPtr) {
    if (*BufPtr != '\t')
      continue;
    
    // Okay, we found a tab.  This tab will turn into at least one character,
    // but it depends on which 'virtual column' it is in.  Compute that now.
    unsigned VCol = 0;
    while (BufPtr-VCol != MainBufStart && BufPtr[-VCol-1] != '\t' &&
           BufPtr[-VCol-1] != '\n' && BufPtr[-VCol-1] != '\r')
      ++VCol;
    
    // Okay, now that we know the virtual column, we know how many spaces to
    // insert.  We assume 8-character tab-stops.
    unsigned Spaces = 8-(VCol & 7);
    
    // Get the location of the tab.
    SourceLocation TabLoc =
      SourceLocation::getFileLoc(MainFileID, BufPtr-MainBufStart);
    
    // Rewrite the single tab character into a sequence of spaces.
    Rewrite.ReplaceText(TabLoc, 1, "        ", Spaces);
  }
  
  // Get the buffer corresponding to MainFileID.  If we haven't changed it, then
  // we are done.
  if (const RewriteBuffer *RewriteBuf = 
          Rewrite.getRewriteBufferFor(MainFileID)) {
    printf("Changed:\n");
    std::string S(RewriteBuf->begin(), RewriteBuf->end());
    printf("%s\n", S.c_str());
  } else {
    printf("No changes\n");
  }
}
