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
    SourceManager *SM;
    unsigned MainFileID;
  public:
    void Initialize(ASTContext &Context, unsigned mainFileID) {
      SM = &Context.SourceMgr;
      MainFileID = mainFileID;
    }
    
    virtual void HandleTopLevelDecl(Decl *D);

    
    ~RewriteTest();
  };
}

ASTConsumer *clang::CreateCodeRewriterTest() { return new RewriteTest(); }

void RewriteTest::HandleTopLevelDecl(Decl *D) {
  // Nothing to do here yet.
#if 0
  if (NamedDecl *ND = dyn_cast<NamedDecl>(D))
    if (ND->getName())
      printf("%s\n", ND->getName());
#endif
}



RewriteTest::~RewriteTest() {
  Rewriter Rewrite(*SM);
  
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
    Rewrite.ReplaceText(TabLoc, 1, "xxxxxxxxxxx", Spaces);
  }
  
  // Get the buffer corresponding to MainFileID.  If we haven't changed it, then
  // we are done.
  if (const RewriteBuffer *RewriteBuf = 
          Rewrite.getRewriteBufferFor(MainFileID)) {
    RewriteBuf = 0;
    printf("Changed\n");
  } else {
    printf("No changes\n");
  }
}
