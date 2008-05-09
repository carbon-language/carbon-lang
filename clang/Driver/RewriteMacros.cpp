//===--- RewriteMacros.cpp - Rewrite macros into their expansions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This code rewrites macro invocations into their expansions.  This gives you
// a macro expanded file that retains comments and #includes.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Path.h"
#include <fstream>
using namespace clang;

/// RewriteMacrosInInput - Implement -rewrite-macros mode.
void clang::RewriteMacrosInInput(Preprocessor &PP,const std::string &InFileName,
                                 const std::string &OutFileName) {
  SourceManager &SM = PP.getSourceManager();
  
  Rewriter Rewrite;
  Rewrite.setSourceMgr(SM);

  // Get the ID and start/end of the main file.
  unsigned MainFileID = SM.getMainFileID();
  //const llvm::MemoryBuffer *MainBuf = SM.getBuffer(MainFileID);
  //const char *MainFileStart = MainBuf->getBufferStart();
  //const char *MainFileEnd = MainBuf->getBufferEnd();
 
  
  // Create the output file.
  
  std::ostream *OutFile;
  if (OutFileName == "-") {
    OutFile = llvm::cout.stream();
  } else if (!OutFileName.empty()) {
    OutFile = new std::ofstream(OutFileName.c_str(), 
                                std::ios_base::binary|std::ios_base::out);
  } else if (InFileName == "-") {
    OutFile = llvm::cout.stream();
  } else {
    llvm::sys::Path Path(InFileName);
    Path.eraseSuffix();
    Path.appendSuffix("cpp");
    OutFile = new std::ofstream(Path.toString().c_str(), 
                                std::ios_base::binary|std::ios_base::out);
  }

  // Get the buffer corresponding to MainFileID.  If we haven't changed it, then
  // we are done.
  if (const RewriteBuffer *RewriteBuf = 
      Rewrite.getRewriteBufferFor(MainFileID)) {
    //printf("Changed:\n");
    *OutFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
  } else {
    fprintf(stderr, "No changes\n");
  }
}
