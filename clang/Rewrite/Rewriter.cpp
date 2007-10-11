//===--- Rewriter.cpp - Code rewriting interface --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Rewriter class, which is used for code
//  transformations.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/Rewriter.h"
#include "clang/Basic/SourceManager.h"
using namespace clang;


void RewriteBuffer::RemoveText(unsigned OrigOffset, unsigned Size) {
  // FIXME:
}

void RewriteBuffer::InsertText(unsigned OrigOffset,
                               const char *StrData, unsigned StrLen) {
  // FIXME:
}



//===----------------------------------------------------------------------===//
// Rewriter class
//===----------------------------------------------------------------------===//

/// getEditBuffer - Get or create a RewriteBuffer for the specified FileID.
///
RewriteBuffer &Rewriter::getEditBuffer(unsigned FileID) {
  std::map<unsigned, RewriteBuffer>::iterator I =
    RewriteBuffers.lower_bound(FileID);
  if (I != RewriteBuffers.end() && I->first == FileID) 
    return I->second;
  I = RewriteBuffers.insert(I, std::make_pair(FileID, RewriteBuffer()));
  
  std::pair<const char*, const char*> MB = SourceMgr.getBufferData(FileID);
  I->second.Initialize(MB.first, MB.second);
  
  return I->second;
}


void Rewriter::ReplaceText(SourceLocation Start, unsigned OrigLength,
                           const char *NewStr, unsigned NewLength) {
  assert(isRewritable(Start) && "Not a rewritable location!");
  
}
