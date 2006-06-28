//===--- ScratchBuffer.cpp - Scratch space for forming tokens -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the ScratchBuffer interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/ScratchBuffer.h"
#include "clang/Basic/SourceBuffer.h"
#include "clang/Basic/SourceManager.h"
using namespace llvm;
using namespace clang;

// ScratchBufSize - The size of each chunk of scratch memory.  Slightly less
//than a page, almost certainly enough for anything. :)
static const unsigned ScratchBufSize = 4060;

ScratchBuffer::ScratchBuffer(SourceManager &SM) : SourceMgr(SM), CurBuffer(0) {
  // Set BytesUsed so that the first call to getToken will require an alloc.
  BytesUsed = ScratchBufSize;
  FileID = 0;
}


/// getToken - Splat the specified text into a temporary SourceBuffer and
/// return a SourceLocation that refers to the token.  The SourceLoc value
/// gives a virtual location that the token will appear to be from.
SourceLocation ScratchBuffer::getToken(const char *Buf, unsigned Len,
                                       SourceLocation SourceLoc) {
  if (BytesUsed+Len > ScratchBufSize)
    AllocScratchBuffer(Len);

  // Copy the token data into the buffer.
  memcpy(CurBuffer+BytesUsed, Buf, Len);

  // Create the initial SourceLocation.
  SourceLocation Loc(FileID, BytesUsed);
  assert(BytesUsed < (1 << SourceLocation::FilePosBits) &&
         "Out of range file position!");
  
  // FIXME: Merge SourceLoc into it.
  
  // Remember that we used these bytes.
  BytesUsed += Len;
  
  return Loc;
}

void ScratchBuffer::AllocScratchBuffer(unsigned RequestLen) {
  // Only pay attention to the requested length if it is larger than our default
  // page size.  If it is, we allocate an entire chunk for it.  This is to
  // support gigantic tokens, which almost certainly won't happen. :)
  if (RequestLen < ScratchBufSize)
    RequestLen = ScratchBufSize;
  
  SourceBuffer *Buf = 
    SourceBuffer::getNewMemBuffer(RequestLen, "<scratch space>");
  FileID = SourceMgr.createFileIDForMemBuffer(Buf);
  CurBuffer = const_cast<char*>(Buf->getBufferStart());
  BytesUsed = 0;
}
