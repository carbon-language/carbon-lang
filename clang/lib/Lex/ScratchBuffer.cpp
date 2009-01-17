//===--- ScratchBuffer.cpp - Scratch space for forming tokens -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the ScratchBuffer interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/ScratchBuffer.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>
using namespace clang;

// ScratchBufSize - The size of each chunk of scratch memory.  Slightly less
//than a page, almost certainly enough for anything. :)
static const unsigned ScratchBufSize = 4060;

ScratchBuffer::ScratchBuffer(SourceManager &SM) : SourceMgr(SM), CurBuffer(0) {
  // Set BytesUsed so that the first call to getToken will require an alloc.
  BytesUsed = ScratchBufSize;
}

/// getToken - Splat the specified text into a temporary MemoryBuffer and
/// return a SourceLocation that refers to the token.  This is just like the
/// method below, but returns a location that indicates the physloc of the
/// token.
SourceLocation ScratchBuffer::getToken(const char *Buf, unsigned Len) {
  if (BytesUsed+Len > ScratchBufSize)
    AllocScratchBuffer(Len);
  
  // Copy the token data into the buffer.
  memcpy(CurBuffer+BytesUsed, Buf, Len);

  // Remember that we used these bytes.
  BytesUsed += Len;

  assert(BytesUsed-Len < (1 << SourceLocation::FilePosBits) &&
         "Out of range file position!");
  
  return BufferStartLoc.getFileLocWithOffset(BytesUsed-Len);
}


/// getToken - Splat the specified text into a temporary MemoryBuffer and
/// return a SourceLocation that refers to the token.  The SourceLoc value
/// gives a virtual location that the token will appear to be from.
SourceLocation ScratchBuffer::getToken(const char *Buf, unsigned Len,
                                       SourceLocation SourceLoc) {
  // Map the physloc to the specified sourceloc.
  return SourceMgr.getInstantiationLoc(getToken(Buf, Len), SourceLoc);
}

void ScratchBuffer::AllocScratchBuffer(unsigned RequestLen) {
  // Only pay attention to the requested length if it is larger than our default
  // page size.  If it is, we allocate an entire chunk for it.  This is to
  // support gigantic tokens, which almost certainly won't happen. :)
  if (RequestLen < ScratchBufSize)
    RequestLen = ScratchBufSize;
  
  llvm::MemoryBuffer *Buf = 
    llvm::MemoryBuffer::getNewMemBuffer(RequestLen, "<scratch space>");
  FileID FID = SourceMgr.createFileIDForMemBuffer(Buf);
  BufferStartLoc = SourceMgr.getLocForStartOfFile(FID);
  CurBuffer = const_cast<char*>(Buf->getBufferStart());
  BytesUsed = 0;
}
