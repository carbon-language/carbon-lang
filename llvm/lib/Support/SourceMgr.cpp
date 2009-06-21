//===- SourceMgr.cpp - Manager for Simple Source Buffers & Diagnostics ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SourceMgr class.  This class is used as a simple
// substrate for diagnostics, #include handling, and other low level things for
// simple parsers.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

SourceMgr::~SourceMgr() {
  while (!Buffers.empty()) {
    delete Buffers.back().Buffer;
    Buffers.pop_back();
  }
}

/// AddIncludeFile - Search for a file with the specified name in the current
/// directory or in one of the IncludeDirs.  If no file is found, this returns
/// ~0, otherwise it returns the buffer ID of the stacked file.
unsigned SourceMgr::AddIncludeFile(const std::string &Filename,
                                   SMLoc IncludeLoc) {
  
  MemoryBuffer *NewBuf = MemoryBuffer::getFile(Filename.c_str());

  // If the file didn't exist directly, see if it's in an include path.
  for (unsigned i = 0, e = IncludeDirectories.size(); i != e && !NewBuf; ++i) {
    std::string IncFile = IncludeDirectories[i] + "/" + Filename;
    NewBuf = MemoryBuffer::getFile(IncFile.c_str());
  }
 
  if (NewBuf == 0) return ~0U;

  return AddNewSourceBuffer(NewBuf, IncludeLoc);
}


/// FindBufferContainingLoc - Return the ID of the buffer containing the
/// specified location, returning -1 if not found.
int SourceMgr::FindBufferContainingLoc(SMLoc Loc) const {
  for (unsigned i = 0, e = Buffers.size(); i != e; ++i)
    if (Loc.getPointer() >= Buffers[i].Buffer->getBufferStart() &&
        // Use <= here so that a pointer to the null at the end of the buffer
        // is included as part of the buffer.
        Loc.getPointer() <= Buffers[i].Buffer->getBufferEnd())
      return i;
  return -1;
}

/// FindLineNumber - Find the line number for the specified location in the
/// specified file.  This is not a fast method.
unsigned SourceMgr::FindLineNumber(SMLoc Loc, int BufferID) const {
  if (BufferID == -1) BufferID = FindBufferContainingLoc(Loc);
  assert(BufferID != -1 && "Invalid Location!");
  
  MemoryBuffer *Buff = getBufferInfo(BufferID).Buffer;
  
  // Count the number of \n's between the start of the file and the specified
  // location.
  unsigned LineNo = 1;
  
  const char *Ptr = Buff->getBufferStart();

  for (; SMLoc::getFromPointer(Ptr) != Loc; ++Ptr)
    if (*Ptr == '\n') ++LineNo;
  return LineNo;
}

void SourceMgr::PrintIncludeStack(SMLoc IncludeLoc) const {
  if (IncludeLoc == SMLoc()) return;  // Top of stack.
  
  int CurBuf = FindBufferContainingLoc(IncludeLoc);
  assert(CurBuf != -1 && "Invalid or unspecified location!");

  PrintIncludeStack(getBufferInfo(CurBuf).IncludeLoc);
  
  errs() << "Included from "
         << getBufferInfo(CurBuf).Buffer->getBufferIdentifier()
         << ":" << FindLineNumber(IncludeLoc, CurBuf) << ":\n";
}


void SourceMgr::PrintMessage(SMLoc Loc, const std::string &Msg) const {
  raw_ostream &OS = errs();
  
  // First thing to do: find the current buffer containing the specified
  // location.
  int CurBuf = FindBufferContainingLoc(Loc);
  assert(CurBuf != -1 && "Invalid or unspecified location!");
  
  PrintIncludeStack(getBufferInfo(CurBuf).IncludeLoc);
  
  MemoryBuffer *CurMB = getBufferInfo(CurBuf).Buffer;
  
  
  OS << "Parsing " << CurMB->getBufferIdentifier() << ":"
     << FindLineNumber(Loc, CurBuf) << ": ";
  
  OS << Msg << "\n";
  
  // Scan backward to find the start of the line.
  const char *LineStart = Loc.getPointer();
  while (LineStart != CurMB->getBufferStart() && 
         LineStart[-1] != '\n' && LineStart[-1] != '\r')
    --LineStart;
  // Get the end of the line.
  const char *LineEnd = Loc.getPointer();
  while (LineEnd != CurMB->getBufferEnd() && 
         LineEnd[0] != '\n' && LineEnd[0] != '\r')
    ++LineEnd;
  // Print out the line.
  OS << std::string(LineStart, LineEnd) << "\n";
  // Print out spaces before the caret.
  for (const char *Pos = LineStart; Pos != Loc.getPointer(); ++Pos)
    OS << (*Pos == '\t' ? '\t' : ' ');
  OS << "^\n";
}
