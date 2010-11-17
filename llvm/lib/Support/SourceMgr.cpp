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

#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  struct LineNoCacheTy {
    int LastQueryBufferID;
    const char *LastQuery;
    unsigned LineNoOfQuery;
  };
}

static LineNoCacheTy *getCache(void *Ptr) {
  return (LineNoCacheTy*)Ptr;
}


SourceMgr::~SourceMgr() {
  // Delete the line # cache if allocated.
  if (LineNoCacheTy *Cache = getCache(LineNoCache))
    delete Cache;

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

  // If we have a line number cache, and if the query is to a later point in the
  // same file, start searching from the last query location.  This optimizes
  // for the case when multiple diagnostics come out of one file in order.
  if (LineNoCacheTy *Cache = getCache(LineNoCache))
    if (Cache->LastQueryBufferID == BufferID &&
        Cache->LastQuery <= Loc.getPointer()) {
      Ptr = Cache->LastQuery;
      LineNo = Cache->LineNoOfQuery;
    }

  // Scan for the location being queried, keeping track of the number of lines
  // we see.
  for (; SMLoc::getFromPointer(Ptr) != Loc; ++Ptr)
    if (*Ptr == '\n') ++LineNo;


  // Allocate the line number cache if it doesn't exist.
  if (LineNoCache == 0)
    LineNoCache = new LineNoCacheTy();

  // Update the line # cache.
  LineNoCacheTy &Cache = *getCache(LineNoCache);
  Cache.LastQueryBufferID = BufferID;
  Cache.LastQuery = Ptr;
  Cache.LineNoOfQuery = LineNo;
  return LineNo;
}

void SourceMgr::PrintIncludeStack(SMLoc IncludeLoc, raw_ostream &OS) const {
  if (IncludeLoc == SMLoc()) return;  // Top of stack.

  int CurBuf = FindBufferContainingLoc(IncludeLoc);
  assert(CurBuf != -1 && "Invalid or unspecified location!");

  PrintIncludeStack(getBufferInfo(CurBuf).IncludeLoc, OS);

  OS << "Included from "
     << getBufferInfo(CurBuf).Buffer->getBufferIdentifier()
     << ":" << FindLineNumber(IncludeLoc, CurBuf) << ":\n";
}


/// GetMessage - Return an SMDiagnostic at the specified location with the
/// specified string.
///
/// @param Type - If non-null, the kind of message (e.g., "error") which is
/// prefixed to the message.
SMDiagnostic SourceMgr::GetMessage(SMLoc Loc, const Twine &Msg,
                                   const char *Type, bool ShowLine) const {

  // First thing to do: find the current buffer containing the specified
  // location.
  int CurBuf = FindBufferContainingLoc(Loc);
  assert(CurBuf != -1 && "Invalid or unspecified location!");

  MemoryBuffer *CurMB = getBufferInfo(CurBuf).Buffer;

  // Scan backward to find the start of the line.
  const char *LineStart = Loc.getPointer();
  while (LineStart != CurMB->getBufferStart() &&
         LineStart[-1] != '\n' && LineStart[-1] != '\r')
    --LineStart;

  std::string LineStr;
  if (ShowLine) {
    // Get the end of the line.
    const char *LineEnd = Loc.getPointer();
    while (LineEnd != CurMB->getBufferEnd() &&
           LineEnd[0] != '\n' && LineEnd[0] != '\r')
      ++LineEnd;
    LineStr = std::string(LineStart, LineEnd);
  }

  std::string PrintedMsg;
  raw_string_ostream OS(PrintedMsg);
  if (Type)
    OS << Type << ": ";
  OS << Msg;

  return SMDiagnostic(*this, Loc,
                      CurMB->getBufferIdentifier(), FindLineNumber(Loc, CurBuf),
                      Loc.getPointer()-LineStart, OS.str(),
                      LineStr, ShowLine);
}

void SourceMgr::PrintMessage(SMLoc Loc, const Twine &Msg,
                             const char *Type, bool ShowLine) const {
  // Report the message with the diagnostic handler if present.
  if (DiagHandler) {
    DiagHandler(GetMessage(Loc, Msg, Type, ShowLine), DiagContext);
    return;
  }
  
  raw_ostream &OS = errs();

  int CurBuf = FindBufferContainingLoc(Loc);
  assert(CurBuf != -1 && "Invalid or unspecified location!");
  PrintIncludeStack(getBufferInfo(CurBuf).IncludeLoc, OS);

  GetMessage(Loc, Msg, Type, ShowLine).Print(0, OS);
}

//===----------------------------------------------------------------------===//
// SMDiagnostic Implementation
//===----------------------------------------------------------------------===//

void SMDiagnostic::Print(const char *ProgName, raw_ostream &S) const {
  if (ProgName && ProgName[0])
    S << ProgName << ": ";

  if (!Filename.empty()) {
    if (Filename == "-")
      S << "<stdin>";
    else
      S << Filename;

    if (LineNo != -1) {
      S << ':' << LineNo;
      if (ColumnNo != -1)
        S << ':' << (ColumnNo+1);
    }
    S << ": ";
  }

  S << Message << '\n';

  if (LineNo != -1 && ColumnNo != -1 && ShowLine) {
    S << LineContents << '\n';

    // Print out spaces/tabs before the caret.
    for (unsigned i = 0; i != unsigned(ColumnNo); ++i)
      S << (LineContents[i] == '\t' ? '\t' : ' ');
    S << "^\n";
  }
}


