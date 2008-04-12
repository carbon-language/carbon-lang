//===--- Rewriter.cpp - Code rewriting interface --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Rewriter class, which is used for code
//  transformations.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/Rewriter.h"
#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include <sstream>
using namespace clang;


/// getMappedOffset - Given an offset into the original SourceBuffer that this
/// RewriteBuffer is based on, map it into the offset space of the
/// RewriteBuffer.
unsigned RewriteBuffer::getMappedOffset(unsigned OrigOffset,
                                        bool AfterInserts) const {
  unsigned ResultOffset = 0;
#if !defined(USE_VECTOR)
  ResultOffset += Deltas.getDeltaAt(OrigOffset+AfterInserts);
#else
  unsigned DeltaIdx = 0;
  
  // Move past any deltas that are relevant.
  // FIXME: binary search.
  for (; DeltaIdx != Deltas.size() && 
       Deltas[DeltaIdx].FileLoc < OrigOffset; ++DeltaIdx)
    ResultOffset += Deltas[DeltaIdx].Delta;
  
  if (AfterInserts)
    for (; DeltaIdx != Deltas.size() &&
         OrigOffset == Deltas[DeltaIdx].FileLoc; ++DeltaIdx)
      ResultOffset += Deltas[DeltaIdx].Delta;
#endif
  
 // printf("Map: %d/%d -> %d\n", OrigOffset, AfterInserts, ResultOffset);
  return ResultOffset+OrigOffset;
}

/// AddDelta - When a change is made that shifts around the text buffer, this
/// method is used to record that info.
void RewriteBuffer::AddDelta(unsigned OrigOffset, int Change) {
 // printf("AddDelta: %d/%d\n", OrigOffset, Change);
#if !defined(USE_VECTOR)
  return Deltas.AddDelta(OrigOffset, Change);
#else
  assert(Change != 0 && "Not changing anything");
  unsigned DeltaIdx = 0;
  
  // Skip over any unrelated deltas.
  for (; DeltaIdx != Deltas.size() && 
       Deltas[DeltaIdx].FileLoc < OrigOffset; ++DeltaIdx)
    ;
  
  // If there is no a delta for this offset, insert a new delta record.
  if (DeltaIdx == Deltas.size() || OrigOffset != Deltas[DeltaIdx].FileLoc) {
    // If this is a removal, check to see if this can be folded into
    // a delta at the end of the deletion.  For example, if we have:
    //  ABCXDEF (X inserted after C) and delete C, we want to end up with no
    // delta because X basically replaced C.
    if (Change < 0 && DeltaIdx != Deltas.size() &&
        OrigOffset-Change == Deltas[DeltaIdx].FileLoc) {
      // Adjust the start of the delta to be the start of the deleted region.
      Deltas[DeltaIdx].FileLoc += Change;
      Deltas[DeltaIdx].Delta += Change;

      // If the delta becomes a noop, remove it.
      if (Deltas[DeltaIdx].Delta == 0)
        Deltas.erase(Deltas.begin()+DeltaIdx);
      return;
    }
    
    // Otherwise, create an entry and return.
    Deltas.insert(Deltas.begin()+DeltaIdx, 
                  SourceDelta::get(OrigOffset, Change));
    return;
  }
  
  // Otherwise, we found a delta record at this offset, adjust it.
  Deltas[DeltaIdx].Delta += Change;
  
  // If it is now dead, remove it.
  if (Deltas[DeltaIdx].Delta == 0)
    Deltas.erase(Deltas.begin()+DeltaIdx);
#endif
}


void RewriteBuffer::RemoveText(unsigned OrigOffset, unsigned Size) {
  // Nothing to remove, exit early.
  if (Size == 0) return;

  unsigned RealOffset = getMappedOffset(OrigOffset, true);
  assert(RealOffset+Size < Buffer.size() && "Invalid location");
  
  // Remove the dead characters.
  RewriteRope::iterator I = Buffer.getAtOffset(RealOffset);
  Buffer.erase(I, I+Size);

  // Add a delta so that future changes are offset correctly.
  AddDelta(OrigOffset, -Size);
}

void RewriteBuffer::InsertText(unsigned OrigOffset,
                               const char *StrData, unsigned StrLen,
                               bool InsertAfter) {
  
  // Nothing to insert, exit early.
  if (StrLen == 0) return;
  
  unsigned RealOffset = getMappedOffset(OrigOffset, InsertAfter);
  assert(RealOffset <= Buffer.size() && "Invalid location");

  // Insert the new characters.
  Buffer.insert(Buffer.getAtOffset(RealOffset), StrData, StrData+StrLen);
  
  // Add a delta so that future changes are offset correctly.
  AddDelta(OrigOffset, StrLen);
}

/// ReplaceText - This method replaces a range of characters in the input
/// buffer with a new string.  This is effectively a combined "remove/insert"
/// operation.
void RewriteBuffer::ReplaceText(unsigned OrigOffset, unsigned OrigLength,
                                const char *NewStr, unsigned NewLength) {
  unsigned RealOffset = getMappedOffset(OrigOffset, true);
  assert(RealOffset+OrigLength <= Buffer.size() && "Invalid location");

  // Overwrite the common piece.
  unsigned CommonLength = std::min(OrigLength, NewLength);
  std::copy(NewStr, NewStr+CommonLength, Buffer.getAtOffset(RealOffset));
  
  // If replacing without shifting around, just overwrite the text.
  if (OrigLength == NewLength)
    return;

  // If inserting more than existed before, this is like an insertion.
  if (NewLength > OrigLength) {
    Buffer.insert(Buffer.getAtOffset(RealOffset+OrigLength),
                  NewStr+OrigLength, NewStr+NewLength);
  } else {
    // If inserting less than existed before, this is like a removal.
    RewriteRope::iterator I = Buffer.getAtOffset(RealOffset+NewLength);
    Buffer.erase(I, I+(OrigLength-NewLength));
  }
  AddDelta(OrigOffset, NewLength-OrigLength);
}


//===----------------------------------------------------------------------===//
// Rewriter class
//===----------------------------------------------------------------------===//

/// getRangeSize - Return the size in bytes of the specified range if they
/// are in the same file.  If not, this returns -1.
int Rewriter::getRangeSize(SourceRange Range) const {
  if (!isRewritable(Range.getBegin()) ||
      !isRewritable(Range.getEnd())) return -1;
  
  unsigned StartOff, StartFileID;
  unsigned EndOff  , EndFileID;
  
  StartOff = getLocationOffsetAndFileID(Range.getBegin(), StartFileID);
  EndOff   = getLocationOffsetAndFileID(Range.getEnd(), EndFileID);
  
  if (StartFileID != EndFileID)
    return -1;
  
  // If edits have been made to this buffer, the delta between the range may
  // have changed.
  std::map<unsigned, RewriteBuffer>::const_iterator I =
    RewriteBuffers.find(StartFileID);
  if (I != RewriteBuffers.end()) {
    const RewriteBuffer &RB = I->second;
    EndOff = RB.getMappedOffset(EndOff, true);
    StartOff = RB.getMappedOffset(StartOff);
  }

  
  // Adjust the end offset to the end of the last token, instead of being the
  // start of the last token.
  EndOff += Lexer::MeasureTokenLength(Range.getEnd(), *SourceMgr);
  
  return EndOff-StartOff;
}


unsigned Rewriter::getLocationOffsetAndFileID(SourceLocation Loc,
                                              unsigned &FileID) const {
  std::pair<unsigned,unsigned> V = SourceMgr->getDecomposedFileLoc(Loc);
  FileID = V.first;
  return V.second;
}


/// getEditBuffer - Get or create a RewriteBuffer for the specified FileID.
///
RewriteBuffer &Rewriter::getEditBuffer(unsigned FileID) {
  std::map<unsigned, RewriteBuffer>::iterator I =
    RewriteBuffers.lower_bound(FileID);
  if (I != RewriteBuffers.end() && I->first == FileID) 
    return I->second;
  I = RewriteBuffers.insert(I, std::make_pair(FileID, RewriteBuffer()));
  
  std::pair<const char*, const char*> MB = SourceMgr->getBufferData(FileID);
  I->second.Initialize(MB.first, MB.second);
  
  return I->second;
}

/// InsertText - Insert the specified string at the specified location in the
/// original buffer.
bool Rewriter::InsertText(SourceLocation Loc, const char *StrData,
                          unsigned StrLen, bool InsertAfter) {
  if (!isRewritable(Loc)) return true;
  unsigned FileID;
  unsigned StartOffs = getLocationOffsetAndFileID(Loc, FileID);
  getEditBuffer(FileID).InsertText(StartOffs, StrData, StrLen, InsertAfter);
  return false;
}

/// RemoveText - Remove the specified text region.
bool Rewriter::RemoveText(SourceLocation Start, unsigned Length) {
  if (!isRewritable(Start)) return true;
  unsigned FileID;
  unsigned StartOffs = getLocationOffsetAndFileID(Start, FileID);
  getEditBuffer(FileID).RemoveText(StartOffs, Length);
  return false;
}

/// ReplaceText - This method replaces a range of characters in the input
/// buffer with a new string.  This is effectively a combined "remove/insert"
/// operation.
bool Rewriter::ReplaceText(SourceLocation Start, unsigned OrigLength,
                           const char *NewStr, unsigned NewLength) {
  if (!isRewritable(Start)) return true;
  unsigned StartFileID;
  unsigned StartOffs = getLocationOffsetAndFileID(Start, StartFileID);
  
  getEditBuffer(StartFileID).ReplaceText(StartOffs, OrigLength,
                                         NewStr, NewLength);
  return false;
}

/// ReplaceStmt - This replaces a Stmt/Expr with another, using the pretty
/// printer to generate the replacement code.  This returns true if the input
/// could not be rewritten, or false if successful.
bool Rewriter::ReplaceStmt(Stmt *From, Stmt *To) {
  // Measaure the old text.
  int Size = getRangeSize(From->getSourceRange());
  if (Size == -1)
    return true;
  
  // Get the new text.
  std::ostringstream S;
  To->printPretty(S);
  const std::string &Str = S.str();

  ReplaceText(From->getLocStart(), Size, &Str[0], Str.size());
  return false;
}


