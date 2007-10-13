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

/// getMappedOffset - Given an offset into the original SourceBuffer that this
/// RewriteBuffer is based on, map it into the offset space of the
/// RewriteBuffer.
unsigned RewriteBuffer::getMappedOffset(unsigned OrigOffset,
                                        bool AfterInserts) const {
  unsigned ResultOffset = OrigOffset;
  unsigned DeltaIdx = 0;
  
  // Move past any deltas that are relevant.
  // FIXME: binary search.
  for (; DeltaIdx != Deltas.size() && 
       Deltas[DeltaIdx].FileLoc < OrigOffset; ++DeltaIdx)
    ResultOffset += Deltas[DeltaIdx].Delta;

  if (AfterInserts && DeltaIdx != Deltas.size() && 
      OrigOffset == Deltas[DeltaIdx].FileLoc)
    ResultOffset += Deltas[DeltaIdx].Delta;
  return ResultOffset;
}

/// AddDelta - When a change is made that shifts around the text buffer, this
/// method is used to record that info.
void RewriteBuffer::AddDelta(unsigned OrigOffset, int Change) {
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
  if (Deltas[DeltaIdx].Delta)
    Deltas.erase(Deltas.begin()+DeltaIdx);
}


void RewriteBuffer::RemoveText(unsigned OrigOffset, unsigned Size) {
  // Nothing to remove, exit early.
  if (Size == 0) return;

  unsigned RealOffset = getMappedOffset(OrigOffset, true);
  assert(RealOffset+Size < Buffer.size() && "Invalid location");
  
  // Remove the dead characters.
  Buffer.erase(Buffer.begin()+RealOffset, Buffer.begin()+RealOffset+Size);

  // Add a delta so that future changes are offset correctly.
  AddDelta(OrigOffset, -Size);
}

void RewriteBuffer::InsertText(unsigned OrigOffset,
                               const char *StrData, unsigned StrLen) {
  if (StrLen == 0) return;
  // FIXME:
}

/// ReplaceText - This method replaces a range of characters in the input
/// buffer with a new string.  This is effectively a combined "remove/insert"
/// operation.
void RewriteBuffer::ReplaceText(unsigned OrigOffset, unsigned OrigLength,
                                const char *NewStr, unsigned NewLength) {
  RemoveText(OrigOffset, OrigLength);
  return;
  
  unsigned MappedOffs = getMappedOffset(OrigOffset);
  // TODO: FIXME location.
  assert(OrigOffset+OrigLength <= Buffer.size() && "Invalid location");
  if (OrigLength == NewLength) {
    // If replacing without shifting around, just overwrite the text.
    memcpy(&Buffer[OrigOffset], NewStr, NewLength);
    return;
  }
}


//===----------------------------------------------------------------------===//
// Rewriter class
//===----------------------------------------------------------------------===//

unsigned Rewriter::getLocationOffsetAndFileID(SourceLocation Loc,
                                              unsigned &FileID) const {
  std::pair<unsigned,unsigned> V = SourceMgr.getDecomposedFileLoc(Loc);
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
  
  std::pair<const char*, const char*> MB = SourceMgr.getBufferData(FileID);
  I->second.Initialize(MB.first, MB.second);
  
  return I->second;
}


void Rewriter::ReplaceText(SourceLocation Start, unsigned OrigLength,
                           const char *NewStr, unsigned NewLength) {
  assert(isRewritable(Start) && "Not a rewritable location!");
  unsigned StartFileID;
  unsigned StartOffs = getLocationOffsetAndFileID(Start, StartFileID);
  
  getEditBuffer(StartFileID).ReplaceText(StartOffs, OrigLength,
                                         NewStr, NewLength);
}
