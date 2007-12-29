//===--- Rewriter.h - Code rewriting interface ------------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_REWRITER_H
#define LLVM_CLANG_REWRITER_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Rewrite/RewriteRope.h"
#include <map>
#include <vector>

namespace clang {
  class SourceManager;
  class Rewriter;
  class Stmt;
  
/// SourceDelta - As code in the original input buffer is added and deleted,
/// SourceDelta records are used to keep track of how the input SourceLocation
/// object is mapped into the output buffer.
struct SourceDelta {
  unsigned FileLoc;
  int Delta;
  
  static SourceDelta get(unsigned Loc, int D) {
    SourceDelta Delta;
    Delta.FileLoc = Loc;
    Delta.Delta = D;
    return Delta;
  }
};


/// RewriteBuffer - As code is rewritten, SourceBuffer's from the original
/// input with modifications get a new RewriteBuffer associated with them.  The
/// RewriteBuffer captures the modified text itself as well as information used
/// to map between SourceLocation's in the original input and offsets in the
/// RewriteBuffer.  For example, if text is inserted into the buffer, any
/// locations after the insertion point have to be mapped.
class RewriteBuffer {
  friend class Rewriter;
  /// Deltas - Keep track of all the deltas in the source code due to insertions
  /// and deletions.  These are kept in sorted order based on the FileLoc.
  std::vector<SourceDelta> Deltas;
  
  /// Buffer - This is the actual buffer itself.  Note that using a vector or
  /// string is a horribly inefficient way to do this, we should use a rope
  /// instead.
  typedef RewriteRope BufferTy;
  BufferTy Buffer;
public:
  typedef BufferTy::const_iterator iterator;
  iterator begin() const { return Buffer.begin(); }
  iterator end() const { return Buffer.end(); }
  
private:  // Methods only usable by Rewriter.
  
  /// Initialize - Start this rewrite buffer out with a copy of the unmodified
  /// input buffer.
  void Initialize(const char *BufStart, const char *BufEnd) {
    Buffer.assign(BufStart, BufEnd);
  }
  
  /// getMappedOffset - Given an offset into the original SourceBuffer that this
  /// RewriteBuffer is based on, map it into the offset space of the
  /// RewriteBuffer.  If AfterInserts is true and if the OrigOffset indicates a
  /// position where text is inserted, the location returned will be after any
  /// inserted text at the position.
  unsigned getMappedOffset(unsigned OrigOffset, bool AfterInserts = false)const;
  
  
  /// AddDelta - When a change is made that shifts around the text buffer, this
  /// method is used to record that info.
  void AddDelta(unsigned OrigOffset, int Change);
  
  /// RemoveText - Remove the specified text.
  void RemoveText(unsigned OrigOffset, unsigned Size);
  
  /// InsertText - Insert some text at the specified point, where the offset in
  /// the buffer is specified relative to the original SourceBuffer.
  ///
  /// TODO: Consider a bool to indicate whether the text is inserted 'before' or
  /// after the atomic point: i.e. whether the atomic point is moved to after
  /// the inserted text or not.
  void InsertText(unsigned OrigOffset, const char *StrData, unsigned StrLen);
  
  /// ReplaceText - This method replaces a range of characters in the input
  /// buffer with a new string.  This is effectively a combined "remove/insert"
  /// operation.
  void ReplaceText(unsigned OrigOffset, unsigned OrigLength,
                   const char *NewStr, unsigned NewLength);
  
};
  

/// Rewriter - This is the main interface to the rewrite buffers.  Its primary
/// job is to dispatch high-level requests to the low-level RewriteBuffers that
/// are involved.
class Rewriter {
  SourceManager *SourceMgr;
  
  std::map<unsigned, RewriteBuffer> RewriteBuffers;
public:
  explicit Rewriter(SourceManager &SM) : SourceMgr(&SM) {}
  explicit Rewriter() : SourceMgr(0) {}
  
  void setSourceMgr(SourceManager &SM) { SourceMgr = &SM; }
  
  /// isRewritable - Return true if this location is a raw file location, which
  /// is rewritable.  Locations from macros, etc are not rewritable.
  static bool isRewritable(SourceLocation Loc) {
    return Loc.isFileID();
  }

  /// getRangeSize - Return the size in bytes of the specified range if they
  /// are in the same file.  If not, this returns -1.
  int getRangeSize(SourceRange Range) const;
  
  /// InsertText - Insert the specified string at the specified location in the
  /// original buffer.  This method is only valid on rewritable source
  /// locations.
  void InsertText(SourceLocation Loc, const char *StrData, unsigned StrLen);
  
  /// RemoveText - Remove the specified text region.  This method is only valid
  /// on a rewritable source location.
  void RemoveText(SourceLocation Start, unsigned Length);
  
  /// ReplaceText - This method replaces a range of characters in the input
  /// buffer with a new string.  This is effectively a combined "remove/insert"
  /// operation.
  void ReplaceText(SourceLocation Start, unsigned OrigLength,
                   const char *NewStr, unsigned NewLength);
  
  /// ReplaceStmt - This replaces a Stmt/Expr with another, using the pretty
  /// printer to generate the replacement code.  This returns true if the input
  /// could not be rewritten, or false if successful.
  bool ReplaceStmt(Stmt *From, Stmt *To);
  
  /// getRewriteBufferFor - Return the rewrite buffer for the specified FileID.
  /// If no modification has been made to it, return null.
  const RewriteBuffer *getRewriteBufferFor(unsigned FileID) const {
    std::map<unsigned, RewriteBuffer>::const_iterator I =
      RewriteBuffers.find(FileID);
    return I == RewriteBuffers.end() ? 0 : &I->second;
  }
private:
  RewriteBuffer &getEditBuffer(unsigned FileID);
  unsigned getLocationOffsetAndFileID(SourceLocation Loc,
                                      unsigned &FileID) const;
};
  
} // end namespace clang

#endif
