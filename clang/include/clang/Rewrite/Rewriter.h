//===--- Rewriter.h - Code rewriting interface ------------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_REWRITER_H
#define LLVM_CLANG_REWRITER_H

#include "clang/Basic/SourceLocation.h"
#include <map>
#include <vector>

namespace clang {
  class SourceManager;
  class Rewriter;
  
/// SourceDelta - As code in the original input buffer is added and deleted,
/// SourceDelta records are used to keep track of how the input SourceLocation
/// object is mapped into the output buffer.
struct SourceDelta {
  unsigned FileLoc;
  int Delta;
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
  std::vector<char> Buffer;
public:

  
  
private:  // Methods only usable by Rewriter.
  
  /// Initialize - Start this rewrite buffer out with a copy of the unmodified
  /// input buffer.
  void Initialize(const char *BufStart, const char *BufEnd) {
    Buffer.assign(BufStart, BufEnd);
  }
  
  /// RemoveText - Remove the specified text.
  void RemoveText(unsigned OrigOffset, unsigned Size);
  
  /// InsertText - Insert some text at the specified point, where the offset in
  /// the buffer is specified relative to the original SourceBuffer.
  ///
  /// TODO: Consider a bool to indicate whether the text is inserted 'before' or
  /// after the atomic point: i.e. whether the atomic point is moved to after
  /// the inserted text or not.
  void InsertText(unsigned OrigOffset, const char *StrData, unsigned StrLen);
};
  

/// Rewriter - This is the main interface to the rewrite buffers.  Its primary
/// job is to dispatch high-level requests to the low-level RewriteBuffers that
/// are involved.
class Rewriter {
  SourceManager &SourceMgr;
  
  std::map<unsigned, RewriteBuffer> RewriteBuffers;
public:
  explicit Rewriter(SourceManager &SM) : SourceMgr(SM) {}
  
  /// isRewritable - Return true if this location is a raw file location, which
  /// is rewritable.  Locations from macros, etc are not rewritable.
  static bool isRewritable(SourceLocation Loc) {
    return Loc.isFileID();
  }
  
  /// InsertText - Insert the specified string at the specified location in the
  /// original buffer.  This method is only valid on rewritable source
  /// locations.
  void InsertText(SourceLocation Loc, const char *StrData, unsigned StrLen);
  
  /// RemoveText - Remove the specified text region.  This method is only valid
  /// on rewritable source locations.
  void RemoveText(SourceLocation Start, SourceLocation End);
  
  
  void ReplaceText(SourceLocation Start, unsigned OrigLength,
                   const char *NewStr, unsigned NewLength);
  
  // TODO: Replace Stmt/Expr with another.  Return bool to indicate whether the
  // locations were rewritable.
  
  /// getRewriteBufferFor - Return the rewrite buffer for the specified FileID.
  /// If no modification has been made to it, return null.
  const RewriteBuffer *getRewriteBufferFor(unsigned FileID) const {
    std::map<unsigned, RewriteBuffer>::const_iterator I =
      RewriteBuffers.find(FileID);
    return I == RewriteBuffers.end() ? 0 : &I->second;
  }
private:
  RewriteBuffer &getEditBuffer(unsigned FileID);
};
  
} // end namespace clang

#endif
