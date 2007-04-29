//===--- SourceBuffer.h - C Language Family Source Buffer -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the SourceBuffer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SOURCEBUFFER_H
#define LLVM_CLANG_SOURCEBUFFER_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
namespace clang {

/// SourceBuffer - This interface provides simple read-only access to the raw
/// bits in a source file in a memory efficient way.  In addition to basic
/// access to the characters in the file, this interface guarantees you can read
/// one character past the end of the file, and that this character will read as
/// '\0'.
class SourceBuffer {
  const char *BufferStart; // Start of the buffer.
  const char *BufferEnd;   // End of the buffer.

  /// MustDeleteBuffer - True if we allocated this buffer.  If so, the
  /// destructor must know the delete[] it.
  bool MustDeleteBuffer;
protected:
  SourceBuffer() : MustDeleteBuffer(false) {}
  void init(const char *BufStart, const char *BufEnd);
  void initCopyOf(const char *BufStart, const char *BufEnd);
public:
  virtual ~SourceBuffer();
  
  const char *getBufferStart() const { return BufferStart; }
  const char *getBufferEnd() const   { return BufferEnd; }
  unsigned getBufferSize() const { return BufferEnd-BufferStart; }
  
  /// getBufferIdentifier - Return an identifier for this buffer, typically the
  /// filename it was read from.
  virtual const char *getBufferIdentifier() const {
    return "Unknown buffer";
  }

  /// getFile - Open the specified file as a SourceBuffer, returning a new
  /// SourceBuffer if successful, otherwise returning null.  If FileSize is
  /// specified, this means that the client knows that the file exists and that
  /// it has the specified size.
  static SourceBuffer *getFile(const char *FilenameStart, unsigned FnSize,
                               int64_t FileSize = -1);

  /// getMemBuffer - Open the specified memory range as a SourceBuffer.  Note
  /// that EndPtr[0] must be a null byte and be accessible!
  static SourceBuffer *getMemBuffer(const char *StartPtr, const char *EndPtr,
                                    const char *BufferName = "");
  
  /// getNewMemBuffer - Allocate a new SourceBuffer of the specified size that
  /// is completely initialized to zeros.  Note that the caller should
  /// initialize the memory allocated by this method.  The memory is owned by
  /// the SourceBuffer object.
  static SourceBuffer *getNewMemBuffer(unsigned Size,
                                       const char *BufferName = "");
  
  /// getNewUninitMemBuffer - Allocate a new SourceBuffer of the specified size
  /// that is not initialized.  Note that the caller should initialize the
  /// memory allocated by this method.  The memory is owned by the SourceBuffer
  /// object.
  static SourceBuffer *getNewUninitMemBuffer(unsigned Size,
                                             const char *BufferName = "");
  
  /// getSTDIN - Read all of stdin into a file buffer, and return it.  This
  /// fails if stdin is empty.
  static SourceBuffer *getSTDIN();
};

} // end namespace clang
} // end namespace llvm

#endif
