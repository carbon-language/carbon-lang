//===--- SourceLocation.h - Compact identifier for Source Files -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the SourceLocation class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SOURCELOCATION_H
#define LLVM_CLANG_SOURCELOCATION_H

namespace llvm {
namespace clang {
    
/// SourceLocation - This is a carefully crafted 32-bit identifier that encodes
/// a full include stack, line and column number information for a position in
/// an input translation unit.
class SourceLocation {
  unsigned ID;
public:
  enum {
    FileIDBits  = 14,
    FilePosBits = 32-FileIDBits
  };

  SourceLocation() : ID(0) {}  // 0 is an invalid FileID.
  
  /// SourceLocation constructor - Create a new SourceLocation object with the 
  /// specified FileID and FilePos.
  SourceLocation(unsigned FileID, unsigned FilePos) {
    // If a FilePos is larger than (1<<FilePosBits), the SourceManager makes
    // enough consequtive FileIDs that we have one for each chunk.
    if (FilePos >= (1 << FilePosBits)) {
      FileID += FilePos >> FilePosBits;
      FilePos &= (1 << FilePosBits)-1;
    }
    
    // FIXME: Find a way to handle out of FileID bits!  Maybe MaxFileID is an
    // escape of some sort?
    if (FileID >= (1 << FileIDBits))
      FileID = (1 << FileIDBits)-1;
    
    ID = (FileID << FilePosBits) | FilePos;
  }
  
  /// isValid - Return true if this is a valid SourceLocation object.  Invalid
  /// SourceLocations are often used when events have no corresponding location
  /// in the source (e.g. a diagnostic is required for a command line option).
  ///
  bool isValid() const { return ID != 0; }
  
  /// getFileID - Return the file identifier for this SourceLocation.  This
  /// FileID can be used with the SourceManager object to obtain an entire
  /// include stack for a file position reference.
  unsigned getFileID() const { return ID >> FilePosBits; }
  
  /// getRawFilePos - Return the byte offset from the start of the file-chunk
  /// referred to by FileID.  This method should not be used to get the offset
  /// from the start of the file, instead you should use
  /// SourceManager::getFilePos.  This method will be incorrect for large files.
  unsigned getRawFilePos() const { return ID & ((1 << FilePosBits)-1); }
  
  
  /// getRawEncoding - When a SourceLocation itself cannot be used, this returns
  /// an (opaque) 32-bit integer encoding for it.  This should only be passed
  /// to SourceLocation::getFromRawEncoding, it should not be inspected
  /// directly.
  unsigned getRawEncoding() const { return ID; }
  
  /// getFromRawEncoding - Turn a raw encoding of a SourceLocation object into
  /// a real SourceLocation.
  static SourceLocation getFromRawEncoding(unsigned Encoding) {
    SourceLocation X;
    X.ID = Encoding;
    return X;
  }
};

inline bool operator==(const SourceLocation &LHS, const SourceLocation &RHS) {
  return LHS.getRawEncoding() == RHS.getRawEncoding();
}

inline bool operator!=(const SourceLocation &LHS, const SourceLocation &RHS) {
  return !(LHS == RHS);
}

/// SourceRange - a trival tuple used to represent a source range.
class SourceRange {
  SourceLocation B;
  SourceLocation E;
public:
  SourceRange(): B(SourceLocation()), E(SourceLocation()) {}
  SourceRange(SourceLocation loc) : B(loc), E(loc) {}
  SourceRange(SourceLocation begin, SourceLocation end) : B(begin), E(end) {}
    
  SourceLocation Begin() const { return B; }
  SourceLocation End() const { return E; }
  
  bool isValid() const { return B.isValid() && E.isValid(); }
};
  
}  // end namespace clang
}  // end namespace llvm

#endif
