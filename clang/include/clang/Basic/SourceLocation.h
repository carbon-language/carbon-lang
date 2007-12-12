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

#include <cassert>
#include "llvm/Bitcode/SerializationFwd.h"

namespace clang {
  
class SourceManager;
    
/// SourceLocation - This is a carefully crafted 32-bit identifier that encodes
/// a full include stack, line and column number information for a position in
/// an input translation unit.
class SourceLocation {
  unsigned ID;
public:
  enum {
    // FileID Layout:
    // bit 31: 0 -> FileID, 1 -> MacroID (invalid for FileID)
    //     30...17 -> FileID of source location, index into SourceManager table.
    FileIDBits  = 14,
    //      0...16 -> Index into the chunk of the specified FileID.
    FilePosBits = 32-1-FileIDBits,
    
    // MacroID Layout:
    // bit 31: 1 -> MacroID, 0 -> FileID (invalid for MacroID)

    // bit 30: 1 -> Start of macro expansion marker.
    MacroStartOfExpansionBit = 30,
    // bit 29: 1 -> End of macro expansion marker.
    MacroEndOfExpansionBit = 29,
    // bits 28...9 -> MacroID number.
    MacroIDBits       = 20,
    // bits 8...0  -> Macro Physical offset
    MacroPhysOffsBits = 9,
    
    
    // Useful constants.
    ChunkSize = (1 << FilePosBits)
  };

  SourceLocation() : ID(0) {}  // 0 is an invalid FileID.
  
  bool isFileID() const { return (ID >> 31) == 0; }
  bool isMacroID() const { return (ID >> 31) != 0; }
  
  /// isValid - Return true if this is a valid SourceLocation object.  Invalid
  /// SourceLocations are often used when events have no corresponding location
  /// in the source (e.g. a diagnostic is required for a command line option).
  ///
  bool isValid() const { return ID != 0; }
  bool isInvalid() const { return ID == 0; }
  
  static SourceLocation getFileLoc(unsigned FileID, unsigned FilePos) {
    SourceLocation L;
    // If a FilePos is larger than (1<<FilePosBits), the SourceManager makes
    // enough consequtive FileIDs that we have one for each chunk.
    if (FilePos >= ChunkSize) {
      FileID += FilePos >> FilePosBits;
      FilePos &= ChunkSize-1;
    }
    
    // FIXME: Find a way to handle out of FileID bits!  Maybe MaxFileID is an
    // escape of some sort?
    assert(FileID < (1 << FileIDBits) && "Out of fileid's");
    
    L.ID = (FileID << FilePosBits) | FilePos;
    return L;
  }
  
  static bool isValidMacroPhysOffs(int Val) {
    if (Val >= 0)
      return Val < (1 << (MacroPhysOffsBits-1));
    return -Val < (1 << (MacroPhysOffsBits-1));
  }
  
  static SourceLocation getMacroLoc(unsigned MacroID, int PhysOffs,
                                    bool isExpansionStart, bool isExpansionEnd){
    assert(MacroID < (1 << MacroIDBits) && "Too many macros!");
    assert(isValidMacroPhysOffs(PhysOffs) && "Physoffs too large!");
    
    // Mask off sign bits.
    PhysOffs &= (1 << MacroPhysOffsBits)-1;
    
    SourceLocation L;
    L.ID = (1 << 31) |
           (isExpansionStart << MacroStartOfExpansionBit) |
           (isExpansionEnd << MacroEndOfExpansionBit) |
           (MacroID << MacroPhysOffsBits) |
           PhysOffs;
    return L;
  }
  
  
  /// getFileID - Return the file identifier for this SourceLocation.  This
  /// FileID can be used with the SourceManager object to obtain an entire
  /// include stack for a file position reference.
  unsigned getFileID() const {
    assert(isFileID() && "can't get the file id of a non-file sloc!");
    return ID >> FilePosBits;
  }
  
  /// getRawFilePos - Return the byte offset from the start of the file-chunk
  /// referred to by FileID.  This method should not be used to get the offset
  /// from the start of the file, instead you should use
  /// SourceManager::getFilePos.  This method will be incorrect for large files.
  unsigned getRawFilePos() const { 
    assert(isFileID() && "can't get the file id of a non-file sloc!");
    return ID & (ChunkSize-1);
  }

  unsigned getMacroID() const {
    assert(isMacroID() && "Is not a macro id!");
    return (ID >> MacroPhysOffsBits) & ((1 << MacroIDBits)-1);
  }
  
  int getMacroPhysOffs() const {
    assert(isMacroID() && "Is not a macro id!");
    int Val = ID & ((1 << MacroPhysOffsBits)-1);
    // Sign extend it properly.
    unsigned ShAmt = sizeof(int)*8 - MacroPhysOffsBits;
    return (Val << ShAmt) >> ShAmt;
  }
  
  /// getFileLocWithOffset - Return a source location with the specified offset
  /// from this file SourceLocation.
  SourceLocation getFileLocWithOffset(int Offset) const {
    unsigned FileID = getFileID();
    Offset += getRawFilePos();
    // Handle negative offsets correctly.
    while (Offset < 0) {
      --FileID;
      Offset += ChunkSize;
    }
    return getFileLoc(FileID, Offset);
  }
  
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
  
  /// Emit - Emit this SourceLocation object to Bitcode.
  void Emit(llvm::Serializer& S) const;
  
  /// ReadVal - Read a SourceLocation object from Bitcode.
  static SourceLocation ReadVal(llvm::Deserializer& D);
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
    
  SourceLocation getBegin() const { return B; }
  SourceLocation getEnd() const { return E; }
  
  void setBegin(SourceLocation b) { B = b; }
  void setEnd(SourceLocation e) { E = e; }
  
  bool isValid() const { return B.isValid() && E.isValid(); }
  
  /// Emit - Emit this SourceRange object to Bitcode.
  void Emit(llvm::Serializer& S) const;    

  /// ReadVal - Read a SourceRange object from Bitcode.
  static SourceRange ReadVal(llvm::Deserializer& D);
};
  
/// FullSourceLoc - A tuple containing both a SourceLocation
///  and its associated SourceManager.  Useful for argument passing to functions
///  that expect both objects.
class FullSourceLoc {
  SourceLocation Loc;
  const SourceManager* SrcMgr;
public:
  // Creates a FullSourceLoc where isValid() returns false.
  explicit FullSourceLoc() 
    : Loc(SourceLocation()), SrcMgr((SourceManager*) 0) {}

  explicit FullSourceLoc(SourceLocation loc, const SourceManager& smgr) 
    : Loc(loc), SrcMgr(&smgr) {
    assert (loc.isValid() && "SourceLocation must be valid!");
  }
    
  bool isValid() const { return Loc.isValid(); }
  
  SourceLocation getSourceLocation() const { return Loc; }
  
  const SourceManager& getManager() const {
    assert (SrcMgr && "SourceManager is NULL.");
    return *SrcMgr;
  }
};

}  // end namespace clang

#endif
