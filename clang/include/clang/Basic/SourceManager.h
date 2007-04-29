//===--- SourceManager.h - Track and cache source files ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the SourceManager interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SOURCEMANAGER_H
#define LLVM_CLANG_SOURCEMANAGER_H

#include "clang/Basic/SourceLocation.h"
#include <vector>
#include <map>
#include <list>

namespace llvm {
class MemoryBuffer;
  
namespace clang {
  
class SourceManager;
class FileEntry;
class IdentifierTokenInfo;

/// SrcMgr - Private classes that are part of the SourceManager implementation.
///
namespace SrcMgr {
  /// FileInfo - Once instance of this struct is kept for every file loaded or
  /// used.  This object owns the MemoryBuffer object.
  struct FileInfo {
    /// Buffer - The actual buffer containing the characters from the input
    /// file.
    const MemoryBuffer *Buffer;
    
    /// SourceLineCache - A new[]'d array of offsets for each source line.  This
    /// is lazily computed.
    ///
    unsigned *SourceLineCache;
    
    /// NumLines - The number of lines in this FileInfo.  This is only valid if
    /// SourceLineCache is non-null.
    unsigned NumLines;
  };
  
  typedef std::pair<const FileEntry * const, FileInfo> InfoRec;

  /// FileIDInfo - Information about a FileID, basically just the logical file
  /// that it represents and include stack information.  A SourceLocation is a
  /// byte offset from the start of this.
  ///
  /// FileID's are used to compute the location of a character in memory as well
  /// as the logical source location, which can be differ from the physical
  /// location.  It is different when #line's are active or when macros have
  /// been expanded.
  ///
  /// Each FileID has include stack information, indicating where it came from.
  /// For the primary translation unit, it comes from SourceLocation() aka 0.
  ///
  /// There are three types of FileID's:
  ///   1. Normal MemoryBuffer (file).  These are represented by a "InfoRec *",
  ///      describing the source file, and a Chunk number, which factors into
  ///      the SourceLocation's offset from the start of the buffer.
  ///   2. Macro Expansions.  These indicate that the logical location is
  ///      totally different than the physical location.  The logical source
  ///      location is specified by the IncludeLoc.  The physical location is
  ///      the FilePos of the token's SourceLocation combined with the FileID
  ///      from MacroTokenFileID.
  ///
  struct FileIDInfo {
    enum FileIDType {
      NormalBuffer,
      MacroExpansion
    };
    
    /// The type of this FileID.
    FileIDType IDType;
    
    /// IncludeLoc - The location of the #include that brought in this file.
    /// This SourceLocation object has a FileId of 0 for the main file.
    SourceLocation IncludeLoc;
    
    /// This union is discriminated by IDType.
    ///
    union {
      struct NormalBufferInfo {
        /// ChunkNo - Really large buffers are broken up into chunks that are
        /// each (1 << SourceLocation::FilePosBits) in size.  This specifies the
        /// chunk number of this FileID.
        unsigned ChunkNo;
        
        /// FileInfo - Information about the source buffer itself.
        ///
        const InfoRec *Info;
      } NormalBuffer;
      
      /// MacroTokenFileID - This is the File ID that contains the characters
      /// that make up the expanded token.
      unsigned MacroTokenFileID;
    } u;
    
    /// getNormalBuffer - Return a FileIDInfo object for a normal buffer
    /// reference.
    static FileIDInfo getNormalBuffer(SourceLocation IL, unsigned CN,
                                      const InfoRec *Inf) {
      FileIDInfo X;
      X.IDType = NormalBuffer;
      X.IncludeLoc = IL;
      X.u.NormalBuffer.ChunkNo = CN;
      X.u.NormalBuffer.Info = Inf;
      return X;
    }
    
    /// getMacroExpansion - Return a FileID for a macro expansion.  IL specifies
    /// the instantiation location, and MacroFID specifies the FileID that the
    /// token's characters come from. 
    static FileIDInfo getMacroExpansion(SourceLocation IL,
                                        unsigned MacroFID) {
      FileIDInfo X;
      X.IDType = MacroExpansion;
      X.IncludeLoc = IL;
      X.u.MacroTokenFileID = MacroFID;
      return X;
    }
    
    unsigned getNormalBufferChunkNo() const {
      assert(IDType == NormalBuffer && "Not a normal buffer!");
      return u.NormalBuffer.ChunkNo;
    }

    const InfoRec *getNormalBufferInfo() const {
      assert(IDType == NormalBuffer && "Not a normal buffer!");
      return u.NormalBuffer.Info;
    }
  };
}  // end SrcMgr namespace.


/// SourceManager - This file handles loading and caching of source files into
/// memory.  This object owns the MemoryBuffer objects for all of the loaded
/// files and assigns unique FileID's for each unique #include chain.
///
/// The SourceManager can be queried for information about SourceLocation
/// objects, turning them into either physical or logical locations.  Physical
/// locations represent where the bytes corresponding to a token came from and
/// logical locations represent where the location is in the user's view.  In
/// the case of a macro expansion, for example, the physical location indicates
/// where the expanded token came from and the logical location specifies where
/// it was expanded.  Logical locations are also influenced by #line directives,
/// etc.
class SourceManager {
  /// FileInfos - Memoized information about all of the files tracked by this
  /// SourceManager.
  std::map<const FileEntry *, SrcMgr::FileInfo> FileInfos;
  
  /// MemBufferInfos - Information about various memory buffers that we have
  /// read in.  This is a list, instead of a vector, because we need pointers to
  /// the FileInfo objects to be stable.
  std::list<SrcMgr::InfoRec> MemBufferInfos;
  
  /// FileIDs - Information about each FileID.  FileID #0 is not valid, so all
  /// entries are off by one.
  std::vector<SrcMgr::FileIDInfo> FileIDs;
  
  /// LastInstantiationLoc_* - Cache the last instantiation request for fast
  /// lookup.  Macros often want many tokens instantated at the same location.
  SourceLocation LastInstantiationLoc_InstantLoc;
  unsigned       LastInstantiationLoc_MacroFID;
  unsigned       LastInstantiationLoc_Result;
public:
  SourceManager() { LastInstantiationLoc_MacroFID = ~0U; }
  ~SourceManager();
  
  /// createFileID - Create a new FileID that represents the specified file
  /// being #included from the specified IncludePosition.  This returns 0 on
  /// error and translates NULL into standard input.
  unsigned createFileID(const FileEntry *SourceFile, SourceLocation IncludePos){
    const SrcMgr::InfoRec *IR = getInfoRec(SourceFile);
    if (IR == 0) return 0;    // Error opening file?
    return createFileID(IR, IncludePos);
  }
  
  /// createFileIDForMemBuffer - Create a new FileID that represents the
  /// specified memory buffer.  This does no caching of the buffer and takes
  /// ownership of the MemoryBuffer, so only pass a MemoryBuffer to this once.
  unsigned createFileIDForMemBuffer(const MemoryBuffer *Buffer) {
    return createFileID(createMemBufferInfoRec(Buffer), SourceLocation());
  }
  
  /// getInstantiationLoc - Return a new SourceLocation that encodes the fact
  /// that a token from physloc PhysLoc should actually be referenced from
  /// InstantiationLoc.
  SourceLocation getInstantiationLoc(SourceLocation PhysLoc,
                                     SourceLocation InstantiationLoc);
  
  /// getBuffer - Return the buffer for the specified FileID.
  ///
  const MemoryBuffer *getBuffer(unsigned FileID) const {
    return getFileInfo(FileID)->Buffer;
  }
  
  /// getIncludeLoc - Return the location of the #include for the specified
  /// FileID.
  SourceLocation getIncludeLoc(unsigned FileID) const;
  
  /// getFilePos - This (efficient) method returns the offset from the start of
  /// the file that the specified SourceLocation represents.  This returns the
  /// location of the physical character data, not the logical file position.
  unsigned getFilePos(SourceLocation Loc) const {
    const SrcMgr::FileIDInfo *FIDInfo = getFIDInfo(Loc.getFileID());

    // For Macros, the physical loc is specified by the MacroTokenFileID.
    if (FIDInfo->IDType == SrcMgr::FileIDInfo::MacroExpansion)
      FIDInfo = &FileIDs[FIDInfo->u.MacroTokenFileID-1];
    
    // If this file has been split up into chunks, factor in the chunk number
    // that the FileID references.
    unsigned ChunkNo = FIDInfo->getNormalBufferChunkNo();
    return Loc.getRawFilePos() + (ChunkNo << SourceLocation::FilePosBits);
  }
  
  /// getCharacterData - Return a pointer to the start of the specified location
  /// in the appropriate MemoryBuffer.
  const char *getCharacterData(SourceLocation SL) const;
  
  /// getColumnNumber - Return the column # for the specified include position.
  /// this is significantly cheaper to compute than the line number.  This
  /// returns zero if the column number isn't known.
  unsigned getColumnNumber(SourceLocation Loc) const;
  
  /// getLineNumber - Given a SourceLocation, return the physical line number
  /// for the position indicated.  This requires building and caching a table of
  /// line offsets for the MemoryBuffer, so this is not cheap: use only when
  /// about to emit a diagnostic.
  unsigned getLineNumber(SourceLocation Loc);
  
  /// getSourceFilePos - This method returns the *logical* offset from the start
  /// of the file that the specified SourceLocation represents.  This returns
  /// the location of the *logical* character data, not the physical file
  /// position.  In the case of macros, for example, this returns where the
  /// macro was instantiated, not where the characters for the macro can be
  /// found.
  unsigned getSourceFilePos(SourceLocation Loc) const;
    
  /// getSourceName - This method returns the name of the file or buffer that
  /// the SourceLocation specifies.  This can be modified with #line directives,
  /// etc.
  std::string getSourceName(SourceLocation Loc);

  /// getFileEntryForFileID - Return the FileEntry record for the specified
  /// FileID if one exists.
  const FileEntry *getFileEntryForFileID(unsigned FileID) const {
    assert(FileID-1 < FileIDs.size() && "Invalid FileID!");
    return FileIDs[FileID-1].getNormalBufferInfo()->first;
  }
  
  /// Given a SourceLocation object, return the logical location referenced by
  /// the ID.  This logical location is subject to #line directives, etc.
  SourceLocation getLogicalLoc(SourceLocation Loc) const {
    if (Loc.getFileID() == 0) return Loc;
    
    const SrcMgr::FileIDInfo *FIDInfo = getFIDInfo(Loc.getFileID());
    if (FIDInfo->IDType == SrcMgr::FileIDInfo::MacroExpansion)
      return FIDInfo->IncludeLoc;
    return Loc;
  }
  
  /// getPhysicalLoc - Given a SourceLocation object, return the physical
  /// location referenced by the ID.
  SourceLocation getPhysicalLoc(SourceLocation Loc) const {
    if (Loc.getFileID() == 0) return Loc;
    
    // For Macros, the physical loc is specified by the MacroTokenFileID.
    const SrcMgr::FileIDInfo *FIDInfo = getFIDInfo(Loc.getFileID());
    if (FIDInfo->IDType == SrcMgr::FileIDInfo::MacroExpansion)
      return SourceLocation(FIDInfo->u.MacroTokenFileID,
                            Loc.getRawFilePos());
    return Loc;
  }
  
  /// PrintStats - Print statistics to stderr.
  ///
  void PrintStats() const;
private:
  /// createFileID - Create a new fileID for the specified InfoRec and include
  /// position.  This works regardless of whether the InfoRec corresponds to a
  /// file or some other input source.
  unsigned createFileID(const SrcMgr::InfoRec *File, SourceLocation IncludePos);
    
  /// getInfoRec - Create or return a cached FileInfo for the specified file.
  /// This returns null on failure.
  const SrcMgr::InfoRec *getInfoRec(const FileEntry *SourceFile);
  
  /// createMemBufferInfoRec - Create a new info record for the specified memory
  /// buffer.  This does no caching.
  const SrcMgr::InfoRec *createMemBufferInfoRec(const MemoryBuffer *Buffer);

  const SrcMgr::FileIDInfo *getFIDInfo(unsigned FileID) const {
    assert(FileID-1 < FileIDs.size() && "Invalid FileID!");
    return &FileIDs[FileID-1];
  }
    
  /// Return the InfoRec structure for the specified FileID.  This is always the
  /// physical reference for the ID.
  const SrcMgr::InfoRec *getInfoRec(const SrcMgr::FileIDInfo *FIDInfo) const {
    // For Macros, the physical loc is specified by the MacroTokenFileID.
    if (FIDInfo->IDType == SrcMgr::FileIDInfo::MacroExpansion)
      FIDInfo = &FileIDs[FIDInfo->u.MacroTokenFileID-1];
    return FIDInfo->getNormalBufferInfo();
  }
  const SrcMgr::InfoRec *getInfoRec(unsigned FileID) const {
    return getInfoRec(getFIDInfo(FileID));
  }
  
  SrcMgr::FileInfo *getFileInfo(const SrcMgr::FileIDInfo *FIDInfo) const {
    if (const SrcMgr::InfoRec *IR = getInfoRec(FIDInfo))
      return const_cast<SrcMgr::FileInfo *>(&IR->second);
    return 0;
  }
  SrcMgr::FileInfo *getFileInfo(unsigned FileID) const {
    if (const SrcMgr::InfoRec *IR = getInfoRec(FileID))
      return const_cast<SrcMgr::FileInfo *>(&IR->second);
    return 0;
  }
  SrcMgr::FileInfo *getFileInfo(const FileEntry *SourceFile) {
    if (const SrcMgr::InfoRec *IR = getInfoRec(SourceFile))
      return const_cast<SrcMgr::FileInfo *>(&IR->second);
    return 0;
  }
};


}  // end namespace clang
}  // end namespace llvm

#endif
