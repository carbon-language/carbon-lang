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
namespace clang {
  
class SourceBuffer;
class SourceManager;
class FileEntry;
class IdentifierTokenInfo;

/// SrcMgr - Private classes that are part of the SourceManager implementation.
///
namespace SrcMgr {
  /// FileInfo - Once instance of this struct is kept for every file loaded or
  /// used.  This object owns the SourceBuffer object.
  struct FileInfo {
    /// Buffer - The actual buffer containing the characters from the input
    /// file.
    const SourceBuffer *Buffer;
    
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
  ///   1. Normal SourceBuffer (file).  These are represented by a "InfoRec *",
  ///      describing the source file, and a Chunk number, which factors into
  ///      the SourceLocation's offset from the start of the buffer.
  ///   2. Macro Expansions.  These indicate that the logical location is
  ///      totally different than the physical location.  The logical source
  ///      location is specified with an explicit SourceLocation object.
  ///
  struct FileIDInfo {
    enum FileIDType {
      NormalBuffer,
      MacroExpansion
    };
    
    /// The type of this FileID.
    FileIDType IDType : 2;
    
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
      
      /// MacroTokenLoc - This is the raw encoding of a SourceLocation which
      /// indicates the physical location of the macro token.
      unsigned MacroTokenLoc;
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
    /// the instantiation location, 
    static FileIDInfo getMacroExpansion(SourceLocation IL,
                                        SourceLocation TokenLoc) {
      FileIDInfo X;
      X.IDType = MacroExpansion;
      X.IncludeLoc = IL;
      X.u.MacroTokenLoc = TokenLoc.getRawEncoding();
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
/// memory.  This object owns the SourceBuffer objects for all of the loaded
/// files and assigns unique FileID's for each unique #include chain.
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
public:
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
  /// ownership of the SourceBuffer, so only pass a SourceBuffer to this once.
  unsigned createFileIDForMemBuffer(const SourceBuffer *Buffer) {
    return createFileID(createMemBufferInfoRec(Buffer), SourceLocation());
  }
  
  
  /// getBuffer - Return the buffer for the specified FileID.
  ///
  const SourceBuffer *getBuffer(unsigned FileID) {
    return getFileInfo(FileID)->Buffer;
  }
  
  /// getIncludeLoc - Return the location of the #include for the specified
  /// FileID.
  SourceLocation getIncludeLoc(unsigned FileID) const {
    assert(FileID-1 < FileIDs.size() && "Invalid FileID!");
    return FileIDs[FileID-1].IncludeLoc;
  }
  
  /// getFilePos - This (efficient) method returns the offset from the start of
  /// the file that the specified SourceLocation represents.
  unsigned getFilePos(SourceLocation IncludePos) const {
    assert(IncludePos.getFileID()-1 < FileIDs.size() && "Invalid FileID!");
    // If this file has been split up into chunks, factor in the chunk number
    // that the FileID references.
    unsigned ChunkNo=FileIDs[IncludePos.getFileID()-1].getNormalBufferChunkNo();
    return IncludePos.getRawFilePos() +
           (ChunkNo << SourceLocation::FilePosBits);
  }
  
  /// getCharacterData - Return a pointer to the start of the specified location
  /// in the appropriate SourceBuffer.  This returns null if it cannot be
  /// computed (e.g. invalid SourceLocation).
  const char *getCharacterData(SourceLocation SL) const;
  
  /// getColumnNumber - Return the column # for the specified include position.
  /// this is significantly cheaper to compute than the line number.  This
  /// returns zero if the column number isn't known.
  unsigned getColumnNumber(SourceLocation Loc) const;
  
  /// getLineNumber - Given a SourceLocation, return the physical line number
  /// for the position indicated.  This requires building and caching a table of
  /// line offsets for the SourceBuffer, so this is not cheap: use only when
  /// about to emit a diagnostic.
  unsigned getLineNumber(SourceLocation Loc);

  /// getFileEntryForFileID - Return the FileEntry record for the specified
  /// FileID if one exists.
  const FileEntry *getFileEntryForFileID(unsigned FileID) const {
    assert(FileID-1 < FileIDs.size() && "Invalid FileID!");
    return FileIDs[FileID-1].getNormalBufferInfo()->first;
  }
  
  /// PrintStats - Print statistics to stderr.
  ///
  void PrintStats() const;
private:
  /// createFileID - Create a new fileID for the specified InfoRec and include
  /// position.  This works regardless of whether the InfoRec corresponds to a
  /// file or some other input source.
  unsigned createFileID(const SrcMgr::InfoRec *File, SourceLocation IncludePos);
    
  /// getFileInfo - Create or return a cached FileInfo for the specified file.
  /// This returns null on failure.
  const SrcMgr::InfoRec *getInfoRec(const FileEntry *SourceFile);
  
  /// createMemBufferInfoRec - Create a new info record for the specified memory
  /// buffer.  This does no caching.
  const SrcMgr::InfoRec *createMemBufferInfoRec(const SourceBuffer *Buffer);

  const SrcMgr::InfoRec *getInfoRec(unsigned FileID) const {
    assert(FileID-1 < FileIDs.size() && "Invalid FileID!");
    return FileIDs[FileID-1].getNormalBufferInfo();
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
