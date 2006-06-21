//===--- SourceManager.cpp - Track and cache source files -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the SourceManager interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceBuffer.h"
#include "llvm/System/Path.h"
#include <algorithm>
#include <iostream>
using namespace llvm;
using namespace clang;
using namespace SrcMgr;

SourceManager::~SourceManager() {
  for (std::map<const FileEntry *, FileInfo>::iterator I = FileInfos.begin(),
       E = FileInfos.end(); I != E; ++I) {
    delete I->second.Buffer;
    delete[] I->second.SourceLineCache;
  }
  
  for (std::list<InfoRec>::iterator I = MemBufferInfos.begin(), 
       E = MemBufferInfos.end(); I != E; ++I) {
    delete I->second.Buffer;
    delete[] I->second.SourceLineCache;
  }
}

/// getFileInfo - Create or return a cached FileInfo for the specified file.
///
const InfoRec *
SourceManager::getInfoRec(const FileEntry *FileEnt) {
  assert(FileEnt && "Didn't specify a file entry to use?");
  // Do we already have information about this file?
  std::map<const FileEntry *, FileInfo>::iterator I = 
    FileInfos.lower_bound(FileEnt);
  if (I != FileInfos.end() && I->first == FileEnt)
    return &*I;
  
  // Nope, get information.
  const SourceBuffer *File;
  try {
    File = clang::SourceBuffer::getFile(FileEnt);
    if (File == 0)
      return 0;
  } catch (...) {
    return 0;
  }

  const InfoRec &Entry =
    *FileInfos.insert(I, std::make_pair(FileEnt, FileInfo()));
  FileInfo &Info = const_cast<FileInfo &>(Entry.second);

  Info.Buffer = File;
  Info.SourceLineCache = 0;
  Info.NumLines = 0;
  return &Entry;
}


/// createMemBufferInfoRec - Create a new info record for the specified memory
/// buffer.  This does no caching.
const InfoRec *
SourceManager::createMemBufferInfoRec(const SourceBuffer *Buffer) {
  // Add a new info record to the MemBufferInfos list and return it.
  FileInfo FI;
  FI.Buffer = Buffer;
  FI.SourceLineCache = 0;
  FI.NumLines = 0;
  MemBufferInfos.push_back(InfoRec(0, FI));
  return &MemBufferInfos.back();
}


/// createFileID - Create a new fileID for the specified InfoRec and include
/// position.  This works regardless of whether the InfoRec corresponds to a
/// file or some other input source.
unsigned SourceManager::createFileID(const InfoRec *File,
                                     SourceLocation IncludePos) {
  // If FileEnt is really large (e.g. it's a large .i file), we may not be able
  // to fit an arbitrary position in the file in the FilePos field.  To handle
  // this, we create one FileID for each chunk of the file that fits in a
  // FilePos field.
  unsigned FileSize = File->second.Buffer->getBufferSize();
  if (FileSize+1 < (1 << SourceLocation::FilePosBits)) {
    FileIDs.push_back(FileIDInfo::getNormalBuffer(IncludePos, 0, File));
    return FileIDs.size();
  }
  
  // Create one FileID for each chunk of the file.
  unsigned Result = FileIDs.size()+1;

  unsigned ChunkNo = 0;
  while (1) {
    FileIDs.push_back(FileIDInfo::getNormalBuffer(IncludePos, ChunkNo++, File));

    if (FileSize+1 < (1 << SourceLocation::FilePosBits)) break;
    FileSize -= (1 << SourceLocation::FilePosBits);
  }

  return Result;
}

/// createFileIDForMacroExp - Return a new FileID for a macro expansion at
/// SourcePos, where the macro token character came from PhysicalFileID.
///
unsigned SourceManager::createFileIDForMacroExp(SourceLocation SourcePos, 
                                                unsigned PhysicalFileID) {
  FileIDs.push_back(FileIDInfo::getMacroExpansion(SourcePos, PhysicalFileID));
  return FileIDs.size();
}


/// getCharacterData - Return a pointer to the start of the specified location
/// in the appropriate SourceBuffer.  This returns null if it cannot be
/// computed (e.g. invalid SourceLocation).
const char *SourceManager::getCharacterData(SourceLocation SL) const {
  if (unsigned FileID = SL.getFileID())
    return getFileInfo(FileID)->Buffer->getBufferStart() + getFilePos(SL);
  return 0;
}


/// getColumnNumber - Return the column # for the specified include position.
/// this is significantly cheaper to compute than the line number.  This returns
/// zero if the column number isn't known.
unsigned SourceManager::getColumnNumber(SourceLocation Loc) const {
  unsigned FileID = Loc.getFileID();
  if (FileID == 0) return 0;
  
  // If this is a macro, we need to get the instantiation location.
  const SrcMgr::FileIDInfo *FIDInfo = getFIDInfo(FileID);
  if (FIDInfo->IDType == SrcMgr::FileIDInfo::MacroExpansion) {
    Loc = FIDInfo->IncludeLoc;
    FileID = Loc.getFileID();
  }
  
  unsigned FilePos = getFilePos(Loc);
  const SourceBuffer *Buffer = getBuffer(FileID);
  const char *Buf = Buffer->getBufferStart();

  unsigned LineStart = FilePos;
  while (LineStart && Buf[LineStart-1] != '\n' && Buf[LineStart-1] != '\r')
    --LineStart;
  return FilePos-LineStart+1;
}

/// getSourceName - This method returns the name of the file or buffer that
/// the SourceLocation specifies.  This can be modified with #line directives,
/// etc.
std::string SourceManager::getSourceName(SourceLocation Loc) {
  unsigned FileID = Loc.getFileID();
  if (FileID == 0) return "";
  
  // If this is a macro, we need to get the instantiation location.
  const SrcMgr::FileIDInfo *FIDInfo = getFIDInfo(FileID);
  if (FIDInfo->IDType == SrcMgr::FileIDInfo::MacroExpansion) {
    Loc = FIDInfo->IncludeLoc;
    FIDInfo = getFIDInfo(Loc.getFileID());
  }
  
  return getFileInfo(FIDInfo)->Buffer->getBufferIdentifier();
}


/// getLineNumber - Given a SourceLocation, return the physical line number
/// for the position indicated.  This requires building and caching a table of
/// line offsets for the SourceBuffer, so this is not cheap: use only when
/// about to emit a diagnostic.
unsigned SourceManager::getLineNumber(SourceLocation Loc) {
  unsigned FileID = Loc.getFileID();
  // If this is a macro, we need to get the instantiation location.
  const SrcMgr::FileIDInfo *FIDInfo = getFIDInfo(FileID);
  if (FIDInfo->IDType == SrcMgr::FileIDInfo::MacroExpansion) {
    Loc = FIDInfo->IncludeLoc;
    FileID = Loc.getFileID();
    FIDInfo = getFIDInfo(FileID);
  }

  FileInfo *FileInfo = getFileInfo(FileID);
  
  // If this is the first use of line information for this buffer, compute the
  /// SourceLineCache for it on demand. 
  if (FileInfo->SourceLineCache == 0) {
    const SourceBuffer *Buffer = FileInfo->Buffer;
    
    // Find the file offsets of all of the *physical* source lines.  This does
    // not look at trigraphs, escaped newlines, or anything else tricky.
    std::vector<unsigned> LineOffsets;
    
    // Line #1 starts at char 0.
    LineOffsets.push_back(0);
    
    const unsigned char *Buf = (const unsigned char *)Buffer->getBufferStart();
    const unsigned char *End = (const unsigned char *)Buffer->getBufferEnd();
    unsigned Offs = 0;
    while (1) {
      // Skip over the contents of the line.
      // TODO: Vectorize this?  This is very performance sensitive for programs
      // with lots of diagnostics.
      const unsigned char *NextBuf = (const unsigned char *)Buf;
      while (*NextBuf != '\n' && *NextBuf != '\r' && *NextBuf != '\0')
        ++NextBuf;
      Offs += NextBuf-Buf;
      Buf = NextBuf;
      
      if (Buf[0] == '\n' || Buf[0] == '\r') {
        // If this is \n\r or \r\n, skip both characters.
        if ((Buf[1] == '\n' || Buf[1] == '\r') && Buf[0] != Buf[1])
          ++Offs, ++Buf;
        ++Offs, ++Buf;
        LineOffsets.push_back(Offs);
      } else {
        // Otherwise, this is a null.  If end of file, exit.
        if (Buf == End) break;
        // Otherwise, skip the null.
        ++Offs, ++Buf;
      }
    }
    LineOffsets.push_back(Offs);
    
    // Copy the offsets into the FileInfo structure.
    FileInfo->NumLines = LineOffsets.size();
    FileInfo->SourceLineCache = new unsigned[LineOffsets.size()];
    std::copy(LineOffsets.begin(), LineOffsets.end(),
              FileInfo->SourceLineCache);
  }

  // Okay, we know we have a line number table.  Do a binary search to find the
  // line number that this character position lands on.
  unsigned NumLines = FileInfo->NumLines;
  unsigned *SourceLineCache = FileInfo->SourceLineCache;
    
  // TODO: If this is performance sensitive, we could try doing simple radix
  // type approaches to make good (tight?) initial guesses based on the
  // assumption that all lines are the same average size.
  unsigned *Pos = std::lower_bound(SourceLineCache, SourceLineCache+NumLines,
                                   getFilePos(Loc)+1);
  return Pos-SourceLineCache;
}

/// PrintStats - Print statistics to stderr.
///
void SourceManager::PrintStats() const {
  std::cerr << "\n*** Source Manager Stats:\n";
  std::cerr << FileInfos.size() << " files mapped, " << MemBufferInfos.size()
            << " mem buffers mapped, " << FileIDs.size() 
            << " file ID's allocated.\n";
  unsigned NumBuffers = 0, NumMacros = 0;
  for (unsigned i = 0, e = FileIDs.size(); i != e; ++i) {
    if (FileIDs[i].IDType == FileIDInfo::NormalBuffer)
      ++NumBuffers;
    else if (FileIDs[i].IDType == FileIDInfo::MacroExpansion)
      ++NumMacros;
    else
      assert(0 && "Unknown FileID!");
  }
  std::cerr << "  " << NumBuffers << " normal buffer FileID's, "
            << NumMacros << " macro expansion FileID's.\n";
    
  
  
  unsigned NumLineNumsComputed = 0;
  unsigned NumFileBytesMapped = 0;
  for (std::map<const FileEntry *, FileInfo>::const_iterator I = 
       FileInfos.begin(), E = FileInfos.end(); I != E; ++I) {
    NumLineNumsComputed += I->second.SourceLineCache != 0;
    NumFileBytesMapped  += I->second.Buffer->getBufferSize();
  }
  std::cerr << NumFileBytesMapped << " bytes of files mapped, "
            << NumLineNumsComputed << " files with line #'s computed.\n";
}
