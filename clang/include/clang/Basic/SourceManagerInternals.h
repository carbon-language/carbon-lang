//===--- SourceManagerInternals.h - SourceManager Internals -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the implementation details of the SourceManager
//  class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SOURCEMANAGER_INTERNALS_H
#define LLVM_CLANG_SOURCEMANAGER_INTERNALS_H

#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringMap.h"
#include <map>

namespace clang {

//===----------------------------------------------------------------------===//
// Line Table Implementation
//===----------------------------------------------------------------------===//

struct LineEntry {
  /// FileOffset - The offset in this file that the line entry occurs at.
  unsigned FileOffset;
  
  /// LineNo - The presumed line number of this line entry: #line 4.
  unsigned LineNo;
  
  /// FilenameID - The ID of the filename identified by this line entry:
  /// #line 4 "foo.c".  This is -1 if not specified.
  int FilenameID;
  
  /// Flags - Set the 0 if no flags, 1 if a system header, 
  SrcMgr::CharacteristicKind FileKind;
  
  /// IncludeOffset - This is the offset of the virtual include stack location,
  /// which is manipulated by GNU linemarker directives.  If this is 0 then
  /// there is no virtual #includer.
  unsigned IncludeOffset;
  
  static LineEntry get(unsigned Offs, unsigned Line, int Filename,
                       SrcMgr::CharacteristicKind FileKind,
                       unsigned IncludeOffset) {
    LineEntry E;
    E.FileOffset = Offs;
    E.LineNo = Line;
    E.FilenameID = Filename;
    E.FileKind = FileKind;
    E.IncludeOffset = IncludeOffset;
    return E;
  }
};

// needed for FindNearestLineEntry (upper_bound of LineEntry)
inline bool operator<(const LineEntry &lhs, const LineEntry &rhs) {
  // FIXME: should check the other field?
  return lhs.FileOffset < rhs.FileOffset;
}

inline bool operator<(const LineEntry &E, unsigned Offset) {
  return E.FileOffset < Offset;
}

inline bool operator<(unsigned Offset, const LineEntry &E) {
  return Offset < E.FileOffset;
}
  
/// LineTableInfo - This class is used to hold and unique data used to
/// represent #line information.
class LineTableInfo {
  /// FilenameIDs - This map is used to assign unique IDs to filenames in
  /// #line directives.  This allows us to unique the filenames that
  /// frequently reoccur and reference them with indices.  FilenameIDs holds
  /// the mapping from string -> ID, and FilenamesByID holds the mapping of ID
  /// to string.
  llvm::StringMap<unsigned, llvm::BumpPtrAllocator> FilenameIDs;
  std::vector<llvm::StringMapEntry<unsigned>*> FilenamesByID;
  
  /// LineEntries - This is a map from FileIDs to a list of line entries (sorted
  /// by the offset they occur in the file.
  std::map<unsigned, std::vector<LineEntry> > LineEntries;
public:
  LineTableInfo() {
  }
  
  void clear() {
    FilenameIDs.clear();
    FilenamesByID.clear();
    LineEntries.clear();
  }
  
  ~LineTableInfo() {}
  
  unsigned getLineTableFilenameID(const char *Ptr, unsigned Len);
  const char *getFilename(unsigned ID) const {
    assert(ID < FilenamesByID.size() && "Invalid FilenameID");
    return FilenamesByID[ID]->getKeyData();
  }
  unsigned getNumFilenames() const { return FilenamesByID.size(); }

  void AddLineNote(unsigned FID, unsigned Offset,
                   unsigned LineNo, int FilenameID);
  void AddLineNote(unsigned FID, unsigned Offset,
                   unsigned LineNo, int FilenameID,
                   unsigned EntryExit, SrcMgr::CharacteristicKind FileKind);

  
  /// FindNearestLineEntry - Find the line entry nearest to FID that is before
  /// it.  If there is no line entry before Offset in FID, return null.
  const LineEntry *FindNearestLineEntry(unsigned FID, unsigned Offset);

  // Low-level access
  typedef std::map<unsigned, std::vector<LineEntry> >::iterator iterator;
  iterator begin() { return LineEntries.begin(); }
  iterator end() { return LineEntries.end(); }

  /// \brief Add a new line entry that has already been encoded into
  /// the internal representation of the line table.
  void AddEntry(unsigned FID, const std::vector<LineEntry> &Entries);
};

} // end namespace clang

#endif
