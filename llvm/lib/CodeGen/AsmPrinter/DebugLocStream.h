//===--- lib/CodeGen/DebugLocStream.h - DWARF debug_loc stream --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DEBUGLOCSTREAM_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DEBUGLOCSTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "ByteStreamer.h"

namespace llvm {
class DwarfCompileUnit;
class MCSymbol;

/// \brief Byte stream of .debug_loc entries.
///
/// Stores a unified stream of .debug_loc entries.  There's \a List for each
/// variable/inlined-at pair, and an \a Entry for each \a DebugLocEntry.
///
/// FIXME: Why do we have comments even when it's an object stream?
/// FIXME: Do we need all these temp symbols?
/// FIXME: Why not output directly to the output stream?
class DebugLocStream {
public:
  struct List {
    DwarfCompileUnit *CU;
    MCSymbol *Label;
    size_t EntryOffset;
    List(DwarfCompileUnit *CU, MCSymbol *Label, size_t EntryOffset)
        : CU(CU), Label(Label), EntryOffset(EntryOffset) {}
  };
  struct Entry {
    const MCSymbol *BeginSym;
    const MCSymbol *EndSym;
    size_t ByteOffset;
    size_t CommentOffset;
    Entry(const MCSymbol *BeginSym, const MCSymbol *EndSym, size_t ByteOffset,
          size_t CommentOffset)
        : BeginSym(BeginSym), EndSym(EndSym), ByteOffset(ByteOffset),
          CommentOffset(CommentOffset) {}
  };

private:
  SmallVector<List, 4> Lists;
  SmallVector<Entry, 32> Entries;
  SmallString<256> DWARFBytes;
  SmallVector<std::string, 32> Comments;

public:
  size_t getNumLists() const { return Lists.size(); }
  const List &getList(size_t LI) const { return Lists[LI]; }
  ArrayRef<List> getLists() const { return Lists; }

  /// \brief Start a new .debug_loc entry list.
  ///
  /// Start a new .debug_loc entry list.  Return the new list's index so it can
  /// be retrieved later via \a getList().
  ///
  /// Until the next call, \a startEntry() will add entries to this list.
  size_t startList(DwarfCompileUnit *CU, MCSymbol *Label) {
    size_t LI = Lists.size();
    Lists.emplace_back(CU, Label, Entries.size());
    return LI;
  }

  /// \brief Start a new .debug_loc entry.
  ///
  /// Until the next call, bytes added to the stream will be added to this
  /// entry.
  void startEntry(const MCSymbol *BeginSym, const MCSymbol *EndSym) {
    Entries.emplace_back(BeginSym, EndSym, DWARFBytes.size(), Comments.size());
  }

  BufferByteStreamer getStreamer() {
    return BufferByteStreamer(DWARFBytes, Comments);
  }

  ArrayRef<Entry> getEntries(const List &L) const {
    size_t LI = getIndex(L);
    return makeArrayRef(Entries)
        .slice(Lists[LI].EntryOffset, getNumEntries(LI));
  }

  ArrayRef<char> getBytes(const Entry &E) const {
    size_t EI = getIndex(E);
    return makeArrayRef(DWARFBytes.begin(), DWARFBytes.end())
        .slice(Entries[EI].ByteOffset, getNumBytes(EI));
  }
  ArrayRef<std::string> getComments(const Entry &E) const {
    size_t EI = getIndex(E);
    return makeArrayRef(Comments)
        .slice(Entries[EI].CommentOffset, getNumComments(EI));
  }

private:
  size_t getIndex(const List &L) const {
    assert(&Lists.front() <= &L && &L <= &Lists.back() &&
           "Expected valid list");
    return &L - &Lists.front();
  }
  size_t getIndex(const Entry &E) const {
    assert(&Entries.front() <= &E && &E <= &Entries.back() &&
           "Expected valid entry");
    return &E - &Entries.front();
  }
  size_t getNumEntries(size_t LI) const {
    if (LI + 1 == Lists.size())
      return Entries.size() - Lists[LI].EntryOffset;
    return Lists[LI + 1].EntryOffset - Lists[LI].EntryOffset;
  }
  size_t getNumBytes(size_t EI) const {
    if (EI + 1 == Entries.size())
      return DWARFBytes.size() - Entries[EI].ByteOffset;
    return Entries[EI + 1].ByteOffset - Entries[EI].ByteOffset;
  }
  size_t getNumComments(size_t EI) const {
    if (EI + 1 == Entries.size())
      return Comments.size() - Entries[EI].CommentOffset;
    return Entries[EI + 1].CommentOffset - Entries[EI].CommentOffset;
  }
};
}
#endif
