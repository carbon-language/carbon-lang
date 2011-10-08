//===- Archive.cpp - ar File Format implementation --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ArchiveObjectFile class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Archive.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace object;

namespace {
const StringRef Magic = "!<arch>\n";

struct ArchiveMemberHeader {
  char Name[16];
  char LastModified[12];
  char UID[6];
  char GID[6];
  char AccessMode[8];
  char Size[10]; //< Size of data, not including header or padding.
  char Terminator[2];

  ///! Get the name without looking up long names.
  StringRef getName() const {
    char EndCond = Name[0] == '/' ? ' ' : '/';
    StringRef::size_type end = StringRef(Name, sizeof(Name)).find(EndCond);
    if (end == StringRef::npos)
      end = sizeof(Name);
    assert(end <= sizeof(Name) && end > 0);
    // Don't include the EndCond if there is one.
    return StringRef(Name, end);
  }

  uint64_t getSize() const {
    APInt ret;
    StringRef(Size, sizeof(Size)).getAsInteger(10, ret);
    return ret.getZExtValue();
  }
};

const ArchiveMemberHeader *ToHeader(const char *base) {
  return reinterpret_cast<const ArchiveMemberHeader *>(base);
}
}

Archive::Child Archive::Child::getNext() const {
  size_t SpaceToSkip = sizeof(ArchiveMemberHeader) +
    ToHeader(Data.data())->getSize();
  // If it's odd, add 1 to make it even.
  if (SpaceToSkip & 1)
    ++SpaceToSkip;

  const char *NextLoc = Data.data() + SpaceToSkip;

  // Check to see if this is past the end of the archive.
  if (NextLoc >= Parent->Data->getBufferEnd())
    return Child(Parent, StringRef(0, 0));

  size_t NextSize = sizeof(ArchiveMemberHeader) +
    ToHeader(NextLoc)->getSize();

  return Child(Parent, StringRef(NextLoc, NextSize));
}

error_code Archive::Child::getName(StringRef &Result) const {
  StringRef name = ToHeader(Data.data())->getName();
  // Check if it's a special name.
  if (name[0] == '/') {
    if (name.size() == 1) { // Linker member.
      Result = name;
      return object_error::success;
    }
    if (name.size() == 2 && name[1] == '/') { // String table.
      Result = name;
      return object_error::success;
    }
    // It's a long name.
    // Get the offset.
    APInt offset;
    name.substr(1).getAsInteger(10, offset);
    const char *addr = Parent->StringTable->Data.begin()
                       + sizeof(ArchiveMemberHeader)
                       + offset.getZExtValue();
    // Verify it.
    if (Parent->StringTable == Parent->end_children()
        || addr < (Parent->StringTable->Data.begin()
                   + sizeof(ArchiveMemberHeader))
        || addr > (Parent->StringTable->Data.begin()
                   + sizeof(ArchiveMemberHeader)
                   + Parent->StringTable->getSize()))
      return object_error::parse_failed;
    Result = addr;
    return object_error::success;
  }
  // It's a simple name.
  if (name[name.size() - 1] == '/')
    Result = name.substr(0, name.size() - 1);
  else
    Result = name;
  return object_error::success;
}

uint64_t Archive::Child::getSize() const {
  return ToHeader(Data.data())->getSize();
}

MemoryBuffer *Archive::Child::getBuffer() const {
  StringRef name;
  if (getName(name)) return NULL;
  return MemoryBuffer::getMemBuffer(Data.substr(sizeof(ArchiveMemberHeader),
                                                getSize()),
                                    name,
                                    false);
}

error_code Archive::Child::getAsBinary(OwningPtr<Binary> &Result) const {
  OwningPtr<Binary> ret;
  if (error_code ec =
    createBinary(getBuffer(), ret))
    return ec;
  Result.swap(ret);
  return object_error::success;
}

Archive::Archive(MemoryBuffer *source, error_code &ec)
  : Binary(Binary::isArchive, source)
  , StringTable(Child(this, StringRef(0, 0))) {
  // Check for sufficient magic.
  if (!source || source->getBufferSize()
                 < (8 + sizeof(ArchiveMemberHeader) + 2) // Smallest archive.
              || StringRef(source->getBufferStart(), 8) != Magic) {
    ec = object_error::invalid_file_type;
    return;
  }

  // Get the string table. It's the 3rd member.
  child_iterator StrTable = begin_children();
  child_iterator e = end_children();
  for (int i = 0; StrTable != e && i < 2; ++StrTable, ++i) {}

  // Check to see if there were 3 members, or the 3rd member wasn't named "//".
  StringRef name;
  if (StrTable != e && !StrTable->getName(name) && name == "//")
    StringTable = StrTable;

  ec = object_error::success;
}

Archive::child_iterator Archive::begin_children() const {
  const char *Loc = Data->getBufferStart() + Magic.size();
  size_t Size = sizeof(ArchiveMemberHeader) +
    ToHeader(Loc)->getSize();
  return Child(this, StringRef(Loc, Size));
}

Archive::child_iterator Archive::end_children() const {
  return Child(this, StringRef(0, 0));
}

namespace llvm {

} // end namespace llvm
