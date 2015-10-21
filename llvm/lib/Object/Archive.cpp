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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace object;
using namespace llvm::support::endian;

static const char *const Magic = "!<arch>\n";
static const char *const ThinMagic = "!<thin>\n";

void Archive::anchor() { }

StringRef ArchiveMemberHeader::getName() const {
  char EndCond;
  if (Name[0] == '/' || Name[0] == '#')
    EndCond = ' ';
  else
    EndCond = '/';
  llvm::StringRef::size_type end =
      llvm::StringRef(Name, sizeof(Name)).find(EndCond);
  if (end == llvm::StringRef::npos)
    end = sizeof(Name);
  assert(end <= sizeof(Name) && end > 0);
  // Don't include the EndCond if there is one.
  return llvm::StringRef(Name, end);
}

ErrorOr<uint32_t> ArchiveMemberHeader::getSize() const {
  uint32_t Ret;
  if (llvm::StringRef(Size, sizeof(Size)).rtrim(" ").getAsInteger(10, Ret))
    return object_error::parse_failed; // Size is not a decimal number.
  return Ret;
}

sys::fs::perms ArchiveMemberHeader::getAccessMode() const {
  unsigned Ret;
  if (StringRef(AccessMode, sizeof(AccessMode)).rtrim(" ").getAsInteger(8, Ret))
    llvm_unreachable("Access mode is not an octal number.");
  return static_cast<sys::fs::perms>(Ret);
}

sys::TimeValue ArchiveMemberHeader::getLastModified() const {
  unsigned Seconds;
  if (StringRef(LastModified, sizeof(LastModified)).rtrim(" ")
          .getAsInteger(10, Seconds))
    llvm_unreachable("Last modified time not a decimal number.");

  sys::TimeValue Ret;
  Ret.fromEpochTime(Seconds);
  return Ret;
}

unsigned ArchiveMemberHeader::getUID() const {
  unsigned Ret;
  if (StringRef(UID, sizeof(UID)).rtrim(" ").getAsInteger(10, Ret))
    llvm_unreachable("UID time not a decimal number.");
  return Ret;
}

unsigned ArchiveMemberHeader::getGID() const {
  unsigned Ret;
  if (StringRef(GID, sizeof(GID)).rtrim(" ").getAsInteger(10, Ret))
    llvm_unreachable("GID time not a decimal number.");
  return Ret;
}

Archive::Child::Child(const Archive *Parent, const char *Start,
                      std::error_code *EC)
    : Parent(Parent) {
  if (!Start)
    return;

  uint64_t Size = sizeof(ArchiveMemberHeader);
  Data = StringRef(Start, Size);
  if (!isThinMember()) {
    ErrorOr<uint64_t> MemberSize = getRawSize();
    if (MemberSize.getError()) {
      assert (EC && "Error must be caught");
      *EC = MemberSize.getError();
      return;
    }
    Size += MemberSize.get();
    Data = StringRef(Start, Size);
  }

  // Setup StartOfFile and PaddingBytes.
  StartOfFile = sizeof(ArchiveMemberHeader);
  // Don't include attached name.
  StringRef Name = getRawName();
  if (Name.startswith("#1/")) {
    uint64_t NameSize;
    if (Name.substr(3).rtrim(" ").getAsInteger(10, NameSize)) {
      if (EC)
        *EC = object_error::parse_failed; // Long name offset is not an integer.
      return;
    }
    StartOfFile += NameSize;
  }
}

ErrorOr<std::unique_ptr<Archive::Child>> Archive::Child::create(
  const Archive *Parent, const char *Start) {
  std::error_code EC;
  std::unique_ptr<Archive::Child> Ret(new Archive::Child(Parent, Start, &EC));
  if (EC)
    return EC;
  return std::move(Ret);
}

ErrorOr<uint64_t> Archive::Child::getSize() const {
  if (Parent->IsThin) {
    ErrorOr<uint32_t> Size = getHeader()->getSize();
    if (std::error_code EC = Size.getError())
      return EC;
    return Size.get();
  }
  return Data.size() - StartOfFile;
}

ErrorOr<uint64_t> Archive::Child::getRawSize() const {
  ErrorOr<uint32_t> Size = getHeader()->getSize();
  if (std::error_code EC = Size.getError())
    return EC;
  return Size.get();
}

bool Archive::Child::isThinMember() const {
  StringRef Name = getHeader()->getName();
  return Parent->IsThin && Name != "/" && Name != "//";
}

ErrorOr<StringRef> Archive::Child::getBuffer() const {
  if (!isThinMember()) {
    ErrorOr<uint32_t> Size = getSize();
    if (std::error_code EC = Size.getError())
      return EC;
    return StringRef(Data.data() + StartOfFile, Size.get());
  }
  ErrorOr<StringRef> Name = getName();
  if (std::error_code EC = Name.getError())
    return EC;
  SmallString<128> FullName = sys::path::parent_path(
      Parent->getMemoryBufferRef().getBufferIdentifier());
  sys::path::append(FullName, *Name);
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(FullName);
  if (std::error_code EC = Buf.getError())
    return EC;
  Parent->ThinBuffers.push_back(std::move(*Buf));
  return Parent->ThinBuffers.back()->getBuffer();
}

ErrorOr<Archive::Child> Archive::Child::getNext() const {
  size_t SpaceToSkip = Data.size();
  // If it's odd, add 1 to make it even.
  size_t Pad = 0;
  if (SpaceToSkip & 1)
    Pad++;

  const char *NextLoc = Data.data() + SpaceToSkip + Pad;

  // Check to see if this is at the end of the archive.
  if (NextLoc == Parent->Data.getBufferEnd() ||
      NextLoc == Parent->Data.getBufferEnd() - Pad )
    return Child(Parent, nullptr, nullptr);

  // Check to see if this is past the end of the archive.
  if (NextLoc > Parent->Data.getBufferEnd())
    return object_error::parse_failed;

  auto ChildOrErr = Child::create(Parent, NextLoc);
  if (std::error_code EC = ChildOrErr.getError())
    return EC;
  return std::move(*ChildOrErr.get());
}

uint64_t Archive::Child::getChildOffset() const {
  const char *a = Parent->Data.getBuffer().data();
  const char *c = Data.data();
  uint64_t offset = c - a;
  return offset;
}

ErrorOr<StringRef> Archive::Child::getName() const {
  StringRef name = getRawName();
  // Check if it's a special name.
  if (name[0] == '/') {
    if (name.size() == 1) // Linker member.
      return name;
    if (name.size() == 2 && name[1] == '/') // String table.
      return name;
    // It's a long name.
    // Get the offset.
    std::size_t offset;
    if (name.substr(1).rtrim(" ").getAsInteger(10, offset))
      return object_error::parse_failed; // Long name offset is not an integer.
    // Check for bad stringtable iterator.
    if (std::error_code EC = Parent->StringTable->getError())
      return EC;
    const char *addr = (*Parent->StringTable)->Data.begin()
                       + sizeof(ArchiveMemberHeader)
                       + offset;
    // Verify it.
    auto Size = (*Parent->StringTable)->getSize();
    if (std::error_code EC = Size.getError())
      return EC;
    if (Parent->StringTable == Parent->child_end()
        || addr < ((*Parent->StringTable)->Data.begin()
                   + sizeof(ArchiveMemberHeader))
        || addr > ((*Parent->StringTable)->Data.begin()
                   + sizeof(ArchiveMemberHeader)
                   + Size.get()))
      return object_error::parse_failed;

    // GNU long file names end with a "/\n".
    if (Parent->kind() == K_GNU || Parent->kind() == K_MIPS64) {
      StringRef::size_type End = StringRef(addr).find('\n');
      return StringRef(addr, End - 1);
    }
    return StringRef(addr);
  } else if (name.startswith("#1/")) {
    uint64_t name_size;
    if (name.substr(3).rtrim(" ").getAsInteger(10, name_size))
      return object_error::parse_failed; // Long name offset is not an integer.
    return Data.substr(sizeof(ArchiveMemberHeader), name_size)
        .rtrim(StringRef("\0", 1));
  }
  // It's a simple name.
  if (name[name.size() - 1] == '/')
    return name.substr(0, name.size() - 1);
  return name;
}

ErrorOr<MemoryBufferRef> Archive::Child::getMemoryBufferRef() const {
  ErrorOr<StringRef> NameOrErr = getName();
  if (std::error_code EC = NameOrErr.getError())
    return EC;
  StringRef Name = NameOrErr.get();
  ErrorOr<StringRef> Buf = getBuffer();
  if (std::error_code EC = Buf.getError())
    return EC;
  return MemoryBufferRef(*Buf, Name);
}

ErrorOr<std::unique_ptr<Binary>>
Archive::Child::getAsBinary(LLVMContext *Context) const {
  ErrorOr<MemoryBufferRef> BuffOrErr = getMemoryBufferRef();
  if (std::error_code EC = BuffOrErr.getError())
    return EC;

  return createBinary(BuffOrErr.get(), Context);
}

ErrorOr<std::unique_ptr<Archive>> Archive::create(MemoryBufferRef Source) {
  std::error_code EC;
  std::unique_ptr<Archive> Ret(new Archive(Source, EC));
  if (EC)
    return EC;
  return std::move(Ret);
}

Archive::Archive(MemoryBufferRef Source, std::error_code &ec)
    : Binary(Binary::ID_Archive, Source), SymbolTable(child_end()),
      StringTable(child_end()), FirstRegular(child_end()) {
  StringRef Buffer = Data.getBuffer();
  // Check for sufficient magic.
  if (Buffer.startswith(ThinMagic)) {
    IsThin = true;
  } else if (Buffer.startswith(Magic)) {
    IsThin = false;
  } else {
    ec = object_error::invalid_file_type;
    return;
  }

  // Get the special members.
  child_iterator i = child_begin(false);
  child_iterator e = child_end();

  if (!*i || i == e) {
    ec = i->getError();
    return;
  }

  StringRef Name = (*i)->getRawName();

  // Below is the pattern that is used to figure out the archive format
  // GNU archive format
  //  First member : / (may exist, if it exists, points to the symbol table )
  //  Second member : // (may exist, if it exists, points to the string table)
  //  Note : The string table is used if the filename exceeds 15 characters
  // BSD archive format
  //  First member : __.SYMDEF or "__.SYMDEF SORTED" (the symbol table)
  //  There is no string table, if the filename exceeds 15 characters or has a
  //  embedded space, the filename has #1/<size>, The size represents the size
  //  of the filename that needs to be read after the archive header
  // COFF archive format
  //  First member : /
  //  Second member : / (provides a directory of symbols)
  //  Third member : // (may exist, if it exists, contains the string table)
  //  Note: Microsoft PE/COFF Spec 8.3 says that the third member is present
  //  even if the string table is empty. However, lib.exe does not in fact
  //  seem to create the third member if there's no member whose filename
  //  exceeds 15 characters. So the third member is optional.

  if (Name == "__.SYMDEF") {
    Format = K_BSD;
    SymbolTable = i;
    ++i;
    if (!*i) {
      ec = i->getError();
      return;
    }

    FirstRegular = i;
    ec = std::error_code();
    return;
  }

  if (Name.startswith("#1/")) {
    Format = K_BSD;
    // We know this is BSD, so getName will work since there is no string table.
    ErrorOr<StringRef> NameOrErr = (*i)->getName();
    ec = NameOrErr.getError();
    if (ec)
      return;
    Name = NameOrErr.get();
    if (Name == "__.SYMDEF SORTED" || Name == "__.SYMDEF") {
      SymbolTable = i;
      ++i;
      if (!*i) {
        ec = i->getError();
        return;
      }
    }
    FirstRegular = i;
    return;
  }

  // MIPS 64-bit ELF archives use a special format of a symbol table.
  // This format is marked by `ar_name` field equals to "/SYM64/".
  // For detailed description see page 96 in the following document:
  // http://techpubs.sgi.com/library/manuals/4000/007-4658-001/pdf/007-4658-001.pdf

  bool has64SymTable = false;
  if (Name == "/" || Name == "/SYM64/") {
    SymbolTable = i;
    if (Name == "/SYM64/")
      has64SymTable = true;

    ++i;
    if (!*i || i == e) {
      ec = i->getError();
      return;
    }
    Name = (*i)->getRawName();
  }

  if (Name == "//") {
    Format = has64SymTable ? K_MIPS64 : K_GNU;
    StringTable = i;
    ++i;
    if (!*i) {
      ec = i->getError();
      return;
    }
    FirstRegular = i;
    ec = std::error_code();
    return;
  }

  if (Name[0] != '/') {
    Format = has64SymTable ? K_MIPS64 : K_GNU;
    FirstRegular = i;
    ec = std::error_code();
    return;
  }

  if (Name != "/") {
    ec = object_error::parse_failed;
    return;
  }

  Format = K_COFF;
  SymbolTable = i;

  ++i;
  if (!*i) {
    ec = i->getError();
    return;
  }
  if (i == e) {
    FirstRegular = i;
    ec = std::error_code();
    return;
  }

  Name = (*i)->getRawName();

  if (Name == "//") {
    StringTable = i;
    ++i;
    if (!*i) {
      ec = i->getError();
      return;
    }
  }

  FirstRegular = i;
  ec = std::error_code();
}

Archive::child_iterator Archive::child_begin(bool SkipInternal) const {
  if (Data.getBufferSize() == 8) // empty archive.
    return child_end();

  if (SkipInternal)
    return FirstRegular;

  const char *Loc = Data.getBufferStart() + strlen(Magic);
  auto ChildOrErr = Child::create(this, Loc);
  if (std::error_code EC = ChildOrErr.getError())
    return child_iterator(EC);
  Child c = *(ChildOrErr.get());
  return child_iterator(c);
}

Archive::child_iterator Archive::child_end() const {
  // This with a second argument of nullptr can't return an Error.
  auto ChildOrErr = Child::create(this, nullptr);
  if (ChildOrErr.getError())
    llvm_unreachable("Can't create Archive::child_end().");
  Child c = *(ChildOrErr.get());
  return child_iterator(c);
}

StringRef Archive::Symbol::getName() const {
  return Parent->getSymbolTable().begin() + StringIndex;
}

ErrorOr<Archive::child_iterator> Archive::Symbol::getMember() const {
  const char *Buf = Parent->getSymbolTable().begin();
  const char *Offsets = Buf;
  if (Parent->kind() == K_MIPS64)
    Offsets += sizeof(uint64_t);
  else
    Offsets += sizeof(uint32_t);
  uint32_t Offset = 0;
  if (Parent->kind() == K_GNU) {
    Offset = read32be(Offsets + SymbolIndex * 4);
  } else if (Parent->kind() == K_MIPS64) {
    Offset = read64be(Offsets + SymbolIndex * 8);
  } else if (Parent->kind() == K_BSD) {
    // The SymbolIndex is an index into the ranlib structs that start at
    // Offsets (the first uint32_t is the number of bytes of the ranlib
    // structs).  The ranlib structs are a pair of uint32_t's the first
    // being a string table offset and the second being the offset into
    // the archive of the member that defines the symbol.  Which is what
    // is needed here.
    Offset = read32le(Offsets + SymbolIndex * 8 + 4);
  } else {
    // Skip offsets.
    uint32_t MemberCount = read32le(Buf);
    Buf += MemberCount * 4 + 4;

    uint32_t SymbolCount = read32le(Buf);
    if (SymbolIndex >= SymbolCount)
      return object_error::parse_failed;

    // Skip SymbolCount to get to the indices table.
    const char *Indices = Buf + 4;

    // Get the index of the offset in the file member offset table for this
    // symbol.
    uint16_t OffsetIndex = read16le(Indices + SymbolIndex * 2);
    // Subtract 1 since OffsetIndex is 1 based.
    --OffsetIndex;

    if (OffsetIndex >= MemberCount)
      return object_error::parse_failed;

    Offset = read32le(Offsets + OffsetIndex * 4);
  }

  const char *Loc = Parent->getData().begin() + Offset;
  auto ChildOrErr = Child::create(Parent, Loc);
  if (std::error_code EC = ChildOrErr.getError())
    return EC;
  child_iterator Iter(std::move(*ChildOrErr.get()));
  return Iter;
}

Archive::Symbol Archive::Symbol::getNext() const {
  Symbol t(*this);
  if (Parent->kind() == K_BSD) {
    // t.StringIndex is an offset from the start of the __.SYMDEF or
    // "__.SYMDEF SORTED" member into the string table for the ranlib
    // struct indexed by t.SymbolIndex .  To change t.StringIndex to the
    // offset in the string table for t.SymbolIndex+1 we subtract the
    // its offset from the start of the string table for t.SymbolIndex
    // and add the offset of the string table for t.SymbolIndex+1.

    // The __.SYMDEF or "__.SYMDEF SORTED" member starts with a uint32_t
    // which is the number of bytes of ranlib structs that follow.  The ranlib
    // structs are a pair of uint32_t's the first being a string table offset
    // and the second being the offset into the archive of the member that
    // define the symbol. After that the next uint32_t is the byte count of
    // the string table followed by the string table.
    const char *Buf = Parent->getSymbolTable().begin();
    uint32_t RanlibCount = 0;
    RanlibCount = read32le(Buf) / 8;
    // If t.SymbolIndex + 1 will be past the count of symbols (the RanlibCount)
    // don't change the t.StringIndex as we don't want to reference a ranlib
    // past RanlibCount.
    if (t.SymbolIndex + 1 < RanlibCount) {
      const char *Ranlibs = Buf + 4;
      uint32_t CurRanStrx = 0;
      uint32_t NextRanStrx = 0;
      CurRanStrx = read32le(Ranlibs + t.SymbolIndex * 8);
      NextRanStrx = read32le(Ranlibs + (t.SymbolIndex + 1) * 8);
      t.StringIndex -= CurRanStrx;
      t.StringIndex += NextRanStrx;
    }
  } else {
    // Go to one past next null.
    t.StringIndex = Parent->getSymbolTable().find('\0', t.StringIndex) + 1;
  }
  ++t.SymbolIndex;
  return t;
}

Archive::symbol_iterator Archive::symbol_begin() const {
  if (!hasSymbolTable())
    return symbol_iterator(Symbol(this, 0, 0));

  const char *buf = getSymbolTable().begin();
  if (kind() == K_GNU) {
    uint32_t symbol_count = 0;
    symbol_count = read32be(buf);
    buf += sizeof(uint32_t) + (symbol_count * (sizeof(uint32_t)));
  } else if (kind() == K_MIPS64) {
    uint64_t symbol_count = read64be(buf);
    buf += sizeof(uint64_t) + (symbol_count * (sizeof(uint64_t)));
  } else if (kind() == K_BSD) {
    // The __.SYMDEF or "__.SYMDEF SORTED" member starts with a uint32_t
    // which is the number of bytes of ranlib structs that follow.  The ranlib
    // structs are a pair of uint32_t's the first being a string table offset
    // and the second being the offset into the archive of the member that
    // define the symbol. After that the next uint32_t is the byte count of
    // the string table followed by the string table.
    uint32_t ranlib_count = 0;
    ranlib_count = read32le(buf) / 8;
    const char *ranlibs = buf + 4;
    uint32_t ran_strx = 0;
    ran_strx = read32le(ranlibs);
    buf += sizeof(uint32_t) + (ranlib_count * (2 * (sizeof(uint32_t))));
    // Skip the byte count of the string table.
    buf += sizeof(uint32_t);
    buf += ran_strx;
  } else {
    uint32_t member_count = 0;
    uint32_t symbol_count = 0;
    member_count = read32le(buf);
    buf += 4 + (member_count * 4); // Skip offsets.
    symbol_count = read32le(buf);
    buf += 4 + (symbol_count * 2); // Skip indices.
  }
  uint32_t string_start_offset = buf - getSymbolTable().begin();
  return symbol_iterator(Symbol(this, 0, string_start_offset));
}

Archive::symbol_iterator Archive::symbol_end() const {
  return symbol_iterator(Symbol(this, getNumberOfSymbols(), 0));
}

uint32_t Archive::getNumberOfSymbols() const {
  if (!hasSymbolTable())
    return 0;
  const char *buf = getSymbolTable().begin();
  if (kind() == K_GNU)
    return read32be(buf);
  if (kind() == K_MIPS64)
    return read64be(buf);
  if (kind() == K_BSD)
    return read32le(buf) / 8;
  uint32_t member_count = 0;
  member_count = read32le(buf);
  buf += 4 + (member_count * 4); // Skip offsets.
  return read32le(buf);
}

Archive::child_iterator Archive::findSym(StringRef name) const {
  Archive::symbol_iterator bs = symbol_begin();
  Archive::symbol_iterator es = symbol_end();

  for (; bs != es; ++bs) {
    StringRef SymName = bs->getName();
    if (SymName == name) {
      ErrorOr<Archive::child_iterator> ResultOrErr = bs->getMember();
      // FIXME: Should we really eat the error?
      if (ResultOrErr.getError())
        return child_end();
      return ResultOrErr.get();
    }
  }
  return child_end();
}

bool Archive::hasSymbolTable() const {
  return SymbolTable != child_end();
}
