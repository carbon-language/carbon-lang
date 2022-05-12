//===- ArchiveWriter.cpp - ar File Format implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the writeArchive function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ArchiveWriter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using namespace llvm;

NewArchiveMember::NewArchiveMember(MemoryBufferRef BufRef)
    : Buf(MemoryBuffer::getMemBuffer(BufRef, false)),
      MemberName(BufRef.getBufferIdentifier()) {}

Expected<NewArchiveMember>
NewArchiveMember::getOldMember(const object::Archive::Child &OldMember,
                               bool Deterministic) {
  Expected<llvm::MemoryBufferRef> BufOrErr = OldMember.getMemoryBufferRef();
  if (!BufOrErr)
    return BufOrErr.takeError();

  NewArchiveMember M;
  M.Buf = MemoryBuffer::getMemBuffer(*BufOrErr, false);
  M.MemberName = M.Buf->getBufferIdentifier();
  if (!Deterministic) {
    auto ModTimeOrErr = OldMember.getLastModified();
    if (!ModTimeOrErr)
      return ModTimeOrErr.takeError();
    M.ModTime = ModTimeOrErr.get();
    Expected<unsigned> UIDOrErr = OldMember.getUID();
    if (!UIDOrErr)
      return UIDOrErr.takeError();
    M.UID = UIDOrErr.get();
    Expected<unsigned> GIDOrErr = OldMember.getGID();
    if (!GIDOrErr)
      return GIDOrErr.takeError();
    M.GID = GIDOrErr.get();
    Expected<sys::fs::perms> AccessModeOrErr = OldMember.getAccessMode();
    if (!AccessModeOrErr)
      return AccessModeOrErr.takeError();
    M.Perms = AccessModeOrErr.get();
  }
  return std::move(M);
}

Expected<NewArchiveMember> NewArchiveMember::getFile(StringRef FileName,
                                                     bool Deterministic) {
  sys::fs::file_status Status;
  auto FDOrErr = sys::fs::openNativeFileForRead(FileName);
  if (!FDOrErr)
    return FDOrErr.takeError();
  sys::fs::file_t FD = *FDOrErr;
  assert(FD != sys::fs::kInvalidFile);

  if (auto EC = sys::fs::status(FD, Status))
    return errorCodeToError(EC);

  // Opening a directory doesn't make sense. Let it fail.
  // Linux cannot open directories with open(2), although
  // cygwin and *bsd can.
  if (Status.type() == sys::fs::file_type::directory_file)
    return errorCodeToError(make_error_code(errc::is_a_directory));

  ErrorOr<std::unique_ptr<MemoryBuffer>> MemberBufferOrErr =
      MemoryBuffer::getOpenFile(FD, FileName, Status.getSize(), false);
  if (!MemberBufferOrErr)
    return errorCodeToError(MemberBufferOrErr.getError());

  if (auto EC = sys::fs::closeFile(FD))
    return errorCodeToError(EC);

  NewArchiveMember M;
  M.Buf = std::move(*MemberBufferOrErr);
  M.MemberName = M.Buf->getBufferIdentifier();
  if (!Deterministic) {
    M.ModTime = std::chrono::time_point_cast<std::chrono::seconds>(
        Status.getLastModificationTime());
    M.UID = Status.getUser();
    M.GID = Status.getGroup();
    M.Perms = Status.permissions();
  }
  return std::move(M);
}

template <typename T>
static void printWithSpacePadding(raw_ostream &OS, T Data, unsigned Size) {
  uint64_t OldPos = OS.tell();
  OS << Data;
  unsigned SizeSoFar = OS.tell() - OldPos;
  assert(SizeSoFar <= Size && "Data doesn't fit in Size");
  OS.indent(Size - SizeSoFar);
}

static bool isDarwin(object::Archive::Kind Kind) {
  return Kind == object::Archive::K_DARWIN ||
         Kind == object::Archive::K_DARWIN64;
}

static bool isBSDLike(object::Archive::Kind Kind) {
  switch (Kind) {
  case object::Archive::K_GNU:
  case object::Archive::K_GNU64:
    return false;
  case object::Archive::K_BSD:
  case object::Archive::K_DARWIN:
  case object::Archive::K_DARWIN64:
    return true;
  case object::Archive::K_AIXBIG:
  case object::Archive::K_COFF:
    break;
  }
  llvm_unreachable("not supported for writting");
}

template <class T>
static void print(raw_ostream &Out, object::Archive::Kind Kind, T Val) {
  support::endian::write(Out, Val,
                         isBSDLike(Kind) ? support::little : support::big);
}

static void printRestOfMemberHeader(
    raw_ostream &Out, const sys::TimePoint<std::chrono::seconds> &ModTime,
    unsigned UID, unsigned GID, unsigned Perms, uint64_t Size) {
  printWithSpacePadding(Out, sys::toTimeT(ModTime), 12);

  // The format has only 6 chars for uid and gid. Truncate if the provided
  // values don't fit.
  printWithSpacePadding(Out, UID % 1000000, 6);
  printWithSpacePadding(Out, GID % 1000000, 6);

  printWithSpacePadding(Out, format("%o", Perms), 8);
  printWithSpacePadding(Out, Size, 10);
  Out << "`\n";
}

static void
printGNUSmallMemberHeader(raw_ostream &Out, StringRef Name,
                          const sys::TimePoint<std::chrono::seconds> &ModTime,
                          unsigned UID, unsigned GID, unsigned Perms,
                          uint64_t Size) {
  printWithSpacePadding(Out, Twine(Name) + "/", 16);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms, Size);
}

static void
printBSDMemberHeader(raw_ostream &Out, uint64_t Pos, StringRef Name,
                     const sys::TimePoint<std::chrono::seconds> &ModTime,
                     unsigned UID, unsigned GID, unsigned Perms, uint64_t Size) {
  uint64_t PosAfterHeader = Pos + 60 + Name.size();
  // Pad so that even 64 bit object files are aligned.
  unsigned Pad = offsetToAlignment(PosAfterHeader, Align(8));
  unsigned NameWithPadding = Name.size() + Pad;
  printWithSpacePadding(Out, Twine("#1/") + Twine(NameWithPadding), 16);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms,
                          NameWithPadding + Size);
  Out << Name;
  while (Pad--)
    Out.write(uint8_t(0));
}

static bool useStringTable(bool Thin, StringRef Name) {
  return Thin || Name.size() >= 16 || Name.contains('/');
}

static bool is64BitKind(object::Archive::Kind Kind) {
  switch (Kind) {
  case object::Archive::K_GNU:
  case object::Archive::K_BSD:
  case object::Archive::K_DARWIN:
  case object::Archive::K_COFF:
  case object::Archive::K_AIXBIG:
    return false;
  case object::Archive::K_DARWIN64:
  case object::Archive::K_GNU64:
    return true;
  }
  llvm_unreachable("not supported for writting");
}

static void
printMemberHeader(raw_ostream &Out, uint64_t Pos, raw_ostream &StringTable,
                  StringMap<uint64_t> &MemberNames, object::Archive::Kind Kind,
                  bool Thin, const NewArchiveMember &M,
                  sys::TimePoint<std::chrono::seconds> ModTime, uint64_t Size) {
  if (isBSDLike(Kind))
    return printBSDMemberHeader(Out, Pos, M.MemberName, ModTime, M.UID, M.GID,
                                M.Perms, Size);
  if (!useStringTable(Thin, M.MemberName))
    return printGNUSmallMemberHeader(Out, M.MemberName, ModTime, M.UID, M.GID,
                                     M.Perms, Size);
  Out << '/';
  uint64_t NamePos;
  if (Thin) {
    NamePos = StringTable.tell();
    StringTable << M.MemberName << "/\n";
  } else {
    auto Insertion = MemberNames.insert({M.MemberName, uint64_t(0)});
    if (Insertion.second) {
      Insertion.first->second = StringTable.tell();
      StringTable << M.MemberName << "/\n";
    }
    NamePos = Insertion.first->second;
  }
  printWithSpacePadding(Out, NamePos, 15);
  printRestOfMemberHeader(Out, ModTime, M.UID, M.GID, M.Perms, Size);
}

namespace {
struct MemberData {
  std::vector<unsigned> Symbols;
  std::string Header;
  StringRef Data;
  StringRef Padding;
};
} // namespace

static MemberData computeStringTable(StringRef Names) {
  unsigned Size = Names.size();
  unsigned Pad = offsetToAlignment(Size, Align(2));
  std::string Header;
  raw_string_ostream Out(Header);
  printWithSpacePadding(Out, "//", 48);
  printWithSpacePadding(Out, Size + Pad, 10);
  Out << "`\n";
  Out.flush();
  return {{}, std::move(Header), Names, Pad ? "\n" : ""};
}

static sys::TimePoint<std::chrono::seconds> now(bool Deterministic) {
  using namespace std::chrono;

  if (!Deterministic)
    return time_point_cast<seconds>(system_clock::now());
  return sys::TimePoint<seconds>();
}

static bool isArchiveSymbol(const object::BasicSymbolRef &S) {
  Expected<uint32_t> SymFlagsOrErr = S.getFlags();
  if (!SymFlagsOrErr)
    // TODO: Actually report errors helpfully.
    report_fatal_error(SymFlagsOrErr.takeError());
  if (*SymFlagsOrErr & object::SymbolRef::SF_FormatSpecific)
    return false;
  if (!(*SymFlagsOrErr & object::SymbolRef::SF_Global))
    return false;
  if (*SymFlagsOrErr & object::SymbolRef::SF_Undefined)
    return false;
  return true;
}

static void printNBits(raw_ostream &Out, object::Archive::Kind Kind,
                       uint64_t Val) {
  if (is64BitKind(Kind))
    print<uint64_t>(Out, Kind, Val);
  else
    print<uint32_t>(Out, Kind, Val);
}

static uint64_t computeSymbolTableSize(object::Archive::Kind Kind,
                                       uint64_t NumSyms, uint64_t OffsetSize,
                                       StringRef StringTable,
                                       uint32_t *Padding = nullptr) {
  assert((OffsetSize == 4 || OffsetSize == 8) && "Unsupported OffsetSize");
  uint64_t Size = OffsetSize; // Number of entries
  if (isBSDLike(Kind))
    Size += NumSyms * OffsetSize * 2; // Table
  else
    Size += NumSyms * OffsetSize; // Table
  if (isBSDLike(Kind))
    Size += OffsetSize; // byte count
  Size += StringTable.size();
  // ld64 expects the members to be 8-byte aligned for 64-bit content and at
  // least 4-byte aligned for 32-bit content.  Opt for the larger encoding
  // uniformly.
  // We do this for all bsd formats because it simplifies aligning members.
  uint32_t Pad = offsetToAlignment(Size, Align(isBSDLike(Kind) ? 8 : 2));
  Size += Pad;
  if (Padding)
    *Padding = Pad;
  return Size;
}

static void writeSymbolTableHeader(raw_ostream &Out, object::Archive::Kind Kind,
                                   bool Deterministic, uint64_t Size) {
  if (isBSDLike(Kind)) {
    const char *Name = is64BitKind(Kind) ? "__.SYMDEF_64" : "__.SYMDEF";
    printBSDMemberHeader(Out, Out.tell(), Name, now(Deterministic), 0, 0, 0,
                         Size);
  } else {
    const char *Name = is64BitKind(Kind) ? "/SYM64" : "";
    printGNUSmallMemberHeader(Out, Name, now(Deterministic), 0, 0, 0, Size);
  }
}

static void writeSymbolTable(raw_ostream &Out, object::Archive::Kind Kind,
                             bool Deterministic, ArrayRef<MemberData> Members,
                             StringRef StringTable) {
  // We don't write a symbol table on an archive with no members -- except on
  // Darwin, where the linker will abort unless the archive has a symbol table.
  if (StringTable.empty() && !isDarwin(Kind))
    return;

  unsigned NumSyms = 0;
  for (const MemberData &M : Members)
    NumSyms += M.Symbols.size();

  uint64_t OffsetSize = is64BitKind(Kind) ? 8 : 4;
  uint32_t Pad;
  uint64_t Size = computeSymbolTableSize(Kind, NumSyms, OffsetSize, StringTable, &Pad);
  writeSymbolTableHeader(Out, Kind, Deterministic, Size);

  uint64_t Pos = Out.tell() + Size;

  if (isBSDLike(Kind))
    printNBits(Out, Kind, NumSyms * 2 * OffsetSize);
  else
    printNBits(Out, Kind, NumSyms);

  for (const MemberData &M : Members) {
    for (unsigned StringOffset : M.Symbols) {
      if (isBSDLike(Kind))
        printNBits(Out, Kind, StringOffset);
      printNBits(Out, Kind, Pos); // member offset
    }
    Pos += M.Header.size() + M.Data.size() + M.Padding.size();
  }

  if (isBSDLike(Kind))
    // byte count of the string table
    printNBits(Out, Kind, StringTable.size());
  Out << StringTable;

  while (Pad--)
    Out.write(uint8_t(0));
}

static Expected<std::vector<unsigned>>
getSymbols(MemoryBufferRef Buf, raw_ostream &SymNames, bool &HasObject) {
  std::vector<unsigned> Ret;

  // In the scenario when LLVMContext is populated SymbolicFile will contain a
  // reference to it, thus SymbolicFile should be destroyed first.
  LLVMContext Context;
  std::unique_ptr<object::SymbolicFile> Obj;

  const file_magic Type = identify_magic(Buf.getBuffer());
  // Treat unsupported file types as having no symbols.
  if (!object::SymbolicFile::isSymbolicFile(Type, &Context))
    return Ret;
  if (Type == file_magic::bitcode) {
    auto ObjOrErr = object::SymbolicFile::createSymbolicFile(
        Buf, file_magic::bitcode, &Context);
    if (!ObjOrErr)
      return ObjOrErr.takeError();
    Obj = std::move(*ObjOrErr);
  } else {
    auto ObjOrErr = object::SymbolicFile::createSymbolicFile(Buf);
    if (!ObjOrErr)
      return ObjOrErr.takeError();
    Obj = std::move(*ObjOrErr);
  }

  HasObject = true;
  for (const object::BasicSymbolRef &S : Obj->symbols()) {
    if (!isArchiveSymbol(S))
      continue;
    Ret.push_back(SymNames.tell());
    if (Error E = S.printName(SymNames))
      return std::move(E);
    SymNames << '\0';
  }
  return Ret;
}

static Expected<std::vector<MemberData>>
computeMemberData(raw_ostream &StringTable, raw_ostream &SymNames,
                  object::Archive::Kind Kind, bool Thin, bool Deterministic,
                  bool NeedSymbols, ArrayRef<NewArchiveMember> NewMembers) {
  static char PaddingData[8] = {'\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n'};

  // This ignores the symbol table, but we only need the value mod 8 and the
  // symbol table is aligned to be a multiple of 8 bytes
  uint64_t Pos = 0;

  std::vector<MemberData> Ret;
  bool HasObject = false;

  // Deduplicate long member names in the string table and reuse earlier name
  // offsets. This especially saves space for COFF Import libraries where all
  // members have the same name.
  StringMap<uint64_t> MemberNames;

  // UniqueTimestamps is a special case to improve debugging on Darwin:
  //
  // The Darwin linker does not link debug info into the final
  // binary. Instead, it emits entries of type N_OSO in in the output
  // binary's symbol table, containing references to the linked-in
  // object files. Using that reference, the debugger can read the
  // debug data directly from the object files. Alternatively, an
  // invocation of 'dsymutil' will link the debug data from the object
  // files into a dSYM bundle, which can be loaded by the debugger,
  // instead of the object files.
  //
  // For an object file, the N_OSO entries contain the absolute path
  // path to the file, and the file's timestamp. For an object
  // included in an archive, the path is formatted like
  // "/absolute/path/to/archive.a(member.o)", and the timestamp is the
  // archive member's timestamp, rather than the archive's timestamp.
  //
  // However, this doesn't always uniquely identify an object within
  // an archive -- an archive file can have multiple entries with the
  // same filename. (This will happen commonly if the original object
  // files started in different directories.) The only way they get
  // distinguished, then, is via the timestamp. But this process is
  // unable to find the correct object file in the archive when there
  // are two files of the same name and timestamp.
  //
  // Additionally, timestamp==0 is treated specially, and causes the
  // timestamp to be ignored as a match criteria.
  //
  // That will "usually" work out okay when creating an archive not in
  // deterministic timestamp mode, because the objects will probably
  // have been created at different timestamps.
  //
  // To ameliorate this problem, in deterministic archive mode (which
  // is the default), on Darwin we will emit a unique non-zero
  // timestamp for each entry with a duplicated name. This is still
  // deterministic: the only thing affecting that timestamp is the
  // order of the files in the resultant archive.
  //
  // See also the functions that handle the lookup:
  // in lldb: ObjectContainerBSDArchive::Archive::FindObject()
  // in llvm/tools/dsymutil: BinaryHolder::GetArchiveMemberBuffers().
  bool UniqueTimestamps = Deterministic && isDarwin(Kind);
  std::map<StringRef, unsigned> FilenameCount;
  if (UniqueTimestamps) {
    for (const NewArchiveMember &M : NewMembers)
      FilenameCount[M.MemberName]++;
    for (auto &Entry : FilenameCount)
      Entry.second = Entry.second > 1 ? 1 : 0;
  }

  for (const NewArchiveMember &M : NewMembers) {
    std::string Header;
    raw_string_ostream Out(Header);

    MemoryBufferRef Buf = M.Buf->getMemBufferRef();
    StringRef Data = Thin ? "" : Buf.getBuffer();

    // ld64 expects the members to be 8-byte aligned for 64-bit content and at
    // least 4-byte aligned for 32-bit content.  Opt for the larger encoding
    // uniformly.  This matches the behaviour with cctools and ensures that ld64
    // is happy with archives that we generate.
    unsigned MemberPadding =
        isDarwin(Kind) ? offsetToAlignment(Data.size(), Align(8)) : 0;
    unsigned TailPadding =
        offsetToAlignment(Data.size() + MemberPadding, Align(2));
    StringRef Padding = StringRef(PaddingData, MemberPadding + TailPadding);

    sys::TimePoint<std::chrono::seconds> ModTime;
    if (UniqueTimestamps)
      // Increment timestamp for each file of a given name.
      ModTime = sys::toTimePoint(FilenameCount[M.MemberName]++);
    else
      ModTime = M.ModTime;

    uint64_t Size = Buf.getBufferSize() + MemberPadding;
    if (Size > object::Archive::MaxMemberSize) {
      std::string StringMsg =
          "File " + M.MemberName.str() + " exceeds size limit";
      return make_error<object::GenericBinaryError>(
          std::move(StringMsg), object::object_error::parse_failed);
    }

    printMemberHeader(Out, Pos, StringTable, MemberNames, Kind, Thin, M,
                      ModTime, Size);
    Out.flush();

    std::vector<unsigned> Symbols;
    if (NeedSymbols) {
      Expected<std::vector<unsigned>> SymbolsOrErr =
          getSymbols(Buf, SymNames, HasObject);
      if (auto E = SymbolsOrErr.takeError())
        return std::move(E);
      Symbols = std::move(*SymbolsOrErr);
    }

    Pos += Header.size() + Data.size() + Padding.size();
    Ret.push_back({std::move(Symbols), std::move(Header), Data, Padding});
  }
  // If there are no symbols, emit an empty symbol table, to satisfy Solaris
  // tools, older versions of which expect a symbol table in a non-empty
  // archive, regardless of whether there are any symbols in it.
  if (HasObject && SymNames.tell() == 0)
    SymNames << '\0' << '\0' << '\0';
  return Ret;
}

namespace llvm {

static ErrorOr<SmallString<128>> canonicalizePath(StringRef P) {
  SmallString<128> Ret = P;
  std::error_code Err = sys::fs::make_absolute(Ret);
  if (Err)
    return Err;
  sys::path::remove_dots(Ret, /*removedotdot*/ true);
  return Ret;
}

// Compute the relative path from From to To.
Expected<std::string> computeArchiveRelativePath(StringRef From, StringRef To) {
  ErrorOr<SmallString<128>> PathToOrErr = canonicalizePath(To);
  ErrorOr<SmallString<128>> DirFromOrErr = canonicalizePath(From);
  if (!PathToOrErr || !DirFromOrErr)
    return errorCodeToError(std::error_code(errno, std::generic_category()));

  const SmallString<128> &PathTo = *PathToOrErr;
  const SmallString<128> &DirFrom = sys::path::parent_path(*DirFromOrErr);

  // Can't construct a relative path between different roots
  if (sys::path::root_name(PathTo) != sys::path::root_name(DirFrom))
    return sys::path::convert_to_slash(PathTo);

  // Skip common prefixes
  auto FromTo =
      std::mismatch(sys::path::begin(DirFrom), sys::path::end(DirFrom),
                    sys::path::begin(PathTo));
  auto FromI = FromTo.first;
  auto ToI = FromTo.second;

  // Construct relative path
  SmallString<128> Relative;
  for (auto FromE = sys::path::end(DirFrom); FromI != FromE; ++FromI)
    sys::path::append(Relative, sys::path::Style::posix, "..");

  for (auto ToE = sys::path::end(PathTo); ToI != ToE; ++ToI)
    sys::path::append(Relative, sys::path::Style::posix, *ToI);

  return std::string(Relative.str());
}

static Error writeArchiveToStream(raw_ostream &Out,
                                  ArrayRef<NewArchiveMember> NewMembers,
                                  bool WriteSymtab, object::Archive::Kind Kind,
                                  bool Deterministic, bool Thin) {
  assert((!Thin || !isBSDLike(Kind)) && "Only the gnu format has a thin mode");

  SmallString<0> SymNamesBuf;
  raw_svector_ostream SymNames(SymNamesBuf);
  SmallString<0> StringTableBuf;
  raw_svector_ostream StringTable(StringTableBuf);

  Expected<std::vector<MemberData>> DataOrErr =
      computeMemberData(StringTable, SymNames, Kind, Thin, Deterministic,
                        WriteSymtab, NewMembers);
  if (Error E = DataOrErr.takeError())
    return E;
  std::vector<MemberData> &Data = *DataOrErr;

  if (!StringTableBuf.empty())
    Data.insert(Data.begin(), computeStringTable(StringTableBuf));

  // We would like to detect if we need to switch to a 64-bit symbol table.
  if (WriteSymtab) {
    uint64_t MaxOffset = 8; // For the file signature.
    uint64_t LastOffset = MaxOffset;
    uint64_t NumSyms = 0;
    for (const auto &M : Data) {
      // Record the start of the member's offset
      LastOffset = MaxOffset;
      // Account for the size of each part associated with the member.
      MaxOffset += M.Header.size() + M.Data.size() + M.Padding.size();
      NumSyms += M.Symbols.size();
    }

    // We assume 32-bit offsets to see if 32-bit symbols are possible or not.
    uint64_t SymtabSize = computeSymbolTableSize(Kind, NumSyms, 4, SymNamesBuf);
    auto computeSymbolTableHeaderSize =
        [=] {
          SmallString<0> TmpBuf;
          raw_svector_ostream Tmp(TmpBuf);
          writeSymbolTableHeader(Tmp, Kind, Deterministic, SymtabSize);
          return TmpBuf.size();
        };
    LastOffset += computeSymbolTableHeaderSize() + SymtabSize;

    // The SYM64 format is used when an archive's member offsets are larger than
    // 32-bits can hold. The need for this shift in format is detected by
    // writeArchive. To test this we need to generate a file with a member that
    // has an offset larger than 32-bits but this demands a very slow test. To
    // speed the test up we use this environment variable to pretend like the
    // cutoff happens before 32-bits and instead happens at some much smaller
    // value.
    uint64_t Sym64Threshold = 1ULL << 32;
    const char *Sym64Env = std::getenv("SYM64_THRESHOLD");
    if (Sym64Env)
      StringRef(Sym64Env).getAsInteger(10, Sym64Threshold);

    // If LastOffset isn't going to fit in a 32-bit varible we need to switch
    // to 64-bit. Note that the file can be larger than 4GB as long as the last
    // member starts before the 4GB offset.
    if (LastOffset >= Sym64Threshold) {
      if (Kind == object::Archive::K_DARWIN)
        Kind = object::Archive::K_DARWIN64;
      else
        Kind = object::Archive::K_GNU64;
    }
  }

  if (Thin)
    Out << "!<thin>\n";
  else
    Out << "!<arch>\n";

  if (WriteSymtab)
    writeSymbolTable(Out, Kind, Deterministic, Data, SymNamesBuf);

  for (const MemberData &M : Data)
    Out << M.Header << M.Data << M.Padding;

  Out.flush();
  return Error::success();
}

Error writeArchive(StringRef ArcName, ArrayRef<NewArchiveMember> NewMembers,
                   bool WriteSymtab, object::Archive::Kind Kind,
                   bool Deterministic, bool Thin,
                   std::unique_ptr<MemoryBuffer> OldArchiveBuf) {
  Expected<sys::fs::TempFile> Temp =
      sys::fs::TempFile::create(ArcName + ".temp-archive-%%%%%%%.a");
  if (!Temp)
    return Temp.takeError();
  raw_fd_ostream Out(Temp->FD, false);

  if (Error E = writeArchiveToStream(Out, NewMembers, WriteSymtab, Kind,
                                     Deterministic, Thin)) {
    if (Error DiscardError = Temp->discard())
      return joinErrors(std::move(E), std::move(DiscardError));
    return E;
  }

  // At this point, we no longer need whatever backing memory
  // was used to generate the NewMembers. On Windows, this buffer
  // could be a mapped view of the file we want to replace (if
  // we're updating an existing archive, say). In that case, the
  // rename would still succeed, but it would leave behind a
  // temporary file (actually the original file renamed) because
  // a file cannot be deleted while there's a handle open on it,
  // only renamed. So by freeing this buffer, this ensures that
  // the last open handle on the destination file, if any, is
  // closed before we attempt to rename.
  OldArchiveBuf.reset();

  return Temp->keep(ArcName);
}

Expected<std::unique_ptr<MemoryBuffer>>
writeArchiveToBuffer(ArrayRef<NewArchiveMember> NewMembers, bool WriteSymtab,
                     object::Archive::Kind Kind, bool Deterministic,
                     bool Thin) {
  SmallVector<char, 0> ArchiveBufferVector;
  raw_svector_ostream ArchiveStream(ArchiveBufferVector);

  if (Error E = writeArchiveToStream(ArchiveStream, NewMembers, WriteSymtab,
                                     Kind, Deterministic, Thin))
    return std::move(E);

  return std::make_unique<SmallVectorMemoryBuffer>(
      std::move(ArchiveBufferVector), /*RequiresNullTerminator=*/false);
}

} // namespace llvm
