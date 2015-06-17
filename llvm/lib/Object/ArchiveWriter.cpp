//===- ArchiveWriter.cpp - ar File Format implementation --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the writeArchive function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ArchiveWriter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using namespace llvm;

NewArchiveIterator::NewArchiveIterator() {}

NewArchiveIterator::NewArchiveIterator(object::Archive::child_iterator I,
                                       StringRef Name)
    : IsNewMember(false), Name(Name), OldI(I) {}

NewArchiveIterator::NewArchiveIterator(StringRef NewFilename, StringRef Name)
    : IsNewMember(true), Name(Name), NewFilename(NewFilename) {}

StringRef NewArchiveIterator::getName() const { return Name; }

bool NewArchiveIterator::isNewMember() const { return IsNewMember; }

object::Archive::child_iterator NewArchiveIterator::getOld() const {
  assert(!IsNewMember);
  return OldI;
}

StringRef NewArchiveIterator::getNew() const {
  assert(IsNewMember);
  return NewFilename;
}

llvm::ErrorOr<int>
NewArchiveIterator::getFD(sys::fs::file_status &NewStatus) const {
  assert(IsNewMember);
  int NewFD;
  if (auto EC = sys::fs::openFileForRead(NewFilename, NewFD))
    return EC;
  assert(NewFD != -1);

  if (auto EC = sys::fs::status(NewFD, NewStatus))
    return EC;

  // Opening a directory doesn't make sense. Let it fail.
  // Linux cannot open directories with open(2), although
  // cygwin and *bsd can.
  if (NewStatus.type() == sys::fs::file_type::directory_file)
    return make_error_code(errc::is_a_directory);

  return NewFD;
}

template <typename T>
static void printWithSpacePadding(raw_fd_ostream &OS, T Data, unsigned Size,
				  bool MayTruncate = false) {
  uint64_t OldPos = OS.tell();
  OS << Data;
  unsigned SizeSoFar = OS.tell() - OldPos;
  if (Size > SizeSoFar) {
    OS.indent(Size - SizeSoFar);
  } else if (Size < SizeSoFar) {
    assert(MayTruncate && "Data doesn't fit in Size");
    // Some of the data this is used for (like UID) can be larger than the
    // space available in the archive format. Truncate in that case.
    OS.seek(OldPos + Size);
  }
}

static void print32BE(raw_ostream &Out, uint32_t Val) {
  support::endian::Writer<support::big>(Out).write(Val);
}

static void printRestOfMemberHeader(raw_fd_ostream &Out,
                                    const sys::TimeValue &ModTime, unsigned UID,
                                    unsigned GID, unsigned Perms,
                                    unsigned Size) {
  printWithSpacePadding(Out, ModTime.toEpochTime(), 12);
  printWithSpacePadding(Out, UID, 6, true);
  printWithSpacePadding(Out, GID, 6, true);
  printWithSpacePadding(Out, format("%o", Perms), 8);
  printWithSpacePadding(Out, Size, 10);
  Out << "`\n";
}

static void printMemberHeader(raw_fd_ostream &Out, StringRef Name,
                              const sys::TimeValue &ModTime, unsigned UID,
                              unsigned GID, unsigned Perms, unsigned Size) {
  printWithSpacePadding(Out, Twine(Name) + "/", 16);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms, Size);
}

static void printMemberHeader(raw_fd_ostream &Out, unsigned NameOffset,
                              const sys::TimeValue &ModTime, unsigned UID,
                              unsigned GID, unsigned Perms, unsigned Size) {
  Out << '/';
  printWithSpacePadding(Out, NameOffset, 15);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms, Size);
}

static void writeStringTable(raw_fd_ostream &Out,
                             ArrayRef<NewArchiveIterator> Members,
                             std::vector<unsigned> &StringMapIndexes) {
  unsigned StartOffset = 0;
  for (ArrayRef<NewArchiveIterator>::iterator I = Members.begin(),
                                              E = Members.end();
       I != E; ++I) {
    StringRef Name = I->getName();
    if (Name.size() < 16)
      continue;
    if (StartOffset == 0) {
      printWithSpacePadding(Out, "//", 58);
      Out << "`\n";
      StartOffset = Out.tell();
    }
    StringMapIndexes.push_back(Out.tell() - StartOffset);
    Out << Name << "/\n";
  }
  if (StartOffset == 0)
    return;
  if (Out.tell() % 2)
    Out << '\n';
  int Pos = Out.tell();
  Out.seek(StartOffset - 12);
  printWithSpacePadding(Out, Pos - StartOffset, 10);
  Out.seek(Pos);
}

// Returns the offset of the first reference to a member offset.
static ErrorOr<unsigned>
writeSymbolTable(raw_fd_ostream &Out, ArrayRef<NewArchiveIterator> Members,
                 ArrayRef<MemoryBufferRef> Buffers,
                 std::vector<unsigned> &MemberOffsetRefs) {
  unsigned StartOffset = 0;
  unsigned MemberNum = 0;
  std::string NameBuf;
  raw_string_ostream NameOS(NameBuf);
  unsigned NumSyms = 0;
  LLVMContext Context;
  for (ArrayRef<NewArchiveIterator>::iterator I = Members.begin(),
                                              E = Members.end();
       I != E; ++I, ++MemberNum) {
    MemoryBufferRef MemberBuffer = Buffers[MemberNum];
    ErrorOr<std::unique_ptr<object::SymbolicFile>> ObjOrErr =
        object::SymbolicFile::createSymbolicFile(
            MemberBuffer, sys::fs::file_magic::unknown, &Context);
    if (!ObjOrErr)
      continue;  // FIXME: check only for "not an object file" errors.
    object::SymbolicFile &Obj = *ObjOrErr.get();

    if (!StartOffset) {
      printMemberHeader(Out, "", sys::TimeValue::now(), 0, 0, 0, 0);
      StartOffset = Out.tell();
      print32BE(Out, 0);
    }

    for (const object::BasicSymbolRef &S : Obj.symbols()) {
      uint32_t Symflags = S.getFlags();
      if (Symflags & object::SymbolRef::SF_FormatSpecific)
        continue;
      if (!(Symflags & object::SymbolRef::SF_Global))
        continue;
      if (Symflags & object::SymbolRef::SF_Undefined)
        continue;
      if (auto EC = S.printName(NameOS))
        return EC;
      NameOS << '\0';
      ++NumSyms;
      MemberOffsetRefs.push_back(MemberNum);
      print32BE(Out, 0);
    }
  }
  Out << NameOS.str();

  if (StartOffset == 0)
    return 0;

  if (Out.tell() % 2)
    Out << '\0';

  unsigned Pos = Out.tell();
  Out.seek(StartOffset - 12);
  printWithSpacePadding(Out, Pos - StartOffset, 10);
  Out.seek(StartOffset);
  print32BE(Out, NumSyms);
  Out.seek(Pos);
  return StartOffset + 4;
}

std::pair<StringRef, std::error_code>
llvm::writeArchive(StringRef ArcName,
                   std::vector<NewArchiveIterator> &NewMembers,
                   bool WriteSymtab) {
  SmallString<128> TmpArchive;
  int TmpArchiveFD;
  if (auto EC = sys::fs::createUniqueFile(ArcName + ".temp-archive-%%%%%%%.a",
                                          TmpArchiveFD, TmpArchive))
    return std::make_pair(ArcName, EC);

  tool_output_file Output(TmpArchive, TmpArchiveFD);
  raw_fd_ostream &Out = Output.os();
  Out << "!<arch>\n";

  std::vector<unsigned> MemberOffsetRefs;

  std::vector<std::unique_ptr<MemoryBuffer>> Buffers;
  std::vector<MemoryBufferRef> Members;
  std::vector<sys::fs::file_status> NewMemberStatus;

  for (unsigned I = 0, N = NewMembers.size(); I < N; ++I) {
    NewArchiveIterator &Member = NewMembers[I];
    MemoryBufferRef MemberRef;

    if (Member.isNewMember()) {
      StringRef Filename = Member.getNew();
      NewMemberStatus.resize(NewMemberStatus.size() + 1);
      sys::fs::file_status &Status = NewMemberStatus.back();
      ErrorOr<int> FD = Member.getFD(Status);
      if (auto EC = FD.getError())
        return std::make_pair(Filename, EC);
      ErrorOr<std::unique_ptr<MemoryBuffer>> MemberBufferOrErr =
          MemoryBuffer::getOpenFile(FD.get(), Filename, Status.getSize(),
                                    false);
      if (auto EC = MemberBufferOrErr.getError())
        return std::make_pair(Filename, EC);
      if (close(FD.get()) != 0)
        return std::make_pair(Filename,
                              std::error_code(errno, std::generic_category()));
      Buffers.push_back(std::move(MemberBufferOrErr.get()));
      MemberRef = Buffers.back()->getMemBufferRef();
    } else {
      object::Archive::child_iterator OldMember = Member.getOld();
      ErrorOr<MemoryBufferRef> MemberBufferOrErr =
          OldMember->getMemoryBufferRef();
      if (auto EC = MemberBufferOrErr.getError())
        return std::make_pair("", EC);
      MemberRef = MemberBufferOrErr.get();
    }
    Members.push_back(MemberRef);
  }

  unsigned MemberReferenceOffset = 0;
  if (WriteSymtab) {
    ErrorOr<unsigned> MemberReferenceOffsetOrErr =
        writeSymbolTable(Out, NewMembers, Members, MemberOffsetRefs);
    if (auto EC = MemberReferenceOffsetOrErr.getError())
      return std::make_pair(ArcName, EC);
    MemberReferenceOffset = MemberReferenceOffsetOrErr.get();
  }

  std::vector<unsigned> StringMapIndexes;
  writeStringTable(Out, NewMembers, StringMapIndexes);

  unsigned MemberNum = 0;
  unsigned LongNameMemberNum = 0;
  unsigned NewMemberNum = 0;
  std::vector<unsigned> MemberOffset;
  for (std::vector<NewArchiveIterator>::iterator I = NewMembers.begin(),
                                                 E = NewMembers.end();
       I != E; ++I, ++MemberNum) {

    unsigned Pos = Out.tell();
    MemberOffset.push_back(Pos);

    MemoryBufferRef File = Members[MemberNum];
    if (I->isNewMember()) {
      StringRef FileName = I->getNew();
      const sys::fs::file_status &Status = NewMemberStatus[NewMemberNum];
      NewMemberNum++;

      StringRef Name = sys::path::filename(FileName);
      if (Name.size() < 16)
        printMemberHeader(Out, Name, Status.getLastModificationTime(),
                          Status.getUser(), Status.getGroup(),
                          Status.permissions(), Status.getSize());
      else
        printMemberHeader(Out, StringMapIndexes[LongNameMemberNum++],
                          Status.getLastModificationTime(), Status.getUser(),
                          Status.getGroup(), Status.permissions(),
                          Status.getSize());
    } else {
      object::Archive::child_iterator OldMember = I->getOld();
      StringRef Name = I->getName();

      if (Name.size() < 16)
        printMemberHeader(Out, Name, OldMember->getLastModified(),
                          OldMember->getUID(), OldMember->getGID(),
                          OldMember->getAccessMode(), OldMember->getSize());
      else
        printMemberHeader(Out, StringMapIndexes[LongNameMemberNum++],
                          OldMember->getLastModified(), OldMember->getUID(),
                          OldMember->getGID(), OldMember->getAccessMode(),
                          OldMember->getSize());
    }

    Out << File.getBuffer();

    if (Out.tell() % 2)
      Out << '\n';
  }

  if (MemberReferenceOffset) {
    Out.seek(MemberReferenceOffset);
    for (unsigned MemberNum : MemberOffsetRefs)
      print32BE(Out, MemberOffset[MemberNum]);
  }

  Output.keep();
  Out.close();
  sys::fs::rename(TmpArchive, ArcName);
  return std::make_pair("", std::error_code());
}
