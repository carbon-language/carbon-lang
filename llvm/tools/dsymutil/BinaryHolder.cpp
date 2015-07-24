//===-- BinaryHolder.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that aims to be a dropin replacement for
// Darwin's dsymutil.
//
//===----------------------------------------------------------------------===//

#include "BinaryHolder.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace dsymutil {

Triple BinaryHolder::getTriple(const object::MachOObjectFile &Obj) {
  // If a ThumbTriple is returned, use it instead of the standard
  // one. This is because the thumb triple always allows to create a
  // target, whereas the non-thumb one might not.
  Triple ThumbTriple;
  Triple T = Obj.getArch(nullptr, &ThumbTriple);
  return ThumbTriple.getArch() ? ThumbTriple : T;
}

void BinaryHolder::changeBackingMemoryBuffer(
    std::unique_ptr<MemoryBuffer> &&Buf) {
  CurrentArchive.reset();
  CurrentObjectFile.reset();

  CurrentMemoryBuffer = std::move(Buf);
}

ErrorOr<MemoryBufferRef>
BinaryHolder::GetMemoryBufferForFile(StringRef Filename,
                                     sys::TimeValue Timestamp) {
  if (Verbose)
    outs() << "trying to open '" << Filename << "'\n";

  // Try that first as it doesn't involve any filesystem access.
  if (auto ErrOrArchiveMember = GetArchiveMemberBuffer(Filename, Timestamp))
    return *ErrOrArchiveMember;

  // If the name ends with a closing paren, there is a huge chance
  // it is an archive member specification.
  if (Filename.endswith(")"))
    if (auto ErrOrArchiveMember =
            MapArchiveAndGetMemberBuffer(Filename, Timestamp))
      return *ErrOrArchiveMember;

  // Otherwise, just try opening a standard file. If this is an
  // archive member specifiaction and any of the above didn't handle it
  // (either because the archive is not there anymore, or because the
  // archive doesn't contain the requested member), this will still
  // provide a sensible error message.
  auto ErrOrFile = MemoryBuffer::getFileOrSTDIN(Filename);
  if (auto Err = ErrOrFile.getError())
    return Err;

  changeBackingMemoryBuffer(std::move(*ErrOrFile));
  if (Verbose)
    outs() << "\tloaded file.\n";
  return CurrentMemoryBuffer->getMemBufferRef();
}

ErrorOr<MemoryBufferRef>
BinaryHolder::GetArchiveMemberBuffer(StringRef Filename,
                                     sys::TimeValue Timestamp) {
  if (!CurrentArchive)
    return make_error_code(errc::no_such_file_or_directory);

  StringRef CurArchiveName = CurrentArchive->getFileName();
  if (!Filename.startswith(Twine(CurArchiveName, "(").str()))
    return make_error_code(errc::no_such_file_or_directory);

  // Remove the archive name and the parens around the archive member name.
  Filename = Filename.substr(CurArchiveName.size() + 1).drop_back();

  for (const auto &Child : CurrentArchive->children()) {
    if (auto NameOrErr = Child.getName())
      if (*NameOrErr == Filename) {
        if (Timestamp != sys::TimeValue::PosixZeroTime() &&
            Timestamp != Child.getLastModified()) {
          if (Verbose)
            outs() << "\tmember had timestamp mismatch.\n";
          continue;
        }
        if (Verbose)
          outs() << "\tfound member in current archive.\n";
        return Child.getMemoryBufferRef();
      }
  }

  return make_error_code(errc::no_such_file_or_directory);
}

ErrorOr<MemoryBufferRef>
BinaryHolder::MapArchiveAndGetMemberBuffer(StringRef Filename,
                                           sys::TimeValue Timestamp) {
  StringRef ArchiveFilename = Filename.substr(0, Filename.find('('));

  auto ErrOrBuff = MemoryBuffer::getFileOrSTDIN(ArchiveFilename);
  if (auto Err = ErrOrBuff.getError())
    return Err;

  if (Verbose)
    outs() << "\topened new archive '" << ArchiveFilename << "'\n";

  changeBackingMemoryBuffer(std::move(*ErrOrBuff));
  auto ErrOrArchive =
      object::Archive::create(CurrentMemoryBuffer->getMemBufferRef());
  if (auto Err = ErrOrArchive.getError())
    return Err;

  CurrentArchive = std::move(*ErrOrArchive);

  return GetArchiveMemberBuffer(Filename, Timestamp);
}

ErrorOr<const object::ObjectFile &>
BinaryHolder::GetObjectFile(StringRef Filename, sys::TimeValue Timestamp) {
  auto ErrOrMemBufferRef = GetMemoryBufferForFile(Filename, Timestamp);
  if (auto Err = ErrOrMemBufferRef.getError())
    return Err;

  auto ErrOrObjectFile =
      object::ObjectFile::createObjectFile(*ErrOrMemBufferRef);
  if (auto Err = ErrOrObjectFile.getError())
    return Err;

  CurrentObjectFile = std::move(*ErrOrObjectFile);
  return *CurrentObjectFile;
}
}
}
