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
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace dsymutil {

ErrorOr<MemoryBufferRef>
BinaryHolder::GetMemoryBufferForFile(StringRef Filename) {
  if (Verbose)
    outs() << "trying to open '" << Filename << "'\n";

  // Try that first as it doesn't involve any filesystem access.
  if (auto ErrOrArchiveMember = GetArchiveMemberBuffer(Filename))
    return *ErrOrArchiveMember;

  // If the name ends with a closing paren, there is a huge chance
  // it is an archive member specification.
  if (Filename.endswith(")"))
    if (auto ErrOrArchiveMember = MapArchiveAndGetMemberBuffer(Filename))
      return *ErrOrArchiveMember;

  // Otherwise, just try opening a standard file. If this is an
  // archive member specifiaction and any of the above didn't handle it
  // (either because the archive is not there anymore, or because the
  // archive doesn't contain the requested member), this will still
  // provide a sensible error message.
  auto ErrOrFile = MemoryBuffer::getFileOrSTDIN(Filename);
  if (auto Err = ErrOrFile.getError())
    return Err;

  if (Verbose)
    outs() << "\tloaded file.\n";
  CurrentArchive.reset();
  CurrentMemoryBuffer = std::move(ErrOrFile.get());
  return CurrentMemoryBuffer->getMemBufferRef();
}

ErrorOr<MemoryBufferRef>
BinaryHolder::GetArchiveMemberBuffer(StringRef Filename) {
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
        if (Verbose)
          outs() << "\tfound member in current archive.\n";
        return Child.getMemoryBufferRef();
      }
  }

  return make_error_code(errc::no_such_file_or_directory);
}

ErrorOr<MemoryBufferRef>
BinaryHolder::MapArchiveAndGetMemberBuffer(StringRef Filename) {
  StringRef ArchiveFilename = Filename.substr(0, Filename.find('('));

  auto ErrOrBuff = MemoryBuffer::getFileOrSTDIN(ArchiveFilename);
  if (auto Err = ErrOrBuff.getError())
    return Err;

  if (Verbose)
    outs() << "\topened new archive '" << ArchiveFilename << "'\n";
  auto ErrOrArchive = object::Archive::create((*ErrOrBuff)->getMemBufferRef());
  if (auto Err = ErrOrArchive.getError())
    return Err;

  CurrentArchive = std::move(*ErrOrArchive);
  CurrentMemoryBuffer = std::move(*ErrOrBuff);

  return GetArchiveMemberBuffer(Filename);
}

ErrorOr<const object::ObjectFile &>
BinaryHolder::GetObjectFile(StringRef Filename) {
  auto ErrOrMemBufferRef = GetMemoryBufferForFile(Filename);
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
