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
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace dsymutil {

static std::pair<StringRef, StringRef>
getArchiveAndObjectName(StringRef Filename) {
  StringRef Archive = Filename.substr(0, Filename.find('('));
  StringRef Object = Filename.substr(Archive.size() + 1).drop_back();
  return {Archive, Object};
}

static bool isArchive(StringRef Filename) { return Filename.endswith(")"); }

static std::vector<MemoryBufferRef>
getMachOFatMemoryBuffers(StringRef Filename, MemoryBuffer &Mem,
                         object::MachOUniversalBinary &Fat) {
  std::vector<MemoryBufferRef> Buffers;
  StringRef FatData = Fat.getData();
  for (auto It = Fat.begin_objects(), End = Fat.end_objects(); It != End;
       ++It) {
    StringRef ObjData = FatData.substr(It->getOffset(), It->getSize());
    Buffers.emplace_back(ObjData, Filename);
  }
  return Buffers;
}

Error CachedBinaryHolder::ArchiveEntry::load(StringRef Filename,
                                             TimestampTy Timestamp,
                                             bool Verbose) {
  StringRef ArchiveFilename = getArchiveAndObjectName(Filename).first;

  // Try to load archive and force it to be memory mapped.
  auto ErrOrBuff = MemoryBuffer::getFileOrSTDIN(ArchiveFilename, -1, false);
  if (auto Err = ErrOrBuff.getError())
    return errorCodeToError(Err);

  MemoryBuffer = std::move(*ErrOrBuff);

  if (Verbose)
    WithColor::note() << "opened archive '" << ArchiveFilename << "'\n";

  // Load one or more archive buffers, depending on whether we're dealing with
  // a fat binary.
  std::vector<MemoryBufferRef> ArchiveBuffers;

  auto ErrOrFat =
      object::MachOUniversalBinary::create(MemoryBuffer->getMemBufferRef());
  if (!ErrOrFat) {
    consumeError(ErrOrFat.takeError());
    ArchiveBuffers.push_back(MemoryBuffer->getMemBufferRef());
  } else {
    FatBinary = std::move(*ErrOrFat);
    FatBinaryName = ArchiveFilename;
    ArchiveBuffers =
        getMachOFatMemoryBuffers(FatBinaryName, *MemoryBuffer, *FatBinary);
  }

  // Finally, try to load the archives.
  Archives.reserve(ArchiveBuffers.size());
  for (auto MemRef : ArchiveBuffers) {
    auto ErrOrArchive = object::Archive::create(MemRef);
    if (!ErrOrArchive)
      return ErrOrArchive.takeError();
    Archives.push_back(std::move(*ErrOrArchive));
  }

  return Error::success();
}

Error CachedBinaryHolder::ObjectEntry::load(StringRef Filename, bool Verbose) {
  // Try to load regular binary and force it to be memory mapped.
  auto ErrOrBuff = MemoryBuffer::getFileOrSTDIN(Filename, -1, false);
  if (auto Err = ErrOrBuff.getError())
    return errorCodeToError(Err);

  MemoryBuffer = std::move(*ErrOrBuff);

  if (Verbose)
    WithColor::note() << "opened object.\n";

  // Load one or more object buffers, depending on whether we're dealing with a
  // fat binary.
  std::vector<MemoryBufferRef> ObjectBuffers;

  auto ErrOrFat =
      object::MachOUniversalBinary::create(MemoryBuffer->getMemBufferRef());
  if (!ErrOrFat) {
    consumeError(ErrOrFat.takeError());
    ObjectBuffers.push_back(MemoryBuffer->getMemBufferRef());
  } else {
    FatBinary = std::move(*ErrOrFat);
    FatBinaryName = Filename;
    ObjectBuffers =
        getMachOFatMemoryBuffers(FatBinaryName, *MemoryBuffer, *FatBinary);
  }

  Objects.reserve(ObjectBuffers.size());
  for (auto MemRef : ObjectBuffers) {
    auto ErrOrObjectFile = object::ObjectFile::createObjectFile(MemRef);
    if (!ErrOrObjectFile)
      return ErrOrObjectFile.takeError();
    Objects.push_back(std::move(*ErrOrObjectFile));
  }

  return Error::success();
}

std::vector<const object::ObjectFile *>
CachedBinaryHolder::ObjectEntry::getObjects() const {
  std::vector<const object::ObjectFile *> Result;
  Result.reserve(Objects.size());
  for (auto &Object : Objects) {
    Result.push_back(Object.get());
  }
  return Result;
}
Expected<const object::ObjectFile &>
CachedBinaryHolder::ObjectEntry::getObject(const Triple &T) const {
  for (const auto &Obj : Objects) {
    if (const auto *MachO = dyn_cast<object::MachOObjectFile>(Obj.get())) {
      if (MachO->getArchTriple().str() == T.str())
        return *MachO;
    } else if (Obj->getArch() == T.getArch())
      return *Obj;
  }
  return errorCodeToError(object::object_error::arch_not_found);
}

Expected<const CachedBinaryHolder::ObjectEntry &>
CachedBinaryHolder::ArchiveEntry::getObjectEntry(StringRef Filename,
                                                 TimestampTy Timestamp,
                                                 bool Verbose) {
  StringRef ArchiveFilename;
  StringRef ObjectFilename;
  std::tie(ArchiveFilename, ObjectFilename) = getArchiveAndObjectName(Filename);

  // Try the cache first.
  KeyTy Key = {ObjectFilename, Timestamp};

  {
    std::lock_guard<std::mutex> Lock(MemberCacheMutex);
    if (MemberCache.count(Key))
      return MemberCache[Key];
  }

  // Create a new ObjectEntry, but don't add it to the cache yet. Loading of
  // the archive members might fail and we don't want to lock the whole archive
  // during this operation.
  ObjectEntry OE;

  for (const auto &Archive : Archives) {
    Error Err = Error::success();
    for (auto Child : Archive->children(Err)) {
      if (auto NameOrErr = Child.getName()) {
        if (*NameOrErr == ObjectFilename) {
          auto ModTimeOrErr = Child.getLastModified();
          if (!ModTimeOrErr)
            return ModTimeOrErr.takeError();

          if (Timestamp != sys::TimePoint<>() &&
              Timestamp != ModTimeOrErr.get()) {
            if (Verbose)
              WithColor::warning() << "member has timestamp mismatch.\n";
            continue;
          }

          if (Verbose)
            WithColor::note() << "found member in current archive.\n";

          auto ErrOrMem = Child.getMemoryBufferRef();
          if (!ErrOrMem)
            return ErrOrMem.takeError();

          auto ErrOrObjectFile =
              object::ObjectFile::createObjectFile(*ErrOrMem);
          if (!ErrOrObjectFile)
            return ErrOrObjectFile.takeError();

          OE.Objects.push_back(std::move(*ErrOrObjectFile));
        }
      }
    }
    if (Err)
      return std::move(Err);
  }

  if (OE.Objects.empty())
    return errorCodeToError(errc::no_such_file_or_directory);

  std::lock_guard<std::mutex> Lock(MemberCacheMutex);
  MemberCache.try_emplace(Key, std::move(OE));
  return MemberCache[Key];
}

Expected<const CachedBinaryHolder::ObjectEntry &>
CachedBinaryHolder::getObjectEntry(StringRef Filename, TimestampTy Timestamp) {
  if (Verbose)
    WithColor::note() << "trying to open '" << Filename << "'\n";

  // If this is an archive, we might have either the object or the archive
  // cached. In this case we can load it without accessing the file system.
  if (isArchive(Filename)) {
    StringRef ArchiveFilename = getArchiveAndObjectName(Filename).first;
    std::lock_guard<std::mutex> Lock(ArchiveCacheMutex);
    if (ArchiveCache.count(ArchiveFilename)) {
      return ArchiveCache[ArchiveFilename].getObjectEntry(Filename, Timestamp);
    } else {
      ArchiveEntry &AE = ArchiveCache[ArchiveFilename];
      auto Err = AE.load(Filename, Timestamp, Verbose);
      if (Err) {
        ArchiveCache.erase(ArchiveFilename);
        // Don't return the error here: maybe the file wasn't an archive.
        llvm::consumeError(std::move(Err));
      } else {
        return ArchiveCache[ArchiveFilename].getObjectEntry(Filename,
                                                            Timestamp);
      }
    }
  }

  // If this is an object, we might have it cached. If not we'll have to load
  // it from the file system and cache it now.
  std::lock_guard<std::mutex> Lock(ObjectCacheMutex);
  if (!ObjectCache.count(Filename)) {
    ObjectEntry &OE = ObjectCache[Filename];
    auto Err = OE.load(Filename);
    if (Err) {
      ObjectCache.erase(Filename);
      return std::move(Err);
    }
  }

  return ObjectCache[Filename];
}

void CachedBinaryHolder::clear() {
  std::lock_guard<std::mutex> ArchiveLock(ArchiveCacheMutex);
  std::lock_guard<std::mutex> ObjectLock(ObjectCacheMutex);
  ArchiveCache.clear();
  ObjectCache.clear();
}

void BinaryHolder::changeBackingMemoryBuffer(
    std::unique_ptr<MemoryBuffer> &&Buf) {
  CurrentArchives.clear();
  CurrentObjectFiles.clear();
  CurrentFatBinary.reset();

  CurrentMemoryBuffer = std::move(Buf);
}

ErrorOr<std::vector<MemoryBufferRef>> BinaryHolder::GetMemoryBuffersForFile(
    StringRef Filename, sys::TimePoint<std::chrono::seconds> Timestamp) {
  if (Verbose)
    outs() << "trying to open '" << Filename << "'\n";

  // Try that first as it doesn't involve any filesystem access.
  if (auto ErrOrArchiveMembers = GetArchiveMemberBuffers(Filename, Timestamp))
    return *ErrOrArchiveMembers;

  // If the name ends with a closing paren, there is a huge chance
  // it is an archive member specification.
  if (Filename.endswith(")"))
    if (auto ErrOrArchiveMembers =
            MapArchiveAndGetMemberBuffers(Filename, Timestamp))
      return *ErrOrArchiveMembers;

  // Otherwise, just try opening a standard file. If this is an
  // archive member specifiaction and any of the above didn't handle it
  // (either because the archive is not there anymore, or because the
  // archive doesn't contain the requested member), this will still
  // provide a sensible error message.
  auto ErrOrFile = MemoryBuffer::getFileOrSTDIN(Filename, -1, false);
  if (auto Err = ErrOrFile.getError())
    return Err;

  changeBackingMemoryBuffer(std::move(*ErrOrFile));
  if (Verbose)
    outs() << "\tloaded file.\n";

  auto ErrOrFat = object::MachOUniversalBinary::create(
      CurrentMemoryBuffer->getMemBufferRef());
  if (!ErrOrFat) {
    consumeError(ErrOrFat.takeError());
    // Not a fat binary must be a standard one. Return a one element vector.
    return std::vector<MemoryBufferRef>{CurrentMemoryBuffer->getMemBufferRef()};
  }

  CurrentFatBinary = std::move(*ErrOrFat);
  CurrentFatBinaryName = Filename;
  return getMachOFatMemoryBuffers(CurrentFatBinaryName, *CurrentMemoryBuffer,
                                  *CurrentFatBinary);
}

ErrorOr<std::vector<MemoryBufferRef>> BinaryHolder::GetArchiveMemberBuffers(
    StringRef Filename, sys::TimePoint<std::chrono::seconds> Timestamp) {
  if (CurrentArchives.empty())
    return make_error_code(errc::no_such_file_or_directory);

  StringRef CurArchiveName = CurrentArchives.front()->getFileName();
  if (!Filename.startswith(Twine(CurArchiveName, "(").str()))
    return make_error_code(errc::no_such_file_or_directory);

  // Remove the archive name and the parens around the archive member name.
  Filename = Filename.substr(CurArchiveName.size() + 1).drop_back();

  std::vector<MemoryBufferRef> Buffers;
  Buffers.reserve(CurrentArchives.size());

  for (const auto &CurrentArchive : CurrentArchives) {
    Error Err = Error::success();
    for (auto Child : CurrentArchive->children(Err)) {
      if (auto NameOrErr = Child.getName()) {
        if (*NameOrErr == Filename) {
          auto ModTimeOrErr = Child.getLastModified();
          if (!ModTimeOrErr)
            return errorToErrorCode(ModTimeOrErr.takeError());
          if (Timestamp != sys::TimePoint<>() &&
              Timestamp != ModTimeOrErr.get()) {
            if (Verbose)
              outs() << "\tmember had timestamp mismatch.\n";
            continue;
          }
          if (Verbose)
            outs() << "\tfound member in current archive.\n";
          auto ErrOrMem = Child.getMemoryBufferRef();
          if (!ErrOrMem)
            return errorToErrorCode(ErrOrMem.takeError());
          Buffers.push_back(*ErrOrMem);
        }
      }
    }
    if (Err)
      return errorToErrorCode(std::move(Err));
  }

  if (Buffers.empty())
    return make_error_code(errc::no_such_file_or_directory);
  return Buffers;
}

ErrorOr<std::vector<MemoryBufferRef>>
BinaryHolder::MapArchiveAndGetMemberBuffers(
    StringRef Filename, sys::TimePoint<std::chrono::seconds> Timestamp) {
  StringRef ArchiveFilename = Filename.substr(0, Filename.find('('));

  auto ErrOrBuff = MemoryBuffer::getFileOrSTDIN(ArchiveFilename, -1, false);
  if (auto Err = ErrOrBuff.getError())
    return Err;

  if (Verbose)
    outs() << "\topened new archive '" << ArchiveFilename << "'\n";

  changeBackingMemoryBuffer(std::move(*ErrOrBuff));
  std::vector<MemoryBufferRef> ArchiveBuffers;
  auto ErrOrFat = object::MachOUniversalBinary::create(
      CurrentMemoryBuffer->getMemBufferRef());
  if (!ErrOrFat) {
    consumeError(ErrOrFat.takeError());
    // Not a fat binary must be a standard one.
    ArchiveBuffers.push_back(CurrentMemoryBuffer->getMemBufferRef());
  } else {
    CurrentFatBinary = std::move(*ErrOrFat);
    CurrentFatBinaryName = ArchiveFilename;
    ArchiveBuffers = getMachOFatMemoryBuffers(
        CurrentFatBinaryName, *CurrentMemoryBuffer, *CurrentFatBinary);
  }

  for (auto MemRef : ArchiveBuffers) {
    auto ErrOrArchive = object::Archive::create(MemRef);
    if (!ErrOrArchive)
      return errorToErrorCode(ErrOrArchive.takeError());
    CurrentArchives.push_back(std::move(*ErrOrArchive));
  }
  return GetArchiveMemberBuffers(Filename, Timestamp);
}

ErrorOr<const object::ObjectFile &>
BinaryHolder::getObjfileForArch(const Triple &T) {
  for (const auto &Obj : CurrentObjectFiles) {
    if (const auto *MachO = dyn_cast<object::MachOObjectFile>(Obj.get())) {
      if (MachO->getArchTriple().str() == T.str())
        return *MachO;
    } else if (Obj->getArch() == T.getArch())
      return *Obj;
  }

  return make_error_code(object::object_error::arch_not_found);
}

ErrorOr<std::vector<const object::ObjectFile *>>
BinaryHolder::GetObjectFiles(StringRef Filename,
                             sys::TimePoint<std::chrono::seconds> Timestamp) {
  auto ErrOrMemBufferRefs = GetMemoryBuffersForFile(Filename, Timestamp);
  if (auto Err = ErrOrMemBufferRefs.getError())
    return Err;

  std::vector<const object::ObjectFile *> Objects;
  Objects.reserve(ErrOrMemBufferRefs->size());

  CurrentObjectFiles.clear();
  for (auto MemBuf : *ErrOrMemBufferRefs) {
    auto ErrOrObjectFile = object::ObjectFile::createObjectFile(MemBuf);
    if (!ErrOrObjectFile)
      return errorToErrorCode(ErrOrObjectFile.takeError());

    Objects.push_back(ErrOrObjectFile->get());
    CurrentObjectFiles.push_back(std::move(*ErrOrObjectFile));
  }

  return std::move(Objects);
}
} // namespace dsymutil
} // namespace llvm
