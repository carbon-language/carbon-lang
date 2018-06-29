//===-- BinaryHolder.h - Utility class for accessing binaries -------------===//
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
#ifndef LLVM_TOOLS_DSYMUTIL_BINARYHOLDER_H
#define LLVM_TOOLS_DSYMUTIL_BINARYHOLDER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"

#include <mutex>

namespace llvm {
namespace dsymutil {

/// The CachedBinaryHolder class is responsible for creating and owning
/// ObjectFiles and their underlying MemoryBuffers. It differs from a simple
/// OwningBinary in that it handles accessing and caching of archives and its
/// members.
class CachedBinaryHolder {
public:
  using TimestampTy = sys::TimePoint<std::chrono::seconds>;

  CachedBinaryHolder(bool Verbose = false) : Verbose(Verbose) {}

  // Forward declarations for friend declaration.
  class ObjectEntry;
  class ArchiveEntry;

  /// Base class shared by cached entries, representing objects and archives.
  class EntryBase {
  protected:
    std::unique_ptr<MemoryBuffer> MemoryBuffer;
    std::unique_ptr<object::MachOUniversalBinary> FatBinary;
    std::string FatBinaryName;
  };

  /// Cached entry holding one or more (in case of a fat binary) object files.
  class ObjectEntry : public EntryBase {
  public:
    /// Load the given object binary in memory.
    Error load(StringRef Filename, bool Verbose = false);

    /// Access all owned ObjectFiles.
    std::vector<const object::ObjectFile *> getObjects() const;

    /// Access to a derived version of all the currently owned ObjectFiles. The
    /// conversion might be invalid, in which case an Error is returned.
    template <typename ObjectFileType>
    Expected<std::vector<const ObjectFileType *>> getObjectsAs() const {
      std::vector<const object::ObjectFile *> Result;
      Result.reserve(Objects.size());
      for (auto &Object : Objects) {
        const auto *Derived = dyn_cast<ObjectFileType>(Object.get());
        if (!Derived)
          return errorCodeToError(object::object_error::invalid_file_type);
        Result.push_back(Derived);
      }
      return Result;
    }

    /// Access the owned ObjectFile with architecture \p T.
    Expected<const object::ObjectFile &> getObject(const Triple &T) const;

    /// Access to a derived version of the currently owned ObjectFile with
    /// architecture \p T. The conversion must be known to be valid.
    template <typename ObjectFileType>
    Expected<const ObjectFileType &> getObjectAs(const Triple &T) const {
      auto Object = getObject(T);
      if (!Object)
        return Object.takeError();
      return cast<ObjectFileType>(*Object);
    }

  private:
    std::vector<std::unique_ptr<object::ObjectFile>> Objects;
    friend ArchiveEntry;
  };

  /// Cached entry holding one or more (in the of a fat binary) archive files.
  class ArchiveEntry : public EntryBase {
  public:
    struct KeyTy {
      std::string Filename;
      TimestampTy Timestamp;

      KeyTy() : Filename(), Timestamp() {}
      KeyTy(StringRef Filename, TimestampTy Timestamp)
          : Filename(Filename.str()), Timestamp(Timestamp) {}
    };

    /// Load the given object binary in memory.
    Error load(StringRef Filename, TimestampTy Timestamp, bool Verbose = false);

    Expected<const ObjectEntry &> getObjectEntry(StringRef Filename,
                                                 TimestampTy Timestamp,
                                                 bool Verbose = false);

  private:
    std::vector<std::unique_ptr<object::Archive>> Archives;
    DenseMap<KeyTy, ObjectEntry> MemberCache;
    std::mutex MemberCacheMutex;
  };

  Expected<const ObjectEntry &> getObjectEntry(StringRef Filename,
                                               TimestampTy Timestamp);

  void clear();

private:
  /// Cache of static archives. Objects that are part of a static archive are
  /// stored under this object, rather than in the map below.
  StringMap<ArchiveEntry> ArchiveCache;
  std::mutex ArchiveCacheMutex;

  /// Object entries for objects that are not in a static archive.
  StringMap<ObjectEntry> ObjectCache;
  std::mutex ObjectCacheMutex;

  bool Verbose;
};

/// The BinaryHolder class is responsible for creating and owning ObjectFile
/// objects and their underlying MemoryBuffer. This is different from a simple
/// OwningBinary in that it handles accessing to archive members.
///
/// As an optimization, this class will reuse an already mapped and parsed
/// Archive object if 2 successive requests target the same archive file (Which
/// is always the case in debug maps).
/// Currently it only owns one memory buffer at any given time, meaning that a
/// mapping request will invalidate the previous memory mapping.
class BinaryHolder {
  std::vector<std::unique_ptr<object::Archive>> CurrentArchives;
  std::unique_ptr<MemoryBuffer> CurrentMemoryBuffer;
  std::vector<std::unique_ptr<object::ObjectFile>> CurrentObjectFiles;
  std::unique_ptr<object::MachOUniversalBinary> CurrentFatBinary;
  std::string CurrentFatBinaryName;
  bool Verbose;

  /// Get the MemoryBufferRefs for the file specification in \p
  /// Filename from the current archive. Multiple buffers are returned
  /// when there are multiple architectures available for the
  /// requested file.
  ///
  /// This function performs no system calls, it just looks up a
  /// potential match for the given \p Filename in the currently
  /// mapped archive if there is one.
  ErrorOr<std::vector<MemoryBufferRef>>
  GetArchiveMemberBuffers(StringRef Filename,
                          sys::TimePoint<std::chrono::seconds> Timestamp);

  /// Interpret Filename as an archive member specification map the
  /// corresponding archive to memory and return the MemoryBufferRefs
  /// corresponding to the described member. Multiple buffers are
  /// returned when there are multiple architectures available for the
  /// requested file.
  ErrorOr<std::vector<MemoryBufferRef>>
  MapArchiveAndGetMemberBuffers(StringRef Filename,
                                sys::TimePoint<std::chrono::seconds> Timestamp);

  /// Return the MemoryBufferRef that holds the memory mapping for the
  /// given \p Filename. This function will try to parse archive
  /// member specifications of the form /path/to/archive.a(member.o).
  ///
  /// The returned MemoryBufferRefs points to a buffer owned by this
  /// object. The buffer is valid until the next call to
  /// GetMemoryBufferForFile() on this object.
  /// Multiple buffers are returned when there are multiple
  /// architectures available for the requested file.
  ErrorOr<std::vector<MemoryBufferRef>>
  GetMemoryBuffersForFile(StringRef Filename,
                          sys::TimePoint<std::chrono::seconds> Timestamp);

  void changeBackingMemoryBuffer(std::unique_ptr<MemoryBuffer> &&MemBuf);
  ErrorOr<const object::ObjectFile &> getObjfileForArch(const Triple &T);

public:
  BinaryHolder(bool Verbose) : Verbose(Verbose) {}

  /// Get the ObjectFiles designated by the \p Filename. This
  /// might be an archive member specification of the form
  /// /path/to/archive.a(member.o).
  ///
  /// Calling this function invalidates the previous mapping owned by
  /// the BinaryHolder. Multiple buffers are returned when there are
  /// multiple architectures available for the requested file.
  ErrorOr<std::vector<const object::ObjectFile *>>
  GetObjectFiles(StringRef Filename,
                 sys::TimePoint<std::chrono::seconds> Timestamp =
                     sys::TimePoint<std::chrono::seconds>());

  /// Wraps GetObjectFiles() to return a derived ObjectFile type.
  template <typename ObjectFileType>
  ErrorOr<std::vector<const ObjectFileType *>>
  GetFilesAs(StringRef Filename,
             sys::TimePoint<std::chrono::seconds> Timestamp =
                 sys::TimePoint<std::chrono::seconds>()) {
    auto ErrOrObjFile = GetObjectFiles(Filename, Timestamp);
    if (auto Err = ErrOrObjFile.getError())
      return Err;

    std::vector<const ObjectFileType *> Objects;
    Objects.reserve((*ErrOrObjFile).size());
    for (const auto &Obj : *ErrOrObjFile) {
      const auto *Derived = dyn_cast<ObjectFileType>(Obj);
      if (!Derived)
        return make_error_code(object::object_error::invalid_file_type);
      Objects.push_back(Derived);
    }
    return std::move(Objects);
  }

  /// Access the currently owned ObjectFile with architecture \p T. As
  /// successful call to GetObjectFiles() or GetFilesAs() must have
  /// been performed before calling this.
  ErrorOr<const object::ObjectFile &> Get(const Triple &T) {
    return getObjfileForArch(T);
  }

  /// Get and cast to a subclass of the currently owned ObjectFile. The
  /// conversion must be known to be valid.
  template <typename ObjectFileType>
  ErrorOr<const ObjectFileType &> GetAs(const Triple &T) {
    auto ErrOrObj = Get(T);
    if (auto Err = ErrOrObj.getError())
      return Err;
    return cast<ObjectFileType>(*ErrOrObj);
  }
};
} // namespace dsymutil

template <>
struct DenseMapInfo<dsymutil::CachedBinaryHolder::ArchiveEntry::KeyTy> {

  static inline dsymutil::CachedBinaryHolder::ArchiveEntry::KeyTy
  getEmptyKey() {
    return dsymutil::CachedBinaryHolder::ArchiveEntry::KeyTy();
  }

  static inline dsymutil::CachedBinaryHolder::ArchiveEntry::KeyTy
  getTombstoneKey() {
    return dsymutil::CachedBinaryHolder::ArchiveEntry::KeyTy("/", {});
  }

  static unsigned
  getHashValue(const dsymutil::CachedBinaryHolder::ArchiveEntry::KeyTy &K) {
    return hash_combine(DenseMapInfo<StringRef>::getHashValue(K.Filename),
                        DenseMapInfo<unsigned>::getHashValue(
                            K.Timestamp.time_since_epoch().count()));
  }

  static bool
  isEqual(const dsymutil::CachedBinaryHolder::ArchiveEntry::KeyTy &LHS,
          const dsymutil::CachedBinaryHolder::ArchiveEntry::KeyTy &RHS) {
    return LHS.Filename == RHS.Filename && LHS.Timestamp == RHS.Timestamp;
  }
};

} // namespace llvm
#endif
