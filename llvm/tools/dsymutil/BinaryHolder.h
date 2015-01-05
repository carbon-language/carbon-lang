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

#include "llvm/Object/Archive.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"

namespace llvm {
namespace dsymutil {

/// \brief The BinaryHolder class is responsible for creating and
/// owning ObjectFile objects and their underlying MemoryBuffer. This
/// is different from a simple OwningBinary in that it handles
/// accessing to archive members.
///
/// As an optimization, this class will reuse an already mapped and
/// parsed Archive object if 2 successive requests target the same
/// archive file (Which is always the case in debug maps).
/// Currently it only owns one memory buffer at any given time,
/// meaning that a mapping request will invalidate the previous memory
/// mapping.
class BinaryHolder {
  std::unique_ptr<object::Archive> CurrentArchive;
  std::unique_ptr<MemoryBuffer> CurrentMemoryBuffer;
  std::unique_ptr<object::ObjectFile> CurrentObjectFile;
  bool Verbose;

  /// \brief Get the MemoryBufferRef for the file specification in \p
  /// Filename from the current archive.
  ///
  /// This function performs no system calls, it just looks up a
  /// potential match for the given \p Filename in the currently
  /// mapped archive if there is one.
  ErrorOr<MemoryBufferRef> GetArchiveMemberBuffer(StringRef Filename);

  /// \brief Interpret Filename as an archive member specification,
  /// map the corresponding archive to memory and return the
  /// MemoryBufferRef corresponding to the described member.
  ErrorOr<MemoryBufferRef> MapArchiveAndGetMemberBuffer(StringRef Filename);

  /// \brief Return the MemoryBufferRef that holds the memory
  /// mapping for the given \p Filename. This function will try to
  /// parse archive member specifications of the form
  /// /path/to/archive.a(member.o).
  ///
  /// The returned MemoryBufferRef points to a buffer owned by this
  /// object. The buffer is valid until the next call to
  /// GetMemoryBufferForFile() on this object.
  ErrorOr<MemoryBufferRef> GetMemoryBufferForFile(StringRef Filename);

public:
  BinaryHolder(bool Verbose) : Verbose(Verbose) {}

  /// \brief Get the ObjectFile designated by the \p Filename. This
  /// might be an archive member specification of the form
  /// /path/to/archive.a(member.o).
  ///
  /// Calling this function invalidates the previous mapping owned by
  /// the BinaryHolder.
  ErrorOr<const object::ObjectFile &> GetObjectFile(StringRef Filename);

  /// \brief Wraps GetObjectFile() to return a derived ObjectFile type.
  template <typename ObjectFileType>
  ErrorOr<const ObjectFileType &> GetFileAs(StringRef Filename) {
    auto ErrOrObjFile = GetObjectFile(Filename);
    if (auto Err = ErrOrObjFile.getError())
      return Err;
    if (const auto *Derived = dyn_cast<ObjectFileType>(CurrentObjectFile.get()))
      return *Derived;
    return make_error_code(object::object_error::invalid_file_type);
  }

  /// \brief Access the currently owned ObjectFile. As successfull
  /// call to GetObjectFile() or GetFileAs() must have been performed
  /// before calling this.
  const object::ObjectFile &Get() {
    assert(CurrentObjectFile);
    return *CurrentObjectFile;
  }

  /// \brief Access to a derived version of the currently owned
  /// ObjectFile. The conversion must be known to be valid.
  template <typename ObjectFileType> const ObjectFileType &GetAs() {
    return cast<ObjectFileType>(*CurrentObjectFile);
  }
};
}
}
#endif
