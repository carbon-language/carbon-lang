//===- Binary.h - A generic binary file -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Binary class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_BINARY_H
#define LLVM_OBJECT_BINARY_H

#include "llvm/Object/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"

namespace llvm {

class MemoryBuffer;
class StringRef;

namespace object {

class Binary {
private:
  Binary() LLVM_DELETED_FUNCTION;
  Binary(const Binary &other) LLVM_DELETED_FUNCTION;

  unsigned int TypeID;

protected:
  MemoryBuffer *Data;

  Binary(unsigned int Type, MemoryBuffer *Source);

  enum {
    ID_Archive,
    ID_MachOUniversalBinary,
    // Object and children.
    ID_StartObjects,
    ID_COFF,

    ID_ELF32L, // ELF 32-bit, little endian
    ID_ELF32B, // ELF 32-bit, big endian
    ID_ELF64L, // ELF 64-bit, little endian
    ID_ELF64B, // ELF 64-bit, big endian

    ID_MachO32L, // MachO 32-bit, little endian
    ID_MachO32B, // MachO 32-bit, big endian
    ID_MachO64L, // MachO 64-bit, little endian
    ID_MachO64B, // MachO 64-bit, big endian

    ID_EndObjects
  };

  static inline unsigned int getELFType(bool isLE, bool is64Bits) {
    if (isLE)
      return is64Bits ? ID_ELF64L : ID_ELF32L;
    else
      return is64Bits ? ID_ELF64B : ID_ELF32B;
  }

  static unsigned int getMachOType(bool isLE, bool is64Bits) {
    if (isLE)
      return is64Bits ? ID_MachO64L : ID_MachO32L;
    else
      return is64Bits ? ID_MachO64B : ID_MachO32B;
  }

public:
  virtual ~Binary();

  StringRef getData() const;
  StringRef getFileName() const;

  // Cast methods.
  unsigned int getType() const { return TypeID; }

  // Convenience methods
  bool isObject() const {
    return TypeID > ID_StartObjects && TypeID < ID_EndObjects;
  }

  bool isArchive() const {
    return TypeID == ID_Archive;
  }

  bool isMachOUniversalBinary() const {
    return TypeID == ID_MachOUniversalBinary;
  }

  bool isELF() const {
    return TypeID >= ID_ELF32L && TypeID <= ID_ELF64B;
  }

  bool isMachO() const {
    return TypeID >= ID_MachO32L && TypeID <= ID_MachO64B;
  }

  bool isCOFF() const {
    return TypeID == ID_COFF;
  }

  bool isLittleEndian() const {
    return !(TypeID == ID_ELF32B || TypeID == ID_ELF64B ||
             TypeID == ID_MachO32B || TypeID == ID_MachO64B);
  }
};

/// @brief Create a Binary from Source, autodetecting the file type.
///
/// @param Source The data to create the Binary from. Ownership is transferred
///        to the Binary if successful. If an error is returned,
///        Source is destroyed by createBinary before returning.
ErrorOr<Binary *> createBinary(MemoryBuffer *Source,
                               sys::fs::file_magic Type =
                                   sys::fs::file_magic::unknown);

ErrorOr<Binary *> createBinary(StringRef Path);
}
}

#endif
