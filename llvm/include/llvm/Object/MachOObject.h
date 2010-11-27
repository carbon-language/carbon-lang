//===- MachOObject.h - Mach-O Object File Wrapper ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHOOBJECT_H
#define LLVM_OBJECT_MACHOOBJECT_H

#include <string>
#include "llvm/ADT/OwningPtr.h"

namespace llvm {

class MemoryBuffer;

namespace object {

/// \brief Wrapper object for manipulating Mach-O object files.
///
/// This class is designed to implement a full-featured, efficient, portable,
/// and robust Mach-O interface to Mach-O object files. It does not attempt to
/// smooth over rough edges in the Mach-O format or generalize access to object
/// independent features.
///
/// The class is designed around accessing the Mach-O object which is expected
/// to be fully loaded into memory.
///
/// This class is *not* suitable for concurrent use. For efficient operation,
/// the class uses APIs which rely on the ability to cache the results of
/// certain calls in internal objects which are not safe for concurrent
/// access. This allows the API to be zero-copy on the common paths.
//
// FIXME: It would be cool if we supported a "paged" MemoryBuffer
// implementation. This would allow us to implement a more sensible version of
// MemoryObject which can work like a MemoryBuffer, but be more efficient for
// objects which are in the current address space.
class MachOObject {
public:

private:
  OwningPtr<MemoryBuffer> Buffer;

  /// Whether the object is little endian.
  bool IsLittleEndian;
  /// Whether the object is 64-bit.
  bool Is64Bit;

private:
  MachOObject(MemoryBuffer *Buffer, bool IsLittleEndian, bool Is64Bit);

public:
  /// \brief Load a Mach-O object from a MemoryBuffer object.
  ///
  /// \param Buffer - The buffer to load the object from. This routine takes
  /// exclusive ownership of the buffer (which is passed to the returned object
  /// on success).
  /// \param ErrorStr [out] - If given, will be set to a user readable error
  /// message on failure.
  /// \returns The loaded object, or null on error.
  static MachOObject *LoadFromBuffer(MemoryBuffer *Buffer,
                                     std::string *ErrorStr = 0);

  /// @name Object Header Information
  /// @{
  /// @}
};

} // end namespace object
} // end namespace llvm

#endif
