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
#include "llvm/ADT/InMemoryStruct.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Object/MachOFormat.h"

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
  struct LoadCommandInfo {
    /// The load command information.
    macho::LoadCommand Command;

    /// The offset to the start of the load command in memory.
    uint64_t Offset;
  };

private:
  OwningPtr<MemoryBuffer> Buffer;

  /// Whether the object is little endian.
  bool IsLittleEndian;
  /// Whether the object is 64-bit.
  bool Is64Bit;
  /// Whether the object is swapped endianness from the host.
  bool IsSwappedEndian;

  /// The cached information on the load commands.
  LoadCommandInfo *LoadCommands;
  mutable unsigned NumLoadedCommands;

  /// The cached copy of the header.
  macho::Header Header;
  macho::Header64Ext Header64Ext;

private:
  MachOObject(MemoryBuffer *Buffer, bool IsLittleEndian, bool Is64Bit);

public:
  ~MachOObject();

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

  /// @name File Information
  /// @{

  bool isLittleEndian() const { return IsLittleEndian; }
  bool isSwappedEndian() const { return IsSwappedEndian; }
  bool is64Bit() const { return Is64Bit; }

  unsigned getHeaderSize() const {
    return Is64Bit ? macho::Header64Size : macho::Header32Size;
  }

  /// @}
  /// @name Object Header Access
  /// @{

  const macho::Header &getHeader() const { return Header; }
  const macho::Header64Ext &getHeader64Ext() const {
    assert(is64Bit() && "Invalid access!");
    return Header64Ext;
  }

  /// @}
  /// @name Object Structure Access
  /// @{

  /// \brief Retrieve the information for the given load command.
  const LoadCommandInfo &getLoadCommandInfo(unsigned Index) const;

  void ReadSegmentLoadCommand(
    const LoadCommandInfo &LCI,
    InMemoryStruct<macho::SegmentLoadCommand> &Res) const;
  void ReadSegment64LoadCommand(
    const LoadCommandInfo &LCI,
    InMemoryStruct<macho::Segment64LoadCommand> &Res) const;

  /// @}
};

} // end namespace object
} // end namespace llvm

#endif
