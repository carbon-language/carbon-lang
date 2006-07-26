//===- llvm/System/Memory.h - Memory Support --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Memory class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_MEMORY_H
#define LLVM_SYSTEM_MEMORY_H

#include <string>
#include "llvm/System/IncludeFile.h"

namespace llvm {
namespace sys {

  /// This class encapsulates the notion of a memory block which has an address
  /// and a size. It is used by the Memory class (a friend) as the result of
  /// various memory allocation operations.
  /// @see Memory
  /// @brief Memory block abstraction.
  class MemoryBlock {
  public:
    void *base() const { return Address; }
    unsigned size() const { return Size; }
  private:
    void *Address;    ///< Address of first byte of memory area
    unsigned Size;    ///< Size, in bytes of the memory area
    friend class Memory;
  };

  /// This class provides various memory handling functions that manipulate
  /// MemoryBlock instances.
  /// @since 1.4
  /// @brief An abstraction for memory operations.
  class Memory {
    /// @name Functions
    /// @{
    public:
      /// This method allocates a block of Read/Write/Execute memory that is
      /// suitable for executing dynamically generated code (e.g. JIT). An
      /// attempt to allocate \p NumBytes bytes of virtual memory is made.
      /// \p NearBlock may point to an existing allocation in which case
      /// an attempt is made to allocate more memory near the existing block.
      ///
      /// On success, this returns a non-null memory block, otherwise it returns
      /// a null memory block and fills in *ErrMsg.
      /// 
      /// @brief Allocate Read/Write/Execute memory.
      static MemoryBlock AllocateRWX(unsigned NumBytes,
                                     const MemoryBlock *NearBlock,
                                     std::string *ErrMsg = 0);

      /// This method releases a block of Read/Write/Execute memory that was
      /// allocated with the AllocateRWX method. It should not be used to
      /// release any memory block allocated any other way.
      ///
      /// On success, this returns false, otherwise it returns true and fills
      /// in *ErrMsg.
      /// @throws std::string if an error occurred.
      /// @brief Release Read/Write/Execute memory.
      static bool ReleaseRWX(MemoryBlock &block, std::string *ErrMsg = 0);
    /// @}
  };
}
}

FORCE_DEFINING_FILE_TO_BE_LINKED(SystemMemory)

#endif
