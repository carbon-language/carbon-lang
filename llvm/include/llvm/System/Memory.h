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

#ifndef LLVM_SYSTEM_PATH_H
#define LLVM_SYSTEM_PATH_H

#include <string>

namespace llvm {
namespace sys {

  /// This class provides an abstraction for various memory handling functions
  /// @since 1.4
  /// @brief An abstraction for operating system paths.
  class Memory {
    /// @name Functions
    /// @{
    public:
      Memory() { Address = 0; AllocSize = 0; }
      ~Memory() { ReleaseRWX(*this); }

      /// @throws std::string if an error occurred
      static void* AllocateRWX(Memory& block, unsigned NumBytes);

      /// @throws std::string if an error occurred
      static void ReleaseRWX(Memory& block);

      char* base() const { return reinterpret_cast<char*>(Address); }
      unsigned size() const { return AllocSize; }
    /// @}
    /// @name Data
    /// @{
    private:
      void * Address;        // Address of first byte of memory area
      unsigned AllocSize;    // Size, in bytes of the memory area
    /// @}
  };
}
}

// vim: sw=2

#endif
