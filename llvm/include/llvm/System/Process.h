//===- llvm/System/Process.h ------------------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Process class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_PROCESS_H
#define LLVM_SYSTEM_PROCESS_H

namespace llvm {
namespace sys {

  /// This class provides an abstraction for getting information about the
  /// currently executing process. 
  /// @since 1.4
  /// @brief An abstraction for operating system processes.
  class Process {
    /// @name Accessors
    /// @{
    public:
      /// This static function will return the operating system's virtual memory
      /// page size.
      /// @returns The number of bytes in a virtual memory page.
      /// @throws nothing
      /// @brief Get the virtual memory page size
      static unsigned GetPageSize();

    /// @}
  };
}
}

// vim: sw=2

#endif
