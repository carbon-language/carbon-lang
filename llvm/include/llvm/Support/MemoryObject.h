//===- MemoryObject.h - Abstract memory interface ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MEMORYOBJECT_H
#define MEMORYOBJECT_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

/// MemoryObject - Abstract base class for contiguous addressable memory.
///   Necessary for cases in which the memory is in another process, in a
///   file, or on a remote machine.
class MemoryObject {
public:
  /// Destructor      - Override as necessary.
  ~MemoryObject() {
  }
  
  /// getBase         - Returns the lowest valid address in the region.
  ///
  /// @result         - The lowest valid address.
  virtual uintptr_t getBase() const = 0;
  
  /// getExtent       - Returns the size of the region in bytes.  (The region is
  ///                   contiguous, so the highest valid address of the region 
  ///                   is getBase() + getExtent() - 1).
  ///
  /// @result         - The size of the region.
  virtual uintptr_t getExtent() const = 0;
  
  /// readByte        - Tries to read a single byte from the region.
  ///
  /// @param address  - The address of the byte, in the same space as getBase().
  /// @param ptr      - A pointer to a byte to be filled in.  Must be non-NULL.
  /// @result         - 0 if successful; -1 if not.  Failure may be due to a
  ///                   bounds violation or an implementation-specific error.
  virtual int readByte(uintptr_t address, uint8_t* ptr) const = 0;
  
  /// readByte        - Tries to read a contiguous range of bytes from the
  ///                   region, up to the end of the region.
  ///                   You should override this function if there is a quicker
  ///                   way than going back and forth with individual bytes.
  ///
  /// @param address  - The address of the first byte, in the same space as 
  ///                   getBase().
  /// @param size     - The maximum number of bytes to copy.
  /// @param buf      - A pointer to a buffer to be filled in.  Must be non-NULL
  ///                   and large enough to hold size bytes.
  /// @result         - The number of bytes copied if successful; (uintptr_t)-1
  ///                   if not.
  ///                   Failure may be due to a bounds violation or an
  ///                   implementation-specific error.
  virtual uintptr_t readBytes(uintptr_t address,
                              uintptr_t size,
                              uint8_t* buf) const {
    uintptr_t current = address;
    uintptr_t limit = getBase() + getExtent();
    
    while(current - address < size && current < limit) {
      if(readByte(current, &buf[(current - address)]))
        return (uintptr_t)-1;
      
      current++;
    }
    
    return current - address;
  }
};

}

#endif

