//===- llvm/System/MappedFile.h - MappedFile OS Concept ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::MappedFile class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_MAPPEDFILE_H
#define LLVM_SYSTEM_MAPPEDFILE_H

#include "llvm/System/Path.h"

namespace llvm {
namespace sys {

  /// Forward declare a class used for holding platform specific information
  /// that needs to be
  struct MappedFileInfo;

  /// This class provides an abstraction for a memory mapped file in the
  /// operating system's filesystem. It provides platform independent operations
  /// for mapping a file into memory for read access.
  class MappedFile {
    sys::PathWithStatus Path;        ///< Path to the file.
    void *BasePtr;                   ///< Pointer to the base memory address
    mutable MappedFileInfo *MapInfo; ///< Platform specific info for the mapping
    
    MappedFile& operator=(const MappedFile &that); // DO NOT IMPLEMENT
    MappedFile(const MappedFile &that); // DO NOT IMPLEMENT
  public:
    MappedFile() : BasePtr(0), MapInfo(0) {}

    /// Destruct a MappedFile and release all memory associated with it.
    ~MappedFile() { close(); }

  public:  // Accessors

    /// This function determines if the file is currently mapped or not.
    bool isMapped() const { return BasePtr != 0; }

    /// getBase - Returns a const void* pointer to the base address of the file
    /// mapping. This is the memory address of the first byte in the file.
    const void *getBase() const { return BasePtr; }

    /// This function returns a reference to the sys::Path object kept by the
    /// MappedFile object. This contains the path to the file that is or
    /// will be mapped.
    const sys::PathWithStatus &path() const { return Path; }

    /// This function returns the number of bytes in the file.
    size_t size() const;

  public:  // Mutators
    
    /// Open a file to be mapped and get its size but don't map it yet.  Return
    /// true on error.
    bool open(const sys::Path &P, std::string *ErrMsg = 0) {
      Path = P;
      return initialize(ErrMsg);
    }

    /// unmap - Remove the mapped file from memory. If the file was mapped for
    /// write access, the memory contents will be automatically synchronized
    /// with the file's disk contents.
    void unmap();

    /// map - Reserve space for the file, map it into memory, and return a
    /// pointer to it.  This returns the base memory address of the mapped file
    /// or 0 if an error occurred.
    void *map(std::string* ErrMsg = 0);

    void close() { if (MapInfo) terminate(); }

  private:
    bool initialize(std::string *ErrMsg); 
    void terminate();
  };
} // end namespace sys
} // end namespace llvm

#endif
