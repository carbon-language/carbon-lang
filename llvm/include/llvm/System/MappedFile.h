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
  /// for mapping a file into memory for both read and write access. This class
  /// does not provide facilities for finding the file or operating on paths to
  /// files. The sys::Path class is used for that.
  class MappedFile {
    sys::PathWithStatus Path;        ///< Path to the file.
    unsigned Options;                ///< Options used to create the mapping
    void *BasePtr;                   ///< Pointer to the base memory address
    mutable MappedFileInfo *MapInfo; ///< Platform specific info for the mapping
    
    MappedFile& operator=(const MappedFile &that); // DO NOT IMPLEMENT
    MappedFile(const MappedFile &that); // DO NOT IMPLEMENT
  public:
    enum MappingOptions {
      READ_ACCESS    = 0x0001,  ///< Map the file for reading
      WRITE_ACCESS   = 0x0002,  ///< Map the file for write access
      EXEC_ACCESS    = 0x0004,  ///< Map the file for execution access
      SHARED_MAPPING = 0x0008   ///< Map the file shared with other processes
    };
    
    MappedFile() : Options(READ_ACCESS), BasePtr(0), MapInfo(0) {}

    /// Destruct a MappedFile and release all memory associated with it.
    ~MappedFile() { close(); }

  public:  // Accessors

    /// This function determines if the file is currently mapped or not.
    bool isMapped() const { return BasePtr != 0; }

    /// This function returns a void* pointer to the base address of the file
    /// mapping. This is the memory address of the first byte in the file.
    /// Note that although a non-const pointer is returned, the memory might
    /// not actually be writable, depending on the MappingOptions used when
    /// the MappedFile was opened.
    void* base() const { return BasePtr; }

    /// This function returns a char* pointer to the base address of the file
    /// mapping. This is the memory address of the first byte in the file.
    /// Note that although a non-const pointer is returned, the memory might
    /// not actually be writable, depending on the MappingOptions used when
    /// the MappedFile was opened.
    char* charBase() const { return reinterpret_cast<char*>(BasePtr); }

    /// This function returns a reference to the sys::Path object kept by the
    /// MappedFile object. This contains the path to the file that is or
    /// will be mapped.
    const sys::Path& path() const { return Path; }

    /// This function returns the number of bytes in the file.
    size_t size() const;

  public:  // Mutators
    
    /// Open a file to be mapped and get its size but don't map it yet.  Return
    /// true on error.
    bool open(const sys::Path &P, int options = READ_ACCESS,
              std::string *ErrMsg = 0) {
      Path = P;
      Options = options;
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

    /// resize - This method causes the size of the file, and consequently the
    /// size of the mapping to be set. This is logically the same as unmap(),
    /// adjust size of the file, map(). Consequently, when calling this
    /// function, the caller should not rely on previous results of the
    /// map(), base(), or baseChar() members as they may point to invalid
    /// areas of memory after this call.
    bool resize(size_t new_size, std::string *ErrMsg = 0);

    void close() { if (MapInfo) terminate(); }

  private:
    bool initialize(std::string *ErrMsg); 
    void terminate();
  };
} // end namespace sys
} // end namespace llvm

#endif
