//===- llvm/System/MappedFile.h - MappedFile OS Concept ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::MappedFile class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_MAPPEDFILE_H
#define LLVM_SYSTEM_MAPPEDFILE_H

#include "llvm/System/Path.h"
#include "llvm/System/IncludeFile.h"

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
  /// @since 1.4
  /// @brief An abstraction for memory mapped files.
  class MappedFile {
  /// @name Types
  /// @{
  public:
    enum MappingOptions {
      READ_ACCESS = 0x0001,     ///< Map the file for reading
      WRITE_ACCESS = 0x0002,    ///< Map the file for write access
      EXEC_ACCESS = 0x0004,     ///< Map the file for execution access
      SHARED_MAPPING = 0x0008   ///< Map the file shared with other processes
    };
  /// @}
  /// @name Constructors
  /// @{
  public:
    /// Construct a MappedFile to the \p path in the operating system's file
    /// system with the mapping \p options provided.
    /// @throws std::string if an error occurs
    MappedFile() : path_(), options_(READ_ACCESS), base_(0), info_(0) {}

    /// Destruct a MappedFile and release all memory associated with it.
    /// @throws std::string if an error occurs
    ~MappedFile() { if (info_) terminate(); }

  /// @}
  /// @name Accessors
  /// @{
  public:
    /// This function determines if the file is currently mapped or not.
    /// @returns true iff the file is mapped into memory, false otherwise
    /// @brief Determine if a MappedFile is currently mapped
    /// @throws nothing
    bool isMapped() const { return base_ != 0; }

    /// This function returns a void* pointer to the base address of the file
    /// mapping. This is the memory address of the first byte in the file.
    /// Note that although a non-const pointer is returned, the memory might
    /// not actually be writable, depending on the MappingOptions used when
    /// the MappedFile was opened.
    /// @returns The base pointer to the memory mapped file.
    /// @brief Obtain the base pointer to the memory mapped file.
    /// @throws nothing
    void* base() const { return base_; }

    /// This function returns a char* pointer to the base address of the file
    /// mapping. This is the memory address of the first byte in the file.
    /// Note that although a non-const pointer is returned, the memory might
    /// not actually be writable, depending on the MappingOptions used when
    /// the MappedFile was opened.
    /// @returns The base pointer to the memory mapped file as a char pointer.
    /// @brief Obtain the base pointer to the memory mapped file.
    /// @throws nothing
    char* charBase() const { return reinterpret_cast<char*>(base_); }

    /// This function returns a reference to the sys::Path object kept by the
    /// MappedFile object. This contains the path to the file that is or
    /// will be mapped.
    /// @returns sys::Path containing the path name.
    /// @brief Returns the mapped file's path as a sys::Path
    /// @throws nothing
    const sys::Path& path() const { return path_; }

    /// This function returns the number of bytes in the file.
    /// @throws std::string if an error occurs
    size_t size() const;

  /// @}
  /// @name Mutators
  /// @{
  public:
    /// Open a file to be mapped and get its size but don't map it yet.
    /// @returns true if an error occurred
    bool open(
      const sys::Path& p, ///< Path to file to be mapped
      int options = READ_ACCESS, ///< Access mode for the mapping
      std::string* ErrMsg = 0 ///< Optional error string pointer
    ) {
      path_ = p;
      options_ = options;
      return initialize(ErrMsg);
    }

    /// The mapped file is removed from memory. If the file was mapped for
    /// write access, the memory contents will be automatically synchronized
    /// with the file's disk contents.
    /// @brief Remove the file mapping from memory.
    void unmap();

    /// The mapped file is put into memory.
    /// @returns The base memory address of the mapped file or 0 if an error
    /// occurred.
    /// @brief Map the file into memory.
    void* map(
      std::string* ErrMsg ///< Optional error string pointer
    );

    /// This method causes the size of the file, and consequently the size
    /// of the mapping to be set. This is logically the same as unmap(),
    /// adjust size of the file, map(). Consequently, when calling this
    /// function, the caller should not rely on previous results of the
    /// map(), base(), or baseChar() members as they may point to invalid
    /// areas of memory after this call.
    /// @throws std::string if an error occurs
    /// @brief Set the size of the file and memory mapping.
    void size(size_t new_size);

    void close() { if (info_) terminate(); }

  /// @}
  /// @name Implementation
  /// @{
  private:
    /// @brief Initialize platform-specific portion
    bool initialize(std::string* ErrMsg); 

    /// @brief Terminate platform-specific portion
    void terminate();  

  /// @}
  /// @name Data
  /// @{
  private:
    sys::Path path_;       ///< Path to the file.
    int options_;          ///< Options used to create the mapping
    void* base_;           ///< Pointer to the base memory address
    mutable MappedFileInfo* info_; ///< Platform specific info for the mapping

  /// @}
  /// @name Disabled
  /// @{
  private:
    ///< Disallow assignment
    MappedFile& operator = ( const MappedFile & that );
    ///< Disallow copying
    MappedFile(const MappedFile& that);
  /// @}
  };
}
}

FORCE_DEFINING_FILE_TO_BE_LINKED(SystemMappedFile)

#endif
