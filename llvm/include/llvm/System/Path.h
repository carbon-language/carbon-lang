//===- Path.h - Implement the Path class ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (C) 2003 eXtensible Systems, Inc. All Rights Reserved.
//
// This program is open source software; you can redistribute it and/or modify
// it under the terms of the University of Illinois Open Source License. See
// LICENSE.TXT (distributed with this software) for details.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Path class. The Path class provides the
// operating system agnostic concept of the name of a file or directory.
//
//===----------------------------------------------------------------------===//
/// @file lib/System/Path.h
/// @author Reid Spencer <raspencer@x10sys.com> (original author)
/// @version \verbatim $Id$ \endverbatim
/// @date 2004/08/14
/// @since 1.4
/// @brief Declares the llvm::sys::Path class.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_PATH_H
#define LLVM_SYSTEM_PATH_H

#include "llvm/System/ErrorCode.h"

namespace llvm {
namespace sys {

  /// This class provides an abstraction for the name of a path
  /// to a file or directory in the filesystem.
  /// @since 1.4
  /// @brief An abstraction for operating system paths.
  class Path : public std::string 
  {
  /// @name Types
  /// @{
  public:
    /// @brief An enumerator for specifying special ways to construct
    /// the path.
    enum ConstructSpecial {
      CONSTRUCT_TEMP_FILE = 1, ///< make a temporary file name
      CONSTRUCT_TEMP_DIR = 2,  ///< Make a temporary directory name
    };

  /// @}
  /// @name Constructors
  /// @{
  public:
    /// Forward declare this because constructors need to use it.
    /// @returns true if the path is valid
    /// @brief Determines if the path is valid (properly formed) or not.
    inline bool is_valid() const throw();

    /// Creates a null (empty) path
    /// @brief Default Constructor
    Path () throw() {}

    /// Creates a path from char*
    /// @brief char* converter
    Path ( const char * name ) throw() : std::string(name) {
      assert(is_valid());
    }

    /// @brief std::string converter
    Path ( const std::string& name ) throw() : std::string(name) {
      assert(is_valid());
    }

    /// Copies the path with copy-on-write semantics. The \p this Path
    /// will reference \p the that Path until one of them is modified
    /// at which point a full copy is taken before the write.
    /// @brief Copy Constructor
    Path ( const Path & that ) throw() : std::string(that) {
      assert(is_valid());
    }

    /// This constructor will construct the path using a special 
    /// construction argument that will pre-fill the path with the name
    /// of some special path name
    /// @brief Construct a special, well known path
    Path( ConstructSpecial ) throw();

    /// Releases storage associated with the Path object
    /// @brief Destructor
    ~Path () throw() {}

  /// @}
  /// @name Operators
  /// @{
  public:
    /// Makes a copy of \p that to \p this with copy-on-write semantics.
    /// @returns \p this
    /// @brief Assignment Operator
    Path & operator = ( const Path & that ) throw() {
      this->assign ( that );
      return *this;
    }

    /// Comparies \p this Path with \p that Path for equality.
    /// @returns true if \p this and \p that refer to the same item.
    /// @brief Equality Operator
    bool operator ==( const Path & that ) const throw() {
      return 0 == this->compare( that ) ;
    }

    /// Comparies \p this Path with \p that Path for inequality.
    /// @returns true if \p this and \p that refer to different items.
    /// @brief Inequality Operator
    bool operator !=( const Path & that ) const throw() {
      return 0 != this->compare( that );
    }

  /// @}
  /// @name Accessors
  /// @{
  public:
    /// @returns true if the path could reference a file
    /// @brief Determines if the path is valid for a file reference.
    bool is_file() const throw();

    /// @returns true if the path could reference a directory
    /// @brief Determines if the path is valid for a directory reference.
    bool is_directory() const throw();

    /// @brief Fills and zero terminates a buffer with the path
    inline void fill( char* buffer, uint32_t len ) const throw();

  /// @}
  /// @name Mutators
  /// @{
  public:
    /// This ensures that the pathname is terminated with a /
    /// @brief Make the path reference a directory.
    ErrorCode make_directory() throw();

    /// This ensures that the pathname is not terminated with a /
    /// @brief Makes the path reference a file.
    ErrorCode make_file() throw();

    /// the file system.
    /// @returns true if the pathname references an existing file.
    /// @brief Determines if the path is a file or directory in
    ErrorCode exists()  throw();

    /// The \p dirname is added to the end of the Path.
    /// @param dirname A string providing the directory name to
    /// be appended to the path.
    /// @brief Appends the name of a directory.
    ErrorCode append_directory( const std::string & dirname ) throw();

    /// The \p filename is added to the end of the Path.
    /// @brief Appends the name of a file.
    ErrorCode append_file( const std::string& filename ) throw();

    /// Directories will have no entries. Files will be zero length. If
    /// the file or directory already exists, no error results.
    /// @throws SystemException if any error occurs.
    /// @brief Causes the file or directory to exist in the filesystem.
    ErrorCode create( bool create_parents = false ) throw();

    ErrorCode create_directory() throw();
    ErrorCode create_directories() throw() ;
    ErrorCode create_file() throw() ;

    /// Directories must be empty before they can be removed. If not,
    /// an error will result. Files will be unlinked, even if another
    /// process is using them.
    /// @brief Removes the file or directory from the filesystem.
    ErrorCode remove() throw();
    ErrorCode remove_directory() throw() ;
    ErrorCode remove_file() throw() ; 

    /// Temporary file names are a common need. This method uses the
    /// template in the existing path and uniquifies a file name based
    /// on the template.
    ErrorCode make_temp_file() throw();

    /// Temporary directory names are a common need. This method uses
    /// the template in the existing path and uniqufies a directory
    /// name based on the template.
    ErrorCode make_temp_directory() throw();

    /// Find library.
    ErrorCode find_lib( const char * file ) throw();
  /// @}

  };

  inline bool Path::is_valid() const throw() {
    if ( empty() ) return false;
    return true;
  }

  inline void Path::fill( char* buffer, size_t bufflen ) const throw() {
    size_t pathlen = length();
    assert( bufflen > pathlen );
    size_t copylen = pathlen <? (bufflen - 1);
    this->copy(buffer, copylen, 0 );
    buffer[ copylen ] = 0;
  }
}
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab

#endif
