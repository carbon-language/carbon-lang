//===- llvm/System/Path.h ---------------------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Path class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_PATH_H
#define LLVM_SYSTEM_PATH_H

#include <string>

namespace llvm {
namespace sys {

  /// This class provides an abstraction for the name of a path
  /// to a file or directory in the filesystem and various basic operations
  /// on it.
  /// @since 1.4
  /// @brief An abstraction for operating system paths.
  class Path : public std::string {
  /// @name Constructors
  /// @{
  public:
    /// Creates a null (empty) path
    /// @brief Default Constructor
    Path () : std::string() {}

    /// Creates a path from char*
    /// @brief char* converter
    Path ( const char * name ) : std::string(name) {
      assert(is_valid());
    }

    /// @brief std::string converter
    Path ( const std::string& name ) : std::string(name){
      assert(is_valid());
    };

    /// Copies the path with copy-on-write semantics. The \p this Path
    /// will reference \p the that Path until one of them is modified
    /// at which point a full copy is taken before the write.
    /// @brief Copy Constructor
    Path ( const Path & that ) : std::string(that) {}

    /// Releases storage associated with the Path object
    /// @brief Destructor
    ~Path ( void ) {};

  /// @}
  /// @name Operators
  /// @{
  public:
    /// Makes a copy of \p that to \p this with copy-on-write semantics.
    /// @returns \p this
    /// @brief Assignment Operator
    Path & operator = ( const Path & that ) {
      this->assign (that);
      return *this;
    }

    /// Comparies \p this Path with \p that Path for equality.
    /// @returns true if \p this and \p that refer to the same item.
    /// @brief Equality Operator
    bool operator ==( const Path & that ) const {
      return 0 == this->compare( that ) ;
    }

    /// Comparies \p this Path with \p that Path for inequality.
    /// @returns true if \p this and \p that refer to different items.
    /// @brief Inequality Operator
    bool operator !=( const Path & that ) const {
      return 0 != this->compare( that );
    }

  /// @}
  /// @name Accessors
  /// @{
  public:
    /// @returns true if the path is valid
    /// @brief Determines if the path is valid (properly formed) or not.
    bool is_valid() const;

    /// @returns true if the path could reference a file
    /// @brief Determines if the path is valid for a file reference.
    bool is_file() const;

    /// @returns true if the path could reference a directory
    /// @brief Determines if the path is valid for a directory reference.
    bool is_directory() const;

    /// @brief Fills and zero terminates a buffer with the path
    void fill( char* buffer, unsigned len ) const;

  /// @}
  /// @name Mutators
  /// @{
  public:
      /// This ensures that the pathname is terminated with a /
      /// @brief Make the path reference a directory.
      void make_directory();

      /// This ensures that the pathname is not terminated with a /
      /// @brief Makes the path reference a file.
      void make_file();

      /// the file system.
      /// @returns true if the pathname references an existing file.
      /// @brief Determines if the path is a file or directory in
      bool exists();

      /// The \p dirname is added to the end of the Path.
      /// @param dirname A string providing the directory name to
      /// be appended to the path.
      /// @brief Appends the name of a directory.
      void append_directory( const std::string& dirname ) {
        this->append( dirname );
        make_directory();
      }

      /// The \p filename is added to the end of the Path.
      /// @brief Appends the name of a file.
      void append_file( const std::string& filename ) {
        this->append( filename );
      }

      /// Directories will have no entries. Files will be zero length. If
      /// the file or directory already exists, no error results.
      /// @throws SystemException if any error occurs.
      /// @brief Causes the file or directory to exist in the filesystem.
      void create( bool create_parents = false );

      void create_directory( void );
      void create_directories( void );
      void create_file( void );

      /// Directories must be empty before they can be removed. If not,
      /// an error will result. Files will be unlinked, even if another
      /// process is using them.
      /// @brief Removes the file or directory from the filesystem.
      void remove( void );
      void remove_directory( void );
      void remove_file( void ); 

      /// Find library.
      void find_lib( const char * file );
  /// @}
  };
}
}

// vim: sw=2

#endif
