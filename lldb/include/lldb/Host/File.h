//===-- File.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_File_h_
#define liblldb_File_h_
#if defined(__cplusplus)

#include "lldb/lldb-private.h"
#include "lldb/Host/FileSpec.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class File File.h "lldb/Host/File.h"
/// @brief A file class.
///
/// A file class that divides abstracts the LLDB core from host file
/// functionality.
//----------------------------------------------------------------------
class File
{
public:
    
    enum OpenOptions
    {
        eOpenOptionRead                 = (1u << 0),    // Open file for reading
        eOpenOptionWrite                = (1u << 1),    // Open file for writing
        eOpenOptionAppend               = (1u << 2),    // Don't truncate file when opening, append to end of file
        eOpenOptionNonBlocking          = (1u << 3),    // File reads
        eOpenOptionCanCreate            = (1u << 4),    // Create file if doesn't already exist
        eOpenOptionCanCreateNewOnly     = (1u << 5),    // Can create file only if it doesn't already exist
        eOpenOptionTruncate             = (1u << 6),    // Truncate file when opening existing
        eOpenOptionSharedLock           = (1u << 7),    // Open file and get shared lock
        eOpenOptionExclusiveLock        = (1u << 8)     // Open file and get exclusive lock
    };
    
    enum Permissions
    {
        ePermissionsUserRead        = (1u << 0),
        ePermissionsUserWrite       = (1u << 1),
        ePermissionsUserExecute     = (1u << 2),
        ePermissionsGroupRead       = (1u << 3),
        ePermissionsGroupWrite      = (1u << 4),
        ePermissionsGroupExecute    = (1u << 5),
        ePermissionsWorldRead       = (1u << 6),
        ePermissionsWorldWrite      = (1u << 7),
        ePermissionsWorldExecute    = (1u << 8)
    };

    File() : m_file_desc (-1)
    {
    }

    //------------------------------------------------------------------
    /// Constructor with path.
    ///
    /// Takes a path to a file which can be just a filename, or a full
    /// path. If \a path is not NULL or empty, this function will call
    /// FileSpec::SetFile (const char *path, bool resolve).
    ///
    /// @param[in] path
    ///     The full or partial path to a file.
    ///
    /// @param[in] options
    ///     Options to use when opening (see OpenOptions)
    ///
    /// @param[in] permissions
    ///     Options to use when opening (see OpenOptions)
    ///
    /// @see FileSpec::SetFile (const char *path, bool resolve)
    //------------------------------------------------------------------
    File (const char *path,
          uint32_t options,
          uint32_t permissions);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual in case this class is subclassed.
    //------------------------------------------------------------------
    virtual
    ~File ();

    bool
    IsValid () const
    {
        return m_file_desc >= 0;
    }

    //------------------------------------------------------------------
    /// Convert to pointer operator.
    ///
    /// This allows code to check a File object to see if it
    /// contains anything valid using code such as:
    ///
    /// @code
    /// File file(...);
    /// if (file)
    /// { ...
    /// @endcode
    ///
    /// @return
    ///     A pointer to this object if either the directory or filename
    ///     is valid, NULL otherwise.
    //------------------------------------------------------------------
    operator
    bool () const
    {
        return m_file_desc >= 0;
    }

    //------------------------------------------------------------------
    /// Logical NOT operator.
    ///
    /// This allows code to check a File object to see if it is
    /// invalid using code such as:
    ///
    /// @code
    /// File file(...);
    /// if (!file)
    /// { ...
    /// @endcode
    ///
    /// @return
    ///     Returns \b true if the object has an empty directory and
    ///     filename, \b false otherwise.
    //------------------------------------------------------------------
    bool
    operator! () const
    {
        return m_file_desc < 0;
    }

    //------------------------------------------------------------------
    /// Get the file spec for this file.
    ///
    /// @return
    ///     A reference to the file specification object.
    //------------------------------------------------------------------
    Error
    GetFileSpec (FileSpec &file_spec) const;
    
    Error
    Open (const char *path,
          uint32_t options,
          uint32_t permissions);

    Error
    Close ();
    
    Error
    Read (void *dst, size_t &num_bytes);
          
    Error
    Write (const void *src, size_t &num_bytes);

    Error
    SeekFromStart (off_t& offset);
    
    Error
    SeekFromCurrent (off_t& offset);
    
    Error
    SeekFromEnd (off_t& offset);

    Error
    Sync ();

protected:
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    int m_file_desc; ///< The open file handle or NULL if the file isn't opened
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_FileSpec_h_
