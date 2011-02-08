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
    /// File::Open (const char *path, uint32_t options, uint32_t permissions).
    ///
    /// @param[in] path
    ///     The full or partial path to a file.
    ///
    /// @param[in] options
    ///     Options to use when opening (see File::OpenOptions)
    ///
    /// @param[in] permissions
    ///     Options to use when opening (see File::Permissions)
    ///
    /// @see File::Open (const char *path, uint32_t options, uint32_t permissions)
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
    
    //------------------------------------------------------------------
    /// Open a file for read/writing with the specified options.
    ///
    /// Takes a path to a file which can be just a filename, or a full
    /// path.
    ///
    /// @param[in] path
    ///     The full or partial path to a file.
    ///
    /// @param[in] options
    ///     Options to use when opening (see File::OpenOptions)
    ///
    /// @param[in] permissions
    ///     Options to use when opening (see File::Permissions)
    //------------------------------------------------------------------
    Error
    Open (const char *path,
          uint32_t options,
          uint32_t permissions);

    Error
    Close ();
    
    //------------------------------------------------------------------
    /// Read bytes from a file from the current file position.
    ///
    /// NOTE: This function is NOT thread safe. Use the read function
    /// that takes an "off_t &offset" to ensure correct operation in
    /// multi-threaded environments.
    ///
    /// @param[in] buf
    ///     A buffer where to put the bytes that are read.
    ///
    /// @param[in/out] num_bytes
    ///     The number of bytes to read form the current file position
    ///     which gets modified with the number of bytes that were read.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Read (void *buf, size_t &num_bytes);
          
    //------------------------------------------------------------------
    /// Write bytes to a file at the current file position.
    ///
    /// NOTE: This function is NOT thread safe. Use the write function
    /// that takes an "off_t &offset" to ensure correct operation in
    /// multi-threaded environments.
    ///
    /// @param[in] buf
    ///     A buffer where to put the bytes that are read.
    ///
    /// @param[in/out] num_bytes
    ///     The number of bytes to write to the current file position
    ///     which gets modified with the number of bytes that were 
    ///     written.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Write (const void *buf, size_t &num_bytes);

    //------------------------------------------------------------------
    /// Seek to an offset relative to the beginning of the file.
    ///
    /// NOTE: This function is NOT thread safe, other threads that 
    /// access this object might also change the current file position.
    /// For thread safe reads and writes see the following functions:
    /// @see File::Read (void *, size_t, off_t &)
    /// @see File::Write (const void *, size_t, off_t &)
    ///
    /// @param[in/out] offset
    ///     The offset to seek to within the file relative to the 
    ///     beginning of the file which gets filled in the the resulting
    ///     absolute file offset.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    SeekFromStart (off_t& offset);
    
    //------------------------------------------------------------------
    /// Seek to an offset relative to the current file position.
    ///
    /// NOTE: This function is NOT thread safe, other threads that 
    /// access this object might also change the current file position.
    /// For thread safe reads and writes see the following functions:
    /// @see File::Read (void *, size_t, off_t &)
    /// @see File::Write (const void *, size_t, off_t &)
    ///
    /// @param[in/out] offset
    ///     The offset to seek to within the file relative to the 
    ///     current file position. On return this parameter gets filled 
    ///     in the the resulting absolute file offset.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    SeekFromCurrent (off_t& offset);
    
    //------------------------------------------------------------------
    /// Seek to an offset relative to the end of the file.
    ///
    /// NOTE: This function is NOT thread safe, other threads that 
    /// access this object might also change the current file position.
    /// For thread safe reads and writes see the following functions:
    /// @see File::Read (void *, size_t, off_t &)
    /// @see File::Write (const void *, size_t, off_t &)
    ///
    /// @param[in/out] offset
    ///     The offset to seek to within the file relative to the 
    ///     end of the file which gets filled in the the resulting
    ///     absolute file offset.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    SeekFromEnd (off_t& offset);

    //------------------------------------------------------------------
    /// Read bytes from a file from the specified file offset.
    ///
    /// NOTE: This function is thread safe in that clients manager their
    /// own file position markers and reads on other threads won't mess
    /// up the current read.
    ///
    /// @param[in] buf
    ///     A buffer where to put the bytes that are read.
    ///
    /// @param[in/out] num_bytes
    ///     The number of bytes to read form the current file position
    ///     which gets modified with the number of bytes that were read.
    ///
    /// @param[in/out] offset
    ///     The offset within the file from which to read \a num_bytes
    ///     bytes. This offset gets incremented by the number of bytes
    ///     that were read.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Read (void *dst, size_t &num_bytes, off_t &offset);
    
    //------------------------------------------------------------------
    /// Write bytes to a file at the specified file offset.
    ///
    /// NOTE: This function is thread safe in that clients manager their
    /// own file position markers, though clients will need to implement
    /// their own locking externally to avoid multiple people writing
    /// to the file at the same time.
    ///
    /// @param[in] buf
    ///     A buffer containing the bytes to write.
    ///
    /// @param[in/out] num_bytes
    ///     The number of bytes to write to the file at offset \a offset.
    ///     \a num_bytes gets modified with the number of bytes that 
    ///     were read.
    ///
    /// @param[in/out] offset
    ///     The offset within the file at which to write \a num_bytes
    ///     bytes. This offset gets incremented by the number of bytes
    ///     that were written.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Write (const void *src, size_t &num_bytes, off_t &offset);

    
    //------------------------------------------------------------------
    /// Sync to disk.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
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
#endif  // liblldb_File_h_
