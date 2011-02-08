//===-- FileSpec.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/Host/File.h"

#include <fcntl.h>

#include "lldb/Core/Error.h"

using namespace lldb;
using namespace lldb_private;

File::File(const char *path, uint32_t options, uint32_t permissions) :
    m_file_desc (-1)
{
    Open (path, options, permissions);
}

File::~File()
{
    Close ();
}

Error
File::Open (const char *path, uint32_t options, uint32_t permissions)
{
    Error error;
    if (IsValid())
        Close ();

    int oflag = 0;
    if (options & eOpenOptionRead && 
        options & eOpenOptionWrite )
        oflag |= O_RDWR;
    else if (options & eOpenOptionRead)
        oflag |= O_RDONLY;
    else if (options & eOpenOptionWrite)
        oflag |= O_WRONLY;
    
    if (options & eOpenOptionNonBlocking)
        oflag |= O_NONBLOCK;

    if (options & eOpenOptionAppend)
        oflag |= O_APPEND;

    if (options & eOpenOptionCanCreate)
        oflag |= O_CREAT;
    
    if (options & eOpenOptionCanCreateNewOnly)
        oflag |= O_CREAT | O_EXCL;
    
    if (options & eOpenOptionTruncate)
        oflag |= O_TRUNC;
    
    if (options & eOpenOptionSharedLock)
        oflag |= O_SHLOCK;

    if (options & eOpenOptionExclusiveLock)
        oflag |= O_EXLOCK;

    mode_t mode = 0;
    if (permissions & ePermissionsUserRead)     mode |= S_IRUSR;
    if (permissions & ePermissionsUserWrite)    mode |= S_IWUSR;
    if (permissions & ePermissionsUserExecute)  mode |= S_IXUSR;
    if (permissions & ePermissionsGroupRead)    mode |= S_IRGRP;
    if (permissions & ePermissionsGroupWrite)   mode |= S_IWGRP;
    if (permissions & ePermissionsGroupExecute) mode |= S_IXGRP;
    if (permissions & ePermissionsWorldRead)    mode |= S_IROTH;
    if (permissions & ePermissionsWorldWrite)   mode |= S_IWOTH;
    if (permissions & ePermissionsWorldExecute) mode |= S_IXOTH;

    m_file_desc = ::open(path, oflag, mode);
    if (m_file_desc == -1)
        error.SetErrorToErrno();
    
    return error;
}

Error
File::Close ()
{
    Error error;
    if (IsValid ())
    {
        if (::close (m_file_desc) != 0)
            error.SetErrorToErrno();
        m_file_desc = -1;
    }
    return error;
}


Error
File::GetFileSpec (FileSpec &file_spec) const
{
    Error error;
    if (IsValid ())
    {
        char path[PATH_MAX];
        if (::fcntl(m_file_desc, F_GETPATH, path) == -1)
            error.SetErrorToErrno();
        else
            file_spec.SetFile (path, false);
    }
    else 
        error.SetErrorString("invalid file handle");

    if (error.Fail())
        file_spec.Clear();
    return error;
}

Error
File::SeekFromStart (off_t& offset)
{
    Error error;
    if (IsValid ())
    {
        offset = ::lseek (m_file_desc, offset, SEEK_SET);

        if (offset == -1)
            error.SetErrorToErrno();
    }
    else 
    {
        error.SetErrorString("invalid file handle");
    }
    return error;
}

Error
File::SeekFromCurrent (off_t& offset)
{
    Error error;
    if (IsValid ())
    {
        offset = ::lseek (m_file_desc, offset, SEEK_CUR);
        
        if (offset == -1)
            error.SetErrorToErrno();
    }
    else 
    {
        error.SetErrorString("invalid file handle");
    }
    return error;
}

Error
File::SeekFromEnd (off_t& offset)
{
    Error error;
    if (IsValid ())
    {
        offset = ::lseek (m_file_desc, offset, SEEK_CUR);
        
        if (offset == -1)
            error.SetErrorToErrno();
    }
    else 
    {
        error.SetErrorString("invalid file handle");
    }
    return error;
}

Error
File::Sync ()
{
    Error error;
    if (IsValid ())
    {
        if (::fsync (m_file_desc) == -1)
            error.SetErrorToErrno();
    }
    else 
    {
        error.SetErrorString("invalid file handle");
    }
    return error;
}
Error
File::Read (void *buf, size_t &num_bytes)
{
    Error error;
    if (IsValid ())
    {
        ssize_t bytes_read = ::read (m_file_desc, buf, num_bytes);
        if (bytes_read == -1)
        {
            error.SetErrorToErrno();
            num_bytes = 0;
        }
        else
            num_bytes = bytes_read;
    }
    else 
    {
        num_bytes = 0;
        error.SetErrorString("invalid file handle");
    }
    return error;
}
          
Error
File::Write (const void *buf, size_t &num_bytes)
{
    Error error;
    if (IsValid())
    {
        ssize_t bytes_written = ::write (m_file_desc, buf, num_bytes);
        if (bytes_written == -1)
        {
            error.SetErrorToErrno();
            num_bytes = 0;
        }
        else
            num_bytes = bytes_written;
    }
    else 
    {
        num_bytes = 0;
        error.SetErrorString("invalid file handle");
    }
    return error;
}

