//===-- FileSystem.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSystem.h"

// C includes
#include <sys/mount.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef __linux__
#include <sys/statfs.h>
#include <sys/mount.h>
#include <linux/magic.h>
#endif

// lldb Includes
#include "lldb/Core/Error.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Host.h"

using namespace lldb;
using namespace lldb_private;

FileSpec::PathSyntax
FileSystem::GetNativePathSyntax()
{
    return FileSpec::ePathSyntaxPosix;
}

Error
FileSystem::MakeDirectory(const char *path, uint32_t file_permissions)
{
    Error error;
    if (path && path[0])
    {
        if (::mkdir(path, file_permissions) != 0)
        {
            error.SetErrorToErrno();
            switch (error.GetError())
            {
                case ENOENT:
                {
                    // Parent directory doesn't exist, so lets make it if we can
                    FileSpec spec(path, false);
                    if (spec.GetDirectory() && spec.GetFilename())
                    {
                        // Make the parent directory and try again
                        Error error2 = MakeDirectory(spec.GetDirectory().GetCString(), file_permissions);
                        if (error2.Success())
                        {
                            // Try and make the directory again now that the parent directory was made successfully
                            if (::mkdir(path, file_permissions) == 0)
                                error.Clear();
                            else
                                error.SetErrorToErrno();
                        }
                    }
                }
                break;

                case EEXIST:
                {
                    FileSpec path_spec(path, false);
                    if (path_spec.IsDirectory())
                        error.Clear(); // It is a directory and it already exists
                }
                break;
            }
        }
    }
    else
    {
        error.SetErrorString("empty path");
    }
    return error;
}

Error
FileSystem::DeleteDirectory(const char *path, bool recurse)
{
    Error error;
    if (path && path[0])
    {
        if (recurse)
        {
            StreamString command;
            command.Printf("rm -rf \"%s\"", path);
            int status = ::system(command.GetString().c_str());
            if (status != 0)
                error.SetError(status, eErrorTypeGeneric);
        }
        else
        {
            if (::rmdir(path) != 0)
                error.SetErrorToErrno();
        }
    }
    else
    {
        error.SetErrorString("empty path");
    }
    return error;
}

Error
FileSystem::GetFilePermissions(const char *path, uint32_t &file_permissions)
{
    Error error;
    struct stat file_stats;
    if (::stat(path, &file_stats) == 0)
    {
        // The bits in "st_mode" currently match the definitions
        // for the file mode bits in unix.
        file_permissions = file_stats.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
    }
    else
    {
        error.SetErrorToErrno();
    }
    return error;
}

Error
FileSystem::SetFilePermissions(const char *path, uint32_t file_permissions)
{
    Error error;
    if (::chmod(path, file_permissions) != 0)
        error.SetErrorToErrno();
    return error;
}

lldb::user_id_t
FileSystem::GetFileSize(const FileSpec &file_spec)
{
    return file_spec.GetByteSize();
}

bool
FileSystem::GetFileExists(const FileSpec &file_spec)
{
    return file_spec.Exists();
}

Error
FileSystem::Hardlink(const char *src, const char *dst)
{
    Error error;
    if (::link(dst, src) == -1)
        error.SetErrorToErrno();
    return error;
}

Error
FileSystem::Symlink(const char *src, const char *dst)
{
    Error error;
    if (::symlink(dst, src) == -1)
        error.SetErrorToErrno();
    return error;
}

Error
FileSystem::Unlink(const char *path)
{
    Error error;
    if (::unlink(path) == -1)
        error.SetErrorToErrno();
    return error;
}

Error
FileSystem::Readlink(const char *path, char *buf, size_t buf_len)
{
    Error error;
    ssize_t count = ::readlink(path, buf, buf_len);
    if (count < 0)
        error.SetErrorToErrno();
    else if (static_cast<size_t>(count) < (buf_len - 1))
        buf[count] = '\0'; // Success
    else
        error.SetErrorString("'buf' buffer is too small to contain link contents");
    return error;
}

static bool IsLocal(const struct statfs& info)
{
#ifdef __linux__
    #define CIFS_MAGIC_NUMBER 0xFF534D42
    switch ((uint32_t)info.f_type)
    {
    case NFS_SUPER_MAGIC:
    case SMB_SUPER_MAGIC:
    case CIFS_MAGIC_NUMBER:
        return false;
    default:
        return true;
    }
#else
    return (info.f_flags & MNT_LOCAL) != 0;
#endif
}

bool
FileSystem::IsLocal(const FileSpec &spec)
{
    struct statfs statfs_info;
    std::string path (spec.GetPath());
    if (statfs(path.c_str(), &statfs_info) == 0)
        return ::IsLocal(statfs_info);
    return false;
}
