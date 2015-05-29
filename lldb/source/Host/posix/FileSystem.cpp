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
FileSystem::MakeDirectory(const FileSpec &file_spec, uint32_t file_permissions)
{
    if (file_spec)
    {
        Error error;
        if (::mkdir(file_spec.GetCString(), file_permissions) == -1)
        {
            error.SetErrorToErrno();
            errno = 0;
            switch (error.GetError())
            {
                case ENOENT:
                {
                    // Parent directory doesn't exist, so lets make it if we can
                    // Make the parent directory and try again
                    FileSpec parent_file_spec{file_spec.GetDirectory().GetCString(), false};
                    error = MakeDirectory(parent_file_spec, file_permissions);
                    if (error.Fail())
                        return error;
                    // Try and make the directory again now that the parent directory was made successfully
                    if (::mkdir(file_spec.GetCString(), file_permissions) == -1)
                    {
                        error.SetErrorToErrno();
                        return error;
                    }
                }
                case EEXIST:
                {
                    if (file_spec.IsDirectory())
                        return Error{}; // It is a directory and it already exists
                }
            }
        }
        return error;
    }
    return Error{"empty path"};
}

Error
FileSystem::DeleteDirectory(const FileSpec &file_spec, bool recurse)
{
    Error error;
    if (file_spec)
    {
        if (recurse)
        {
            StreamString command;
            command.Printf("rm -rf \"%s\"", file_spec.GetCString());
            int status = ::system(command.GetString().c_str());
            if (status != 0)
                error.SetError(status, eErrorTypeGeneric);
        }
        else
        {
            if (::rmdir(file_spec.GetCString()) != 0)
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
FileSystem::GetFilePermissions(const FileSpec &file_spec, uint32_t &file_permissions)
{
    Error error;
    struct stat file_stats;
    if (::stat(file_spec.GetCString(), &file_stats) == 0)
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
FileSystem::SetFilePermissions(const FileSpec &file_spec, uint32_t file_permissions)
{
    Error error;
    if (::chmod(file_spec.GetCString(), file_permissions) != 0)
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
FileSystem::Hardlink(const FileSpec &src, const FileSpec &dst)
{
    Error error;
    if (::link(dst.GetCString(), src.GetCString()) == -1)
        error.SetErrorToErrno();
    return error;
}

Error
FileSystem::Symlink(const FileSpec &src, const FileSpec &dst)
{
    Error error;
    if (::symlink(dst.GetCString(), src.GetCString()) == -1)
        error.SetErrorToErrno();
    return error;
}

Error
FileSystem::Unlink(const FileSpec &file_spec)
{
    Error error;
    if (::unlink(file_spec.GetCString()) == -1)
        error.SetErrorToErrno();
    return error;
}

Error
FileSystem::Readlink(const FileSpec &src, FileSpec &dst)
{
    Error error;
    char buf[PATH_MAX];
    ssize_t count = ::readlink(src.GetCString(), buf, sizeof(buf) - 1);
    if (count < 0)
        error.SetErrorToErrno();
    else
    {
        buf[count] = '\0'; // Success
        dst.SetFile(buf, false);
    }
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
