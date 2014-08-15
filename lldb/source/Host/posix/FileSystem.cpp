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
#include <sys/stat.h>
#include <sys/types.h>

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

bool
FileSystem::CalculateMD5(const FileSpec &file_spec, uint64_t &low, uint64_t &high)
{
#if defined(__APPLE__)
    StreamString md5_cmd_line;
    md5_cmd_line.Printf("md5 -q '%s'", file_spec.GetPath().c_str());
    std::string hash_string;
    Error err = Host::RunShellCommand(md5_cmd_line.GetData(), NULL, NULL, NULL, &hash_string, 60);
    if (err.Fail())
        return false;
    // a correctly formed MD5 is 16-bytes, that is 32 hex digits
    // if the output is any other length it is probably wrong
    if (hash_string.size() != 32)
        return false;
    std::string part1(hash_string, 0, 16);
    std::string part2(hash_string, 16);
    const char *part1_cstr = part1.c_str();
    const char *part2_cstr = part2.c_str();
    high = ::strtoull(part1_cstr, NULL, 16);
    low = ::strtoull(part2_cstr, NULL, 16);
    return true;
#else
    // your own MD5 implementation here
    return false;
#endif
}
