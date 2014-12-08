//===-- FileSystem.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/windows.h"

#include <shellapi.h>

#include "lldb/Host/FileSystem.h"

using namespace lldb_private;

FileSpec::PathSyntax
FileSystem::GetNativePathSyntax()
{
    return FileSpec::ePathSyntaxWindows;
}

Error
FileSystem::MakeDirectory(const char *path, uint32_t file_permissions)
{
    // On Win32, the mode parameter is ignored, as Windows files and directories support a
    // different permission model than POSIX.
    Error error;
    if (!::CreateDirectory(path, NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
}

Error
FileSystem::DeleteDirectory(const char *path, bool recurse)
{
    Error error;
    if (!recurse)
    {
        BOOL result = ::RemoveDirectory(path);
        if (!result)
            error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    }
    else
    {
        // SHFileOperation() accepts a list of paths, and so must be double-null-terminated to
        // indicate the end of the list.
        std::string path_buffer(path);
        path_buffer.push_back(0);

        SHFILEOPSTRUCT shfos = {0};
        shfos.wFunc = FO_DELETE;
        shfos.pFrom = path_buffer.c_str();
        shfos.fFlags = FOF_NO_UI;

        int result = ::SHFileOperation(&shfos);
        // TODO(zturner): Correctly handle the intricacies of SHFileOperation return values.
        if (result != 0)
            error.SetErrorStringWithFormat("SHFileOperation failed");
    }
    return error;
}

Error
FileSystem::GetFilePermissions(const char *path, uint32_t &file_permissions)
{
    Error error;
    error.SetErrorStringWithFormat("%s is not supported on this host", __PRETTY_FUNCTION__);
    return error;
}

Error
FileSystem::SetFilePermissions(const char *path, uint32_t file_permissions)
{
    Error error;
    error.SetErrorStringWithFormat("%s is not supported on this host", __PRETTY_FUNCTION__);
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
FileSystem::Symlink(const char *linkname, const char *target)
{
    Error error;
    DWORD attrib = ::GetFileAttributes(target);
    if (attrib == INVALID_FILE_ATTRIBUTES)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }
    bool is_directory = !!(attrib & FILE_ATTRIBUTE_DIRECTORY);
    DWORD flag = is_directory ? SYMBOLIC_LINK_FLAG_DIRECTORY : 0;
    BOOL result = ::CreateSymbolicLink(linkname, target, flag);
    if (!result)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
}

Error
FileSystem::Unlink(const char *path)
{
    Error error;
    BOOL result = ::DeleteFile(path);
    if (!result)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
}

Error
FileSystem::Readlink(const char *path, char *buf, size_t buf_len)
{
    Error error;
    HANDLE h = ::CreateFile(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                            FILE_FLAG_OPEN_REPARSE_POINT, NULL);
    if (h == INVALID_HANDLE_VALUE)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }

    // Subtract 1 from the path length since this function does not add a null terminator.
    DWORD result = ::GetFinalPathNameByHandle(h, buf, buf_len - 1, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (result == 0)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);

    ::CloseHandle(h);
    return error;
}

bool
FileSystem::CalculateMD5(const FileSpec &file_spec, uint64_t &low, uint64_t &high)
{
    return false;
}
