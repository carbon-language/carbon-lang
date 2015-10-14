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
#include <sys/stat.h>
#include <sys/types.h>

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/windows/AutoHandle.h"
#include "llvm/Support/FileSystem.h"

using namespace lldb_private;

const char *
FileSystem::DEV_NULL = "nul";

FileSpec::PathSyntax
FileSystem::GetNativePathSyntax()
{
    return FileSpec::ePathSyntaxWindows;
}

Error
FileSystem::MakeDirectory(const FileSpec &file_spec, uint32_t file_permissions)
{
    // On Win32, the mode parameter is ignored, as Windows files and directories support a
    // different permission model than POSIX.
    Error error;
    const auto err_code = llvm::sys::fs::create_directories(file_spec.GetPath(), true);
    if (err_code)
    {
        error.SetErrorString(err_code.message().c_str());
    }

    return error;
}

Error
FileSystem::DeleteDirectory(const FileSpec &file_spec, bool recurse)
{
    Error error;
    if (!recurse)
    {
        BOOL result = ::RemoveDirectory(file_spec.GetCString());
        if (!result)
            error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    }
    else
    {
        // SHFileOperation() accepts a list of paths, and so must be double-null-terminated to
        // indicate the end of the list.
        std::string path_buffer{file_spec.GetPath()};
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
FileSystem::GetFilePermissions(const FileSpec &file_spec, uint32_t &file_permissions)
{
    Error error;
    // Beware that Windows's permission model is different from Unix's, and it's
    // not clear if this API is supposed to check ACLs.  To match the caller's
    // expectations as closely as possible, we'll use Microsoft's _stat, which
    // attempts to emulate POSIX stat.  This should be good enough for basic
    // checks like FileSpec::Readable.
    struct _stat file_stats;
    if (::_stat(file_spec.GetCString(), &file_stats) == 0)
    {
        // The owner permission bits in "st_mode" currently match the definitions
        // for the owner file mode bits.
        file_permissions = file_stats.st_mode & (_S_IREAD | _S_IWRITE | _S_IEXEC);
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
FileSystem::Hardlink(const FileSpec &src, const FileSpec &dst)
{
    Error error;
    if (!::CreateHardLink(src.GetCString(), dst.GetCString(), nullptr))
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
}

int
FileSystem::GetHardlinkCount(const FileSpec &file_spec)
{
    HANDLE file_handle = ::CreateFile(file_spec.GetCString(),
                                      FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ,
                                      nullptr,
                                      OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL,
                                      nullptr);

    if (file_handle == INVALID_HANDLE_VALUE)
      return -1;

    AutoHandle auto_file_handle(file_handle);
    BY_HANDLE_FILE_INFORMATION file_info;
    if (::GetFileInformationByHandle(file_handle, &file_info))
        return file_info.nNumberOfLinks;

    return -1;
}

Error
FileSystem::Symlink(const FileSpec &src, const FileSpec &dst)
{
    Error error;
    DWORD attrib = ::GetFileAttributes(dst.GetCString());
    if (attrib == INVALID_FILE_ATTRIBUTES)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }
    bool is_directory = !!(attrib & FILE_ATTRIBUTE_DIRECTORY);
    DWORD flag = is_directory ? SYMBOLIC_LINK_FLAG_DIRECTORY : 0;
    BOOL result = ::CreateSymbolicLink(src.GetCString(), dst.GetCString(), flag);
    if (!result)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
}

Error
FileSystem::Unlink(const FileSpec &file_spec)
{
    Error error;
    BOOL result = ::DeleteFile(file_spec.GetCString());
    if (!result)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
}

Error
FileSystem::Readlink(const FileSpec &src, FileSpec &dst)
{
    Error error;
    HANDLE h = ::CreateFile(src.GetCString(), GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT, NULL);
    if (h == INVALID_HANDLE_VALUE)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }

    char buf[PATH_MAX];
    // Subtract 1 from the path length since this function does not add a null terminator.
    DWORD result = ::GetFinalPathNameByHandle(h, buf, sizeof(buf) - 1,
            FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (result == 0)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    else
        dst.SetFile(buf, false);

    ::CloseHandle(h);
    return error;
}

Error
FileSystem::ResolveSymbolicLink(const FileSpec &src, FileSpec &dst)
{
    return Error("ResolveSymbolicLink() isn't implemented on Windows");
}

bool
FileSystem::IsLocal(const FileSpec &spec)
{
    if (spec)
    {
        // TODO: return true if the file is on a locally mounted file system
        return true;
    }

    return false;
}
