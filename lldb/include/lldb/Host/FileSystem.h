//===-- FileSystem.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_FileSystem_h
#define liblldb_Host_FileSystem_h

#include <stdint.h>

#include "lldb/lldb-types.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/FileSpec.h"

namespace lldb_private
{
class FileSystem
{
  public:
    static FileSpec::PathSyntax GetNativePathSyntax();

    static Error MakeDirectory(const char *path, uint32_t mode);
    static Error DeleteDirectory(const char *path, bool recurse);

    static Error GetFilePermissions(const char *path, uint32_t &file_permissions);
    static Error SetFilePermissions(const char *path, uint32_t file_permissions);
    static lldb::user_id_t GetFileSize(const FileSpec &file_spec);
    static bool GetFileExists(const FileSpec &file_spec);

    static Error Symlink(const char *src, const char *dst);
    static Error Readlink(const char *path, char *buf, size_t buf_len);
    static Error Unlink(const char *path);

    static bool CalculateMD5(const FileSpec &file_spec, uint64_t &low, uint64_t &high);
};
}

#endif
