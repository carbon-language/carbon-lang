//===-- PlatformPOSIX.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformPOSIX_h_
#define liblldb_PlatformPOSIX_h_

// C Includes
// C++ Includes

#include <memory>

// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Target/Platform.h"

class PlatformPOSIX : public lldb_private::Platform
{
public:
    PlatformPOSIX (bool is_host);
    
    virtual
    ~PlatformPOSIX();
    
    //------------------------------------------------------------
    // lldb_private::Platform functions
    //------------------------------------------------------------
    virtual lldb_private::OptionGroupOptions*
    GetConnectionOptions (lldb_private::CommandInterpreter& interpreter);
    
    virtual lldb_private::Error
    PutFile (const lldb_private::FileSpec& source,
             const lldb_private::FileSpec& destination,
             uint32_t uid = UINT32_MAX,
             uint32_t gid = UINT32_MAX);
    
    virtual lldb::user_id_t
    OpenFile (const lldb_private::FileSpec& file_spec,
              uint32_t flags,
              uint32_t mode,
              lldb_private::Error &error);
    
    virtual bool
    CloseFile (lldb::user_id_t fd,
               lldb_private::Error &error);
    
    virtual uint64_t
    ReadFile (lldb::user_id_t fd,
              uint64_t offset,
              void *dst,
              uint64_t dst_len,
              lldb_private::Error &error);
    
    virtual uint64_t
    WriteFile (lldb::user_id_t fd,
               uint64_t offset,
               const void* src,
               uint64_t src_len,
               lldb_private::Error &error);
    
    virtual lldb::user_id_t
    GetFileSize (const lldb_private::FileSpec& file_spec);

    virtual lldb_private::Error
    CreateSymlink(const char *src, const char *dst);

    virtual lldb_private::Error
    GetFile (const lldb_private::FileSpec& source,
             const lldb_private::FileSpec& destination);
    
    virtual lldb_private::ConstString
    GetRemoteWorkingDirectory();
    
    virtual bool
    SetRemoteWorkingDirectory(const lldb_private::ConstString &path);
    
    virtual lldb_private::Error
    RunShellCommand (const char *command,           // Shouldn't be NULL
                     const char *working_dir,       // Pass NULL to use the current working directory
                     int *status_ptr,               // Pass NULL if you don't want the process exit status
                     int *signo_ptr,                // Pass NULL if you don't want the signal that caused the process to exit
                     std::string *command_output,   // Pass NULL if you don't want the command output
                     uint32_t timeout_sec);         // Timeout in seconds to wait for shell program to finish
    
    virtual lldb_private::Error
    MakeDirectory (const char *path, uint32_t mode);
    
    virtual lldb_private::Error
    GetFilePermissions (const char *path, uint32_t &file_permissions);

    virtual lldb_private::Error
    SetFilePermissions (const char *path, uint32_t file_permissions);

    virtual bool
    GetFileExists (const lldb_private::FileSpec& file_spec);
    
    virtual lldb_private::Error
    Unlink (const char *path);

    virtual std::string
    GetPlatformSpecificConnectionInformation();
    
    virtual bool
    CalculateMD5 (const lldb_private::FileSpec& file_spec,
                  uint64_t &low,
                  uint64_t &high);

protected:
    std::unique_ptr<lldb_private::OptionGroupOptions> m_options;
        
    lldb::PlatformSP m_remote_platform_sp; // Allow multiple ways to connect to a remote POSIX-compliant OS
    
private:
    DISALLOW_COPY_AND_ASSIGN (PlatformPOSIX);
    
};

#endif  // liblldb_PlatformPOSIX_h_
