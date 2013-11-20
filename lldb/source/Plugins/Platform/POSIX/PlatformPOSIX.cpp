//===-- PlatformPOSIX.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformPOSIX.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"

using namespace lldb;
using namespace lldb_private;


//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformPOSIX::PlatformPOSIX (bool is_host) :
Platform(is_host),  // This is the local host platform
m_remote_platform_sp ()
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformPOSIX::~PlatformPOSIX()
{
}

lldb_private::OptionGroupOptions*
PlatformPOSIX::GetConnectionOptions (lldb_private::CommandInterpreter& interpreter)
{
    if (m_options.get() == NULL)
    {
        m_options.reset(new OptionGroupOptions(interpreter));
        m_options->Append(new OptionGroupPlatformRSync());
        m_options->Append(new OptionGroupPlatformSSH());
        m_options->Append(new OptionGroupPlatformCaching());
    }
    return m_options.get();
}

lldb_private::Error
PlatformPOSIX::RunShellCommand (const char *command,           // Shouldn't be NULL
                                const char *working_dir,       // Pass NULL to use the current working directory
                                int *status_ptr,               // Pass NULL if you don't want the process exit status
                                int *signo_ptr,                // Pass NULL if you don't want the signal that caused the process to exit
                                std::string *command_output,   // Pass NULL if you don't want the command output
                                uint32_t timeout_sec)         // Timeout in seconds to wait for shell program to finish
{
    if (IsHost())
        return Host::RunShellCommand(command, working_dir, status_ptr, signo_ptr, command_output, timeout_sec);
    else
    {
        if (m_remote_platform_sp)
            return m_remote_platform_sp->RunShellCommand(command, working_dir, status_ptr, signo_ptr, command_output, timeout_sec);
        else
            return Error("unable to run a remote command without a platform");
    }
}

Error
PlatformPOSIX::MakeDirectory (const char *path, uint32_t file_permissions)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->MakeDirectory(path, file_permissions);
    else
        return Platform::MakeDirectory(path ,file_permissions);
}

Error
PlatformPOSIX::GetFilePermissions (const char *path, uint32_t &file_permissions)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetFilePermissions(path, file_permissions);
    else
        return Platform::GetFilePermissions(path ,file_permissions);
}

Error
PlatformPOSIX::SetFilePermissions (const char *path, uint32_t file_permissions)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->MakeDirectory(path, file_permissions);
    else
        return Platform::SetFilePermissions(path ,file_permissions);
}

lldb::user_id_t
PlatformPOSIX::OpenFile (const FileSpec& file_spec,
                         uint32_t flags,
                         uint32_t mode,
                         Error &error)
{
    if (IsHost())
        return Host::OpenFile(file_spec, flags, mode, error);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->OpenFile(file_spec, flags, mode, error);
    else
        return Platform::OpenFile(file_spec, flags, mode, error);
}

bool
PlatformPOSIX::CloseFile (lldb::user_id_t fd, Error &error)
{
    if (IsHost())
        return Host::CloseFile(fd, error);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->CloseFile(fd, error);
    else
        return Platform::CloseFile(fd, error);
}

uint64_t
PlatformPOSIX::ReadFile (lldb::user_id_t fd,
                         uint64_t offset,
                         void *dst,
                         uint64_t dst_len,
                         Error &error)
{
    if (IsHost())
        return Host::ReadFile(fd, offset, dst, dst_len, error);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->ReadFile(fd, offset, dst, dst_len, error);
    else
        return Platform::ReadFile(fd, offset, dst, dst_len, error);
}

uint64_t
PlatformPOSIX::WriteFile (lldb::user_id_t fd,
                          uint64_t offset,
                          const void* src,
                          uint64_t src_len,
                          Error &error)
{
    if (IsHost())
        return Host::WriteFile(fd, offset, src, src_len, error);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->WriteFile(fd, offset, src, src_len, error);
    else
        return Platform::WriteFile(fd, offset, src, src_len, error);
}

static uint32_t
chown_file(Platform *platform,
           const char* path,
           uint32_t uid = UINT32_MAX,
           uint32_t gid = UINT32_MAX)
{
    if (!platform || !path || *path == 0)
        return UINT32_MAX;
    
    if (uid == UINT32_MAX && gid == UINT32_MAX)
        return 0;   // pretend I did chown correctly - actually I just didn't care
    
    StreamString command;
    command.PutCString("chown ");
    if (uid != UINT32_MAX)
        command.Printf("%d",uid);
    if (gid != UINT32_MAX)
        command.Printf(":%d",gid);
    command.Printf("%s",path);
    int status;
    platform->RunShellCommand(command.GetData(),
                              NULL,
                              &status,
                              NULL,
                              NULL,
                              10);
    return status;
}

lldb_private::Error
PlatformPOSIX::PutFile (const lldb_private::FileSpec& source,
                         const lldb_private::FileSpec& destination,
                         uint32_t uid,
                         uint32_t gid)
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));

    if (IsHost())
    {
        if (FileSpec::Equal(source, destination, true))
            return Error();
        // cp src dst
        // chown uid:gid dst
        std::string src_path (source.GetPath());
        if (src_path.empty())
            return Error("unable to get file path for source");
        std::string dst_path (destination.GetPath());
        if (dst_path.empty())
            return Error("unable to get file path for destination");
        StreamString command;
        command.Printf("cp %s %s", src_path.c_str(), dst_path.c_str());
        int status;
        RunShellCommand(command.GetData(),
                        NULL,
                        &status,
                        NULL,
                        NULL,
                        10);
        if (status != 0)
            return Error("unable to perform copy");
        if (uid == UINT32_MAX && gid == UINT32_MAX)
            return Error();
        if (chown_file(this,dst_path.c_str(),uid,gid) != 0)
            return Error("unable to perform chown");
        return Error();
    }
    else if (m_remote_platform_sp)
    {
        if (GetSupportsRSync())
        {
            std::string src_path (source.GetPath());
            if (src_path.empty())
                return Error("unable to get file path for source");
            std::string dst_path (destination.GetPath());
            if (dst_path.empty())
                return Error("unable to get file path for destination");
            StreamString command;
            if (GetIgnoresRemoteHostname())
            {
                if (!GetRSyncPrefix())
                    command.Printf("rsync %s %s %s",
                                   GetRSyncOpts(),
                                   src_path.c_str(),
                                   dst_path.c_str());
                else
                    command.Printf("rsync %s %s %s%s",
                                   GetRSyncOpts(),
                                   src_path.c_str(),
                                   GetRSyncPrefix(),
                                   dst_path.c_str());
            }
            else
                command.Printf("rsync %s %s %s:%s",
                               GetRSyncOpts(),
                               src_path.c_str(),
                               GetHostname(),
                               dst_path.c_str());
            if (log)
                log->Printf("[PutFile] Running command: %s\n", command.GetData());
            int retcode;
            Host::RunShellCommand(command.GetData(),
                                  NULL,
                                  &retcode,
                                  NULL,
                                  NULL,
                                  60);
            if (retcode == 0)
            {
                // Don't chown a local file for a remote system
//                if (chown_file(this,dst_path.c_str(),uid,gid) != 0)
//                    return Error("unable to perform chown");
                return Error();
            }
            // if we are still here rsync has failed - let's try the slow way before giving up
        }
        
        if (log)
            log->Printf ("PlatformPOSIX::PutFile(src='%s', dst='%s', uid=%u, gid=%u)",
                         source.GetPath().c_str(),
                         destination.GetPath().c_str(),
                         uid,
                         gid); // REMOVE THIS PRINTF PRIOR TO CHECKIN
        // open
        // read, write, read, write, ...
        // close
        // chown uid:gid dst
        if (log)
            log->Printf("[PutFile] Using block by block transfer....\n");
        
        uint32_t source_open_options = File::eOpenOptionRead;
        if (source.GetFileType() == FileSpec::eFileTypeSymbolicLink)
            source_open_options |= File::eOpenoptionDontFollowSymlinks;

        File source_file(source, source_open_options, lldb::eFilePermissionsUserRW);
        Error error;
        uint32_t permissions = source_file.GetPermissions(error);
        if (permissions == 0)
            permissions = lldb::eFilePermissionsFileDefault;

        if (!source_file.IsValid())
            return Error("unable to open source file");
        lldb::user_id_t dest_file = OpenFile (destination,
                                              File::eOpenOptionCanCreate | File::eOpenOptionWrite | File::eOpenOptionTruncate,
                                              permissions,
                                              error);
        if (log)
            log->Printf ("dest_file = %" PRIu64 "\n", dest_file);
        if (error.Fail())
            return error;
        if (dest_file == UINT64_MAX)
            return Error("unable to open target file");
        lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024, 0));
        uint64_t offset = 0;
        while (error.Success())
        {
            size_t bytes_read = buffer_sp->GetByteSize();
            error = source_file.Read(buffer_sp->GetBytes(), bytes_read);
            if (bytes_read)
            {
                WriteFile(dest_file, offset, buffer_sp->GetBytes(), bytes_read, error);
                offset += bytes_read;
            }
            else
                break;
        }
        CloseFile(dest_file, error);
        if (uid == UINT32_MAX && gid == UINT32_MAX)
            return error;
        // This is remopve, don't chown a local file...
//        std::string dst_path (destination.GetPath());
//        if (chown_file(this,dst_path.c_str(),uid,gid) != 0)
//            return Error("unable to perform chown");
        return error;
    }
    return Platform::PutFile(source,destination,uid,gid);
}

lldb::user_id_t
PlatformPOSIX::GetFileSize (const FileSpec& file_spec)
{
    if (IsHost())
        return Host::GetFileSize(file_spec);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->GetFileSize(file_spec);
    else
        return Platform::GetFileSize(file_spec);
}

Error
PlatformPOSIX::CreateSymlink(const char *src, const char *dst)
{
    if (IsHost())
        return Host::Symlink(src, dst);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->CreateSymlink(src, dst);
    else
        return Platform::CreateSymlink(src, dst);
}

bool
PlatformPOSIX::GetFileExists (const FileSpec& file_spec)
{
    if (IsHost())
        return file_spec.Exists();
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->GetFileExists(file_spec);
    else
        return Platform::GetFileExists(file_spec);
}

Error
PlatformPOSIX::Unlink (const char *path)
{
    if (IsHost())
        return Host::Unlink (path);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->Unlink(path);
    else
        return Platform::Unlink(path);
}

lldb_private::Error
PlatformPOSIX::GetFile (const lldb_private::FileSpec& source /* remote file path */,
                        const lldb_private::FileSpec& destination /* local file path */)
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));

    // Check the args, first.
    std::string src_path (source.GetPath());
    if (src_path.empty())
        return Error("unable to get file path for source");
    std::string dst_path (destination.GetPath());
    if (dst_path.empty())
        return Error("unable to get file path for destination");
    if (IsHost())
    {
        if (FileSpec::Equal(source, destination, true))
            return Error("local scenario->source and destination are the same file path: no operation performed");
        // cp src dst
        StreamString cp_command;
        cp_command.Printf("cp %s %s", src_path.c_str(), dst_path.c_str());
        int status;
        RunShellCommand(cp_command.GetData(),
                        NULL,
                        &status,
                        NULL,
                        NULL,
                        10);
        if (status != 0)
            return Error("unable to perform copy");
        return Error();
    }
    else if (m_remote_platform_sp)
    {
        if (GetSupportsRSync())
        {
            StreamString command;
            if (GetIgnoresRemoteHostname())
            {
                if (!GetRSyncPrefix())
                    command.Printf("rsync %s %s %s",
                                   GetRSyncOpts(),
                                   src_path.c_str(),
                                   dst_path.c_str());
                else
                    command.Printf("rsync %s %s%s %s",
                                   GetRSyncOpts(),
                                   GetRSyncPrefix(),
                                   src_path.c_str(),
                                   dst_path.c_str());
            }
            else
                command.Printf("rsync %s %s:%s %s",
                               GetRSyncOpts(),
                               m_remote_platform_sp->GetHostname(),
                               src_path.c_str(),
                               dst_path.c_str());
            if (log)
                log->Printf("[GetFile] Running command: %s\n", command.GetData());
            int retcode;
            Host::RunShellCommand(command.GetData(),
                                  NULL,
                                  &retcode,
                                  NULL,
                                  NULL,
                                  60);
            if (retcode == 0)
                return Error();
            // If we are here, rsync has failed - let's try the slow way before giving up
        }
        // open src and dst
        // read/write, read/write, read/write, ...
        // close src
        // close dst
        if (log)
            log->Printf("[GetFile] Using block by block transfer....\n");
        Error error;
        user_id_t fd_src = OpenFile (source,
                                     File::eOpenOptionRead,
                                     lldb::eFilePermissionsFileDefault,
                                     error);

        if (fd_src == UINT64_MAX)
            return Error("unable to open source file");

        uint32_t permissions = 0;
        error = GetFilePermissions(source.GetPath().c_str(), permissions);
        
        if (permissions == 0)
            permissions = lldb::eFilePermissionsFileDefault;

        user_id_t fd_dst = Host::OpenFile(destination,
                                          File::eOpenOptionCanCreate | File::eOpenOptionWrite | File::eOpenOptionTruncate,
                                          permissions,
                                          error);

        if (fd_dst == UINT64_MAX)
        {
            if (error.Success())
                error.SetErrorString("unable to open destination file");
        }

        if (error.Success())
        {
            lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024, 0));
            uint64_t offset = 0;
            error.Clear();
            while (error.Success())
            {
                const uint64_t n_read = ReadFile (fd_src,
                                                  offset,
                                                  buffer_sp->GetBytes(),
                                                  buffer_sp->GetByteSize(),
                                                  error);
                if (error.Fail())
                    break;
                if (n_read == 0)
                    break;
                if (Host::WriteFile(fd_dst,
                                    offset,
                                    buffer_sp->GetBytes(),
                                    n_read,
                                    error) != n_read)
                {
                    if (!error.Fail())
                        error.SetErrorString("unable to write to destination file");
                        break;
                }
                offset += n_read;
            }
        }
        // Ignore the close error of src.
        if (fd_src != UINT64_MAX)
            CloseFile(fd_src, error);
        // And close the dst file descriptot.
        if (fd_dst != UINT64_MAX && !Host::CloseFile(fd_dst, error))
        {
            if (!error.Fail())
                error.SetErrorString("unable to close destination file");

        }
        return error;
    }
    return Platform::GetFile(source,destination);
}

std::string
PlatformPOSIX::GetPlatformSpecificConnectionInformation()
{
    StreamString stream;
    if (GetSupportsRSync())
    {
        stream.PutCString("rsync");
        if ( (GetRSyncOpts() && *GetRSyncOpts()) ||
             (GetRSyncPrefix() && *GetRSyncPrefix()) ||
             GetIgnoresRemoteHostname())
        {
            stream.Printf(", options: ");
            if (GetRSyncOpts() && *GetRSyncOpts())
                stream.Printf("'%s' ",GetRSyncOpts());
            stream.Printf(", prefix: ");
            if (GetRSyncPrefix() && *GetRSyncPrefix())
                stream.Printf("'%s' ",GetRSyncPrefix());
            if (GetIgnoresRemoteHostname())
                stream.Printf("ignore remote-hostname ");
        }
    }
    if (GetSupportsSSH())
    {
        stream.PutCString("ssh");
        if (GetSSHOpts() && *GetSSHOpts())
            stream.Printf(", options: '%s' ",GetSSHOpts());
    }
    if (GetLocalCacheDirectory() && *GetLocalCacheDirectory())
        stream.Printf("cache dir: %s",GetLocalCacheDirectory());
    if (stream.GetSize())
        return stream.GetData();
    else
        return "";
}

bool
PlatformPOSIX::CalculateMD5 (const FileSpec& file_spec,
                            uint64_t &low,
                            uint64_t &high)
{
    if (IsHost())
        return Platform::CalculateMD5 (file_spec, low, high);
    if (m_remote_platform_sp)
        return m_remote_platform_sp->CalculateMD5(file_spec, low, high);
    return false;
}

lldb_private::ConstString
PlatformPOSIX::GetRemoteWorkingDirectory()
{
    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteWorkingDirectory();
    else
        return Platform::GetRemoteWorkingDirectory();
}

bool
PlatformPOSIX::SetRemoteWorkingDirectory(const lldb_private::ConstString &path)
{
    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->SetRemoteWorkingDirectory(path);
    else
        return Platform::SetRemoteWorkingDirectory(path);
}

