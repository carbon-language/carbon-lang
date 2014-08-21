//===-- HostInfoPosix.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Core/Log.h"
#include "lldb/Host/posix/HostInfoPosix.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <netdb.h>
#include <limits.h>
#include <unistd.h>

using namespace lldb_private;

size_t
HostInfoPosix::GetPageSize()
{
    return ::getpagesize();
}

bool
HostInfoPosix::GetHostname(std::string &s)
{
    char hostname[PATH_MAX];
    hostname[sizeof(hostname) - 1] = '\0';
    if (::gethostname(hostname, sizeof(hostname) - 1) == 0)
    {
        struct hostent *h = ::gethostbyname(hostname);
        if (h)
            s.assign(h->h_name);
        else
            s.assign(hostname);
        return true;
    }
    return false;
}

bool
HostInfoPosix::ComputeSupportExeDirectory(FileSpec &file_spec)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);

    FileSpec lldb_file_spec;
    if (!GetLLDBPath(lldb::ePathTypeLLDBShlibDir, lldb_file_spec))
        return false;

    char raw_path[PATH_MAX];
    lldb_file_spec.GetPath(raw_path, sizeof(raw_path));

    // Most Posix systems (e.g. Linux/*BSD) will attempt to replace a */lib with */bin as the base
    // directory for helper exe programs.  This will fail if the /lib and /bin directories are
    // rooted in entirely different trees.
    if (log)
        log->Printf("HostInfoPosix::ComputeSupportExeDirectory() attempting to derive the bin path (ePathTypeSupportExecutableDir) from "
                    "this path: %s",
                    raw_path);
    char *lib_pos = ::strstr(raw_path, "/lib");
    if (lib_pos != nullptr)
    {
        // First terminate the raw path at the start of lib.
        *lib_pos = '\0';

        // Now write in bin in place of lib.
        ::strncpy(lib_pos, "/bin", PATH_MAX - (lib_pos - raw_path));

        if (log)
            log->Printf("Host::%s() derived the bin path as: %s", __FUNCTION__, raw_path);
    }
    else
    {
        if (log)
            log->Printf("Host::%s() failed to find /lib/liblldb within the shared lib path, bailing on bin path construction",
                        __FUNCTION__);
    }
    file_spec.SetFile(raw_path, true);
    return (bool)file_spec.GetDirectory();
}

bool
HostInfoPosix::ComputeHeaderDirectory(FileSpec &file_spec)
{
    file_spec.SetFile("/opt/local/include/lldb", false);
    return true;
}

bool
HostInfoPosix::ComputePythonDirectory(FileSpec &file_spec)
{
    FileSpec lldb_file_spec;
    if (!GetLLDBPath(lldb::ePathTypeLLDBShlibDir, lldb_file_spec))
        return false;

    char raw_path[PATH_MAX];
    lldb_file_spec.GetPath(raw_path, sizeof(raw_path));

    llvm::SmallString<256> python_version_dir;
    llvm::raw_svector_ostream os(python_version_dir);
    os << "/python" << PY_MAJOR_VERSION << '.' << PY_MINOR_VERSION << "/site-packages";
    os.flush();

    // We may get our string truncated. Should we protect this with an assert?
    ::strncat(raw_path, python_version_dir.c_str(), sizeof(raw_path) - strlen(raw_path) - 1);

    file_spec.SetFile(raw_path, true);
    return true;
}
