//===-- HostInfoLinux.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Log.h"
#include "lldb/Host/linux/HostInfoLinux.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/utsname.h>

#include <algorithm>

using namespace lldb_private;

namespace
{
struct HostInfoLinuxFields
{
    HostInfoLinuxFields()
        : m_os_major(0)
        , m_os_minor(0)
        , m_os_update(0)
    {
    }

    std::string m_distribution_id;
    uint32_t m_os_major;
    uint32_t m_os_minor;
    uint32_t m_os_update;
};

HostInfoLinuxFields *g_fields = nullptr;
}

void
HostInfoLinux::Initialize()
{
    HostInfoPosix::Initialize();

    g_fields = new HostInfoLinuxFields();
}

bool
HostInfoLinux::GetOSVersion(uint32_t &major, uint32_t &minor, uint32_t &update)
{
    static bool is_initialized = false;
    static bool success = false;

    if (!is_initialized)
    {
        is_initialized = true;
        struct utsname un;

        if (uname(&un))
            goto finished;

        int status = sscanf(un.release, "%u.%u.%u", &g_fields->m_os_major, &g_fields->m_os_minor, &g_fields->m_os_update);
        if (status == 3)
        {
            success = true;
            goto finished;
        }

        // Some kernels omit the update version, so try looking for just "X.Y" and
        // set update to 0.
        g_fields->m_os_update = 0;
        status = sscanf(un.release, "%u.%u", &g_fields->m_os_major, &g_fields->m_os_minor);
        success = !!(status == 2);
    }

finished:
    major = g_fields->m_os_major;
    minor = g_fields->m_os_minor;
    update = g_fields->m_os_update;
    return success;
}

llvm::StringRef
HostInfoLinux::GetDistributionId()
{
    static bool is_initialized = false;
    // Try to run 'lbs_release -i', and use that response
    // for the distribution id.

    if (!is_initialized)
    {
        is_initialized = true;

        Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST));
        if (log)
            log->Printf("attempting to determine Linux distribution...");

        // check if the lsb_release command exists at one of the
        // following paths
        const char *const exe_paths[] = {"/bin/lsb_release", "/usr/bin/lsb_release"};

        for (size_t exe_index = 0; exe_index < sizeof(exe_paths) / sizeof(exe_paths[0]); ++exe_index)
        {
            const char *const get_distribution_info_exe = exe_paths[exe_index];
            if (access(get_distribution_info_exe, F_OK))
            {
                // this exe doesn't exist, move on to next exe
                if (log)
                    log->Printf("executable doesn't exist: %s", get_distribution_info_exe);
                continue;
            }

            // execute the distribution-retrieval command, read output
            std::string get_distribution_id_command(get_distribution_info_exe);
            get_distribution_id_command += " -i";

            FILE *file = popen(get_distribution_id_command.c_str(), "r");
            if (!file)
            {
                if (log)
                    log->Printf("failed to run command: \"%s\", cannot retrieve "
                                "platform information",
                                get_distribution_id_command.c_str());
                break;
            }

            // retrieve the distribution id string.
            char distribution_id[256] = {'\0'};
            if (fgets(distribution_id, sizeof(distribution_id) - 1, file) != NULL)
            {
                if (log)
                    log->Printf("distribution id command returned \"%s\"", distribution_id);

                const char *const distributor_id_key = "Distributor ID:\t";
                if (strstr(distribution_id, distributor_id_key))
                {
                    // strip newlines
                    std::string id_string(distribution_id + strlen(distributor_id_key));
                    id_string.erase(std::remove(id_string.begin(), id_string.end(), '\n'), id_string.end());

                    // lower case it and convert whitespace to underscores
                    std::transform(id_string.begin(), id_string.end(), id_string.begin(), [](char ch)
                                   {
                        return tolower(isspace(ch) ? '_' : ch);
                    });

                    g_fields->m_distribution_id = id_string;
                    if (log)
                        log->Printf("distribution id set to \"%s\"", g_fields->m_distribution_id.c_str());
                }
                else
                {
                    if (log)
                        log->Printf("failed to find \"%s\" field in \"%s\"", distributor_id_key, distribution_id);
                }
            }
            else
            {
                if (log)
                    log->Printf("failed to retrieve distribution id, \"%s\" returned no"
                                " lines",
                                get_distribution_id_command.c_str());
            }

            // clean up the file
            pclose(file);
        }
    }

    return g_fields->m_distribution_id.c_str();
}

FileSpec
HostInfoLinux::GetProgramFileSpec()
{
    static FileSpec g_program_filespec;

    if (!g_program_filespec)
    {
        char exe_path[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (len > 0)
        {
            exe_path[len] = 0;
            g_program_filespec.SetFile(exe_path, false);
        }
    }

    return g_program_filespec;
}

bool
HostInfoLinux::ComputeSystemPluginsDirectory(FileSpec &file_spec)
{
    file_spec.SetFile("/usr/lib/lldb", true);
    return true;
}

bool
HostInfoLinux::ComputeUserPluginsDirectory(FileSpec &file_spec)
{
    // XDG Base Directory Specification
    // http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
    // If XDG_DATA_HOME exists, use that, otherwise use ~/.local/share/lldb.
    FileSpec lldb_file_spec;
    const char *xdg_data_home = getenv("XDG_DATA_HOME");
    if (xdg_data_home && xdg_data_home[0])
    {
        std::string user_plugin_dir(xdg_data_home);
        user_plugin_dir += "/lldb";
        lldb_file_spec.SetFile(user_plugin_dir.c_str(), true);
    }
    else
        lldb_file_spec.SetFile("~/.local/share/lldb", true);

    return true;
}

void
HostInfoLinux::ComputeHostArchitectureSupport(ArchSpec &arch_32, ArchSpec &arch_64)
{
    HostInfoPosix::ComputeHostArchitectureSupport(arch_32, arch_64);

    const char *distribution_id = GetDistributionId().data();

    // On Linux, "unknown" in the vendor slot isn't what we want for the default
    // triple.  It's probably an artifact of config.guess.
    if (arch_32.IsValid())
    {
        arch_32.SetDistributionId(distribution_id);
        if (arch_32.GetTriple().getVendor() == llvm::Triple::UnknownVendor)
            arch_32.GetTriple().setVendorName("");
    }
    if (arch_64.IsValid())
    {
        arch_64.SetDistributionId(distribution_id);
        if (arch_64.GetTriple().getVendor() == llvm::Triple::UnknownVendor)
            arch_64.GetTriple().setVendorName("");
    }
}
