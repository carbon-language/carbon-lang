//===-- Platform.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Platform_h_
#define liblldb_Platform_h_

// C Includes
// C++ Includes
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Host/Mutex.h"

namespace lldb_private {

    //----------------------------------------------------------------------
    /// @class Platform Platform.h "lldb/Target/Platform.h"
    /// @brief A plug-in interface definition class for debug platform that
    /// includes many platform abilities such as:
    ///     @li getting platform information such as supported architectures,
    ///         supported binary file formats and more
    ///     @li launching new processes
    ///     @li attaching to existing processes
    ///     @li download/upload files
    ///     @li execute shell commands
    ///     @li listing and getting info for existing processes
    ///     @li attaching and possibly debugging the platform's kernel
    //----------------------------------------------------------------------
    class Platform : public PluginInterface
    {
    public:

        //------------------------------------------------------------------
        /// Get the native host platform plug-in. 
        ///
        /// There should only be one of these for each host that LLDB runs
        /// upon that should be statically compiled in and registered using
        /// preprocessor macros or other similar build mechanisms in a 
        /// PlatformSubclass::Initialize() function.
        ///
        /// This platform will be used as the default platform when launching
        /// or attaching to processes unless another platform is specified.
        //------------------------------------------------------------------
        static lldb::PlatformSP
        GetDefaultPlatform ();

        static void
        SetDefaultPlatform (const lldb::PlatformSP &platform_sp);

        static lldb::PlatformSP
        Create (const char *platform_name, Error &error);

        static uint32_t
        GetNumConnectedRemotePlatforms ();
        
        static lldb::PlatformSP
        GetConnectedRemotePlatformAtIndex (uint32_t idx);

        //------------------------------------------------------------------
        /// Default Constructor
        //------------------------------------------------------------------
        Platform (bool is_host_platform);

        //------------------------------------------------------------------
        /// Destructor.
        ///
        /// The destructor is virtual since this class is designed to be
        /// inherited from by the plug-in instance.
        //------------------------------------------------------------------
        virtual
        ~Platform();

        //------------------------------------------------------------------
        /// Set the target's executable based off of the existing 
        /// architecture information in \a target given a path to an 
        /// executable \a exe_file.
        ///
        /// Each platform knows the architectures that it supports and can
        /// select the correct architecture slice within \a exe_file by 
        /// inspecting the architecture in \a target. If the target had an
        /// architecture specified, then in can try and obey that request
        /// and optionally fail if the architecture doesn't match up.
        /// If no architecture is specified, the platform should select the
        /// default architecture from \a exe_file. Any application bundles
        /// or executable wrappers can also be inspected for the actual
        /// application binary within the bundle that should be used.
        ///
        /// @return
        ///     Returns \b true if this Platform plug-in was able to find
        ///     a suitable executable, \b false otherwise.
        //------------------------------------------------------------------
        virtual Error
        ResolveExecutable (const FileSpec &exe_file,
                           const ArchSpec &arch,
                           lldb::ModuleSP &module_sp);

        bool
        GetOSVersion (uint32_t &major, 
                      uint32_t &minor, 
                      uint32_t &update);
           
        bool
        SetOSVersion (uint32_t major, 
                      uint32_t minor, 
                      uint32_t update);

        bool
        GetOSBuildString (std::string &s);
        
        bool
        GetOSKernelDescription (std::string &s);

        const char *
        GetHostname ();

        virtual const char *
        GetDescription () = 0;

        //------------------------------------------------------------------
        /// Report the current status for this platform. 
        ///
        /// The returned string usually involves returning the OS version
        /// (if available), and any SDK directory that might be being used
        /// for local file caching, and if connected a quick blurb about
        /// what this platform is connected to.
        //------------------------------------------------------------------        
        virtual void
        GetStatus (Stream &strm);

        //------------------------------------------------------------------
        // Subclasses must be able to fetch the current OS version
        //
        // Remote classes must be connected for this to succeed. Local 
        // subclasses don't need to override this function as it will just
        // call the Host::GetOSVersion().
        //------------------------------------------------------------------
        virtual bool
        GetRemoteOSVersion ()
        {
            return false;
        }

        virtual bool
        GetRemoteOSBuildString (std::string &s)
        {
            s.clear();
            return false;
        }
        
        virtual bool
        GetRemoteOSKernelDescription (std::string &s)
        {
            s.clear();
            return false;
        }

        // Remote Platform subclasses need to override this function
        virtual ArchSpec
        GetRemoteSystemArchitecture ()
        {
            return ArchSpec(); // Return an invalid architecture
        }

        // Remote subclasses should override this and return a valid instance
        // name if connected.
        virtual const char *
        GetRemoteHostname ()
        {
            return NULL;
        }

        //------------------------------------------------------------------
        /// Locate a file for a platform.
        ///
        /// The default implementation of this function will return the same
        /// file patch in \a local_file as was in \a platform_file.
        ///
        /// @param[in] platform_file
        ///     The platform file path to locate and cache locally.
        ///
        /// @param[out] local_file
        ///     A locally cached version of the platform file. For platforms
        ///     that describe the current host computer, this will just be
        ///     the same file. For remote platforms, this file might come from
        ///     and SDK directory, or might need to be sync'ed over to the
        ///     current machine for efficient debugging access.
        ///
        /// @return
        ///     An error object.
        //------------------------------------------------------------------
        virtual Error
        GetFile (const FileSpec &platform_file, 
                 const UUID *uuid_ptr,
                 FileSpec &local_file);

        virtual Error
        ConnectRemote (Args& args);

        virtual Error
        DisconnectRemote ();

        //------------------------------------------------------------------
        /// Get the platform's supported architectures in the order in which
        /// they should be searched.
        ///
        /// @param[in] idx
        ///     A zero based architecture index
        ///
        /// @param[out] arch
        ///     A copy of the archgitecture at index if the return value is
        ///     \b true.
        ///
        /// @return
        ///     \b true if \a arch was filled in and is valid, \b false 
        ///     otherwise.
        //------------------------------------------------------------------
        virtual bool
        GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch) = 0;

        virtual size_t
        GetSoftwareBreakpointTrapOpcode (Target &target,
                                         BreakpointSite *bp_site) = 0;

        //------------------------------------------------------------------
        /// Launch a new process.
        ///
        /// Launch a new process by spawning a new process using the
        /// target object's executable module's file as the file to launch.
        /// Arguments are given in \a argv, and the environment variables
        /// are in \a envp. Standard input and output files can be
        /// optionally re-directed to \a stdin_path, \a stdout_path, and
        /// \a stderr_path.
        ///
        /// This function is not meant to be overridden by Process
        /// subclasses. It will first call Process::WillLaunch (Module *)
        /// and if that returns \b true, Process::DoLaunch (Module*,
        /// char const *[],char const *[],const char *,const char *,
        /// const char *) will be called to actually do the launching. If
        /// DoLaunch returns \b true, then Process::DidLaunch() will be
        /// called.
        ///
        /// @param[in] argv
        ///     The argument array.
        ///
        /// @param[in] envp
        ///     The environment array.
        ///
        /// @param[in] launch_flags
        ///     Flags to modify the launch (@see lldb::LaunchFlags)
        ///
        /// @param[in] stdin_path
        ///     The path to use when re-directing the STDIN of the new
        ///     process. If all stdXX_path arguments are NULL, a pseudo
        ///     terminal will be used.
        ///
        /// @param[in] stdout_path
        ///     The path to use when re-directing the STDOUT of the new
        ///     process. If all stdXX_path arguments are NULL, a pseudo
        ///     terminal will be used.
        ///
        /// @param[in] stderr_path
        ///     The path to use when re-directing the STDERR of the new
        ///     process. If all stdXX_path arguments are NULL, a pseudo
        ///     terminal will be used.
        ///
        /// @param[in] working_directory
        ///     The working directory to have the child process run in
        ///
        /// @return
        ///     An error object. Call GetID() to get the process ID if
        ///     the error object is success.
        //------------------------------------------------------------------
//        virtual lldb::ProcessSP
//        Launch (char const *argv[],
//                char const *envp[],
//                uint32_t launch_flags,
//                const char *stdin_path,
//                const char *stdout_path,
//                const char *stderr_path,
//                const char *working_directory,
//                Error &error) = 0;

        //------------------------------------------------------------------
        /// Attach to an existing process using a process ID.
        ///
        /// This function is not meant to be overridden by Process
        /// subclasses. It will first call Process::WillAttach (lldb::pid_t)
        /// and if that returns \b true, Process::DoAttach (lldb::pid_t) will
        /// be called to actually do the attach. If DoAttach returns \b
        /// true, then Process::DidAttach() will be called.
        ///
        /// @param[in] pid
        ///     The process ID that we should attempt to attach to.
        ///
        /// @return
        ///     Returns \a pid if attaching was successful, or
        ///     LLDB_INVALID_PROCESS_ID if attaching fails.
        //------------------------------------------------------------------
//        virtual lldb::ProcessSP
//        Attach (lldb::pid_t pid, 
//                Error &error) = 0;

        //------------------------------------------------------------------
        /// Attach to an existing process by process name.
        ///
        /// This function is not meant to be overridden by Process
        /// subclasses. It will first call
        /// Process::WillAttach (const char *) and if that returns \b
        /// true, Process::DoAttach (const char *) will be called to
        /// actually do the attach. If DoAttach returns \b true, then
        /// Process::DidAttach() will be called.
        ///
        /// @param[in] process_name
        ///     A process name to match against the current process list.
        ///
        /// @return
        ///     Returns \a pid if attaching was successful, or
        ///     LLDB_INVALID_PROCESS_ID if attaching fails.
        //------------------------------------------------------------------
//        virtual lldb::ProcessSP
//        Attach (const char *process_name, 
//                bool wait_for_launch, 
//                Error &error) = 0;
        
        virtual uint32_t
        FindProcessesByName (const char *name, 
                             NameMatchType name_match_type,
                             ProcessInfoList &proc_infos) = 0;

        virtual bool
        GetProcessInfo (lldb::pid_t pid, ProcessInfo &proc_info) = 0;

        const std::string &
        GetRemoteURL () const
        {
            return m_remote_url;
        }

        bool
        IsHost () const
        {
            return m_is_host;    // Is this the default host platform?
        }

        bool
        IsRemote () const
        {
            return !m_is_host;
        }
        
        virtual bool
        IsConnected () const
        {
            // Remote subclasses should override this function
            return IsHost();
        }
        
        const ArchSpec &
        GetSystemArchitecture();

        void
        SetSystemArchitecture (const ArchSpec &arch)
        {
            m_system_arch = arch;
            if (IsHost())
                m_os_version_set_while_connected = m_system_arch.IsValid();
        }

    protected:
        bool m_is_host;
        // Set to true when we are able to actually set the OS version while 
        // being connected. For remote platforms, we might set the version ahead
        // of time before we actually connect and this version might change when
        // we actually connect to a remote platform. For the host platform this
        // will be set to the once we call Host::GetOSVersion().
        bool m_os_version_set_while_connected;
        bool m_system_arch_set_while_connected;
        std::string m_remote_url;
        std::string m_name;
        uint32_t m_major_os_version;
        uint32_t m_minor_os_version;
        uint32_t m_update_os_version;
        ArchSpec m_system_arch; // The architecture of the kernel or the remote platform
    private:
        DISALLOW_COPY_AND_ASSIGN (Platform);
    };

    
    class PlatformList
    {
    public:
        PlatformList() :
            m_mutex (Mutex::eMutexTypeRecursive),
            m_platforms ()
        {
        }
        
        ~PlatformList()
        {
        }
        
        void
        Append (const lldb::PlatformSP &platform_sp, bool set_selected)
        {
            Mutex::Locker locker (m_mutex);
            m_platforms.push_back (platform_sp);
            if (set_selected)
                m_selected_platform_sp = m_platforms.back();
        }

        size_t
        GetSize()
        {
            Mutex::Locker locker (m_mutex);
            return m_platforms.size();
        }

        lldb::PlatformSP
        GetAtIndex (uint32_t idx)
        {
            lldb::PlatformSP platform_sp;
            {
                Mutex::Locker locker (m_mutex);
                if (idx < m_platforms.size())
                    platform_sp = m_platforms[idx];
            }
            return platform_sp;
        }

        //------------------------------------------------------------------
        /// Select the active platform.
        ///
        /// In order to debug remotely, other platform's can be remotely
        /// connected to and set as the selected platform for any subsequent
        /// debugging. This allows connection to remote targets and allows
        /// the ability to discover process info, launch and attach to remote
        /// processes.
        //------------------------------------------------------------------
        lldb::PlatformSP
        GetSelectedPlatform ()
        {
            Mutex::Locker locker (m_mutex);
            if (!m_selected_platform_sp && !m_platforms.empty())
                m_selected_platform_sp = m_platforms.front();
            
            return m_selected_platform_sp;
        }

        void
        SetSelectedPlatform (const lldb::PlatformSP &platform_sp)
        {
            if (platform_sp)
            {
                Mutex::Locker locker (m_mutex);
                const size_t num_platforms = m_platforms.size();
                for (size_t idx=0; idx<num_platforms; ++idx)
                {
                    if (m_platforms[idx].get() == platform_sp.get())
                    {
                        m_selected_platform_sp = m_platforms[idx];
                        return;
                    }
                }
                m_platforms.push_back (platform_sp);
                m_selected_platform_sp = m_platforms.back();
            }
        }

    protected:
        typedef std::vector<lldb::PlatformSP> collection;
        mutable Mutex m_mutex;
        collection m_platforms;
        lldb::PlatformSP m_selected_platform_sp;

    private:
        DISALLOW_COPY_AND_ASSIGN (PlatformList);
    };
} // namespace lldb_private

#endif  // liblldb_Platform_h_
