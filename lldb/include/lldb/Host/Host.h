//===-- Host.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_h_
#define liblldb_Host_h_
#if defined(__cplusplus)


#include "lldb/lldb-private.h"
#include "lldb/Core/StringList.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Host Host.h "lldb/Host/Host.h"
/// @brief A class that provides host computer information.
///
/// Host is a class that answers information about the host operating
/// system.
//----------------------------------------------------------------------
class Host
{
public:
    typedef bool (*MonitorChildProcessCallback) (void *callback_baton,
                                                 lldb::pid_t pid,
                                                 int signal,    // Zero for no signal
                                                 int status);   // Exit value of process if signal is zero

    //------------------------------------------------------------------
    /// Start monitoring a child process.
    ///
    /// Allows easy monitoring of child processes. \a callback will be
    /// called when the child process exits or if it gets a signal. The
    /// callback will only be called with signals if \a monitor_signals
    /// is \b true. \a callback will usually be called from another
    /// thread so the callback function must be thread safe.
    ///
    /// When the callback gets called, the return value indicates if
    /// minotoring should stop. If \b true is returned from \a callback
    /// the information will be removed. If \b false is returned then
    /// monitoring will continue. If the child process exits, the
    /// monitoring will automatically stop after the callback returned
    /// ragardless of the callback return value.
    ///
    /// @param[in] callback
    ///     A function callback to call when a child receives a signal
    ///     (if \a monitor_signals is true) or a child exits.
    ///
    /// @param[in] callback_baton
    ///     A void * of user data that will be pass back when
    ///     \a callback is called.
    ///
    /// @param[in] pid
    ///     The process ID of a child process to monitor, -1 for all
    ///     processes.
    ///
    /// @param[in] monitor_signals
    ///     If \b true the callback will get called when the child
    ///     process gets a signal. If \b false, the callback will only
    ///     get called if the child process exits.
    ///
    /// @return
    ///     A thread handle that can be used to cancel the thread that
    ///     was spawned to monitor \a pid.
    ///
    /// @see static void Host::StopMonitoringChildProcess (uint32_t)
    //------------------------------------------------------------------
    static lldb::thread_t
    StartMonitoringChildProcess (MonitorChildProcessCallback callback,
                                 void *callback_baton,
                                 lldb::pid_t pid,
                                 bool monitor_signals);

    //------------------------------------------------------------------
    /// Get the host page size.
    ///
    /// @return
    ///     The size in bytes of a VM page on the host system.
    //------------------------------------------------------------------
    static size_t
    GetPageSize();

    //------------------------------------------------------------------
    /// Returns the endianness of the host system.
    ///
    /// @return
    ///     Returns the endianness of the host system as a lldb::ByteOrder
    ///     enumeration.
    //------------------------------------------------------------------
    static lldb::ByteOrder
    GetByteOrder ();

    //------------------------------------------------------------------
    /// Gets the host kernel architecture.
    ///
    /// @return
    ///     A const architecture object that represents the host kernel
    ///     architecture.
    //------------------------------------------------------------------
    static const ArchSpec &
    GetArchitecture ();

    //------------------------------------------------------------------
    /// Gets the host vendor string.
    ///
    /// @return
    ///     A const string object containing the host vendor name.
    //------------------------------------------------------------------
    static const ConstString &
    GetVendorString ();

    //------------------------------------------------------------------
    /// Gets the host Operating System (OS) string.
    ///
    /// @return
    ///     A const string object containing the host OS name.
    //------------------------------------------------------------------
    static const ConstString &
    GetOSString ();

    //------------------------------------------------------------------
    /// Gets the host target triple as a const string.
    ///
    /// @return
    ///     A const string object containing the host target triple.
    //------------------------------------------------------------------
    static const ConstString &
    GetTargetTriple ();

    //------------------------------------------------------------------
    /// Get the process ID for the calling process.
    ///
    /// @return
    ///     The process ID for the current process.
    //------------------------------------------------------------------
    static lldb::pid_t
    GetCurrentProcessID ();

    //------------------------------------------------------------------
    /// Get the thread ID for the calling thread in the current process.
    ///
    /// @return
    ///     The thread ID for the calling thread in the current process.
    //------------------------------------------------------------------
    static lldb::tid_t
    GetCurrentThreadID ();

    static const char *
    GetSignalAsCString (int signo);

    static void
    WillTerminate ();
    //------------------------------------------------------------------
    /// Host specific thread created function call.
    ///
    /// This function call lets the current host OS do any thread
    /// specific initialization that it needs, including naming the
    /// thread. No cleanup routine is exptected to be called
    ///
    /// @param[in] name
    ///     The current thread's name in the current process.
    //------------------------------------------------------------------
    static void
    ThreadCreated (const char *name);

    static lldb::thread_t
    ThreadCreate (const char *name,
                  lldb::thread_func_t function,
                  lldb::thread_arg_t thread_arg,
                  Error *err);

    static bool
    ThreadCancel (lldb::thread_t thread,
                  Error *error);

    static bool
    ThreadDetach (lldb::thread_t thread,
                  Error *error);
    static bool
    ThreadJoin (lldb::thread_t thread,
                lldb::thread_result_t *thread_result_ptr,
                Error *error);

    //------------------------------------------------------------------
    /// Gets the name of a thread in a process.
    ///
    /// This function will name a thread in a process using it's own
    /// thread name pool, and also will attempt to set a thread name
    /// using any supported host OS APIs.
    ///
    /// @param[in] pid
    ///     The process ID in which we are trying to get the name of
    ///     a thread.
    ///
    /// @param[in] tid
    ///     The thread ID for which we are trying retrieve the name of.
    ///
    /// @return
    ///     A NULL terminate C string name that is owned by a static
    ///     global string pool, or NULL if there is no matching thread
    ///     name. This string does not need to be freed.
    //------------------------------------------------------------------
    static const char *
    GetThreadName (lldb::pid_t pid, lldb::tid_t tid);

    //------------------------------------------------------------------
    /// Sets the name of a thread in the current process.
    ///
    /// @param[in] pid
    ///     The process ID in which we are trying to name a thread.
    ///
    /// @param[in] tid
    ///     The thread ID which we are trying to name.
    ///
    /// @param[in] name
    ///     The current thread's name in the current process to \a name.
    ///
    /// @return
    ///     \b true if the thread name was able to be set, \b false
    ///     otherwise.
    //------------------------------------------------------------------
    static void
    SetThreadName (lldb::pid_t pid, lldb::tid_t tid, const char *name);

    //------------------------------------------------------------------
    /// Gets the FileSpec of the current process (the process that
    /// that is running the LLDB code).
    ///
    /// @return
    ///     \b A file spec with the program name.
    //------------------------------------------------------------------
    static FileSpec
    GetProgramFileSpec ();

    //------------------------------------------------------------------
    /// Given an address in the current process (the process that
    /// is running the LLDB code), return the name of the module that
    /// it comes from. This can be useful when you need to know the
    /// path to the shared library that your code is running in for
    /// loading resources that are relative to your binary.
    ///
    /// @param[in] host_addr
    ///     The pointer to some code in the current process.
    ///
    /// @return
    ///     \b A file spec with the module that contains \a host_addr,
    ///     which may be invalid if \a host_addr doesn't fall into
    ///     any valid module address range.
    //------------------------------------------------------------------
    static FileSpec
    GetModuleFileSpecForHostAddress (const void *host_addr);


    //------------------------------------------------------------------
    /// When executable files may live within a directory, where the 
    /// directory represents an executable bundle (like the MacOSX 
    /// app bundles), the locate the executable within the containing
    /// bundle.
    ///
    /// @param[in,out] file
    ///     A file spec that currently points to the bundle that will
    ///     be filled in with the executable path within the bundle
    ///     if \b true is returned. Otherwise \a file is left untouched.
    ///
    /// @return
    ///     \b true if \a file was resolved, \b false if this function
    ///     was not able to resolve the path.
    //------------------------------------------------------------------
    static bool
    ResolveExecutableInBundle (FileSpec &file);

    //------------------------------------------------------------------
    /// Find a resource files that are related to LLDB.
    ///
    /// Operating systems have different ways of storing shared 
    /// libraries and related resources. This function abstracts the
    /// access to these paths.
    ///
    /// @param[in] path_type
    ///     The type of LLDB resource path you are looking for. If the
    ///     enumeration ends with "Dir", then only the \a file_spec's 
    ///     directory member gets filled in.
    ///
    /// @param[in] file_spec
    ///     A file spec that gets filled in with the appriopriate path.
    ///
    /// @return
    ///     \b true if \a resource_path was resolved, \a false otherwise.
    //------------------------------------------------------------------
    static bool
    GetLLDBPath (lldb::PathType path_type,
                 FileSpec &file_spec);

    //------------------------------------------------------------------
    /// Set a string that can be displayed if host application crashes.
    ///
    /// Some operating systems have the ability to print a description
    /// for shared libraries when a program crashes. If the host OS
    /// supports such a mechanism, it should be implemented to help
    /// with crash triage.
    ///
    /// @param[in] format
    ///     A printf format that will be used to form a new crash
    ///     description string.
    //------------------------------------------------------------------
    static void
    SetCrashDescriptionWithFormat (const char *format, ...);

    static void
    SetCrashDescription (const char *description);

    static uint32_t
    ListProcessesMatchingName (const char *name, StringList &matches, std::vector<lldb::pid_t> &pids);
    
    static ArchSpec
    GetArchSpecForExistingProcess (lldb::pid_t pid);
    
    static ArchSpec
    GetArchSpecForExistingProcess (const char *process_name);
    
    static lldb::pid_t
    LaunchApplication (const FileSpec &app_file_spec);

    static lldb::pid_t
    LaunchInNewTerminal (const char *tty_name,  // Optional partial or full tty name ("/dev/ttys000" or "ttys000")
                         const char **argv,   // argv[0] is executable
                         const char **envp,
                         const ArchSpec *arch_spec,
                         bool stop_at_entry,
                         bool disable_aslr);
    
    static bool
    OpenFileInExternalEditor (const FileSpec &file_spec, uint32_t line_no);

    static void
    Backtrace (Stream &strm, uint32_t max_frames);
    
    static size_t
    GetEnvironment (StringList &env);
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Host_h_
