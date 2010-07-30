//===-- debugserver.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <sys/socket.h>
#include <sys/types.h>
#include <errno.h>
#include <getopt.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/sysctl.h>
#include <string>
#include <vector>
#include <asl.h>

#include "CFString.h"
#include "DNB.h"
#include "DNBLog.h"
#include "DNBTimer.h"
#include "PseudoTerminal.h"
#include "RNBContext.h"
#include "RNBServices.h"
#include "RNBSocket.h"
#include "RNBRemote.h"
#include "SysSignal.h"

// Global PID in case we get a signal and need to stop the process...
nub_process_t g_pid = INVALID_NUB_PROCESS;

//----------------------------------------------------------------------
// Run loop modes which determine which run loop function will be called
//----------------------------------------------------------------------
typedef enum
{
    eRNBRunLoopModeInvalid = 0,
    eRNBRunLoopModeGetStartModeFromRemoteProtocol,
    eRNBRunLoopModeInferiorAttaching,
    eRNBRunLoopModeInferiorLaunching,
    eRNBRunLoopModeInferiorExecuting,
    eRNBRunLoopModeExit
} RNBRunLoopMode;


//----------------------------------------------------------------------
// Global Variables
//----------------------------------------------------------------------
RNBRemoteSP g_remoteSP;
static int g_lockdown_opt  = 0;
static int g_applist_opt = 0;
static nub_launch_flavor_t g_launch_flavor = eLaunchFlavorDefault;

int g_isatty = 0;

#define RNBLogSTDOUT(fmt, ...) do { if (g_isatty) { fprintf(stdout, fmt, ## __VA_ARGS__); } else { _DNBLog(0, fmt, ## __VA_ARGS__); } } while (0)
#define RNBLogSTDERR(fmt, ...) do { if (g_isatty) { fprintf(stderr, fmt, ## __VA_ARGS__); } else { _DNBLog(0, fmt, ## __VA_ARGS__); } } while (0)

//----------------------------------------------------------------------
// Run Loop function prototypes
//----------------------------------------------------------------------
RNBRunLoopMode RNBRunLoopGetStartModeFromRemote (RNBRemoteSP &remote);
RNBRunLoopMode RNBRunLoopInferiorExecuting (RNBRemoteSP &remote);


//----------------------------------------------------------------------
// Get our program path and arguments from the remote connection.
// We will need to start up the remote connection without a PID, get the
// arguments, wait for the new process to finish launching and hit its
// entry point,  and then return the run loop mode that should come next.
//----------------------------------------------------------------------
RNBRunLoopMode
RNBRunLoopGetStartModeFromRemote (RNBRemoteSP &remoteSP)
{
    std::string packet;

    if (remoteSP.get() != NULL)
    {
        RNBRemote* remote = remoteSP.get();
        RNBContext& ctx = remote->Context();
        uint32_t event_mask = RNBContext::event_read_packet_available;

        // Spin waiting to get the A packet.
        while (1)
        {
            DNBLogThreadedIf (LOG_RNB_MAX, "%s ctx.Events().WaitForSetEvents( 0x%08x ) ...",__FUNCTION__, event_mask);
            nub_event_t set_events = ctx.Events().WaitForSetEvents(event_mask);
            DNBLogThreadedIf (LOG_RNB_MAX, "%s ctx.Events().WaitForSetEvents( 0x%08x ) => 0x%08x", __FUNCTION__, event_mask, set_events);

            if (set_events & RNBContext::event_read_packet_available)
            {
                rnb_err_t err = rnb_err;
                RNBRemote::PacketEnum type;

                err = remote->HandleReceivedPacket (&type);

                // check if we tried to attach to a process
                if (type == RNBRemote::vattach || type == RNBRemote::vattachwait)
                {
                    if (err == rnb_success)
                        return eRNBRunLoopModeInferiorExecuting;
                    else
                    {
                        RNBLogSTDERR ("error: attach failed.");
                        return eRNBRunLoopModeExit;
                    }
                }

                if (err == rnb_success)
                {
                    // If we got our arguments we are ready to launch using the arguments
                    // and any environment variables we received.
                    if (type == RNBRemote::set_argv)
                    {
                        return eRNBRunLoopModeInferiorLaunching;
                    }
                }
                else if (err == rnb_not_connected)
                {
                    RNBLogSTDERR ("error: connection lost.");
                    return eRNBRunLoopModeExit;
                }
                else
                {
                    // a catch all for any other gdb remote packets that failed
                    DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s Error getting packet.",__FUNCTION__);
                    continue;
                }

                DNBLogThreadedIf (LOG_RNB_MINIMAL, "#### %s", __FUNCTION__);
            }
            else
            {
                DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s Connection closed before getting \"A\" packet.", __FUNCTION__);
                return eRNBRunLoopModeExit;
            }
        }
    }
    return eRNBRunLoopModeExit;
}


//----------------------------------------------------------------------
// This run loop mode will wait for the process to launch and hit its
// entry point. It will currently ignore all events except for the
// process state changed event, where it watches for the process stopped
// or crash process state.
//----------------------------------------------------------------------
RNBRunLoopMode
RNBRunLoopLaunchInferior (RNBRemoteSP &remote, const char *stdio_path)
{
    RNBContext& ctx = remote->Context();

    // The Process stuff takes a c array, the RNBContext has a vector...
    // So make up a c array.

    DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s Launching '%s'...", __FUNCTION__, ctx.ArgumentAtIndex(0));

    size_t inferior_argc = ctx.ArgumentCount();
    // Initialize inferior_argv with inferior_argc + 1 NULLs
    std::vector<const char *> inferior_argv(inferior_argc + 1, NULL);

    size_t i;
    for (i = 0; i < inferior_argc; i++)
        inferior_argv[i] = ctx.ArgumentAtIndex(i);

    // Pass the environment array the same way:

    size_t inferior_envc = ctx.EnvironmentCount();
    // Initialize inferior_argv with inferior_argc + 1 NULLs
    std::vector<const char *> inferior_envp(inferior_envc + 1, NULL);

    for (i = 0; i < inferior_envc; i++)
        inferior_envp[i] = ctx.EnvironmentAtIndex(i);

    // Our launch type hasn't been set to anything concrete, so we need to
    // figure our how we are going to launch automatically.

    nub_launch_flavor_t launch_flavor = g_launch_flavor;
    if (launch_flavor == eLaunchFlavorDefault)
    {
        // Our default launch method is posix spawn
        launch_flavor = eLaunchFlavorPosixSpawn;

#if defined (__arm__)
        // Check if we have an app bundle, if so launch using SpringBoard.
        if (strstr(inferior_argv[0], ".app"))
        {
            launch_flavor = eLaunchFlavorSpringBoard;
        }
#endif
    }

    ctx.SetLaunchFlavor(launch_flavor);
    char resolved_path[PATH_MAX];

    // If we fail to resolve the path to our executable, then just use what we
    // were given and hope for the best
    if ( !DNBResolveExecutablePath (inferior_argv[0], resolved_path, sizeof(resolved_path)) )
        ::strncpy(resolved_path, inferior_argv[0], sizeof(resolved_path));

    char launch_err_str[PATH_MAX];
    launch_err_str[0] = '\0';
    nub_process_t pid = DNBProcessLaunch (resolved_path,
                                          &inferior_argv[0],
                                          &inferior_envp[0],
                                          stdio_path,
                                          launch_flavor,
                                          launch_err_str,
                                          sizeof(launch_err_str));

    g_pid = pid;

    if (pid == INVALID_NUB_PROCESS && strlen(launch_err_str) > 0)
    {
        DNBLogThreaded ("%s DNBProcessLaunch() returned error: '%s'", __FUNCTION__, launch_err_str);
        ctx.LaunchStatus().SetError(-1, DNBError::Generic);
        ctx.LaunchStatus().SetErrorString(launch_err_str);
    }
    else
        ctx.LaunchStatus().Clear();

    if (remote->Comm().IsConnected())
    {
        // It we are connected already, the next thing gdb will do is ask
        // whether the launch succeeded, and if not, whether there is an
        // error code.  So we need to fetch one packet from gdb before we wait
        // on the stop from the target.

        uint32_t event_mask = RNBContext::event_read_packet_available;
        nub_event_t set_events = ctx.Events().WaitForSetEvents(event_mask);

        if (set_events & RNBContext::event_read_packet_available)
        {
            rnb_err_t err = rnb_err;
            RNBRemote::PacketEnum type;

            err = remote->HandleReceivedPacket (&type);

            if (err != rnb_success)
            {
                DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s Error getting packet.", __FUNCTION__);
                return eRNBRunLoopModeExit;
            }
            if (type != RNBRemote::query_launch_success)
            {
                DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s Didn't get the expected qLaunchSuccess packet.", __FUNCTION__);
            }
        }
    }

    while (pid != INVALID_NUB_PROCESS)
    {
        // Wait for process to start up and hit entry point
        DNBLogThreadedIf (LOG_RNB_EVENTS, "%s DNBProcessWaitForEvent (%4.4x, eEventProcessRunningStateChanged | eEventProcessStoppedStateChanged, true, INFINITE)...", __FUNCTION__, pid);
        nub_event_t set_events = DNBProcessWaitForEvents (pid, eEventProcessRunningStateChanged | eEventProcessStoppedStateChanged, true, NULL);
        DNBLogThreadedIf (LOG_RNB_EVENTS, "%s DNBProcessWaitForEvent (%4.4x, eEventProcessRunningStateChanged | eEventProcessStoppedStateChanged, true, INFINITE) => 0x%8.8x", __FUNCTION__, pid, set_events);

        if (set_events == 0)
        {
            pid = INVALID_NUB_PROCESS;
            g_pid = pid;
        }
        else
        {
            if (set_events & (eEventProcessRunningStateChanged | eEventProcessStoppedStateChanged))
            {
                nub_state_t pid_state = DNBProcessGetState (pid);
                DNBLogThreadedIf (LOG_RNB_EVENTS, "%s process %4.4x state changed (eEventProcessStateChanged): %s", __FUNCTION__, pid, DNBStateAsString(pid_state));

                switch (pid_state)
                {
                    default:
                    case eStateInvalid:
                    case eStateUnloaded:
                    case eStateAttaching:
                    case eStateLaunching:
                    case eStateSuspended:
                        break;  // Ignore

                    case eStateRunning:
                    case eStateStepping:
                        // Still waiting to stop at entry point...
                        break;

                    case eStateStopped:
                    case eStateCrashed:
                        ctx.SetProcessID(pid);
                        return eRNBRunLoopModeInferiorExecuting;

                    case eStateDetached:
                    case eStateExited:
                        pid = INVALID_NUB_PROCESS;
                        g_pid = pid;
                        return eRNBRunLoopModeExit;
                }
            }

            DNBProcessResetEvents(pid, set_events);
        }
    }

    return eRNBRunLoopModeExit;
}


//----------------------------------------------------------------------
// This run loop mode will wait for the process to launch and hit its
// entry point. It will currently ignore all events except for the
// process state changed event, where it watches for the process stopped
// or crash process state.
//----------------------------------------------------------------------
RNBRunLoopMode
RNBRunLoopLaunchAttaching (RNBRemoteSP &remote, nub_process_t attach_pid, nub_process_t& pid)
{
    RNBContext& ctx = remote->Context();

    DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s Attaching to pid %i...", __FUNCTION__, attach_pid);
    char err_str[1024];
    pid = DNBProcessAttach (attach_pid, NULL, err_str, sizeof(err_str));
    g_pid = pid;

    if (pid == INVALID_NUB_PROCESS)
    {
        ctx.LaunchStatus().SetError(-1, DNBError::Generic);
        if (err_str[0])
            ctx.LaunchStatus().SetErrorString(err_str);
        return eRNBRunLoopModeExit;
    }
    else
    {

        ctx.SetProcessID(pid);
        return eRNBRunLoopModeInferiorExecuting;
    }
}

//----------------------------------------------------------------------
// Watch for signals:
// SIGINT: so we can halt our inferior. (disabled for now)
// SIGPIPE: in case our child process dies
//----------------------------------------------------------------------
int g_sigint_received = 0;
int g_sigpipe_received = 0;
void
signal_handler(int signo)
{
    DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s (%s)", __FUNCTION__, SysSignal::Name(signo));

    switch (signo)
    {
        case SIGINT:
            g_sigint_received++;
            if (g_pid != INVALID_NUB_PROCESS)
            {
                // Only send a SIGINT once...
                if (g_sigint_received == 1)
                {
                    switch (DNBProcessGetState (g_pid))
                    {
                        case eStateRunning:
                        case eStateStepping:
                            DNBProcessSignal (g_pid, SIGSTOP);
                            return;
                    }
                }
            }
            exit (SIGINT);
            break;

        case SIGPIPE:
            g_sigpipe_received = 1;
            break;
    }
}

// Return the new run loop mode based off of the current process state
RNBRunLoopMode
HandleProcessStateChange (RNBRemoteSP &remote, bool initialize)
{
    RNBContext& ctx = remote->Context();
    nub_process_t pid = ctx.ProcessID();

    if (pid == INVALID_NUB_PROCESS)
    {
        DNBLogThreadedIf (LOG_RNB_MINIMAL, "#### %s error: pid invalid, exiting...", __FUNCTION__);
        return eRNBRunLoopModeExit;
    }
    nub_state_t pid_state = DNBProcessGetState (pid);

    DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s (&remote, initialize=%i)  pid_state = %s", __FUNCTION__, (int)initialize, DNBStateAsString (pid_state));

    switch (pid_state)
    {
        case eStateInvalid:
        case eStateUnloaded:
            // Something bad happened
            return eRNBRunLoopModeExit;
            break;

        case eStateAttaching:
        case eStateLaunching:
            return eRNBRunLoopModeInferiorExecuting;

        case eStateSuspended:
        case eStateCrashed:
        case eStateStopped:
            // If we stop due to a signal, so clear the fact that we got a SIGINT
            // so we can stop ourselves again (but only while our inferior
            // process is running..)
            g_sigint_received = 0;
            if (initialize == false)
            {
                // Compare the last stop count to our current notion of a stop count
                // to make sure we don't notify more than once for a given stop.
                nub_size_t prev_pid_stop_count = ctx.GetProcessStopCount();
                bool pid_stop_count_changed = ctx.SetProcessStopCount(DNBProcessGetStopCount(pid));
                if (pid_stop_count_changed)
                {
                    remote->FlushSTDIO();

                    if (ctx.GetProcessStopCount() == 1)
                    {
                        DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s (&remote, initialize=%i)  pid_state = %s pid_stop_count %u (old %u)) Notify??? no, first stop...", __FUNCTION__, (int)initialize, DNBStateAsString (pid_state), ctx.GetProcessStopCount(), prev_pid_stop_count);
                    }
                    else
                    {

                        DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s (&remote, initialize=%i)  pid_state = %s pid_stop_count %u (old %u)) Notify??? YES!!!", __FUNCTION__, (int)initialize, DNBStateAsString (pid_state), ctx.GetProcessStopCount(), prev_pid_stop_count);
                        remote->NotifyThatProcessStopped ();
                    }
                }
                else
                {
                    DNBLogThreadedIf (LOG_RNB_MINIMAL, "%s (&remote, initialize=%i)  pid_state = %s pid_stop_count %u (old %u)) Notify??? skipping...", __FUNCTION__, (int)initialize, DNBStateAsString (pid_state), ctx.GetProcessStopCount(), prev_pid_stop_count);
                }
            }
            return eRNBRunLoopModeInferiorExecuting;

        case eStateStepping:
        case eStateRunning:
            return eRNBRunLoopModeInferiorExecuting;

        case eStateExited:
            remote->HandlePacket_last_signal(NULL);
            return eRNBRunLoopModeExit;

    }

    // Catch all...
    return eRNBRunLoopModeExit;
}
// This function handles the case where our inferior program is stopped and
// we are waiting for gdb remote protocol packets. When a packet occurs that
// makes the inferior run, we need to leave this function with a new state
// as the return code.
RNBRunLoopMode
RNBRunLoopInferiorExecuting (RNBRemoteSP &remote)
{
    DNBLogThreadedIf (LOG_RNB_MINIMAL, "#### %s", __FUNCTION__);
    RNBContext& ctx = remote->Context();

    // Init our mode and set 'is_running' based on the current process state
    RNBRunLoopMode mode = HandleProcessStateChange (remote, true);

    while (ctx.ProcessID() != INVALID_NUB_PROCESS)
    {

        std::string set_events_str;
        uint32_t event_mask = ctx.NormalEventBits();

        if (!ctx.ProcessStateRunning())
        {
            // Clear the stdio bits if we are not running so we don't send any async packets
            event_mask &= ~RNBContext::event_proc_stdio_available;
        }

        // We want to make sure we consume all process state changes and have
        // whomever is notifying us to wait for us to reset the event bit before
        // continuing.
        //ctx.Events().SetResetAckMask (RNBContext::event_proc_state_changed);

        DNBLogThreadedIf (LOG_RNB_EVENTS, "%s ctx.Events().WaitForSetEvents(0x%08x) ...",__FUNCTION__, event_mask);
        nub_event_t set_events = ctx.Events().WaitForSetEvents(event_mask);
        DNBLogThreadedIf (LOG_RNB_EVENTS, "%s ctx.Events().WaitForSetEvents(0x%08x) => 0x%08x (%s)",__FUNCTION__, event_mask, set_events, ctx.EventsAsString(set_events, set_events_str));

        if (set_events)
        {
            if ((set_events & RNBContext::event_proc_thread_exiting) ||
                (set_events & RNBContext::event_proc_stdio_available))
            {
                remote->FlushSTDIO();
            }

            if (set_events & RNBContext::event_read_packet_available)
            {
                // handleReceivedPacket will take care of resetting the
                // event_read_packet_available events when there are no more...
                set_events ^= RNBContext::event_read_packet_available;

                if (ctx.ProcessStateRunning())
                {
                    if (remote->HandleAsyncPacket() == rnb_not_connected)
                    {
                        // TODO: connect again? Exit?
                    }
                }
                else
                {
                    if (remote->HandleReceivedPacket() == rnb_not_connected)
                    {
                        // TODO: connect again? Exit?
                    }
                }
            }

            if (set_events & RNBContext::event_proc_state_changed)
            {
                mode = HandleProcessStateChange (remote, false);
                ctx.Events().ResetEvents(RNBContext::event_proc_state_changed);
                set_events ^= RNBContext::event_proc_state_changed;
            }

            if (set_events & RNBContext::event_proc_thread_exiting)
            {
                mode = eRNBRunLoopModeExit;
            }

            if (set_events & RNBContext::event_read_thread_exiting)
            {
                // Out remote packet receiving thread exited, exit for now.
                if (ctx.HasValidProcessID())
                {
                    // TODO: We should add code that will leave the current process
                    // in its current state and listen for another connection...
                    if (ctx.ProcessStateRunning())
                    {
                        DNBProcessKill (ctx.ProcessID());
                    }
                }
                mode = eRNBRunLoopModeExit;
            }
        }

        // Reset all event bits that weren't reset for now...
        if (set_events != 0)
            ctx.Events().ResetEvents(set_events);

        if (mode != eRNBRunLoopModeInferiorExecuting)
            break;
    }

    return mode;
}


//----------------------------------------------------------------------
// Convenience function to set up the remote listening port
// Returns 1 for success 0 for failure.
//----------------------------------------------------------------------

static int
StartListening (RNBRemoteSP remoteSP, int listen_port)
{
    if (!remoteSP->Comm().IsConnected())
    {
        RNBLogSTDOUT ("Listening to port %i...\n", listen_port);
        if (remoteSP->Comm().Listen(listen_port) != rnb_success)
        {
            RNBLogSTDERR ("Failed to get connection from a remote gdb process.\n");
            return 0;
        }
        else
        {
            remoteSP->StartReadRemoteDataThread();
        }
    }
    return 1;
}

//----------------------------------------------------------------------
// ASL Logging callback that can be registered with DNBLogSetLogCallback
//----------------------------------------------------------------------
void
ASLLogCallback(void *baton, uint32_t flags, const char *format, va_list args)
{
    if (format == NULL)
        return;
    static aslmsg g_aslmsg = NULL;
    if (g_aslmsg == NULL)
    {
        g_aslmsg = ::asl_new (ASL_TYPE_MSG);
        char asl_key_sender[PATH_MAX];
        snprintf(asl_key_sender, sizeof(asl_key_sender), "com.apple.%s-%g", DEBUGSERVER_PROGRAM_NAME, DEBUGSERVER_VERSION_NUM);
        ::asl_set (g_aslmsg, ASL_KEY_SENDER, asl_key_sender);
    }

    int asl_level;
    if (flags & DNBLOG_FLAG_FATAL)        asl_level = ASL_LEVEL_CRIT;
    else if (flags & DNBLOG_FLAG_ERROR)   asl_level = ASL_LEVEL_ERR;
    else if (flags & DNBLOG_FLAG_WARNING) asl_level = ASL_LEVEL_WARNING;
    else if (flags & DNBLOG_FLAG_VERBOSE) asl_level = ASL_LEVEL_WARNING; //ASL_LEVEL_INFO;
    else                                  asl_level = ASL_LEVEL_WARNING; //ASL_LEVEL_DEBUG;

    ::asl_vlog (NULL, g_aslmsg, asl_level, format, args);
}

//----------------------------------------------------------------------
// FILE based Logging callback that can be registered with
// DNBLogSetLogCallback
//----------------------------------------------------------------------
void
FileLogCallback(void *baton, uint32_t flags, const char *format, va_list args)
{
    if (baton == NULL || format == NULL)
        return;

    ::vfprintf ((FILE *)baton, format, args);
    ::fprintf ((FILE *)baton, "\n");
}


void
show_usage_and_exit (int exit_code)
{
    RNBLogSTDERR ("Usage:\n  %s host:port [program-name program-arg1 program-arg2 ...]\n", DEBUGSERVER_PROGRAM_NAME);
    RNBLogSTDERR ("  %s /path/file [program-name program-arg1 program-arg2 ...]\n", DEBUGSERVER_PROGRAM_NAME);
    RNBLogSTDERR ("  %s host:port --attach=<pid>\n", DEBUGSERVER_PROGRAM_NAME);
    RNBLogSTDERR ("  %s /path/file --attach=<pid>\n", DEBUGSERVER_PROGRAM_NAME);
    RNBLogSTDERR ("  %s host:port --attach=<process_name>\n", DEBUGSERVER_PROGRAM_NAME);
    RNBLogSTDERR ("  %s /path/file --attach=<process_name>\n", DEBUGSERVER_PROGRAM_NAME);
    exit (exit_code);
}


//----------------------------------------------------------------------
// option descriptors for getopt_long()
//----------------------------------------------------------------------
static struct option g_long_options[] =
{
    { "attach",             required_argument,  NULL,               'a' },
    { "debug",              no_argument,        NULL,               'g' },
    { "verbose",            no_argument,        NULL,               'v' },
    { "lockdown",           no_argument,        &g_lockdown_opt,    1   },  // short option "-k"
    { "applist",            no_argument,        &g_applist_opt,     1   },  // short option "-t"
    { "log-file",           required_argument,  NULL,               'l' },
    { "log-flags",          required_argument,  NULL,               'f' },
    { "launch",             required_argument,  NULL,               'x' },  // Valid values are "auto", "posix-spawn", "fork-exec", "springboard" (arm only)
    { "waitfor",            required_argument,  NULL,               'w' },  // Wait for a process whose name starts with ARG
    { "waitfor-interval",   required_argument,  NULL,               'i' },  // Time in usecs to wait between sampling the pid list when waiting for a process by name
    { "waitfor-duration",   required_argument,  NULL,               'd' },  // The time in seconds to wait for a process to show up by name
    { "native-regs",        no_argument,        NULL,               'r' },  // Specify to use the native registers instead of the gdb defaults for the architecture.
    { "stdio-path",         required_argument,  NULL,               's' },  // Set the STDIO path to be used when launching applications
    { "setsid",             no_argument,        NULL,               'S' },  // call setsid() to make debugserver run in its own sessions
    { NULL,                 0,                  NULL,               0   }
};


//----------------------------------------------------------------------
// main
//----------------------------------------------------------------------
int
main (int argc, char *argv[])
{
    g_isatty = ::isatty (STDIN_FILENO);

    //  ::printf ("uid=%u euid=%u gid=%u egid=%u\n",
    //            getuid(),
    //            geteuid(),
    //            getgid(),
    //            getegid());


    //    signal (SIGINT, signal_handler);
    signal (SIGPIPE, signal_handler);
    signal (SIGHUP, signal_handler);

    int i;
    int attach_pid = INVALID_NUB_PROCESS;

    FILE* log_file = NULL;
    uint32_t log_flags = 0;
    // Parse our options
    int ch;
    int long_option_index = 0;
    int use_native_registers = 0;
    int debug = 0;
    std::string compile_options;
    std::string waitfor_pid_name;           // Wait for a process that starts with this name
    std::string attach_pid_name;
    std::string stdio_path;
    useconds_t waitfor_interval = 1000;     // Time in usecs between process lists polls when waiting for a process by name, default 1 msec.
    useconds_t waitfor_duration = 0;        // Time in seconds to wait for a process by name, 0 means wait forever.

#if !defined (DNBLOG_ENABLED)
    compile_options += "(no-logging) ";
#endif

    RNBRunLoopMode start_mode = eRNBRunLoopModeExit;

    while ((ch = getopt_long(argc, argv, "a:d:gi:vktl:f:w:x:r", g_long_options, &long_option_index)) != -1)
    {
        DNBLogDebug("option: ch == %c (0x%2.2x) --%s%c%s\n",
                    ch, (uint8_t)ch,
                    g_long_options[long_option_index].name,
                    g_long_options[long_option_index].has_arg ? '=' : ' ',
                    optarg ? optarg : "");
        switch (ch)
        {
            case 0:   // Any optional that auto set themselves will return 0
                break;

            case 'a':
                if (optarg && optarg[0])
                {
                    if (isdigit(optarg[0]))
                    {
                        char *end = NULL;
                        attach_pid = strtoul(optarg, &end, 0);
                        if (end == NULL || *end != '\0')
                        {
                            RNBLogSTDERR ("error: invalid pid option '%s'\n", optarg);
                            exit (4);
                        }
                    }
                    else
                    {
                        attach_pid_name = optarg;
                    }
                    start_mode = eRNBRunLoopModeInferiorAttaching;
                }
                break;

                // --waitfor=NAME
            case 'w':
                if (optarg && optarg[0])
                {
                    waitfor_pid_name = optarg;
                    start_mode = eRNBRunLoopModeInferiorAttaching;
                }
                break;

                // --waitfor-interval=USEC
            case 'i':
                if (optarg && optarg[0])
                {
                    char *end = NULL;
                    waitfor_interval = strtoul(optarg, &end, 0);
                    if (end == NULL || *end != '\0')
                    {
                        RNBLogSTDERR ("error: invalid waitfor-interval option value '%s'.\n", optarg);
                        exit (6);
                    }
                }
                break;

                // --waitfor-duration=SEC
            case 'd':
                if (optarg && optarg[0])
                {
                    char *end = NULL;
                    waitfor_duration = strtoul(optarg, &end, 0);
                    if (end == NULL || *end != '\0')
                    {
                        RNBLogSTDERR ("error: invalid waitfor-duration option value '%s'.\n", optarg);
                        exit (7);
                    }
                }
                break;

            case 'x':
                if (optarg && optarg[0])
                {
                    if (strcasecmp(optarg, "auto") == 0)
                        g_launch_flavor = eLaunchFlavorDefault;
                    else if (strcasestr(optarg, "posix") == optarg)
                        g_launch_flavor = eLaunchFlavorPosixSpawn;
                    else if (strcasestr(optarg, "fork") == optarg)
                        g_launch_flavor = eLaunchFlavorForkExec;
#if defined (__arm__)
                    else if (strcasestr(optarg, "spring") == optarg)
                        g_launch_flavor = eLaunchFlavorSpringBoard;
#endif
                    else
                    {
                        RNBLogSTDERR ("error: invalid TYPE for the --launch=TYPE (-x TYPE) option: '%s'\n", optarg);
                        RNBLogSTDERR ("Valid values TYPE are:\n");
                        RNBLogSTDERR ("  auto    Auto-detect the best launch method to use.\n");
                        RNBLogSTDERR ("  posix   Launch the executable using posix_spawn.\n");
                        RNBLogSTDERR ("  fork    Launch the executable using fork and exec.\n");
#if defined (__arm__)
                        RNBLogSTDERR ("  spring  Launch the executable through Springboard.\n");
#endif
                        exit (5);
                    }
                }
                break;

            case 'l': // Set Log File
                if (optarg && optarg[0])
                {
                    if (strcasecmp(optarg, "stdout") == 0)
                        log_file = stdout;
                    else if (strcasecmp(optarg, "stderr") == 0)
                        log_file = stderr;
                    else
                        log_file = fopen(optarg, "w+");

                    if (log_file == NULL)
                    {
                        const char *errno_str = strerror(errno);
                        RNBLogSTDERR ("Failed to open log file '%s' for writing: errno = %i (%s)", optarg, errno, errno_str ? errno_str : "unknown error");
                    }
                }
                break;

            case 'f': // Log Flags
                if (optarg && optarg[0])
                    log_flags = strtoul(optarg, NULL, 0);
                break;

            case 'g':
                debug = 1;
                DNBLogSetDebug(1);
                break;

            case 't':
                g_applist_opt = 1;
                break;

            case 'k':
                g_lockdown_opt = 1;
                break;

            case 'r':
                use_native_registers = 1;
                break;

            case 'v':
                DNBLogSetVerbose(1);
                break;

            case 's':
                stdio_path = optarg;
                break;

            case 'S':
                // Put debugserver into a new session. Terminals group processes
                // into sessions and when a special terminal key sequences
                // (like control+c) are typed they can cause signals to go out to
                // all processes in a session. Using this --setsid (-S) option
                // will cause debugserver to run in its own sessions and be free
                // from such issues.
                //
                // This is useful when debugserver is spawned from a command
                // line application that uses debugserver to do the debugging,
                // yet that application doesn't want debugserver receiving the
                // signals sent to the session (i.e. dying when anyone hits ^C).
                setsid();
                break;
        }
    }

    // Skip any options we consumed with getopt_long
    argc -= optind;
    argv += optind;

    g_remoteSP.reset (new RNBRemote (use_native_registers));

    RNBRemote *remote = g_remoteSP.get();
    if (remote == NULL)
    {
        RNBLogSTDERR ("error: failed to create a remote connection class\n");
        return -1;
    }

    RNBContext& ctx = remote->Context();


    // It is ok for us to set NULL as the logfile (this will disable any logging)

    if (log_file != NULL)
    {
        DNBLogSetLogCallback(FileLogCallback, log_file);
        // If our log file was set, yet we have no log flags, log everything!
        if (log_flags == 0)
            log_flags = LOG_ALL | LOG_RNB_ALL;

        DNBLogSetLogMask (log_flags);
    }
    else
    {
        // Enable DNB logging
        DNBLogSetLogCallback(ASLLogCallback, NULL);
        DNBLogSetLogMask (log_flags);

    }

    if (DNBLogEnabled())
    {
        for (i=0; i<argc; i++)
            DNBLogDebug("argv[%i] = %s", i, argv[i]);
    }

    // Now that we have read in the options and enabled logging, initialize
    // the rest of RNBRemote
    RNBRemote::InitializeRegisters (use_native_registers);


    // as long as we're dropping remotenub in as a replacement for gdbserver,
    // explicitly note that this is not gdbserver.

    RNBLogSTDOUT ("%s-%g %sfor %s.\n",
                  DEBUGSERVER_PROGRAM_NAME,
                  DEBUGSERVER_VERSION_NUM,
                  compile_options.c_str(),
                  RNB_ARCH);

    int listen_port = INT32_MAX;
    char str[PATH_MAX];

    if (g_lockdown_opt == 0 && g_applist_opt == 0)
    {
        // Make sure we at least have port
        if (argc < 1)
        {
            show_usage_and_exit (1);
        }
        // accept 'localhost:' prefix on port number

        int items_scanned = ::sscanf (argv[0], "%[^:]:%i", str, &listen_port);
        if (items_scanned == 2)
        {
            DNBLogDebug("host = '%s'  port = %i", str, listen_port);
        }
        else if (argv[0][0] == '/')
        {
            listen_port = INT32_MAX;
            strncpy(str, argv[0], sizeof(str));
        }
        else
        {
            show_usage_and_exit (2);
        }

        // We just used the 'host:port' or the '/path/file' arg...
        argc--;
        argv++;

    }

    //  If we know we're waiting to attach, we don't need any of this other info.
    if (start_mode != eRNBRunLoopModeInferiorAttaching)
    {
        if (argc == 0 || g_lockdown_opt)
        {
            if (g_lockdown_opt != 0)
            {
                // Work around for SIGPIPE crashes due to posix_spawn issue.
                // We have to close STDOUT and STDERR, else the first time we
                // try and do any, we get SIGPIPE and die as posix_spawn is
                // doing bad things with our file descriptors at the moment.
                int null = open("/dev/null", O_RDWR);
                dup2(null, STDOUT_FILENO);
                dup2(null, STDERR_FILENO);
            }
            else if (g_applist_opt != 0)
            {
                // List all applications we are able to see
                std::string applist_plist;
                int err = ListApplications(applist_plist, false, false);
                if (err == 0)
                {
                    fputs (applist_plist.c_str(), stdout);
                }
                else
                {
                    RNBLogSTDERR ("error: ListApplications returned error %i\n", err);
                }
                // Exit with appropriate error if we were asked to list the applications
                // with no other args were given (and we weren't trying to do this over
                // lockdown)
                return err;
            }

            DNBLogDebug("Get args from remote protocol...");
            start_mode = eRNBRunLoopModeGetStartModeFromRemoteProtocol;
        }
        else
        {
            start_mode = eRNBRunLoopModeInferiorLaunching;
            // Fill in the argv array in the context from the rest of our args.
            // Skip the name of this executable and the port number
            for (int i = 0; i < argc; i++)
            {
                DNBLogDebug("inferior_argv[%i] = '%s'", i, argv[i]);
                ctx.PushArgument (argv[i]);
            }
        }
    }

    if (start_mode == eRNBRunLoopModeExit)
        return -1;

    RNBRunLoopMode mode = start_mode;
    char err_str[1024] = {'\0'};

    while (mode != eRNBRunLoopModeExit)
    {
        switch (mode)
        {
            case eRNBRunLoopModeGetStartModeFromRemoteProtocol:
#if defined (__arm__)
                if (g_lockdown_opt)
                {
                    if (!g_remoteSP->Comm().IsConnected())
                    {
                        if (g_remoteSP->Comm().ConnectToService () != rnb_success)
                        {
                            RNBLogSTDERR ("Failed to get connection from a remote gdb process.\n");
                            mode = eRNBRunLoopModeExit;
                        }
                        else if (g_applist_opt != 0)
                        {
                            // List all applications we are able to see
                            std::string applist_plist;
                            if (ListApplications(applist_plist, false, false) == 0)
                            {
                                DNBLogDebug("Task list: %s", applist_plist.c_str());

                                g_remoteSP->Comm().Write(applist_plist.c_str(), applist_plist.size());
                                // Issue a read that will never yield any data until the other side
                                // closes the socket so this process doesn't just exit and cause the
                                // socket to close prematurely on the other end and cause data loss.
                                std::string buf;
                                g_remoteSP->Comm().Read(buf);
                            }
                            g_remoteSP->Comm().Disconnect(false);
                            mode = eRNBRunLoopModeExit;
                            break;
                        }
                        else
                        {
                            // Start watching for remote packets
                            g_remoteSP->StartReadRemoteDataThread();
                        }
                    }
                }
                else
#endif
                    if (listen_port != INT32_MAX)
                    {
                        if (!StartListening (g_remoteSP, listen_port))
                            mode = eRNBRunLoopModeExit;
                    }
                    else if (str[0] == '/')
                    {
                        if (g_remoteSP->Comm().OpenFile (str))
                            mode = eRNBRunLoopModeExit;
                    }
                if (mode != eRNBRunLoopModeExit)
                {
                    RNBLogSTDOUT ("Got a connection, waiting for process information for launching or attaching.\n");

                    mode = RNBRunLoopGetStartModeFromRemote (g_remoteSP);
                }
                break;

            case eRNBRunLoopModeInferiorAttaching:
                if (!waitfor_pid_name.empty())
                {
                    // Set our end wait time if we are using a waitfor-duration
                    // option that may have been specified
                    struct timespec attach_timeout_abstime, *timeout_ptr = NULL;
                    if (waitfor_duration != 0)
                    {
                        DNBTimer::OffsetTimeOfDay(&attach_timeout_abstime, waitfor_duration, 0);
                        timeout_ptr = &attach_timeout_abstime;
                    }
                    nub_launch_flavor_t launch_flavor = g_launch_flavor;
                    if (launch_flavor == eLaunchFlavorDefault)
                    {
                        // Our default launch method is posix spawn
                        launch_flavor = eLaunchFlavorPosixSpawn;

#if defined (__arm__)
                        // Check if we have an app bundle, if so launch using SpringBoard.
                        if (waitfor_pid_name.find (".app") != std::string::npos)
                        {
                            launch_flavor = eLaunchFlavorSpringBoard;
                        }
#endif
                    }

                    ctx.SetLaunchFlavor(launch_flavor);

                    nub_process_t pid = DNBProcessAttachWait (waitfor_pid_name.c_str(), launch_flavor, timeout_ptr, waitfor_interval, err_str, sizeof(err_str));
                    g_pid = pid;

                    if (pid == INVALID_NUB_PROCESS)
                    {
                        ctx.LaunchStatus().SetError(-1, DNBError::Generic);
                        if (err_str[0])
                            ctx.LaunchStatus().SetErrorString(err_str);
                        RNBLogSTDERR ("error: failed to attach to process named: \"%s\" %s", waitfor_pid_name.c_str(), err_str);
                        mode = eRNBRunLoopModeExit;
                    }
                    else
                    {
                        ctx.SetProcessID(pid);
                        mode = eRNBRunLoopModeInferiorExecuting;
                    }
                }
                else if (attach_pid != INVALID_NUB_PROCESS)
                {

                    RNBLogSTDOUT ("Attaching to process %i...\n", attach_pid);
                    nub_process_t attached_pid;
                    mode = RNBRunLoopLaunchAttaching (g_remoteSP, attach_pid, attached_pid);
                    if (mode != eRNBRunLoopModeInferiorExecuting)
                    {
                        const char *error_str = remote->Context().LaunchStatus().AsString();
                        RNBLogSTDERR ("error: failed to attach process %i: %s\n", attach_pid, error_str ? error_str : "unknown error.");
                        mode = eRNBRunLoopModeExit;
                    }
                }
                else if (!attach_pid_name.empty ())
                {
                    struct timespec attach_timeout_abstime, *timeout_ptr = NULL;
                    if (waitfor_duration != 0)
                    {
                        DNBTimer::OffsetTimeOfDay(&attach_timeout_abstime, waitfor_duration, 0);
                        timeout_ptr = &attach_timeout_abstime;
                    }

                    nub_process_t pid = DNBProcessAttachByName (attach_pid_name.c_str(), timeout_ptr, err_str, sizeof(err_str));
                    g_pid = pid;
                    if (pid == INVALID_NUB_PROCESS)
                    {
                        ctx.LaunchStatus().SetError(-1, DNBError::Generic);
                        if (err_str[0])
                            ctx.LaunchStatus().SetErrorString(err_str);
                        RNBLogSTDERR ("error: failed to attach to process named: \"%s\" %s", waitfor_pid_name.c_str(), err_str);
                        mode = eRNBRunLoopModeExit;
                    }
                    else
                    {
                        ctx.SetProcessID(pid);
                        mode = eRNBRunLoopModeInferiorExecuting;
                    }

                }
                else
                {
                    RNBLogSTDERR ("error: asked to attach with empty name and invalid PID.");
                    mode = eRNBRunLoopModeExit;
                }

                if (mode != eRNBRunLoopModeExit)
                {
                    if (listen_port != INT32_MAX)
                    {
                        if (!StartListening (g_remoteSP, listen_port))
                            mode = eRNBRunLoopModeExit;
                    }
                    else if (str[0] == '/')
                    {
                        if (g_remoteSP->Comm().OpenFile (str))
                            mode = eRNBRunLoopModeExit;
                    }
                    if (mode != eRNBRunLoopModeExit)
                        RNBLogSTDOUT ("Got a connection, waiting for debugger instructions for process %d.\n", attach_pid);
                }
                break;

            case eRNBRunLoopModeInferiorLaunching:
                mode = RNBRunLoopLaunchInferior (g_remoteSP, stdio_path.empty() ? NULL : stdio_path.c_str());

                if (mode == eRNBRunLoopModeInferiorExecuting)
                {
                    if (listen_port != INT32_MAX)
                    {
                        if (!StartListening (g_remoteSP, listen_port))
                            mode = eRNBRunLoopModeExit;
                    }
                    else if (str[0] == '/')
                    {
                        if (g_remoteSP->Comm().OpenFile (str))
                            mode = eRNBRunLoopModeExit;
                    }

                    if (mode != eRNBRunLoopModeExit)
                        RNBLogSTDOUT ("Got a connection, waiting for debugger instructions.\n");
                }
                else
                {
                    const char *error_str = remote->Context().LaunchStatus().AsString();
                    RNBLogSTDERR ("error: failed to launch process %s: %s\n", argv[0], error_str ? error_str : "unknown error.");
                }
                break;

            case eRNBRunLoopModeInferiorExecuting:
                mode = RNBRunLoopInferiorExecuting(g_remoteSP);
                break;

            default:
                mode = eRNBRunLoopModeExit;
            case eRNBRunLoopModeExit:
                break;
        }
    }

    g_remoteSP->StopReadRemoteDataThread ();
    g_remoteSP->Context().SetProcessID(INVALID_NUB_PROCESS);

    return 0;
}
