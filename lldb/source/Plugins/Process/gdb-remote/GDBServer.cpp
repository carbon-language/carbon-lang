//===-- GDBServer.cpp -------------------------------------------*- C++ -*-===//
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

#include "GDBServerLog.h"
#include "GDBRemoteSession.h"

using namespace lldb;

//----------------------------------------------------------------------
// Run loop modes which determine which run loop function will be called
//----------------------------------------------------------------------
typedef enum
{
    eDCGSRunLoopModeInvalid = 0,
    eDCGSRunLoopModeGetStartModeFromRemoteProtocol,
    eDCGSRunLoopModeInferiorAttaching,
    eDCGSRunLoopModeInferiorLaunching,
    eDCGSRunLoopModeInferiorExecuting,
    eDCGSRunLoopModeInferiorKillOrDetach,
    eDCGSRunLoopModeExit
} GSRunLoopMode;

typedef enum
{
    eLaunchFlavorDefault = 0,
    eLaunchFlavorPosixSpawn,
#if defined (__arm__)
    eLaunchFlavorSpringBoard,
#endif
    eLaunchFlavorForkExec,
} GSLaunchFlavor;

typedef lldb::shared_ptr<GDBRemoteSession> GDBRemoteSP;

typedef struct HandleBroadcastEventInfo
{
    TargetSP target_sp;
    GDBRemoteSP remote_sp;
    GSRunLoopMode mode;

    Target *
    GetTarget ()
    {
        return target_sp.get();
    }

    Process *
    GetProcess()
    {
        if (target_sp.get())
            return target_sp->GetProcess().get();
        return NULL;
    }

    GDBRemoteSession *
    GetRemote ()
    {
        return remote_sp.get();
    }

};


//----------------------------------------------------------------------
// Global Variables
//----------------------------------------------------------------------
static int g_lockdown_opt  = 0;
static int g_applist_opt = 0;
static GSLaunchFlavor g_launch_flavor = eLaunchFlavorDefault;
int g_isatty = 0;

//----------------------------------------------------------------------
// Run Loop function prototypes
//----------------------------------------------------------------------
void GSRunLoopGetStartModeFromRemote (HandleBroadcastEventInfo *info);
void GSRunLoopInferiorExecuting (HandleBroadcastEventInfo *info);


//----------------------------------------------------------------------
// Get our program path and arguments from the remote connection.
// We will need to start up the remote connection without a PID, get the
// arguments, wait for the new process to finish launching and hit its
// entry point,  and then return the run loop mode that should come next.
//----------------------------------------------------------------------
void
GSRunLoopGetStartModeFromRemote (HandleBroadcastEventInfo *info)
{
    std::string packet;

    Target *target = info->GetTarget();
    GDBRemoteSession *remote = info->GetRemote();
    if (target != NULL && remote != NULL)
    {
        // Spin waiting to get the A packet.
        while (1)
        {
            gdb_err_t err = gdb_err;
            GDBRemoteSession::PacketEnum type;

            err = remote->HandleReceivedPacket (&type);

            // check if we tried to attach to a process
            if (type == GDBRemoteSession::vattach || type == GDBRemoteSession::vattachwait)
            {
                if (err == gdb_success)
                {
                    info->mode = eDCGSRunLoopModeInferiorExecuting;
                    return;
                }
                else
                {
                    Log::STDERR ("error: attach failed.");
                    info->mode = eDCGSRunLoopModeExit;
                    return;
                }
            }

            if (err == gdb_success)
            {
                // If we got our arguments we are ready to launch using the arguments
                // and any environment variables we received.
                if (type == GDBRemoteSession::set_argv)
                {
                    info->mode = eDCGSRunLoopModeInferiorLaunching;
                    return;
                }
            }
            else if (err == gdb_not_connected)
            {
                Log::STDERR ("error: connection lost.");
                info->mode = eDCGSRunLoopModeExit;
                return;
            }
            else
            {
                // a catch all for any other gdb remote packets that failed
                GDBServerLog::LogIf (GS_LOG_MINIMAL, "%s Error getting packet.",__FUNCTION__);
                continue;
            }

            GDBServerLog::LogIf (GS_LOG_MINIMAL, "#### %s", __FUNCTION__);
        }
    }
    info->mode = eDCGSRunLoopModeExit;
}


//----------------------------------------------------------------------
// This run loop mode will wait for the process to launch and hit its
// entry point. It will currently ignore all events except for the
// process state changed event, where it watches for the process stopped
// or crash process state.
//----------------------------------------------------------------------
GSRunLoopMode
GSRunLoopLaunchInferior (HandleBroadcastEventInfo *info)
{
    // The Process stuff takes a c array, the GSContext has a vector...
    // So make up a c array.
    Target *target = info->GetTarget();
    GDBRemoteSession *remote = info->GetRemote();
    Process* process = info->GetProcess();

    if (process == NULL)
        return eDCGSRunLoopModeExit;

    GDBServerLog::LogIf (GS_LOG_MINIMAL, "%s Launching '%s'...", __FUNCTION__, target->GetExecutableModule()->GetFileSpec().GetFilename().AsCString());

    // Our launch type hasn't been set to anything concrete, so we need to
    // figure our how we are going to launch automatically.

    GSLaunchFlavor launch_flavor = g_launch_flavor;
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

    //ctx.SetLaunchFlavor(launch_flavor);

    const char *stdio_file = NULL;
    lldb::pid_t pid = process->Launch (remote->GetARGV(), remote->GetENVP(), stdio_file, stdio_file, stdio_file);

    if (pid == LLDB_INVALID_PROCESS_ID)
    {
        Log::STDERR ("error: process launch failed: %s", process->GetError().AsCString());
    }
    else
    {
        if (remote->IsConnected())
        {
            // It we are connected already, the next thing gdb will do is ask
            // whether the launch succeeded, and if not, whether there is an
            // error code.  So we need to fetch one packet from gdb before we wait
            // on the stop from the target.
            gdb_err_t err = gdb_err;
            GDBRemoteSession::PacketEnum type;

            err = remote->HandleReceivedPacket (&type);

            if (err != gdb_success)
            {
                GDBServerLog::LogIf (GS_LOG_MINIMAL, "%s Error getting packet.", __FUNCTION__);
                return eDCGSRunLoopModeExit;
            }
            if (type != GDBRemoteSession::query_launch_success)
            {
                GDBServerLog::LogIf (GS_LOG_MINIMAL, "%s Didn't get the expected qLaunchSuccess packet.", __FUNCTION__);
            }
        }
    }

    Listener listener("GSRunLoopLaunchInferior");
    listener.StartListeningForEvents (process, Process::eBroadcastBitStateChanged);
    while (process->GetID() != LLDB_INVALID_PROCESS_ID)
    {
        uint32_t event_mask = 0;
        while (listener.WaitForEvent(NULL, &event_mask))
        {
            if (event_mask & Process::eBroadcastBitStateChanged)
            {
                Event event;
                StateType event_state;
                while ((event_state = process->GetNextEvent (&event)))
                if (StateIsStoppedState(event_state))
                {
                    GDBServerLog::LogIf (GS_LOG_EVENTS, "%s process %4.4x stopped with state %s", __FUNCTION__, pid, StateAsCString(event_state));

                    switch (event_state)
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
                        return eDCGSRunLoopModeInferiorExecuting;

                    case eStateDetached:
                    case eStateExited:
                        pid = LLDB_INVALID_PROCESS_ID;
                        return eDCGSRunLoopModeExit;
                    }
                }

                if (event_state = eStateInvalid)
                    break;
            }
        }
    }

    return eDCGSRunLoopModeExit;
}


//----------------------------------------------------------------------
// This run loop mode will wait for the process to launch and hit its
// entry point. It will currently ignore all events except for the
// process state changed event, where it watches for the process stopped
// or crash process state.
//----------------------------------------------------------------------
GSRunLoopMode
GSRunLoopLaunchAttaching (HandleBroadcastEventInfo *info, lldb::pid_t& pid)
{
    Process* process = info->GetProcess();

    GDBServerLog::LogIf (GS_LOG_MINIMAL, "%s Attaching to pid %i...", __FUNCTION__, pid);
    pid = process->Attach(pid);

    if (pid == LLDB_INVALID_PROCESS_ID)
        return eDCGSRunLoopModeExit;
    return eDCGSRunLoopModeInferiorExecuting;
}

//----------------------------------------------------------------------
// Watch for signals:
// SIGINT: so we can halt our inferior. (disabled for now)
// SIGPIPE: in case our child process dies
//----------------------------------------------------------------------
lldb::pid_t g_pid;
int g_sigpipe_received = 0;
void
signal_handler(int signo)
{
    GDBServerLog::LogIf (GS_LOG_MINIMAL, "%s (%s)", __FUNCTION__, Host::GetSignalAsCString(signo));

    switch (signo)
    {
//  case SIGINT:
//      DNBProcessKill (g_pid, signo);
//      break;

    case SIGPIPE:
        g_sigpipe_received = 1;
        break;
    }
}

// Return the new run loop mode based off of the current process state
void
HandleProcessStateChange (HandleBroadcastEventInfo *info, bool initialize)
{
    Process *process = info->GetProcess();
    if (process == NULL)
    {
        info->mode = eDCGSRunLoopModeExit;
        return;
    }

    if (process->GetID() == LLDB_INVALID_PROCESS_ID)
    {
        GDBServerLog::LogIf (GS_LOG_MINIMAL, "#### %s error: pid invalid, exiting...", __FUNCTION__);
        info->mode = eDCGSRunLoopModeExit;
        return;
    }
    StateType pid_state = process->GetState ();

    GDBServerLog::LogIf (GS_LOG_MINIMAL, "%s (info, initialize=%i)  pid_state = %s", __FUNCTION__, (int)initialize, StateAsCString(pid_state));

    switch (pid_state)
    {
    case eStateInvalid:
    case eStateUnloaded:
        // Something bad happened
        info->mode = eDCGSRunLoopModeExit;
        return;

    case eStateAttaching:
    case eStateLaunching:
        info->mode = eDCGSRunLoopModeInferiorExecuting;
        return;

    case eStateSuspended:
    case eStateCrashed:
    case eStateStopped:
        if (initialize == false)
        {
            // Compare the last stop count to our current notion of a stop count
            // to make sure we don't notify more than once for a given stop.
            static uint32_t g_prev_stop_id = 0;
            uint32_t stop_id = process->GetStopID();
            bool pid_stop_count_changed = g_prev_stop_id != stop_id;
            if (pid_stop_count_changed)
            {
                info->GetRemote()->FlushSTDIO();

                if (stop_id == 1)
                {
                    GDBServerLog::LogIf (GS_LOG_MINIMAL, "%s (&remote, initialize=%i)  pid_state = %s pid_stop_count %u (old %u)) Notify??? no, first stop...", __FUNCTION__, (int)initialize, StateAsCString (pid_state), stop_id, g_prev_stop_id);
                }
                else
                {

                    GDBServerLog::LogIf (GS_LOG_MINIMAL, "%s (&remote, initialize=%i)  pid_state = %s pid_stop_count %u (old %u)) Notify??? YES!!!", __FUNCTION__, (int)initialize, StateAsCString (pid_state), stop_id, g_prev_stop_id);
                    info->GetRemote()->NotifyThatProcessStopped ();
                }
            }
            else
            {
                GDBServerLog::LogIf (GS_LOG_MINIMAL, "%s (&remote, initialize=%i)  pid_state = %s pid_stop_count %u (old %u)) Notify??? skipping...", __FUNCTION__, (int)initialize, StateAsCString (pid_state), stop_id, g_prev_stop_id);
            }
        }
        info->mode = eDCGSRunLoopModeInferiorExecuting;
        return;

    case eStateStepping:
    case eStateRunning:
        info->mode = eDCGSRunLoopModeInferiorExecuting;
        return;

    case eStateExited:
        info->GetRemote()->HandlePacket_last_signal (NULL);
        info->mode = eDCGSRunLoopModeExit;
        return;

    }

    // Catch all...
    info->mode = eDCGSRunLoopModeExit;
}

bool
CommunicationHandleBroadcastEvent (Broadcaster *broadcaster, uint32_t event_mask, void *baton)
{
    HandleBroadcastEventInfo *info = (HandleBroadcastEventInfo *)baton;
    Process *process = info->GetProcess();

    if (process == NULL)
    {
        info->mode = eDCGSRunLoopModeExit;
        return true;
    }

    if (event_mask & Communication::eBroadcastBitPacketAvailable)
    {
        if (process->IsRunning())
        {
            if (info->GetRemote()->HandleAsyncPacket() == gdb_not_connected)
                info->mode = eDCGSRunLoopModeExit;
        }
        else
        {
            if (info->GetRemote()->HandleReceivedPacket() == gdb_not_connected)
                info->mode = eDCGSRunLoopModeExit;
        }
    }
    if (event_mask & Communication::eBroadcastBitReadThreadDidExit)
    {
        info->mode = eDCGSRunLoopModeExit;
    }
    if (event_mask & Communication::eBroadcastBitDisconnected)
    {
        info->mode = eDCGSRunLoopModeExit;
    }

    return true;

}

bool
ProcessHandleBroadcastEvent (Broadcaster *broadcaster, uint32_t event_mask, void *baton)
{
    HandleBroadcastEventInfo *info = (HandleBroadcastEventInfo *)baton;
    Process *process = info->GetProcess();
    if (process == NULL)
    {
        info->mode = eDCGSRunLoopModeExit;
        return true;
    }

    if (event_mask & Process::eBroadcastBitStateChanged)
    {
        // Consume all available process events with no timeout
        Event event;
        StateType process_state;
        while ((process_state = process->GetNextEvent (&event)) != eStateInvalid)
        {
            if (StateIsStoppedState(process_state))
                info->GetRemote()->FlushSTDIO();
            HandleProcessStateChange (info, false);

            if (info->mode != eDCGSRunLoopModeInferiorExecuting)
                break;
        }
    }
    else
    if (event_mask & (Process::eBroadcastBitSTDOUT | Process::eBroadcastBitSTDERR))
    {
        info->GetRemote()->FlushSTDIO();
    }
    return true;
}

// This function handles the case where our inferior program is stopped and
// we are waiting for gdb remote protocol packets. When a packet occurs that
// makes the inferior run, we need to leave this function with a new state
// as the return code.
void
GSRunLoopInferiorExecuting (HandleBroadcastEventInfo *info)
{
    GDBServerLog::LogIf (GS_LOG_MINIMAL, "#### %s", __FUNCTION__);

    // Init our mode and set 'is_running' based on the current process state
    HandleProcessStateChange (info, true);

    uint32_t desired_mask, acquired_mask;
    Listener listener("GSRunLoopInferiorExecuting");

    desired_mask =  Communication::eBroadcastBitPacketAvailable |
                    Communication::eBroadcastBitReadThreadDidExit  |
                    Communication::eBroadcastBitDisconnected;

    acquired_mask = listener.StartListeningForEvents (&(info->GetRemote()->GetPacketComm()),
                                      desired_mask,
                                      CommunicationHandleBroadcastEvent,
                                      info);

    assert (acquired_mask == desired_mask);
    desired_mask = GDBRemotePacket::eBroadcastBitPacketAvailable;

    acquired_mask = listener.StartListeningForEvents (&(info->GetRemote()->GetPacketComm()),
                                            desired_mask,
                                            CommunicationHandleBroadcastEvent,
                                            info);

    assert (acquired_mask == desired_mask);

    desired_mask =  Process::eBroadcastBitStateChanged |
                    Process::eBroadcastBitSTDOUT |
                    Process::eBroadcastBitSTDERR ;
    acquired_mask = listener.StartListeningForEvents (info->GetProcess (),
                                      desired_mask,
                                      ProcessHandleBroadcastEvent,
                                      info);

    assert (acquired_mask == desired_mask);

    Process *process = info->GetProcess();

    while (process->IsAlive())
    {
        if (!info->GetRemote()->IsConnected())
        {
            info->mode = eDCGSRunLoopModeInferiorKillOrDetach;
            break;
        }

        // We want to make sure we consume all process state changes and have
        // whomever is notifying us to wait for us to reset the event bit before
        // continuing.
        //ctx.Events().SetResetAckMask (GSContext::event_proc_state_changed);
        uint32_t event_mask = 0;
        Broadcaster *broadcaster = listener.WaitForEvent(NULL, &event_mask);
        if (broadcaster)
        {
            listener.HandleBroadcastEvent(broadcaster, event_mask);
        }
    }
}


//----------------------------------------------------------------------
// Convenience function to set up the remote listening port
// Returns 1 for success 0 for failure.
//----------------------------------------------------------------------

static bool
StartListening (HandleBroadcastEventInfo *info, int listen_port)
{
    if (!info->GetRemote()->IsConnected())
    {
        Log::STDOUT ("Listening to port %i...\n", listen_port);
        char connect_url[256];
        snprintf(connect_url, sizeof(connect_url), "listen://%i", listen_port);

        Communication &comm = info->remote_sp->GetPacketComm();
        comm.SetConnection (new ConnectionFileDescriptor);

        if (comm.Connect (connect_url))
        {
            if (comm.StartReadThread())
                return true;

            Log::STDERR ("Failed to start the communication read thread.\n", connect_url);
            comm.Disconnect();
        }
        else
        {
            Log::STDERR ("Failed to connection to %s.\n", connect_url);
        }
        return false;
    }
    return true;
}

//----------------------------------------------------------------------
// ASL Logging callback that can be registered with DNBLogSetLogDCScriptInterpreter::Type
//----------------------------------------------------------------------
//void
//ASLLogDCScriptInterpreter::Type(void *baton, uint32_t flags, const char *format, va_list args)
//{
//    if (format == NULL)
//      return;
//    static aslmsg g_aslmsg = NULL;
//    if (g_aslmsg == NULL)
//    {
//        g_aslmsg = ::asl_new (ASL_TYPE_MSG);
//        char asl_key_sender[PATH_MAX];
//        snprintf(asl_key_sender, sizeof(asl_key_sender), "com.apple.dc-gdbserver-%g", dc_gdbserverVersionNumber);
//        ::asl_set (g_aslmsg, ASL_KEY_SENDER, asl_key_sender);
//    }
//
//    int asl_level;
//    if (flags & DNBLOG_FLAG_FATAL)        asl_level = ASL_LEVEL_CRIT;
//    else if (flags & DNBLOG_FLAG_ERROR)   asl_level = ASL_LEVEL_ERR;
//    else if (flags & DNBLOG_FLAG_WARNING) asl_level = ASL_LEVEL_WARNING;
//    else if (flags & DNBLOG_FLAG_VERBOSE) asl_level = ASL_LEVEL_WARNING; //ASL_LEVEL_INFO;
//    else                                  asl_level = ASL_LEVEL_WARNING; //ASL_LEVEL_DEBUG;
//
//    ::asl_vlog (NULL, g_aslmsg, asl_level, format, args);
//}

//----------------------------------------------------------------------
// FILE based Logging callback that can be registered with
// DNBLogSetLogDCScriptInterpreter::Type
//----------------------------------------------------------------------
void
FileLogDCScriptInterpreter::Type(void *baton, uint32_t flags, const char *format, va_list args)
{
    if (baton == NULL || format == NULL)
      return;

    ::vfprintf ((FILE *)baton, format, args);
    ::fprintf ((FILE *)baton, "\n");
}

//----------------------------------------------------------------------
// option descriptors for getopt_long()
//----------------------------------------------------------------------
static struct option g_long_options[] =
{
    { "arch",       required_argument,  NULL,               'c' },
    { "attach",     required_argument,  NULL,               'a' },
    { "debug",      no_argument,        NULL,               'g' },
    { "verbose",    no_argument,        NULL,               'v' },
    { "lockdown",   no_argument,        &g_lockdown_opt,    1   },  // short option "-k"
    { "applist",    no_argument,        &g_applist_opt,     1   },  // short option "-t"
    { "log-file",   required_argument,  NULL,               'l' },
    { "log-flags",  required_argument,  NULL,               'f' },
    { "launch",     required_argument,  NULL,               'x' },  // Valid values are "auto", "posix-spawn", "fork-exec", "springboard" (arm only)
    { "waitfor",    required_argument,  NULL,               'w' },  // Wait for a process whose namet starts with ARG
    { "waitfor-interval", required_argument,    NULL,       'i' },  // Time in usecs to wait between sampling the pid list when waiting for a process by name
    { "waitfor-duration", required_argument,    NULL,       'd' },  // The time in seconds to wait for a process to show up by name
    { NULL,         0,                  NULL,               0   }
};

extern const double dc_gdbserverVersionNumber;
int
main (int argc, char *argv[])
{
    Initialize();
    Host::ThreadCreated ("[main]");

    g_isatty = ::isatty (STDIN_FILENO);

//  signal (SIGINT, signal_handler);
    signal (SIGPIPE, signal_handler);

    Log *log = GDBServerLog::GetLogIfAllCategoriesSet(GS_LOG_ALL);
    const char *this_exe_name = argv[0];
    int i;
    int attach_pid = LLDB_INVALID_PROCESS_ID;
    for (i=0; i<argc; i++)
        GDBServerLog::LogIf(GS_LOG_DEBUG, "argv[%i] = %s", i, argv[i]);

    FILE* log_file = NULL;
    uint32_t log_flags = 0;
    // Parse our options
    int ch;
    int long_option_index = 0;
    int debug = 0;
    std::string waitfor_pid_name;           // Wait for a process that starts with this name
    std::string attach_pid_name;
    useconds_t waitfor_interval = 1000;     // Time in usecs between process lists polls when waiting for a process by name, default 1 msec.
    useconds_t waitfor_duration = 0;        // Time in seconds to wait for a process by name, 0 means wait forever.
    ArchSpec arch;
    GSRunLoopMode start_mode = eDCGSRunLoopModeExit;

    while ((ch = getopt_long(argc, argv, "a:c:d:gi:vktl:f:w:x:", g_long_options, &long_option_index)) != -1)
    {
//      DNBLogDebug("option: ch == %c (0x%2.2x) --%s%c%s\n",
//                    ch, (uint8_t)ch,
//                    g_long_options[long_option_index].name,
//                    g_long_options[long_option_index].has_arg ? '=' : ' ',
//                    optarg ? optarg : "");
        switch (ch)
        {
        case 0:   // Any optional that auto set themselves will return 0
            break;

        case 'c':
            arch.SetArch(optarg);
            if (!arch.IsValid())
            {
                Log::STDERR ("error: invalid arch string '%s'\n", optarg);
                exit (8);
            }
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
                        Log::STDERR ("error: invalid pid option '%s'\n", optarg);
                        exit (4);
                    }
                }
                else
                {
                    attach_pid_name = optarg;
                }
                start_mode = eDCGSRunLoopModeInferiorAttaching;
            }
            break;

        // --waitfor=NAME
        case 'w':
            if (optarg && optarg[0])
            {
                waitfor_pid_name = optarg;
                start_mode = eDCGSRunLoopModeInferiorAttaching;
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
                    Log::STDERR ("error: invalid waitfor-interval option value '%s'.\n", optarg);
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
                    Log::STDERR ("error: invalid waitfor-duration option value '%s'.\n", optarg);
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
                    Log::STDERR ("error: invalid TYPE for the --launch=TYPE (-x TYPE) option: '%s'\n", optarg);
                    Log::STDERR ("Valid values TYPE are:\n");
                    Log::STDERR ("  auto    Auto-detect the best launch method to use.\n");
                    Log::STDERR ("  posix   Launch the executable using posix_spawn.\n");
                    Log::STDERR ("  fork    Launch the executable using fork and exec.\n");
#if defined (__arm__)
                    Log::STDERR ("  spring  Launch the executable through Springboard.\n");
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
                    Log::STDERR ("Failed to open log file '%s' for writing: errno = %i (%s)", optarg, errno, errno_str ? errno_str : "unknown error");
                }
            }
            break;

        case 'f': // Log Flags
            if (optarg && optarg[0])
                log_flags = strtoul(optarg, NULL, 0);
            break;

        case 'g':
            debug = 1;
            //DNBLogSetDebug(1);
            break;

        case 't':
            g_applist_opt = 1;
            break;

        case 'k':
            g_lockdown_opt = 1;
            break;

        case 'v':
            //DNBLogSetVerbose(1);
            break;
        }
    }

    // Skip any options we consumed with getopt_long
    argc -= optind;
    argv += optind;

    // It is ok for us to set NULL as the logfile (this will disable any logging)

//    if (log_file != NULL)
//    {
//        DNBLogSetLogDCScriptInterpreter::Type(FileLogDCScriptInterpreter::Type, log_file);
//        // If our log file was set, yet we have no log flags, log everything!
//        if (log_flags == 0)
//            log_flags = LOG_ALL | LOG_DCGS_ALL;
//
//        DNBLogSetLogMask (log_flags);
//    }
//    else
//    {
//        // Enable DNB logging
//        DNBLogSetLogDCScriptInterpreter::Type(ASLLogDCScriptInterpreter::Type, NULL);
//        DNBLogSetLogMask (log_flags);
//
//    }

    // as long as we're dropping remotenub in as a replacement for gdbserver,
    // explicitly note that this is not gdbserver.

    Log::STDOUT ("debugserver-%g \n", dc_gdbserverVersionNumber);
    int listen_port = -1;
    if (g_lockdown_opt == 0 && g_applist_opt == 0)
    {
        // Make sure we at least have port
        if (argc < 1)
        {
            Log::STDERR ("Usage: %s host:port [program-name program-arg1 program-arg2 ...]\n", this_exe_name);
            exit (1);
        }
        // accept 'localhost:' prefix on port number

        std::string host_str;
        std::string port_str(argv[0]);

        // We just used the host:port arg...
        argc--;
        argv++;

        size_t port_idx = port_str.find(':');
        if (port_idx != std::string::npos)
        {
            host_str.assign(port_str, 0, port_idx);
            port_str.erase(0, port_idx + 1);
        }

        if (port_str.empty())
        {
            Log::STDERR ("error: no port specified\nUsage: %s host:port [program-name program-arg1 program-arg2 ...]\n", this_exe_name);
            exit (2);
        }
        else if (port_str.find_first_not_of("0123456789") != std::string::npos)
        {
            Log::STDERR ("error: port must be an integer: %s\nUsage: %s host:port [program-name program-arg1 program-arg2 ...]\n", port_str.c_str(), this_exe_name);
            exit (3);
        }
        //DNBLogDebug("host_str = '%s'  port_str = '%s'", host_str.c_str(), port_str.c_str());
        listen_port = atoi (port_str.c_str());
    }


    // We must set up some communications now.

    FileSpec exe_spec;
    if (argv[0])
        exe_spec.SetFile (argv[0]);

    HandleBroadcastEventInfo info;
    info.target_sp = TargetList::SharedList().CreateTarget(&exe_spec, &arch);
    ProcessSP process_sp (info.target_sp->CreateProcess ());
    info.remote_sp.reset (new GDBRemoteSession (process_sp));

    info.remote_sp->SetLog (log);
    StreamString sstr;
    sstr.Printf("ConnectionFileDescriptor(%s)", argv[0]);

    if (info.remote_sp.get() == NULL)
    {
        Log::STDERR ("error: failed to create a GDBRemoteSession class\n");
        return -1;
    }



    //  If we know we're waiting to attach, we don't need any of this other info.
    if (start_mode != eDCGSRunLoopModeInferiorAttaching)
    {
        if (argc == 0 || g_lockdown_opt)
        {
            if (g_lockdown_opt != 0)
            {
                // Work around for SIGPIPE crashes due to posix_spawn issue. We have to close
                // STDOUT and STDERR, else the first time we try and do any, we get SIGPIPE and
                // die as posix_spawn is doing bad things with our file descriptors at the moment.
                int null = open("/dev/null", O_RDWR);
                dup2(null, STDOUT_FILENO);
                dup2(null, STDERR_FILENO);
            }
            else if (g_applist_opt != 0)
            {
//                // List all applications we are able to see
//                std::string applist_plist;
//                int err = ListApplications(applist_plist, false, false);
//                if (err == 0)
//                {
//                    fputs (applist_plist.c_str(), stdout);
//                }
//                else
//                {
//                    Log::STDERR ("error: ListApplications returned error %i\n", err);
//                }
//                // Exit with appropriate error if we were asked to list the applications
//                // with no other args were given (and we weren't trying to do this over
//                // lockdown)
//                return err;
                return 0;
            }

            //DNBLogDebug("Get args from remote protocol...");
            start_mode = eDCGSRunLoopModeGetStartModeFromRemoteProtocol;
        }
        else
        {
            start_mode = eDCGSRunLoopModeInferiorLaunching;
            // Fill in the argv array in the context from the rest of our args.
            // Skip the name of this executable and the port number
            info.remote_sp->SetArguments (argc, argv);
        }
    }

    if (start_mode == eDCGSRunLoopModeExit)
      return -1;

    info.mode = start_mode;

    while (info.mode != eDCGSRunLoopModeExit)
    {
        switch (info.mode)
        {
        case eDCGSRunLoopModeGetStartModeFromRemoteProtocol:
 #if defined (__arm__)
            if (g_lockdown_opt)
            {
                if (!info.remote_sp->GetCommunication()->IsConnected())
                {
                    if (info.remote_sp->GetCommunication()->ConnectToService () != gdb_success)
                    {
                        Log::STDERR ("Failed to get connection from a remote gdb process.\n");
                        info.mode = eDCGSRunLoopModeExit;
                    }
                    else if (g_applist_opt != 0)
                    {
                        // List all applications we are able to see
                        std::string applist_plist;
                        if (ListApplications(applist_plist, false, false) == 0)
                        {
                            //DNBLogDebug("Task list: %s", applist_plist.c_str());

                            info.remote_sp->GetCommunication()->Write(applist_plist.c_str(), applist_plist.size());
                            // Issue a read that will never yield any data until the other side
                            // closes the socket so this process doesn't just exit and cause the
                            // socket to close prematurely on the other end and cause data loss.
                            std::string buf;
                            info.remote_sp->GetCommunication()->Read(buf);
                        }
                        info.remote_sp->GetCommunication()->Disconnect(false);
                        info.mode = eDCGSRunLoopModeExit;
                        break;
                    }
                    else
                    {
                        // Start watching for remote packets
                        info.remote_sp->StartReadRemoteDataThread();
                    }
                }
            }
            else
#endif
            {
                if (StartListening (&info, listen_port))
                    Log::STDOUT ("Got a connection, waiting for process information for launching or attaching.\n");
                else
                    info.mode = eDCGSRunLoopModeExit;
            }

            if (info.mode != eDCGSRunLoopModeExit)
                GSRunLoopGetStartModeFromRemote (&info);
            break;

        case eDCGSRunLoopModeInferiorAttaching:
            if (!waitfor_pid_name.empty())
            {
                // Set our end wait time if we are using a waitfor-duration
                // option that may have been specified

                TimeValue attach_timeout_abstime;
                if (waitfor_duration != 0)
                {
                    attach_timeout_abstime = TimeValue::Now();
                    attach_timeout_abstime.OffsetWithSeconds (waitfor_duration);
                }
                GSLaunchFlavor launch_flavor = g_launch_flavor;
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

                //ctx.SetLaunchFlavor(launch_flavor);


                lldb::pid_t pid = info.GetProcess()->Attach (waitfor_pid_name.c_str());
                if (pid == LLDB_INVALID_PROCESS_ID)
                {
                    info.GetRemote()->GetLaunchError() = info.GetProcess()->GetError();
                    Log::STDERR ("error: failed to attach to process named: \"%s\" %s", waitfor_pid_name.c_str(), info.GetRemote()->GetLaunchError().AsCString());
                    info.mode = eDCGSRunLoopModeExit;
                }
                else
                {
                    info.mode = eDCGSRunLoopModeInferiorExecuting;
                }
            }
            else if (attach_pid != LLDB_INVALID_PROCESS_ID)
            {
                Log::STDOUT ("Attaching to process %i...\n", attach_pid);
                info.mode = GSRunLoopLaunchAttaching (&info, attach_pid);
                if (info.mode != eDCGSRunLoopModeInferiorExecuting)
                {
                    const char *error_str = info.GetRemote()->GetLaunchError().AsCString();
                    Log::STDERR ("error: failed to attach process %i: %s\n", attach_pid, error_str ? error_str : "unknown error.");
                    info.mode = eDCGSRunLoopModeExit;
                }
            }
            else if (!attach_pid_name.empty ())
            {
                lldb::pid_t pid = info.GetProcess()->Attach (waitfor_pid_name.c_str());
                if (pid == LLDB_INVALID_PROCESS_ID)
                {
                    info.GetRemote()->GetLaunchError() = info.GetProcess()->GetError();
                    Log::STDERR ("error: failed to attach to process named: \"%s\" %s", waitfor_pid_name.c_str(), info.GetRemote()->GetLaunchError().AsCString());
                    info.mode = eDCGSRunLoopModeExit;
                }
                else
                {
                    info.mode = eDCGSRunLoopModeInferiorExecuting;
                }
            }
            else
            {
                Log::STDERR ("error: asked to attach with empty name and invalid PID.");
                info.mode = eDCGSRunLoopModeExit;
            }

            if (info.mode != eDCGSRunLoopModeExit)
            {
                if (StartListening (&info, listen_port))
                    Log::STDOUT ("Got a connection, waiting for debugger instructions for process %d.\n", attach_pid);
                else
                    info.mode = eDCGSRunLoopModeExit;
            }
            break;

        case eDCGSRunLoopModeInferiorLaunching:
            info.mode = GSRunLoopLaunchInferior (&info);

            if (info.mode == eDCGSRunLoopModeInferiorExecuting)
            {
                if (StartListening (&info, listen_port))
                    Log::STDOUT ("Got a connection, waiting for debugger instructions for task \"%s\".\n", argv[0]);
                else
                    info.mode = eDCGSRunLoopModeExit;
            }
            else
            {
                Log::STDERR ("error: failed to launch process %s: %s\n", argv[0], info.GetRemote()->GetLaunchError().AsCString());
            }
            break;

        case eDCGSRunLoopModeInferiorExecuting:
            GSRunLoopInferiorExecuting (&info);
            break;

        case eDCGSRunLoopModeInferiorKillOrDetach:
            {
                Process *process = info.GetProcess();
                if (process && process->IsAlive())
                {
                    process->Kill(SIGCONT);
                    process->Kill(SIGKILL);
                }
            }
            info.mode = eDCGSRunLoopModeExit;
            break;

        default:
          info.mode = eDCGSRunLoopModeExit;
        case eDCGSRunLoopModeExit:
            break;
        }
    }

    return 0;
}
