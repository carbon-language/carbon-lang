//===-- lldb-gdbserver.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <signal.h>
#include <unistd.h>
#endif

// C++ Includes

// Other libraries and framework includes
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/ConnectionMachPort.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/Pipe.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServer.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"

#ifndef LLGS_PROGRAM_NAME
#define LLGS_PROGRAM_NAME "lldb-gdbserver"
#endif

#ifndef LLGS_VERSION_STR
#define LLGS_VERSION_STR "local_build"
#endif

using namespace lldb;
using namespace lldb_private;

// lldb-gdbserver state

namespace
{
HostThread s_listen_thread;
    std::unique_ptr<ConnectionFileDescriptor> s_listen_connection_up;
    std::string s_listen_url;
}

//----------------------------------------------------------------------
// option descriptors for getopt_long_only()
//----------------------------------------------------------------------

int g_debug = 0;
int g_verbose = 0;

static struct option g_long_options[] =
{
    { "debug",              no_argument,        &g_debug,           1   },
    { "platform",           required_argument,  NULL,               'p' },
    { "verbose",            no_argument,        &g_verbose,         1   },
    { "lldb-command",       required_argument,  NULL,               'c' },
    { "log-file",           required_argument,  NULL,               'l' },
    { "log-flags",          required_argument,  NULL,               'f' },
    { "attach",             required_argument,  NULL,               'a' },
    { "named-pipe",         required_argument,  NULL,               'P' },
    { "native-regs",        no_argument,        NULL,               'r' },  // Specify to use the native registers instead of the gdb defaults for the architecture.  NOTE: this is a do-nothing arg as it's behavior is default now.  FIXME remove call from lldb-platform.
    { "reverse-connect",    no_argument,        NULL,               'R' },  // Specifies that llgs attaches to the client address:port rather than llgs listening for a connection from address on port.
    { "setsid",             no_argument,        NULL,               'S' },  // Call setsid() to make llgs run in its own session.
    { NULL,                 0,                  NULL,               0   }
};


//----------------------------------------------------------------------
// Watch for signals
//----------------------------------------------------------------------
static int g_sigpipe_received = 0;
static int g_sighup_received_count = 0;

#ifndef _WIN32

void
signal_handler(int signo)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    fprintf (stderr, "lldb-gdbserver:%s received signal %d\n", __FUNCTION__, signo);
    if (log)
        log->Printf ("lldb-gdbserver:%s received signal %d", __FUNCTION__, signo);

    switch (signo)
    {
    case SIGPIPE:
        g_sigpipe_received = 1;
        break;
    case SIGHUP:
        ++g_sighup_received_count;

        // For now, swallow SIGHUP.
        if (log)
            log->Printf ("lldb-gdbserver:%s swallowing SIGHUP (receive count=%d)", __FUNCTION__, g_sighup_received_count);
        signal (SIGHUP, signal_handler);
        break;
    }
}
#endif // #ifndef _WIN32

static void
display_usage (const char *progname)
{
    fprintf(stderr, "Usage:\n  %s [--log-file log-file-path] [--log-flags flags] [--lldb-command command]* [--platform platform_name] [--setsid] [--named-pipe named-pipe-path] [--native-regs] [--attach pid] [[HOST]:PORT] "
            "[-- PROGRAM ARG1 ARG2 ...]\n", progname);
    exit(0);
}

static void
dump_available_platforms (FILE *output_file)
{
    fprintf (output_file, "Available platform plugins:\n");
    for (int i = 0; ; ++i)
    {
        const char *plugin_name = PluginManager::GetPlatformPluginNameAtIndex (i);
        const char *plugin_desc = PluginManager::GetPlatformPluginDescriptionAtIndex (i);

        if (!plugin_name || !plugin_desc)
            break;

        fprintf (output_file, "%s\t%s\n", plugin_name, plugin_desc);
    }

    if ( Platform::GetHostPlatform () )
    {
        // add this since the default platform doesn't necessarily get registered by
        // the plugin name (e.g. 'host' doesn't show up as a
        // registered platform plugin even though it's the default).
        fprintf (output_file, "%s\tDefault platform for this host.\n", Platform::GetHostPlatform ()->GetPluginName ().AsCString ());
    }
}

static void
run_lldb_commands (const lldb::DebuggerSP &debugger_sp, const std::vector<std::string> &lldb_commands)
{
    for (const auto &lldb_command : lldb_commands)
    {
        printf("(lldb) %s\n", lldb_command.c_str ());

        lldb_private::CommandReturnObject result;
        debugger_sp->GetCommandInterpreter ().HandleCommand (lldb_command.c_str (), eLazyBoolNo, result);
        const char *output = result.GetOutputData ();
        if (output && output[0])
            puts (output);
    }
}

static lldb::PlatformSP
setup_platform (const std::string &platform_name)
{
    lldb::PlatformSP platform_sp;

    if (platform_name.empty())
    {
        printf ("using the default platform: ");
        platform_sp = Platform::GetHostPlatform ();
        printf ("%s\n", platform_sp->GetPluginName ().AsCString ());
        return platform_sp;
    }

    Error error;
    platform_sp = Platform::Create (lldb_private::ConstString(platform_name), error);
    if (error.Fail ())
    {
        // the host platform isn't registered with that name (at
        // least, not always.  Check if the given name matches
        // the default platform name.  If so, use it.
        if ( Platform::GetHostPlatform () && ( Platform::GetHostPlatform ()->GetPluginName () == ConstString (platform_name.c_str()) ) )
        {
            platform_sp = Platform::GetHostPlatform ();
        }
        else
        {
            fprintf (stderr, "error: failed to create platform with name '%s'\n", platform_name.c_str());
            dump_available_platforms (stderr);
            exit (1);
        }
    }
    printf ("using platform: %s\n", platform_name.c_str ());

    return platform_sp;
}

void
handle_attach_to_pid (GDBRemoteCommunicationServer &gdb_server, lldb::pid_t pid)
{
    Error error = gdb_server.AttachToProcess (pid);
    if (error.Fail ())
    {
        fprintf (stderr, "error: failed to attach to pid %" PRIu64 ": %s\n", pid, error.AsCString());
        exit(1);
    }
}

void
handle_attach_to_process_name (GDBRemoteCommunicationServer &gdb_server, const std::string &process_name)
{
    // FIXME implement.
}

void
handle_attach (GDBRemoteCommunicationServer &gdb_server, const std::string &attach_target)
{
    assert (!attach_target.empty () && "attach_target cannot be empty");

    // First check if the attach_target is convertable to a long. If so, we'll use it as a pid.
    char *end_p = nullptr;
    const long int pid = strtol (attach_target.c_str (), &end_p, 10);

    // We'll call it a match if the entire argument is consumed.
    if (end_p && static_cast<size_t> (end_p - attach_target.c_str ()) == attach_target.size ())
        handle_attach_to_pid (gdb_server, static_cast<lldb::pid_t> (pid));
    else
        handle_attach_to_process_name (gdb_server, attach_target);
}

void
handle_launch (GDBRemoteCommunicationServer &gdb_server, int argc, const char *const argv[])
{
    Error error;
    error = gdb_server.SetLaunchArguments (argv, argc);
    if (error.Fail ())
    {
        fprintf (stderr, "error: failed to set launch args for '%s': %s\n", argv[0], error.AsCString());
        exit(1);
    }

    unsigned int launch_flags = eLaunchFlagStopAtEntry | eLaunchFlagDebug;

    error = gdb_server.SetLaunchFlags (launch_flags);
    if (error.Fail ())
    {
        fprintf (stderr, "error: failed to set launch flags for '%s': %s\n", argv[0], error.AsCString());
        exit(1);
    }

    error = gdb_server.LaunchProcess ();
    if (error.Fail ())
    {
        fprintf (stderr, "error: failed to launch '%s': %s\n", argv[0], error.AsCString());
        exit(1);
    }
}

static lldb::thread_result_t
ListenThread (lldb::thread_arg_t /* arg */)
{
    Error error;

    if (s_listen_connection_up)
    {
        // Do the listen on another thread so we can continue on...
        if (s_listen_connection_up->Connect(s_listen_url.c_str(), &error) != eConnectionStatusSuccess)
            s_listen_connection_up.reset();
    }
    return nullptr;
}

static Error
StartListenThread (const char *hostname, uint16_t port)
{
    Error error;
    if (s_listen_thread.IsJoinable())
    {
        error.SetErrorString("listen thread already running");
    }
    else
    {
        char listen_url[512];
        if (hostname && hostname[0])
            snprintf(listen_url, sizeof(listen_url), "listen://%s:%i", hostname, port);
        else
            snprintf(listen_url, sizeof(listen_url), "listen://%i", port);

        s_listen_url = listen_url;
        s_listen_connection_up.reset (new ConnectionFileDescriptor ());
        s_listen_thread = ThreadLauncher::LaunchThread(listen_url, ListenThread, nullptr, &error);
    }
    return error;
}

static bool
JoinListenThread ()
{
    if (s_listen_thread.IsJoinable())
        s_listen_thread.Join(nullptr);
    return true;
}

Error
writePortToPipe (const char *const named_pipe_path, const uint16_t port)
{
    Pipe port_name_pipe;
    // Wait for 10 seconds for pipe to be opened.
    auto error = port_name_pipe.OpenAsWriterWithTimeout (named_pipe_path, false, std::chrono::microseconds (10 * 1000000));
    if (error.Fail ())
        return error;

    char port_str[64];
    const auto port_str_len = ::snprintf (port_str, sizeof (port_str), "%u", port);

    size_t bytes_written = 0;
    // Write the port number as a C string with the NULL terminator.
    return port_name_pipe.Write (port_str, port_str_len + 1, bytes_written);
}

void
ConnectToRemote (GDBRemoteCommunicationServer &gdb_server, bool reverse_connect, const char *const host_and_port, const char *const progname, const char *const named_pipe_path)
{
    Error error;

    if (host_and_port && host_and_port[0])
    {
        // Parse out host and port.
        std::string final_host_and_port;
        std::string connection_host;
        std::string connection_port;
        uint32_t connection_portno = 0;

        // If host_and_port starts with ':', default the host to be "localhost" and expect the remainder to be the port.
        if (host_and_port[0] == ':')
            final_host_and_port.append ("localhost");
        final_host_and_port.append (host_and_port);

        const std::string::size_type colon_pos = final_host_and_port.find (':');
        if (colon_pos != std::string::npos)
        {
            connection_host = final_host_and_port.substr (0, colon_pos);
            connection_port = final_host_and_port.substr (colon_pos + 1);
            connection_portno = Args::StringToUInt32 (connection_port.c_str (), 0);
        }
        else
        {
            fprintf (stderr, "failed to parse host and port from connection string '%s'\n", final_host_and_port.c_str ());
            display_usage (progname);
            exit (1);
        }

        if (reverse_connect)
        {
            // llgs will connect to the gdb-remote client.

            // Ensure we have a port number for the connection.
            if (connection_portno == 0)
            {
                fprintf (stderr, "error: port number must be specified on when using reverse connect");
                exit (1);
            }

            // Build the connection string.
            char connection_url[512];
            snprintf(connection_url, sizeof(connection_url), "connect://%s", final_host_and_port.c_str ());

            // Create the connection.
            std::unique_ptr<ConnectionFileDescriptor> connection_up (new ConnectionFileDescriptor ());
            connection_up.reset (new ConnectionFileDescriptor ());
            auto connection_result = connection_up->Connect (connection_url, &error);
            if (connection_result != eConnectionStatusSuccess)
            {
                fprintf (stderr, "error: failed to connect to client at '%s' (connection status: %d)", connection_url, static_cast<int> (connection_result));
                exit (-1);
            }
            if (error.Fail ())
            {
                fprintf (stderr, "error: failed to connect to client at '%s': %s", connection_url, error.AsCString ());
                exit (-1);
            }

            // We're connected.
            printf ("Connection established.\n");
            gdb_server.SetConnection (connection_up.release());
        }
        else
        {
            // llgs will listen for connections on the given port from the given address.
            // Start the listener on a new thread.  We need to do this so we can resolve the
            // bound listener port.
            StartListenThread(connection_host.c_str (), static_cast<uint16_t> (connection_portno));
            printf ("Listening to port %s for a connection from %s...\n", connection_port.c_str (), connection_host.c_str ());

            // If we have a named pipe to write the port number back to, do that now.
            if (named_pipe_path && named_pipe_path[0] && connection_portno == 0)
            {
                const uint16_t bound_port = s_listen_connection_up->GetListeningPort (10);
                if (bound_port > 0)
                {
                    error = writePortToPipe (named_pipe_path, bound_port);
                    if (error.Fail ())
                    {
                        fprintf (stderr, "failed to write to the named pipe \'%s\': %s", named_pipe_path, error.AsCString());
                    }
                }
                else
                {
                    fprintf (stderr, "unable to get the bound port for the listening connection\n");
                }
            }

            // Join the listener thread.
            if (!JoinListenThread ())
            {
                fprintf (stderr, "failed to join the listener thread\n");
                display_usage (progname);
                exit (1);
            }

            // Ensure we connected.
            if (s_listen_connection_up)
            {
                printf ("Connection established.\n");
                gdb_server.SetConnection (s_listen_connection_up.release());
            }
            else
            {
                fprintf (stderr, "failed to connect to '%s': %s\n", final_host_and_port.c_str (), error.AsCString ());
                display_usage (progname);
                exit (1);
            }
        }
    }

    if (gdb_server.IsConnected())
    {
        // After we connected, we need to get an initial ack from...
        if (gdb_server.HandshakeWithClient(&error))
        {
            // We'll use a half a second timeout interval so that an exit conditions can
            // be checked that often.
            const uint32_t TIMEOUT_USEC = 500000;

            bool interrupt = false;
            bool done = false;
            while (!interrupt && !done && (g_sighup_received_count < 2))
            {
                const GDBRemoteCommunication::PacketResult result = gdb_server.GetPacketAndSendResponse (TIMEOUT_USEC, error, interrupt, done);
                if ((result != GDBRemoteCommunication::PacketResult::Success) &&
                    (result != GDBRemoteCommunication::PacketResult::ErrorReplyTimeout))
                {
                    // We're bailing out - we only support successful handling and timeouts.
                    fprintf(stderr, "leaving packet loop due to PacketResult %d\n", result);
                    break;
                }
            }

            if (error.Fail())
            {
                fprintf(stderr, "error: %s\n", error.AsCString());
            }
        }
        else
        {
            fprintf(stderr, "error: handshake with client failed\n");
        }
    }
    else
    {
        fprintf (stderr, "no connection information provided, unable to run\n");
        display_usage (progname);
        exit (1);
    }
}

//----------------------------------------------------------------------
// main
//----------------------------------------------------------------------
int
main (int argc, char *argv[])
{
#ifndef _WIN32
    // Setup signal handlers first thing.
    signal (SIGPIPE, signal_handler);
    signal (SIGHUP, signal_handler);
#endif

    const char *progname = argv[0];
    int long_option_index = 0;
    StreamSP log_stream_sp;
    Args log_args;
    Error error;
    int ch;
    std::string platform_name;
    std::string attach_target;
    std::string named_pipe_path;
    bool reverse_connect = false;

    Debugger::Initialize (NULL);

    lldb::DebuggerSP debugger_sp = Debugger::CreateInstance ();

    debugger_sp->SetInputFileHandle(stdin, false);
    debugger_sp->SetOutputFileHandle(stdout, false);
    debugger_sp->SetErrorFileHandle(stderr, false);

    // ProcessLaunchInfo launch_info;
    ProcessAttachInfo attach_info;

    bool show_usage = false;
    int option_error = 0;
#if __GLIBC__
    optind = 0;
#else
    optreset = 1;
    optind = 1;
#endif

    std::string short_options(OptionParser::GetShortOptionString(g_long_options));

    std::vector<std::string> lldb_commands;

    while ((ch = getopt_long_only(argc, argv, short_options.c_str(), g_long_options, &long_option_index)) != -1)
    {
        switch (ch)
        {
        case 0:   // Any optional that auto set themselves will return 0
            break;

        case 'l': // Set Log File
            if (optarg && optarg[0])
            {
                if ((strcasecmp(optarg, "stdout") == 0) || (strcmp(optarg, "/dev/stdout") == 0))
                {
                    log_stream_sp.reset (new StreamFile (stdout, false));
                }
                else if ((strcasecmp(optarg, "stderr") == 0) || (strcmp(optarg, "/dev/stderr") == 0))
                {
                    log_stream_sp.reset (new StreamFile (stderr, false));
                }
                else
                {
                    FILE *log_file = fopen(optarg, "w");
                    if (log_file)
                    {
                        setlinebuf(log_file);
                        log_stream_sp.reset (new StreamFile (log_file, true));
                    }
                    else
                    {
                        const char *errno_str = strerror(errno);
                        fprintf (stderr, "Failed to open log file '%s' for writing: errno = %i (%s)", optarg, errno, errno_str ? errno_str : "unknown error");
                    }

                }
            }
            break;

        case 'f': // Log Flags
            if (optarg && optarg[0])
                log_args.AppendArgument(optarg);
            break;

        case 'c': // lldb commands
            if (optarg && optarg[0])
                lldb_commands.push_back(optarg);
            break;

        case 'p': // platform name
            if (optarg && optarg[0])
                platform_name = optarg;
            break;

        case 'P': // named pipe
            if (optarg && optarg[0])
                named_pipe_path = optarg;
            break;

        case 'r':
            // Do nothing, native regs is the default these days
            break;

        case 'R':
            reverse_connect = true;
            break;

#ifndef _WIN32
        case 'S':
            // Put llgs into a new session. Terminals group processes
            // into sessions and when a special terminal key sequences
            // (like control+c) are typed they can cause signals to go out to
            // all processes in a session. Using this --setsid (-S) option
            // will cause debugserver to run in its own sessions and be free
            // from such issues.
            //
            // This is useful when llgs is spawned from a command
            // line application that uses llgs to do the debugging,
            // yet that application doesn't want llgs receiving the
            // signals sent to the session (i.e. dying when anyone hits ^C).
            {
                const ::pid_t new_sid = setsid();
                if (new_sid == -1)
                {
                    const char *errno_str = strerror(errno);
                    fprintf (stderr, "failed to set new session id for %s (%s)\n", LLGS_PROGRAM_NAME, errno_str ? errno_str : "<no error string>");
                }
            }
            break;
#endif

        case 'a': // attach {pid|process_name}
            if (optarg && optarg[0])
                attach_target = optarg;
                break;

        case 'h':   /* fall-through is intentional */
        case '?':
            show_usage = true;
            break;
        }
    }

    if (show_usage || option_error)
    {
        display_usage(progname);
        exit(option_error);
    }

    if (log_stream_sp)
    {
        if (log_args.GetArgumentCount() == 0)
            log_args.AppendArgument("default");
        ProcessGDBRemoteLog::EnableLog (log_stream_sp, 0,log_args.GetConstArgumentVector(), log_stream_sp.get());
    }

    // Skip any options we consumed with getopt_long_only.
    argc -= optind;
    argv += optind;

    if (argc == 0)
    {
        display_usage(progname);
        exit(255);
    }

    // Run any commands requested.
    run_lldb_commands (debugger_sp, lldb_commands);

    // Setup the platform that GDBRemoteCommunicationServer will use.
    lldb::PlatformSP platform_sp = setup_platform (platform_name);

    const bool is_platform = false;
    GDBRemoteCommunicationServer gdb_server (is_platform, platform_sp, debugger_sp);

    const char *const host_and_port = argv[0];
    argc -= 1;
    argv += 1;

    // Any arguments left over are for the the program that we need to launch. If there
    // are no arguments, then the GDB server will start up and wait for an 'A' packet
    // to launch a program, or a vAttach packet to attach to an existing process, unless
    // explicitly asked to attach with the --attach={pid|program_name} form.
    if (!attach_target.empty ())
        handle_attach (gdb_server, attach_target);
    else if (argc > 0)
        handle_launch (gdb_server, argc, argv);

    // Print version info.
    printf("%s-%s", LLGS_PROGRAM_NAME, LLGS_VERSION_STR);

    ConnectToRemote (gdb_server, reverse_connect, host_and_port, progname, named_pipe_path.c_str ());

    Debugger::Terminate ();

    fprintf(stderr, "lldb-gdbserver exiting...\n");

    return 0;
}
