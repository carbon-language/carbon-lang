//===-- lldb-platform.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

// C Includes
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// C++ Includes

// Other libraries and framework includes
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/ConnectionMachPort.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Host/OptionParser.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServer.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// option descriptors for getopt_long_only()
//----------------------------------------------------------------------

int g_debug = 0;
int g_verbose = 0;
int g_stay_alive = 0;

static struct option g_long_options[] =
{
    { "debug",              no_argument,        &g_debug,           1   },
    { "verbose",            no_argument,        &g_verbose,         1   },
    { "stay-alive",         no_argument,        &g_stay_alive,      1   },
    { "log-file",           required_argument,  NULL,               'l' },
    { "log-flags",          required_argument,  NULL,               'f' },
    { "listen",             required_argument,  NULL,               'L' },
    { "port-offset",        required_argument,  NULL,               'p' },
    { "gdbserver-port",     required_argument,  NULL,               'P' },
    { "min-gdbserver-port", required_argument,  NULL,               'm' },
    { "max-gdbserver-port", required_argument,  NULL,               'M' },
    { NULL,                 0,                  NULL,               0   }
};

#if defined (__APPLE__)
#define LOW_PORT    (IPPORT_RESERVED)
#define HIGH_PORT   (IPPORT_HIFIRSTAUTO)
#else
#define LOW_PORT    (1024u)
#define HIGH_PORT   (49151u)
#endif


//----------------------------------------------------------------------
// Watch for signals
//----------------------------------------------------------------------
void
signal_handler(int signo)
{
    switch (signo)
    {
    case SIGHUP:
        // Use SIGINT first, if that does not work, use SIGHUP as a last resort.
        // And we should not call exit() here because it results in the global destructors
        // to be invoked and wreaking havoc on the threads still running.
        Host::SystemLog(Host::eSystemLogWarning, "SIGHUP received, exiting lldb-platform...\n");
        abort();
        break;
    }
}

static void
display_usage (const char *progname)
{
    fprintf(stderr, "Usage:\n  %s [--log-file log-file-path] [--log-flags flags] --listen port\n", progname);
    exit(0);
}

//----------------------------------------------------------------------
// main
//----------------------------------------------------------------------
int
main (int argc, char *argv[])
{
    const char *progname = argv[0];
    signal (SIGPIPE, SIG_IGN);
    signal (SIGHUP, signal_handler);
    int long_option_index = 0;
    StreamSP log_stream_sp;
    Args log_args;
    Error error;
    std::string listen_host_port;
    int ch;
    Debugger::Initialize();
    
    GDBRemoteCommunicationServer::PortMap gdbserver_portmap;
    int min_gdbserver_port = 0;
    int max_gdbserver_port = 0;
    uint16_t port_offset = 0;
    
    bool show_usage = false;
    int option_error = 0;
    // Enable LLDB log channels...
    StreamSP stream_sp (new StreamFile(stdout, false));
    const char *log_channels[] = { "platform", "host", "process", NULL };
    EnableLog (stream_sp, 0, log_channels, NULL);
    
    std::string short_options(OptionParser::GetShortOptionString(g_long_options));
                            
#if __GLIBC__
    optind = 0;
#else
    optreset = 1;
    optind = 1;
#endif

    while ((ch = getopt_long_only(argc, argv, short_options.c_str(), g_long_options, &long_option_index)) != -1)
    {
//        DNBLogDebug("option: ch == %c (0x%2.2x) --%s%c%s\n",
//                    ch, (uint8_t)ch,
//                    g_long_options[long_option_index].name,
//                    g_long_options[long_option_index].has_arg ? '=' : ' ',
//                    optarg ? optarg : "");
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
        
        case 'L':
            listen_host_port.append (optarg);
            break;

        case 'p':
            {
                char *end = NULL;
                long tmp_port_offset = strtoul(optarg, &end, 0);
                if (end && *end == '\0')
                {
                    if (LOW_PORT <= tmp_port_offset && tmp_port_offset <= HIGH_PORT)
                    {
                        port_offset = (uint16_t)tmp_port_offset;
                    }
                    else
                    {
                        fprintf (stderr, "error: port offset %li is not in the valid user port range of %u - %u\n", tmp_port_offset, LOW_PORT, HIGH_PORT);
                        option_error = 5;
                    }
                }
                else
                {
                    fprintf (stderr, "error: invalid port offset string %s\n", optarg);
                    option_error = 4;
                }
            }
            break;
                
        case 'P':
        case 'm':
        case 'M':
            {
                char *end = NULL;
                long portnum = strtoul(optarg, &end, 0);
                if (end && *end == '\0')
                {
                    if (LOW_PORT <= portnum && portnum <= HIGH_PORT)
                    {
                        if (ch  == 'P')
                            gdbserver_portmap[(uint16_t)portnum] = LLDB_INVALID_PROCESS_ID;
                        else if (ch == 'm')
                            min_gdbserver_port = portnum;
                        else
                            max_gdbserver_port = portnum;
                    }
                    else
                    {
                        fprintf (stderr, "error: port number %li is not in the valid user port range of %u - %u\n", portnum, LOW_PORT, HIGH_PORT);
                        option_error = 1;
                    }
                }
                else
                {
                    fprintf (stderr, "error: invalid port number string %s\n", optarg);
                    option_error = 2;
                }
            }
            break;
            
        case 'h':   /* fall-through is intentional */
        case '?':
            show_usage = true;
            break;
        }
    }
    
    // Make a port map for a port range that was specified.
    if (min_gdbserver_port < max_gdbserver_port)
    {
        for (uint16_t port = min_gdbserver_port; port < max_gdbserver_port; ++port)
            gdbserver_portmap[port] = LLDB_INVALID_PROCESS_ID;
    }
    else if (min_gdbserver_port != max_gdbserver_port)
    {
        fprintf (stderr, "error: --min-gdbserver-port (%u) is greater than --max-gdbserver-port (%u)\n", min_gdbserver_port, max_gdbserver_port);
        option_error = 3;
        
    }

    // Print usage and exit if no listening port is specified.
    if (listen_host_port.empty())
        show_usage = true;
    
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

    // Skip any options we consumed with getopt_long_only
    argc -= optind;
    argv += optind;


    do {
        GDBRemoteCommunicationServer gdb_server (true);
        
        if (port_offset > 0)
            gdb_server.SetPortOffset(port_offset);

        if (!gdbserver_portmap.empty())
        {
            gdb_server.SetPortMap(std::move(gdbserver_portmap));
        }

        if (!listen_host_port.empty())
        {
            std::unique_ptr<ConnectionFileDescriptor> conn_ap(new ConnectionFileDescriptor());
            if (conn_ap.get())
            {
                for (int j = 0; j < listen_host_port.size(); j++)
                {
                    char c = listen_host_port[j];
                    if (c > '9' || c < '0')
                        printf("WARNING: passing anything but a number as argument to --listen will most probably make connecting impossible.\n");
                }
                std::auto_ptr<ConnectionFileDescriptor> conn_ap(new ConnectionFileDescriptor());
                if (conn_ap.get())
                {
                    std::string connect_url ("listen://");
                    connect_url.append(listen_host_port.c_str());

                    printf ("Listening for a connection on %s...\n", listen_host_port.c_str());
                    if (conn_ap->Connect(connect_url.c_str(), &error) == eConnectionStatusSuccess)
                    {
                        printf ("Connection established.\n");
                        gdb_server.SetConnection (conn_ap.release());
                    }
                }
            }

            if (gdb_server.IsConnected())
            {
                // After we connected, we need to get an initial ack from...
                if (gdb_server.HandshakeWithClient(&error))
                {
                    bool interrupt = false;
                    bool done = false;
                    while (!interrupt && !done)
                    {
                        if (!gdb_server.GetPacketAndSendResponse (UINT32_MAX, error, interrupt, done))
                            break;
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
        }
    } while (g_stay_alive);

    Debugger::Terminate();

    fprintf(stderr, "lldb-platform exiting...\n");

    return 0;
}
