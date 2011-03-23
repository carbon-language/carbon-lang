//===-- lldb-platform.cpp ---------------------------------------*- C++ -*-===//
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
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// C++ Includes

// Other libraries and framework includes
#include "lldb/Core/Error.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "GDBRemoteCommunicationServer.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// option descriptors for getopt_long()
//----------------------------------------------------------------------

int g_debug = 0;
int g_verbose = 0;

static struct option g_long_options[] =
{
    { "debug",              no_argument,        &g_debug,           1   },
    { "verbose",            no_argument,        &g_verbose,         1   },
    { "log-file",           required_argument,  NULL,               'l' },
    { "log-flags",          required_argument,  NULL,               'f' },
    { "listen",             required_argument,  NULL,               'L' },
    { NULL,                 0,                  NULL,               0   }
};

//----------------------------------------------------------------------
// Watch for signals
//----------------------------------------------------------------------
int g_sigpipe_received = 0;
void
signal_handler(int signo)
{
    switch (signo)
    {
    case SIGPIPE:
        g_sigpipe_received = 1;
        break;
    }
}

//----------------------------------------------------------------------
// main
//----------------------------------------------------------------------
int
main (int argc, char *argv[])
{
    signal (SIGPIPE, signal_handler);
    int long_option_index = 0;
    FILE* log_file = NULL;
    uint32_t log_flags = 0;
    std::string connect_url;
    char ch;

    while ((ch = getopt_long(argc, argv, "l:f:", g_long_options, &long_option_index)) != -1)
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
                if (strcasecmp(optarg, "stdout") == 0)
                    log_file = stdout;
                else if (strcasecmp(optarg, "stderr") == 0)
                    log_file = stderr;
                else
                {
                    log_file = fopen(optarg, "w");
                    if (log_file != NULL)
                        setlinebuf(log_file);
                }
                
                if (log_file == NULL)
                {
                    const char *errno_str = strerror(errno);
                    fprintf (stderr, "Failed to open log file '%s' for writing: errno = %i (%s)", optarg, errno, errno_str ? errno_str : "unknown error");
                }
            }
            break;

        case 'f': // Log Flags
            if (optarg && optarg[0])
                log_flags = strtoul(optarg, NULL, 0);
            break;
        
        case 'L':
            connect_url.assign ("connect://");
            connect_url.append (optarg);
            break;
        }
    }
    
    // Skip any options we consumed with getopt_long
    argc -= optind;
    argv += optind;


    GDBRemoteCommunicationServer gdb_server;
    Error error;
    if (!connect_url.empty())
    {
        std::auto_ptr<ConnectionFileDescriptor> conn_ap(new ConnectionFileDescriptor());
        if (conn_ap.get())
        {
            const uint32_t max_retry_count = 50;
            uint32_t retry_count = 0;
            while (!gdb_server.IsConnected())
            {
                if (conn_ap->Connect(connect_url.c_str(), &error) == eConnectionStatusSuccess)
                {
                    gdb_server.SetConnection (conn_ap.release());
                    break;
                }
                retry_count++;

                if (retry_count >= max_retry_count)
                    break;

                usleep (100000);
            }
        }
    }


    if (gdb_server.IsConnected())
    {
        if (gdb_server.StartReadThread(&error))
        {
            bool interrupt = false;
            bool done = false;
            while (!interrupt && !done)
            {
                gdb_server.GetPacketAndSendResponse(NULL, interrupt, done);
            }
        }
        else
        {
        }
    }

    return 0;
}
