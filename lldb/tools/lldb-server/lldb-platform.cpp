//===-- lldb-platform.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#if defined(__APPLE__)
#include <netinet/in.h>
#endif
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if !defined(_WIN32)
#include <sys/wait.h>
#endif
#include <fstream>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include "Acceptor.h"
#include "LLDBServerUtilities.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerPlatform.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/HostGetOpt.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

// option descriptors for getopt_long_only()

static int g_debug = 0;
static int g_verbose = 0;
static int g_server = 0;

static struct option g_long_options[] = {
    {"debug", no_argument, &g_debug, 1},
    {"verbose", no_argument, &g_verbose, 1},
    {"log-file", required_argument, nullptr, 'l'},
    {"log-channels", required_argument, nullptr, 'c'},
    {"listen", required_argument, nullptr, 'L'},
    {"port-offset", required_argument, nullptr, 'p'},
    {"gdbserver-port", required_argument, nullptr, 'P'},
    {"min-gdbserver-port", required_argument, nullptr, 'm'},
    {"max-gdbserver-port", required_argument, nullptr, 'M'},
    {"socket-file", required_argument, nullptr, 'f'},
    {"server", no_argument, &g_server, 1},
    {nullptr, 0, nullptr, 0}};

#if defined(__APPLE__)
#define LOW_PORT (IPPORT_RESERVED)
#define HIGH_PORT (IPPORT_HIFIRSTAUTO)
#else
#define LOW_PORT (1024u)
#define HIGH_PORT (49151u)
#endif

#if !defined(_WIN32)
// Watch for signals
static void signal_handler(int signo) {
  switch (signo) {
  case SIGHUP:
    // Use SIGINT first, if that does not work, use SIGHUP as a last resort.
    // And we should not call exit() here because it results in the global
    // destructors
    // to be invoked and wreaking havoc on the threads still running.
    Host::SystemLog(Host::eSystemLogWarning,
                    "SIGHUP received, exiting lldb-server...\n");
    abort();
    break;
  }
}
#endif

static void display_usage(const char *progname, const char *subcommand) {
  fprintf(stderr, "Usage:\n  %s %s [--log-file log-file-name] [--log-channels "
                  "log-channel-list] [--port-file port-file-path] --server "
                  "--listen port\n",
          progname, subcommand);
  exit(0);
}

static Status save_socket_id_to_file(const std::string &socket_id,
                                     const FileSpec &file_spec) {
  FileSpec temp_file_spec(file_spec.GetDirectory().GetStringRef());
  Status error(llvm::sys::fs::create_directory(temp_file_spec.GetPath()));
  if (error.Fail())
    return Status("Failed to create directory %s: %s",
                  temp_file_spec.GetCString(), error.AsCString());

  llvm::SmallString<64> temp_file_path;
  temp_file_spec.AppendPathComponent("port-file.%%%%%%");
  temp_file_path = temp_file_spec.GetPath();

  Status status;
  if (auto Err =
          handleErrors(llvm::writeFileAtomically(
                           temp_file_path, file_spec.GetPath(), socket_id),
                       [&status, &file_spec](const AtomicFileWriteError &E) {
                         std::string ErrorMsgBuffer;
                         llvm::raw_string_ostream S(ErrorMsgBuffer);
                         E.log(S);

                         switch (E.Error) {
                         case atomic_write_error::failed_to_create_uniq_file:
                           status = Status("Failed to create temp file: %s",
                                           ErrorMsgBuffer.c_str());
                           break;
                         case atomic_write_error::output_stream_error:
                           status = Status("Failed to write to port file.");
                           break;
                         case atomic_write_error::failed_to_rename_temp_file:
                           status = Status("Failed to rename file %s to %s: %s",
                                           ErrorMsgBuffer.c_str(),
                                           file_spec.GetPath().c_str(),
                                           ErrorMsgBuffer.c_str());
                           break;
                         }
                       })) {
    return Status("Failed to atomically write file %s",
                  file_spec.GetPath().c_str());
  }
  return status;
}

// main
int main_platform(int argc, char *argv[]) {
  const char *progname = argv[0];
  const char *subcommand = argv[1];
  argc--;
  argv++;
#if !defined(_WIN32)
  signal(SIGPIPE, SIG_IGN);
  signal(SIGHUP, signal_handler);
#endif
  int long_option_index = 0;
  Status error;
  std::string listen_host_port;
  int ch;

  std::string log_file;
  StringRef
      log_channels; // e.g. "lldb process threads:gdb-remote default:linux all"

  GDBRemoteCommunicationServerPlatform::PortMap gdbserver_portmap;
  int min_gdbserver_port = 0;
  int max_gdbserver_port = 0;
  uint16_t port_offset = 0;

  FileSpec socket_file;
  bool show_usage = false;
  int option_error = 0;
  int socket_error = -1;

  std::string short_options(OptionParser::GetShortOptionString(g_long_options));

#if __GLIBC__
  optind = 0;
#else
  optreset = 1;
  optind = 1;
#endif

  while ((ch = getopt_long_only(argc, argv, short_options.c_str(),
                                g_long_options, &long_option_index)) != -1) {
    switch (ch) {
    case 0: // Any optional that auto set themselves will return 0
      break;

    case 'L':
      listen_host_port.append(optarg);
      break;

    case 'l': // Set Log File
      if (optarg && optarg[0])
        log_file.assign(optarg);
      break;

    case 'c': // Log Channels
      if (optarg && optarg[0])
        log_channels = StringRef(optarg);
      break;

    case 'f': // Socket file
      if (optarg && optarg[0])
        socket_file.SetFile(optarg, FileSpec::Style::native);
      break;

    case 'p': {
      if (!llvm::to_integer(optarg, port_offset)) {
        WithColor::error() << "invalid port offset string " << optarg << "\n";
        option_error = 4;
        break;
      }
      if (port_offset < LOW_PORT || port_offset > HIGH_PORT) {
        WithColor::error() << llvm::formatv(
            "port offset {0} is not in the "
            "valid user port range of {1} - {2}\n",
            port_offset, LOW_PORT, HIGH_PORT);
        option_error = 5;
      }
    } break;

    case 'P':
    case 'm':
    case 'M': {
      uint16_t portnum;
      if (!llvm::to_integer(optarg, portnum)) {
        WithColor::error() << "invalid port number string " << optarg << "\n";
        option_error = 2;
        break;
      }
      if (portnum < LOW_PORT || portnum > HIGH_PORT) {
        WithColor::error() << llvm::formatv(
            "port number {0} is not in the "
            "valid user port range of {1} - {2}\n",
            portnum, LOW_PORT, HIGH_PORT);
        option_error = 1;
        break;
      }
      if (ch == 'P')
        gdbserver_portmap.AllowPort(portnum);
      else if (ch == 'm')
        min_gdbserver_port = portnum;
      else
        max_gdbserver_port = portnum;
    } break;

    case 'h': /* fall-through is intentional */
    case '?':
      show_usage = true;
      break;
    }
  }

  if (!LLDBServerUtilities::SetupLogging(log_file, log_channels, 0))
    return -1;

  // Make a port map for a port range that was specified.
  if (min_gdbserver_port && min_gdbserver_port < max_gdbserver_port) {
    gdbserver_portmap = GDBRemoteCommunicationServerPlatform::PortMap(
        min_gdbserver_port, max_gdbserver_port);
  } else if (min_gdbserver_port || max_gdbserver_port) {
    WithColor::error() << llvm::formatv(
        "--min-gdbserver-port ({0}) is not lower than "
        "--max-gdbserver-port ({1})\n",
        min_gdbserver_port, max_gdbserver_port);
    option_error = 3;
  }

  // Print usage and exit if no listening port is specified.
  if (listen_host_port.empty())
    show_usage = true;

  if (show_usage || option_error) {
    display_usage(progname, subcommand);
    exit(option_error);
  }

  // Skip any options we consumed with getopt_long_only.
  argc -= optind;
  argv += optind;
  lldb_private::Args inferior_arguments;
  inferior_arguments.SetArguments(argc, const_cast<const char **>(argv));

  const bool children_inherit_listen_socket = false;
  // the test suite makes many connections in parallel, let's not miss any.
  // The highest this should get reasonably is a function of the number
  // of target CPUs. For now, let's just use 100.
  const int backlog = 100;

  std::unique_ptr<Acceptor> acceptor_up(Acceptor::Create(
      listen_host_port, children_inherit_listen_socket, error));
  if (error.Fail()) {
    fprintf(stderr, "failed to create acceptor: %s", error.AsCString());
    exit(socket_error);
  }

  error = acceptor_up->Listen(backlog);
  if (error.Fail()) {
    printf("failed to listen: %s\n", error.AsCString());
    exit(socket_error);
  }
  if (socket_file) {
    error =
        save_socket_id_to_file(acceptor_up->GetLocalSocketId(), socket_file);
    if (error.Fail()) {
      fprintf(stderr, "failed to write socket id to %s: %s\n",
              socket_file.GetPath().c_str(), error.AsCString());
      return 1;
    }
  }

  do {
    GDBRemoteCommunicationServerPlatform platform(
        acceptor_up->GetSocketProtocol(), acceptor_up->GetSocketScheme());

    if (port_offset > 0)
      platform.SetPortOffset(port_offset);

    if (!gdbserver_portmap.empty()) {
      platform.SetPortMap(std::move(gdbserver_portmap));
    }

    const bool children_inherit_accept_socket = true;
    Connection *conn = nullptr;
    error = acceptor_up->Accept(children_inherit_accept_socket, conn);
    if (error.Fail()) {
      WithColor::error() << error.AsCString() << '\n';
      exit(socket_error);
    }
    printf("Connection established.\n");
    if (g_server) {
      // Collect child zombie processes.
#if !defined(_WIN32)
      while (waitpid(-1, nullptr, WNOHANG) > 0)
        ;
#endif
      if (fork()) {
        // Parent doesn't need a connection to the lldb client
        delete conn;

        // Parent will continue to listen for new connections.
        continue;
      } else {
        // Child process will handle the connection and exit.
        g_server = 0;
        // Listening socket is owned by parent process.
        acceptor_up.release();
      }
    } else {
      // If not running as a server, this process will not accept
      // connections while a connection is active.
      acceptor_up.reset();
    }
    platform.SetConnection(std::unique_ptr<Connection>(conn));

    if (platform.IsConnected()) {
      if (inferior_arguments.GetArgumentCount() > 0) {
        lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
        llvm::Optional<uint16_t> port = 0;
        std::string socket_name;
        Status error = platform.LaunchGDBServer(inferior_arguments,
                                                "", // hostname
                                                pid, port, socket_name);
        if (error.Success())
          platform.SetPendingGdbServer(pid, *port, socket_name);
        else
          fprintf(stderr, "failed to start gdbserver: %s\n", error.AsCString());
      }

      bool interrupt = false;
      bool done = false;
      while (!interrupt && !done) {
        if (platform.GetPacketAndSendResponse(llvm::None, error, interrupt,
                                              done) !=
            GDBRemoteCommunication::PacketResult::Success)
          break;
      }

      if (error.Fail())
        WithColor::error() << error.AsCString() << '\n';
    }
  } while (g_server);

  fprintf(stderr, "lldb-server exiting...\n");

  return 0;
}
