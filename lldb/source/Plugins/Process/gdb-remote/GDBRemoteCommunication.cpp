//===-- GDBRemoteCommunication.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "GDBRemoteCommunication.h"

// C Includes
#include <string.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Target/Process.h"

// Project includes
#include "ProcessGDBRemoteLog.h"

#define DEBUGSERVER_BASENAME    "debugserver"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// GDBRemoteCommunication constructor
//----------------------------------------------------------------------
GDBRemoteCommunication::GDBRemoteCommunication(const char *comm_name, 
                                               const char *listener_name, 
                                               bool is_platform) :
    Communication(comm_name),
    m_packet_timeout (60),
    m_rx_packet_listener (listener_name),
    m_sequence_mutex (Mutex::eMutexTypeRecursive),
    m_public_is_running (false),
    m_private_is_running (false),
    m_send_acks (true),
    m_is_platform (is_platform)
{
    m_rx_packet_listener.StartListeningForEvents(this,
                                                 Communication::eBroadcastBitPacketAvailable  |
                                                 Communication::eBroadcastBitReadThreadDidExit);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
GDBRemoteCommunication::~GDBRemoteCommunication()
{
    m_rx_packet_listener.StopListeningForEvents(this,
                                                Communication::eBroadcastBitPacketAvailable  |
                                                Communication::eBroadcastBitReadThreadDidExit);
    if (IsConnected())
    {
        StopReadThread();
        Disconnect();
    }
}

char
GDBRemoteCommunication::CalculcateChecksum (const char *payload, size_t payload_length)
{
    int checksum = 0;

    // We only need to compute the checksum if we are sending acks
    if (GetSendAcks ())
    {
        for (size_t i = 0; i < payload_length; ++i)
            checksum += payload[i];
    }
    return checksum & 255;
}

size_t
GDBRemoteCommunication::SendAck ()
{
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS));
    if (log)
        log->Printf ("send packet: +");
    ConnectionStatus status = eConnectionStatusSuccess;
    char ack_char = '+';
    return Write (&ack_char, 1, status, NULL);
}

size_t
GDBRemoteCommunication::SendNack ()
{
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS));
    if (log)
        log->Printf ("send packet: -");
    ConnectionStatus status = eConnectionStatusSuccess;
    char nack_char = '-';
    return Write (&nack_char, 1, status, NULL);
}

size_t
GDBRemoteCommunication::SendPacket (lldb_private::StreamString &payload)
{
    Mutex::Locker locker(m_sequence_mutex);
    const std::string &p (payload.GetString());
    return SendPacketNoLock (p.c_str(), p.size());
}

size_t
GDBRemoteCommunication::SendPacket (const char *payload)
{
    Mutex::Locker locker(m_sequence_mutex);
    return SendPacketNoLock (payload, ::strlen (payload));
}

size_t
GDBRemoteCommunication::SendPacket (const char *payload, size_t payload_length)
{
    Mutex::Locker locker(m_sequence_mutex);
    return SendPacketNoLock (payload, payload_length);
}

size_t
GDBRemoteCommunication::SendPacketNoLock (const char *payload, size_t payload_length)
{
    if (IsConnected())
    {
        StreamString packet(0, 4, eByteOrderBig);

        packet.PutChar('$');
        packet.Write (payload, payload_length);
        packet.PutChar('#');
        packet.PutHex8(CalculcateChecksum (payload, payload_length));

        LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS));
        if (log)
            log->Printf ("send packet: %s", packet.GetData());
        ConnectionStatus status = eConnectionStatusSuccess;
        size_t bytes_written = Write (packet.GetData(), packet.GetSize(), status, NULL);
        if (bytes_written == packet.GetSize())
        {
            if (GetSendAcks ())
            {
                if (GetAck () != '+')
                {
                    printf("get ack failed...");
                    return 0;
                }
            }
        }
        else
        {
            LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS));
            if (log)
                log->Printf ("error: failed to send packet: %s", packet.GetData());
        }
        return bytes_written;
    }
    return 0;
}

char
GDBRemoteCommunication::GetAck ()
{
    StringExtractorGDBRemote packet;
    if (WaitForPacket (packet, m_packet_timeout) == 1)
        return packet.GetChar();
    return 0;
}

bool
GDBRemoteCommunication::GetSequenceMutex (Mutex::Locker& locker)
{
    return locker.TryLock (m_sequence_mutex.GetMutex());
}


bool
GDBRemoteCommunication::WaitForNotRunningPrivate (const TimeValue *timeout_ptr)
{
    return m_private_is_running.WaitForValueEqualTo (false, timeout_ptr, NULL);
}

size_t
GDBRemoteCommunication::WaitForPacket (StringExtractorGDBRemote &packet, uint32_t timeout_seconds)
{
    Mutex::Locker locker(m_sequence_mutex);
    TimeValue timeout_time;
    timeout_time = TimeValue::Now();
    timeout_time.OffsetWithSeconds (timeout_seconds);
    return WaitForPacketNoLock (packet, &timeout_time);
}

size_t
GDBRemoteCommunication::WaitForPacket (StringExtractorGDBRemote &packet, const TimeValue* timeout_time_ptr)
{
    Mutex::Locker locker(m_sequence_mutex);
    return WaitForPacketNoLock (packet, timeout_time_ptr);
}

size_t
GDBRemoteCommunication::WaitForPacketNoLock (StringExtractorGDBRemote &packet, const TimeValue* timeout_time_ptr)
{
    bool checksum_error = false;
    packet.Clear ();

    EventSP event_sp;

    if (m_rx_packet_listener.WaitForEvent (timeout_time_ptr, event_sp))
    {
        const uint32_t event_type = event_sp->GetType();
        if (event_type | Communication::eBroadcastBitPacketAvailable)
        {
            const EventDataBytes *event_bytes = EventDataBytes::GetEventDataFromEvent(event_sp.get());
            if (event_bytes)
            {
                const char * packet_data =  (const char *)event_bytes->GetBytes();
                LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS));
                if (log)
                    log->Printf ("read packet: %s", packet_data);
                const size_t packet_size =  event_bytes->GetByteSize();
                if (packet_data && packet_size > 0)
                {
                    std::string &packet_str = packet.GetStringRef();
                    if (packet_data[0] == '$')
                    {
                        bool success = false;
                        if (packet_size < 4)
                            ::fprintf (stderr, "Packet that starts with $ is too short: '%s'\n", packet_data);
                        else if (packet_data[packet_size-3] != '#' || 
                                 !::isxdigit (packet_data[packet_size-2]) || 
                                 !::isxdigit (packet_data[packet_size-1]))
                            ::fprintf (stderr, "Invalid checksum footer for packet: '%s'\n", packet_data);
                        else
                            success = true;
                        
                        if (success)
                            packet_str.assign (packet_data + 1, packet_size - 4);
                        if (GetSendAcks ())
                        {
                            char packet_checksum = strtol (&packet_data[packet_size-2], NULL, 16);
                            char actual_checksum = CalculcateChecksum (&packet_str[0], packet_str.size());
                            checksum_error = packet_checksum != actual_checksum;
                            // Send the ack or nack if needed
                            if (checksum_error || !success)
                                SendNack();
                            else
                                SendAck();
                        }
                    }
                    else
                    {
                        packet_str.assign (packet_data, packet_size);
                    }
                    return packet_str.size();
                }
            }
        }
        else if (event_type | Communication::eBroadcastBitReadThreadDidExit)
        {
            // Our read thread exited on us so just fall through and return zero...
            Disconnect();
        }
    }
    return 0;
}

void
GDBRemoteCommunication::AppendBytesToCache (const uint8_t *src, size_t src_len, bool broadcast, 
                                            ConnectionStatus status)
{
    // Put the packet data into the buffer in a thread safe fashion
    Mutex::Locker locker(m_bytes_mutex);
    m_bytes.append ((const char *)src, src_len);

    // Parse up the packets into gdb remote packets
    while (!m_bytes.empty())
    {
        // end_idx must be one past the last valid packet byte. Start
        // it off with an invalid value that is the same as the current
        // index.
        size_t end_idx = 0;

        switch (m_bytes[0])
        {
            case '+':       // Look for ack
            case '-':       // Look for cancel
            case '\x03':    // ^C to halt target
                end_idx = 1;  // The command is one byte long...
                break;

            case '$':
                // Look for a standard gdb packet?
                end_idx = m_bytes.find('#');
                if (end_idx != std::string::npos)
                {
                    if (end_idx + 2 < m_bytes.size())
                    {
                        end_idx += 3;
                    }
                    else
                    {
                        // Checksum bytes aren't all here yet
                        end_idx = std::string::npos;
                    }
                }
                break;

            default:
                break;
        }

        if (end_idx == std::string::npos)
        {
            //ProcessGDBRemoteLog::LogIf (GDBR_LOG_PACKETS | GDBR_LOG_VERBOSE, "GDBRemoteCommunication::%s packet not yet complete: '%s'",__FUNCTION__, m_bytes.c_str());
            return;
        }
        else if (end_idx > 0)
        {
            // We have a valid packet...
            assert (end_idx <= m_bytes.size());
            std::auto_ptr<EventDataBytes> event_bytes_ap (new EventDataBytes (&m_bytes[0], end_idx));
            ProcessGDBRemoteLog::LogIf (GDBR_LOG_COMM, "got full packet: %s", event_bytes_ap->GetBytes());
            BroadcastEvent (eBroadcastBitPacketAvailable, event_bytes_ap.release());
            m_bytes.erase(0, end_idx);
        }
        else
        {
            assert (1 <= m_bytes.size());
            ProcessGDBRemoteLog::LogIf (GDBR_LOG_COMM, "GDBRemoteCommunication::%s tossing junk byte at %c",__FUNCTION__, m_bytes[0]);
            m_bytes.erase(0, 1);
        }
    }
}

Error
GDBRemoteCommunication::StartDebugserverProcess (const char *debugserver_url,
                                                 const char *unix_socket_name,  // For handshaking
                                                 lldb_private::ProcessLaunchInfo &launch_info)
{
    Error error;
    // If we locate debugserver, keep that located version around
    static FileSpec g_debugserver_file_spec;
    
    // This function will fill in the launch information for the debugserver
    // instance that gets launched.
    launch_info.Clear();
    
    char debugserver_path[PATH_MAX];
    FileSpec &debugserver_file_spec = launch_info.GetExecutableFile();
    
    // Always check to see if we have an environment override for the path
    // to the debugserver to use and use it if we do.
    const char *env_debugserver_path = getenv("LLDB_DEBUGSERVER_PATH");
    if (env_debugserver_path)
        debugserver_file_spec.SetFile (env_debugserver_path, false);
    else
        debugserver_file_spec = g_debugserver_file_spec;
    bool debugserver_exists = debugserver_file_spec.Exists();
    if (!debugserver_exists)
    {
        // The debugserver binary is in the LLDB.framework/Resources
        // directory. 
        if (Host::GetLLDBPath (ePathTypeSupportExecutableDir, debugserver_file_spec))
        {
            debugserver_file_spec.GetFilename().SetCString(DEBUGSERVER_BASENAME);
            debugserver_exists = debugserver_file_spec.Exists();
            if (debugserver_exists)
            {
                g_debugserver_file_spec = debugserver_file_spec;
            }
            else
            {
                g_debugserver_file_spec.Clear();
                debugserver_file_spec.Clear();
            }
        }
    }
    
    if (debugserver_exists)
    {
        debugserver_file_spec.GetPath (debugserver_path, sizeof(debugserver_path));

        Args &debugserver_args = launch_info.GetArguments();
        debugserver_args.Clear();
        char arg_cstr[PATH_MAX];
        
        // Start args with "debugserver /file/path -r --"
        debugserver_args.AppendArgument(debugserver_path);
        debugserver_args.AppendArgument(debugserver_url);
        // use native registers, not the GDB registers
        debugserver_args.AppendArgument("--native-regs");   
        // make debugserver run in its own session so signals generated by 
        // special terminal key sequences (^C) don't affect debugserver
        debugserver_args.AppendArgument("--setsid");
        
        if (unix_socket_name && unix_socket_name[0])
        {
            debugserver_args.AppendArgument("--unix-socket");
            debugserver_args.AppendArgument(unix_socket_name);
        }

        const char *env_debugserver_log_file = getenv("LLDB_DEBUGSERVER_LOG_FILE");
        if (env_debugserver_log_file)
        {
            ::snprintf (arg_cstr, sizeof(arg_cstr), "--log-file=%s", env_debugserver_log_file);
            debugserver_args.AppendArgument(arg_cstr);
        }
        
        const char *env_debugserver_log_flags = getenv("LLDB_DEBUGSERVER_LOG_FLAGS");
        if (env_debugserver_log_flags)
        {
            ::snprintf (arg_cstr, sizeof(arg_cstr), "--log-flags=%s", env_debugserver_log_flags);
            debugserver_args.AppendArgument(arg_cstr);
        }
        //            debugserver_args.AppendArgument("--log-file=/tmp/debugserver.txt");
        //            debugserver_args.AppendArgument("--log-flags=0x802e0e");
        
        // We currently send down all arguments, attach pids, or attach 
        // process names in dedicated GDB server packets, so we don't need
        // to pass them as arguments. This is currently because of all the
        // things we need to setup prior to launching: the environment,
        // current working dir, file actions, etc.
#if 0
        // Now append the program arguments
        if (inferior_argv)
        {
            // Terminate the debugserver args so we can now append the inferior args
            debugserver_args.AppendArgument("--");
            
            for (int i = 0; inferior_argv[i] != NULL; ++i)
                debugserver_args.AppendArgument (inferior_argv[i]);
        }
        else if (attach_pid != LLDB_INVALID_PROCESS_ID)
        {
            ::snprintf (arg_cstr, sizeof(arg_cstr), "--attach=%u", attach_pid);
            debugserver_args.AppendArgument (arg_cstr);
        }
        else if (attach_name && attach_name[0])
        {
            if (wait_for_launch)
                debugserver_args.AppendArgument ("--waitfor");
            else
                debugserver_args.AppendArgument ("--attach");
            debugserver_args.AppendArgument (attach_name);
        }
#endif
        
        // Close STDIN, STDOUT and STDERR. We might need to redirect them
        // to "/dev/null" if we run into any problems.
//        launch_info.AppendCloseFileAction (STDIN_FILENO);
//        launch_info.AppendCloseFileAction (STDOUT_FILENO);
//        launch_info.AppendCloseFileAction (STDERR_FILENO);
        
        error = Host::LaunchProcess(launch_info);
    }
    else
    {
        error.SetErrorStringWithFormat ("Unable to locate " DEBUGSERVER_BASENAME ".\n");
    }
    return error;
}

