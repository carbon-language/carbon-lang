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
// C++ Includes
// Other libraries and framework includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/TimeValue.h"

// Project includes
#include "Utility/StringExtractorGDBRemote.h"
#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// GDBRemoteCommunication constructor
//----------------------------------------------------------------------
GDBRemoteCommunication::GDBRemoteCommunication() :
    Communication("gdb-remote.packets"),
    m_send_acks (true),
    m_thread_suffix_supported (false),
    m_rx_packet_listener ("gdbremote.rx_packet"),
    m_sequence_mutex (Mutex::eMutexTypeRecursive),
    m_public_is_running (false),
    m_private_is_running (false),
    m_async_mutex (Mutex::eMutexTypeRecursive),
    m_async_packet_predicate (false),
    m_async_packet (),
    m_async_response (),
    m_async_timeout (UINT32_MAX),
    m_async_signal (-1),
    m_arch(),
    m_os(),
    m_vendor(),
    m_byte_order(eByteOrderHost),
    m_pointer_byte_size(0)
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
    if (m_send_acks)
    {
        for (size_t i = 0; i < payload_length; ++i)
            checksum += payload[i];
    }
    return checksum & 255;
}

size_t
GDBRemoteCommunication::SendAck ()
{
    ProcessGDBRemoteLog::LogIf (GDBR_LOG_PACKETS, "send packet: +");
    ConnectionStatus status = eConnectionStatusSuccess;
    char ack_char = '+';
    return Write (&ack_char, 1, status, NULL) == 1;
}

size_t
GDBRemoteCommunication::SendNack ()
{
    ProcessGDBRemoteLog::LogIf (GDBR_LOG_PACKETS, "send packet: -");
    ConnectionStatus status = eConnectionStatusSuccess;
    char nack_char = '-';
    return Write (&nack_char, 1, status, NULL) == 1;
}

size_t
GDBRemoteCommunication::SendPacketAndWaitForResponse
(
    const char *payload,
    StringExtractorGDBRemote &response,
    uint32_t timeout_seconds,
    bool send_async
)
{
    return SendPacketAndWaitForResponse (payload, 
                                         ::strlen (payload),
                                         response,
                                         timeout_seconds,
                                         send_async);
}

size_t
GDBRemoteCommunication::SendPacketAndWaitForResponse
(
    const char *payload,
    size_t payload_length,
    StringExtractorGDBRemote &response,
    uint32_t timeout_seconds,
    bool send_async
)
{
    Mutex::Locker locker;
    TimeValue timeout_time;
    timeout_time = TimeValue::Now();
    timeout_time.OffsetWithSeconds (timeout_seconds);
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));

    if (GetSequenceMutex (locker))
    {
        if (SendPacketNoLock (payload, strlen(payload)))
            return WaitForPacketNoLock (response, &timeout_time);
    }
    else
    {
        if (send_async)
        {
            Mutex::Locker async_locker (m_async_mutex);
            m_async_packet.assign(payload, payload_length);
            m_async_timeout = timeout_seconds;
            m_async_packet_predicate.SetValue (true, eBroadcastNever);
            
            if (log) 
                log->Printf ("async: async packet = %s", m_async_packet.c_str());

            bool timed_out = false;
            bool sent_interrupt = false;
            if (SendInterrupt(locker, 2, sent_interrupt, timed_out))
            {
                if (sent_interrupt)
                {
                    if (log) 
                        log->Printf ("async: sent interrupt");
                    if (m_async_packet_predicate.WaitForValueEqualTo (false, &timeout_time, &timed_out))
                    {
                        if (log) 
                            log->Printf ("async: got response");
                        response = m_async_response;
                        return response.GetStringRef().size();
                    }
                    else
                    {
                        if (log) 
                            log->Printf ("async: timed out waiting for response");
                    }
                    
                    // Make sure we wait until the continue packet has been sent again...
                    if (m_private_is_running.WaitForValueEqualTo (true, &timeout_time, &timed_out))
                    {
                        if (log) 
                            log->Printf ("async: timed out waiting for process to resume");
                    }
                }
                else
                {
                    // We had a racy condition where we went to send the interrupt
                    // yet we were able to get the loc
                }
            }
            else
            {
                if (log) 
                    log->Printf ("async: failed to interrupt");
            }
        }
        else
        {
            if (log) 
                log->Printf ("mutex taken and send_async == false, aborting packet");
        }
    }
    return 0;
}

//template<typename _Tp>
//class ScopedValueChanger
//{
//public:
//    // Take a value reference and the value to assign it to when this class
//    // instance goes out of scope.
//    ScopedValueChanger (_Tp &value_ref, _Tp value) :
//        m_value_ref (value_ref),
//        m_value (value)
//    {
//    }
//
//    // This object is going out of scope, change the value pointed to by
//    // m_value_ref to the value we got during construction which was stored in
//    // m_value;
//    ~ScopedValueChanger ()
//    {
//        m_value_ref = m_value;
//    }
//protected:
//    _Tp &m_value_ref;   // A reference to the value we will change when this object destructs
//    _Tp m_value;        // The value to assign to m_value_ref when this goes out of scope.
//};

StateType
GDBRemoteCommunication::SendContinuePacketAndWaitForResponse
(
    ProcessGDBRemote *process,
    const char *payload,
    size_t packet_length,
    StringExtractorGDBRemote &response
)
{
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));
    if (log)
        log->Printf ("GDBRemoteCommunication::%s ()", __FUNCTION__);

    Mutex::Locker locker(m_sequence_mutex);
    StateType state = eStateRunning;

    BroadcastEvent(eBroadcastBitRunPacketSent, NULL);
    m_public_is_running.SetValue (true, eBroadcastNever);
    // Set the starting continue packet into "continue_packet". This packet
    // make change if we are interrupted and we continue after an async packet...
    std::string continue_packet(payload, packet_length);
    
    while (state == eStateRunning)
    {
        if (log)
            log->Printf ("GDBRemoteCommunication::%s () sending continue packet: %s", __FUNCTION__, continue_packet.c_str());
        if (SendPacket(continue_packet.c_str(), continue_packet.size()) == 0)
            state = eStateInvalid;
        
        m_private_is_running.SetValue (true, eBroadcastNever);

        if (log)
            log->Printf ("GDBRemoteCommunication::%s () WaitForPacket(%.*s)", __FUNCTION__);

        if (WaitForPacket (response, (TimeValue*)NULL))
        {
            if (response.Empty())
                state = eStateInvalid;
            else
            {
                const char stop_type = response.GetChar();
                if (log)
                    log->Printf ("GDBRemoteCommunication::%s () got packet: %s", __FUNCTION__, response.GetStringRef().c_str());
                switch (stop_type)
                {
                case 'T':
                case 'S':
                    // Privately notify any internal threads that we have stopped
                    // in case we wanted to interrupt our process, yet we might
                    // send a packet and continue without returning control to the
                    // user.
                    m_private_is_running.SetValue (false, eBroadcastAlways);
                    if (m_async_signal != -1)
                    {
                        if (log) 
                            log->Printf ("async: send signo = %s", Host::GetSignalAsCString (m_async_signal));

                        // Save off the async signal we are supposed to send
                        const int async_signal = m_async_signal;
                        // Clear the async signal member so we don't end up
                        // sending the signal multiple times...
                        m_async_signal = -1;
                        // Check which signal we stopped with
                        uint8_t signo = response.GetHexU8(255);
                        if (signo == async_signal)
                        {
                            if (log) 
                                log->Printf ("async: stopped with signal %s, we are done running", Host::GetSignalAsCString (signo));

                            // We already stopped with a signal that we wanted
                            // to stop with, so we are done
                            response.SetFilePos (0);
                        }
                        else
                        {
                            // We stopped with a different signal that the one
                            // we wanted to stop with, so now we must resume
                            // with the signal we want
                            char signal_packet[32];
                            int signal_packet_len = 0;
                            signal_packet_len = ::snprintf (signal_packet,
                                                            sizeof (signal_packet),
                                                            "C%2.2x",
                                                            async_signal);

                            if (log) 
                                log->Printf ("async: stopped with signal %s, resume with %s", 
                                                   Host::GetSignalAsCString (signo),
                                                   Host::GetSignalAsCString (async_signal));

                            // Set the continue packet to resume...
                            continue_packet.assign(signal_packet, signal_packet_len);
                            continue;
                        }
                    }
                    else if (m_async_packet_predicate.GetValue())
                    {
                        // We are supposed to send an asynchronous packet while
                        // we are running. 
                        m_async_response.Clear();
                        if (m_async_packet.empty())
                        {
                            if (log) 
                                log->Printf ("async: error: empty async packet");                            

                        }
                        else
                        {
                            if (log) 
                                log->Printf ("async: sending packet: %s", 
                                             m_async_packet.c_str());
                            
                            SendPacketAndWaitForResponse (&m_async_packet[0], 
                                                          m_async_packet.size(),
                                                          m_async_response,
                                                          m_async_timeout,
                                                          false);
                        }
                        // Let the other thread that was trying to send the async
                        // packet know that the packet has been sent and response is
                        // ready...
                        m_async_packet_predicate.SetValue(false, eBroadcastAlways);

                        // Set the continue packet to resume...
                        continue_packet.assign (1, 'c');
                        continue;
                    }
                    // Stop with signal and thread info
                    state = eStateStopped;
                    break;

                case 'W':
                case 'X':
                    // process exited
                    state = eStateExited;
                    break;

                case 'O':
                    // STDOUT
                    {
                        std::string inferior_stdout;
                        inferior_stdout.reserve(response.GetBytesLeft () / 2);
                        char ch;
                        while ((ch = response.GetHexU8()) != '\0')
                            inferior_stdout.append(1, ch);
                        process->AppendSTDOUT (inferior_stdout.c_str(), inferior_stdout.size());
                    }
                    break;

                case 'E':
                    // ERROR
                    state = eStateInvalid;
                    break;

                default:
                    if (log)
                        log->Printf ("GDBRemoteCommunication::%s () unrecognized async packet", __FUNCTION__);
                    break;
                }
            }
        }
        else
        {
            if (log)
                log->Printf ("GDBRemoteCommunication::%s () WaitForPacket(...) => false", __FUNCTION__);
            state = eStateInvalid;
        }
    }
    if (log)
        log->Printf ("GDBRemoteCommunication::%s () => %s", __FUNCTION__, StateAsCString(state));
    response.SetFilePos(0);
    m_private_is_running.SetValue (false, eBroadcastAlways);
    m_public_is_running.SetValue (false, eBroadcastAlways);
    return state;
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

        ProcessGDBRemoteLog::LogIf (GDBR_LOG_PACKETS, "send packet: %s", packet.GetData());
        ConnectionStatus status = eConnectionStatusSuccess;
        size_t bytes_written = Write (packet.GetData(), packet.GetSize(), status, NULL);
        if (bytes_written == packet.GetSize())
        {
            if (m_send_acks)
            {
                if (GetAck (1) != '+')
                    return 0;
            }
        }
        else
        {
            ProcessGDBRemoteLog::LogIf (GDBR_LOG_PACKETS, "error: failed to send packet: %s", packet.GetData());
        }
        return bytes_written;
    }
    return 0;
}

char
GDBRemoteCommunication::GetAck (uint32_t timeout_seconds)
{
    StringExtractorGDBRemote response;
    if (WaitForPacket (response, timeout_seconds) == 1)
        return response.GetChar();
    return 0;
}

bool
GDBRemoteCommunication::GetSequenceMutex (Mutex::Locker& locker)
{
    return locker.TryLock (m_sequence_mutex.GetMutex());
}

bool
GDBRemoteCommunication::SendAsyncSignal (int signo)
{
    m_async_signal = signo;
    bool timed_out = false;
    bool sent_interrupt = false;
    Mutex::Locker locker;
    if (SendInterrupt (locker, 1, sent_interrupt, timed_out))
        return true;
    m_async_signal = -1;
    return false;
}

// This function takes a mutex locker as a parameter in case the GetSequenceMutex
// actually succeeds. If it doesn't succeed in acquiring the sequence mutex 
// (the expected result), then it will send the halt packet. If it does succeed
// then the caller that requested the interrupt will want to keep the sequence
// locked down so that no one else can send packets while the caller has control.
// This function usually gets called when we are running and need to stop the 
// target. It can also be used when we are running and and we need to do something
// else (like read/write memory), so we need to interrupt the running process
// (gdb remote protocol requires this), and do what we need to do, then resume.

bool
GDBRemoteCommunication::SendInterrupt 
(
    Mutex::Locker& locker, 
    uint32_t seconds_to_wait_for_stop,             
    bool &sent_interrupt,
    bool &timed_out
)
{
    sent_interrupt = false;
    timed_out = false;
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));

    if (IsRunning())
    {
        // Only send an interrupt if our debugserver is running...
        if (GetSequenceMutex (locker) == false)
        {
            // Someone has the mutex locked waiting for a response or for the
            // inferior to stop, so send the interrupt on the down low...
            char ctrl_c = '\x03';
            ConnectionStatus status = eConnectionStatusSuccess;
            TimeValue timeout;
            if (seconds_to_wait_for_stop)
            {
                timeout = TimeValue::Now();
                timeout.OffsetWithSeconds (seconds_to_wait_for_stop);
            }
            size_t bytes_written = Write (&ctrl_c, 1, status, NULL);
            ProcessGDBRemoteLog::LogIf (GDBR_LOG_PACKETS | GDBR_LOG_PROCESS, "send packet: \\x03");
            if (bytes_written > 0)
            {
                sent_interrupt = true;
                if (seconds_to_wait_for_stop)
                {
                    m_private_is_running.WaitForValueEqualTo (false, &timeout, &timed_out);
                    if (log)
                        log->Printf ("GDBRemoteCommunication::%s () - sent interrupt, private state stopped", __FUNCTION__);

                }
                else
                {
                    if (log)
                        log->Printf ("GDBRemoteCommunication::%s () - sent interrupt, not waiting for stop...", __FUNCTION__);                    
                }
                return true;
            }
            else
            {
                if (log)
                    log->Printf ("GDBRemoteCommunication::%s () - failed to write interrupt", __FUNCTION__);
            }
        }
        else
        {
            if (log)
                log->Printf ("GDBRemoteCommunication::%s () - got sequence mutex without having to interrupt", __FUNCTION__);
        }
    }
    return false;
}

bool
GDBRemoteCommunication::WaitForNotRunning (const TimeValue *timeout_ptr)
{
    return m_public_is_running.WaitForValueEqualTo (false, timeout_ptr, NULL);
}

bool
GDBRemoteCommunication::WaitForNotRunningPrivate (const TimeValue *timeout_ptr)
{
    return m_private_is_running.WaitForValueEqualTo (false, timeout_ptr, NULL);
}

size_t
GDBRemoteCommunication::WaitForPacket (StringExtractorGDBRemote &response, uint32_t timeout_seconds)
{
    Mutex::Locker locker(m_sequence_mutex);
    TimeValue timeout_time;
    timeout_time = TimeValue::Now();
    timeout_time.OffsetWithSeconds (timeout_seconds);
    return WaitForPacketNoLock (response, &timeout_time);
}

size_t
GDBRemoteCommunication::WaitForPacket (StringExtractorGDBRemote &response, const TimeValue* timeout_time_ptr)
{
    Mutex::Locker locker(m_sequence_mutex);
    return WaitForPacketNoLock (response, timeout_time_ptr);
}

size_t
GDBRemoteCommunication::WaitForPacketNoLock (StringExtractorGDBRemote &response, const TimeValue* timeout_time_ptr)
{
    bool checksum_error = false;
    response.Clear ();

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
                ProcessGDBRemoteLog::LogIf (GDBR_LOG_PACKETS, "read packet: %s", packet_data);
                const size_t packet_size =  event_bytes->GetByteSize();
                if (packet_data && packet_size > 0)
                {
                    std::string &response_str = response.GetStringRef();
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
                            response_str.assign (packet_data + 1, packet_size - 4);
                        if (m_send_acks)
                        {
                            char packet_checksum = strtol (&packet_data[packet_size-2], NULL, 16);
                            char actual_checksum = CalculcateChecksum (&response_str[0], response_str.size());
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
                        response_str.assign (packet_data, packet_size);
                    }
                    return response_str.size();
                }
            }
        }
        else if (Communication::eBroadcastBitReadThreadDidExit)
        {
            // Our read thread exited on us so just fall through and return zero...
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

lldb::pid_t
GDBRemoteCommunication::GetCurrentProcessID (uint32_t timeout_seconds)
{
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse("qC", strlen("qC"), response, timeout_seconds, false))
    {
        if (response.GetChar() == 'Q')
            if (response.GetChar() == 'C')
                return response.GetHexMaxU32 (false, LLDB_INVALID_PROCESS_ID);
    }
    return LLDB_INVALID_PROCESS_ID;
}

bool
GDBRemoteCommunication::GetLaunchSuccess (uint32_t timeout_seconds, std::string &error_str)
{
    error_str.clear();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse("qLaunchSuccess", strlen("qLaunchSuccess"), response, timeout_seconds, false))
    {
        if (response.IsOKPacket())
            return true;
        if (response.GetChar() == 'E')
        {
            // A string the describes what failed when launching...
            error_str = response.GetStringRef().substr(1);
        }
        else
        {
            error_str.assign ("unknown error occurred launching process");
        }
    }
    else
    {
        error_str.assign ("failed to send the qLaunchSuccess packet");
    }
    return false;
}

int
GDBRemoteCommunication::SendArgumentsPacket (char const *argv[], uint32_t timeout_seconds)
{
    if (argv && argv[0])
    {
        StreamString packet;
        packet.PutChar('A');
        const char *arg;
        for (uint32_t i = 0; (arg = argv[i]) != NULL; ++i)
        {
            const int arg_len = strlen(arg);
            if (i > 0)
                packet.PutChar(',');
            packet.Printf("%i,%i,", arg_len * 2, i);
            packet.PutBytesAsRawHex8(arg, arg_len, eByteOrderHost, eByteOrderHost);
        }

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, timeout_seconds, false))
        {
            if (response.IsOKPacket())
                return 0;
            uint8_t error = response.GetError();
            if (error)
                return error;
        }
    }
    return -1;
}

int
GDBRemoteCommunication::SendEnvironmentPacket (char const *name_equal_value, uint32_t timeout_seconds)
{
    if (name_equal_value && name_equal_value[0])
    {
        StreamString packet;
        packet.Printf("QEnvironment:%s", name_equal_value);
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, timeout_seconds, false))
        {
            if (response.IsOKPacket())
                return 0;
            uint8_t error = response.GetError();
            if (error)
                return error;
        }
    }
    return -1;
}

bool
GDBRemoteCommunication::GetHostInfo (uint32_t timeout_seconds)
{
    m_arch.Clear();
    m_os.Clear();
    m_vendor.Clear();
    m_byte_order = eByteOrderHost;
    m_pointer_byte_size = 0;

    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse ("qHostInfo", response, timeout_seconds, false))
    {
        if (response.IsUnsupportedPacket())
            return false;

        
        std::string name;
        std::string value;
        while (response.GetNameColonValue(name, value))
        {
            if (name.compare("cputype") == 0)
            {
                // exception type in big endian hex
                m_arch.SetCPUType(Args::StringToUInt32 (value.c_str(), LLDB_INVALID_CPUTYPE, 0));
            }
            else if (name.compare("cpusubtype") == 0)
            {
                // exception count in big endian hex
                m_arch.SetCPUSubtype(Args::StringToUInt32 (value.c_str(), 0, 0));
            }
            else if (name.compare("ostype") == 0)
            {
                // exception data in big endian hex
                m_os.SetCString(value.c_str());
            }
            else if (name.compare("vendor") == 0)
            {
                m_vendor.SetCString(value.c_str());
            }
            else if (name.compare("endian") == 0)
            {
                if (value.compare("little") == 0)
                    m_byte_order = eByteOrderLittle;
                else if (value.compare("big") == 0)
                    m_byte_order = eByteOrderBig;
                else if (value.compare("pdp") == 0)
                    m_byte_order = eByteOrderPDP;
            }
            else if (name.compare("ptrsize") == 0)
            {
                m_pointer_byte_size = Args::StringToUInt32 (value.c_str(), 0, 0);
            }
        }
    }
    return HostInfoIsValid();
}

int
GDBRemoteCommunication::SendAttach 
(
    lldb::pid_t pid, 
    uint32_t timeout_seconds, 
    StringExtractorGDBRemote& response
)
{
    if (pid != LLDB_INVALID_PROCESS_ID)
    {
        StreamString packet;
        packet.Printf("vAttach;%x", pid);
        
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, timeout_seconds, false))
        {
            if (response.IsErrorPacket())
                return response.GetError();
            return 0;
        }
    }
    return -1;
}

const lldb_private::ArchSpec &
GDBRemoteCommunication::GetHostArchitecture ()
{
    if (!HostInfoIsValid ())
        GetHostInfo (1);
    return m_arch;
}

const lldb_private::ConstString &
GDBRemoteCommunication::GetOSString ()
{
    if (!HostInfoIsValid ())
        GetHostInfo (1);
    return m_os;
}

const lldb_private::ConstString &
GDBRemoteCommunication::GetVendorString()
{
    if (!HostInfoIsValid ())
        GetHostInfo (1);
    return m_vendor;
}

lldb::ByteOrder
GDBRemoteCommunication::GetByteOrder ()
{
    if (!HostInfoIsValid ())
        GetHostInfo (1);
    return m_byte_order;
}

uint32_t
GDBRemoteCommunication::GetAddressByteSize ()
{
    if (!HostInfoIsValid ())
        GetHostInfo (1);
    return m_pointer_byte_size;
}

addr_t
GDBRemoteCommunication::AllocateMemory (size_t size, uint32_t permissions, uint32_t timeout_seconds)
{
    char packet[64];
    ::snprintf (packet, sizeof(packet), "_M%zx,%s%s%s", size,
                permissions & lldb::ePermissionsReadable ? "r" : "",
                permissions & lldb::ePermissionsWritable ? "w" : "",
                permissions & lldb::ePermissionsExecutable ? "x" : "");
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse (packet, response, timeout_seconds, false))
    {
        if (!response.IsErrorPacket())
            return response.GetHexMaxU64(false, LLDB_INVALID_ADDRESS);
    }
    return LLDB_INVALID_ADDRESS;
}

bool
GDBRemoteCommunication::DeallocateMemory (addr_t addr, uint32_t timeout_seconds)
{
    char packet[64];
    snprintf(packet, sizeof(packet), "_m%llx", (uint64_t)addr);
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse (packet, response, timeout_seconds, false))
    {
        if (response.IsOKPacket())
            return true;
    }
    return false;
}

