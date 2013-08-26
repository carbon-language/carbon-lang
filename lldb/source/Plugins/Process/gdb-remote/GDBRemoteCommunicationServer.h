//===-- GDBRemoteCommunicationServer.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GDBRemoteCommunicationServer_h_
#define liblldb_GDBRemoteCommunicationServer_h_

// C Includes
// C++ Includes
#include <vector>
#include <set>
// Other libraries and framework includes
// Project includes
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"
#include "GDBRemoteCommunication.h"

class ProcessGDBRemote;
class StringExtractorGDBRemote;

class GDBRemoteCommunicationServer : public GDBRemoteCommunication
{
public:
    enum
    {
        eBroadcastBitRunPacketSent = kLoUserBroadcastBit
    };
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteCommunicationServer(bool is_platform);

    virtual
    ~GDBRemoteCommunicationServer();

    bool
    GetPacketAndSendResponse (uint32_t timeout_usec,
                              lldb_private::Error &error,
                              bool &interrupt, 
                              bool &quit);

    virtual bool
    GetThreadSuffixSupported ()
    {
        return true;
    }

    // After connecting, do a little handshake with the client to make sure
    // we are at least communicating
    bool
    HandshakeWithClient (lldb_private::Error *error_ptr);

    // Set both ports to zero to let the platform automatically bind to 
    // a port chosen by the OS.
    void
    SetPortRange (uint16_t lo_port_num, uint16_t hi_port_num)
    {
        m_lo_port_num = lo_port_num;
        m_hi_port_num = hi_port_num;
        m_next_port = m_lo_port_num;
        m_use_port_range = true;
    }

    // If we are using a port range, get and update the next port to be used variable.
    // Otherwise, just return 0.
    uint16_t
    GetAndUpdateNextPort ()
    {
        if (!m_use_port_range)
            return 0;
        uint16_t val = m_next_port;
        if (++m_next_port > m_hi_port_num)
            m_next_port = m_lo_port_num;
        return val;
    }

protected:
    //typedef std::map<uint16_t, lldb::pid_t> PortToPIDMap;

    lldb::thread_t m_async_thread;
    lldb_private::ProcessLaunchInfo m_process_launch_info;
    lldb_private::Error m_process_launch_error;
    std::set<lldb::pid_t> m_spawned_pids;
    lldb_private::Mutex m_spawned_pids_mutex;
    lldb_private::ProcessInstanceInfoList m_proc_infos;
    uint32_t m_proc_infos_index;
    uint16_t m_lo_port_num;
    uint16_t m_hi_port_num;
    //PortToPIDMap m_port_to_pid_map;
    uint16_t m_next_port;
    bool m_use_port_range;
    

    size_t
    SendUnimplementedResponse (const char *packet);

    size_t
    SendErrorResponse (uint8_t error);

    size_t
    SendOKResponse ();

    bool
    Handle_A (StringExtractorGDBRemote &packet);
    
    bool
    Handle_qLaunchSuccess (StringExtractorGDBRemote &packet);

    bool
    Handle_qHostInfo (StringExtractorGDBRemote &packet);
    
    bool
    Handle_qLaunchGDBServer (StringExtractorGDBRemote &packet);
    
    bool
    Handle_qKillSpawnedProcess (StringExtractorGDBRemote &packet);

    bool
    Handle_qPlatform_IO_MkDir (StringExtractorGDBRemote &packet);
    
    bool
    Handle_qProcessInfoPID (StringExtractorGDBRemote &packet);
    
    bool
    Handle_qfProcessInfo (StringExtractorGDBRemote &packet);
    
    bool 
    Handle_qsProcessInfo (StringExtractorGDBRemote &packet);

    bool
    Handle_qC (StringExtractorGDBRemote &packet);

    bool 
    Handle_qUserName (StringExtractorGDBRemote &packet);

    bool 
    Handle_qGroupName (StringExtractorGDBRemote &packet);

    bool
    Handle_qSpeedTest (StringExtractorGDBRemote &packet);

    bool
    Handle_QEnvironment  (StringExtractorGDBRemote &packet);
    
    bool
    Handle_QLaunchArch (StringExtractorGDBRemote &packet);
    
    bool
    Handle_QSetDisableASLR (StringExtractorGDBRemote &packet);

    bool
    Handle_QSetWorkingDir (StringExtractorGDBRemote &packet);

    bool
    Handle_QStartNoAckMode (StringExtractorGDBRemote &packet);

    bool
    Handle_QSetSTDIN (StringExtractorGDBRemote &packet);

    bool
    Handle_QSetSTDOUT (StringExtractorGDBRemote &packet);

    bool
    Handle_QSetSTDERR (StringExtractorGDBRemote &packet);
    
    bool
    Handle_vFile_Open (StringExtractorGDBRemote &packet);

    bool
    Handle_vFile_Close (StringExtractorGDBRemote &packet);

    bool
    Handle_vFile_pRead (StringExtractorGDBRemote &packet);

    bool
    Handle_vFile_pWrite (StringExtractorGDBRemote &packet);

    bool
    Handle_vFile_Size (StringExtractorGDBRemote &packet);
    
    bool
    Handle_vFile_Mode (StringExtractorGDBRemote &packet);

    bool
    Handle_vFile_Exists (StringExtractorGDBRemote &packet);

    bool
    Handle_vFile_Stat (StringExtractorGDBRemote &packet);
    
    bool
    Handle_vFile_MD5 (StringExtractorGDBRemote &packet);
    
    bool
    Handle_qPlatform_RunCommand (StringExtractorGDBRemote &packet);

private:
    bool
    DebugserverProcessReaped (lldb::pid_t pid);
    
    static bool
    ReapDebugserverProcess (void *callback_baton,
                            lldb::pid_t pid,
                            bool exited,
                            int signal,
                            int status);

    //------------------------------------------------------------------
    // For GDBRemoteCommunicationServer only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunicationServer);
};

#endif  // liblldb_GDBRemoteCommunicationServer_h_
