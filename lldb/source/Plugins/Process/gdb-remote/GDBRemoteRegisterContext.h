//===-- GDBRemoteRegisterContext.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_GDBRemoteRegisterContext_h_
#define lldb_GDBRemoteRegisterContext_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Target/RegisterContext.h"
#include "Plugins/Process/Utility/DynamicRegisterInfo.h"

#include "GDBRemoteCommunicationClient.h"

class ThreadGDBRemote;
class ProcessGDBRemote;
class StringExtractor;

class GDBRemoteDynamicRegisterInfo :
    public DynamicRegisterInfo
{
public:
    GDBRemoteDynamicRegisterInfo () :
        DynamicRegisterInfo()
    {
    }

    ~GDBRemoteDynamicRegisterInfo ()
    {
    }

    void
    HardcodeARMRegisters(bool from_scratch);

};

class GDBRemoteRegisterContext : public lldb_private::RegisterContext
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteRegisterContext (ThreadGDBRemote &thread,
                              uint32_t concrete_frame_idx,
                              GDBRemoteDynamicRegisterInfo &reg_info,
                              bool read_all_at_once);

    virtual
    ~GDBRemoteRegisterContext ();

    //------------------------------------------------------------------
    // Subclasses must override these functions
    //------------------------------------------------------------------
    virtual void
    InvalidateAllRegisters ();

    virtual size_t
    GetRegisterCount ();

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex (size_t reg);

    virtual size_t
    GetRegisterSetCount ();

    virtual const lldb_private::RegisterSet *
    GetRegisterSet (size_t reg_set);

    virtual bool
    ReadRegister (const lldb_private::RegisterInfo *reg_info, lldb_private::RegisterValue &value);

    virtual bool
    WriteRegister (const lldb_private::RegisterInfo *reg_info, const lldb_private::RegisterValue &value);
    
    virtual bool
    ReadAllRegisterValues (lldb::DataBufferSP &data_sp);

    virtual bool
    WriteAllRegisterValues (const lldb::DataBufferSP &data_sp);

    virtual uint32_t
    ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num);

protected:
    friend class ThreadGDBRemote;

    bool
    ReadRegisterBytes (const lldb_private::RegisterInfo *reg_info,
                       lldb_private::DataExtractor &data);

    bool
    WriteRegisterBytes (const lldb_private::RegisterInfo *reg_info,
                        lldb_private::DataExtractor &data, 
                        uint32_t data_offset);

    bool
    PrivateSetRegisterValue (uint32_t reg, StringExtractor &response);
    
    void
    SetAllRegisterValid (bool b);

    bool
    GetRegisterIsValid (uint32_t reg) const
    {
#if defined (LLDB_CONFIGURATION_DEBUG)
        assert (reg < m_reg_valid.size());
#endif
        if (reg < m_reg_valid.size())
            return m_reg_valid[reg];
        return false;
    }

    void
    SetRegisterIsValid (const lldb_private::RegisterInfo *reg_info, bool valid)
    {
        if (reg_info)
            return SetRegisterIsValid (reg_info->kinds[lldb::eRegisterKindLLDB], valid);
    }

    void
    SetRegisterIsValid (uint32_t reg, bool valid)
    {
#if defined (LLDB_CONFIGURATION_DEBUG)
        assert (reg < m_reg_valid.size());
#endif
        if (reg < m_reg_valid.size())
            m_reg_valid[reg] = valid;
    }

    void
    SyncThreadState(lldb_private::Process *process);  // Assumes the sequence mutex has already been acquired.
    
    GDBRemoteDynamicRegisterInfo &m_reg_info;
    std::vector<bool> m_reg_valid;
    lldb_private::DataExtractor m_reg_data;
    bool m_read_all_at_once;

private:
    // Helper function for ReadRegisterBytes().
    bool GetPrimordialRegister(const lldb_private::RegisterInfo *reg_info,
                               GDBRemoteCommunicationClient &gdb_comm);
    // Helper function for WriteRegisterBytes().
    bool SetPrimordialRegister(const lldb_private::RegisterInfo *reg_info,
                               GDBRemoteCommunicationClient &gdb_comm);

    //------------------------------------------------------------------
    // For GDBRemoteRegisterContext only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteRegisterContext);
};

#endif  // lldb_GDBRemoteRegisterContext_h_
