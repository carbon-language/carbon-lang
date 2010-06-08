//===-- GDBRemoteRegisterContext.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GDBRemoteRegisterContext.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
// Project includes
#include "StringExtractorGDBRemote.h"
#include "ProcessGDBRemote.h"
#include "ThreadGDBRemote.h"
#include "ARM_GCC_Registers.h"
#include "ARM_DWARF_Registers.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// GDBRemoteRegisterContext constructor
//----------------------------------------------------------------------
GDBRemoteRegisterContext::GDBRemoteRegisterContext
(
    ThreadGDBRemote &thread,
    StackFrame *frame,
    GDBRemoteDynamicRegisterInfo &reg_info,
    bool read_all_at_once
) :
    RegisterContext (thread, frame),
    m_reg_info (reg_info),
    m_reg_valid (),
    m_reg_data (),
    m_read_all_at_once (read_all_at_once)
{
    // Resize our vector of bools to contain one bool for every register.
    // We will use these boolean values to know when a register value
    // is valid in m_reg_data.
    m_reg_valid.resize (reg_info.GetNumRegisters());

    // Make a heap based buffer that is big enough to store all registers
    DataBufferSP reg_data_sp(new DataBufferHeap (reg_info.GetRegisterDataByteSize(), 0));
    m_reg_data.SetData (reg_data_sp);

}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
GDBRemoteRegisterContext::~GDBRemoteRegisterContext()
{
}

ProcessGDBRemote &
GDBRemoteRegisterContext::GetGDBProcess()
{
    return static_cast<ProcessGDBRemote &>(m_thread.GetProcess());
}

ThreadGDBRemote &
GDBRemoteRegisterContext::GetGDBThread()
{
    return static_cast<ThreadGDBRemote &>(m_thread);
}

void
GDBRemoteRegisterContext::Invalidate ()
{
    SetAllRegisterValid (false);
}

void
GDBRemoteRegisterContext::SetAllRegisterValid (bool b)
{
    std::vector<bool>::iterator pos, end = m_reg_valid.end();
    for (pos = m_reg_valid.begin(); pos != end; ++pos)
        *pos = b;
}

size_t
GDBRemoteRegisterContext::GetRegisterCount ()
{
    return m_reg_info.GetNumRegisters ();
}

const lldb::RegisterInfo *
GDBRemoteRegisterContext::GetRegisterInfoAtIndex (uint32_t reg)
{
    return m_reg_info.GetRegisterInfoAtIndex (reg);
}

size_t
GDBRemoteRegisterContext::GetRegisterSetCount ()
{
    return m_reg_info.GetNumRegisterSets ();
}



const lldb::RegisterSet *
GDBRemoteRegisterContext::GetRegisterSet (uint32_t reg_set)
{
    return m_reg_info.GetRegisterSet (reg_set);
}



bool
GDBRemoteRegisterContext::ReadRegisterValue (uint32_t reg, Scalar &value)
{
    // Read the register
    if (ReadRegisterBytes (reg, m_reg_data))
    {
        const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
        uint32_t offset = reg_info->byte_offset;
        switch (reg_info->encoding)
        {
        case eEncodingUint:
            switch (reg_info->byte_size)
            {
            case 1:
            case 2:
            case 4:
                value = m_reg_data.GetMaxU32 (&offset, reg_info->byte_size);
                return true;

            case 8:
                value = m_reg_data.GetMaxU64 (&offset, reg_info->byte_size);
                return true;
            }
            break;

        case eEncodingSint:
            switch (reg_info->byte_size)
            {
            case 1:
            case 2:
            case 4:
                value = (int32_t)m_reg_data.GetMaxU32 (&offset, reg_info->byte_size);
                return true;

            case 8:
                value = m_reg_data.GetMaxS64 (&offset, reg_info->byte_size);
                return true;
            }
            break;

        case eEncodingIEEE754:
            switch (reg_info->byte_size)
            {
            case sizeof (float):
                value = m_reg_data.GetFloat (&offset);
                return true;

            case sizeof (double):
                value = m_reg_data.GetDouble (&offset);
                return true;

            case sizeof (long double):
                value = m_reg_data.GetLongDouble (&offset);
                return true;
            }
            break;
        }
    }
    return false;
}


bool
GDBRemoteRegisterContext::ReadRegisterBytes (uint32_t reg, DataExtractor &data)
{
    GDBRemoteCommunication &gdb_comm = GetGDBProcess().GetGDBRemote();
// FIXME: This check isn't right because IsRunning checks the Public state, but this
// is work you need to do - for instance in ShouldStop & friends - before the public 
// state has been changed.
//    if (gdb_comm.IsRunning())
//        return false;

    if (m_reg_valid_stop_id != m_thread.GetProcess().GetStopID())
    {
        Invalidate();
        m_reg_valid_stop_id = m_thread.GetProcess().GetStopID();
    }
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
    assert (reg_info);
    if (m_reg_valid[reg] == false)
    {
        Mutex::Locker locker;
        if (gdb_comm.GetSequenceMutex (locker))
        {
            if (GetGDBProcess().SetCurrentGDBRemoteThread(m_thread.GetID()))
            {
                char packet[32];
                StringExtractorGDBRemote response;
                int packet_len;
                if (m_read_all_at_once)
                {
                    // Get all registers in one packet
                    packet_len = ::snprintf (packet, sizeof(packet), "g");
                    assert (packet_len < (sizeof(packet) - 1));
                    if (gdb_comm.SendPacketAndWaitForResponse(packet, response, 1, false))
                    {
                        if (response.IsNormalPacket())
                            if (response.GetHexBytes ((void *)m_reg_data.GetDataStart(), m_reg_data.GetByteSize(), '\xcc') == m_reg_data.GetByteSize())
                                SetAllRegisterValid (true);
                    }
                }
                else
                {
                    // Get each register individually
                    packet_len = ::snprintf (packet, sizeof(packet), "p%x", reg, false);
                    assert (packet_len < (sizeof(packet) - 1));
                    if (gdb_comm.SendPacketAndWaitForResponse(packet, response, 1, false))
                        if (response.GetHexBytes ((uint8_t*)m_reg_data.PeekData(reg_info->byte_offset, reg_info->byte_size), reg_info->byte_size, '\xcc') == reg_info->byte_size)
                            m_reg_valid[reg] = true;
                }
            }
        }
    }

    bool reg_is_valid = m_reg_valid[reg];
    if (reg_is_valid)
    {
        if (&data != &m_reg_data)
        {
            // If we aren't extracting into our own buffer (which
            // only happens when this function is called from
            // ReadRegisterValue(uint32_t, Scalar&)) then
            // we transfer bytes from our buffer into the data
            // buffer that was passed in
            data.SetByteOrder (m_reg_data.GetByteOrder());
            data.SetData (m_reg_data, reg_info->byte_offset, reg_info->byte_size);
        }
    }
    return reg_is_valid;
}


bool
GDBRemoteRegisterContext::WriteRegisterValue (uint32_t reg, const Scalar &value)
{
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
    if (reg_info)
    {
        DataExtractor data;
        if (value.GetData (data, reg_info->byte_size))
            return WriteRegisterBytes (reg, data, 0);
    }
    return false;
}


bool
GDBRemoteRegisterContext::WriteRegisterBytes (uint32_t reg, DataExtractor &data, uint32_t data_offset)
{
    GDBRemoteCommunication &gdb_comm = GetGDBProcess().GetGDBRemote();
// FIXME: This check isn't right because IsRunning checks the Public state, but this
// is work you need to do - for instance in ShouldStop & friends - before the public 
// state has been changed.
//    if (gdb_comm.IsRunning())
//        return false;

    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);

    if (reg_info)
    {
        // Grab a pointer to where we are going to put this register
        uint8_t *dst = (uint8_t *)m_reg_data.PeekData(reg_info->byte_offset, reg_info->byte_size);

        if (dst == NULL)
            return false;

        // Grab a pointer to where we are going to grab the new value from
        const uint8_t *src = data.PeekData(0, reg_info->byte_size);

        if (src == NULL)
            return false;

        if (data.GetByteOrder() == m_reg_data.GetByteOrder())
        {
            // No swapping, just copy the bytes
            ::memcpy (dst, src, reg_info->byte_size);
        }
        else
        {
            // Swap the bytes
            for (uint32_t i=0; i<reg_info->byte_size; ++i)
                dst[i] = src[reg_info->byte_size - 1 - i];
        }

        Mutex::Locker locker;
        if (gdb_comm.GetSequenceMutex (locker))
        {
            if (GetGDBProcess().SetCurrentGDBRemoteThread(m_thread.GetID()))
            {
                uint32_t offset, end_offset;
                StreamString packet;
                StringExtractorGDBRemote response;
                if (m_read_all_at_once)
                {
                    // Get all registers in one packet
                    packet.PutChar ('G');
                    offset = 0;
                    end_offset = m_reg_data.GetByteSize();

                    packet.PutBytesAsRawHex8 (m_reg_data.GetDataStart(),
                                              m_reg_data.GetByteSize(),
                                              eByteOrderHost,
                                              eByteOrderHost);
                    
                    // Invalidate all register values
                    Invalidate ();
                    
                    if (gdb_comm.SendPacketAndWaitForResponse(packet.GetString().c_str(),
                                                              packet.GetString().size(),
                                                              response,
                                                              1,
                                                              false))
                    {
                        SetAllRegisterValid (false);
                        if (response.IsOKPacket())
                        {
                            return true;
                        }
                    }
                }
                else
                {
                    // Get each register individually
                    packet.Printf ("P%x=", reg);
                    packet.PutBytesAsRawHex8 (m_reg_data.PeekData(reg_info->byte_offset, reg_info->byte_size),
                                              reg_info->byte_size,
                                              eByteOrderHost,
                                              eByteOrderHost);

                    // Invalidate just this register
                    m_reg_valid[reg] = false;
                    if (gdb_comm.SendPacketAndWaitForResponse(packet.GetString().c_str(),
                                                              packet.GetString().size(),
                                                              response,
                                                              1,
                                                              false))
                    {
                        if (response.IsOKPacket())
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}


bool
GDBRemoteRegisterContext::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    GDBRemoteCommunication &gdb_comm = GetGDBProcess().GetGDBRemote();
    StringExtractorGDBRemote response;
    if (gdb_comm.SendPacketAndWaitForResponse("g", response, 1, false))
    {
        if (response.IsErrorPacket())
            return false;
            
        response.GetStringRef().insert(0, 1, 'G');
        data_sp.reset (new DataBufferHeap(response.GetStringRef().data(), 
                                          response.GetStringRef().size()));
        return true;
    }
    return false;
}

bool
GDBRemoteRegisterContext::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    GDBRemoteCommunication &gdb_comm = GetGDBProcess().GetGDBRemote();
    StringExtractorGDBRemote response;
    if (gdb_comm.SendPacketAndWaitForResponse((const char *)data_sp->GetBytes(), 
                                              data_sp->GetByteSize(), 
                                              response, 
                                              1, 
                                              false))
    {
        if (response.IsOKPacket())
            return true;
    }
    return false;
}


uint32_t
GDBRemoteRegisterContext::ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num)
{
    return m_reg_info.ConvertRegisterKindToRegisterNumber (kind, num);
}

void
GDBRemoteDynamicRegisterInfo::HardcodeARMRegisters()
{
    static lldb::RegisterInfo
    g_register_infos[] =
    {
        //  NAME        ALT     SZ  OFF   ENCODING           FORMAT          NUM      COMPILER            DWARF               GENERIC
        //  ======      ======= ==  ====  =============      ============    ===  ===============     ===============     =========
        {   "r0",       NULL,   4,    0,  eEncodingUint,     eFormatHex,      0,  { gcc_r0,               dwarf_r0,           LLDB_INVALID_REGNUM     }},
        {   "r1",       NULL,   4,    4,  eEncodingUint,     eFormatHex,      1,  { gcc_r1,               dwarf_r1,           LLDB_INVALID_REGNUM     }},
        {   "r2",       NULL,   4,    8,  eEncodingUint,     eFormatHex,      2,  { gcc_r2,               dwarf_r2,           LLDB_INVALID_REGNUM     }},
        {   "r3",       NULL,   4,   12,  eEncodingUint,     eFormatHex,      3,  { gcc_r3,               dwarf_r3,           LLDB_INVALID_REGNUM     }},
        {   "r4",       NULL,   4,   16,  eEncodingUint,     eFormatHex,      4,  { gcc_r4,               dwarf_r4,           LLDB_INVALID_REGNUM     }},
        {   "r5",       NULL,   4,   20,  eEncodingUint,     eFormatHex,      5,  { gcc_r5,               dwarf_r5,           LLDB_INVALID_REGNUM     }},
        {   "r6",       NULL,   4,   24,  eEncodingUint,     eFormatHex,      6,  { gcc_r6,               dwarf_r6,           LLDB_INVALID_REGNUM     }},
        {   "r7",       NULL,   4,   28,  eEncodingUint,     eFormatHex,      7,  { gcc_r7,               dwarf_r7,           LLDB_REGNUM_GENERIC_FP  }},
        {   "r8",       NULL,   4,   32,  eEncodingUint,     eFormatHex,      8,  { gcc_r8,               dwarf_r8,           LLDB_INVALID_REGNUM     }},
        {   "r9",       NULL,   4,   36,  eEncodingUint,     eFormatHex,      9,  { gcc_r9,               dwarf_r9,           LLDB_INVALID_REGNUM     }},
        {   "r10",      NULL,   4,   40,  eEncodingUint,     eFormatHex,     10,  { gcc_r10,              dwarf_r10,          LLDB_INVALID_REGNUM     }},
        {   "r11",      NULL,   4,   44,  eEncodingUint,     eFormatHex,     11,  { gcc_r11,              dwarf_r11,          LLDB_INVALID_REGNUM     }},
        {   "r12",      NULL,   4,   48,  eEncodingUint,     eFormatHex,     12,  { gcc_r12,              dwarf_r12,          LLDB_INVALID_REGNUM     }},
        {   "sp",      "r13",   4,   52,  eEncodingUint,     eFormatHex,     13,  { gcc_sp,               dwarf_sp,           LLDB_REGNUM_GENERIC_SP  }},
        {   "lr",      "r14",   4,   56,  eEncodingUint,     eFormatHex,     14,  { gcc_lr,               dwarf_lr,           LLDB_REGNUM_GENERIC_RA  }},
        {   "pc",      "r15",   4,   60,  eEncodingUint,     eFormatHex,     15,  { gcc_pc,               dwarf_pc,           LLDB_REGNUM_GENERIC_PC  }},
        {   NULL,       NULL,  12,   64,  eEncodingIEEE754,  eFormatFloat,   16,  { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS }},
        {   NULL,       NULL,  12,   76,  eEncodingIEEE754,  eFormatFloat,   17,  { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS }},
        {   NULL,       NULL,  12,   88,  eEncodingIEEE754,  eFormatFloat,   18,  { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS }},
        {   NULL,       NULL,  12,  100,  eEncodingIEEE754,  eFormatFloat,   19,  { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS }},
        {   NULL,       NULL,  12,  112,  eEncodingIEEE754,  eFormatFloat,   20,  { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS }},
        {   NULL,       NULL,  12,  124,  eEncodingIEEE754,  eFormatFloat,   21,  { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS }},
        {   NULL,       NULL,  12,  136,  eEncodingIEEE754,  eFormatFloat,   22,  { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS }},
        {   NULL,       NULL,  12,  148,  eEncodingIEEE754,  eFormatFloat,   23,  { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS }},
        {   NULL,       NULL,  12,  160,  eEncodingIEEE754,  eFormatFloat,   24,  { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS }},
        {   "cpsr",     "psr",  4,  172,  eEncodingUint,     eFormatHex,     25,  { gcc_cpsr,             dwarf_cpsr,         LLDB_REGNUM_GENERIC_FLAGS   }},
        {   "s0",       NULL,   4,  176,  eEncodingIEEE754,  eFormatFloat,   26,  { LLDB_INVALID_REGNUM,  dwarf_s0,           LLDB_INVALID_REGNUM     }},
        {   "s1",       NULL,   4,  180,  eEncodingIEEE754,  eFormatFloat,   27,  { LLDB_INVALID_REGNUM,  dwarf_s1,           LLDB_INVALID_REGNUM     }},
        {   "s2",       NULL,   4,  184,  eEncodingIEEE754,  eFormatFloat,   28,  { LLDB_INVALID_REGNUM,  dwarf_s2,           LLDB_INVALID_REGNUM     }},
        {   "s3",       NULL,   4,  188,  eEncodingIEEE754,  eFormatFloat,   29,  { LLDB_INVALID_REGNUM,  dwarf_s3,           LLDB_INVALID_REGNUM     }},
        {   "s4",       NULL,   4,  192,  eEncodingIEEE754,  eFormatFloat,   30,  { LLDB_INVALID_REGNUM,  dwarf_s4,           LLDB_INVALID_REGNUM     }},
        {   "s5",       NULL,   4,  196,  eEncodingIEEE754,  eFormatFloat,   31,  { LLDB_INVALID_REGNUM,  dwarf_s5,           LLDB_INVALID_REGNUM     }},
        {   "s6",       NULL,   4,  200,  eEncodingIEEE754,  eFormatFloat,   32,  { LLDB_INVALID_REGNUM,  dwarf_s6,           LLDB_INVALID_REGNUM     }},
        {   "s7",       NULL,   4,  204,  eEncodingIEEE754,  eFormatFloat,   33,  { LLDB_INVALID_REGNUM,  dwarf_s7,           LLDB_INVALID_REGNUM     }},
        {   "s8",       NULL,   4,  208,  eEncodingIEEE754,  eFormatFloat,   34,  { LLDB_INVALID_REGNUM,  dwarf_s8,           LLDB_INVALID_REGNUM     }},
        {   "s9",       NULL,   4,  212,  eEncodingIEEE754,  eFormatFloat,   35,  { LLDB_INVALID_REGNUM,  dwarf_s9,           LLDB_INVALID_REGNUM     }},
        {   "s10",      NULL,   4,  216,  eEncodingIEEE754,  eFormatFloat,   36,  { LLDB_INVALID_REGNUM,  dwarf_s10,          LLDB_INVALID_REGNUM     }},
        {   "s11",      NULL,   4,  220,  eEncodingIEEE754,  eFormatFloat,   37,  { LLDB_INVALID_REGNUM,  dwarf_s11,          LLDB_INVALID_REGNUM     }},
        {   "s12",      NULL,   4,  224,  eEncodingIEEE754,  eFormatFloat,   38,  { LLDB_INVALID_REGNUM,  dwarf_s12,          LLDB_INVALID_REGNUM     }},
        {   "s13",      NULL,   4,  228,  eEncodingIEEE754,  eFormatFloat,   39,  { LLDB_INVALID_REGNUM,  dwarf_s13,          LLDB_INVALID_REGNUM     }},
        {   "s14",      NULL,   4,  232,  eEncodingIEEE754,  eFormatFloat,   40,  { LLDB_INVALID_REGNUM,  dwarf_s14,          LLDB_INVALID_REGNUM     }},
        {   "s15",      NULL,   4,  236,  eEncodingIEEE754,  eFormatFloat,   41,  { LLDB_INVALID_REGNUM,  dwarf_s15,          LLDB_INVALID_REGNUM     }},
        {   "s16",      NULL,   4,  240,  eEncodingIEEE754,  eFormatFloat,   42,  { LLDB_INVALID_REGNUM,  dwarf_s16,          LLDB_INVALID_REGNUM     }},
        {   "s17",      NULL,   4,  244,  eEncodingIEEE754,  eFormatFloat,   43,  { LLDB_INVALID_REGNUM,  dwarf_s17,          LLDB_INVALID_REGNUM     }},
        {   "s18",      NULL,   4,  248,  eEncodingIEEE754,  eFormatFloat,   44,  { LLDB_INVALID_REGNUM,  dwarf_s18,          LLDB_INVALID_REGNUM     }},
        {   "s19",      NULL,   4,  252,  eEncodingIEEE754,  eFormatFloat,   45,  { LLDB_INVALID_REGNUM,  dwarf_s19,          LLDB_INVALID_REGNUM     }},
        {   "s20",      NULL,   4,  256,  eEncodingIEEE754,  eFormatFloat,   46,  { LLDB_INVALID_REGNUM,  dwarf_s20,          LLDB_INVALID_REGNUM     }},
        {   "s21",      NULL,   4,  260,  eEncodingIEEE754,  eFormatFloat,   47,  { LLDB_INVALID_REGNUM,  dwarf_s21,          LLDB_INVALID_REGNUM     }},
        {   "s22",      NULL,   4,  264,  eEncodingIEEE754,  eFormatFloat,   48,  { LLDB_INVALID_REGNUM,  dwarf_s22,          LLDB_INVALID_REGNUM     }},
        {   "s23",      NULL,   4,  268,  eEncodingIEEE754,  eFormatFloat,   49,  { LLDB_INVALID_REGNUM,  dwarf_s23,          LLDB_INVALID_REGNUM     }},
        {   "s24",      NULL,   4,  272,  eEncodingIEEE754,  eFormatFloat,   50,  { LLDB_INVALID_REGNUM,  dwarf_s24,          LLDB_INVALID_REGNUM     }},
        {   "s25",      NULL,   4,  276,  eEncodingIEEE754,  eFormatFloat,   51,  { LLDB_INVALID_REGNUM,  dwarf_s25,          LLDB_INVALID_REGNUM     }},
        {   "s26",      NULL,   4,  280,  eEncodingIEEE754,  eFormatFloat,   52,  { LLDB_INVALID_REGNUM,  dwarf_s26,          LLDB_INVALID_REGNUM     }},
        {   "s27",      NULL,   4,  284,  eEncodingIEEE754,  eFormatFloat,   53,  { LLDB_INVALID_REGNUM,  dwarf_s27,          LLDB_INVALID_REGNUM     }},
        {   "s28",      NULL,   4,  288,  eEncodingIEEE754,  eFormatFloat,   54,  { LLDB_INVALID_REGNUM,  dwarf_s28,          LLDB_INVALID_REGNUM     }},
        {   "s29",      NULL,   4,  292,  eEncodingIEEE754,  eFormatFloat,   55,  { LLDB_INVALID_REGNUM,  dwarf_s29,          LLDB_INVALID_REGNUM     }},
        {   "s30",      NULL,   4,  296,  eEncodingIEEE754,  eFormatFloat,   56,  { LLDB_INVALID_REGNUM,  dwarf_s30,          LLDB_INVALID_REGNUM     }},
        {   "s31",      NULL,   4,  300,  eEncodingIEEE754,  eFormatFloat,   57,  { LLDB_INVALID_REGNUM,  dwarf_s31,          LLDB_INVALID_REGNUM     }},
        {   "fpscr",    NULL,   4,  304,  eEncodingUint,     eFormatHex,     58,  { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,LLDB_INVALID_REGNUM     }},
        {   "d16",      NULL,   8,  308,  eEncodingIEEE754,  eFormatFloat,   59,  { LLDB_INVALID_REGNUM,  dwarf_d16,          LLDB_INVALID_REGNUM     }},
        {   "d17",      NULL,   8,  316,  eEncodingIEEE754,  eFormatFloat,   60,  { LLDB_INVALID_REGNUM,  dwarf_d17,          LLDB_INVALID_REGNUM     }},
        {   "d18",      NULL,   8,  324,  eEncodingIEEE754,  eFormatFloat,   61,  { LLDB_INVALID_REGNUM,  dwarf_d18,          LLDB_INVALID_REGNUM     }},
        {   "d19",      NULL,   8,  332,  eEncodingIEEE754,  eFormatFloat,   62,  { LLDB_INVALID_REGNUM,  dwarf_d19,          LLDB_INVALID_REGNUM     }},
        {   "d20",      NULL,   8,  340,  eEncodingIEEE754,  eFormatFloat,   63,  { LLDB_INVALID_REGNUM,  dwarf_d20,          LLDB_INVALID_REGNUM     }},
        {   "d21",      NULL,   8,  348,  eEncodingIEEE754,  eFormatFloat,   64,  { LLDB_INVALID_REGNUM,  dwarf_d21,          LLDB_INVALID_REGNUM     }},
        {   "d22",      NULL,   8,  356,  eEncodingIEEE754,  eFormatFloat,   65,  { LLDB_INVALID_REGNUM,  dwarf_d22,          LLDB_INVALID_REGNUM     }},
        {   "d23",      NULL,   8,  364,  eEncodingIEEE754,  eFormatFloat,   66,  { LLDB_INVALID_REGNUM,  dwarf_d23,          LLDB_INVALID_REGNUM     }},
        {   "d24",      NULL,   8,  372,  eEncodingIEEE754,  eFormatFloat,   67,  { LLDB_INVALID_REGNUM,  dwarf_d24,          LLDB_INVALID_REGNUM     }},
        {   "d25",      NULL,   8,  380,  eEncodingIEEE754,  eFormatFloat,   68,  { LLDB_INVALID_REGNUM,  dwarf_d25,          LLDB_INVALID_REGNUM     }},
        {   "d26",      NULL,   8,  388,  eEncodingIEEE754,  eFormatFloat,   69,  { LLDB_INVALID_REGNUM,  dwarf_d26,          LLDB_INVALID_REGNUM     }},
        {   "d27",      NULL,   8,  396,  eEncodingIEEE754,  eFormatFloat,   70,  { LLDB_INVALID_REGNUM,  dwarf_d27,          LLDB_INVALID_REGNUM     }},
        {   "d28",      NULL,   8,  404,  eEncodingIEEE754,  eFormatFloat,   71,  { LLDB_INVALID_REGNUM,  dwarf_d28,          LLDB_INVALID_REGNUM     }},
        {   "d29",      NULL,   8,  412,  eEncodingIEEE754,  eFormatFloat,   72,  { LLDB_INVALID_REGNUM,  dwarf_d29,          LLDB_INVALID_REGNUM     }},
        {   "d30",      NULL,   8,  420,  eEncodingIEEE754,  eFormatFloat,   73,  { LLDB_INVALID_REGNUM,  dwarf_d30,          LLDB_INVALID_REGNUM     }},
        {   "d31",      NULL,   8,  428,  eEncodingIEEE754,  eFormatFloat,   74,  { LLDB_INVALID_REGNUM,  dwarf_d31,          LLDB_INVALID_REGNUM     }},
    };
    static const uint32_t num_registers = sizeof (g_register_infos)/sizeof (lldb::RegisterInfo);
    static ConstString gpr_reg_set ("General Purpose Registers");
    static ConstString vfp_reg_set ("Floating Point Registers");
    for (uint32_t i=0; i<num_registers; ++i)
    {
        ConstString name;
        ConstString alt_name;
        if (g_register_infos[i].name && g_register_infos[i].name[0])
            name.SetCString(g_register_infos[i].name);
        if (g_register_infos[i].alt_name && g_register_infos[i].alt_name[0])
            alt_name.SetCString(g_register_infos[i].alt_name);

        AddRegister (g_register_infos[i], name, alt_name, i < 26 ? gpr_reg_set : vfp_reg_set);
    }
}

