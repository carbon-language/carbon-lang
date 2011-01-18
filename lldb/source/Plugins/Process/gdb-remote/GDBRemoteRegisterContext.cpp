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
#include "Utility/StringExtractorGDBRemote.h"
#include "ProcessGDBRemote.h"
#include "ThreadGDBRemote.h"
#include "Utility/ARM_GCC_Registers.h"
#include "Utility/ARM_DWARF_Registers.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// GDBRemoteRegisterContext constructor
//----------------------------------------------------------------------
GDBRemoteRegisterContext::GDBRemoteRegisterContext
(
    ThreadGDBRemote &thread,
    uint32_t concrete_frame_idx,
    GDBRemoteDynamicRegisterInfo &reg_info,
    bool read_all_at_once
) :
    RegisterContext (thread, concrete_frame_idx),
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
GDBRemoteRegisterContext::InvalidateAllRegisters ()
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

        default:
            break;
        }        
    }
    return false;
}

void
GDBRemoteRegisterContext::PrivateSetRegisterValue (uint32_t reg, StringExtractor &response)
{
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
    assert (reg_info);

    // Invalidate if needed
    InvalidateIfNeeded(false);

    const uint32_t reg_byte_size = reg_info->byte_size;
    const size_t bytes_copied = response.GetHexBytes (const_cast<uint8_t*>(m_reg_data.PeekData(reg_info->byte_offset, reg_byte_size)), reg_byte_size, '\xcc');
    bool success = bytes_copied == reg_byte_size;
    if (success)
    {
        m_reg_valid[reg] = true;
    }
    else if (bytes_copied > 0)
    {
        // Only set register is valid to false if we copied some bytes, else 
        // leave it as it was.
        m_reg_valid[reg] = false;
    }
}


bool
GDBRemoteRegisterContext::ReadRegisterBytes (uint32_t reg, DataExtractor &data)
{
    GDBRemoteCommunication &gdb_comm = GetGDBProcess().GetGDBRemote();

    InvalidateIfNeeded(false);

    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
    assert (reg_info);
    if (!m_reg_valid[reg])
    {
        Mutex::Locker locker;
        if (gdb_comm.GetSequenceMutex (locker))
        {
            const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
            if (thread_suffix_supported || GetGDBProcess().SetCurrentGDBRemoteThread(m_thread.GetID()))
            {
                char packet[64];
                StringExtractorGDBRemote response;
                int packet_len = 0;
                if (m_read_all_at_once)
                {
                    // Get all registers in one packet
                    if (thread_suffix_supported)
                        packet_len = ::snprintf (packet, sizeof(packet), "g;thread:%4.4x;", m_thread.GetID());
                    else
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
                    if (thread_suffix_supported)
                        packet_len = ::snprintf (packet, sizeof(packet), "p%x;thread:%4.4x;", reg, m_thread.GetID());
                    else
                        packet_len = ::snprintf (packet, sizeof(packet), "p%x", reg);
                    assert (packet_len < (sizeof(packet) - 1));
                    if (gdb_comm.SendPacketAndWaitForResponse(packet, response, 1, false))
                        PrivateSetRegisterValue (reg, response);
                }
            }
        }

        // Make sure we got a valid register value after reading it
        if (!m_reg_valid[reg])
            return false;
    }

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
    return true;
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
        uint8_t *dst = const_cast<uint8_t*>(m_reg_data.PeekData(reg_info->byte_offset, reg_info->byte_size));

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
            const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
            if (thread_suffix_supported || GetGDBProcess().SetCurrentGDBRemoteThread(m_thread.GetID()))
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
                    
                    if (thread_suffix_supported)
                        packet.Printf (";thread:%4.4x;", m_thread.GetID());

                    // Invalidate all register values
                    InvalidateIfNeeded (true);

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

                    if (thread_suffix_supported)
                        packet.Printf (";thread:%4.4x;", m_thread.GetID());

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
    
    Mutex::Locker locker;
    if (gdb_comm.GetSequenceMutex (locker))
    {
        char packet[32];
        const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
        if (thread_suffix_supported || GetGDBProcess().SetCurrentGDBRemoteThread(m_thread.GetID()))
        {
            int packet_len = 0;
            if (thread_suffix_supported)
                packet_len = ::snprintf (packet, sizeof(packet), "g;thread:%4.4x", m_thread.GetID());
            else
                packet_len = ::snprintf (packet, sizeof(packet), "g");
            assert (packet_len < (sizeof(packet) - 1));

            if (gdb_comm.SendPacketAndWaitForResponse(packet, packet_len, response, 1, false))
            {
                if (response.IsErrorPacket())
                    return false;
                
                response.GetStringRef().insert(0, 1, 'G');
                if (thread_suffix_supported)
                {
                    char thread_id_cstr[64];
                    ::snprintf (thread_id_cstr, sizeof(thread_id_cstr), ";thread:%4.4x;", m_thread.GetID());
                    response.GetStringRef().append (thread_id_cstr);
                }
                data_sp.reset (new DataBufferHeap (response.GetStringRef().c_str(), 
                                                   response.GetStringRef().size()));
                return true;
            }
        }
    }
    return false;
}

bool
GDBRemoteRegisterContext::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    if (!data_sp || data_sp->GetBytes() == NULL || data_sp->GetByteSize() == 0)
        return false;

    GDBRemoteCommunication &gdb_comm = GetGDBProcess().GetGDBRemote();
    StringExtractorGDBRemote response;
    Mutex::Locker locker;
    if (gdb_comm.GetSequenceMutex (locker))
    {
        const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
        if (thread_suffix_supported || GetGDBProcess().SetCurrentGDBRemoteThread(m_thread.GetID()))
        {
            if (gdb_comm.SendPacketAndWaitForResponse((const char *)data_sp->GetBytes(), 
                                                      data_sp->GetByteSize(), 
                                                      response, 
                                                      1, 
                                                      false))
            {
                if (response.IsOKPacket())
                    return true;
            }
        }
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
        //  NAME        ALT     SZ  OFF   ENCODING           FORMAT            COMPILER              DWARF               GENERIC              GDB                    LLDB NATIVE
        //  ======      ======= ==  ====  =============      ============    ===============         ===============     =========             =====                   ===========
        {   "r0",       NULL,   4,    0,  eEncodingUint,     eFormatHex,     { gcc_r0,               dwarf_r0,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    0  }},
        {   "r1",       NULL,   4,    4,  eEncodingUint,     eFormatHex,     { gcc_r1,               dwarf_r1,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    1 }},
        {   "r2",       NULL,   4,    8,  eEncodingUint,     eFormatHex,     { gcc_r2,               dwarf_r2,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    2 }},
        {   "r3",       NULL,   4,   12,  eEncodingUint,     eFormatHex,     { gcc_r3,               dwarf_r3,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    3 }},
        {   "r4",       NULL,   4,   16,  eEncodingUint,     eFormatHex,     { gcc_r4,               dwarf_r4,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    4 }},
        {   "r5",       NULL,   4,   20,  eEncodingUint,     eFormatHex,     { gcc_r5,               dwarf_r5,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    5 }},
        {   "r6",       NULL,   4,   24,  eEncodingUint,     eFormatHex,     { gcc_r6,               dwarf_r6,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    6 }},
        {   "r7",       NULL,   4,   28,  eEncodingUint,     eFormatHex,     { gcc_r7,               dwarf_r7,           LLDB_REGNUM_GENERIC_FP,  LLDB_INVALID_REGNUM, 7 }},
        {   "r8",       NULL,   4,   32,  eEncodingUint,     eFormatHex,     { gcc_r8,               dwarf_r8,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    8 }},
        {   "r9",       NULL,   4,   36,  eEncodingUint,     eFormatHex,     { gcc_r9,               dwarf_r9,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    9 }},
        {   "r10",      NULL,   4,   40,  eEncodingUint,     eFormatHex,     { gcc_r10,              dwarf_r10,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    10 }},
        {   "r11",      NULL,   4,   44,  eEncodingUint,     eFormatHex,     { gcc_r11,              dwarf_r11,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    11 }},
        {   "r12",      NULL,   4,   48,  eEncodingUint,     eFormatHex,     { gcc_r12,              dwarf_r12,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,    12 }},
        {   "sp",      "r13",   4,   52,  eEncodingUint,     eFormatHex,     { gcc_sp,               dwarf_sp,           LLDB_REGNUM_GENERIC_SP,  LLDB_INVALID_REGNUM, 13 }},
        {   "lr",      "r14",   4,   56,  eEncodingUint,     eFormatHex,     { gcc_lr,               dwarf_lr,           LLDB_REGNUM_GENERIC_RA,  LLDB_INVALID_REGNUM, 14 }},
        {   "pc",      "r15",   4,   60,  eEncodingUint,     eFormatHex,     { gcc_pc,               dwarf_pc,           LLDB_REGNUM_GENERIC_PC,  LLDB_INVALID_REGNUM, 15 }},
        {   NULL,       NULL,  12,   64,  eEncodingIEEE754,  eFormatFloat,   { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM, 16 }},
        {   NULL,       NULL,  12,   76,  eEncodingIEEE754,  eFormatFloat,   { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM, 17 }},
        {   NULL,       NULL,  12,   88,  eEncodingIEEE754,  eFormatFloat,   { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM, 18 }},
        {   NULL,       NULL,  12,  100,  eEncodingIEEE754,  eFormatFloat,   { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM, 19 }},
        {   NULL,       NULL,  12,  112,  eEncodingIEEE754,  eFormatFloat,   { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM, 20 }},
        {   NULL,       NULL,  12,  124,  eEncodingIEEE754,  eFormatFloat,   { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM, 21 }},
        {   NULL,       NULL,  12,  136,  eEncodingIEEE754,  eFormatFloat,   { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM, 22 }},
        {   NULL,       NULL,  12,  148,  eEncodingIEEE754,  eFormatFloat,   { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM, 23 }},
        {   NULL,       NULL,  12,  160,  eEncodingIEEE754,  eFormatFloat,   { LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS, LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM, 24 }},
        {   "cpsr",     "psr",  4,  172,  eEncodingUint,     eFormatHex,     { gcc_cpsr,             dwarf_cpsr,         LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM,  25 }},
        {   "s0",       NULL,   4,  176,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s0,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     26 }},
        {   "s1",       NULL,   4,  180,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s1,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     27 }},
        {   "s2",       NULL,   4,  184,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s2,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     28 }},
        {   "s3",       NULL,   4,  188,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s3,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     29 }},
        {   "s4",       NULL,   4,  192,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s4,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     30 }},
        {   "s5",       NULL,   4,  196,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s5,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     31 }},
        {   "s6",       NULL,   4,  200,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s6,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     32 }},
        {   "s7",       NULL,   4,  204,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s7,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     33 }},
        {   "s8",       NULL,   4,  208,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s8,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     34 }},
        {   "s9",       NULL,   4,  212,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s9,           LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     35 }},
        {   "s10",      NULL,   4,  216,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s10,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     36 }},
        {   "s11",      NULL,   4,  220,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s11,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     37 }},
        {   "s12",      NULL,   4,  224,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s12,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     38 }},
        {   "s13",      NULL,   4,  228,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s13,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     39 }},
        {   "s14",      NULL,   4,  232,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s14,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     40 }},
        {   "s15",      NULL,   4,  236,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s15,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     41 }},
        {   "s16",      NULL,   4,  240,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s16,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     42 }},
        {   "s17",      NULL,   4,  244,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s17,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     43 }},
        {   "s18",      NULL,   4,  248,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s18,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     44 }},
        {   "s19",      NULL,   4,  252,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s19,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     45 }},
        {   "s20",      NULL,   4,  256,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s20,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     46 }},
        {   "s21",      NULL,   4,  260,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s21,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     47 }},
        {   "s22",      NULL,   4,  264,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s22,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     48 }},
        {   "s23",      NULL,   4,  268,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s23,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     49 }},
        {   "s24",      NULL,   4,  272,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s24,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     50 }},
        {   "s25",      NULL,   4,  276,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s25,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     51 }},
        {   "s26",      NULL,   4,  280,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s26,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     52 }},
        {   "s27",      NULL,   4,  284,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s27,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     53 }},
        {   "s28",      NULL,   4,  288,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s28,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     54 }},
        {   "s29",      NULL,   4,  292,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s29,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     55 }},
        {   "s30",      NULL,   4,  296,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s30,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     56 }},
        {   "s31",      NULL,   4,  300,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_s31,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     57 }},
        {   "fpscr",    NULL,   4,  304,  eEncodingUint,     eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     58 }},
        {   "d16",      NULL,   8,  308,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d16,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     59 }},
        {   "d17",      NULL,   8,  316,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d17,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     60 }},
        {   "d18",      NULL,   8,  324,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d18,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     61 }},
        {   "d19",      NULL,   8,  332,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d19,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     62 }},
        {   "d20",      NULL,   8,  340,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d20,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     63 }},
        {   "d21",      NULL,   8,  348,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d21,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     64 }},
        {   "d22",      NULL,   8,  356,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d22,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     65 }},
        {   "d23",      NULL,   8,  364,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d23,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     66 }},
        {   "d24",      NULL,   8,  372,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d24,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     67 }},
        {   "d25",      NULL,   8,  380,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d25,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     68 }},
        {   "d26",      NULL,   8,  388,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d26,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     69 }},
        {   "d27",      NULL,   8,  396,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d27,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     70 }},
        {   "d28",      NULL,   8,  404,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d28,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     71 }},
        {   "d29",      NULL,   8,  412,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d29,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     72 }},
        {   "d30",      NULL,   8,  420,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d30,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     73 }},
        {   "d31",      NULL,   8,  428,  eEncodingIEEE754,  eFormatFloat,   { LLDB_INVALID_REGNUM,  dwarf_d31,          LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,     74 }},
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

