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
#include "lldb/Core/RegisterValue.h"
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

const RegisterInfo *
GDBRemoteRegisterContext::GetRegisterInfoAtIndex (uint32_t reg)
{
    return m_reg_info.GetRegisterInfoAtIndex (reg);
}

size_t
GDBRemoteRegisterContext::GetRegisterSetCount ()
{
    return m_reg_info.GetNumRegisterSets ();
}



const RegisterSet *
GDBRemoteRegisterContext::GetRegisterSet (uint32_t reg_set)
{
    return m_reg_info.GetRegisterSet (reg_set);
}



bool
GDBRemoteRegisterContext::ReadRegister (const RegisterInfo *reg_info, RegisterValue &value)
{
    // Read the register
    if (ReadRegisterBytes (reg_info, value, m_reg_data))
    {
        const bool partial_data_ok = false;
        Error error (value.SetValueFromData(reg_info, m_reg_data, reg_info->byte_offset, partial_data_ok));
        return error.Success();
    }
    return false;
}

bool
GDBRemoteRegisterContext::PrivateSetRegisterValue (uint32_t reg, StringExtractor &response)
{
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
    if (reg_info == NULL)
        return false;

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
    return success;
}


bool
GDBRemoteRegisterContext::ReadRegisterBytes (const RegisterInfo *reg_info, RegisterValue &value, DataExtractor &data)
{
    GDBRemoteCommunicationClient &gdb_comm (GetGDBProcess().GetGDBRemote());

    InvalidateIfNeeded(false);

    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

    if (!m_reg_valid[reg])
    {
        Mutex::Locker locker;
        if (gdb_comm.GetSequenceMutex (locker))
        {
            const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
            if (thread_suffix_supported || GetGDBProcess().GetGDBRemote().SetCurrentThread(m_thread.GetID()))
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
                    if (gdb_comm.SendPacketAndWaitForResponse(packet, response, false))
                    {
                        if (response.IsNormalResponse())
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
                    if (gdb_comm.SendPacketAndWaitForResponse(packet, response, false))
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
GDBRemoteRegisterContext::WriteRegister (const RegisterInfo *reg_info,
                                         const RegisterValue &value)
{
    DataExtractor data;
    if (value.GetData (data))
        return WriteRegisterBytes (reg_info, value, data, 0);
    return false;
}


bool
GDBRemoteRegisterContext::WriteRegisterBytes (const lldb_private::RegisterInfo *reg_info, const RegisterValue &value, DataExtractor &data, uint32_t data_offset)
{
    GDBRemoteCommunicationClient &gdb_comm (GetGDBProcess().GetGDBRemote());
// FIXME: This check isn't right because IsRunning checks the Public state, but this
// is work you need to do - for instance in ShouldStop & friends - before the public 
// state has been changed.
//    if (gdb_comm.IsRunning())
//        return false;

    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

    // Grab a pointer to where we are going to put this register
    uint8_t *dst = const_cast<uint8_t*>(m_reg_data.PeekData(reg_info->byte_offset, reg_info->byte_size));

    if (dst == NULL)
        return false;
    
    
    if (data.CopyByteOrderedData (data_offset,                  // src offset
                                  reg_info->byte_size,          // src length
                                  dst,                          // dst
                                  reg_info->byte_size,          // dst length
                                  m_reg_data.GetByteOrder()))   // dst byte order
    {
        Mutex::Locker locker;
        if (gdb_comm.GetSequenceMutex (locker))
        {
            const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
            if (thread_suffix_supported || GetGDBProcess().GetGDBRemote().SetCurrentThread(m_thread.GetID()))
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
                                              lldb::endian::InlHostByteOrder(),
                                              lldb::endian::InlHostByteOrder());
                    
                    if (thread_suffix_supported)
                        packet.Printf (";thread:%4.4x;", m_thread.GetID());

                    // Invalidate all register values
                    InvalidateIfNeeded (true);

                    if (gdb_comm.SendPacketAndWaitForResponse(packet.GetString().c_str(),
                                                              packet.GetString().size(),
                                                              response,
                                                              false))
                    {
                        SetAllRegisterValid (false);
                        if (response.IsOKResponse())
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
                                              lldb::endian::InlHostByteOrder(),
                                              lldb::endian::InlHostByteOrder());

                    if (thread_suffix_supported)
                        packet.Printf (";thread:%4.4x;", m_thread.GetID());

                    // Invalidate just this register
                    m_reg_valid[reg] = false;
                    if (gdb_comm.SendPacketAndWaitForResponse(packet.GetString().c_str(),
                                                              packet.GetString().size(),
                                                              response,
                                                              false))
                    {
                        if (response.IsOKResponse())
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
    GDBRemoteCommunicationClient &gdb_comm (GetGDBProcess().GetGDBRemote());
    StringExtractorGDBRemote response;
    
    Mutex::Locker locker;
    if (gdb_comm.GetSequenceMutex (locker))
    {
        char packet[32];
        const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
        if (thread_suffix_supported || GetGDBProcess().GetGDBRemote().SetCurrentThread(m_thread.GetID()))
        {
            int packet_len = 0;
            if (thread_suffix_supported)
                packet_len = ::snprintf (packet, sizeof(packet), "g;thread:%4.4x", m_thread.GetID());
            else
                packet_len = ::snprintf (packet, sizeof(packet), "g");
            assert (packet_len < (sizeof(packet) - 1));

            if (gdb_comm.SendPacketAndWaitForResponse(packet, packet_len, response, false))
            {
                if (response.IsErrorResponse())
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

    GDBRemoteCommunicationClient &gdb_comm (GetGDBProcess().GetGDBRemote());
    StringExtractorGDBRemote response;
    Mutex::Locker locker;
    if (gdb_comm.GetSequenceMutex (locker))
    {
        const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
        if (thread_suffix_supported || GetGDBProcess().GetGDBRemote().SetCurrentThread(m_thread.GetID()))
        {
            if (gdb_comm.SendPacketAndWaitForResponse((const char *)data_sp->GetBytes(), 
                                                      data_sp->GetByteSize(), 
                                                      response, 
                                                      false))
            {
                if (response.IsOKResponse())
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
    static RegisterInfo g_register_infos[] = {
//   NAME    ALT    SZ  OFF  ENCODING          FORMAT          COMPILER             DWARF                GENERIC                 GDB    LLDB
//   ======  ====== === ===  =============     ============    ===================  ===================  ======================  ===    ====
    { "r0",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r0,              dwarf_r0,            LLDB_INVALID_REGNUM,     0,      0 }},
    { "r1",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r1,              dwarf_r1,            LLDB_INVALID_REGNUM,     1,      1 }},
    { "r2",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r2,              dwarf_r2,            LLDB_INVALID_REGNUM,     2,      2 }},
    { "r3",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r3,              dwarf_r3,            LLDB_INVALID_REGNUM,     3,      3 }},
    { "r4",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r4,              dwarf_r4,            LLDB_INVALID_REGNUM,     4,      4 }},
    { "r5",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r5,              dwarf_r5,            LLDB_INVALID_REGNUM,     5,      5 }},
    { "r6",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r6,              dwarf_r6,            LLDB_INVALID_REGNUM,     6,      6 }},
    { "r7",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r7,              dwarf_r7,            LLDB_REGNUM_GENERIC_FP,  7,      7 }},
    { "r8",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r8,              dwarf_r8,            LLDB_INVALID_REGNUM,     8,      8 }},
    { "r9",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r9,              dwarf_r9,            LLDB_INVALID_REGNUM,     9,      9 }},
    { "r10",  NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r10,             dwarf_r10,           LLDB_INVALID_REGNUM,    10,     10 }},
    { "r11",  NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r11,             dwarf_r11,           LLDB_INVALID_REGNUM,    11,     11 }},
    { "r12",  NULL,   4,   0, eEncodingUint,    eFormatHex,   { gcc_r12,             dwarf_r12,           LLDB_INVALID_REGNUM,    12,     12 }},
    { "sp",   "r13",  4,   0, eEncodingUint,    eFormatHex,   { gcc_sp,              dwarf_sp,            LLDB_REGNUM_GENERIC_SP, 13,     13 }},
    { "lr",   "r14",  4,   0, eEncodingUint,    eFormatHex,   { gcc_lr,              dwarf_lr,            LLDB_REGNUM_GENERIC_RA, 14,     14 }},
    { "pc",   "r15",  4,   0, eEncodingUint,    eFormatHex,   { gcc_pc,              dwarf_pc,            LLDB_REGNUM_GENERIC_PC, 15,     15 }},
    { "f0",   NULL,  12,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    16,     16 }},
    { "f1",   NULL,  12,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    17,     17 }},
    { "f2",   NULL,  12,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    18,     18 }},
    { "f3",   NULL,  12,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    19,     19 }},
    { "f4",   NULL,  12,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    20,     20 }},
    { "f5",   NULL,  12,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    21,     21 }},
    { "f6",   NULL,  12,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    22,     22 }},
    { "f7",   NULL,  12,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    23,     23 }},
    { "fps",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    24,     24 }},
    { "cpsr", "psr",  4,   0, eEncodingUint,    eFormatHex,   { gcc_cpsr,            dwarf_cpsr,          LLDB_INVALID_REGNUM,    25,     25 }},
    { "s0",   NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s0,            LLDB_INVALID_REGNUM,    26,     26 }},
    { "s1",   NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s1,            LLDB_INVALID_REGNUM,    27,     27 }},
    { "s2",   NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s2,            LLDB_INVALID_REGNUM,    28,     28 }},
    { "s3",   NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s3,            LLDB_INVALID_REGNUM,    29,     29 }},
    { "s4",   NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s4,            LLDB_INVALID_REGNUM,    30,     30 }},
    { "s5",   NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s5,            LLDB_INVALID_REGNUM,    31,     31 }},
    { "s6",   NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s6,            LLDB_INVALID_REGNUM,    32,     32 }},
    { "s7",   NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s7,            LLDB_INVALID_REGNUM,    33,     33 }},
    { "s8",   NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s8,            LLDB_INVALID_REGNUM,    34,     34 }},
    { "s9",   NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s9,            LLDB_INVALID_REGNUM,    35,     35 }},
    { "s10",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s10,           LLDB_INVALID_REGNUM,    36,     36 }},
    { "s11",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s11,           LLDB_INVALID_REGNUM,    37,     37 }},
    { "s12",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s12,           LLDB_INVALID_REGNUM,    38,     38 }},
    { "s13",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s13,           LLDB_INVALID_REGNUM,    39,     39 }},
    { "s14",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s14,           LLDB_INVALID_REGNUM,    40,     40 }},
    { "s15",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s15,           LLDB_INVALID_REGNUM,    41,     41 }},
    { "s16",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s16,           LLDB_INVALID_REGNUM,    42,     42 }},
    { "s17",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s17,           LLDB_INVALID_REGNUM,    43,     43 }},
    { "s18",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s18,           LLDB_INVALID_REGNUM,    44,     44 }},
    { "s19",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s19,           LLDB_INVALID_REGNUM,    45,     45 }},
    { "s20",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s20,           LLDB_INVALID_REGNUM,    46,     46 }},
    { "s21",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s21,           LLDB_INVALID_REGNUM,    47,     47 }},
    { "s22",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s22,           LLDB_INVALID_REGNUM,    48,     48 }},
    { "s23",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s23,           LLDB_INVALID_REGNUM,    49,     49 }},
    { "s24",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s24,           LLDB_INVALID_REGNUM,    50,     50 }},
    { "s25",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s25,           LLDB_INVALID_REGNUM,    51,     51 }},
    { "s26",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s26,           LLDB_INVALID_REGNUM,    52,     52 }},
    { "s27",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s27,           LLDB_INVALID_REGNUM,    53,     53 }},
    { "s28",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s28,           LLDB_INVALID_REGNUM,    54,     54 }},
    { "s29",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s29,           LLDB_INVALID_REGNUM,    55,     55 }},
    { "s30",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s30,           LLDB_INVALID_REGNUM,    56,     56 }},
    { "s31",  NULL,   4,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_s31,           LLDB_INVALID_REGNUM,    57,     57 }},
    { "fpscr",NULL,   4,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    58,     58 }},
    { "d16",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d16,           LLDB_INVALID_REGNUM,    59,     59 }},
    { "d17",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d17,           LLDB_INVALID_REGNUM,    60,     60 }},
    { "d18",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d18,           LLDB_INVALID_REGNUM,    61,     61 }},
    { "d19",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d19,           LLDB_INVALID_REGNUM,    62,     62 }},
    { "d20",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d20,           LLDB_INVALID_REGNUM,    63,     63 }},
    { "d21",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d21,           LLDB_INVALID_REGNUM,    64,     64 }},
    { "d22",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d22,           LLDB_INVALID_REGNUM,    65,     65 }},
    { "d23",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d23,           LLDB_INVALID_REGNUM,    66,     66 }},
    { "d24",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d24,           LLDB_INVALID_REGNUM,    67,     67 }},
    { "d25",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d25,           LLDB_INVALID_REGNUM,    68,     68 }},
    { "d26",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d26,           LLDB_INVALID_REGNUM,    69,     69 }},
    { "d27",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d27,           LLDB_INVALID_REGNUM,    70,     70 }},
    { "d28",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d28,           LLDB_INVALID_REGNUM,    71,     71 }},
    { "d29",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d29,           LLDB_INVALID_REGNUM,    72,     72 }},
    { "d30",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d30,           LLDB_INVALID_REGNUM,    73,     73 }},
    { "d31",  NULL,   8,   0, eEncodingIEEE754, eFormatHex,   { LLDB_INVALID_REGNUM, dwarf_d31,           LLDB_INVALID_REGNUM,    74,     74 }},
    };

    static const uint32_t num_registers = sizeof (g_register_infos)/sizeof (RegisterInfo);
    static ConstString gpr_reg_set ("General Purpose Registers");
    static ConstString sfp_reg_set ("Software Floating Point Registers");
    static ConstString vfp_reg_set ("Floating Point Registers");
    uint32_t i;
    // Calculate the offsets of the registers
    if (g_register_infos[2].byte_offset == 0)
    {
        uint32_t byte_offset = 0;
        for (i=0; i<num_registers; ++i)
        {
            g_register_infos[i].byte_offset = byte_offset;
            byte_offset += g_register_infos[i].byte_size;
        }
    }
    for (i=0; i<num_registers; ++i)
    {
        ConstString name;
        ConstString alt_name;
        if (g_register_infos[i].name && g_register_infos[i].name[0])
            name.SetCString(g_register_infos[i].name);
        if (g_register_infos[i].alt_name && g_register_infos[i].alt_name[0])
            alt_name.SetCString(g_register_infos[i].alt_name);
        
        if (i <= 15 || i == 25)
            AddRegister (g_register_infos[i], name, alt_name, gpr_reg_set);
        else if (i <= 24)
            AddRegister (g_register_infos[i], name, alt_name, sfp_reg_set);
        else
            AddRegister (g_register_infos[i], name, alt_name, vfp_reg_set);
    }
}

