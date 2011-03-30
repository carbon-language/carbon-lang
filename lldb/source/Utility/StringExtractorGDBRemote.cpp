//===-- StringExtractorGDBRemote.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <string.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "Utility/StringExtractorGDBRemote.h"



StringExtractorGDBRemote::ResponseType
StringExtractorGDBRemote::GetResponseType () const
{
    if (m_packet.empty())
        return eUnsupported;

    switch (m_packet[0])
    {
    case 'E':
        if (m_packet.size() == 3 &&
            isxdigit(m_packet[1]) &&
            isxdigit(m_packet[2]))
            return eError;
        break;

    case 'O':
        if (m_packet.size() == 2 && m_packet[1] == 'K')
            return eOK;
        break;

    case '+':
        if (m_packet.size() == 1)
            return eAck;
        break;

    case '-':
        if (m_packet.size() == 1)
            return eNack;
        break;
    }
    return eResponse;
}

StringExtractorGDBRemote::ServerPacketType
StringExtractorGDBRemote::GetServerPacketType () const
{
    // Empty is not a supported packet...
    if (m_packet.empty())
        return eServerPacketType_invalid;

    const char *packet_cstr = m_packet.c_str();
    switch (m_packet[0])
    {
    case '\x03':
        if (m_packet.size() == 1)
            return eServerPacketType_interrupt;
        break;

    case '-':
        if (m_packet.size() == 1)
            return eServerPacketType_nack;
        break;

    case '+':
        if (m_packet.size() == 1)
            return eServerPacketType_ack;
        break;

    case 'Q':
        if (strcmp (packet_cstr, "QStartNoAckMode") == 0)
            return eServerPacketType_QStartNoAckMode;
        break;
            
    case 'q':
        if (packet_cstr[1] == 'H' && 0 == ::strcmp (packet_cstr, "qHostInfo"))
            return eServerPacketType_qHostInfo;
        else if (packet_cstr[1] == 'P' && 0 == ::strncmp(packet_cstr, "qProcessInfoPID:", strlen("qProcessInfoPID:")))
            return eServerPacketType_qProcessInfoPID;
        else if (packet_cstr[1] == 'f' && 0 == ::strncmp(packet_cstr, "qfProcessInfo", strlen("qfProcessInfo")))
            return eServerPacketType_qfProcessInfo;
        else if (packet_cstr[1] == 'U' && 0 == ::strncmp(packet_cstr, "qUserName:", strlen("qUserName:")))
            return eServerPacketType_qUserName;
        else if (packet_cstr[1] == 'G' && 0 == ::strncmp(packet_cstr, "qGroupName:", strlen("qGroupName:")))
            return eServerPacketType_qGroupName;
        else if (packet_cstr[1] == 's' && 0 == ::strcmp (packet_cstr, "qsProcessInfo"))
            return eServerPacketType_qsProcessInfo;
        break;
    }
    return eServerPacketType_unimplemented;
}

bool
StringExtractorGDBRemote::IsOKResponse() const
{
    return GetResponseType () == eOK;
}


bool
StringExtractorGDBRemote::IsUnsupportedResponse() const
{
    return GetResponseType () == eUnsupported;
}

bool
StringExtractorGDBRemote::IsNormalResponse() const
{
    return GetResponseType () == eResponse;
}

bool
StringExtractorGDBRemote::IsErrorResponse() const
{
    return GetResponseType () == eError &&
           m_packet.size() == 3 &&
           isxdigit(m_packet[1]) &&
           isxdigit(m_packet[2]);
}

uint8_t
StringExtractorGDBRemote::GetError ()
{
    if (GetResponseType() == eError)
    {
        SetFilePos(1);
        return GetHexU8(255);
    }
    return 0;
}
