//===-- StringExtractorGDBRemote.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "StringExtractorGDBRemote.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes



StringExtractorGDBRemote::Type
StringExtractorGDBRemote::GetType () const
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

bool
StringExtractorGDBRemote::IsOKPacket() const
{
    return GetType () == eOK;
}


bool
StringExtractorGDBRemote::IsUnsupportedPacket() const
{
    return GetType () == eUnsupported;
}

bool
StringExtractorGDBRemote::IsNormalPacket() const
{
    return GetType () == eResponse;
}

bool
StringExtractorGDBRemote::IsErrorPacket() const
{
    return GetType () == eError &&
           m_packet.size() == 3 &&
           isxdigit(m_packet[1]) &&
           isxdigit(m_packet[2]);
}

uint8_t
StringExtractorGDBRemote::GetError ()
{
    if (GetType() == eError)
    {
        SetFilePos(1);
        return GetHexU8(255);
    }
    return 0;
}
