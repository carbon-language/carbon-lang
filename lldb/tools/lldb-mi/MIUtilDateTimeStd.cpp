//===-- MIUtilDateTimeStd.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MIUtilDateTimeStd.cpp
//
// Overview:    CMIUtilDateTimeStd implementation.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// In-house headers:
#include "MIUtilDateTimeStd.h"
#include "MICmnResources.h"

//++ ------------------------------------------------------------------------------------
// Details: CMIUtilDateTimeStd constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMIUtilDateTimeStd::CMIUtilDateTimeStd(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMIUtilDateTimeStd destructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMIUtilDateTimeStd::~CMIUtilDateTimeStd(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve system local current date. Format is MM/DD/YYYY.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString - Text description.
// Throws:  None.
//--
CMIUtilString
CMIUtilDateTimeStd::GetDate(void)
{
    CMIUtilString strDate(MIRSRC(IDS_WORD_INVALIDBRKTS));

    std::time(&m_rawTime);
    const std::tm *pTi = std::localtime(&m_rawTime);
    if (std::strftime(&m_pScratch[0], sizeof(m_pScratch), "%d/%m/%y", pTi) > 0)
        strDate = m_pScratch;

    return strDate;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve system local current time. Format is HH:MM:SS 24 hour clock.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString - Text description.
// Throws:  None.
//--
CMIUtilString
CMIUtilDateTimeStd::GetTime(void)
{
    std::time(&m_rawTime);
    const std::tm *pTi = std::localtime(&m_rawTime);
    const CMIUtilString seconds(CMIUtilString::Format("%d", pTi->tm_sec));
    const CMIUtilString zero((seconds.length() == 1) ? "0" : "");
    const CMIUtilString strTime(CMIUtilString::Format("%d:%d:%s%s", pTi->tm_hour, pTi->tm_min, zero.c_str(), seconds.c_str()));

    return strTime;
}
