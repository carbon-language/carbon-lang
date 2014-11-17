//===-- MICmnLLDBBroadcaster.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmnLLDBBroadcaster.cpp
//
// Overview:    CMICmnLLDBBroadcaster implementation.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// In-house headers:
#include "MICmnLLDBBroadcaster.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmnLLDBBroadcaster constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnLLDBBroadcaster::CMICmnLLDBBroadcaster(void)
    : lldb::SBBroadcaster("MI driver")
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmnLLDBBroadcaster destructor.
// Type:    Overridable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnLLDBBroadcaster::~CMICmnLLDBBroadcaster(void)
{
    Shutdown();
}

//++ ------------------------------------------------------------------------------------
// Details: Initialize resources for *this broardcaster object.
// Type:    Method.
// Args:    None.
// Return:  MIstatus::success - Functionality succeeded.
//          MIstatus::failure - Functionality failed.
// Throws:  None.
//--
bool
CMICmnLLDBBroadcaster::Initialize(void)
{
    m_clientUsageRefCnt++;

    if (m_bInitialized)
        return MIstatus::success;

    m_bInitialized = MIstatus::success;

    return m_bInitialized;
}

//++ ------------------------------------------------------------------------------------
// Details: Release resources for *this broardcaster object.
// Type:    Method.
// Args:    None.
// Return:  MIstatus::success - Functionality succeeded.
//          MIstatus::failure - Functionality failed.
// Throws:  None.
//--
bool
CMICmnLLDBBroadcaster::Shutdown(void)
{
    if (--m_clientUsageRefCnt > 0)
        return MIstatus::success;

    if (!m_bInitialized)
        return MIstatus::success;

    m_bInitialized = false;

    return MIstatus::success;
}
