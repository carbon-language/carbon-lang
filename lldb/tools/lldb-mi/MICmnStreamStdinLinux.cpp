//===-- MICmnStreamStdinLinux.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MIUtilStreamStdin.cpp
//
// Overview:    CMICmnStreamStdinLinux implementation.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// Third Party Headers:
#if !defined(_MSC_VER)
#include <sys/select.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#endif              // !defined( _MSC_VER )
#include <string.h> // For std::strerror()

// In-house headers:
#include "MICmnStreamStdinLinux.h"
#include "MICmnLog.h"
#include "MICmnResources.h"
#include "MIUtilSingletonHelper.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmnStreamStdinLinux constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnStreamStdinLinux::CMICmnStreamStdinLinux(void)
    : m_constBufferSize(1024)
    , m_pStdin(nullptr)
    , m_pCmdBuffer(nullptr)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmnStreamStdinLinux destructor.
// Type:    Overridable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnStreamStdinLinux::~CMICmnStreamStdinLinux(void)
{
    Shutdown();
}

//++ ------------------------------------------------------------------------------------
// Details: Initialize resources for *this Stdin stream.
// Type:    Method.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmnStreamStdinLinux::Initialize(void)
{
    if (m_bInitialized)
        return MIstatus::success;

    bool bOk = MIstatus::success;
    CMIUtilString errMsg;

    // Note initialisation order is important here as some resources depend on previous
    MI::ModuleInit<CMICmnLog>(IDS_MI_INIT_ERR_LOG, bOk, errMsg);
    MI::ModuleInit<CMICmnResources>(IDS_MI_INIT_ERR_RESOURCES, bOk, errMsg);

    // Other resources required
    if (bOk)
    {
        m_pCmdBuffer = new MIchar[m_constBufferSize];
        m_pStdin = stdin;
    }

    // Clear error indicators for std input
    ::clearerr(stdin);

    m_bInitialized = bOk;

    if (!bOk)
    {
        CMIUtilString strInitError(CMIUtilString::Format(MIRSRC(IDS_MI_INIT_ERR_OS_STDIN_HANDLER), errMsg.c_str()));
        SetErrorDescription(strInitError);
        return MIstatus::failure;
    }

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Release resources for *this Stdin stream.
// Type:    Method.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmnStreamStdinLinux::Shutdown(void)
{
    if (!m_bInitialized)
        return MIstatus::success;

    m_bInitialized = false;

    ClrErrorDescription();

    bool bOk = MIstatus::success;
    CMIUtilString errMsg;

    // Tidy up
    if (m_pCmdBuffer != nullptr)
    {
        delete[] m_pCmdBuffer;
        m_pCmdBuffer = nullptr;
    }
    m_pStdin = nullptr;

    // Note shutdown order is important here
    MI::ModuleShutdown<CMICmnResources>(IDS_MI_INIT_ERR_RESOURCES, bOk, errMsg);
    MI::ModuleShutdown<CMICmnLog>(IDS_MI_INIT_ERR_LOG, bOk, errMsg);

    if (!bOk)
    {
        SetErrorDescriptionn(MIRSRC(IDS_MI_SHTDWN_ERR_OS_STDIN_HANDLER), errMsg.c_str());
    }

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Determine if stdin has any characters present in its buffer.
// Type:    Method.
// Args:    vwbAvail    - (W) True = There is chars available, false = nothing there.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmnStreamStdinLinux::InputAvailable(bool &vwbAvail)
{
#if !defined(_WIN32)
    // The code below is needed on OSX where lldb-mi hangs when doing -exec-run.
    // The hang seems to come from calling fgets and fileno from different thread.
    // Although this problem was not observed on Linux.
    // A solution based on 'select' was also proposed but it seems to slow things down
    // a lot.
    int nBytesWaiting;
    if (::ioctl(STDIN_FILENO, FIONREAD, &nBytesWaiting) == -1)
    {
        vwbAvail = false;
        return MIstatus::failure;;
    }
    vwbAvail = (nBytesWaiting > 0);
#endif // !defined( _WIN32 )
    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Wait on new line of data from stdin stream (completed by '\n' or '\r').
// Type:    Method.
// Args:    vwErrMsg    - (W) Empty string ok or error description.
// Return:  MIchar * - text buffer pointer or NULL on failure.
// Throws:  None.
//--
const MIchar *
CMICmnStreamStdinLinux::ReadLine(CMIUtilString &vwErrMsg)
{
    vwErrMsg.clear();

    // Read user input
    const MIchar *pText = ::fgets(&m_pCmdBuffer[0], m_constBufferSize, stdin);
    if (pText == nullptr)
    {
        if (::ferror(m_pStdin) != 0)
            vwErrMsg = ::strerror(errno);
        return nullptr;
    }

    // Strip off new line characters
    for (MIchar *pI = m_pCmdBuffer; *pI != '\0'; pI++)
    {
        if ((*pI == '\n') || (*pI == '\r'))
        {
            *pI = '\0';
            break;
        }
    }

    return pText;
}

