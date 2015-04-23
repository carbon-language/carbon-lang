//===-- MIUtilSystemWindows.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER)

// Third party headers
#include <memory> // std::unique_ptr
#include <Windows.h>
#include <WinBase.h> // ::FormatMessage()

// In-house headers:
#include "MIUtilSystemWindows.h"
#include "MICmnResources.h"
#include "MIUtilFileStd.h"

//++ ------------------------------------------------------------------------------------
// Details: CMIUtilSystemWindows constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMIUtilSystemWindows::CMIUtilSystemWindows(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMIUtilSystemWindows destructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMIUtilSystemWindows::~CMIUtilSystemWindows(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the OS system error message for the given system error code.
// Type:    Method.
// Args:    vError      - (R) OS error code value.
//          vrwErrorMsg - (W) The error message.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMIUtilSystemWindows::GetOSErrorMsg(const MIint vError, CMIUtilString &vrwErrorMsg) const
{
    // Reset
    vrwErrorMsg.clear();

    const MIuint nBufLen = 1024;
    std::unique_ptr<char[]> pBuffer;
    pBuffer.reset(new char[nBufLen]);

    // CMIUtilString Format is not used as cannot replicate the behavior of ::FormatMessage which
    // can take into account locality while retrieving the error message from the system.
    const int nLength = ::FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, (DWORD)vError,
                                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPTSTR>(&pBuffer[0]), nBufLen, nullptr);
    bool bOk = MIstatus::success;
    if (nLength != 0)
        vrwErrorMsg = &pBuffer[0];
    else
        bOk = MIstatus::failure;

    return bOk;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve if possible the OS last error description.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString - Error description.
// Throws:  None.
//--
CMIUtilString
CMIUtilSystemWindows::GetOSLastError(void) const
{
    CMIUtilString errorMsg;
    const DWORD dwLastError = ::GetLastError();
    if (dwLastError != 0)
    {
        if (!GetOSErrorMsg(dwLastError, errorMsg))
            errorMsg = MIRSRC(IDE_OS_ERR_RETRIEVING);
    }
    else
        errorMsg = MIRSRC(IDE_OS_ERR_UNKNOWN);

    return errorMsg;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieves the fully qualified path for the this application. If the function
//          fails the string is filled with the error message.
// Type:    Method.
// Args:    vrwFileNamePath   - (W) The excutable's name and path or last error description.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMIUtilSystemWindows::GetExecutablesPath(CMIUtilString &vrwFileNamePath) const
{
    bool bOk = MIstatus::success;
    HMODULE hModule = ::GetModuleHandle(nullptr);
    char pPath[MAX_PATH];
    if (!::GetModuleFileName(hModule, &pPath[0], MAX_PATH))
    {
        bOk = MIstatus::failure;
        vrwFileNamePath = GetOSLastError();
    }
    else
        vrwFileNamePath = &pPath[0];

    return bOk;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieves the fully qualified path for the Log file for this application.
//          If the function fails the string is filled with the error message.
// Type:    Method.
// Args:    vrwFileNamePath   - (W) The Log file's name and path or last error description.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMIUtilSystemWindows::GetLogFilesPath(CMIUtilString &vrwFileNamePath) const
{
    vrwFileNamePath = CMIUtilString(".");
    return MIstatus::success;
}

#endif // #if defined( _MSC_VER )
