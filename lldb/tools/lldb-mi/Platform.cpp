//===-- Platform.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// this file is only relevant for Visual C++
#if defined(_MSC_VER)

#include <process.h>
#include <assert.h>

#include "Platform.h"

// the control handler or SIGINT handler
static sighandler_t _ctrlHandler = NULL;

// the default console control handler
BOOL WINAPI CtrlHandler(DWORD ctrlType)
{
    if (_ctrlHandler != NULL)
    {
        _ctrlHandler(SIGINT);
        return TRUE;
    }
    return FALSE;
}

sighandler_t
signal(int sig, sighandler_t sigFunc)
{
    switch (sig)
    {
        case (SIGINT):
        {
            _ctrlHandler = sigFunc;
            SetConsoleCtrlHandler(CtrlHandler, TRUE);
        }
        break;
        default:
            assert(!"Not implemented!");
    }
    return 0;
}

#endif
