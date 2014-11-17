//===-- MIUtilTermios.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MIUtilTermios.cpp
//
// Overview:    Terminal setting termios functions.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// Third party headers:
#include <stdlib.h>

// In-house headers:
#include "MIUtilTermios.h"
#include "Platform.h"

namespace MIUtilTermios
{
// Instantiations:
static bool g_bOldStdinTermiosIsValid = false; // True = yes valid, false = no valid
static struct termios g_sOldStdinTermios;

//++ ------------------------------------------------------------------------------------
// Details: Reset the terminal settings. This function is added as an ::atexit handler
//          to make sure we clean up. See StdinTerminosSet().
// Type:    Global function.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
void
StdinTermiosReset(void)
{
    if (g_bOldStdinTermiosIsValid)
    {
        g_bOldStdinTermiosIsValid = false;
        ::tcsetattr(STDIN_FILENO, TCSANOW, &g_sOldStdinTermios);
    }
}

//++ ------------------------------------------------------------------------------------
// Details: Set the terminal settings function. StdinTermiosReset() is called when to
//          reset to this to before and application exit.
// Type:    Global function.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
void
StdinTermiosSet(void)
{
    if (::tcgetattr(STDIN_FILENO, &g_sOldStdinTermios) == 0)
    {
        g_bOldStdinTermiosIsValid = true;
        ::atexit(StdinTermiosReset);
    }
}

} // namespace MIUtilTermios
