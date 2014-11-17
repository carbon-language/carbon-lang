//===-- MICmnConfig.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmnConfig.h
//
// Overview:    Common defines to guide feature inclusion at compile time.
//
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--
#pragma once

// 1 = Yes compile MI Driver version, 0 = compile original LLDB driver code only.
// 0 was mainly just for testing purposes and so may be removed at a later time.
#define MICONFIG_COMPILE_MIDRIVER_VERSION 1

// 1 = Show debug process attach modal dialog, 0 = do not show
// For windows only ATM, other OS's code is an infinite loop which a debugger must change a value to continue
#define MICONFIG_DEBUG_SHOW_ATTACH_DBG_DLG 0

// 1 = Compile in and init LLDB driver code alongside MI version, 0 = do not compile in
#define MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER 1

// 1 = Give runtime our own custom buffer, 0 = Use runtime managed buffer
#define MICONFIG_CREATE_OWN_STDIN_BUFFER 0

// 1 = Use the MI driver regardless of --interpreter, 0 = require --interpreter argument
// This depends on MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER
#define MICONFIG_DEFAULT_TO_MI_DRIVER 0

// 1 = Check for stdin before we issue blocking read, 0 = issue blocking call always
#define MICONFIG_POLL_FOR_STD_IN 1

// 1 = Write to MI's Log file warnings about commands that did not handle arguments or
// options present to them by the driver's client, 0 = no warnings given
#define MICONFIG_GIVE_WARNING_CMD_ARGS_NOT_HANDLED 1

// 1 = Enable MI Driver in MI mode to create a local debug session, 0 = Report "Not implemented"
#define MICONFIG_ENABLE_MI_DRIVER_MI_MODE_CMDLINE_ARG_EXECUTABLE_DEBUG_SESSION 0
