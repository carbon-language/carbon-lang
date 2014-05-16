//===-- MIConfig.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIConfig.h
//
// Overview:	Common defines to guide feature inclusion at compile time.
//
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--
#pragma once

// 1 = Yes compile MI version, 0 = compile original LLDB driver
#define MICONFIG_COMPILE_MIDRIVER_VERSION 1	

// 1 = Show modal dialog, 0 = do not show
#define MICONFIG_DEBUG_SHOW_ATTACH_DBG_DLG 0

// 1 = Compile in and init LLDB driver code alongside MI version, 0 = do not use
// ToDo: This has not been fully implemented as may not be required in the future
#define MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER 0

// 1 = Give runtime our own custom buffer, 0 = Use runtime managed buffer
#define MICONFIG_CREATE_OWN_STDIN_BUFFER 0

// 1 = Use the MI driver regardless of --interpreter, 0 = require --interpreter argument
#define MICONFIG_DEFAULT_TO_MI_DRIVER 1

// 1 = Check for stdin before we issue blocking read, 0 = issue blocking call always
#define MICONFIG_POLL_FOR_STD_IN 1

// Temp workaround while needing different triples
// ToDo: Temp workaround while needing different triples - not used ATM, may not be required anymore
//#define MICONFIG_TRIPLE "arm"

// 1 = Write to MI's Log file warnings about commands that did not handle arguments or
// options present to them by the driver's client, 0 = no warnings given
#define MICONFIG_GIVE_WARNING_CMD_ARGS_NOT_HANDLED 1
