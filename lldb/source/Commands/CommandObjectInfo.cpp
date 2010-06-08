//===-- CommandObjectInfo.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectInfo.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectInfo
//-------------------------------------------------------------------------

CommandObjectInfo::CommandObjectInfo () :
CommandObjectCrossref ("info", "Lists the kinds of objects for which you can get information, and shows the syntax for doing so.", "info")
{
}

CommandObjectInfo::~CommandObjectInfo ()
{
}


