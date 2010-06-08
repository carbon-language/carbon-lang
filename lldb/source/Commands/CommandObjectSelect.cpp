//===-- CommandObjectSelect.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectSelect.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectSelect
//-------------------------------------------------------------------------

CommandObjectSelect::CommandObjectSelect () :
    CommandObjectCrossref ("select", "Lists the kinds of objects you can select, and shows syntax for selecting them.", "select")
{
}

CommandObjectSelect::~CommandObjectSelect ()
{
}


