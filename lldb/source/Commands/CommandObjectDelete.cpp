//===-- CommandObjectDelete.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectDelete.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectDelete
//-------------------------------------------------------------------------

CommandObjectDelete::CommandObjectDelete () :
CommandObjectCrossref ("delete", "Lists the kinds of objects you can delete, and shows syntax for deleting them.", "delete")
{
}

CommandObjectDelete::~CommandObjectDelete ()
{
}


