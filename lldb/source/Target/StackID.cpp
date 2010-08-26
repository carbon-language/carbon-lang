//===-- StackID.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackID.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb_private;


bool
lldb_private::operator== (const StackID& lhs, const StackID& rhs)
{
    return lhs.GetCallFrameAddress()    == rhs.GetCallFrameAddress()    && 
           lhs.GetInlineBlockID()       == rhs.GetInlineBlockID()       &&
           lhs.GetStartAddress()        == rhs.GetStartAddress();
}

bool
lldb_private::operator!= (const StackID& lhs, const StackID& rhs)
{
    return lhs.GetCallFrameAddress()    != rhs.GetCallFrameAddress()    || 
           lhs.GetInlineBlockID()       != rhs.GetInlineBlockID()       || 
           lhs.GetStartAddress()        != rhs.GetStartAddress();
}

bool
lldb_private::operator< (const StackID& lhs, const StackID& rhs)
{
    if (lhs.GetCallFrameAddress() < rhs.GetCallFrameAddress())
        return true;
    return lhs.GetInlineBlockID() < rhs.GetInlineBlockID();
}
