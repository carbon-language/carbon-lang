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

//----------------------------------------------------------------------
// StackID constructor
//----------------------------------------------------------------------
StackID::StackID() :
    m_start_address(),
    m_cfa (0),
    m_inline_height (0)
{
}

//----------------------------------------------------------------------
// StackID constructor with args
//----------------------------------------------------------------------
StackID::StackID (const Address& start_address, lldb::addr_t cfa, uint32_t inline_id) :
    m_start_address (start_address),
    m_cfa (cfa),
    m_inline_height (inline_id)
{
}

StackID::StackID (lldb::addr_t cfa, uint32_t inline_id) :
    m_start_address (),
    m_cfa (cfa),
    m_inline_height (inline_id)
{
}

//----------------------------------------------------------------------
// StackID copy constructor
//----------------------------------------------------------------------
StackID::StackID(const StackID& rhs) :
    m_start_address (rhs.m_start_address),
    m_cfa (rhs.m_cfa),
    m_inline_height (rhs.m_inline_height)
{
}

//----------------------------------------------------------------------
// StackID assignment operator
//----------------------------------------------------------------------
const StackID&
StackID::operator=(const StackID& rhs)
{
    if (this != &rhs)
    {
        m_start_address = rhs.m_start_address;
        m_cfa = rhs.m_cfa;
        m_inline_height = rhs.m_inline_height;
    }
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
StackID::~StackID()
{
}

bool
lldb_private::operator== (const StackID& lhs, const StackID& rhs)
{
    return  lhs.GetCallFrameAddress()   == rhs.GetCallFrameAddress() && 
            lhs.GetInlineHeight()       == rhs.GetInlineHeight() &&
            lhs.GetStartAddress()       == rhs.GetStartAddress();
}

bool
lldb_private::operator!= (const StackID& lhs, const StackID& rhs)
{
    return lhs.GetCallFrameAddress() != rhs.GetCallFrameAddress() || 
           lhs.GetInlineHeight()     != rhs.GetInlineHeight() || 
           lhs.GetStartAddress()     != rhs.GetStartAddress();
}

bool
lldb_private::operator< (const StackID& lhs, const StackID& rhs)
{
    return lhs.GetCallFrameAddress() < rhs.GetCallFrameAddress();
}

