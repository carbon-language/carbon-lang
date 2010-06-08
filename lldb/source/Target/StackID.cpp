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
    m_cfa()
{
}

//----------------------------------------------------------------------
// StackID constructor with args
//----------------------------------------------------------------------
StackID::StackID (const Address& start_address, lldb::addr_t cfa) :
    m_start_address (start_address),
    m_cfa (cfa)
{
}

StackID::StackID (lldb::addr_t cfa) :
    m_start_address (),
    m_cfa (cfa)
{
}

//----------------------------------------------------------------------
// StackID copy constructor
//----------------------------------------------------------------------
StackID::StackID(const StackID& rhs) :
    m_start_address (rhs.m_start_address),
    m_cfa (rhs.m_cfa)
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
    }
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
StackID::~StackID()
{
}


const Address&
StackID::GetStartAddress() const
{
    return m_start_address;
}

void
StackID::SetStartAddress(const Address& start_address)
{
    m_start_address = start_address;
}

lldb::addr_t
StackID::GetCallFrameAddress() const
{
    return m_cfa;
}


bool
lldb_private::operator== (const StackID& lhs, const StackID& rhs)
{
    return lhs.GetCallFrameAddress() == rhs.GetCallFrameAddress() && lhs.GetStartAddress() == rhs.GetStartAddress();
}

bool
lldb_private::operator!= (const StackID& lhs, const StackID& rhs)
{
    return lhs.GetCallFrameAddress() != rhs.GetCallFrameAddress() || lhs.GetStartAddress() != rhs.GetStartAddress();
}

bool
lldb_private::operator< (const StackID& lhs, const StackID& rhs)
{
    return lhs.GetCallFrameAddress() < rhs.GetCallFrameAddress();
}

