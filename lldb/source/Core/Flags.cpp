//===-- Flags.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Flags.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// Default Constructor
//----------------------------------------------------------------------
Flags::Flags (ValueType flags) :
    m_flags(flags)
{
}

//----------------------------------------------------------------------
// Copy Constructor
//----------------------------------------------------------------------
Flags::Flags (const Flags& rhs) :
    m_flags(rhs.m_flags)
{
}

//----------------------------------------------------------------------
// Virtual destructor in case anyone inherits from this class.
//----------------------------------------------------------------------
Flags::~Flags ()
{
}

//----------------------------------------------------------------------
// Get accessor for all of the current flag bits.
//----------------------------------------------------------------------
Flags::ValueType
Flags::GetAllFlagBits () const
{
    return m_flags;
}

size_t
Flags::GetBitSize() const
{
    return sizeof (ValueType) * 8;
}

//----------------------------------------------------------------------
// Set accessor for all of the current flag bits.
//----------------------------------------------------------------------
void
Flags::SetAllFlagBits (ValueType flags)
{
    m_flags = flags;
}

//----------------------------------------------------------------------
// Clear one or more bits in our flag bits
//----------------------------------------------------------------------
Flags::ValueType
Flags::Clear (ValueType bits)
{
    m_flags &= ~bits;
    return m_flags;
}

//----------------------------------------------------------------------
// Set one or more bits in our flag bits
//----------------------------------------------------------------------
Flags::ValueType
Flags::Set (ValueType bits)
{
    m_flags |= bits;
    return m_flags;
}

//----------------------------------------------------------------------
// Returns true if any flag bits in "bits" are set
//----------------------------------------------------------------------
bool
Flags::IsSet (ValueType bits) const
{
    return (m_flags & bits) != 0;
}

//----------------------------------------------------------------------
// Returns true if all flag bits in "bits" are clear
//----------------------------------------------------------------------
bool
Flags::IsClear (ValueType bits) const
{
    return (m_flags & bits) == 0;
}


size_t
Flags::SetCount () const
{
    size_t count = 0;
    for (ValueType mask = m_flags; mask; mask >>= 1)
    {
        if (mask & 1)
            ++count;
    }
    return count;
}

size_t
Flags::ClearCount () const
{
    size_t count = 0;
    for (ValueType shift = 0; shift < sizeof(ValueType)*8; ++shift)
    {
        if ((m_flags & (1u << shift)) == 0)
            ++count;
    }
    return count;
}
