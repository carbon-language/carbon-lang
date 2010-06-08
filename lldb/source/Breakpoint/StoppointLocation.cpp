//===-- StoppointLocation.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/StoppointLocation.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// StoppointLocation constructor
//----------------------------------------------------------------------
StoppointLocation::StoppointLocation (break_id_t bid, addr_t addr, tid_t tid, bool hardware) :
    m_loc_id(bid),
    m_tid(tid),
    m_byte_size(0),
    m_addr(addr),
    m_hit_count(0),
    m_hw_preferred(hardware),
    m_hw_index(LLDB_INVALID_INDEX32)
{
}

StoppointLocation::StoppointLocation (break_id_t bid, addr_t addr, tid_t tid, size_t size, bool hardware) :
    m_loc_id(bid),
    m_tid(tid),
    m_byte_size(size),
    m_addr(addr),
    m_hit_count(0),
    m_hw_preferred(hardware),
    m_hw_index(LLDB_INVALID_INDEX32)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
StoppointLocation::~StoppointLocation()
{
}


size_t
StoppointLocation::GetByteSize () const
{
    return m_byte_size;
}

addr_t
StoppointLocation::GetLoadAddress() const
{
    return m_addr;
}

tid_t
StoppointLocation::GetThreadID() const
{
    return m_tid;
}

uint32_t
StoppointLocation::GetHitCount () const
{
    return m_hit_count;
}

bool
StoppointLocation::HardwarePreferred () const
{
    return m_hw_preferred;
}

bool
StoppointLocation::IsHardware () const
{
    return m_hw_index != LLDB_INVALID_INDEX32;
}

uint32_t
StoppointLocation::GetHardwareIndex () const
{
    return m_hw_index;
}

void
StoppointLocation::SetHardwareIndex (uint32_t hw_index)
{
    m_hw_index = hw_index;
}

// RETURNS - true if we should stop at this breakpoint, false if we
// should continue.

bool
StoppointLocation::ShouldStop (StoppointCallbackContext *context)
{
    return true;
}

break_id_t
StoppointLocation::GetID() const
{
    return m_loc_id;
}

void
StoppointLocation::Dump (Stream *stream) const
{

}
