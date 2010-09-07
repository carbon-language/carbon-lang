//===-- TimeValue.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/TimeValue.h"
#include <stddef.h>

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#define NSEC_PER_USEC   1000ull
#define USEC_PER_SEC    1000000ull
#define NSEC_PER_SEC    1000000000ull

using namespace lldb_private;

//----------------------------------------------------------------------
// TimeValue constructor
//----------------------------------------------------------------------
TimeValue::TimeValue() :
    m_nano_seconds (0)
{
}

//----------------------------------------------------------------------
// TimeValue copy constructor
//----------------------------------------------------------------------
TimeValue::TimeValue(const TimeValue& rhs) :
    m_nano_seconds (rhs.m_nano_seconds)
{
}

TimeValue::TimeValue(const struct timespec& ts) :
    m_nano_seconds (ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec)
{
}

TimeValue::TimeValue(const struct timeval& tv) :
    m_nano_seconds (tv.tv_sec * NSEC_PER_SEC + tv.tv_usec * NSEC_PER_USEC)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
TimeValue::~TimeValue()
{
}


uint64_t
TimeValue::GetAsNanoSecondsSinceJan1_1970() const
{
    return m_nano_seconds;
}

uint64_t
TimeValue::GetAsMicroSecondsSinceJan1_1970() const
{
    return m_nano_seconds / NSEC_PER_USEC;
}

struct timespec
TimeValue::GetAsTimeSpec () const
{
    struct timespec ts;
    ts.tv_sec = m_nano_seconds / NSEC_PER_SEC;
    ts.tv_nsec = m_nano_seconds % NSEC_PER_SEC;
    return ts;
}

struct timeval
TimeValue::GetAsTimeVal () const
{
    struct timeval tv;
    tv.tv_sec = m_nano_seconds / NSEC_PER_SEC;
    tv.tv_usec = (m_nano_seconds % NSEC_PER_SEC) / NSEC_PER_USEC;
    return tv;
}

void
TimeValue::Clear ()
{
    m_nano_seconds = 0;
}

bool
TimeValue::IsValid () const
{
    return m_nano_seconds != 0;
}

void
TimeValue::OffsetWithSeconds (uint64_t sec)
{
    m_nano_seconds += sec * NSEC_PER_SEC;
}

void
TimeValue::OffsetWithMicroSeconds (uint64_t usec)
{
    m_nano_seconds += usec * NSEC_PER_USEC;
}

void
TimeValue::OffsetWithNanoSeconds (uint64_t nsec)
{
    m_nano_seconds += nsec;
}

TimeValue
TimeValue::Now()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    TimeValue now(tv);
    return now;
}

//----------------------------------------------------------------------
// TimeValue assignment operator
//----------------------------------------------------------------------
const TimeValue&
TimeValue::operator=(const TimeValue& rhs)
{
    m_nano_seconds = rhs.m_nano_seconds;
    return *this;
}


bool
lldb_private::operator == (const TimeValue &lhs, const TimeValue &rhs)
{
    return lhs.GetAsNanoSecondsSinceJan1_1970() == rhs.GetAsNanoSecondsSinceJan1_1970();
}

bool
lldb_private::operator != (const TimeValue &lhs, const TimeValue &rhs)
{
    return lhs.GetAsNanoSecondsSinceJan1_1970() != rhs.GetAsNanoSecondsSinceJan1_1970();
}

bool
lldb_private::operator <  (const TimeValue &lhs, const TimeValue &rhs)
{
    return lhs.GetAsNanoSecondsSinceJan1_1970() < rhs.GetAsNanoSecondsSinceJan1_1970();
}

bool
lldb_private::operator <= (const TimeValue &lhs, const TimeValue &rhs)
{
    return lhs.GetAsNanoSecondsSinceJan1_1970() <= rhs.GetAsNanoSecondsSinceJan1_1970();
}

bool
lldb_private::operator >  (const TimeValue &lhs, const TimeValue &rhs)
{
    return lhs.GetAsNanoSecondsSinceJan1_1970() > rhs.GetAsNanoSecondsSinceJan1_1970();
}

bool
lldb_private::operator >= (const TimeValue &lhs, const TimeValue &rhs)
{
    return lhs.GetAsNanoSecondsSinceJan1_1970() >= rhs.GetAsNanoSecondsSinceJan1_1970();
}

uint64_t
lldb_private::operator - (const TimeValue &lhs, const TimeValue &rhs)
{
    return lhs.GetAsNanoSecondsSinceJan1_1970() - rhs.GetAsNanoSecondsSinceJan1_1970();
}


