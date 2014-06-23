//===-- SBUnixSignals.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-defines.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Core/Log.h"

#include "lldb/API/SBUnixSignals.h"

using namespace lldb;
using namespace lldb_private;

SBUnixSignals::SBUnixSignals ()
{}

SBUnixSignals::SBUnixSignals (const SBUnixSignals &rhs) :
    m_opaque_wp(rhs.m_opaque_wp)
{
}

SBUnixSignals::SBUnixSignals (ProcessSP &process_sp) :
    m_opaque_wp(process_sp)
{
}

const SBUnixSignals&
SBUnixSignals::operator = (const SBUnixSignals& rhs)
{
    if (this != &rhs)
        m_opaque_wp = rhs.m_opaque_wp;
    return *this;
}

SBUnixSignals::~SBUnixSignals()
{
}

ProcessSP
SBUnixSignals::GetSP() const
{
    return m_opaque_wp.lock();
}

void
SBUnixSignals::SetSP (const ProcessSP &process_sp)
{
    m_opaque_wp = process_sp;
}

void
SBUnixSignals::Clear ()
{
    m_opaque_wp.reset();
}

bool
SBUnixSignals::IsValid() const
{
    return (bool) GetSP();
}

const char *
SBUnixSignals::GetSignalAsCString (int32_t signo) const
{
    ProcessSP process_sp(GetSP());
    if (process_sp) return process_sp->GetUnixSignals().GetSignalAsCString(signo);
    return NULL;
}

int32_t
SBUnixSignals::GetSignalNumberFromName (const char *name) const
{
    ProcessSP process_sp(GetSP());
    if (process_sp) return process_sp->GetUnixSignals().GetSignalNumberFromName(name);
    return -1;
}

bool
SBUnixSignals::GetShouldSuppress (int32_t signo) const
{
    ProcessSP process_sp(GetSP());
    if (process_sp) return process_sp->GetUnixSignals().GetShouldSuppress(signo);
    return false;
}

bool
SBUnixSignals::SetShouldSuppress (int32_t signo, bool value)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    ProcessSP process_sp(GetSP());

    if (log)
    {
        log->Printf ("SBUnixSignals(%p)::SetShouldSuppress (signo=%d, value=%d)",
                     static_cast<void*>(process_sp.get()),
                     signo,
                     value);
    }

    if (process_sp) return process_sp->GetUnixSignals().SetShouldSuppress(signo, value);
    return false;
}

bool
SBUnixSignals::GetShouldStop (int32_t signo) const
{
    ProcessSP process_sp(GetSP());
    if (process_sp) return process_sp->GetUnixSignals().GetShouldStop(signo);
    return false;
}

bool
SBUnixSignals::SetShouldStop (int32_t signo, bool value)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    ProcessSP process_sp(GetSP());

    if (log)
    {
        log->Printf ("SBUnixSignals(%p)::SetShouldStop (signo=%d, value=%d)",
                     static_cast<void*>(process_sp.get()),
                     signo,
                     value);
    }

    if (process_sp) return process_sp->GetUnixSignals().SetShouldStop(signo, value);
    return false;
}

bool
SBUnixSignals::GetShouldNotify (int32_t signo) const
{
    ProcessSP process_sp(GetSP());
    if (process_sp) return process_sp->GetUnixSignals().GetShouldNotify(signo);
    return false;
}

bool
SBUnixSignals::SetShouldNotify (int32_t signo, bool value)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    ProcessSP process_sp(GetSP());

    if (log)
    {
        log->Printf ("SBUnixSignals(%p)::SetShouldNotify (signo=%d, value=%d)",
                     static_cast<void*>(process_sp.get()),
                     signo,
                     value);
    }

    if (process_sp) return process_sp->GetUnixSignals().SetShouldNotify(signo, value);
    return false;
}

int32_t
SBUnixSignals::GetNumSignals () const
{
    if (auto process_sp = GetSP())
    {
        // only valid while we hold process_sp
        UnixSignals *unix_signals_ptr = &process_sp->GetUnixSignals();
        int32_t num_signals = 0;
        for (int32_t signo = unix_signals_ptr->GetFirstSignalNumber();
             signo != LLDB_INVALID_SIGNAL_NUMBER;
             signo = unix_signals_ptr->GetNextSignalNumber(signo))
        {
            num_signals++;
        }
        return num_signals;
    }
    return LLDB_INVALID_SIGNAL_NUMBER;
}

int32_t
SBUnixSignals::GetSignalAtIndex (int32_t index) const
{
    if (auto process_sp = GetSP())
    {
        // only valid while we hold process_sp
        UnixSignals *unix_signals_ptr = &process_sp->GetUnixSignals();
        int32_t idx = 0;
        for (int32_t signo = unix_signals_ptr->GetFirstSignalNumber();
             signo != LLDB_INVALID_SIGNAL_NUMBER;
             signo = unix_signals_ptr->GetNextSignalNumber(signo))
        {
            if (index == idx) return signo;
            idx++;
        }
    }
    return LLDB_INVALID_SIGNAL_NUMBER;
}
