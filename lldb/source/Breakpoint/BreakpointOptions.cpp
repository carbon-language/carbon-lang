//===-- BreakpointOptions.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointOptions.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"
#include "lldb/Core/StringList.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"

using namespace lldb;
using namespace lldb_private;

bool
BreakpointOptions::NullCallback (void *baton, StoppointCallbackContext *context, lldb::user_id_t break_id, lldb::user_id_t break_loc_id)
{
    return true;
}

//----------------------------------------------------------------------
// BreakpointOptions constructor
//----------------------------------------------------------------------
BreakpointOptions::BreakpointOptions() :
    m_callback (BreakpointOptions::NullCallback),
    m_callback_is_synchronous (false),
    m_callback_baton_sp (),
    m_enabled (true),
    m_ignore_count (0),
    m_thread_id (LLDB_INVALID_THREAD_ID)
{
}

//----------------------------------------------------------------------
// BreakpointOptions copy constructor
//----------------------------------------------------------------------
BreakpointOptions::BreakpointOptions(const BreakpointOptions& rhs) :
    m_callback (rhs.m_callback),
    m_callback_baton_sp (rhs.m_callback_baton_sp),
    m_callback_is_synchronous (rhs.m_callback_is_synchronous),
    m_enabled (rhs.m_enabled),
    m_ignore_count (rhs.m_ignore_count),
    m_thread_id (rhs.m_thread_id)
{
}

//----------------------------------------------------------------------
// BreakpointOptions assignment operator
//----------------------------------------------------------------------
const BreakpointOptions&
BreakpointOptions::operator=(const BreakpointOptions& rhs)
{
    m_callback = rhs.m_callback;
    m_callback_baton_sp = rhs.m_callback_baton_sp;
    m_callback_is_synchronous = rhs.m_callback_is_synchronous;
    m_enabled = rhs.m_enabled;
    m_ignore_count = rhs.m_ignore_count;
    m_thread_id = rhs.m_thread_id;
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
BreakpointOptions::~BreakpointOptions()
{
}

//------------------------------------------------------------------
// Callbacks
//------------------------------------------------------------------
void
BreakpointOptions::SetCallback (BreakpointHitCallback callback, const BatonSP &callback_baton_sp, bool callback_is_synchronous)
{
    m_callback_is_synchronous = callback_is_synchronous;
    m_callback = callback;
    m_callback_baton_sp = callback_baton_sp;
}

void
BreakpointOptions::ClearCallback ()
{
    m_callback = NULL;
    m_callback_baton_sp.reset();
}

Baton *
BreakpointOptions::GetBaton ()
{
    return m_callback_baton_sp.get();
}

bool
BreakpointOptions::InvokeCallback (StoppointCallbackContext *context, 
                                   lldb::user_id_t break_id,
                                   lldb::user_id_t break_loc_id)
{
    if (m_callback && context->is_synchronous == IsCallbackSynchronous())
    {
        return m_callback (m_callback_baton_sp ? m_callback_baton_sp->m_data : NULL,
                           context, 
                           break_id, 
                           break_loc_id);
    }
    else
        return true;
}

//------------------------------------------------------------------
// Enabled/Ignore Count
//------------------------------------------------------------------
bool
BreakpointOptions::IsEnabled () const
{
    return m_enabled;
}

void
BreakpointOptions::SetEnabled (bool enabled)
{
    m_enabled = enabled;
}

int32_t
BreakpointOptions::GetIgnoreCount () const
{
    return m_ignore_count;
}

void
BreakpointOptions::SetIgnoreCount (int32_t n)
{
    m_ignore_count = n;
}

void
BreakpointOptions::SetThreadID (lldb::tid_t thread_id)
{
    m_thread_id = thread_id;
}

lldb::tid_t
BreakpointOptions::GetThreadID () const
{
    return m_thread_id;
}



void
BreakpointOptions::CommandBaton::GetDescription (Stream *s, lldb::DescriptionLevel level) const
{
    s->Indent("Breakpoint commands:\n");
    CommandData *data = (CommandData *)m_data;
    
    s->IndentMore ();
    if (data && data->user_source.GetSize() > 0)
    {
        const size_t num_strings = data->user_source.GetSize();
        for (size_t i = 0; i < num_strings; ++i)
        {
            s->Indent(data->user_source.GetStringAtIndex(i));
            s->EOL();
        }
    }
    else
    {
        s->PutCString ("No commands.\n");
    }
    s->IndentLess ();
}

