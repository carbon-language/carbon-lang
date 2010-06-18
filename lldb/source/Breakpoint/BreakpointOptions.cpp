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
#include "lldb/Target/ThreadSpec.h"

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
    m_thread_spec_ap (NULL)
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
    m_thread_spec_ap (NULL)
{
    if (rhs.m_thread_spec_ap.get() != NULL)
        m_thread_spec_ap.reset (new ThreadSpec(*rhs.m_thread_spec_ap.get()));
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
    if (rhs.m_thread_spec_ap.get() != NULL)
        m_thread_spec_ap.reset(new ThreadSpec(*rhs.m_thread_spec_ap.get()));
    return *this;
}

BreakpointOptions *
BreakpointOptions::CopyOptionsNoCallback (BreakpointOptions &orig)
{
    BreakpointHitCallback orig_callback = orig.m_callback;
    lldb::BatonSP orig_callback_baton_sp = orig.m_callback_baton_sp;
    bool orig_is_sync = orig.m_callback_is_synchronous;
    
    orig.ClearCallback();
    BreakpointOptions *ret_val = new BreakpointOptions(orig);
    
    orig.SetCallback (orig_callback, orig_callback_baton_sp, orig_is_sync);
    
    return ret_val;
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

const Baton *
BreakpointOptions::GetBaton () const
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

bool
BreakpointOptions::HasCallback ()
{
    return m_callback != BreakpointOptions::NullCallback;
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

const ThreadSpec *
BreakpointOptions::GetThreadSpec () const
{
    return m_thread_spec_ap.get();
}

ThreadSpec *
BreakpointOptions::GetThreadSpec ()
{
    if (m_thread_spec_ap.get() == NULL)
        m_thread_spec_ap.reset (new ThreadSpec());
        
    return m_thread_spec_ap.get();
}

void
BreakpointOptions::SetThreadID (lldb::tid_t thread_id)
{
    GetThreadSpec()->SetTID(thread_id);
}

void
BreakpointOptions::GetDescription (Stream *s, lldb::DescriptionLevel level) const
{

    // Figure out if there are any options not at their default value, and only print 
    // anything if there are:
    
    if (m_ignore_count != 0 || !m_enabled || (GetThreadSpec() != NULL && GetThreadSpec()->HasSpecification ()))
    {
        if (level == lldb::eDescriptionLevelVerbose)
        {
            s->EOL ();
            s->IndentMore();
            s->Indent();
            s->PutCString("Breakpoint Options:\n");
            s->IndentMore();
            s->Indent();
        }
        else
            s->PutCString(" Options: ");
                
        if (m_ignore_count > 0)
            s->Printf("ignore: %d ", m_ignore_count);
        s->Printf("%sabled ", m_enabled ? "en" : "dis");
        
        if (m_thread_spec_ap.get())
            m_thread_spec_ap->GetDescription (s, level);
        else if (level == eDescriptionLevelBrief)
            s->PutCString ("thread spec: no ");
        if (level == lldb::eDescriptionLevelFull)
        {
            s->IndentLess();
            s->IndentMore();
        }
    }
            
    if (m_callback_baton_sp.get())
    {
        if (level != eDescriptionLevelBrief)
            s->EOL();
        m_callback_baton_sp->GetDescription (s, level);
    }
    else if (level == eDescriptionLevelBrief)
        s->PutCString ("commands: no ");
    
}

void
BreakpointOptions::CommandBaton::GetDescription (Stream *s, lldb::DescriptionLevel level) const
{
    CommandData *data = (CommandData *)m_data;

    if (level == eDescriptionLevelBrief)
    {
        if (data && data->user_source.GetSize() > 0)
            s->PutCString("commands: yes ");
        else
            s->PutCString("commands: no ");
        return;
    }
    
    s->IndentMore ();
    s->Indent("Breakpoint commands:\n");
    
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
    s->IndentLess ();
}

