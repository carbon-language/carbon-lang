//===-- LanguageRuntime.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

LanguageRuntime*
LanguageRuntime::FindPlugin (Process *process, lldb::LanguageType language)
{
    std::auto_ptr<LanguageRuntime> language_runtime_ap;
    LanguageRuntimeCreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetLanguageRuntimeCreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        language_runtime_ap.reset (create_callback(process, language));

        if (language_runtime_ap.get())
            return language_runtime_ap.release();
    }

    return NULL;
}

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
LanguageRuntime::LanguageRuntime(Process *process) :
    m_process (process)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
LanguageRuntime::~LanguageRuntime()
{
}

BreakpointSP
LanguageRuntime::CreateExceptionBreakpoint(
    Target &target, 
    lldb::LanguageType language, 
    bool catch_bp, 
    bool throw_bp, 
    bool is_internal)
{
    BreakpointSP exc_breakpt_sp;
    BreakpointResolverSP resolver_sp(new ExceptionBreakpointResolver(NULL, language, catch_bp, throw_bp));
    SearchFilterSP filter_sp(target.GetSearchFilterForModule(NULL));
    
    exc_breakpt_sp = target.CreateBreakpoint (filter_sp, resolver_sp, is_internal);
    
    return exc_breakpt_sp;
}

LanguageRuntime::ExceptionBreakpointResolver::ExceptionBreakpointResolver (Breakpoint *bkpt,
                        LanguageType language,
                        bool catch_bp,
                        bool throw_bp) :
    BreakpointResolver (bkpt, BreakpointResolver::ExceptionResolver),
    m_language (language),
    m_catch_bp (catch_bp),
    m_throw_bp (throw_bp)

{
}
                        
void
LanguageRuntime::ExceptionBreakpointResolver::GetDescription (Stream *s)
{
    s->Printf ("Exception breakpoint (catch: %s throw: %s)", 
           m_catch_bp ? "on" : "off",
           m_throw_bp ? "on" : "off");
       
    SetActualResolver();
    if (m_actual_resolver_sp)
    {
        s->Printf (" using: ");
        m_actual_resolver_sp->GetDescription (s);
    }
    else
        s->Printf (" the correct runtime exception handler will be determined when you run");
}

bool
LanguageRuntime::ExceptionBreakpointResolver::SetActualResolver()
{
    ProcessSP process_sp = m_process_wp.lock();
    
    // See if our process weak pointer is still good:
    if (!process_sp)
    {
        // If not, our resolver is no good, so chuck that.  Then see if we can get the 
        // target's new process.
        m_actual_resolver_sp.reset();
        if (m_breakpoint)
        {
            Target &target = m_breakpoint->GetTarget();
            process_sp = target.GetProcessSP();
            if (process_sp)
            {
                m_process_wp = process_sp;
                process_sp = m_process_wp.lock();
            }
        }
    }
    
    if (process_sp)
    {
        if (m_actual_resolver_sp)
            return true;
        else
        {
            // If we have a process but not a resolver, set one now.
            LanguageRuntime *runtime = process_sp->GetLanguageRuntime(m_language);
            if (runtime)
            {
                m_actual_resolver_sp = runtime->CreateExceptionResolver (m_breakpoint, m_catch_bp, m_throw_bp);
                return (bool) m_actual_resolver_sp;
            }
            else
                return false;
        }
    }
    else
        return false;
}

Searcher::CallbackReturn
LanguageRuntime::ExceptionBreakpointResolver::SearchCallback (SearchFilter &filter,
                SymbolContext &context,
                Address *addr,
                bool containing)
{
    
    if (!SetActualResolver())
    {
        return eCallbackReturnStop;
    }
    else
        return m_actual_resolver_sp->SearchCallback (filter, context, addr, containing);
}

Searcher::Depth
LanguageRuntime::ExceptionBreakpointResolver::GetDepth ()
{
    if (!SetActualResolver())
        return eDepthTarget;
    else
        return m_actual_resolver_sp->GetDepth();
}

static const char *language_names[] =
{
    "unknown",
    "c89",
    "c",
    "ada83",
    "c++",
    "cobol74",
    "cobol85",
    "fortran77",
    "fortran90",
    "pascal83",
    "modula2",
    "java",
    "c99",
    "ada95",
    "fortran95",
    "pli",
    "objective-c",
    "objective-c++",
    "upc",
    "d",
    "python"
};
static uint32_t num_languages = sizeof(language_names) / sizeof (char *);

LanguageType
LanguageRuntime::GetLanguageTypeFromString (const char *string)
{
    for (uint32_t i = 0; i < num_languages; i++)
    {
        if (strcmp (language_names[i], string) == 0)
            return (LanguageType) i;
    }
    return eLanguageTypeUnknown;
}

const char *
LanguageRuntime::GetNameForLanguageType (LanguageType language)
{
    if (language < num_languages)
        return language_names[language];
    else
        return language_names[eLanguageTypeUnknown];
}
        
