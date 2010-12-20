//===-- SBDebugger.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBDebugger.h"

#include "lldb/lldb-include.h"

#include "lldb/API/SBListener.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBInputReader.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/State.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/TargetList.h"

using namespace lldb;
using namespace lldb_private;

void
SBDebugger::Initialize ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger::Initialize ()");

    Debugger::Initialize();
}

void
SBDebugger::Terminate ()
{
    Debugger::Terminate();
}

void
SBDebugger::Clear ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger(%p)::Clear ()", m_opaque_sp.get());
        
    if (m_opaque_sp)
        m_opaque_sp->CleanUpInputReaders ();

    m_opaque_sp.reset();
}

SBDebugger
SBDebugger::Create()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBDebugger debugger;
    debugger.reset(Debugger::CreateInstance());

    if (log)
    {
        SBStream sstr;
        debugger.GetDescription (sstr);
        log->Printf ("SBDebugger::Create () => SBDebugger(%p): %s", debugger.m_opaque_sp.get(), sstr.GetData());
    }

    return debugger;
}

SBDebugger::SBDebugger () :
    m_opaque_sp ()
{
}

SBDebugger::SBDebugger(const SBDebugger &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

SBDebugger &
SBDebugger::operator = (const SBDebugger &rhs)
{
    if (this != &rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
    }
    return *this;
}

SBDebugger::~SBDebugger ()
{
}

bool
SBDebugger::IsValid() const
{
    return m_opaque_sp.get() != NULL;
}


void
SBDebugger::SetAsync (bool b)
{
    if (m_opaque_sp)
        m_opaque_sp->SetAsyncExecution(b);
}

void
SBDebugger::SkipLLDBInitFiles (bool b)
{
    if (m_opaque_sp)
        m_opaque_sp->GetCommandInterpreter().SkipLLDBInitFiles (b);
}

// Shouldn't really be settable after initialization as this could cause lots of problems; don't want users
// trying to switch modes in the middle of a debugging session.
void
SBDebugger::SetInputFileHandle (FILE *fh, bool transfer_ownership)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger(%p)::SetInputFileHandle (fh=%p, transfer_ownership=%i)", m_opaque_sp.get(),
                     fh, transfer_ownership);

    if (m_opaque_sp)
        m_opaque_sp->SetInputFileHandle (fh, transfer_ownership);
}

void
SBDebugger::SetOutputFileHandle (FILE *fh, bool transfer_ownership)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));


    if (log)
        log->Printf ("SBDebugger(%p)::SetOutputFileHandle (fh=%p, transfer_ownership=%i)", m_opaque_sp.get(),
                     fh, transfer_ownership);

    if (m_opaque_sp)
        m_opaque_sp->SetOutputFileHandle (fh, transfer_ownership);
}

void
SBDebugger::SetErrorFileHandle (FILE *fh, bool transfer_ownership)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));


    if (log)
        log->Printf ("SBDebugger(%p)::SetErrorFileHandle (fh=%p, transfer_ownership=%i)", m_opaque_sp.get(),
                     fh, transfer_ownership);

    if (m_opaque_sp)
        m_opaque_sp->SetErrorFileHandle (fh, transfer_ownership);
}

FILE *
SBDebugger::GetInputFileHandle ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetInputFileHandle();
    return NULL;
}

FILE *
SBDebugger::GetOutputFileHandle ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetOutputFileHandle();
    return NULL;
}

FILE *
SBDebugger::GetErrorFileHandle ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetErrorFileHandle();
    return NULL;
}

SBCommandInterpreter
SBDebugger::GetCommandInterpreter ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBCommandInterpreter sb_interpreter;
    if (m_opaque_sp)
        sb_interpreter.reset (&m_opaque_sp->GetCommandInterpreter());

    if (log)
        log->Printf ("SBDebugger(%p)::GetCommandInterpreter () => SBCommandInterpreter(%p)", 
                     m_opaque_sp.get(), sb_interpreter.get());

    return sb_interpreter;
}

void
SBDebugger::HandleCommand (const char *command)
{
    if (m_opaque_sp)
    {
        TargetSP target_sp (m_opaque_sp->GetSelectedTarget());
        Mutex::Locker api_locker;
        if (target_sp)
            api_locker.Reset(target_sp->GetAPIMutex().GetMutex());

        SBCommandInterpreter sb_interpreter(GetCommandInterpreter ());
        SBCommandReturnObject result;

        sb_interpreter.HandleCommand (command, result, false);

        if (GetErrorFileHandle() != NULL)
            result.PutError (GetErrorFileHandle());
        if (GetOutputFileHandle() != NULL)
            result.PutOutput (GetOutputFileHandle());

        if (m_opaque_sp->GetAsyncExecution() == false)
        {
            SBProcess process(GetCommandInterpreter().GetProcess ());
            if (process.IsValid())
            {
                EventSP event_sp;
                Listener &lldb_listener = m_opaque_sp->GetListener();
                while (lldb_listener.GetNextEventForBroadcaster (process.get(), event_sp))
                {
                    SBEvent event(event_sp);
                    HandleProcessEvent (process, event, GetOutputFileHandle(), GetErrorFileHandle());
                }
            }
        }
    }
}

SBListener
SBDebugger::GetListener ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBListener sb_listener;
    if (m_opaque_sp)
        sb_listener.reset(&m_opaque_sp->GetListener(), false);

    if (log)
        log->Printf ("SBDebugger(%p)::GetListener () => SBListener(%p)", m_opaque_sp.get(),
                     sb_listener.get());

    return sb_listener;
}

void
SBDebugger::HandleProcessEvent (const SBProcess &process, const SBEvent &event, FILE *out, FILE *err)
{
    if (!process.IsValid())
        return;

    const uint32_t event_type = event.GetType();
    char stdio_buffer[1024];
    size_t len;

    Mutex::Locker api_locker (process.GetTarget()->GetAPIMutex());
    
    if (event_type & (Process::eBroadcastBitSTDOUT | Process::eBroadcastBitStateChanged))
    {
        // Drain stdout when we stop just in case we have any bytes
        while ((len = process.GetSTDOUT (stdio_buffer, sizeof (stdio_buffer))) > 0)
            if (out != NULL)
                ::fwrite (stdio_buffer, 1, len, out);
    }
    
    if (event_type & (Process::eBroadcastBitSTDERR | Process::eBroadcastBitStateChanged))
    {
        // Drain stderr when we stop just in case we have any bytes
        while ((len = process.GetSTDERR (stdio_buffer, sizeof (stdio_buffer))) > 0)
            if (err != NULL)
                ::fwrite (stdio_buffer, 1, len, err);
    }
    
    if (event_type & Process::eBroadcastBitStateChanged)
    {
        StateType event_state = SBProcess::GetStateFromEvent (event);
        
        if (event_state == eStateInvalid)
            return;
        
        bool is_stopped = StateIsStoppedState (event_state);
        if (!is_stopped)
            process.ReportEventState (event, out);
    }
}

SBSourceManager &
SBDebugger::GetSourceManager ()
{
    static SourceManager g_lldb_source_manager;
    static SBSourceManager g_sb_source_manager (&g_lldb_source_manager);
    return g_sb_source_manager;
}


bool
SBDebugger::GetDefaultArchitecture (char *arch_name, size_t arch_name_len)
{
    if (arch_name && arch_name_len)
    {
        ArchSpec default_arch = lldb_private::Target::GetDefaultArchitecture ();

        if (default_arch.IsValid())
        {
            ::snprintf (arch_name, arch_name_len, "%s", default_arch.AsCString());
            return true;
        }
    }
    if (arch_name && arch_name_len)
        arch_name[0] = '\0';
    return false;
}


bool
SBDebugger::SetDefaultArchitecture (const char *arch_name)
{
    if (arch_name)
    {
        ArchSpec arch (arch_name);
        if (arch.IsValid())
        {
            lldb_private::Target::SetDefaultArchitecture (arch);
            return true;
        }
    }
    return false;
}

ScriptLanguage
SBDebugger::GetScriptingLanguage (const char *script_language_name)
{

    return Args::StringToScriptLanguage (script_language_name,
                                         eScriptLanguageDefault,
                                         NULL);
}

const char *
SBDebugger::GetVersionString ()
{
    return lldb_private::GetVersion();
}

const char *
SBDebugger::StateAsCString (lldb::StateType state)
{
    return lldb_private::StateAsCString (state);
}

bool
SBDebugger::StateIsRunningState (lldb::StateType state)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    const bool result = lldb_private::StateIsRunningState (state);
    if (log)
        log->Printf ("SBDebugger::StateIsRunningState (state=%s) => %i", 
                     lldb_private::StateAsCString (state), result);

    return result;
}

bool
SBDebugger::StateIsStoppedState (lldb::StateType state)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    const bool result = lldb_private::StateIsStoppedState (state);
    if (log)
        log->Printf ("SBDebugger::StateIsStoppedState (state=%s) => %i", 
                     lldb_private::StateAsCString (state), result);

    return result;
}


SBTarget
SBDebugger::CreateTargetWithFileAndTargetTriple (const char *filename,
                                                 const char *target_triple)
{
    SBTarget target;
    if (m_opaque_sp)
    {
        ArchSpec arch;
        FileSpec file_spec (filename, true);
        arch.SetArchFromTargetTriple(target_triple);
        TargetSP target_sp;
        Error error (m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file_spec, arch, NULL, true, target_sp));
        target.reset (target_sp);
    }
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBDebugger(%p)::CreateTargetWithFileAndTargetTriple (filename=\"%s\", triple=%s) => SBTarget(%p)", 
                     m_opaque_sp.get(), filename, target_triple, target.get());
    }

    return target;
}

SBTarget
SBDebugger::CreateTargetWithFileAndArch (const char *filename, const char *archname)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBTarget target;
    if (m_opaque_sp)
    {
        FileSpec file (filename, true);
        ArchSpec arch = lldb_private::Target::GetDefaultArchitecture ();
        TargetSP target_sp;
        Error error;

        if (archname != NULL)
        {
            ArchSpec arch2 (archname);
            error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file, arch2, NULL, true, target_sp);
        }
        else
        {
            if (!arch.IsValid())
                arch = LLDB_ARCH_DEFAULT;

            error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file, arch, NULL, true, target_sp);

            if (error.Fail())
            {
                if (arch == LLDB_ARCH_DEFAULT_32BIT)
                    arch = LLDB_ARCH_DEFAULT_64BIT;
                else
                    arch = LLDB_ARCH_DEFAULT_32BIT;

                error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file, arch, NULL, true, target_sp);
            }
        }

        if (error.Success())
        {
            m_opaque_sp->GetTargetList().SetSelectedTarget (target_sp.get());
            target.reset(target_sp);
        }
    }

    if (log)
    {
        log->Printf ("SBDebugger(%p)::CreateTargetWithFileAndArch (filename=\"%s\", arch=%s) => SBTarget(%p)", 
                     m_opaque_sp.get(), filename, archname, target.get());
    }

    return target;
}

SBTarget
SBDebugger::CreateTarget (const char *filename)
{
    SBTarget target;
    if (m_opaque_sp)
    {
        FileSpec file (filename, true);
        ArchSpec arch = lldb_private::Target::GetDefaultArchitecture ();
        TargetSP target_sp;
        Error error;

        if (!arch.IsValid())
            arch = LLDB_ARCH_DEFAULT;

        error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file, arch, NULL, true, target_sp);

        if (error.Fail())
        {
            if (arch == LLDB_ARCH_DEFAULT_32BIT)
                arch = LLDB_ARCH_DEFAULT_64BIT;
            else
                arch = LLDB_ARCH_DEFAULT_32BIT;

            error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file, arch, NULL, true, target_sp);
        }

        if (error.Success())
        {
            m_opaque_sp->GetTargetList().SetSelectedTarget (target_sp.get());
            target.reset (target_sp);
        }
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBDebugger(%p)::CreateTarget (filename=\"%s\") => SBTarget(%p)", 
                     m_opaque_sp.get(), filename, target.get());
    }
    return target;
}

SBTarget
SBDebugger::GetTargetAtIndex (uint32_t idx)
{
    SBTarget sb_target;
    if (m_opaque_sp)
    {
        // No need to lock, the target list is thread safe
        sb_target.reset(m_opaque_sp->GetTargetList().GetTargetAtIndex (idx));
    }
    return sb_target;
}

SBTarget
SBDebugger::FindTargetWithProcessID (pid_t pid)
{
    SBTarget sb_target;
    if (m_opaque_sp)
    {
        // No need to lock, the target list is thread safe
        sb_target.reset(m_opaque_sp->GetTargetList().FindTargetWithProcessID (pid));
    }
    return sb_target;
}

SBTarget
SBDebugger::FindTargetWithFileAndArch (const char *filename, const char *arch_name)
{
    SBTarget sb_target;
    if (m_opaque_sp && filename && filename[0])
    {
        // No need to lock, the target list is thread safe
        ArchSpec arch;
        if (arch_name)
            arch.SetArch(arch_name);
        TargetSP target_sp (m_opaque_sp->GetTargetList().FindTargetWithExecutableAndArchitecture (FileSpec(filename, false), arch_name ? &arch : NULL));
        sb_target.reset(target_sp);
    }
    return sb_target;
}

SBTarget
SBDebugger::FindTargetWithLLDBProcess (const lldb::ProcessSP &process_sp)
{
    SBTarget sb_target;
    if (m_opaque_sp)
    {
        // No need to lock, the target list is thread safe
        sb_target.reset(m_opaque_sp->GetTargetList().FindTargetWithProcess (process_sp.get()));
    }
    return sb_target;
}


uint32_t
SBDebugger::GetNumTargets ()
{
    if (m_opaque_sp)
    {
        // No need to lock, the target list is thread safe
        return m_opaque_sp->GetTargetList().GetNumTargets ();
    }
    return 0;
}

SBTarget
SBDebugger::GetSelectedTarget ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBTarget sb_target;
    if (m_opaque_sp)
    {
        // No need to lock, the target list is thread safe
        sb_target.reset(m_opaque_sp->GetTargetList().GetSelectedTarget ());
    }

    if (log)
    {
        SBStream sstr;
        sb_target.GetDescription (sstr, lldb::eDescriptionLevelBrief);
        log->Printf ("SBDebugger(%p)::GetSelectedTarget () => SBTarget(%p): %s", m_opaque_sp.get(),
                     sb_target.get(), sstr.GetData());
    }

    return sb_target;
}

void
SBDebugger::DispatchInput (void *baton, const void *data, size_t data_len)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger(%p)::DispatchInput (baton=%p, data=\"%.*s\", size_t=%zu)", m_opaque_sp.get(),
                     baton, (int) data_len, (const char *) data, data_len);

    if (m_opaque_sp)
        m_opaque_sp->DispatchInput ((const char *) data, data_len);
}

void
SBDebugger::DispatchInputInterrupt ()
{
    if (m_opaque_sp)
        m_opaque_sp->DispatchInputInterrupt ();
}

void
SBDebugger::DispatchInputEndOfFile ()
{
    if (m_opaque_sp)
        m_opaque_sp->DispatchInputEndOfFile ();
}

void
SBDebugger::PushInputReader (SBInputReader &reader)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger(%p)::PushInputReader (SBInputReader(%p))", m_opaque_sp.get(), &reader);

    if (m_opaque_sp && reader.IsValid())
    {
        TargetSP target_sp (m_opaque_sp->GetSelectedTarget());
        Mutex::Locker api_locker;
        if (target_sp)
            api_locker.Reset(target_sp->GetAPIMutex().GetMutex());
        InputReaderSP reader_sp(*reader);
        m_opaque_sp->PushInputReader (reader_sp);
    }
}

void
SBDebugger::reset (const lldb::DebuggerSP &debugger_sp)
{
    m_opaque_sp = debugger_sp;
}

Debugger *
SBDebugger::get () const
{
    return m_opaque_sp.get();
}

Debugger &
SBDebugger::ref () const
{
    assert (m_opaque_sp.get());
    return *m_opaque_sp;
}


SBDebugger
SBDebugger::FindDebuggerWithID (int id)
{
    // No need to lock, the debugger list is thread safe
    SBDebugger sb_debugger;
    lldb::DebuggerSP debugger_sp = Debugger::FindDebuggerWithID (id);
    if (debugger_sp)
        sb_debugger.reset (debugger_sp);
    return sb_debugger;
}

const char *
SBDebugger::GetInstanceName()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetInstanceName().AsCString();
    else
        return NULL;
}

SBError
SBDebugger::SetInternalVariable (const char *var_name, const char *value, const char *debugger_instance_name)
{
    lldb::UserSettingsControllerSP root_settings_controller = lldb_private::Debugger::GetSettingsController();

    Error err = root_settings_controller->SetVariable (var_name, value, lldb::eVarSetOperationAssign, true,
                                                       debugger_instance_name);
    SBError sb_error;
    sb_error.SetError (err);

    return sb_error;
}

lldb::SBStringList
SBDebugger::GetInternalVariableValue (const char *var_name, const char *debugger_instance_name)
{
    SBStringList ret_value;
    lldb::SettableVariableType var_type;
    lldb_private::Error err;

    lldb::UserSettingsControllerSP root_settings_controller = lldb_private::Debugger::GetSettingsController();

    StringList value = root_settings_controller->GetVariable (var_name, var_type, debugger_instance_name, err);
    
    if (err.Success())
    {
        for (unsigned i = 0; i != value.GetSize(); ++i)
            ret_value.AppendString (value.GetStringAtIndex(i));
    }
    else
    {
        ret_value.AppendString (err.AsCString());
    }


    return ret_value;
}

uint32_t
SBDebugger::GetTerminalWidth () const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetTerminalWidth ();
    return 0;
}

void
SBDebugger::SetTerminalWidth (uint32_t term_width)
{
    if (m_opaque_sp)
        m_opaque_sp->SetTerminalWidth (term_width);
}

const char *
SBDebugger::GetPrompt() const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    if (log)
        log->Printf ("SBDebugger(%p)::GetPrompt () => \"%s\"", m_opaque_sp.get(), 
                     (m_opaque_sp ? m_opaque_sp->GetPrompt() : ""));

    if (m_opaque_sp)
        return m_opaque_sp->GetPrompt ();
    return 0;
}

void
SBDebugger::SetPrompt (const char *prompt)
{
    if (m_opaque_sp)
        m_opaque_sp->SetPrompt (prompt);
}

    
lldb::ScriptLanguage 
SBDebugger::GetScriptLanguage() const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetScriptLanguage ();
    return eScriptLanguageNone;
}

void
SBDebugger::SetScriptLanguage (lldb::ScriptLanguage script_lang)
{
    if (m_opaque_sp)
    {
        m_opaque_sp->SetScriptLanguage (script_lang);
    }
}

bool
SBDebugger::SetUseExternalEditor (bool value)
{
    if (m_opaque_sp)
        return m_opaque_sp->SetUseExternalEditor (value);
    return false;
}

bool
SBDebugger::GetUseExternalEditor ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetUseExternalEditor ();
    return false;
}

bool
SBDebugger::GetDescription (SBStream &description)
{
    if (m_opaque_sp)
    {
        const char *name = m_opaque_sp->GetInstanceName().AsCString();
        lldb::user_id_t id = m_opaque_sp->GetID();
        description.Printf ("Debugger (instance: \"%s\", id: %d)", name, id);
    }
    else
        description.Printf ("No value");
    
    return true;
}

lldb::user_id_t
SBDebugger::GetID()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetID();
    return LLDB_INVALID_UID;
}
