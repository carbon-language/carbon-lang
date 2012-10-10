//===-- OperatingSystemPython.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_DISABLE_PYTHON

#include "OperatingSystemPython.h"
// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/PythonDataObjects.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/PythonDataObjects.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Target/Thread.h"
#include "Plugins/Process/Utility/DynamicRegisterInfo.h"
#include "Plugins/Process/Utility/RegisterContextMemory.h"
#include "Plugins/Process/Utility/ThreadMemory.h"

using namespace lldb;
using namespace lldb_private;

void
OperatingSystemPython::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
OperatingSystemPython::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

OperatingSystem *
OperatingSystemPython::CreateInstance (Process *process, bool force)
{
    // Python OperatingSystem plug-ins must be requested by name, so force must be true
    if (force)
        return new OperatingSystemPython (process);
    return NULL;
}


const char *
OperatingSystemPython::GetPluginNameStatic()
{
    return "python";
}

const char *
OperatingSystemPython::GetPluginDescriptionStatic()
{
    return "Operating system plug-in that gathers OS information from a python class that implements the necessary OperatingSystem functionality.";
}


OperatingSystemPython::OperatingSystemPython (lldb_private::Process *process) :
    OperatingSystem (process),
    m_thread_list_valobj_sp (),
    m_register_info_ap (),
    m_interpreter(NULL),
    m_python_object(NULL)
{
    if (!process)
        return;
    lldb::TargetSP target_sp = process->CalculateTarget();
    if (!target_sp)
        return;
    m_interpreter = target_sp->GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
    if (m_interpreter)
    {
        // TODO: hardcoded is not good
        auto object_sp = m_interpreter->CreateOSPlugin("operating_system.PlugIn",process->CalculateProcess());
        if (object_sp)
        {
            m_python_object = object_sp->GetObject();
            
            //GetDynamicRegisterInfo (); // COMMENT THIS LINE OUT PRIOR TO CHECKIN!!!
        }
    }
}

OperatingSystemPython::~OperatingSystemPython ()
{
}

DynamicRegisterInfo *
OperatingSystemPython::GetDynamicRegisterInfo ()
{
    if (m_register_info_ap.get() == NULL)
    {
        if (!m_interpreter || !m_python_object)
            return NULL;
        auto object_sp = m_interpreter->OSPlugin_QueryForRegisterInfo(m_interpreter->MakeScriptObject(m_python_object));
        if (!object_sp)
            return NULL;
        PythonDataObject dictionary_data_obj((PyObject*)object_sp->GetObject());
        PythonDataDictionary dictionary = dictionary_data_obj.GetDictionaryObject();
        if (!dictionary)
            return NULL;
        
        m_register_info_ap.reset (new DynamicRegisterInfo (dictionary));
        assert (m_register_info_ap->GetNumRegisters() > 0);
        assert (m_register_info_ap->GetNumRegisterSets() > 0);
    }
    return m_register_info_ap.get();
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
OperatingSystemPython::GetPluginName()
{
    return "OperatingSystemPython";
}

const char *
OperatingSystemPython::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
OperatingSystemPython::GetPluginVersion()
{
    return 1;
}

bool
OperatingSystemPython::UpdateThreadList (ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    
    if (!m_interpreter || !m_python_object)
        return NULL;
    auto object_sp = m_interpreter->OSPlugin_QueryForThreadsInfo(m_interpreter->MakeScriptObject(m_python_object));
    if (!object_sp)
        return NULL;
    PythonDataObject pyobj((PyObject*)object_sp->GetObject());
    PythonDataArray threads_array (pyobj.GetArrayObject());
    if (threads_array)
    {
//        const uint32_t num_old_threads = old_thread_list.GetSize(false);
//        for (uint32_t i=0; i<num_old_threads; ++i)
//        {
//            ThreadSP old_thread_sp(old_thread_list.GetThreadAtIndex(i, false));
//            if (old_thread_sp->GetID() < 0x10000)
//                new_thread_list.AddThread (old_thread_sp);
//        }

        PythonDataString tid_pystr("tid");
        PythonDataString name_pystr("name");
        PythonDataString queue_pystr("queue");
        PythonDataString state_pystr("state");
        PythonDataString stop_reason_pystr("stop_reason");
        
        const uint32_t num_threads = threads_array.GetSize();
        for (uint32_t i=0; i<num_threads; ++i)
        {
            PythonDataDictionary thread_dict(threads_array.GetItemAtIndex(i).GetDictionaryObject());
            if (thread_dict)
            {
                const tid_t tid = thread_dict.GetItemForKeyAsInteger(tid_pystr, LLDB_INVALID_THREAD_ID);
                const char *name = thread_dict.GetItemForKeyAsString (name_pystr);
                const char *queue = thread_dict.GetItemForKeyAsString (queue_pystr);
                //const char *state = thread_dict.GetItemForKeyAsString (state_pystr);
                //const char *stop_reason = thread_dict.GetItemForKeyAsString (stop_reason_pystr);
                
                ThreadSP thread_sp (old_thread_list.FindThreadByID (tid, false));
                if (!thread_sp)
                    thread_sp.reset (new ThreadMemory (*m_process,
                                                       tid,
                                                       name,
                                                       queue));
                new_thread_list.AddThread(thread_sp);

            }
        }
    }
    else
    {
        new_thread_list = old_thread_list;
    }
    return new_thread_list.GetSize(false) > 0;
}

void
OperatingSystemPython::ThreadWasSelected (Thread *thread)
{
}

RegisterContextSP
OperatingSystemPython::CreateRegisterContextForThread (Thread *thread)
{
    RegisterContextSP reg_ctx_sp;
    if (!m_interpreter || !m_python_object || !thread)
        return RegisterContextSP();
    auto object_sp = m_interpreter->OSPlugin_QueryForRegisterContextData (m_interpreter->MakeScriptObject(m_python_object),
                                                                          thread->GetID());

           if (!object_sp)
        return RegisterContextSP();
    
    PythonDataString reg_context_data((PyObject*)object_sp->GetObject());
    if (reg_context_data)
    {
        DataBufferSP data_sp (new DataBufferHeap (reg_context_data.GetString(),
                                                  reg_context_data.GetSize()));
        if (data_sp->GetByteSize())
        {
            RegisterContextMemory *reg_ctx_memory = new RegisterContextMemory (*thread, 0, *GetDynamicRegisterInfo (), LLDB_INVALID_ADDRESS);
            if (reg_ctx_memory)
            {
                reg_ctx_sp.reset(reg_ctx_memory);
                reg_ctx_memory->SetAllRegisterData (data_sp);
            }
        }
    }
    return reg_ctx_sp;
}

StopInfoSP
OperatingSystemPython::CreateThreadStopReason (lldb_private::Thread *thread)
{
    // We should have gotten the thread stop info from the dictionary of data for
    // the thread in the initial call to get_thread_info(), this should have been
    // cached so we can return it here
    StopInfoSP stop_info_sp; //(StopInfo::CreateStopReasonWithSignal (*thread, SIGSTOP));
    return stop_info_sp;
}


#endif // #ifndef LLDB_DISABLE_PYTHON
