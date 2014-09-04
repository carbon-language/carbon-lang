//===-- MemoryHistoryASan.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MemoryHistoryASan.h"

#include "lldb/Target/MemoryHistory.h"

#include "lldb/lldb-private.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Core/Module.h"
#include "Plugins/Process/Utility/HistoryThread.h"
#include "lldb/Core/ValueObject.h"

using namespace lldb;
using namespace lldb_private;

MemoryHistorySP
MemoryHistoryASan::CreateInstance (const ProcessSP &process_sp)
{
    if (!process_sp.get())
        return NULL;
    
    Target & target = process_sp->GetTarget();
    
    bool found_asan_runtime = false;
    
    const ModuleList &target_modules = target.GetImages();
    Mutex::Locker modules_locker(target_modules.GetMutex());
    const size_t num_modules = target_modules.GetSize();
    for (size_t i = 0; i < num_modules; ++i)
    {
        Module *module_pointer = target_modules.GetModulePointerAtIndexUnlocked(i);
        
        SymbolContextList sc_list;
        const bool include_symbols = true;
        const bool append = true;
        const bool include_inlines = true;

        size_t num_matches = module_pointer->FindFunctions(ConstString("__asan_get_alloc_stack"), NULL, eFunctionNameTypeAuto, include_symbols, include_inlines, append, sc_list);
        
        if (num_matches)
        {
            found_asan_runtime = true;
            break;
        }
    }
    
    if (! found_asan_runtime)
        return MemoryHistorySP();

    return MemoryHistorySP(new MemoryHistoryASan(process_sp));
}

void
MemoryHistoryASan::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "ASan memory history provider.",
                                   CreateInstance);
}

void
MemoryHistoryASan::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


ConstString
MemoryHistoryASan::GetPluginNameStatic()
{
    static ConstString g_name("asan");
    return g_name;
}

MemoryHistoryASan::MemoryHistoryASan(const ProcessSP &process_sp)
{
    this->m_process_sp = process_sp;
}

const char *
memory_history_asan_command_format = R"(
    struct t {
        void *alloc_trace[256];
        size_t alloc_count;
        int alloc_tid;
        
        void *free_trace[256];
        size_t free_count;
        int free_tid;
    } t;

    t.alloc_count = ((size_t (*) (void *, void **, size_t, int *))__asan_get_alloc_stack)((void *)0x%)" PRIx64 R"(, t.alloc_trace, 256, &t.alloc_tid);
    t.free_count = ((size_t (*) (void *, void **, size_t, int *))__asan_get_free_stack)((void *)0x%)" PRIx64 R"(, t.free_trace, 256, &t.free_tid);

    t;
)";

#define GET_STACK_FUNCTION_TIMEOUT_USEC 2*1000*1000

HistoryThreads
MemoryHistoryASan::GetHistoryThreads(lldb::addr_t address)
{
    ProcessSP process_sp = m_process_sp;
    ThreadSP thread_sp = m_process_sp->GetThreadList().GetSelectedThread();
    StackFrameSP frame_sp = thread_sp->GetSelectedFrame();

    if (!frame_sp)
    {
        return HistoryThreads();
    }

    ExecutionContext exe_ctx (frame_sp);
    ValueObjectSP return_value_sp;
    StreamString expr;
    expr.Printf(memory_history_asan_command_format, address, address);
    
    EvaluateExpressionOptions options;
    options.SetUnwindOnError(true);
    options.SetTryAllThreads(true);
    options.SetStopOthers(true);
    options.SetIgnoreBreakpoints(true);
    options.SetTimeoutUsec(GET_STACK_FUNCTION_TIMEOUT_USEC);

    if (m_process_sp->GetTarget().EvaluateExpression(expr.GetData(), frame_sp.get(), return_value_sp, options) != eExpressionCompleted)
    {
        return HistoryThreads();
    }
    if (!return_value_sp)
    {
        return HistoryThreads();
    }
    
    HistoryThreads result;
    
    int alloc_count = return_value_sp->GetValueForExpressionPath(".alloc_count")->GetValueAsUnsigned(0);
    int free_count = return_value_sp->GetValueForExpressionPath(".free_count")->GetValueAsUnsigned(0);
    tid_t alloc_tid = return_value_sp->GetValueForExpressionPath(".alloc_tid")->GetValueAsUnsigned(0);
    tid_t free_tid = return_value_sp->GetValueForExpressionPath(".free_tid")->GetValueAsUnsigned(0);
    
    if (alloc_count > 0)
    {
        std::vector<lldb::addr_t> pcs;
        ValueObjectSP trace_sp = return_value_sp->GetValueForExpressionPath(".alloc_trace");
        for (int i = 0; i < alloc_count; i++) {
            addr_t pc = trace_sp->GetChildAtIndex(i, true)->GetValueAsUnsigned(0);
            pcs.push_back(pc);
        }
        
        HistoryThread *history_thread = new HistoryThread(*process_sp, alloc_tid, pcs, 0, false);
        ThreadSP new_thread_sp(history_thread);
        // let's use thread name for the type of history thread, since history threads don't have names anyway
        history_thread->SetThreadName("Memory allocated at");
        result.push_back(new_thread_sp);
    }
    
    if (free_count > 0)
    {
        std::vector<lldb::addr_t> pcs;
        ValueObjectSP trace_sp = return_value_sp->GetValueForExpressionPath(".free_trace");
        for (int i = 0; i < free_count; i++) {
            addr_t pc = trace_sp->GetChildAtIndex(i, true)->GetValueAsUnsigned(0);
            pcs.push_back(pc);
        }
        
        HistoryThread *history_thread = new HistoryThread(*process_sp, free_tid, pcs, 0, false);
        ThreadSP new_thread_sp(history_thread);
        // let's use thread name for the type of history thread, since history threads don't have names anyway
        history_thread->SetThreadName("Memory deallocated at");
        result.push_back(new_thread_sp);
    }
    
    return result;
}
