//===-- lldb-private-interfaces.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_lldb_interfaces_h_
#define liblldb_lldb_interfaces_h_

#if defined(__cplusplus)

#include "lldb/lldb-private.h"

namespace lldb_private
{
    typedef ABI* (*ABICreateInstance) (const ConstString &triple);
    typedef Disassembler* (*DisassemblerCreateInstance) (const ArchSpec &arch);
    typedef DynamicLoader* (*DynamicLoaderCreateInstance) (Process* process);
    typedef ObjectContainer* (*ObjectContainerCreateInstance) (Module* module, lldb::DataBufferSP& dataSP, const FileSpec *file, lldb::addr_t offset, lldb::addr_t length);
    typedef ObjectFile* (*ObjectFileCreateInstance) (Module* module, lldb::DataBufferSP& dataSP, const FileSpec* file, lldb::addr_t offset, lldb::addr_t length);
    typedef LogChannel* (*LogChannelCreateInstance) ();
    typedef LanguageRuntime *(*LanguageRuntimeCreateInstance) (Process *process, lldb::LanguageType language);
    typedef Process* (*ProcessCreateInstance) (Target &target, Listener &listener);
    typedef SymbolFile* (*SymbolFileCreateInstance) (ObjectFile* obj_file);
    typedef SymbolVendor* (*SymbolVendorCreateInstance) (Module *module);   // Module can be NULL for default system symbol vendor
    typedef bool (*BreakpointHitCallback) (void *baton, StoppointCallbackContext *context, lldb::user_id_t break_id, lldb::user_id_t break_loc_id);
    typedef bool (*WatchpointHitCallback) (void *baton, StoppointCallbackContext *context, lldb::user_id_t watch_id, uint32_t type);
    typedef ThreadPlan * (*ThreadPlanShouldStopHereCallback) (ThreadPlan *current_plan, Flags &flags, void *baton);
    typedef UnwindAssemblyProfiler* (*UnwindAssemblyProfilerCreateInstance) (const ArchSpec &arch);
    typedef ArchDefaultUnwindPlan* (*ArchDefaultUnwindPlanCreateInstance) (const ArchSpec &arch);
    typedef ArchVolatileRegs* (*ArchVolatileRegsCreateInstance) (const ArchSpec &arch);
} // namespace lldb_private

#endif  // #if defined(__cplusplus)

#endif  // liblldb_lldb_interfaces_h_
