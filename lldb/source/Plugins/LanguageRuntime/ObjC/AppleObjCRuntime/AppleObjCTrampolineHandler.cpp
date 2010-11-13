//===-- AppleObjCTrampolineHandler.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleObjCTrampolineHandler.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "AppleThreadPlanStepThroughObjCTrampoline.h"

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"

using namespace lldb;
using namespace lldb_private;

        
AppleObjCTrampolineHandler::AppleObjCVTables::VTableRegion::VTableRegion(AppleObjCVTables *owner, lldb::addr_t header_addr) :
    m_valid (true),
    m_owner(owner),
    m_header_addr (header_addr),
    m_code_start_addr(0),
    m_code_end_addr (0),
    m_next_region (0)
{
    SetUpRegion ();
}

void
AppleObjCTrampolineHandler::AppleObjCVTables::VTableRegion::SetUpRegion()
{
    // The header looks like:
    //
    //   uint16_t headerSize
    //   uint16_t descSize
    //   uint32_t descCount
    //   void * next
    //
    // First read in the header:
    
    char memory_buffer[16];
    Process *process = m_owner->GetProcess();
    DataExtractor data(memory_buffer, sizeof(memory_buffer), 
                       process->GetByteOrder(), 
                       process->GetAddressByteSize());
    size_t actual_size = 8 + process->GetAddressByteSize();
    Error error;
    size_t bytes_read = process->ReadMemory (m_header_addr, memory_buffer, actual_size, error);
    if (bytes_read != actual_size)
    {
        m_valid = false;
        return;
    }
    
    uint32_t offset_ptr = 0;
    const uint16_t header_size = data.GetU16(&offset_ptr);
    const uint16_t descriptor_size = data.GetU16(&offset_ptr);
    const size_t num_descriptors = data.GetU32(&offset_ptr);
    
    m_next_region = data.GetPointer(&offset_ptr);
    
    // If the header size is 0, that means we've come in too early before this data is set up.
    // Set ourselves as not valid, and continue.
    if (header_size == 0 || num_descriptors == 0)
    {
        m_valid = false;
        return;
    }
    
    // Now read in all the descriptors:
    // The descriptor looks like:
    //
    // uint32_t offset
    // uint32_t flags
    //
    // Where offset is either 0 - in which case it is unused, or
    // it is the offset of the vtable code from the beginning of the descriptor record.
    // Below, we'll convert that into an absolute code address, since I don't want to have
    // to compute it over and over.
    
    // Ingest the whole descriptor array:
    const lldb::addr_t desc_ptr = m_header_addr + header_size;
    const size_t desc_array_size = num_descriptors * descriptor_size;
    DataBufferSP data_sp(new DataBufferHeap (desc_array_size, '\0'));
    uint8_t* dst = (uint8_t*)data_sp->GetBytes();

    DataExtractor desc_extractor (dst, desc_array_size,
                                  process->GetByteOrder(), 
                                  process->GetAddressByteSize());
    bytes_read = process->ReadMemory(desc_ptr, dst, desc_array_size, error);
    if (bytes_read != desc_array_size)
    {
        m_valid = false;
        return;
    }
    
    // The actual code for the vtables will be laid out consecutively, so I also
    // compute the start and end of the whole code block.

    offset_ptr = 0;
    m_code_start_addr = 0;
    m_code_end_addr = 0;

    for (int i = 0; i < num_descriptors; i++)
    {
        lldb::addr_t start_offset = offset_ptr;
        uint32_t offset = desc_extractor.GetU32 (&offset_ptr);
        uint32_t flags  = desc_extractor.GetU32 (&offset_ptr);
        lldb:addr_t code_addr = desc_ptr + start_offset + offset;
        m_descriptors.push_back (VTableDescriptor(flags, code_addr));
        
        if (m_code_start_addr == 0 || code_addr < m_code_start_addr)
            m_code_start_addr = code_addr;
        if (code_addr > m_code_end_addr)
            m_code_end_addr = code_addr;
            
        offset_ptr = start_offset + descriptor_size;
    }
    // Finally, a little bird told me that all the vtable code blocks are the same size.  
    // Let's compute the blocks and if they are all the same add the size to the code end address:
    lldb::addr_t code_size = 0;
    bool all_the_same = true;
    for (int i = 0; i < num_descriptors - 1; i++)
    {
        lldb::addr_t this_size = m_descriptors[i + 1].code_start - m_descriptors[i].code_start;
        if (code_size == 0)
            code_size = this_size;
        else
        {
            if (this_size != code_size)
                all_the_same = false;
            if (this_size > code_size)
                code_size = this_size;
        }
    }
    if (all_the_same)
        m_code_end_addr += code_size;
}

bool 
AppleObjCTrampolineHandler::AppleObjCVTables::VTableRegion::AddressInRegion (lldb::addr_t addr, uint32_t &flags)
{
    if (!IsValid())
        return false;
        
    if (addr < m_code_start_addr || addr > m_code_end_addr)
        return false;
        
    std::vector<VTableDescriptor>::iterator pos, end = m_descriptors.end();
    for (pos = m_descriptors.begin(); pos != end; pos++)
    {
        if (addr <= (*pos).code_start)
        {
            flags = (*pos).flags;
            return true;
        }
    }
    return false;
}

void
AppleObjCTrampolineHandler::AppleObjCVTables::VTableRegion::Dump (Stream &s)
{
    s.Printf ("Header addr: 0x%llx Code start: 0x%llx Code End: 0x%llx Next: 0x%llx\n", 
              m_header_addr, m_code_start_addr, m_code_end_addr, m_next_region);
    size_t num_elements = m_descriptors.size();
    for (size_t i = 0; i < num_elements; i++)
    {
        s.Indent();
        s.Printf ("Code start: 0x%llx Flags: %d\n", m_descriptors[i].code_start, m_descriptors[i].flags);
    }
}
        
AppleObjCTrampolineHandler::AppleObjCVTables::AppleObjCVTables (ProcessSP &process_sp, ModuleSP &objc_module_sp) :
        m_process_sp(process_sp),
        m_trampoline_header(LLDB_INVALID_ADDRESS),
        m_trampolines_changed_bp_id(LLDB_INVALID_BREAK_ID),
        m_objc_module_sp(objc_module_sp)
{
    
}

AppleObjCTrampolineHandler::AppleObjCVTables::~AppleObjCVTables()
{
    if (m_trampolines_changed_bp_id != LLDB_INVALID_BREAK_ID)
        m_process_sp->GetTarget().RemoveBreakpointByID (m_trampolines_changed_bp_id);
}
    
bool
AppleObjCTrampolineHandler::AppleObjCVTables::InitializeVTableSymbols ()
{
    if (m_trampoline_header != LLDB_INVALID_ADDRESS)
        return true;
    Target &target = m_process_sp->GetTarget();
    
    ModuleList &modules = target.GetImages();
    size_t num_modules = modules.GetSize();
    if (!m_objc_module_sp)
    {
        for (size_t i = 0; i < num_modules; i++)
        {
            if (m_process_sp->GetObjCLanguageRuntime()->IsModuleObjCLibrary (modules.GetModuleAtIndex(i)))
            {
                m_objc_module_sp = modules.GetModuleAtIndex(i);
                break;
            }
        }
    }
    
    if (m_objc_module_sp)
    {
        ConstString trampoline_name ("gdb_objc_trampolines");
        const Symbol *trampoline_symbol = m_objc_module_sp->FindFirstSymbolWithNameAndType(trampoline_name, 
                                                                                   eSymbolTypeData);
        if (trampoline_symbol != NULL)
        {
            const Address &temp_address = trampoline_symbol->GetValue();
            if (!temp_address.IsValid())
                return false;
                
            m_trampoline_header = temp_address.GetLoadAddress(&target);
            if (m_trampoline_header == LLDB_INVALID_ADDRESS)
                return false;
            
            // Next look up the "changed" symbol and set a breakpoint on that...
            ConstString changed_name ("gdb_objc_trampolines_changed");
            const Symbol *changed_symbol = m_objc_module_sp->FindFirstSymbolWithNameAndType(changed_name, 
                                                                                   eSymbolTypeCode);
            if (changed_symbol != NULL)
            {
                const Address &temp_address = changed_symbol->GetValue();
                if (!temp_address.IsValid())
                    return false;
                    
                lldb::addr_t changed_addr = temp_address.GetLoadAddress(&target);
                if (changed_addr != LLDB_INVALID_ADDRESS)
                {
                    BreakpointSP trampolines_changed_bp_sp = target.CreateBreakpoint (changed_addr,
                                                                                          true);
                    if (trampolines_changed_bp_sp != NULL)
                    {
                        m_trampolines_changed_bp_id = trampolines_changed_bp_sp->GetID();
                        trampolines_changed_bp_sp->SetCallback (RefreshTrampolines, this, true);
                        return true;
                    }
                }
            }
        }
    }
    
    return false;
}
    
bool 
AppleObjCTrampolineHandler::AppleObjCVTables::RefreshTrampolines (void *baton, 
                                    StoppointCallbackContext *context, 
                                    lldb::user_id_t break_id, 
                                    lldb::user_id_t break_loc_id)
{
    AppleObjCVTables *vtable_handler = (AppleObjCVTables *) baton;
    if (vtable_handler->InitializeVTableSymbols())
    {
        // The Update function is called with the address of an added region.  So we grab that address, and
        // feed it into ReadRegions.  Of course, our friend the ABI will get the values for us.
        Process *process = context->exe_ctx.process;
        const ABI *abi = process->GetABI();
        
        ClangASTContext *clang_ast_context = process->GetTarget().GetScratchClangASTContext();
        ValueList argument_values;
        Value input_value;
        void *clang_void_ptr_type = clang_ast_context->GetVoidPtrType(false);
        input_value.SetValueType (Value::eValueTypeScalar);
        input_value.SetContext (Value::eContextTypeClangType, clang_void_ptr_type);
        argument_values.PushValue(input_value);
        
        bool success = abi->GetArgumentValues (*(context->exe_ctx.thread), argument_values);
        if (!success)
            return false;
            
        // Now get a pointer value from the zeroth argument.
        Error error;
        DataExtractor data;
        error = argument_values.GetValueAtIndex(0)->GetValueAsData(&(context->exe_ctx), clang_ast_context->getASTContext(), data, 0);
        uint32_t offset_ptr = 0;
        lldb::addr_t region_addr = data.GetPointer(&offset_ptr);
        
        if (region_addr != 0)
            vtable_handler->ReadRegions(region_addr);
    }
    return false;
}

bool
AppleObjCTrampolineHandler::AppleObjCVTables::ReadRegions ()
{
    // The no argument version reads the start region from the value of the gdb_regions_header, and 
    // gets started from there.
    
    m_regions.clear();
    if (!InitializeVTableSymbols())
        return false;
    char memory_buffer[8];
    DataExtractor data(memory_buffer, sizeof(memory_buffer), 
                       m_process_sp->GetByteOrder(), 
                       m_process_sp->GetAddressByteSize());
    Error error;
    size_t bytes_read = m_process_sp->ReadMemory (m_trampoline_header, memory_buffer, m_process_sp->GetAddressByteSize(), error);
    if (bytes_read != m_process_sp->GetAddressByteSize())
        return false;
        
    uint32_t offset_ptr = 0;
    lldb::addr_t region_addr = data.GetPointer(&offset_ptr);
    return ReadRegions (region_addr);
}

bool
AppleObjCTrampolineHandler::AppleObjCVTables::ReadRegions (lldb::addr_t region_addr)
{
    if (!m_process_sp)
        return false;
        
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    
    // We aren't starting at the trampoline symbol.
    InitializeVTableSymbols ();
    lldb::addr_t next_region = region_addr;
    
    // Read in the sizes of the headers.
    while (next_region != 0) 
    {
        m_regions.push_back (VTableRegion(this, next_region));
        if (!m_regions.back().IsValid())
        {
            m_regions.clear();
            return false;
        }
        if (log)
        {
            StreamString s;
            m_regions.back().Dump(s);
            log->Printf("Read vtable region: \n%s", s.GetData());
        }
        
        next_region = m_regions.back().GetNextRegionAddr();
    }
    
    return true;
}
    
bool
AppleObjCTrampolineHandler::AppleObjCVTables::IsAddressInVTables (lldb::addr_t addr, uint32_t &flags)
{
    region_collection::iterator pos, end = m_regions.end();
    for (pos = m_regions.begin(); pos != end; pos++)
    {
        if ((*pos).AddressInRegion (addr, flags))
            return true;
    }
    return false;
}

const AppleObjCTrampolineHandler::DispatchFunction
AppleObjCTrampolineHandler::g_dispatch_functions[] =
{
    // NAME                              STRET  SUPER  FIXUP TYPE
    {"objc_msgSend",                     false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSend_fixup",               false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_fixedup",             false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {"objc_msgSend_stret",               true,  false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSend_stret_fixup",         true,  false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_stret_fixedup",       true,  false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {"objc_msgSend_fpret",               false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSend_fpret_fixup",         false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_fpret_fixedup",       false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {"objc_msgSend_fp2ret",              false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSend_fp2ret_fixup",        false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_fp2ret_fixedup",      false, false, AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {"objc_msgSendSuper",                false, true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper_stret",          true,  true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper2",               false, true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper2_fixup",         false, true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSendSuper2_fixedup",       false, true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {"objc_msgSendSuper2_stret",         true,  true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper2_stret_fixup",   true,  true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpToFix   },
    {"objc_msgSendSuper2_stret_fixedup", true,  true,  AppleObjCTrampolineHandler::DispatchFunction::eFixUpFixed   },
    {NULL}
};

AppleObjCTrampolineHandler::AppleObjCTrampolineHandler (ProcessSP process_sp, ModuleSP objc_module_sp) :
    m_process_sp (process_sp),
    m_objc_module_sp (objc_module_sp),
    m_impl_fn_addr (LLDB_INVALID_ADDRESS),
    m_impl_stret_fn_addr (LLDB_INVALID_ADDRESS)
{
    // Look up the known resolution functions:
    
    ConstString get_impl_name("class_getMethodImplementation");
    ConstString get_impl_stret_name("class_getMethodImplementation_stret");
    
    Target *target = m_process_sp ? &m_process_sp->GetTarget() : NULL;
    const Symbol *class_getMethodImplementation = m_objc_module_sp->FindFirstSymbolWithNameAndType (get_impl_name, eSymbolTypeCode);
    const Symbol *class_getMethodImplementation_stret = m_objc_module_sp->FindFirstSymbolWithNameAndType (get_impl_stret_name, eSymbolTypeCode);
    
    if (class_getMethodImplementation)
        m_impl_fn_addr = class_getMethodImplementation->GetValue().GetLoadAddress(target);
    if  (class_getMethodImplementation_stret)
        m_impl_stret_fn_addr = class_getMethodImplementation_stret->GetValue().GetLoadAddress(target);
    
    // FIXME: Do some kind of logging here.
    if (m_impl_fn_addr == LLDB_INVALID_ADDRESS || m_impl_stret_fn_addr == LLDB_INVALID_ADDRESS)
        return;
        
    // Look up the addresses for the objc dispatch functions and cache them.  For now I'm inspecting the symbol
    // names dynamically to figure out how to dispatch to them.  If it becomes more complicated than this we can 
    // turn the g_dispatch_functions char * array into a template table, and populate the DispatchFunction map
    // from there.

    for (int i = 0; g_dispatch_functions[i].name != NULL; i++)
    {
        ConstString name_const_str(g_dispatch_functions[i].name);
        const Symbol *msgSend_symbol = m_objc_module_sp->FindFirstSymbolWithNameAndType (name_const_str, eSymbolTypeCode);
        if (msgSend_symbol)
        {
            // FixMe: Make g_dispatch_functions static table of DisptachFunctions, and have the map be address->index.
            // Problem is we also need to lookup the dispatch function.  For now we could have a side table of stret & non-stret
            // dispatch functions.  If that's as complex as it gets, we're fine.
            
            lldb::addr_t sym_addr = msgSend_symbol->GetValue().GetLoadAddress(target);
            
            m_msgSend_map.insert(std::pair<lldb::addr_t, int>(sym_addr, i));
        }
    }
    
    // Build our vtable dispatch handler here:
    m_vtables_ap.reset(new AppleObjCVTables(process_sp, m_objc_module_sp));
    if (m_vtables_ap.get())
        m_vtables_ap->ReadRegions();        
}

ThreadPlanSP
AppleObjCTrampolineHandler::GetStepThroughDispatchPlan (Thread &thread, bool stop_others)
{
    ThreadPlanSP ret_plan_sp;
    lldb::addr_t curr_pc = thread.GetRegisterContext()->GetPC();
    
    DispatchFunction this_dispatch;
    bool found_it = false;
    
    MsgsendMap::iterator pos;
    pos = m_msgSend_map.find (curr_pc);
    if (pos != m_msgSend_map.end())
    {
        this_dispatch = g_dispatch_functions[(*pos).second];
        found_it = true;
    }
    
    if (!found_it)
    {
        uint32_t flags;
        if (m_vtables_ap.get())
        {
            found_it = m_vtables_ap->IsAddressInVTables (curr_pc, flags);
            if (found_it)
            {
                this_dispatch.name = "vtable";
                this_dispatch.stret_return 
                        = (flags & AppleObjCVTables::eOBJC_TRAMPOLINE_STRET) == AppleObjCVTables::eOBJC_TRAMPOLINE_STRET;
                this_dispatch.is_super = false;
                this_dispatch.fixedup = DispatchFunction::eFixUpFixed;
            }
        }
    }
    
    if (found_it)
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

        
        lldb::StackFrameSP thread_cur_frame = thread.GetStackFrameAtIndex(0);
        
        Process *process = thread.CalculateProcess();
        const ABI *abi = process->GetABI();
        if (abi == NULL)
            return ret_plan_sp;
            
        Target *target = thread.CalculateTarget();
        
        // FIXME: Since neither the value nor the Clang QualType know their ASTContext, 
        // we have to make sure the type we put in our value list comes from the same ASTContext
        // the ABI will use to get the argument values.  THis is the bottom-most frame's module.

        ClangASTContext *clang_ast_context = target->GetScratchClangASTContext();
        ValueList argument_values;
        Value input_value;
        void *clang_void_ptr_type = clang_ast_context->GetVoidPtrType(false);
        input_value.SetValueType (Value::eValueTypeScalar);
        input_value.SetContext (Value::eContextTypeClangType, clang_void_ptr_type);
        
        int obj_index;
        int sel_index;
        
        // If this is a struct return dispatch, then the first argument is the
        // return struct pointer, and the object is the second, and the selector is the third.
        // Otherwise the object is the first and the selector the second.
        if (this_dispatch.stret_return)
        {
            obj_index = 1;
            sel_index = 2;
            argument_values.PushValue(input_value);
            argument_values.PushValue(input_value);
            argument_values.PushValue(input_value);
        }
        else
        {
            obj_index = 0;
            sel_index = 1;
            argument_values.PushValue(input_value);
            argument_values.PushValue(input_value);
        }

        
        bool success = abi->GetArgumentValues (thread, argument_values);
        if (!success)
            return ret_plan_sp;
        
        // Okay, the first value here is the object, we actually want the class of that object.
        // For now we're just going with the ISA.  
        // FIXME: This should really be the return value of [object class] to properly handle KVO interposition.
        
        Value isa_value(*(argument_values.GetValueAtIndex(obj_index)));
        
        // This is a little cheesy, but since object->isa is the first field, 
        // making the object value a load address value and resolving it will get
        // the pointer sized data pointed to by that value...
        ExecutionContext exec_ctx;
        thread.CalculateExecutionContext (exec_ctx);

        isa_value.SetValueType(Value::eValueTypeLoadAddress);
        isa_value.ResolveValue(&exec_ctx, clang_ast_context->getASTContext());
        
        if (this_dispatch.fixedup == DispatchFunction::eFixUpFixed)
        {
            // For the FixedUp method the Selector is actually a pointer to a 
            // structure, the second field of which is the selector number.
            Value *sel_value = argument_values.GetValueAtIndex(sel_index);
            sel_value->GetScalar() += process->GetAddressByteSize();
            sel_value->SetValueType(Value::eValueTypeLoadAddress);
            sel_value->ResolveValue(&exec_ctx, clang_ast_context->getASTContext());            
        }
        else if (this_dispatch.fixedup == DispatchFunction::eFixUpToFix)
        {   
            // FIXME: If the method dispatch is not "fixed up" then the selector is actually a
            // pointer to the string name of the selector.  We need to look that up...
            // For now I'm going to punt on that and just return no plan.
            if (log)
                log->Printf ("Punting on stepping into un-fixed-up method dispatch.");
            return ret_plan_sp;
        }
        
        // FIXME: If this is a dispatch to the super-class, we need to get the super-class from
        // the class, and disaptch to that instead.
        // But for now I just punt and return no plan.
        if (this_dispatch.is_super)
        {   
            if (log)
                log->Printf ("Punting on stepping into super method dispatch.");
            return ret_plan_sp;
        }
        
        ValueList dispatch_values;
        dispatch_values.PushValue (isa_value);
        dispatch_values.PushValue(*(argument_values.GetValueAtIndex(sel_index)));
        
        if (log)
        {
            log->Printf("Resolving method call for class - 0x%llx and selector - 0x%llx",
                        dispatch_values.GetValueAtIndex(0)->GetScalar().ULongLong(),
                        dispatch_values.GetValueAtIndex(1)->GetScalar().ULongLong());
        }
        ObjCLanguageRuntime *objc_runtime = m_process_sp->GetObjCLanguageRuntime ();
        assert(objc_runtime != NULL);
        lldb::addr_t impl_addr = objc_runtime->LookupInMethodCache (dispatch_values.GetValueAtIndex(0)->GetScalar().ULongLong(),
                                                dispatch_values.GetValueAtIndex(1)->GetScalar().ULongLong());
                                                
        if (impl_addr == LLDB_INVALID_ADDRESS)
        {

            Address resolve_address(NULL, this_dispatch.stret_return ? m_impl_stret_fn_addr : m_impl_fn_addr);
            
            StreamString errors;
            { 
                // Scope for mutex locker:
                Mutex::Locker locker(m_impl_function_mutex);
                if (!m_impl_function.get())
                {
                     m_impl_function.reset(new ClangFunction(process->GetTargetTriple().GetCString(), 
                                                             clang_ast_context, 
                                                             clang_void_ptr_type, 
                                                             resolve_address, 
                                                             dispatch_values));
                            
                    unsigned num_errors = m_impl_function->CompileFunction(errors);
                    if (num_errors)
                    {
                        if (log)
                            log->Printf ("Error compiling function: \"%s\".", errors.GetData());
                        return ret_plan_sp;
                    }
                    
                    errors.Clear();
                    if (!m_impl_function->WriteFunctionWrapper(exec_ctx, errors))
                    {
                        if (log)
                            log->Printf ("Error Inserting function: \"%s\".", errors.GetData());
                        return ret_plan_sp;
                    }
                }
                
            }
            
            errors.Clear();
            
            // Now write down the argument values for this call.
            lldb::addr_t args_addr = LLDB_INVALID_ADDRESS;
            if (!m_impl_function->WriteFunctionArguments (exec_ctx, args_addr, resolve_address, dispatch_values, errors))
                return ret_plan_sp;
        
            ret_plan_sp.reset (new AppleThreadPlanStepThroughObjCTrampoline (thread, this, args_addr, 
                                                                        argument_values.GetValueAtIndex(0)->GetScalar().ULongLong(),
                                                                        dispatch_values.GetValueAtIndex(0)->GetScalar().ULongLong(),
                                                                        dispatch_values.GetValueAtIndex(1)->GetScalar().ULongLong(),
                                                                        stop_others));
            if (log)
            {
                StreamString s;
                ret_plan_sp->GetDescription(&s, eDescriptionLevelFull);
                log->Printf("Using ObjC step plan: %s.\n", s.GetData());
            }
        }
        else
        {
            if (log)
                log->Printf ("Found implementation address in cache: 0x%llx", impl_addr);
                 
            ret_plan_sp.reset (new ThreadPlanRunToAddress (thread, impl_addr, stop_others));
        }
    }
    
    return ret_plan_sp;
}

ClangFunction *
AppleObjCTrampolineHandler::GetLookupImplementationWrapperFunction ()
{
    return m_impl_function.get();
}
