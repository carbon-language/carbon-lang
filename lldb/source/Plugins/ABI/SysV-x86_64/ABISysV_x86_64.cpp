//===-- ABISysV_x86_64.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ABISysV_x86_64.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"

#include "llvm/ADT/Triple.h"

using namespace lldb;
using namespace lldb_private;

static const char *pluginName = "ABISysV_x86_64";
static const char *pluginDesc = "System V ABI for x86_64 targets";
static const char *pluginShort = "abi.sysv-x86_64";



size_t
ABISysV_x86_64::GetRedZoneSize () const
{
    return 128;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
ABISP
ABISysV_x86_64::CreateInstance (const ArchSpec &arch)
{
    static ABISP g_abi_sp;
    if (arch.GetTriple().getArch() == llvm::Triple::x86_64)
    {
        if (!g_abi_sp)
            g_abi_sp.reset (new ABISysV_x86_64);
        return g_abi_sp;
    }
    return ABISP();
}

bool
ABISysV_x86_64::PrepareTrivialCall (Thread &thread, 
                                    addr_t sp, 
                                    addr_t func_addr, 
                                    addr_t return_addr, 
                                    addr_t *arg1_ptr,
                                    addr_t *arg2_ptr,
                                    addr_t *arg3_ptr,
                                    addr_t *arg4_ptr,
                                    addr_t *arg5_ptr,
                                    addr_t *arg6_ptr) const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
        log->Printf("ABISysV_x86_64::PrepareTrivialCall\n(\n  thread = %p\n  sp = 0x%llx\n  func_addr = 0x%llx\n  return_addr = 0x%llx\n  arg1_ptr = %p (0x%llx)\n  arg2_ptr = %p (0x%llx)\n  arg3_ptr = %p (0x%llx)\n)",
                    (void*)&thread,
                    (uint64_t)sp,
                    (uint64_t)func_addr,
                    (uint64_t)return_addr,
                    arg1_ptr, arg1_ptr ? (uint64_t)*arg1_ptr : (uint64_t) 0,
                    arg2_ptr, arg2_ptr ? (uint64_t)*arg2_ptr : (uint64_t) 0,
                    arg3_ptr, arg3_ptr ? (uint64_t)*arg3_ptr : (uint64_t) 0);
    
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return false;
    
    const RegisterInfo *reg_info = NULL;
    if (arg1_ptr)
    {
        reg_info = reg_ctx->GetRegisterInfoByName("rdi", 0);
        if (log)
            log->Printf("About to write arg1 (0x%llx) into %s", (uint64_t)*arg1_ptr, reg_info->name);

        if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg1_ptr))
            return false;

        if (arg2_ptr)
        {
            reg_info = reg_ctx->GetRegisterInfoByName("rsi", 0);
            if (log)
                log->Printf("About to write arg2 (0x%llx) into %s", (uint64_t)*arg2_ptr, reg_info->name);
            if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg2_ptr))
                return false;

            if (arg3_ptr)
            {
                reg_info = reg_ctx->GetRegisterInfoByName("rdx", 0);
                if (log)
                    log->Printf("About to write arg3 (0x%llx) into %s", (uint64_t)*arg3_ptr, reg_info->name);
                if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg3_ptr))
                    return false;

                if (arg4_ptr)
                {
                    reg_info = reg_ctx->GetRegisterInfoByName("rcx", 0);
                    if (log)
                        log->Printf("About to write arg4 (0x%llx) into %s", (uint64_t)*arg4_ptr, reg_info->name);
                    if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg4_ptr))
                        return false;

                    if (arg5_ptr)
                    {
                        reg_info = reg_ctx->GetRegisterInfoByName("r8", 0);
                        if (log)
                            log->Printf("About to write arg5 (0x%llx) into %s", (uint64_t)*arg5_ptr, reg_info->name);
                        if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg5_ptr))
                            return false;

                        if (arg6_ptr)
                        {
                            reg_info = reg_ctx->GetRegisterInfoByName("r9", 0);
                            if (log)
                                log->Printf("About to write arg6 (0x%llx) into %s", (uint64_t)*arg6_ptr, reg_info->name);
                            if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg6_ptr))
                                return false;
                        }
                    }
                }
            }
        }
    }


    // First, align the SP

    if (log)
        log->Printf("16-byte aligning SP: 0x%llx to 0x%llx", (uint64_t)sp, (uint64_t)(sp & ~0xfull));

    sp &= ~(0xfull); // 16-byte alignment

    // The return address is pushed onto the stack (yes after the alignment...)
    sp -= 8;

    RegisterValue reg_value;
    reg_value.SetUInt64 (return_addr);

    if (log)
        log->Printf("Pushing the return address onto the stack: new SP 0x%llx, return address 0x%llx", (uint64_t)sp, (uint64_t)return_addr);

    const RegisterInfo *pc_reg_info = reg_ctx->GetRegisterInfoByName("rip");
    Error error (reg_ctx->WriteRegisterValueToMemory(pc_reg_info, sp, pc_reg_info->byte_size, reg_value));
    if (error.Fail())
        return false;

    // %rsp is set to the actual stack value.

    if (log)
        log->Printf("Writing SP (0x%llx) down", (uint64_t)sp);
    
    if (!reg_ctx->WriteRegisterFromUnsigned (reg_ctx->GetRegisterInfoByName("rsp"), sp))
        return false;

    // %rip is set to the address of the called function.
    
    if (log)
        log->Printf("Writing new IP (0x%llx) down", (uint64_t)func_addr);

    if (!reg_ctx->WriteRegisterFromUnsigned (pc_reg_info, func_addr))
        return false;

    return true;
}

static bool ReadIntegerArgument(Scalar           &scalar,
                                unsigned int     bit_width,
                                bool             is_signed,
                                Thread           &thread,
                                uint32_t         *argument_register_ids,
                                unsigned int     &current_argument_register,
                                addr_t           &current_stack_argument)
{
    if (bit_width > 64)
        return false; // Scalar can't hold large integer arguments
    
    uint64_t arg_contents;
    
    if (current_argument_register < 6)
    {
        arg_contents = thread.GetRegisterContext()->ReadRegisterAsUnsigned(argument_register_ids[current_argument_register], 0);
        current_argument_register++;
    }
    else
    {
        uint8_t arg_data[sizeof(arg_contents)];
        Error error;
        thread.GetProcess().ReadMemory(current_stack_argument, arg_data, sizeof(arg_contents), error);
        DataExtractor arg_data_extractor (arg_data, sizeof(arg_contents), 
                                          thread.GetProcess().GetTarget().GetArchitecture().GetByteOrder(), 
                                          thread.GetProcess().GetTarget().GetArchitecture().GetAddressByteSize());
        uint32_t offset = 0;
        arg_contents = arg_data_extractor.GetMaxU64(&offset, bit_width / 8);
        if (!offset)
            return false;
        current_stack_argument += (bit_width / 8);
    }
    
    if (is_signed)
    {
        switch (bit_width)
        {
        default:
            return false;
        case 8:
            scalar = (int8_t)(arg_contents & 0xff);
            break;
        case 16:
            scalar = (int16_t)(arg_contents & 0xffff);
            break;
        case 32:
            scalar = (int32_t)(arg_contents & 0xffffffff);
            break;
        case 64:
            scalar = (int64_t)arg_contents;
            break;
        }
    }
    else
    {
        switch (bit_width)
        {
        default:
            return false;
        case 8:
            scalar = (uint8_t)(arg_contents & 0xffu);
            break;
        case 16:
            scalar = (uint16_t)(arg_contents & 0xffffu);
            break;
        case 32:
            scalar = (uint32_t)(arg_contents & 0xffffffffu);
            break;
        case 64:
            scalar = (uint64_t)arg_contents;
            break;
        }
    }
    
    return true;
}

bool
ABISysV_x86_64::GetArgumentValues (Thread &thread,
                                   ValueList &values) const
{
    unsigned int num_values = values.GetSize();
    unsigned int value_index;
    
    // For now, assume that the types in the AST values come from the Target's 
    // scratch AST.    
    
    clang::ASTContext *ast_context = thread.CalculateTarget()->GetScratchClangASTContext()->getASTContext();
    
    // Extract the register context so we can read arguments from registers
    
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    
    if (!reg_ctx)
        return false;
    
    // Get the pointer to the first stack argument so we have a place to start 
    // when reading data
    
    addr_t sp = reg_ctx->GetSP(0);
    
    if (!sp)
        return false;
    
    addr_t current_stack_argument = sp + 8; // jump over return address
    
    uint32_t argument_register_ids[6];
    
    argument_register_ids[0] = reg_ctx->GetRegisterInfoByName("rdi", 0)->kinds[eRegisterKindLLDB];
    argument_register_ids[1] = reg_ctx->GetRegisterInfoByName("rsi", 0)->kinds[eRegisterKindLLDB];
    argument_register_ids[2] = reg_ctx->GetRegisterInfoByName("rdx", 0)->kinds[eRegisterKindLLDB];
    argument_register_ids[3] = reg_ctx->GetRegisterInfoByName("rcx", 0)->kinds[eRegisterKindLLDB];
    argument_register_ids[4] = reg_ctx->GetRegisterInfoByName("r8", 0)->kinds[eRegisterKindLLDB];
    argument_register_ids[5] = reg_ctx->GetRegisterInfoByName("r9", 0)->kinds[eRegisterKindLLDB];
    
    unsigned int current_argument_register = 0;
    
    for (value_index = 0;
         value_index < num_values;
         ++value_index)
    {
        Value *value = values.GetValueAtIndex(value_index);
    
        if (!value)
            return false;
        
        // We currently only support extracting values with Clang QualTypes.
        // Do we care about others?
        switch (value->GetContextType())
        {
        default:
            return false;
        case Value::eContextTypeClangType:
            {
                void *value_type = value->GetClangType();
                bool is_signed;
                
                if (ClangASTContext::IsIntegerType (value_type, is_signed))
                {
                    size_t bit_width = ClangASTType::GetClangTypeBitWidth(ast_context, value_type);
                    
                    ReadIntegerArgument(value->GetScalar(),
                                        bit_width, 
                                        is_signed,
                                        thread, 
                                        argument_register_ids, 
                                        current_argument_register,
                                        current_stack_argument);
                }
                else if (ClangASTContext::IsPointerType (value_type))
                {
                    ReadIntegerArgument(value->GetScalar(),
                                        64,
                                        false,
                                        thread,
                                        argument_register_ids, 
                                        current_argument_register,
                                        current_stack_argument);
                }
            }
            break;
        }
    }
    
    return true;
}

bool
ABISysV_x86_64::GetReturnValue (Thread &thread,
                                Value &value) const
{
    switch (value.GetContextType())
    {
        default:
            return false;
        case Value::eContextTypeClangType:
        {
            void *value_type = value.GetClangType();
            bool is_signed;
            
            RegisterContext *reg_ctx = thread.GetRegisterContext().get();
            
            if (!reg_ctx)
                return false;
            
            if (ClangASTContext::IsIntegerType (value_type, is_signed))
            {
                // For now, assume that the types in the AST values come from the Target's 
                // scratch AST.    
                
                clang::ASTContext *ast_context = thread.CalculateTarget()->GetScratchClangASTContext()->getASTContext();
                
                // Extract the register context so we can read arguments from registers
                
                size_t bit_width = ClangASTType::GetClangTypeBitWidth(ast_context, value_type);
                unsigned rax_id = reg_ctx->GetRegisterInfoByName("rax", 0)->kinds[eRegisterKindLLDB];
                
                switch (bit_width)
                {
                default:
                case 128:
                    // Scalar can't hold 128-bit literals, so we don't handle this
                    return false;
                case 64:
                    if (is_signed)
                        value.GetScalar() = (int64_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0));
                    else
                        value.GetScalar() = (uint64_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0));
                    break;
                case 32:
                    if (is_signed)
                        value.GetScalar() = (int32_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xffffffff);
                    else
                        value.GetScalar() = (uint32_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xffffffff);
                    break;
                case 16:
                    if (is_signed)
                        value.GetScalar() = (int16_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xffff);
                    else
                        value.GetScalar() = (uint16_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xffff);
                    break;
                case 8:
                    if (is_signed)
                        value.GetScalar() = (int8_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xff);
                    else
                        value.GetScalar() = (uint8_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xff);
                    break;
                }
            }
            else if (ClangASTContext::IsPointerType (value_type))
            {
                unsigned rax_id = reg_ctx->GetRegisterInfoByName("rax", 0)->kinds[eRegisterKindLLDB];
                value.GetScalar() = (uint64_t)thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0);
            }
            else
            {
                // not handled yet
                return false;
            }
        }
        break;
    }
    
    return true;
}

bool
ABISysV_x86_64::CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan)
{
    uint32_t reg_kind = unwind_plan.GetRegisterKind();
    uint32_t sp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t pc_reg_num = LLDB_INVALID_REGNUM;
    
    switch (reg_kind)
    {
    case eRegisterKindDWARF:
    case eRegisterKindGCC:
        sp_reg_num = gcc_dwarf_rsp;
        pc_reg_num = gcc_dwarf_rip;
        break;

    case eRegisterKindGDB:
        sp_reg_num = gdb_rsp;
        pc_reg_num = gdb_rip;
        break;

    case eRegisterKindGeneric:
        sp_reg_num = LLDB_REGNUM_GENERIC_SP;
        pc_reg_num = LLDB_REGNUM_GENERIC_PC;
        break;
    }

    if (sp_reg_num == LLDB_INVALID_REGNUM ||
        pc_reg_num == LLDB_INVALID_REGNUM)
        return false;

    UnwindPlan::Row row;
    row.SetCFARegister (sp_reg_num);
    row.SetCFAOffset (8);
    row.SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, -8, false);    
    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName (pluginName);
    return true;
}

bool
ABISysV_x86_64::CreateDefaultUnwindPlan (UnwindPlan &unwind_plan)
{
    uint32_t reg_kind = unwind_plan.GetRegisterKind();
    uint32_t fp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t sp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t pc_reg_num = LLDB_INVALID_REGNUM;
    
    switch (reg_kind)
    {
        case eRegisterKindDWARF:
        case eRegisterKindGCC:
            fp_reg_num = gcc_dwarf_rbp;
            sp_reg_num = gcc_dwarf_rsp;
            pc_reg_num = gcc_dwarf_rip;
            break;
            
        case eRegisterKindGDB:
            fp_reg_num = gdb_rbp;
            sp_reg_num = gdb_rsp;
            pc_reg_num = gdb_rip;
            break;
            
        case eRegisterKindGeneric:
            fp_reg_num = LLDB_REGNUM_GENERIC_FP;
            sp_reg_num = LLDB_REGNUM_GENERIC_SP;
            pc_reg_num = LLDB_REGNUM_GENERIC_PC;
            break;
    }

    if (fp_reg_num == LLDB_INVALID_REGNUM ||
        sp_reg_num == LLDB_INVALID_REGNUM ||
        pc_reg_num == LLDB_INVALID_REGNUM)
        return false;

    UnwindPlan::Row row;

    const int32_t ptr_size = 8;
    row.SetCFARegister (LLDB_REGNUM_GENERIC_FP);
    row.SetCFAOffset (2 * ptr_size);
    row.SetOffset (0);
    
    row.SetRegisterLocationToAtCFAPlusOffset(fp_reg_num, ptr_size * -2, true);
    row.SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, ptr_size * -1, true);
    row.SetRegisterLocationToAtCFAPlusOffset(sp_reg_num, ptr_size *  0, true);

    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("x86_64 default unwind plan");
    return true;
}

bool
ABISysV_x86_64::RegisterIsVolatile (const RegisterInfo *reg_info)
{
    return !RegisterIsCalleeSaved (reg_info);
}

bool
ABISysV_x86_64::RegisterIsCalleeSaved (const RegisterInfo *reg_info)
{
    if (reg_info)
    {
        // Volatile registers include: rbx, rbp, rsp, r12, r13, r14, r15, rip
        const char *name = reg_info->name;
        if (name[0] == 'r')
        {
            switch (name[1])
            {
            case '1': // r12, r13, r14, r15
                if (name[2] >= '2' && name[2] <= '5')
                    return name[3] == '\0';
                break;

            case 'b': // rbp, rbx
                if (name[2] == 'p' || name[2] == 'x')
                    return name[3] == '\0';
                break;

            case 'i': // rip
                if (name[2] == 'p')
                    return name[3] == '\0'; 
                break;

            case 's': // rsp
                if (name[2] == 'p')
                    return name[3] == '\0';
                break;

            default:
                break;
            }
        }
    }
    return false;
}



void
ABISysV_x86_64::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);
}

void
ABISysV_x86_64::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ABISysV_x86_64::GetPluginName()
{
    return pluginName;
}

const char *
ABISysV_x86_64::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
ABISysV_x86_64::GetPluginVersion()
{
    return 1;
}

