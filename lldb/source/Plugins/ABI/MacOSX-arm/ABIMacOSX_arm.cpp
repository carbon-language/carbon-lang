//===-- ABIMacOSX_arm.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ABIMacOSX_arm.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "llvm/ADT/Triple.h"

#include "Utility/ARM_DWARF_Registers.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

static const char *pluginName = "ABIMacOSX_arm";
static const char *pluginDesc = "Mac OS X ABI for arm targets";
static const char *pluginShort = "abi.macosx-arm";

size_t
ABIMacOSX_arm::GetRedZoneSize () const
{
    return 0;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
ABISP
ABIMacOSX_arm::CreateInstance (const ArchSpec &arch)
{
    static ABISP g_abi_sp;
    const llvm::Triple::ArchType arch_type = arch.GetTriple().getArch();
    if ((arch_type == llvm::Triple::arm) ||
        (arch_type == llvm::Triple::thumb))
    {
        if (!g_abi_sp)
            g_abi_sp.reset (new ABIMacOSX_arm);
        return g_abi_sp;
    }
    return ABISP();
}

bool
ABIMacOSX_arm::PrepareTrivialCall (Thread &thread, 
                                   addr_t sp, 
                                   addr_t function_addr, 
                                   addr_t return_addr, 
                                   addr_t *arg1_ptr,
                                   addr_t *arg2_ptr,
                                   addr_t *arg3_ptr,
                                   addr_t *arg4_ptr,
                                   addr_t *arg5_ptr,
                                   addr_t *arg6_ptr) const
{
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return false;    

    const uint32_t pc_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    const uint32_t sp_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    const uint32_t ra_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA);

    RegisterValue reg_value;

    if (arg1_ptr)
    {
        reg_value.SetUInt32(*arg1_ptr);
        if (!reg_ctx->WriteRegister (reg_ctx->GetRegisterInfoByName("r0"), reg_value))
            return false;

        if (arg2_ptr)
        {
            reg_value.SetUInt32(*arg2_ptr);
            if (!reg_ctx->WriteRegister (reg_ctx->GetRegisterInfoByName("r1"), reg_value))
                return false;

            if (arg3_ptr)
            {
                reg_value.SetUInt32(*arg3_ptr);
                if (!reg_ctx->WriteRegister (reg_ctx->GetRegisterInfoByName("r2"), reg_value))
                    return false;
                if (arg4_ptr)
                {
                    reg_value.SetUInt32(*arg4_ptr);
                    const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName("r3");
                    if (!reg_ctx->WriteRegister (reg_info, reg_value))
                        return false;
                    if (arg5_ptr)
                    {
                        reg_value.SetUInt32(*arg5_ptr);
                        if (reg_ctx->WriteRegisterValueToMemory (reg_info, sp, reg_info->byte_size, reg_value).Fail())
                            return false;
                        if (arg6_ptr)
                        {
                            reg_value.SetUInt32(*arg6_ptr);
                            if (reg_ctx->WriteRegisterValueToMemory (reg_info, sp + 4, reg_info->byte_size, reg_value).Fail())
                                return false;
                        }
                    }
                }
            }            
        }
    }

    // Set "lr" to the return address into "lr"
    if (!reg_ctx->WriteRegisterFromUnsigned (ra_reg_num, return_addr))
        return false;

    // Set "sp" to the requested value
    if (!reg_ctx->WriteRegisterFromUnsigned (sp_reg_num, sp))
        return false;
    
    // Set "pc" to the address requested
    if (!reg_ctx->WriteRegisterFromUnsigned (pc_reg_num, function_addr))
        return false;
    
    return true;
}


static bool 
ReadIntegerArgument (Scalar &scalar,
                     unsigned int bit_width,
                     bool is_signed,
                     Process &process,
                     addr_t &current_stack_argument)
{
//    if (bit_width > 64)
//        return false; // Scalar can't hold large integer arguments
//    
//    uint64_t arg_contents;
//    uint32_t read_data;
//    Error error;
//    
//    if (bit_width > 32)
//    {
//        if (process.ReadMemory(current_stack_argument, &read_data, sizeof(read_data), error) != sizeof(read_data))
//            return false;
//        
//        arg_contents = read_data;
//        
//        if (process.ReadMemory(current_stack_argument + 4, &read_data, sizeof(read_data), error) != sizeof(read_data))
//            return false;
//        
//        arg_contents |= ((uint64_t)read_data) << 32;
//        
//        current_stack_argument += 8;
//    }
//    else {
//        if (process.ReadMemory(current_stack_argument, &read_data, sizeof(read_data), error) != sizeof(read_data))
//            return false;
//        
//        arg_contents = read_data;
//        
//        current_stack_argument += 4;
//    }
//    
//    if (is_signed)
//    {
//        switch (bit_width)
//        {
//        default:
//            return false;
//        case 8:
//            scalar = (int8_t)(arg_contents & 0xff);
//            break;
//        case 16:
//            scalar = (int16_t)(arg_contents & 0xffff);
//            break;
//        case 32:
//            scalar = (int32_t)(arg_contents & 0xffffffff);
//            break;
//        case 64:
//            scalar = (int64_t)arg_contents;
//            break;
//        }
//    }
//    else
//    {
//        switch (bit_width)
//        {
//        default:
//            return false;
//        case 8:
//            scalar = (uint8_t)(arg_contents & 0xff);
//            break;
//        case 16:
//            scalar = (uint16_t)(arg_contents & 0xffff);
//            break;
//        case 32:
//            scalar = (uint32_t)(arg_contents & 0xffffffff);
//            break;
//        case 64:
//            scalar = (uint64_t)arg_contents;
//            break;
//        }
//    }
//    
//    return true;
    return false;
}

bool
ABIMacOSX_arm::GetArgumentValues (Thread &thread,
                                  ValueList &values) const
{
//    unsigned int num_values = values.GetSize();
//    unsigned int value_index;
//    
//    // Extract the Clang AST context from the PC so that we can figure out type
//    // sizes
//    
//    clang::ASTContext *ast_context = thread.CalculateTarget()->GetScratchClangASTContext()->getASTContext();
//    
//    // Get the pointer to the first stack argument so we have a place to start 
//    // when reading data
//    
//    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
//    
//    if (!reg_ctx)
//        return false;
//    
//    addr_t sp = reg_ctx->GetSP(0);
//    
//    if (!sp)
//        return false;
//    
//    addr_t current_stack_argument = sp + 4; // jump over return address
//    
//    for (value_index = 0;
//         value_index < num_values;
//         ++value_index)
//    {
//        Value *value = values.GetValueAtIndex(value_index);
//        
//        if (!value)
//            return false;
//        
//        // We currently only support extracting values with Clang QualTypes.
//        // Do we care about others?
//        switch (value->GetContextType())
//        {
//            default:
//                return false;
//            case Value::eContextTypeClangType:
//            {
//                void *value_type = value->GetClangType();
//                bool is_signed;
//                
//                if (ClangASTContext::IsIntegerType (value_type, is_signed))
//                {
//                    size_t bit_width = ClangASTType::GetClangTypeBitWidth(ast_context, value_type);
//                    
//                    ReadIntegerArgument(value->GetScalar(),
//                                        bit_width, 
//                                        is_signed,
//                                        thread.GetProcess(), 
//                                        current_stack_argument);
//                }
//                else if (ClangASTContext::IsPointerType (value_type))
//                {
//                    ReadIntegerArgument(value->GetScalar(),
//                                        32,
//                                        false,
//                                        thread.GetProcess(),
//                                        current_stack_argument);
//                }
//            }
//                break;
//        }
//    }
//    
//    return true;
    return false;
}

bool
ABIMacOSX_arm::GetReturnValue (Thread &thread,
                               Value &value) const
{
    switch (value.GetContextType())
    {
        default:
            return false;
        case Value::eContextTypeClangType:
        {
            // Extract the Clang AST context from the PC so that we can figure out type
            // sizes
            
            clang::ASTContext *ast_context = thread.CalculateTarget()->GetScratchClangASTContext()->getASTContext();
            
            // Get the pointer to the first stack argument so we have a place to start 
            // when reading data
            
            RegisterContext *reg_ctx = thread.GetRegisterContext().get();
            
            void *value_type = value.GetClangType();
            bool is_signed;
            
            const RegisterInfo *r0_reg_info = reg_ctx->GetRegisterInfoByName("r0", 0);
            if (ClangASTContext::IsIntegerType (value_type, is_signed))
            {
                size_t bit_width = ClangASTType::GetClangTypeBitWidth(ast_context, value_type);
                
                switch (bit_width)
                {
                    default:
                        return false;
                    case 64:
                    {
                        const RegisterInfo *r1_reg_info = reg_ctx->GetRegisterInfoByName("r1", 0);
                        uint64_t raw_value;
                        raw_value = reg_ctx->ReadRegisterAsUnsigned(r0_reg_info, 0) & UINT32_MAX;
                        raw_value |= ((uint64_t)(reg_ctx->ReadRegisterAsUnsigned(r1_reg_info, 0) & UINT32_MAX)) << 32;
                        if (is_signed)
                            value.GetScalar() = (int64_t)raw_value;
                        else
                            value.GetScalar() = (uint64_t)raw_value;
                    }
                        break;
                    case 32:
                        if (is_signed)
                            value.GetScalar() = (int32_t)(reg_ctx->ReadRegisterAsUnsigned(r0_reg_info, 0) & UINT32_MAX);
                        else
                            value.GetScalar() = (uint32_t)(reg_ctx->ReadRegisterAsUnsigned(r0_reg_info, 0) & UINT32_MAX);
                        break;
                    case 16:
                        if (is_signed)
                            value.GetScalar() = (int16_t)(reg_ctx->ReadRegisterAsUnsigned(r0_reg_info, 0) & UINT16_MAX);
                        else
                            value.GetScalar() = (uint16_t)(reg_ctx->ReadRegisterAsUnsigned(r0_reg_info, 0) & UINT16_MAX);
                        break;
                    case 8:
                        if (is_signed)
                            value.GetScalar() = (int8_t)(reg_ctx->ReadRegisterAsUnsigned(r0_reg_info, 0) & UINT8_MAX);
                        else
                            value.GetScalar() = (uint8_t)(reg_ctx->ReadRegisterAsUnsigned(r0_reg_info, 0) & UINT8_MAX);
                        break;
                }
            }
            else if (ClangASTContext::IsPointerType (value_type))
            {
                uint32_t ptr = thread.GetRegisterContext()->ReadRegisterAsUnsigned(r0_reg_info, 0) & UINT32_MAX;
                value.GetScalar() = ptr;
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
ABIMacOSX_arm::CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan)
{
    uint32_t reg_kind = unwind_plan.GetRegisterKind();
    uint32_t lr_reg_num = LLDB_INVALID_REGNUM;
    uint32_t sp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t pc_reg_num = LLDB_INVALID_REGNUM;
    
    switch (reg_kind)
    {
        case eRegisterKindDWARF:
        case eRegisterKindGCC:
            lr_reg_num = dwarf_lr;
            sp_reg_num = dwarf_sp;
            pc_reg_num = dwarf_pc;
            break;
            
        case eRegisterKindGeneric:
            lr_reg_num = LLDB_REGNUM_GENERIC_RA;
            sp_reg_num = LLDB_REGNUM_GENERIC_SP;
            pc_reg_num = LLDB_REGNUM_GENERIC_PC;
            break;
    }
    
    if (lr_reg_num == LLDB_INVALID_REGNUM ||
        sp_reg_num == LLDB_INVALID_REGNUM ||
        pc_reg_num == LLDB_INVALID_REGNUM)
        return false;

    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);
    
    UnwindPlan::Row row;
    
    // Our previous Call Frame Address is the stack pointer
    row.SetCFARegister (sp_reg_num);
    
    // Our previous PC is in the LR
    row.SetRegisterLocationToRegister(pc_reg_num, lr_reg_num, true);
    unwind_plan.AppendRow (row);
    
    // All other registers are the same.
    
    unwind_plan.SetSourceName (pluginName);
    return true;
}

bool
ABIMacOSX_arm::CreateDefaultUnwindPlan (UnwindPlan &unwind_plan)
{
    uint32_t reg_kind = unwind_plan.GetRegisterKind();
    uint32_t fp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t sp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t pc_reg_num = LLDB_INVALID_REGNUM;
    
    switch (reg_kind)
    {
        case eRegisterKindDWARF:
        case eRegisterKindGCC:
            fp_reg_num = dwarf_r7; // apple uses r7 for all frames. Normal arm uses r11
            sp_reg_num = dwarf_sp;
            pc_reg_num = dwarf_pc;
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
    
    unwind_plan.SetRegisterKind (eRegisterKindGeneric);
    row.SetCFARegister (fp_reg_num);
    row.SetCFAOffset (2 * ptr_size);
    row.SetOffset (0);
    
    row.SetRegisterLocationToAtCFAPlusOffset(fp_reg_num, ptr_size * -2, true);
    row.SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, ptr_size * -1, true);
    
    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("arm-apple-darwin default unwind plan");
    return true;
}

bool
ABIMacOSX_arm::RegisterIsVolatile (const RegisterInfo *reg_info)
{
    if (reg_info)
    {
        // Volatile registers include: ebx, ebp, esi, edi, esp, eip
        const char *name = reg_info->name;
        if (name[0] == 'r')
        {
            switch (name[1])
            {
                case '0': return name[2] == '\0'; // r0
                case '1': 
                    switch (name[2])
                    {
                    case '\0':
                        return true; // r1
                    case '2':
                    case '3':
                        return name[2] == '\0'; // r12 - r13
                    default:
                        break;
                    }
                    break;

                case '2': return name[2] == '\0'; // r2
                case '3': return name[2] == '\0'; // r3
                case '9': return name[2] == '\0'; // r9 (apple-darwin only...)
                    
                break;
            }
        }
        else if (name[0] == 'd')
        {
            switch (name[1])
            {
                case '0': 
                    return name[2] == '\0'; // d0

                case '1':
                    switch (name[2])
                    {
                    case '\0':
                        return true; // d1;
                    case '6':
                    case '7':
                    case '8':
                    case '9':
                        return name[3] == '\0'; // d16 - d19
                    default:
                        break;
                    }
                    break;
                    
                case '2':
                    switch (name[2])
                    {
                    case '\0':
                        return true; // d2;
                    case '0':
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                    case '6':
                    case '7':
                    case '8':
                    case '9':
                        return name[3] == '\0'; // d20 - d29
                    default:
                        break;
                    }
                    break;

                case '3':
                    switch (name[2])
                    {
                    case '\0':
                        return true; // d3;
                    case '0':
                    case '1':
                        return name[3] == '\0'; // d30 - d31
                    default:
                        break;
                    }
                case '4':
                case '5':
                case '6':
                case '7':
                    return name[2] == '\0'; // d4 - d7

                default:
                    break;
            }
        }
        else if (name[0] == 's' && name[1] == 'p' && name[2] == '\0')
            return true;
    }
    return false;
}

void
ABIMacOSX_arm::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);    
}

void
ABIMacOSX_arm::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ABIMacOSX_arm::GetPluginName()
{
    return pluginName;
}

const char *
ABIMacOSX_arm::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
ABIMacOSX_arm::GetPluginVersion()
{
    return 1;
}

