//===-- ABIMacOSX_i386.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ABIMacOSX_i386.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "llvm/ADT/Triple.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

static const char *pluginName = "ABIMacOSX_i386";
static const char *pluginDesc = "Mac OS X ABI for i386 targets";
static const char *pluginShort = "abi.macosx-i386";

size_t
ABIMacOSX_i386::GetRedZoneSize () const
{
    return 0;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
ABISP
ABIMacOSX_i386::CreateInstance (const ArchSpec &arch)
{
    static ABISP g_abi_sp;
    if (arch.GetTriple().getArch() == llvm::Triple::x86)
    {
        if (!g_abi_sp)
            g_abi_sp.reset (new ABIMacOSX_i386);
        return g_abi_sp;
    }
    return ABISP();
}

bool
ABIMacOSX_i386::PrepareTrivialCall (Thread &thread, 
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
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return false;    
    uint32_t pc_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    uint32_t sp_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    
    // When writing a register value down to memory, the register info used 
    // to write memory just needs to have the correct size of a 32 bit register, 
    // the actual register it pertains to is not important, just the size needs 
    // to be correct. Here we use "eax"...
    const RegisterInfo *reg_info_32 = reg_ctx->GetRegisterInfoByName("eax");

    // Make room for the argument(s) on the stack

    Error error;
    RegisterValue reg_value;
    
    // Write any arguments onto the stack
    if (arg1_ptr)
    {
        sp -= 4;
        if (arg2_ptr)
        {
            sp -= 4;
            if (arg3_ptr)
            {
                sp -= 4;
                if (arg4_ptr)
                {
                    sp -= 4;
                    if (arg5_ptr)
                    {
                        sp -= 4;
                        if (arg6_ptr)
                        {
                            sp -= 4;
                        }
                    }
                }
            }
        }
    }

    // Align the SP    
    sp &= ~(16ull-1ull); // 16-byte alignment
    
    if (arg1_ptr)
    {
        reg_value.SetUInt32(*arg1_ptr);
        error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                     sp, 
                                                     reg_info_32->byte_size, 
                                                     reg_value);
        if (error.Fail())
            return false;

        if (arg2_ptr)
        {
            reg_value.SetUInt32(*arg2_ptr);
            // The register info used to write memory just needs to have the correct
            // size of a 32 bit register, the actual register it pertains to is not
            // important, just the size needs to be correct. Here we use "eax"...
            error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                         sp + 4, 
                                                         reg_info_32->byte_size, 
                                                         reg_value);
            if (error.Fail())
                return false;
            
            if (arg3_ptr)
            {
                reg_value.SetUInt32(*arg3_ptr);
                // The register info used to write memory just needs to have the correct
                // size of a 32 bit register, the actual register it pertains to is not
                // important, just the size needs to be correct. Here we use "eax"...
                error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                             sp + 8, 
                                                             reg_info_32->byte_size, 
                                                             reg_value);
                if (error.Fail())
                    return false;

                if (arg4_ptr)
                {
                    reg_value.SetUInt32(*arg4_ptr);
                    // The register info used to write memory just needs to have the correct
                    // size of a 32 bit register, the actual register it pertains to is not
                    // important, just the size needs to be correct. Here we use "eax"...
                    error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                                 sp + 12, 
                                                                 reg_info_32->byte_size, 
                                                                 reg_value);
                    if (error.Fail())
                        return false;
                    if (arg5_ptr)
                    {
                        reg_value.SetUInt32(*arg5_ptr);
                        // The register info used to write memory just needs to have the correct
                        // size of a 32 bit register, the actual register it pertains to is not
                        // important, just the size needs to be correct. Here we use "eax"...
                        error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                                     sp + 16, 
                                                                     reg_info_32->byte_size, 
                                                                     reg_value);
                        if (error.Fail())
                            return false;
                        if (arg6_ptr)
                        {
                            reg_value.SetUInt32(*arg6_ptr);
                            // The register info used to write memory just needs to have the correct
                            // size of a 32 bit register, the actual register it pertains to is not
                            // important, just the size needs to be correct. Here we use "eax"...
                            error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                                         sp + 20, 
                                                                         reg_info_32->byte_size, 
                                                                         reg_value);
                            if (error.Fail())
                                return false;
                        }
                    }
                }
            }
        }
    }
    
    
    // The return address is pushed onto the stack (yes after we just set the
    // alignment above!).
    sp -= 4;
    reg_value.SetUInt32(return_addr);
    error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                 sp, 
                                                 reg_info_32->byte_size, 
                                                 reg_value);
    if (error.Fail())
        return false;
    
    // %esp is set to the actual stack value.
    
    if (!reg_ctx->WriteRegisterFromUnsigned (sp_reg_num, sp))
        return false;
    
    // %eip is set to the address of the called function.
    
    if (!reg_ctx->WriteRegisterFromUnsigned (pc_reg_num, func_addr))
        return false;
    
    return true;
}

bool
ABIMacOSX_i386::PrepareNormalCall (Thread &thread,
                                   addr_t sp,
                                   addr_t func_addr,
                                   addr_t return_addr,
                                   ValueList &args) const
{
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return false;
    Error error;
    uint32_t fp_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP);
    uint32_t pc_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    uint32_t sp_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    
    // Do the argument layout
    
    std::vector <uint32_t> argLayout;   // 4-byte chunks, as discussed in the ABI Function Call Guide
    
    size_t numArgs = args.GetSize();
    size_t index;
    
    for (index = 0; index < numArgs; ++index)
    {
        Value *val = args.GetValueAtIndex(index);
        
        if (!val)
            return false;
        
        switch (val->GetValueType())
        {
        case Value::eValueTypeScalar:
            {
                Scalar &scalar = val->GetScalar();
                switch (scalar.GetType())
                {
                case Scalar::e_void:
                default:
                    return false;
                case Scalar::e_sint: 
                case Scalar::e_uint:
                case Scalar::e_slong:
                case Scalar::e_ulong:
                case Scalar::e_slonglong:
                case Scalar::e_ulonglong:
                    {
                        uint64_t data = scalar.ULongLong();
                        
                        switch (scalar.GetByteSize())
                        {
                        default:
                            return false;
                        case 1:
                            argLayout.push_back((uint32_t)(data & 0xffull));
                            break;
                        case 2:
                            argLayout.push_back((uint32_t)(data & 0xffffull));
                            break;
                        case 4:
                            argLayout.push_back((uint32_t)(data & 0xffffffffull));
                            break;
                        case 8:
                            argLayout.push_back((uint32_t)(data & 0xffffffffull));
                            argLayout.push_back((uint32_t)(data >> 32));
                            break;
                        }
                    }
                    break;
                case Scalar::e_float:
                    {
                        float data = scalar.Float();
                        uint32_t dataRaw = *((uint32_t*)(&data));
                        argLayout.push_back(dataRaw);
                    }
                    break;
                case Scalar::e_double:
                    {
                        double data = scalar.Double();
                        uint32_t *dataRaw = ((uint32_t*)(&data));
                        argLayout.push_back(dataRaw[0]);
                        argLayout.push_back(dataRaw[1]);
                    }
                    break;
                case Scalar::e_long_double:
                    {
                        long double data = scalar.Double();
                        uint32_t *dataRaw = ((uint32_t*)(&data));
                        while ((argLayout.size() * 4) & 0xf)
                            argLayout.push_back(0);
                        argLayout.push_back(dataRaw[0]);
                        argLayout.push_back(dataRaw[1]);
                        argLayout.push_back(dataRaw[2]);
                        argLayout.push_back(dataRaw[3]);
                    }
                    break;
                }
            }
            break;
        case Value::eValueTypeHostAddress:
            switch (val->GetContextType()) 
            {
            default:
                return false;
            case Value::eContextTypeClangType:
                {
                    void *val_type = val->GetClangType();
                    uint32_t cstr_length;
                    
                    if (ClangASTContext::IsCStringType (val_type, cstr_length))
                    {
                        const char *cstr = (const char*)val->GetScalar().ULongLong();
                        cstr_length = strlen(cstr);
                        
                        // Push the string onto the stack immediately.
                        
                        sp -= (cstr_length + 1);
                        
                        if (thread.GetProcess().WriteMemory(sp, cstr, cstr_length + 1, error) != (cstr_length + 1))
                            return false;
                        
                        // Put the address of the string into the argument array.
                        
                        argLayout.push_back((uint32_t)(sp & 0xffffffff));
                    }
                    else
                    {
                        return false;
                    }
                }
                break;
            }
            break;
        case Value::eValueTypeFileAddress:
        case Value::eValueTypeLoadAddress:
        default:
            return false;
        }
    }
    
    // Make room for the arguments on the stack
    
    sp -= 4 * argLayout.size();
    
    // Align the SP
    
    sp &= ~(16ull-1ull); // 16-byte alignment
    
    // Write the arguments on the stack
    
    size_t numChunks = argLayout.size();

    for (index = 0; index < numChunks; ++index)
        if (thread.GetProcess().WriteMemory(sp + (index * 4), &argLayout[index], sizeof(uint32_t), error) != sizeof(uint32_t))
            return false;
    
    // The return address is pushed onto the stack.
    
    sp -= 4;
    uint32_t returnAddressU32 = return_addr;
    if (thread.GetProcess().WriteMemory (sp, &returnAddressU32, sizeof(returnAddressU32), error) != sizeof(returnAddressU32))
        return false;
    
    // %esp is set to the actual stack value.
    
    if (!reg_ctx->WriteRegisterFromUnsigned(sp_reg_num, sp))
        return false;
    
    // %ebp is set to a fake value, in our case 0x0x00000000
    
    if (!reg_ctx->WriteRegisterFromUnsigned(fp_reg_num, 0x00000000))
        return false;
    
    // %eip is set to the address of the called function.
    
    if (!reg_ctx->WriteRegisterFromUnsigned(pc_reg_num, func_addr))
        return false;
    
    return true;    
}

static bool ReadIntegerArgument(Scalar           &scalar,
                                unsigned int     bit_width,
                                bool             is_signed,
                                Process          &process,
                                addr_t           &current_stack_argument)
{
    if (bit_width > 64)
        return false; // Scalar can't hold large integer arguments
    
    uint64_t arg_contents;
    uint32_t read_data;
    Error error;
    
    if (bit_width > 32)
    {
        if (process.ReadMemory(current_stack_argument, &read_data, sizeof(read_data), error) != sizeof(read_data))
            return false;
        
        arg_contents = read_data;
        
        if (process.ReadMemory(current_stack_argument + 4, &read_data, sizeof(read_data), error) != sizeof(read_data))
            return false;
        
        arg_contents |= ((uint64_t)read_data) << 32;
        
        current_stack_argument += 8;
    }
    else {
        if (process.ReadMemory(current_stack_argument, &read_data, sizeof(read_data), error) != sizeof(read_data))
            return false;
        
        arg_contents = read_data;
        
        current_stack_argument += 4;
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
            scalar = (uint8_t)(arg_contents & 0xff);
            break;
        case 16:
            scalar = (uint16_t)(arg_contents & 0xffff);
            break;
        case 32:
            scalar = (uint32_t)(arg_contents & 0xffffffff);
            break;
        case 64:
            scalar = (uint64_t)arg_contents;
            break;
        }
    }
    
    return true;
}

bool
ABIMacOSX_i386::GetArgumentValues (Thread &thread,
                                   ValueList &values) const
{
    unsigned int num_values = values.GetSize();
    unsigned int value_index;
    
    // Extract the Clang AST context from the PC so that we can figure out type
    // sizes
    
    clang::ASTContext *ast_context = thread.CalculateTarget()->GetScratchClangASTContext()->getASTContext();
    
    // Get the pointer to the first stack argument so we have a place to start 
    // when reading data
    
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    
    if (!reg_ctx)
        return false;
    
    addr_t sp = reg_ctx->GetSP(0);
    
    if (!sp)
        return false;
    
    addr_t current_stack_argument = sp + 4; // jump over return address
    
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
                                        thread.GetProcess(), 
                                        current_stack_argument);
                }
                else if (ClangASTContext::IsPointerType (value_type))
                {
                    ReadIntegerArgument(value->GetScalar(),
                                        32,
                                        false,
                                        thread.GetProcess(),
                                        current_stack_argument);
                }
            }
                break;
        }
    }
    
    return true;
}

bool
ABIMacOSX_i386::GetReturnValue (Thread &thread,
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
            
            if (ClangASTContext::IsIntegerType (value_type, is_signed))
            {
                size_t bit_width = ClangASTType::GetClangTypeBitWidth(ast_context, value_type);
                
                unsigned eax_id = reg_ctx->GetRegisterInfoByName("eax", 0)->kinds[eRegisterKindLLDB];
                unsigned edx_id = reg_ctx->GetRegisterInfoByName("edx", 0)->kinds[eRegisterKindLLDB];
                
                switch (bit_width)
                {
                    default:
                    case 128:
                        // Scalar can't hold 128-bit literals, so we don't handle this
                        return false;
                    case 64:
                        uint64_t raw_value;
                        raw_value = thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff;
                        raw_value |= (thread.GetRegisterContext()->ReadRegisterAsUnsigned(edx_id, 0) & 0xffffffff) << 32;
                        if (is_signed)
                            value.GetScalar() = (int64_t)raw_value;
                        else
                            value.GetScalar() = (uint64_t)raw_value;
                        break;
                    case 32:
                        if (is_signed)
                            value.GetScalar() = (int32_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff);
                        else
                            value.GetScalar() = (uint32_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff);
                        break;
                    case 16:
                        if (is_signed)
                            value.GetScalar() = (int16_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffff);
                        else
                            value.GetScalar() = (uint16_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffff);
                        break;
                    case 8:
                        if (is_signed)
                            value.GetScalar() = (int8_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xff);
                        else
                            value.GetScalar() = (uint8_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xff);
                        break;
                }
            }
            else if (ClangASTContext::IsPointerType (value_type))
            {
                unsigned eax_id = reg_ctx->GetRegisterInfoByName("eax", 0)->kinds[eRegisterKindLLDB];
                uint32_t ptr = thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff;
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
ABIMacOSX_i386::CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan)
{
    uint32_t reg_kind = unwind_plan.GetRegisterKind();
    uint32_t sp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t pc_reg_num = LLDB_INVALID_REGNUM;
    
    switch (reg_kind)
    {
        case eRegisterKindDWARF:
            sp_reg_num = dwarf_esp;
            pc_reg_num = dwarf_eip;
            break;

        case eRegisterKindGCC:
            sp_reg_num = gcc_esp;
            pc_reg_num = gcc_eip;
            break;
            
        case eRegisterKindGDB:
            sp_reg_num = gdb_esp;
            pc_reg_num = gdb_eip;
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
    row.SetCFAOffset (4);
    row.SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, -4, false);    
    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName (pluginName);
    return true;
}

bool
ABIMacOSX_i386::CreateDefaultUnwindPlan (UnwindPlan &unwind_plan)
{
    uint32_t reg_kind = unwind_plan.GetRegisterKind();
    uint32_t fp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t sp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t pc_reg_num = LLDB_INVALID_REGNUM;
    
    switch (reg_kind)
    {
        case eRegisterKindDWARF:
            fp_reg_num = dwarf_ebp;
            sp_reg_num = dwarf_esp;
            pc_reg_num = dwarf_eip;
            break;
            
        case eRegisterKindGCC:
            fp_reg_num = gcc_ebp;
            sp_reg_num = gcc_esp;
            pc_reg_num = gcc_eip;
            break;
            
        case eRegisterKindGDB:
            fp_reg_num = gdb_ebp;
            sp_reg_num = gdb_esp;
            pc_reg_num = gdb_eip;
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
    const int32_t ptr_size = 4;

    unwind_plan.SetRegisterKind (eRegisterKindGeneric);
    row.SetCFARegister (fp_reg_num);
    row.SetCFAOffset (2 * ptr_size);
    row.SetOffset (0);
    
    row.SetRegisterLocationToAtCFAPlusOffset(fp_reg_num, ptr_size * -2, true);
    row.SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, ptr_size * -1, true);
    row.SetRegisterLocationToAtCFAPlusOffset(sp_reg_num, ptr_size *  0, true);

    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("i386 default unwind plan");
    return true;
}

bool
ABIMacOSX_i386::RegisterIsVolatile (const RegisterInfo *reg_info)
{
    return RegisterIsCalleeSaved (reg_info);
}

bool
ABIMacOSX_i386::RegisterIsCalleeSaved (const RegisterInfo *reg_info)
{
    if (reg_info)
    {
        // Volatile registers include: ebx, ebp, esi, edi, esp, eip
        const char *name = reg_info->name;
        if (name[0] == 'e')
        {
            switch (name[1])
            {
            case 'b': 
                if (name[2] == 'x' || name[2] == 'p')
                    return name[0] == '\0';
                break;
            case 'd':
                if (name[2] == 'i')
                    return name[0] == '\0';
                break;
            case 'i': 
                if (name[2] == 'p')
                    return name[0] == '\0';
                break;
            case 's':
                if (name[2] == 'i' || name[2] == 'p')
                    return name[0] == '\0';
                break;
            }
        }
    }
    return false;
}

void
ABIMacOSX_i386::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);    
}

void
ABIMacOSX_i386::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ABIMacOSX_i386::GetPluginName()
{
    return pluginName;
}

const char *
ABIMacOSX_i386::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
ABIMacOSX_i386::GetPluginVersion()
{
    return 1;
}

