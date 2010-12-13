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
#include "lldb/Core/Value.h"
#include "lldb/Symbol/ClangASTContext.h"
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
lldb_private::ABI *
ABISysV_x86_64::CreateInstance (const ConstString &triple)
{
    llvm::StringRef tripleStr(triple.GetCString());
    llvm::Triple llvmTriple(tripleStr);

    if (llvmTriple.getArch() != llvm::Triple::x86_64)
        return NULL;

    return new ABISysV_x86_64;
}

bool
ABISysV_x86_64::PrepareTrivialCall (Thread &thread, 
                                    lldb::addr_t sp, 
                                    lldb::addr_t functionAddress, 
                                    lldb::addr_t returnAddress, 
                                    lldb::addr_t arg,
                                    lldb::addr_t *this_arg,
                                    lldb::addr_t *cmd_arg) const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
        log->Printf("ABISysV_x86_64::PrepareTrivialCall\n(\n  thread = %p\n  sp = 0x%llx\n  functionAddress = 0x%llx\n  returnAddress = 0x%llx\n  arg = 0x%llx\n  this_arg = %p(0x%llx)\n  cmd_arg = %p(0x%llx)\n)",
                    (void*)&thread,
                    (uint64_t)sp,
                    (uint64_t)functionAddress,
                    (uint64_t)returnAddress,
                    (void*)arg,
                    this_arg,
                    this_arg ? (uint64_t)*this_arg : (uint64_t)0,
                    cmd_arg,
                    cmd_arg ? (uint64_t)*cmd_arg : (uint64_t)0);
    
    RegisterContext *reg_ctx = thread.GetRegisterContext();
    if (!reg_ctx)
        return false;

    uint32_t rdiID = reg_ctx->GetRegisterInfoByName("rdi", 0)->kinds[eRegisterKindLLDB];
#define CHAIN_RBP
    
#ifndef CHAIN_RBP
    uint32_t rbpID = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP);
#endif
    
    uint32_t ripID = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    uint32_t rspID = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);

    // The argument is in %rdi, and not on the stack.
    
    if (cmd_arg)
    {
        if (log)
            log->PutCString("The trivial call has a self and a _cmd pointer");
        
        uint32_t rsiID = reg_ctx->GetRegisterInfoByName("rsi", 0)->kinds[eRegisterKindLLDB];
        uint32_t rdxID = reg_ctx->GetRegisterInfoByName("rdx", 0)->kinds[eRegisterKindLLDB];
        
        if (log)
            log->Printf("About to write 'self' (0x%llx) into RDI", (uint64_t)*this_arg);
        
        if (!reg_ctx->WriteRegisterFromUnsigned(rdiID, *this_arg))
            return false;
        
        if (log)
            log->Printf("About to write '_cmd' (0x%llx) into RSI", (uint64_t)*cmd_arg);
        
        if (!reg_ctx->WriteRegisterFromUnsigned(rsiID, *this_arg))
            return false;
        
        if (log)
            log->Printf("About to write the argument (0x%llx) into RDX", (uint64_t)arg);
        
        if (!reg_ctx->WriteRegisterFromUnsigned(rdxID, arg))
            return false;    }
    if (this_arg)
    {
        if (log)
            log->PutCString("The trivial call has a this pointer");
        
        uint32_t rsiID = reg_ctx->GetRegisterInfoByName("rsi", 0)->kinds[eRegisterKindLLDB];
        
        if (log)
            log->Printf("About to write 'this' (0x%llx) into RDI", (uint64_t)*this_arg);
        
        if (!reg_ctx->WriteRegisterFromUnsigned(rdiID, *this_arg))
            return false;
        
        if (log)
            log->Printf("About to write the argument (0x%llx) into RSI", (uint64_t)arg);
        
        if (!reg_ctx->WriteRegisterFromUnsigned(rsiID, arg))
            return false;
    }
    else
    {
        if (log)
            log->PutCString("The trivial call does not have a this pointer");
        
        if (log)
            log->Printf("About to write the argument (0x%llx) into RDI", (uint64_t)arg);
        
        if (!reg_ctx->WriteRegisterFromUnsigned(rdiID, arg))
            return false;
    }

    // First, align the SP
    
    if (log)
        log->Printf("16-byte aligning SP: 0x%llx to 0x%llx", (uint64_t)sp, (uint64_t)(sp & ~0xfull));

    sp &= ~(0xfull); // 16-byte alignment

    // The return address is pushed onto the stack.

    sp -= 8;
    uint64_t returnAddressU64 = returnAddress;
    Error error;
    
    if (log)
        log->Printf("Pushing the return address onto the stack: new SP 0x%llx, return address 0x%llx", (uint64_t)sp, (uint64_t)returnAddressU64);
    
    if (thread.GetProcess().WriteMemory (sp, &returnAddressU64, sizeof(returnAddressU64), error) != sizeof(returnAddressU64))
        return false;

    // %rsp is set to the actual stack value.

    if (log)
        log->Printf("Writing SP (0x%llx) down", (uint64_t)sp);
    
    if (!reg_ctx->WriteRegisterFromUnsigned(rspID, sp))
        return false;

#ifndef CHAIN_RBP
    // %rbp is set to a fake value, in our case 0x0000000000000000.

    if (!reg_ctx->WriteRegisterFromUnsigned(rbpID, 0x000000000000000))
        return false;
#endif

    // %rip is set to the address of the called function.
    
    if (log)
        log->Printf("Writing new IP (0x%llx) down", (uint64_t)functionAddress);

    if (!reg_ctx->WriteRegisterFromUnsigned(ripID, functionAddress))
        return false;

    return true;
}

bool
ABISysV_x86_64::PrepareNormalCall (Thread &thread,
                                   lldb::addr_t sp,
                                   lldb::addr_t functionAddress,
                                   lldb::addr_t returnAddress,
                                   ValueList &args) const
{
    return false;    
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
        DataExtractor arg_data_extractor(arg_data, sizeof(arg_contents), thread.GetProcess().GetByteOrder(), thread.GetProcess().GetAddressByteSize());
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
ABISysV_x86_64::GetArgumentValues (Thread &thread,
                                   ValueList &values) const
{
    unsigned int num_values = values.GetSize();
    unsigned int value_index;
    
    // For now, assume that the types in the AST values come from the Target's 
    // scratch AST.    
    
    clang::ASTContext *ast_context = thread.CalculateTarget()->GetScratchClangASTContext()->getASTContext();
    
    // Extract the register context so we can read arguments from registers
    
    RegisterContext *reg_ctx = thread.GetRegisterContext();
    
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
            
            RegisterContext *reg_ctx = thread.GetRegisterContext();
            
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

void
ABISysV_x86_64::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
ABISysV_x86_64::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
ABISysV_x86_64::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}
