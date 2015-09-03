//===-- CXXFormatterFunctions.cpp---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/CXXFormatterFunctions.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSummary.h"

#include "llvm/Support/ConvertUTF.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "lldb/Utility/ProcessStructReader.h"

#include <algorithm>

#if __ANDROID_NDK__
#include <sys/types.h>
#endif

#include "lldb/Host/Time.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

StackFrame*
lldb_private::formatters::GetViableFrame (ExecutionContext exe_ctx)
{
    StackFrame* frame = exe_ctx.GetFramePtr();
    if (frame)
        return frame;
    
    Process* process = exe_ctx.GetProcessPtr();
    if (!process)
        return nullptr;
    
    ThreadSP thread_sp(process->GetThreadList().GetSelectedThread());
    if (thread_sp)
        return thread_sp->GetSelectedFrame().get();
    return nullptr;
}

bool
lldb_private::formatters::ExtractValueFromObjCExpression (ValueObject &valobj,
                                                          const char* target_type,
                                                          const char* selector,
                                                          uint64_t &value)
{
    if (!target_type || !*target_type)
        return false;
    if (!selector || !*selector)
        return false;
    StreamString expr;
    expr.Printf("(%s)[(id)0x%" PRIx64 " %s]",target_type,valobj.GetPointerValue(),selector);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = GetViableFrame(exe_ctx);
    if (!target || !stack_frame)
        return false;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false);
    options.SetUnwindOnError(true);
    options.SetKeepInMemory(true);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               result_sp,
                               options);
    if (!result_sp)
        return false;
    value = result_sp->GetValueAsUnsigned(0);
    return true;
}

bool
lldb_private::formatters::ExtractSummaryFromObjCExpression (ValueObject &valobj,
                                                            const char* target_type,
                                                            const char* selector,
                                                            Stream &stream)
{
    if (!target_type || !*target_type)
        return false;
    if (!selector || !*selector)
        return false;
    StreamString expr;
    expr.Printf("(%s)[(id)0x%" PRIx64 " %s]",target_type,valobj.GetPointerValue(),selector);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = GetViableFrame(exe_ctx);
    if (!target || !stack_frame)
        return false;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false);
    options.SetUnwindOnError(true);
    options.SetKeepInMemory(true);
    options.SetUseDynamic(lldb::eDynamicCanRunTarget);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               result_sp,
                               options);
    if (!result_sp)
        return false;
    stream.Printf("%s",result_sp->GetSummaryAsCString());
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::CallSelectorOnObject (ValueObject &valobj,
                                                const char* return_type,
                                                const char* selector,
                                                uint64_t index)
{
    lldb::ValueObjectSP valobj_sp;
    if (!return_type || !*return_type)
        return valobj_sp;
    if (!selector || !*selector)
        return valobj_sp;
    StreamString expr_path_stream;
    valobj.GetExpressionPath(expr_path_stream, false);
    StreamString expr;
    expr.Printf("(%s)[%s %s:%" PRId64 "]",return_type,expr_path_stream.GetData(),selector,index);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = GetViableFrame(exe_ctx);
    if (!target || !stack_frame)
        return valobj_sp;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false);
    options.SetUnwindOnError(true);
    options.SetKeepInMemory(true);
    options.SetUseDynamic(lldb::eDynamicCanRunTarget);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               valobj_sp,
                               options);
    return valobj_sp;
}

lldb::ValueObjectSP
lldb_private::formatters::CallSelectorOnObject (ValueObject &valobj,
                                                const char* return_type,
                                                const char* selector,
                                                const char* key)
{
    lldb::ValueObjectSP valobj_sp;
    if (!return_type || !*return_type)
        return valobj_sp;
    if (!selector || !*selector)
        return valobj_sp;
    if (!key || !*key)
        return valobj_sp;
    StreamString expr_path_stream;
    valobj.GetExpressionPath(expr_path_stream, false);
    StreamString expr;
    expr.Printf("(%s)[%s %s:%s]",return_type,expr_path_stream.GetData(),selector,key);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = GetViableFrame(exe_ctx);
    if (!target || !stack_frame)
        return valobj_sp;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false);
    options.SetUnwindOnError(true);
    options.SetKeepInMemory(true);
    options.SetUseDynamic(lldb::eDynamicCanRunTarget);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               valobj_sp,
                               options);
    return valobj_sp;
}

bool
lldb_private::formatters::FunctionPointerSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    std::string destination;
    StreamString sstr;
    AddressType func_ptr_address_type = eAddressTypeInvalid;
    addr_t func_ptr_address = valobj.GetPointerValue (&func_ptr_address_type);
    if (func_ptr_address != 0 && func_ptr_address != LLDB_INVALID_ADDRESS)
    {
        switch (func_ptr_address_type)
        {
            case eAddressTypeInvalid:
            case eAddressTypeFile:
            case eAddressTypeHost:
                break;
                
            case eAddressTypeLoad:
            {
                ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
                
                Address so_addr;
                Target *target = exe_ctx.GetTargetPtr();
                if (target && target->GetSectionLoadList().IsEmpty() == false)
                {
                    if (target->GetSectionLoadList().ResolveLoadAddress(func_ptr_address, so_addr))
                    {
                        so_addr.Dump (&sstr,
                                      exe_ctx.GetBestExecutionContextScope(),
                                      Address::DumpStyleResolvedDescription,
                                      Address::DumpStyleSectionNameOffset);
                    }
                }
            }
                break;
        }
    }
    if (sstr.GetSize() > 0)
    {
        stream.Printf("(%s)", sstr.GetData());
        return true;
    }
    else
        return false;
}

bool
lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    stream.Printf("%s",valobj.GetObjectDescription());
    return true;
}

size_t
lldb_private::formatters::ExtractIndexFromString (const char* item_name)
{
    if (!item_name || !*item_name)
        return UINT32_MAX;
    if (*item_name != '[')
        return UINT32_MAX;
    item_name++;
    char* endptr = NULL;
    unsigned long int idx = ::strtoul(item_name, &endptr, 0);
    if (idx == 0 && endptr == item_name)
        return UINT32_MAX;
    if (idx == ULONG_MAX)
        return UINT32_MAX;
    return idx;
}
