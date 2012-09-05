//===-- CXXFormatterFunctions.cpp---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/CXXFormatterFunctions.h"

// needed to get ConvertUTF16/32ToUTF8
#define CLANG_NEEDS_THESE_ONE_DAY
#include "clang/Basic/ConvertUTF.h"

#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

bool
lldb_private::formatters::CodeRunning_Fetcher (ValueObject &valobj,
                                               const char* target_type,
                                               const char* selector,
                                               uint64_t &value)
{
    if (!target_type || !*target_type)
        return false;
    if (!selector || !*selector)
        return false;
    StreamString expr_path_stream;
    valobj.GetExpressionPath(expr_path_stream, false);
    StreamString expr;
    expr.Printf("(%s)[%s %s]",target_type,expr_path_stream.GetData(),selector);
    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
    lldb::ValueObjectSP result_sp;
    Target* target = exe_ctx.GetTargetPtr();
    StackFrame* stack_frame = exe_ctx.GetFramePtr();
    if (!target || !stack_frame)
        return false;
    
    Target::EvaluateExpressionOptions options;
    options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true)
    .SetUseDynamic(lldb::eDynamicCanRunTarget);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               result_sp,
                               options);
    if (!result_sp)
        return false;
    value = result_sp->GetValueAsUnsigned(0);
    return true;
}

template<bool name_entries>
bool
lldb_private::formatters::NSDictionary_SummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;

    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    bool is_64bit = (ptr_size == 8);
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    uint64_t value = 0;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    if (!strcmp(class_name,"__NSDictionaryI"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
    }
    else if (!strcmp(class_name,"__NSDictionaryM"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
    }
    else if (!strcmp(class_name,"__NSCFDictionary"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + (is_64bit ? 20 : 12), ptr_size, 0, error);
        if (error.Fail())
            return false;
        if (is_64bit)
            value &= ~0x0f1f000000000000UL;
            }
    else
    {
        if (!CodeRunning_Fetcher(valobj, "int", "count", value))
            return false;
    }
    
    stream.Printf("%s%llu %s%s",
                  (name_entries ? "@\"" : ""),
                  value,
                  (name_entries ? (value == 1 ? "entry" : "entries") : (value == 1 ? "key/value pair" : "key/value pairs")),
                  (name_entries ? "\"" : ""));
    return true;
}

bool
lldb_private::formatters::NSArray_SummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    uint64_t value = 0;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    if (!strcmp(class_name,"__NSArrayI"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
    }
    else if (!strcmp(class_name,"__NSArrayM"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
    }
    else if (!strcmp(class_name,"__NSCFArray"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + 2 * ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
    }
    else
    {
        if (!CodeRunning_Fetcher(valobj, "int", "count", value))
            return false;
    }
    
    stream.Printf("@\"%llu object%s\"",
                  value,
                  value == 1 ? "" : "s");
    return true;
}

template<bool needs_at>
bool
lldb_private::formatters::NSData_SummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    bool is_64bit = (process_sp->GetAddressByteSize() == 8);
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    uint64_t value = 0;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    if (!strcmp(class_name,"NSConcreteData") ||
        !strcmp(class_name,"NSConcreteMutableData") ||
        !strcmp(class_name,"__NSCFData"))
    {
        uint32_t offset = (is_64bit ? 16 : 8);
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + offset, is_64bit ? 8 : 4, 0, error);
        if (error.Fail())
            return false;
    }
    else
    {
        if (!CodeRunning_Fetcher(valobj, "int", "length", value))
            return false;
    }
    
    stream.Printf("%s%llu byte%s%s",
                  (needs_at ? "@\"" : ""),
                  value,
                  (value > 1 ? "s" : ""),
                  (needs_at ? "\"" : ""));
    
    return true;
}

bool
lldb_private::formatters::NSNumber_SummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!strcmp(class_name,"NSNumber") || !strcmp(class_name,"__NSCFNumber"))
    {
        if (descriptor->IsTagged())
        {
            // we have a call to get info and value bits in the tagged descriptor. but we prefer not to cast and replicate them
            int64_t value = (valobj_addr & ~0x0000000000000000FFL) >> 8;
            uint64_t i_bits = (valobj_addr & 0xF0) >> 4;
            
            switch (i_bits)
            {
                case 0:
                    stream.Printf("(char)%hhd",(char)value);
                    break;
                case 4:
                    stream.Printf("(short)%hd",(short)value);
                    break;
                case 8:
                    stream.Printf("(int)%d",(int)value);
                    break;
                case 12:
                    stream.Printf("(long)%lld",value);
                    break;
                default:
                    stream.Printf("absurd value:(info=%llu, value=%llu",i_bits,value);
                    break;
            }
            return true;
        }
        else
        {
            Error error;
            uint8_t data_type = (process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, 1, 0, error) & 0x1F);
            uint64_t data_location = valobj_addr + 2*ptr_size;
            uint64_t value = 0;
            if (error.Fail())
                return false;
            switch (data_type)
            {
                case 1: // 0B00001
                    value = process_sp->ReadUnsignedIntegerFromMemory(data_location, 1, 0, error);
                    if (error.Fail())
                        return false;
                    stream.Printf("(char)%hhd",(char)value);
                    break;
                case 2: // 0B0010
                    value = process_sp->ReadUnsignedIntegerFromMemory(data_location, 2, 0, error);
                    if (error.Fail())
                        return false;
                    stream.Printf("(short)%hd",(short)value);
                    break;
                case 3: // 0B0011
                    value = process_sp->ReadUnsignedIntegerFromMemory(data_location, 4, 0, error);
                    if (error.Fail())
                        return false;
                    stream.Printf("(int)%d",(int)value);
                    break;
                case 17: // 0B10001
                    data_location += 8;
                case 4: // 0B0100
                    value = process_sp->ReadUnsignedIntegerFromMemory(data_location, 8, 0, error);
                    if (error.Fail())
                        return false;
                    stream.Printf("(long)%lld",value);
                    break;
                case 5: // 0B0101
                {
                    uint32_t flt_as_int = process_sp->ReadUnsignedIntegerFromMemory(data_location, 4, 0, error);
                    if (error.Fail())
                        return false;
                    float flt_value = *((float*)&flt_as_int);
                    stream.Printf("(float)%f",flt_value);
                    break;
                }
                case 6: // 0B0110
                {
                    uint64_t dbl_as_lng = process_sp->ReadUnsignedIntegerFromMemory(data_location, 8, 0, error);
                    if (error.Fail())
                        return false;
                    double dbl_value = *((double*)&dbl_as_lng);
                    stream.Printf("(double)%g",dbl_value);
                    break;
                }
                default:
                    stream.Printf("absurd: dt=%d",data_type);
                    break;
            }
            return true;
        }
    }
    else
    {
        // similar to CodeRunning_Fetcher but uses summary instead of value
        StreamString expr_path_stream;
        valobj.GetExpressionPath(expr_path_stream, false);
        StreamString expr;
        expr.Printf("(NSString*)[%s stringValue]",expr_path_stream.GetData());
        ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
        lldb::ValueObjectSP result_sp;
        Target* target = exe_ctx.GetTargetPtr();
        StackFrame* stack_frame = exe_ctx.GetFramePtr();
        if (!target || !stack_frame)
            return false;
        
        Target::EvaluateExpressionOptions options;
        options.SetCoerceToId(false)
        .SetUnwindOnError(true)
        .SetKeepInMemory(true)
        .SetUseDynamic(lldb::eDynamicCanRunTarget);
        
        target->EvaluateExpression(expr.GetData(),
                                   stack_frame,
                                   result_sp,
                                   options);
        if (!result_sp)
            return false;
        stream.Printf("%s",result_sp->GetSummaryAsCString());
        return true;
    }
}

bool
lldb_private::formatters::NSString_SummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    uint64_t info_bits_location = valobj_addr + ptr_size;
    if (process_sp->GetByteOrder() != lldb::eByteOrderLittle)
        info_bits_location += 3;
        
        Error error;
    
    uint8_t info_bits = process_sp->ReadUnsignedIntegerFromMemory(info_bits_location, 1, 0, error);
    if (error.Fail())
        return false;
    
    bool is_mutable = (info_bits & 1) == 1;
    bool is_inline = (info_bits & 0x60) == 0;
    bool has_explicit_length = (info_bits & (1 | 4)) != 4;
    bool is_unicode = (info_bits & 0x10) == 0x10;
    bool is_special = strcmp(class_name,"NSPathStore2") == 0;
    
    if (strcmp(class_name,"NSString") &&
        strcmp(class_name,"CFStringRef") &&
        strcmp(class_name,"CFMutableStringRef") &&
        strcmp(class_name,"__NSCFConstantString") &&
        strcmp(class_name,"__NSCFString") &&
        strcmp(class_name,"NSCFConstantString") &&
        strcmp(class_name,"NSCFString") &&
        strcmp(class_name,"NSPathStore2"))
    {
        // probably not one of us - bail out
        return false;
    }
    
    if (is_mutable)
    {
        uint64_t location = 2 * ptr_size + valobj_addr;
        location = process_sp->ReadPointerFromMemory(location, error);
        if (error.Fail())
            return false;
        if (has_explicit_length and is_unicode)
        {
            lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024,0));
            size_t data_read = process_sp->ReadMemoryFromInferior(location, (char*)buffer_sp->GetBytes(), 1024, error);
            if (error.Fail())
            {
                stream.Printf("erorr reading pte");
                return true;
            }
            else
                stream.Printf("@\"");
                if (data_read)
                {
                    UTF16 *data_ptr = (UTF16*)buffer_sp->GetBytes();
                    UTF16 *data_end_ptr = data_ptr + 256;
                    
                    while (data_ptr < data_end_ptr)
                    {
                        if (!*data_ptr)
                        {
                            data_end_ptr = data_ptr;
                            break;
                        }
                        data_ptr++;
                    }
                    
                    *data_ptr = 0;
                    data_ptr = (UTF16*)buffer_sp->GetBytes();
                    
                    lldb::DataBufferSP utf8_data_buffer_sp(new DataBufferHeap(1024,0));
                    UTF8* utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes();
                    UTF8* utf8_data_end_ptr = utf8_data_ptr + 1024;
                    
                    ConvertUTF16toUTF8	(	(const UTF16**)&data_ptr,
                                         data_end_ptr,
                                         &utf8_data_ptr,
                                         utf8_data_end_ptr,
                                         lenientConversion);
                    utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes();
                    for (;utf8_data_ptr != utf8_data_end_ptr; utf8_data_ptr++)
                    {
                        if (!*utf8_data_ptr)
                            break;
                        stream.Printf("%c",*utf8_data_ptr);
                    }
                    stream.Printf("\"");
                    return true;
                }
            stream.Printf("\"");
            return true;
        }
        else
        {
            location++;
            lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024,0));
            size_t data_read = process_sp->ReadCStringFromMemory(location, (char*)buffer_sp->GetBytes(), 1024, error);
            if (error.Fail())
                return false;
            if (data_read)
                stream.Printf("@\"%s\"",(char*)buffer_sp->GetBytes());
                return true;
        }
    }
    else if (is_inline && has_explicit_length && !is_unicode && !is_special && !is_mutable)
    {
        uint64_t location = 3 * ptr_size + valobj_addr;
        lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024,0));
        size_t data_read = process_sp->ReadCStringFromMemory(location, (char*)buffer_sp->GetBytes(), 1024, error);
        if (error.Fail())
            return false;
        if (data_read)
            stream.Printf("@\"%s\"",(char*)buffer_sp->GetBytes());
            return true;
    }
    else if (is_unicode)
    {
        uint64_t location = valobj_addr + ptr_size + 4 + (ptr_size == 8 ? 4 : 0);
        if (is_inline)
        {
            if (!has_explicit_length)
            {
                stream.Printf("found new combo");
                return true;
            }
            else
                location += ptr_size;
                }
        else
        {
            location = process_sp->ReadPointerFromMemory(location, error);
            if (error.Fail())
                return false;
        }
        lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024,0));
        size_t data_read = process_sp->ReadMemoryFromInferior(location, (char*)buffer_sp->GetBytes(), 1024, error);
        if (error.Fail())
        {
            stream.Printf("erorr reading pte");
            return true;
        }
        else
            stream.Printf("@\"");
            if (data_read)
            {
                UTF16 *data_ptr = (UTF16*)buffer_sp->GetBytes();
                UTF16 *data_end_ptr = data_ptr + 256;
                
                while (data_ptr < data_end_ptr)
                {
                    if (!*data_ptr)
                    {
                        data_end_ptr = data_ptr;
                        break;
                    }
                    data_ptr++;
                }
                
                *data_ptr = 0;
                data_ptr = (UTF16*)buffer_sp->GetBytes();
                
                lldb::DataBufferSP utf8_data_buffer_sp(new DataBufferHeap(1024,0));
                UTF8* utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes();
                UTF8* utf8_data_end_ptr = utf8_data_ptr + 1024;
                
                ConvertUTF16toUTF8	(	(const UTF16**)&data_ptr,
                                     data_end_ptr,
                                     &utf8_data_ptr,
                                     utf8_data_end_ptr,
                                     lenientConversion);
                utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes();
                for (;utf8_data_ptr != utf8_data_end_ptr; utf8_data_ptr++)
                {
                    if (!*utf8_data_ptr)
                        break;
                    stream.Printf("%c",*utf8_data_ptr);
                }
                stream.Printf("\"");
                return true;
            }
        stream.Printf("\"");
        return true;
    }
    else if (is_special)
    {
        uint64_t location = valobj_addr + (ptr_size == 8 ? 12 : 8);
        lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024,0));
        size_t data_read = process_sp->ReadMemoryFromInferior(location, (char*)buffer_sp->GetBytes(), 1024, error);
        if (error.Fail())
        {
            stream.Printf("erorr reading pte");
            return true;
        }
        else
            stream.Printf("@\"");
            if (data_read)
            {
                UTF16 *data_ptr = (UTF16*)buffer_sp->GetBytes();
                UTF16 *data_end_ptr = data_ptr + 256;
                
                while (data_ptr < data_end_ptr)
                {
                    if (!*data_ptr)
                    {
                        data_end_ptr = data_ptr;
                        break;
                    }
                    data_ptr++;
                }
                
                *data_ptr = 0;
                data_ptr = (UTF16*)buffer_sp->GetBytes();
                
                lldb::DataBufferSP utf8_data_buffer_sp(new DataBufferHeap(1024,0));
                UTF8* utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes();
                UTF8* utf8_data_end_ptr = utf8_data_ptr + 1024;
                
                ConvertUTF16toUTF8	(	(const UTF16**)&data_ptr,
                                     data_end_ptr,
                                     &utf8_data_ptr,
                                     utf8_data_end_ptr,
                                     lenientConversion);
                utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes();
                for (;utf8_data_ptr != utf8_data_end_ptr; utf8_data_ptr++)
                {
                    if (!*utf8_data_ptr)
                        break;
                    stream.Printf("%c",*utf8_data_ptr);
                }
                stream.Printf("\"");
                return true;
            }
        stream.Printf("\"");
        return true;
    }
    else if (is_inline)
    {
        uint64_t location = valobj_addr + ptr_size + 4 + (ptr_size == 8 ? 4 : 0);
        if (!has_explicit_length)
            location++;
        lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024,0));
        size_t data_read = process_sp->ReadCStringFromMemory(location, (char*)buffer_sp->GetBytes(), 1024, error);
        if (error.Fail())
            return false;
        if (data_read)
            stream.Printf("@\"%s\"",(char*)buffer_sp->GetBytes());
            return true;
    }
    else
    {
        uint64_t location = valobj_addr + ptr_size + 4 + (ptr_size == 8 ? 4 : 0);
        location = process_sp->ReadPointerFromMemory(location, error);
        if (error.Fail())
            return false;
        lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024,0));
        size_t data_read = process_sp->ReadCStringFromMemory(location, (char*)buffer_sp->GetBytes(), 1024, error);
        if (error.Fail())
            return false;
        if (data_read)
            stream.Printf("@\"%s\"",(char*)buffer_sp->GetBytes());
            return true;
    }
    
    stream.Printf("class name = %s",class_name);
    return true;
    
}

template bool
lldb_private::formatters::NSDictionary_SummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::NSDictionary_SummaryProvider<false> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::NSData_SummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::NSData_SummaryProvider<false> (ValueObject&, Stream&) ;
