//===-- CXXFormatterFunctions.cpp---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/DataFormatters/CXXFormatterFunctions.h"

#include "llvm/Support/ConvertUTF.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

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
    StackFrame* stack_frame = exe_ctx.GetFramePtr();
    if (!target || !stack_frame)
        return false;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true);
    
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
    StackFrame* stack_frame = exe_ctx.GetFramePtr();
    if (!target || !stack_frame)
        return false;
    
    EvaluateExpressionOptions options;
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
    StackFrame* stack_frame = exe_ctx.GetFramePtr();
    if (!target || !stack_frame)
        return valobj_sp;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true)
    .SetUseDynamic(lldb::eDynamicCanRunTarget);
    
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
    StackFrame* stack_frame = exe_ctx.GetFramePtr();
    if (!target || !stack_frame)
        return valobj_sp;
    
    EvaluateExpressionOptions options;
    options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true)
    .SetUseDynamic(lldb::eDynamicCanRunTarget);
    
    target->EvaluateExpression(expr.GetData(),
                               stack_frame,
                               valobj_sp,
                               options);
    return valobj_sp;
}

// use this call if you already have an LLDB-side buffer for the data
template<typename SourceDataType>
static bool
DumpUTFBufferToStream (ConversionResult (*ConvertFunction) (const SourceDataType**,
                                                            const SourceDataType*,
                                                            UTF8**,
                                                            UTF8*,
                                                            ConversionFlags),
                       DataExtractor& data,
                       Stream& stream,
                       char prefix_token = '@',
                       char quote = '"',
                       int sourceSize = 0)
{
    if (prefix_token != 0)
        stream.Printf("%c",prefix_token);
    if (quote != 0)
        stream.Printf("%c",quote);
    if (data.GetByteSize() && data.GetDataStart() && data.GetDataEnd())
    {
        const int bufferSPSize = data.GetByteSize();
        if (sourceSize == 0)
        {
            const int origin_encoding = 8*sizeof(SourceDataType);
            sourceSize = bufferSPSize/(origin_encoding / 4);
        }
        
        SourceDataType *data_ptr = (SourceDataType*)data.GetDataStart();
        SourceDataType *data_end_ptr = data_ptr + sourceSize;
        
        while (data_ptr < data_end_ptr)
        {
            if (!*data_ptr)
            {
                data_end_ptr = data_ptr;
                break;
            }
            data_ptr++;
        }
        
        data_ptr = (SourceDataType*)data.GetDataStart();
        
        lldb::DataBufferSP utf8_data_buffer_sp;
        UTF8* utf8_data_ptr = nullptr;
        UTF8* utf8_data_end_ptr = nullptr;
        
        if (ConvertFunction)
        {
            utf8_data_buffer_sp.reset(new DataBufferHeap(bufferSPSize,0));
            utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes();
            utf8_data_end_ptr = utf8_data_ptr + bufferSPSize;
            ConvertFunction ( (const SourceDataType**)&data_ptr, data_end_ptr, &utf8_data_ptr, utf8_data_end_ptr, lenientConversion );
            utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes(); // needed because the ConvertFunction will change the value of the data_ptr
        }
        else
        {
            // just copy the pointers - the cast is necessary to make the compiler happy
            // but this should only happen if we are reading UTF8 data
            utf8_data_ptr = (UTF8*)data_ptr;
            utf8_data_end_ptr = (UTF8*)data_end_ptr;
        }
        
        // since we tend to accept partial data (and even partially malformed data)
        // we might end up with no NULL terminator before the end_ptr
        // hence we need to take a slower route and ensure we stay within boundaries
        for (;utf8_data_ptr != utf8_data_end_ptr; utf8_data_ptr++)
        {
            if (!*utf8_data_ptr)
                break;
            stream.Printf("%c",*utf8_data_ptr);
        }
    }
    if (quote != 0)
        stream.Printf("%c",quote);
    return true;
}

template<typename SourceDataType>
static bool
ReadUTFBufferAndDumpToStream (ConversionResult (*ConvertFunction) (const SourceDataType**,
                                                                   const SourceDataType*,
                                                                   UTF8**,
                                                                   UTF8*,
                                                                   ConversionFlags),
                              uint64_t location,
                              const ProcessSP& process_sp,
                              Stream& stream,
                              char prefix_token = '@',
                              char quote = '"',
                              int sourceSize = 0)
{
    if (location == 0 || location == LLDB_INVALID_ADDRESS)
        return false;
    if (!process_sp)
        return false;

    const int origin_encoding = 8*sizeof(SourceDataType);
    if (origin_encoding != 8 && origin_encoding != 16 && origin_encoding != 32)
        return false;
    // if not UTF8, I need a conversion function to return proper UTF8
    if (origin_encoding != 8 && !ConvertFunction)
        return false;

    if (sourceSize == 0)
        sourceSize = process_sp->GetTarget().GetMaximumSizeOfStringSummary();
    const int bufferSPSize = sourceSize * (origin_encoding >> 2);

    Error error;
    lldb::DataBufferSP buffer_sp(new DataBufferHeap(bufferSPSize,0));
    
    if (!buffer_sp->GetBytes())
        return false;
    
    size_t data_read = process_sp->ReadMemoryFromInferior(location, (char*)buffer_sp->GetBytes(), bufferSPSize, error);
    if (error.Fail() || data_read == 0)
    {
        stream.Printf("unable to read data");
        return true;
    }
    
    DataExtractor data(buffer_sp, process_sp->GetByteOrder(), process_sp->GetAddressByteSize());
    
    return DumpUTFBufferToStream(ConvertFunction, data, stream, prefix_token, quote, sourceSize);
}

bool
lldb_private::formatters::Char16StringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    if (!ReadUTFBufferAndDumpToStream<UTF16>(ConvertUTF16toUTF8,valobj_addr,
                                                                 process_sp,
                                                                 stream,
                                                                 'u'))
    {
        stream.Printf("Summary Unavailable");
        return true;
    }

    return true;
}

bool
lldb_private::formatters::Char32StringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    if (!ReadUTFBufferAndDumpToStream<UTF32>(ConvertUTF32toUTF8,valobj_addr,
                                                                 process_sp,
                                                                 stream,
                                                                 'U'))
    {
        stream.Printf("Summary Unavailable");
        return true;
    }
    
    return true;
}

bool
lldb_private::formatters::WCharStringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;

    lldb::addr_t data_addr = 0;
    
    if (valobj.IsPointerType())
        data_addr = valobj.GetValueAsUnsigned(0);
    else if (valobj.IsArrayType())
        data_addr = valobj.GetAddressOf();

    if (data_addr == 0 || data_addr == LLDB_INVALID_ADDRESS)
        return false;

    clang::ASTContext* ast = valobj.GetClangAST();

    if (!ast)
        return false;

    uint32_t wchar_size = ClangASTType::GetClangTypeBitWidth(ast, ClangASTType::GetBasicType(ast, lldb::eBasicTypeWChar).GetOpaqueQualType());

    switch (wchar_size)
    {
        case 8:
            // utf 8
            return ReadUTFBufferAndDumpToStream<UTF8>(nullptr, data_addr,
                                                               process_sp,
                                                               stream,
                                                               'L');
        case 16:
            // utf 16
            return ReadUTFBufferAndDumpToStream<UTF16>(ConvertUTF16toUTF8, data_addr,
                                                                           process_sp,
                                                                           stream,
                                                                           'L');
        case 32:
            // utf 32
            return ReadUTFBufferAndDumpToStream<UTF32>(ConvertUTF32toUTF8, data_addr,
                                                                           process_sp,
                                                                           stream,
                                                                           'L');
        default:
            stream.Printf("size for wchar_t is not valid");
            return true;
    }
    return true;
}

bool
lldb_private::formatters::Char16SummaryProvider (ValueObject& valobj, Stream& stream)
{
    DataExtractor data;
    valobj.GetData(data);
    
    std::string value;
    valobj.GetValueAsCString(lldb::eFormatUnicode16, value);
    if (!value.empty())
        stream.Printf("%s ", value.c_str());

    return DumpUTFBufferToStream<UTF16>(ConvertUTF16toUTF8,data,stream, 'u','\'',1);
}

bool
lldb_private::formatters::Char32SummaryProvider (ValueObject& valobj, Stream& stream)
{
    DataExtractor data;
    valobj.GetData(data);
    
    std::string value;
    valobj.GetValueAsCString(lldb::eFormatUnicode32, value);
    if (!value.empty())
        stream.Printf("%s ", value.c_str());
    
    return DumpUTFBufferToStream<UTF32>(ConvertUTF32toUTF8,data,stream, 'U','\'',1);
}

bool
lldb_private::formatters::WCharSummaryProvider (ValueObject& valobj, Stream& stream)
{
    DataExtractor data;
    valobj.GetData(data);
    
    clang::ASTContext* ast = valobj.GetClangAST();
    
    if (!ast)
        return false;
    
    std::string value;
    
    uint32_t wchar_size = ClangASTType::GetClangTypeBitWidth(ast, ClangASTType::GetBasicType(ast, lldb::eBasicTypeWChar).GetOpaqueQualType());
    
    switch (wchar_size)
    {
        case 8:
            // utf 8
            valobj.GetValueAsCString(lldb::eFormatChar, value);
            if (!value.empty())
                stream.Printf("%s ", value.c_str());
            return DumpUTFBufferToStream<UTF8>(nullptr,
                                               data,
                                               stream,
                                               'L',
                                               '\'',
                                               1);
        case 16:
            // utf 16
            valobj.GetValueAsCString(lldb::eFormatUnicode16, value);
            if (!value.empty())
                stream.Printf("%s ", value.c_str());
            return DumpUTFBufferToStream<UTF16>(ConvertUTF16toUTF8,
                                                data,
                                                stream,
                                                'L',
                                                '\'',
                                                1);
        case 32:
            // utf 32
            valobj.GetValueAsCString(lldb::eFormatUnicode32, value);
            if (!value.empty())
                stream.Printf("%s ", value.c_str());
            return DumpUTFBufferToStream<UTF32>(ConvertUTF32toUTF8,
                                                data,
                                                stream,
                                                'L',
                                                '\'',
                                                1);
        default:
            stream.Printf("size for wchar_t is not valid");
            return true;
    }
    return true;
}

// this function extracts information from a libcxx std::basic_string<>
// irregardless of template arguments. it reports the size (in item count not bytes)
// and the location in memory where the string data can be found
static bool
ExtractLibcxxStringInfo (ValueObject& valobj,
                         ValueObjectSP &location_sp,
                         uint64_t& size)
{
    ValueObjectSP D(valobj.GetChildAtIndexPath({0,0,0,0}));
    if (!D)
        return false;
    
    ValueObjectSP size_mode(D->GetChildAtIndexPath({1,0,0}));
    if (!size_mode)
        return false;
    
    uint64_t size_mode_value(size_mode->GetValueAsUnsigned(0));
    
    if ((size_mode_value & 1) == 0) // this means the string is in short-mode and the data is stored inline
    {
        ValueObjectSP s(D->GetChildAtIndex(1, true));
        if (!s)
            return false;
        size = ((size_mode_value >> 1) % 256);
        location_sp = s->GetChildAtIndex(1, true);
        return (location_sp.get() != nullptr);
    }
    else
    {
        ValueObjectSP l(D->GetChildAtIndex(0, true));
        if (!l)
            return false;
        location_sp = l->GetChildAtIndex(2, true);
        ValueObjectSP size_vo(l->GetChildAtIndex(1, true));
        if (!size_vo || !location_sp)
            return false;
        size = size_vo->GetValueAsUnsigned(0);
        return true;
    }
}

bool
lldb_private::formatters::LibcxxWStringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    uint64_t size = 0;
    ValueObjectSP location_sp((ValueObject*)nullptr);
    if (!ExtractLibcxxStringInfo(valobj, location_sp, size))
        return false;
    if (size == 0)
    {
        stream.Printf("L\"\"");
        return true;
    }
    if (!location_sp)
        return false;
    return WCharStringSummaryProvider(*location_sp.get(), stream);
}

bool
lldb_private::formatters::LibcxxStringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    uint64_t size = 0;
    ValueObjectSP location_sp((ValueObject*)nullptr);
    if (!ExtractLibcxxStringInfo(valobj, location_sp, size))
        return false;
    if (size == 0)
    {
        stream.Printf("\"\"");
        return true;
    }
    if (!location_sp)
        return false;
    Error error;
    if (location_sp->ReadPointedString(stream,
                                       error,
                                       0, // max length is decided by the settings
                                       false) == 0) // do not honor array (terminates on first 0 byte even for a char[])
        stream.Printf("\"\""); // if nothing was read, print an empty string
    return error.Success();
}

bool
lldb_private::formatters::ObjCClassSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj.GetValueAsUnsigned(0)));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;

    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    stream.Printf("%s",class_name);
    return true;
}

template<bool needs_at>
bool
lldb_private::formatters::NSDataSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    if (!class_name || !*class_name)
        return false;
    
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
        if (!ExtractValueFromObjCExpression(valobj, "int", "length", value))
            return false;
    }
    
    stream.Printf("%s%" PRIu64 " byte%s%s",
                  (needs_at ? "@\"" : ""),
                  value,
                  (value > 1 ? "s" : ""),
                  (needs_at ? "\"" : ""));
    
    return true;
}

bool
lldb_private::formatters::NSBundleSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    if (!class_name || !*class_name)
        return false;
    
    if (!strcmp(class_name,"NSBundle"))
    {
        uint64_t offset = 5 * ptr_size;
        ClangASTType type(valobj.GetClangAST(),valobj.GetClangType());
        ValueObjectSP text(valobj.GetSyntheticChildAtOffset(offset, type, true));
        StreamString summary_stream;
        bool was_nsstring_ok = NSStringSummaryProvider(*text.get(), summary_stream);
        if (was_nsstring_ok && summary_stream.GetSize() > 0)
        {
            stream.Printf("%s",summary_stream.GetData());
            return true;
        }
    }
    // this is either an unknown subclass or an NSBundle that comes from [NSBundle mainBundle]
    // which is encoded differently and needs to be handled by running code
    return ExtractSummaryFromObjCExpression(valobj, "NSString*", "bundlePath", stream);
}

bool
lldb_private::formatters::NSNotificationSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    if (!class_name || !*class_name)
        return false;
    
    if (!strcmp(class_name,"NSConcreteNotification"))
    {
        uint64_t offset = ptr_size;
        ClangASTType type(valobj.GetClangAST(),valobj.GetClangType());
        ValueObjectSP text(valobj.GetSyntheticChildAtOffset(offset, type, true));
        StreamString summary_stream;
        bool was_nsstring_ok = NSStringSummaryProvider(*text.get(), summary_stream);
        if (was_nsstring_ok && summary_stream.GetSize() > 0)
        {
            stream.Printf("%s",summary_stream.GetData());
            return true;
        }
    }
    // this is either an unknown subclass or an NSBundle that comes from [NSBundle mainBundle]
    // which is encoded differently and needs to be handled by running code
    return ExtractSummaryFromObjCExpression(valobj, "NSString*", "name", stream);
}

bool
lldb_private::formatters::NSMachPortSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    if (!class_name || !*class_name)
        return false;
    
    uint64_t port_number = 0;
    
    do
    {
        if (!strcmp(class_name,"NSMachPort"))
        {
            uint64_t offset = (ptr_size == 4 ? 12 : 20);
            Error error;
            port_number = process_sp->ReadUnsignedIntegerFromMemory(offset+valobj_addr, 4, 0, error);
            if (error.Success())
                break;
        }
        if (!ExtractValueFromObjCExpression(valobj, "int", "machPort", port_number))
            return false;
    } while (false);
    
    stream.Printf("mach port: %u",(uint32_t)(port_number & 0x00000000FFFFFFFF));
    return true;
}

bool
lldb_private::formatters::CFBagSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    uint32_t count = 0;

    bool is_type_ok = false; // check to see if this is a CFBag we know about
    if (descriptor->IsCFType())
    {
        ConstString type_name(valobj.GetTypeName());
        if (type_name == ConstString("__CFBag") || type_name == ConstString("const struct __CFBag"))
        {
            if (valobj.IsPointerType())
                is_type_ok = true;
        }
    }
    
    if (is_type_ok == false)
    {
        StackFrameSP frame_sp(valobj.GetFrameSP());
        if (!frame_sp)
            return false;
        ValueObjectSP count_sp;
        StreamString expr;
        expr.Printf("(int)CFBagGetCount((void*)0x%" PRIx64 ")",valobj.GetPointerValue());
        if (process_sp->GetTarget().EvaluateExpression(expr.GetData(), frame_sp.get(), count_sp) != eExecutionCompleted)
            return false;
        if (!count_sp)
            return false;
        count = count_sp->GetValueAsUnsigned(0);
    }
    else
    {
        uint32_t offset = 2*ptr_size+4 + valobj_addr;
        Error error;
        count = process_sp->ReadUnsignedIntegerFromMemory(offset, 4, 0, error);
        if (error.Fail())
            return false;
    }
    stream.Printf("@\"%u value%s\"",
                  count,(count == 1 ? "" : "s"));
    return true;
}

bool
lldb_private::formatters::CFBinaryHeapSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    uint32_t count = 0;
    
    bool is_type_ok = false; // check to see if this is a CFBinaryHeap we know about
    if (descriptor->IsCFType())
    {
        ConstString type_name(valobj.GetTypeName());
        if (type_name == ConstString("__CFBinaryHeap") || type_name == ConstString("const struct __CFBinaryHeap"))
        {
            if (valobj.IsPointerType())
                is_type_ok = true;
        }
    }
    
    if (is_type_ok == false)
    {
        StackFrameSP frame_sp(valobj.GetFrameSP());
        if (!frame_sp)
            return false;
        ValueObjectSP count_sp;
        StreamString expr;
        expr.Printf("(int)CFBinaryHeapGetCount((void*)0x%" PRIx64 ")",valobj.GetPointerValue());
        if (process_sp->GetTarget().EvaluateExpression(expr.GetData(), frame_sp.get(), count_sp) != eExecutionCompleted)
            return false;
        if (!count_sp)
            return false;
        count = count_sp->GetValueAsUnsigned(0);
    }
    else
    {
        uint32_t offset = 2*ptr_size;
        Error error;
        count = process_sp->ReadUnsignedIntegerFromMemory(offset, 4, 0, error);
        if (error.Fail())
            return false;
    }
    stream.Printf("@\"%u item%s\"",
                  count,(count == 1 ? "" : "s"));
    return true;
}

bool
lldb_private::formatters::NSNumberSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    if (!class_name || !*class_name)
        return false;
    
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
                    stream.Printf("(long)%" PRId64,value);
                    break;
                default:
                    stream.Printf("unexpected value:(info=%" PRIu64 ", value=%" PRIu64,i_bits,value);
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
                    stream.Printf("(long)%" PRId64,value);
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
        return ExtractSummaryFromObjCExpression(valobj, "NSString*", "stringValue", stream);
    }
}

static bool
ReadAsciiBufferAndDumpToStream (lldb::addr_t location,
                                lldb::ProcessSP& process_sp,
                                Stream& dest,
                                size_t size = 0,
                                Error* error = NULL,
                                size_t *data_read = NULL,
                                char prefix_token = '@',
                                char quote = '"')
{
    Error my_error;
    size_t my_data_read;
    if (!process_sp || location == 0)
        return false;
    
    if (size == 0)
        size = process_sp->GetTarget().GetMaximumSizeOfStringSummary();
    
    lldb::DataBufferSP buffer_sp(new DataBufferHeap(size,0));
    
    my_data_read = process_sp->ReadCStringFromMemory(location, (char*)buffer_sp->GetBytes(), size, my_error);

    if (error)
        *error = my_error;
    if (data_read)
        *data_read = my_data_read;
    
    if (my_error.Fail())
        return false;
    if (my_data_read)
        dest.Printf("%c%c%s%c",prefix_token,quote,(char*)buffer_sp->GetBytes(),quote);
    
    return true;
}

bool
lldb_private::formatters::NSStringSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    if (!class_name || !*class_name)
        return false;
    
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
        // not one of us - but tell me class name
        stream.Printf("class name = %s",class_name);
        return true;
    }
    
    if (is_mutable)
    {
        uint64_t location = 2 * ptr_size + valobj_addr;
        location = process_sp->ReadPointerFromMemory(location, error);
        if (error.Fail())
            return false;
        if (has_explicit_length and is_unicode)
            return ReadUTFBufferAndDumpToStream<UTF16> (ConvertUTF16toUTF8,location, process_sp, stream, '@');
        else
            return ReadAsciiBufferAndDumpToStream(location+1,process_sp,stream);
    }
    else if (is_inline && has_explicit_length && !is_unicode && !is_special && !is_mutable)
    {
        uint64_t location = 3 * ptr_size + valobj_addr;
        return ReadAsciiBufferAndDumpToStream(location,process_sp,stream);
    }
    else if (is_unicode)
    {
        uint64_t location = valobj_addr + 2*ptr_size;
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
        return ReadUTFBufferAndDumpToStream<UTF16> (ConvertUTF16toUTF8, location, process_sp, stream, '@');
    }
    else if (is_special)
    {
        uint64_t location = valobj_addr + (ptr_size == 8 ? 12 : 8);
        return ReadUTFBufferAndDumpToStream<UTF16> (ConvertUTF16toUTF8, location, process_sp, stream, '@');
    }
    else if (is_inline)
    {
        uint64_t location = valobj_addr + 2*ptr_size;
        if (!has_explicit_length)
            location++;
        return ReadAsciiBufferAndDumpToStream(location,process_sp,stream);
    }
    else
    {
        uint64_t location = valobj_addr + 2*ptr_size;
        location = process_sp->ReadPointerFromMemory(location, error);
        if (error.Fail())
            return false;
        return ReadAsciiBufferAndDumpToStream(location,process_sp,stream);
    }
    
    stream.Printf("class name = %s",class_name);
    return true;
    
}

bool
lldb_private::formatters::NSAttributedStringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    TargetSP target_sp(valobj.GetTargetSP());
    if (!target_sp)
        return false;
    uint32_t addr_size = target_sp->GetArchitecture().GetAddressByteSize();
    uint64_t pointee = valobj.GetValueAsUnsigned(0);
    if (!pointee)
        return false;
    pointee += addr_size;
    ClangASTType type(valobj.GetClangAST(),valobj.GetClangType());
    ExecutionContext exe_ctx(target_sp,false);
    ValueObjectSP child_ptr_sp(valobj.CreateValueObjectFromAddress("string_ptr", pointee, exe_ctx, type));
    if (!child_ptr_sp)
        return false;
    DataExtractor data;
    child_ptr_sp->GetData(data);
    ValueObjectSP child_sp(child_ptr_sp->CreateValueObjectFromData("string_data", data, exe_ctx, type));
    child_sp->GetValueAsUnsigned(0);
    if (child_sp)
        return NSStringSummaryProvider(*child_sp, stream);
    return false;
}

bool
lldb_private::formatters::NSMutableAttributedStringSummaryProvider (ValueObject& valobj, Stream& stream)
{
    return NSAttributedStringSummaryProvider(valobj, stream);
}

bool
lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider (ValueObject& valobj, Stream& stream)
{
    stream.Printf("%s",valobj.GetObjectDescription());
    return true;
}

bool
lldb_private::formatters::NSURLSummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    if (!class_name || !*class_name)
        return false;
    
    if (strcmp(class_name, "NSURL") == 0)
    {
        uint64_t offset_text = ptr_size + ptr_size + 8; // ISA + pointer + 8 bytes of data (even on 32bit)
        uint64_t offset_base = offset_text + ptr_size;
        ClangASTType type(valobj.GetClangAST(),valobj.GetClangType());
        ValueObjectSP text(valobj.GetSyntheticChildAtOffset(offset_text, type, true));
        ValueObjectSP base(valobj.GetSyntheticChildAtOffset(offset_base, type, true));
        if (!text)
            return false;
        if (text->GetValueAsUnsigned(0) == 0)
            return false;
        StreamString summary;
        if (!NSStringSummaryProvider(*text, summary))
            return false;
        if (base && base->GetValueAsUnsigned(0))
        {
            if (summary.GetSize() > 0)
                summary.GetString().resize(summary.GetSize()-1);
            summary.Printf(" -- ");
            StreamString base_summary;
            if (NSURLSummaryProvider(*base, base_summary) && base_summary.GetSize() > 0)
                summary.Printf("%s",base_summary.GetSize() > 2 ? base_summary.GetData() + 2 : base_summary.GetData());
        }
        if (summary.GetSize())
        {
            stream.Printf("%s",summary.GetData());
            return true;
        }
    }
    else
    {
        return ExtractSummaryFromObjCExpression(valobj, "NSString*", "description", stream);
    }
    return false;
}

bool
lldb_private::formatters::ObjCBOOLSummaryProvider (ValueObject& valobj, Stream& stream)
{
    const uint32_t type_info = ClangASTContext::GetTypeInfo(valobj.GetClangType(),
                                                            valobj.GetClangAST(),
                                                            NULL);
    
    ValueObjectSP real_guy_sp = valobj.GetSP();
    
    if (type_info & ClangASTContext::eTypeIsPointer)
    {
        Error err;
        real_guy_sp = valobj.Dereference(err);
        if (err.Fail() || !real_guy_sp)
            return false;
    }
    else if (type_info & ClangASTContext::eTypeIsReference)
    {
        real_guy_sp =  valobj.GetChildAtIndex(0, true);
        if (!real_guy_sp)
            return false;
    }
    uint64_t value = real_guy_sp->GetValueAsUnsigned(0);
    if (value == 0)
    {
        stream.Printf("NO");
        return true;
    }
    stream.Printf("YES");
    return true;
}

template <bool is_sel_ptr>
bool
lldb_private::formatters::ObjCSELSummaryProvider (ValueObject& valobj, Stream& stream)
{
    lldb::ValueObjectSP valobj_sp;

    if (!valobj.GetClangAST())
        return false;
    void* char_opaque_type = valobj.GetClangAST()->CharTy.getAsOpaquePtr();
    if (!char_opaque_type)
        return false;
    ClangASTType charstar(valobj.GetClangAST(),ClangASTType::GetPointerType(valobj.GetClangAST(), char_opaque_type));

    ExecutionContext exe_ctx(valobj.GetExecutionContextRef());
    
    if (is_sel_ptr)
    {
        lldb::addr_t data_address = valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
        if (data_address == LLDB_INVALID_ADDRESS)
            return false;
        valobj_sp = ValueObject::CreateValueObjectFromAddress("text", data_address, exe_ctx, charstar);
    }
    else
    {
        DataExtractor data;
        valobj.GetData(data);
        valobj_sp = ValueObject::CreateValueObjectFromData("text", data, exe_ctx, charstar);
    }
    
    if (!valobj_sp)
        return false;
    
    stream.Printf("%s",valobj_sp->GetSummaryAsCString());
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

lldb_private::formatters::VectorIteratorSyntheticFrontEnd::VectorIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp,
                                                                                            ConstString item_name) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_item_name(item_name),
m_item_sp()
{
    if (valobj_sp)
        Update();
}

bool
lldb_private::formatters::VectorIteratorSyntheticFrontEnd::Update()
{
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    
    if (!valobj_sp)
        return false;
    
    ValueObjectSP item_ptr(valobj_sp->GetChildMemberWithName(m_item_name,true));
    if (!item_ptr)
        return false;
    if (item_ptr->GetValueAsUnsigned(0) == 0)
        return false;
    Error err;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    m_item_sp = ValueObject::CreateValueObjectFromAddress("item", item_ptr->GetValueAsUnsigned(0), m_exe_ctx_ref, ClangASTType(item_ptr->GetClangAST(),ClangASTType::GetPointeeType(item_ptr->GetClangType())));
    if (err.Fail())
        m_item_sp.reset();
    return (m_item_sp.get() != NULL);
}

size_t
lldb_private::formatters::VectorIteratorSyntheticFrontEnd::CalculateNumChildren ()
{
    return 1;
}

lldb::ValueObjectSP
lldb_private::formatters::VectorIteratorSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (idx == 0)
        return m_item_sp;
    return lldb::ValueObjectSP();
}

bool
lldb_private::formatters::VectorIteratorSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::VectorIteratorSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (name == ConstString("item"))
        return 0;
    return UINT32_MAX;
}

lldb_private::formatters::VectorIteratorSyntheticFrontEnd::~VectorIteratorSyntheticFrontEnd ()
{
}

template bool
lldb_private::formatters::NSDataSummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::NSDataSummaryProvider<false> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::ObjCSELSummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::ObjCSELSummaryProvider<false> (ValueObject&, Stream&) ;
