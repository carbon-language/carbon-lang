//===-- CXXFormatterFunctions.cpp---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Core/CXXFormatterFunctions.h"

// needed to get ConvertUTF16/32ToUTF8
#define CLANG_NEEDS_THESE_ONE_DAY
#include "clang/Basic/ConvertUTF.h"

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
            sourceSize = bufferSPSize/(origin_encoding >> 2);
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
        
        *data_ptr = 0;
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
    location_sp->ReadPointedString(stream,
                                   error,
                                   0, // max length is decided by the settings
                                   false); // do not honor array (terminates on first 0 byte even for a char[])
    return error.Success();
}

template<bool name_entries>
bool
lldb_private::formatters::NSDictionarySummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    if (!class_name || !*class_name)
        return false;
    
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
        if (!ExtractValueFromObjCExpression(valobj, "int", "count", value))
            return false;
    }
    
    stream.Printf("%s%" PRIu64 " %s%s",
                  (name_entries ? "@\"" : ""),
                  value,
                  (name_entries ? (value == 1 ? "entry" : "entries") : (value == 1 ? "key/value pair" : "key/value pairs")),
                  (name_entries ? "\"" : ""));
    return true;
}

bool
lldb_private::formatters::NSArraySummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    if (!class_name || !*class_name)
        return false;
    
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
        if (!ExtractValueFromObjCExpression(valobj, "int", "count", value))
            return false;
    }
    
    stream.Printf("@\"%" PRIu64 " object%s\"",
                  value,
                  value == 1 ? "" : "s");
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
        // similar to ExtractValueFromObjCExpression but uses summary instead of value
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
            return ReadUTFBufferAndDumpToStream<UTF16> (ConvertUTF16toUTF8,location, process_sp, stream, '@');
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
        return ReadUTFBufferAndDumpToStream<UTF16> (ConvertUTF16toUTF8, location, process_sp, stream, '@');
    }
    else if (is_special)
    {
        uint64_t location = valobj_addr + (ptr_size == 8 ? 12 : 8);
        return ReadUTFBufferAndDumpToStream<UTF16> (ConvertUTF16toUTF8, location, process_sp, stream, '@');
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

bool
lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider (ValueObject& valobj, Stream& stream)
{
    stream.Printf("%s",valobj.GetObjectDescription());
    return true;
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
    lldb::addr_t data_address = LLDB_INVALID_ADDRESS;
    
    if (is_sel_ptr)
        data_address = valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    else
        data_address = valobj.GetAddressOf();

    if (data_address == LLDB_INVALID_ADDRESS)
        return false;
    
    ExecutionContext exe_ctx(valobj.GetExecutionContextRef());
    
    void* char_opaque_type = valobj.GetClangAST()->CharTy.getAsOpaquePtr();
    ClangASTType charstar(valobj.GetClangAST(),ClangASTType::GetPointerType(valobj.GetClangAST(), char_opaque_type));
    
    ValueObjectSP valobj_sp(ValueObject::CreateValueObjectFromAddress("text", data_address, exe_ctx, charstar));
    
    stream.Printf("%s",valobj_sp->GetSummaryAsCString());
    return true;
}

lldb_private::formatters::NSArrayMSyntheticFrontEnd::NSArrayMSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_ptr_size(8),
m_data_32(NULL),
m_data_64(NULL)
{
    if (valobj_sp)
    {
        m_id_type = ClangASTType(valobj_sp->GetClangAST(),valobj_sp->GetClangAST()->ObjCBuiltinIdTy.getAsOpaquePtr());
        Update();
    }
}

size_t
lldb_private::formatters::NSArrayMSyntheticFrontEnd::CalculateNumChildren ()
{
    if (m_data_32)
        return m_data_32->_used;
    if (m_data_64)
        return m_data_64->_used;
    return 0;
}

lldb::ValueObjectSP
lldb_private::formatters::NSArrayMSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (!m_data_32 && !m_data_64)
        return lldb::ValueObjectSP();
    if (idx >= CalculateNumChildren())
        return lldb::ValueObjectSP();
    lldb::addr_t object_at_idx = (m_data_32 ? m_data_32->_data : m_data_64->_data);
    object_at_idx += (idx * m_ptr_size);
    StreamString idx_name;
    idx_name.Printf("[%zu]",idx);
    lldb::ValueObjectSP retval_sp = ValueObject::CreateValueObjectFromAddress(idx_name.GetData(),
                                                                              object_at_idx,
                                                                              m_exe_ctx_ref,
                                                                              m_id_type);
    m_children.push_back(retval_sp);
    return retval_sp;
}

bool
lldb_private::formatters::NSArrayMSyntheticFrontEnd::Update()
{
    m_children.clear();
    ValueObjectSP valobj_sp = m_backend.GetSP();
    m_ptr_size = 0;
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
    if (valobj_sp->IsDynamic())
        valobj_sp = valobj_sp->GetStaticValue();
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    Error error;
    if (valobj_sp->IsPointerType())
    {
        valobj_sp = valobj_sp->Dereference(error);
        if (error.Fail() || !valobj_sp)
            return false;
    }
    error.Clear();
    lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
    if (!process_sp)
        return false;
    m_ptr_size = process_sp->GetAddressByteSize();
    uint64_t data_location = valobj_sp->GetAddressOf() + m_ptr_size;
    if (m_ptr_size == 4)
    {
        m_data_32 = new DataDescriptor_32();
        process_sp->ReadMemory (data_location, m_data_32, sizeof(DataDescriptor_32), error);
    }
    else
    {
        m_data_64 = new DataDescriptor_64();
        process_sp->ReadMemory (data_location, m_data_64, sizeof(DataDescriptor_64), error);
    }
    if (error.Fail())
        return false;
    return false;
}

bool
lldb_private::formatters::NSArrayMSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

static uint32_t
ExtractIndexFromString (const char* item_name)
{
    if (!item_name || !*item_name)
        return UINT32_MAX;
    if (*item_name != '[')
        return UINT32_MAX;
    item_name++;
    uint32_t idx = 0;
    while(*item_name)
    {
        char x = *item_name;
        if (x == ']')
            break;
        if (x < '0' || x > '9')
            return UINT32_MAX;
        idx = 10*idx + (x-'0');
        item_name++;
    }
    return idx;
}

uint32_t
lldb_private::formatters::NSArrayMSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (!m_data_32 && !m_data_64)
        return UINT32_MAX;
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

lldb_private::formatters::NSArrayMSyntheticFrontEnd::~NSArrayMSyntheticFrontEnd ()
{
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
}

lldb_private::formatters::NSArrayISyntheticFrontEnd::NSArrayISyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_ptr_size(8),
m_items(0),
m_data_ptr(0)
{
    if (valobj_sp)
    {
        m_id_type = ClangASTType(valobj_sp->GetClangAST(),valobj_sp->GetClangAST()->ObjCBuiltinIdTy.getAsOpaquePtr());
        Update();
    }
}

lldb_private::formatters::NSArrayISyntheticFrontEnd::~NSArrayISyntheticFrontEnd ()
{
}

uint32_t
lldb_private::formatters::NSArrayISyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

size_t
lldb_private::formatters::NSArrayISyntheticFrontEnd::CalculateNumChildren ()
{
    return m_items;
}

bool
lldb_private::formatters::NSArrayISyntheticFrontEnd::Update()
{
    m_ptr_size = 0;
    m_items = 0;
    m_data_ptr = 0;
    m_children.clear();
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (valobj_sp->IsDynamic())
        valobj_sp = valobj_sp->GetStaticValue();
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    Error error;
    if (valobj_sp->IsPointerType())
    {
        valobj_sp = valobj_sp->Dereference(error);
        if (error.Fail() || !valobj_sp)
            return false;
    }
    error.Clear();
    lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
    if (!process_sp)
        return false;
    m_ptr_size = process_sp->GetAddressByteSize();
    uint64_t data_location = valobj_sp->GetAddressOf() + m_ptr_size;
    m_items = process_sp->ReadPointerFromMemory(data_location, error);
    if (error.Fail())
        return false;
    m_data_ptr = data_location+m_ptr_size;
    return false;
}

bool
lldb_private::formatters::NSArrayISyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSArrayISyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (idx >= CalculateNumChildren())
        return lldb::ValueObjectSP();
    lldb::addr_t object_at_idx = m_data_ptr;
    object_at_idx += (idx * m_ptr_size);
    ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
    if (!process_sp)
        return lldb::ValueObjectSP();
    Error error;
    object_at_idx = process_sp->ReadPointerFromMemory(object_at_idx, error);
    if (error.Fail())
        return lldb::ValueObjectSP();
    StreamString expr;
    expr.Printf("(id)%" PRIu64,object_at_idx);
    StreamString idx_name;
    idx_name.Printf("[%zu]",idx);
    lldb::ValueObjectSP retval_sp = ValueObject::CreateValueObjectFromExpression(idx_name.GetData(), expr.GetData(), m_exe_ctx_ref);
    m_children.push_back(retval_sp);
    return retval_sp;
}

SyntheticChildrenFrontEnd* lldb_private::formatters::NSArraySyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    lldb::ProcessSP process_sp (valobj_sp->GetProcessSP());
    if (!process_sp)
        return NULL;
    ObjCLanguageRuntime *runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    if (!runtime)
        return NULL;
    
    if (!valobj_sp->IsPointerType())
    {
        Error error;
        valobj_sp = valobj_sp->AddressOf(error);
        if (error.Fail() || !valobj_sp)
            return NULL;
    }
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(*valobj_sp.get()));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return NULL;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return NULL;
    
    if (!strcmp(class_name,"__NSArrayI"))
    {
        return (new NSArrayISyntheticFrontEnd(valobj_sp));
    }
    else if (!strcmp(class_name,"__NSArrayM"))
    {
        return (new NSArrayMSyntheticFrontEnd(valobj_sp));
    }
    else
    {
        return (new NSArrayCodeRunningSyntheticFrontEnd(valobj_sp));
    }
}

lldb_private::formatters::NSArrayCodeRunningSyntheticFrontEnd::NSArrayCodeRunningSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get())
{}

size_t
lldb_private::formatters::NSArrayCodeRunningSyntheticFrontEnd::CalculateNumChildren ()
{
    uint64_t count = 0;
    if (ExtractValueFromObjCExpression(m_backend, "int", "count", count))
        return count;
    return 0;
}

lldb::ValueObjectSP
lldb_private::formatters::NSArrayCodeRunningSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    StreamString idx_name;
    idx_name.Printf("[%zu]",idx);
    lldb::ValueObjectSP valobj_sp = CallSelectorOnObject(m_backend,"id","objectAtIndex:",idx);
    if (valobj_sp)
        valobj_sp->SetName(ConstString(idx_name.GetData()));
    return valobj_sp;
}

bool
lldb_private::formatters::NSArrayCodeRunningSyntheticFrontEnd::Update()
{
    return false;
}

bool
lldb_private::formatters::NSArrayCodeRunningSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

uint32_t
lldb_private::formatters::NSArrayCodeRunningSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    return 0;
}

lldb_private::formatters::NSArrayCodeRunningSyntheticFrontEnd::~NSArrayCodeRunningSyntheticFrontEnd ()
{}

SyntheticChildrenFrontEnd* lldb_private::formatters::NSDictionarySyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    
    lldb::ProcessSP process_sp (valobj_sp->GetProcessSP());
    if (!process_sp)
        return NULL;
    ObjCLanguageRuntime *runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    if (!runtime)
        return NULL;

    if (!valobj_sp->IsPointerType())
    {
        Error error;
        valobj_sp = valobj_sp->AddressOf(error);
        if (error.Fail() || !valobj_sp)
            return NULL;
    }
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(*valobj_sp.get()));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return NULL;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return NULL;
    
    if (!strcmp(class_name,"__NSDictionaryI"))
    {
        return (new NSDictionaryISyntheticFrontEnd(valobj_sp));
    }
    else if (!strcmp(class_name,"__NSDictionaryM"))
    {
        return (new NSDictionaryMSyntheticFrontEnd(valobj_sp));
    }
    else
    {
        return (new NSDictionaryCodeRunningSyntheticFrontEnd(valobj_sp));
    }
}

lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::NSDictionaryCodeRunningSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get())
{}

size_t
lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::CalculateNumChildren ()
{
    uint64_t count = 0;
    if (ExtractValueFromObjCExpression(m_backend, "int", "count", count))
        return count;
    return 0;
}

lldb::ValueObjectSP
lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    StreamString idx_name;
    idx_name.Printf("[%zu]",idx);
    StreamString valobj_expr_path;
    m_backend.GetExpressionPath(valobj_expr_path, false);
    StreamString key_fetcher_expr;
    key_fetcher_expr.Printf("(id)[(NSArray*)[%s allKeys] objectAtIndex:%zu]",valobj_expr_path.GetData(),idx);
    StreamString value_fetcher_expr;
    value_fetcher_expr.Printf("(id)[%s objectForKey:%s]",valobj_expr_path.GetData(),key_fetcher_expr.GetData());
    StreamString object_fetcher_expr;
    object_fetcher_expr.Printf("struct __lldb_autogen_nspair { id key; id value; } _lldb_valgen_item; _lldb_valgen_item.key = %s; _lldb_valgen_item.value = %s; _lldb_valgen_item;",key_fetcher_expr.GetData(),value_fetcher_expr.GetData());
    lldb::ValueObjectSP child_sp;
    m_backend.GetTargetSP()->EvaluateExpression(object_fetcher_expr.GetData(), m_backend.GetFrameSP().get(), child_sp,
                                                EvaluateExpressionOptions().SetKeepInMemory(true));
    if (child_sp)
        child_sp->SetName(ConstString(idx_name.GetData()));
    return child_sp;
}

bool
lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::Update()
{
    return false;
}

bool
lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

uint32_t
lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    return 0;
}

lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::~NSDictionaryCodeRunningSyntheticFrontEnd ()
{}

lldb_private::formatters::NSDictionaryISyntheticFrontEnd::NSDictionaryISyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp.get()),
    m_exe_ctx_ref(),
    m_ptr_size(8),
    m_data_32(NULL),
    m_data_64(NULL)
{
    if (valobj_sp)
        Update();
}

lldb_private::formatters::NSDictionaryISyntheticFrontEnd::~NSDictionaryISyntheticFrontEnd ()
{
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
}

uint32_t
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

size_t
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::CalculateNumChildren ()
{
    if (!m_data_32 && !m_data_64)
        return 0;
    return (m_data_32 ? m_data_32->_used : m_data_64->_used);
}

bool
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::Update()
{
    m_children.clear();
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
    m_ptr_size = 0;
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    if (valobj_sp->IsDynamic())
        valobj_sp = valobj_sp->GetStaticValue();
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    Error error;
    if (valobj_sp->IsPointerType())
    {
        valobj_sp = valobj_sp->Dereference(error);
        if (error.Fail() || !valobj_sp)
            return false;
    }
    error.Clear();
    lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
    if (!process_sp)
        return false;
    m_ptr_size = process_sp->GetAddressByteSize();
    uint64_t data_location = valobj_sp->GetAddressOf() + m_ptr_size;
    if (m_ptr_size == 4)
    {
        m_data_32 = new DataDescriptor_32();
        process_sp->ReadMemory (data_location, m_data_32, sizeof(DataDescriptor_32), error);
    }
    else
    {
        m_data_64 = new DataDescriptor_64();
        process_sp->ReadMemory (data_location, m_data_64, sizeof(DataDescriptor_64), error);
    }
    if (error.Fail())
        return false;
    m_data_ptr = data_location + m_ptr_size;
    return false;
}

bool
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    uint32_t num_children = CalculateNumChildren();
    
    if (idx >= num_children)
        return lldb::ValueObjectSP();
    
    if (m_children.empty())
    {
        // do the scan phase
        lldb::addr_t key_at_idx = 0, val_at_idx = 0;
        
        uint32_t tries = 0;
        uint32_t test_idx = 0;
        
        while(tries < num_children)
        {
            key_at_idx = m_data_ptr + (2*test_idx * m_ptr_size);
            val_at_idx = key_at_idx + m_ptr_size;
            ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
            if (!process_sp)
                return lldb::ValueObjectSP();
            Error error;
            key_at_idx = process_sp->ReadPointerFromMemory(key_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            val_at_idx = process_sp->ReadPointerFromMemory(val_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();

            test_idx++;
            
            if (!key_at_idx || !val_at_idx)
                continue;
            tries++;
            
            DictionaryItemDescriptor descriptor = {key_at_idx,val_at_idx,lldb::ValueObjectSP()};
            
            m_children.push_back(descriptor);
        }
    }
    
    if (idx >= m_children.size()) // should never happen
        return lldb::ValueObjectSP();
    
    DictionaryItemDescriptor &dict_item = m_children[idx];
    if (!dict_item.valobj_sp)
    {
        // make the new ValueObject
        StreamString expr;
        expr.Printf("struct __lldb_autogen_nspair { id key; id value; } _lldb_valgen_item; _lldb_valgen_item.key = (id)%" PRIu64 " ; _lldb_valgen_item.value = (id)%" PRIu64 "; _lldb_valgen_item;",dict_item.key_ptr,dict_item.val_ptr);
        StreamString idx_name;
        idx_name.Printf("[%zu]",idx);
        dict_item.valobj_sp = ValueObject::CreateValueObjectFromExpression(idx_name.GetData(), expr.GetData(), m_exe_ctx_ref);
    }
    return dict_item.valobj_sp;
}

lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::NSDictionaryMSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp.get()),
    m_exe_ctx_ref(),
    m_ptr_size(8),
    m_data_32(NULL),
    m_data_64(NULL)
{
    if (valobj_sp)
        Update ();
}

lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::~NSDictionaryMSyntheticFrontEnd ()
{
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
}

uint32_t
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

size_t
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::CalculateNumChildren ()
{
    if (!m_data_32 && !m_data_64)
        return 0;
    return (m_data_32 ? m_data_32->_used : m_data_64->_used);
}

bool
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::Update()
{
    m_children.clear();
    ValueObjectSP valobj_sp = m_backend.GetSP();
    m_ptr_size = 0;
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
    if (!valobj_sp)
        return false;
    if (valobj_sp->IsDynamic())
        valobj_sp = valobj_sp->GetStaticValue();
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    Error error;
    if (valobj_sp->IsPointerType())
    {
        valobj_sp = valobj_sp->Dereference(error);
        if (error.Fail() || !valobj_sp)
            return false;
    }
    error.Clear();
    lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
    if (!process_sp)
        return false;
    m_ptr_size = process_sp->GetAddressByteSize();
    uint64_t data_location = valobj_sp->GetAddressOf() + m_ptr_size;
    if (m_ptr_size == 4)
    {
        m_data_32 = new DataDescriptor_32();
        process_sp->ReadMemory (data_location, m_data_32, sizeof(DataDescriptor_32), error);
    }
    else
    {
        m_data_64 = new DataDescriptor_64();
        process_sp->ReadMemory (data_location, m_data_64, sizeof(DataDescriptor_64), error);
    }
    if (error.Fail())
        return false;
    return false;
}

bool
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    lldb::addr_t m_keys_ptr = (m_data_32 ? m_data_32->_keys_addr : m_data_64->_keys_addr);
    lldb::addr_t m_values_ptr = (m_data_32 ? m_data_32->_objs_addr : m_data_64->_objs_addr);
    
    uint32_t num_children = CalculateNumChildren();
    
    if (idx >= num_children)
        return lldb::ValueObjectSP();
    
    if (m_children.empty())
    {
        // do the scan phase
        lldb::addr_t key_at_idx = 0, val_at_idx = 0;
        
        uint32_t tries = 0;
        uint32_t test_idx = 0;
        
        while(tries < num_children)
        {
            key_at_idx = m_keys_ptr + (test_idx * m_ptr_size);
            val_at_idx = m_values_ptr + (test_idx * m_ptr_size);;
            ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
            if (!process_sp)
                return lldb::ValueObjectSP();
            Error error;
            key_at_idx = process_sp->ReadPointerFromMemory(key_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            val_at_idx = process_sp->ReadPointerFromMemory(val_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            
            test_idx++;
            
            if (!key_at_idx || !val_at_idx)
                continue;
            tries++;
            
            DictionaryItemDescriptor descriptor = {key_at_idx,val_at_idx,lldb::ValueObjectSP()};
            
            m_children.push_back(descriptor);
        }
    }
    
    if (idx >= m_children.size()) // should never happen
        return lldb::ValueObjectSP();
    
    DictionaryItemDescriptor &dict_item = m_children[idx];
    if (!dict_item.valobj_sp)
    {
        // make the new ValueObject
        StreamString expr;
        expr.Printf("struct __lldb_autogen_nspair { id key; id value; } _lldb_valgen_item; _lldb_valgen_item.key = (id)%" PRIu64 " ; _lldb_valgen_item.value = (id)%" PRIu64 "; _lldb_valgen_item;",dict_item.key_ptr,dict_item.val_ptr);
        StreamString idx_name;
        idx_name.Printf("[%zu]",idx);
        dict_item.valobj_sp = ValueObject::CreateValueObjectFromExpression(idx_name.GetData(), expr.GetData(), m_exe_ctx_ref);
    }
    return dict_item.valobj_sp;
}

template bool
lldb_private::formatters::NSDictionarySummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::NSDictionarySummaryProvider<false> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::NSDataSummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::NSDataSummaryProvider<false> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::ObjCSELSummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::ObjCSELSummaryProvider<false> (ValueObject&, Stream&) ;
