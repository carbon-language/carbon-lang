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
    location_sp->ReadPointedString(stream,
                                   error,
                                   0, // max length is decided by the settings
                                   false); // do not honor array (terminates on first 0 byte even for a char[])
    return error.Success();
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
        return ExtractSummaryFromObjCExpression(valobj, "NSString*", "stringValue", stream);
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

lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::LibcxxVectorBoolSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp.get()),
    m_exe_ctx_ref(),
    m_count(0),
    m_base_data_address(0),
    m_options()
    {
        if (valobj_sp)
            Update();
        m_options.SetCoerceToId(false)
                 .SetUnwindOnError(true)
                 .SetKeepInMemory(true)
                 .SetUseDynamic(lldb::eDynamicCanRunTarget);
    }

size_t
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::CalculateNumChildren ()
{
    return m_count;
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (idx >= m_count)
        return ValueObjectSP();
    if (m_base_data_address == 0 || m_count == 0)
        return ValueObjectSP();
    size_t byte_idx = (idx >> 3); // divide by 8 to get byte index
    size_t bit_index = (idx & 7); // efficient idx % 8 for bit index
    lldb::addr_t byte_location = m_base_data_address + byte_idx;
    ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
    if (!process_sp)
        return ValueObjectSP();
    uint8_t byte = 0;
    uint8_t mask = 0;
    Error err;
    size_t bytes_read = process_sp->ReadMemory(byte_location, &byte, 1, err);
    if (err.Fail() || bytes_read == 0)
        return ValueObjectSP();
    switch (bit_index)
    {
        case 0:
            mask = 1; break;
        case 1:
            mask = 2; break;
        case 2:
            mask = 4; break;
        case 3:
            mask = 8; break;
        case 4:
            mask = 16; break;
        case 5:
            mask = 32; break;
        case 6:
            mask = 64; break;
        case 7:
            mask = 128; break;
        default:
            return ValueObjectSP();
    }
    bool bit_set = ((byte & mask) != 0);
    Target& target(process_sp->GetTarget());
    ValueObjectSP retval_sp;
    if (bit_set)
        target.EvaluateExpression("(bool)true", NULL, retval_sp);
    else
        target.EvaluateExpression("(bool)false", NULL, retval_sp);
    StreamString name; name.Printf("[%zu]",idx);
    if (retval_sp)
        retval_sp->SetName(ConstString(name.GetData()));
    return retval_sp;
}

/*(std::__1::vector<std::__1::allocator<bool> >) vBool = {
    __begin_ = 0x00000001001000e0
    __size_ = 56
    __cap_alloc_ = {
        std::__1::__libcpp_compressed_pair_imp<unsigned long, std::__1::allocator<unsigned long> > = {
            __first_ = 1
        }
    }
}*/

bool
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::Update()
{
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    if (valobj_sp->IsDynamic())
        valobj_sp = valobj_sp->GetStaticValue();
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    ValueObjectSP size_sp(valobj_sp->GetChildMemberWithName(ConstString("__size_"), true));
    if (!size_sp)
        return false;
    m_count = size_sp->GetValueAsUnsigned(0);
    if (!m_count)
        return true;
    ValueObjectSP begin_sp(valobj_sp->GetChildMemberWithName(ConstString("__begin_"), true));
    if (!begin_sp)
    {
        m_count = 0;
        return false;
    }
    m_base_data_address = begin_sp->GetValueAsUnsigned(0);
    if (!m_base_data_address)
    {
        m_count = 0;
        return false;
    }
    return true;
}

bool
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (!m_count || !m_base_data_address)
        return UINT32_MAX;
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::~LibcxxVectorBoolSyntheticFrontEnd ()
{}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new LibcxxVectorBoolSyntheticFrontEnd(valobj_sp));
}

lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::LibstdcppVectorBoolSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_count(0),
m_base_data_address(0),
m_options()
{
    if (valobj_sp)
        Update();
    m_options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true)
    .SetUseDynamic(lldb::eDynamicCanRunTarget);
}

size_t
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::CalculateNumChildren ()
{
    return m_count;
}

lldb::ValueObjectSP
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (idx >= m_count)
        return ValueObjectSP();
    if (m_base_data_address == 0 || m_count == 0)
        return ValueObjectSP();
    size_t byte_idx = (idx >> 3); // divide by 8 to get byte index
    size_t bit_index = (idx & 7); // efficient idx % 8 for bit index
    lldb::addr_t byte_location = m_base_data_address + byte_idx;
    ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
    if (!process_sp)
        return ValueObjectSP();
    uint8_t byte = 0;
    uint8_t mask = 0;
    Error err;
    size_t bytes_read = process_sp->ReadMemory(byte_location, &byte, 1, err);
    if (err.Fail() || bytes_read == 0)
        return ValueObjectSP();
    switch (bit_index)
    {
        case 0:
            mask = 1; break;
        case 1:
            mask = 2; break;
        case 2:
            mask = 4; break;
        case 3:
            mask = 8; break;
        case 4:
            mask = 16; break;
        case 5:
            mask = 32; break;
        case 6:
            mask = 64; break;
        case 7:
            mask = 128; break;
        default:
            return ValueObjectSP();
    }
    bool bit_set = ((byte & mask) != 0);
    Target& target(process_sp->GetTarget());
    ValueObjectSP retval_sp;
    if (bit_set)
        target.EvaluateExpression("(bool)true", NULL, retval_sp);
    else
        target.EvaluateExpression("(bool)false", NULL, retval_sp);
    StreamString name; name.Printf("[%zu]",idx);
    if (retval_sp)
        retval_sp->SetName(ConstString(name.GetData()));
    return retval_sp;
}

/*((std::vector<std::allocator<bool> >) vBool = {
 (std::_Bvector_base<std::allocator<bool> >) std::_Bvector_base<std::allocator<bool> > = {
 (std::_Bvector_base<std::allocator<bool> >::_Bvector_impl) _M_impl = {
 (std::_Bit_iterator) _M_start = {
 (std::_Bit_iterator_base) std::_Bit_iterator_base = {
 (_Bit_type *) _M_p = 0x0016b160
 (unsigned int) _M_offset = 0
 }
 }
 (std::_Bit_iterator) _M_finish = {
 (std::_Bit_iterator_base) std::_Bit_iterator_base = {
 (_Bit_type *) _M_p = 0x0016b16c
 (unsigned int) _M_offset = 16
 }
 }
 (_Bit_type *) _M_end_of_storage = 0x0016b170
 }
 }
 }
*/

bool
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::Update()
{
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    if (valobj_sp->IsDynamic())
        valobj_sp = valobj_sp->GetStaticValue();
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    
    ValueObjectSP m_impl_sp(valobj_sp->GetChildMemberWithName(ConstString("_M_impl"), true));
    if (!m_impl_sp)
        return false;
    
    ValueObjectSP m_start_sp(m_impl_sp->GetChildMemberWithName(ConstString("_M_start"), true));
    ValueObjectSP m_finish_sp(m_impl_sp->GetChildMemberWithName(ConstString("_M_finish"), true));
    
    ValueObjectSP start_p_sp, finish_p_sp, finish_offset_sp;
    
    if (!m_start_sp || !m_finish_sp)
        return false;
    
    start_p_sp = m_start_sp->GetChildMemberWithName(ConstString("_M_p"), true);
    finish_p_sp = m_finish_sp->GetChildMemberWithName(ConstString("_M_p"), true);
    finish_offset_sp = m_finish_sp->GetChildMemberWithName(ConstString("_M_offset"), true);
    
    if (!start_p_sp || !finish_offset_sp || !finish_p_sp)
        return false;

    m_base_data_address = start_p_sp->GetValueAsUnsigned(0);
    if (!m_base_data_address)
        return false;
    
    lldb::addr_t end_data_address(finish_p_sp->GetValueAsUnsigned(0));
    if (!end_data_address)
        return false;
    
    if (end_data_address < m_base_data_address)
        return false;
    else
        m_count = finish_offset_sp->GetValueAsUnsigned(0) + (end_data_address-m_base_data_address)*8;

    return true;
}

bool
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (!m_count || !m_base_data_address)
        return UINT32_MAX;
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::~LibstdcppVectorBoolSyntheticFrontEnd ()
{}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new LibstdcppVectorBoolSyntheticFrontEnd(valobj_sp));
}

template bool
lldb_private::formatters::NSDataSummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::NSDataSummaryProvider<false> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::ObjCSELSummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::ObjCSELSummaryProvider<false> (ValueObject&, Stream&) ;
