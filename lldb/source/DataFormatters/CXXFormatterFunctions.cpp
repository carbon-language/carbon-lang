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
lldb_private::formatters::Char16StringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions&)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    ReadStringAndDumpToStreamOptions options(valobj);
    options.SetLocation(valobj_addr);
    options.SetProcessSP(process_sp);
    options.SetStream(&stream);
    options.SetPrefixToken('u');
    
    if (!ReadStringAndDumpToStream<StringElementType::UTF16>(options))
    {
        stream.Printf("Summary Unavailable");
        return true;
    }

    return true;
}

bool
lldb_private::formatters::Char32StringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions&)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    ReadStringAndDumpToStreamOptions options(valobj);
    options.SetLocation(valobj_addr);
    options.SetProcessSP(process_sp);
    options.SetStream(&stream);
    options.SetPrefixToken('U');
    
    if (!ReadStringAndDumpToStream<StringElementType::UTF32>(options))
    {
        stream.Printf("Summary Unavailable");
        return true;
    }
    
    return true;
}

bool
lldb_private::formatters::WCharStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions&)
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

    clang::ASTContext* ast = valobj.GetClangType().GetASTContext();
    
    if (!ast)
        return false;

    ClangASTType wchar_clang_type = ClangASTContext::GetBasicType(ast, lldb::eBasicTypeWChar);
    const uint32_t wchar_size = wchar_clang_type.GetBitSize(nullptr); // Safe to pass NULL for exe_scope here

    ReadStringAndDumpToStreamOptions options(valobj);
    options.SetLocation(data_addr);
    options.SetProcessSP(process_sp);
    options.SetStream(&stream);
    options.SetPrefixToken('L');
    
    switch (wchar_size)
    {
        case 8:
            return ReadStringAndDumpToStream<StringElementType::UTF8>(options);
        case 16:
            return ReadStringAndDumpToStream<StringElementType::UTF16>(options);
        case 32:
            return ReadStringAndDumpToStream<StringElementType::UTF32>(options);
        default:
            stream.Printf("size for wchar_t is not valid");
            return true;
    }
    return true;
}

bool
lldb_private::formatters::Char16SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions&)
{
    DataExtractor data;
    Error error;
    valobj.GetData(data, error);
    
    if (error.Fail())
        return false;
    
    std::string value;
    valobj.GetValueAsCString(lldb::eFormatUnicode16, value);
    if (!value.empty())
        stream.Printf("%s ", value.c_str());

    ReadBufferAndDumpToStreamOptions options(valobj);
    options.SetData(data);
    options.SetStream(&stream);
    options.SetPrefixToken('u');
    options.SetQuote('\'');
    options.SetSourceSize(1);
    
    return ReadBufferAndDumpToStream<StringElementType::UTF16>(options);
}

bool
lldb_private::formatters::Char32SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions&)
{
    DataExtractor data;
    Error error;
    valobj.GetData(data, error);
    
    if (error.Fail())
        return false;
    
    std::string value;
    valobj.GetValueAsCString(lldb::eFormatUnicode32, value);
    if (!value.empty())
        stream.Printf("%s ", value.c_str());
    
    ReadBufferAndDumpToStreamOptions options(valobj);
    options.SetData(data);
    options.SetStream(&stream);
    options.SetPrefixToken('U');
    options.SetQuote('\'');
    options.SetSourceSize(1);
    
    return ReadBufferAndDumpToStream<StringElementType::UTF32>(options);
}

bool
lldb_private::formatters::WCharSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions&)
{
    DataExtractor data;
    Error error;
    valobj.GetData(data, error);
    
    if (error.Fail())
        return false;
    
    ReadBufferAndDumpToStreamOptions options(valobj);
    options.SetData(data);
    options.SetStream(&stream);
    options.SetPrefixToken('L');
    options.SetQuote('\'');
    options.SetSourceSize(1);
    
    return ReadBufferAndDumpToStream<StringElementType::UTF16>(options);
}

// the field layout in a libc++ string (cap, side, data or data, size, cap)
enum LibcxxStringLayoutMode
{
    eLibcxxStringLayoutModeCSD = 0,
    eLibcxxStringLayoutModeDSC = 1,
    eLibcxxStringLayoutModeInvalid = 0xffff
};

// this function abstracts away the layout and mode details of a libc++ string
// and returns the address of the data and the size ready for callers to consume
static bool
ExtractLibcxxStringInfo (ValueObject& valobj,
                         ValueObjectSP &location_sp,
                         uint64_t& size)
{
    ValueObjectSP D(valobj.GetChildAtIndexPath({0,0,0,0}));
    if (!D)
        return false;
    
    ValueObjectSP layout_decider(D->GetChildAtIndexPath({0,0}));
    
    // this child should exist
    if (!layout_decider)
        return false;
    
    ConstString g_data_name("__data_");
    ConstString g_size_name("__size_");
    bool short_mode = false; // this means the string is in short-mode and the data is stored inline
    LibcxxStringLayoutMode layout = (layout_decider->GetName() == g_data_name) ? eLibcxxStringLayoutModeDSC : eLibcxxStringLayoutModeCSD;
    uint64_t size_mode_value = 0;
    
    if (layout == eLibcxxStringLayoutModeDSC)
    {
        ValueObjectSP size_mode(D->GetChildAtIndexPath({1,1,0}));
        if (!size_mode)
            return false;
        
        if (size_mode->GetName() != g_size_name)
        {
            // we are hitting the padding structure, move along
            size_mode = D->GetChildAtIndexPath({1,1,1});
            if (!size_mode)
                return false;
        }
        
        size_mode_value = (size_mode->GetValueAsUnsigned(0));
        short_mode = ((size_mode_value & 0x80) == 0);
    }
    else
    {
        ValueObjectSP size_mode(D->GetChildAtIndexPath({1,0,0}));
        if (!size_mode)
            return false;
        
        size_mode_value = (size_mode->GetValueAsUnsigned(0));
        short_mode = ((size_mode_value & 1) == 0);
    }
    
    if (short_mode)
    {
        ValueObjectSP s(D->GetChildAtIndex(1, true));
        if (!s)
            return false;
        location_sp = s->GetChildAtIndex((layout == eLibcxxStringLayoutModeDSC) ? 0 : 1, true);
        size = (layout == eLibcxxStringLayoutModeDSC) ? size_mode_value : ((size_mode_value >> 1) % 256);
        return (location_sp.get() != nullptr);
    }
    else
    {
        ValueObjectSP l(D->GetChildAtIndex(0, true));
        if (!l)
            return false;
        // we can use the layout_decider object as the data pointer
        location_sp = (layout == eLibcxxStringLayoutModeDSC) ? layout_decider : l->GetChildAtIndex(2, true);
        ValueObjectSP size_vo(l->GetChildAtIndex(1, true));
        if (!size_vo || !location_sp)
            return false;
        size = size_vo->GetValueAsUnsigned(0);
        return true;
    }
}

bool
lldb_private::formatters::LibcxxWStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
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
    return WCharStringSummaryProvider(*location_sp.get(), stream, options);
}

bool
lldb_private::formatters::LibcxxStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& summary_options)
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
    
    DataExtractor extractor;
    if (summary_options.GetCapping() == TypeSummaryCapping::eTypeSummaryCapped)
        size = std::min<decltype(size)>(size, valobj.GetTargetSP()->GetMaximumSizeOfStringSummary());
    location_sp->GetPointeeData(extractor, 0, size);
    
    ReadBufferAndDumpToStreamOptions options(valobj);
    options.SetData(extractor); // none of this matters for a string - pass some defaults
    options.SetStream(&stream);
    options.SetPrefixToken(0);
    options.SetQuote('"');
    options.SetSourceSize(size);
    lldb_private::formatters::ReadBufferAndDumpToStream<lldb_private::formatters::StringElementType::ASCII>(options);
    
    return true;
}

bool
lldb_private::formatters::ObjCClassSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptorFromISA(valobj.GetValueAsUnsigned(0)));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;

    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    stream.Printf("%s",class_name);
    return true;
}

class ObjCClassSyntheticChildrenFrontEnd : public SyntheticChildrenFrontEnd
{
public:
    ObjCClassSyntheticChildrenFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp.get())
    {
    }
    
    virtual size_t
    CalculateNumChildren ()
    {
        return 0;
    }
    
    virtual lldb::ValueObjectSP
    GetChildAtIndex (size_t idx)
    {
        return lldb::ValueObjectSP();
    }
    
    virtual bool
    Update()
    {
        return false;
    }
    
    virtual bool
    MightHaveChildren ()
    {
        return false;
    }
    
    virtual size_t
    GetIndexOfChildWithName (const ConstString &name)
    {
        return UINT32_MAX;
    }
    
    virtual
    ~ObjCClassSyntheticChildrenFrontEnd ()
    {
    }
};

SyntheticChildrenFrontEnd*
lldb_private::formatters::ObjCClassSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    return new ObjCClassSyntheticChildrenFrontEnd(valobj_sp);
}

template<bool needs_at>
bool
lldb_private::formatters::NSDataSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
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
                  (value != 1 ? "s" : ""),
                  (needs_at ? "\"" : ""));
    
    return true;
}

static bool
ReadAsciiBufferAndDumpToStream (lldb::addr_t location,
                                lldb::ProcessSP& process_sp,
                                Stream& dest,
                                uint32_t size = 0,
                                Error* error = NULL,
                                size_t *data_read = NULL,
                                char prefix_token = '@',
                                char quote = '"')
{
    Error my_error;
    size_t my_data_read;
    if (!process_sp || location == 0)
        return false;
    
    if (!size)
        size = process_sp->GetTarget().GetMaximumSizeOfStringSummary();
    else
        size = std::min(size,process_sp->GetTarget().GetMaximumSizeOfStringSummary());
    
    lldb::DataBufferSP buffer_sp(new DataBufferHeap(size,0));
    
    my_data_read = process_sp->ReadCStringFromMemory(location, (char*)buffer_sp->GetBytes(), size, my_error);

    if (error)
        *error = my_error;
    if (data_read)
        *data_read = my_data_read;
    
    if (my_error.Fail())
        return false;
    
    dest.Printf("%c%c",prefix_token,quote);
    
    if (my_data_read)
        dest.Printf("%s",(char*)buffer_sp->GetBytes());
    
    dest.Printf("%c",quote);
    
    return true;
}

bool
lldb_private::formatters::NSTaggedString_SummaryProvider (ObjCLanguageRuntime::ClassDescriptorSP descriptor, Stream& stream)
{
    if (!descriptor)
        return false;
    uint64_t len_bits = 0, data_bits = 0;
    if (!descriptor->GetTaggedPointerInfo(&len_bits,&data_bits,nullptr))
        return false;
    
    static const int g_MaxNonBitmaskedLen = 7; //TAGGED_STRING_UNPACKED_MAXLEN
    static const int g_SixbitMaxLen = 9;
    static const int g_fiveBitMaxLen = 11;
    
    static const char *sixBitToCharLookup = "eilotrm.apdnsIc ufkMShjTRxgC4013" "bDNvwyUL2O856P-B79AFKEWV_zGJ/HYX";
    
    if (len_bits > g_fiveBitMaxLen)
        return false;
    
    // this is a fairly ugly trick - pretend that the numeric value is actually a char*
    // this works under a few assumptions:
    // little endian architecture
    // sizeof(uint64_t) > g_MaxNonBitmaskedLen
    if (len_bits <= g_MaxNonBitmaskedLen)
    {
        stream.Printf("@\"%s\"",(const char*)&data_bits);
        return true;
    }
    
    // if the data is bitmasked, we need to actually process the bytes
    uint8_t bitmask = 0;
    uint8_t shift_offset = 0;
    
    if (len_bits <= g_SixbitMaxLen)
    {
        bitmask = 0x03f;
        shift_offset = 6;
    }
    else
    {
        bitmask = 0x01f;
        shift_offset = 5;
    }
    
    std::vector<uint8_t> bytes;
    bytes.resize(len_bits);
    for (; len_bits > 0; data_bits >>= shift_offset, --len_bits)
    {
        uint8_t packed = data_bits & bitmask;
        bytes.insert(bytes.begin(), sixBitToCharLookup[packed]);
    }
    
    stream.Printf("@\"%s\"",&bytes[0]);
    return true;
}

static ClangASTType
GetNSPathStore2Type (Target &target)
{
    static ConstString g_type_name("__lldb_autogen_nspathstore2");

    ClangASTContext *ast_ctx = target.GetScratchClangASTContext();
    
    if (!ast_ctx)
        return ClangASTType();
    
    ClangASTType voidstar = ast_ctx->GetBasicType(lldb::eBasicTypeVoid).GetPointerType();
    ClangASTType uint32 = ast_ctx->GetIntTypeFromBitSize(32, false);
    
    return ast_ctx->GetOrCreateStructForIdentifier(g_type_name, {
        {"isa",voidstar},
        {"lengthAndRef",uint32},
        {"buffer",voidstar}
    });
}

bool
lldb_private::formatters::NSStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& summary_options)
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
    
    bool is_tagged_ptr = (0 == strcmp(class_name,"NSTaggedPointerString")) && descriptor->GetTaggedPointerInfo();
    // for a tagged pointer, the descriptor has everything we need
    if (is_tagged_ptr)
        return NSTaggedString_SummaryProvider(descriptor, stream);
    
    // if not a tagged pointer that we know about, try the normal route
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
    bool has_null = (info_bits & 8) == 8;
    
    size_t explicit_length = 0;
    if (!has_null && has_explicit_length && !is_special)
    {
        lldb::addr_t explicit_length_offset = 2*ptr_size;
        if (is_mutable && !is_inline)
            explicit_length_offset = explicit_length_offset + ptr_size; //  notInlineMutable.length;
        else if (is_inline)
            explicit_length = explicit_length + 0; // inline1.length;
        else if (!is_inline && !is_mutable)
            explicit_length_offset = explicit_length_offset + ptr_size; // notInlineImmutable1.length;
        else
            explicit_length_offset = 0;

        if (explicit_length_offset)
        {
            explicit_length_offset = valobj_addr + explicit_length_offset;
            explicit_length = process_sp->ReadUnsignedIntegerFromMemory(explicit_length_offset, 4, 0, error);
        }
    }
    
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
        if (has_explicit_length && is_unicode)
        {
            ReadStringAndDumpToStreamOptions options(valobj);
            options.SetLocation(location);
            options.SetProcessSP(process_sp);
            options.SetStream(&stream);
            options.SetPrefixToken('@');
            options.SetQuote('"');
            options.SetSourceSize(explicit_length);
            options.SetNeedsZeroTermination(false);
            options.SetIgnoreMaxLength(summary_options.GetCapping() == TypeSummaryCapping::eTypeSummaryUncapped);
            return ReadStringAndDumpToStream<StringElementType::UTF16>(options);
        }
        else
        {
            ReadStringAndDumpToStreamOptions options(valobj);
            options.SetLocation(location+1);
            options.SetProcessSP(process_sp);
            options.SetStream(&stream);
            options.SetPrefixToken('@');
            options.SetSourceSize(explicit_length);
            options.SetNeedsZeroTermination(false);
            options.SetIgnoreMaxLength(summary_options.GetCapping() == TypeSummaryCapping::eTypeSummaryUncapped);
            return ReadStringAndDumpToStream<StringElementType::ASCII>(options);
        }
    }
    else if (is_inline && has_explicit_length && !is_unicode && !is_special && !is_mutable)
    {
        uint64_t location = 3 * ptr_size + valobj_addr;
        return ReadAsciiBufferAndDumpToStream(location,process_sp,stream,explicit_length);
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
        ReadStringAndDumpToStreamOptions options(valobj);
        options.SetLocation(location);
        options.SetProcessSP(process_sp);
        options.SetStream(&stream);
        options.SetPrefixToken('@');
        options.SetQuote('"');
        options.SetSourceSize(explicit_length);
        options.SetNeedsZeroTermination(has_explicit_length == false);
        options.SetIgnoreMaxLength(summary_options.GetCapping() == TypeSummaryCapping::eTypeSummaryUncapped);
        return ReadStringAndDumpToStream<StringElementType::UTF16> (options);
    }
    else if (is_special)
    {
        ProcessStructReader reader(valobj.GetProcessSP().get(), valobj.GetValueAsUnsigned(0), GetNSPathStore2Type(*valobj.GetTargetSP()));
        explicit_length = reader.GetField<uint32_t>(ConstString("lengthAndRef")) >> 20;
        lldb::addr_t location = valobj.GetValueAsUnsigned(0) + ptr_size + 4;
        
        ReadStringAndDumpToStreamOptions options(valobj);
        options.SetLocation(location);
        options.SetProcessSP(process_sp);
        options.SetStream(&stream);
        options.SetPrefixToken('@');
        options.SetQuote('"');
        options.SetSourceSize(explicit_length);
        options.SetNeedsZeroTermination(has_explicit_length == false);
        options.SetIgnoreMaxLength(summary_options.GetCapping() == TypeSummaryCapping::eTypeSummaryUncapped);
        return ReadStringAndDumpToStream<StringElementType::UTF16> (options);
    }
    else if (is_inline)
    {
        uint64_t location = valobj_addr + 2*ptr_size;
        if (!has_explicit_length)
            location++;
        ReadStringAndDumpToStreamOptions options(valobj);
        options.SetLocation(location);
        options.SetProcessSP(process_sp);
        options.SetStream(&stream);
        options.SetPrefixToken('@');
        options.SetSourceSize(explicit_length);
        options.SetIgnoreMaxLength(summary_options.GetCapping() == TypeSummaryCapping::eTypeSummaryUncapped);
        return ReadStringAndDumpToStream<StringElementType::ASCII>(options);
    }
    else
    {
        uint64_t location = valobj_addr + 2*ptr_size;
        location = process_sp->ReadPointerFromMemory(location, error);
        if (error.Fail())
            return false;
        if (has_explicit_length && !has_null)
            explicit_length++; // account for the fact that there is no NULL and we need to have one added
        ReadStringAndDumpToStreamOptions options(valobj);
        options.SetLocation(location);
        options.SetProcessSP(process_sp);
        options.SetPrefixToken('@');
        options.SetStream(&stream);
        options.SetSourceSize(explicit_length);
        options.SetIgnoreMaxLength(summary_options.GetCapping() == TypeSummaryCapping::eTypeSummaryUncapped);
        return ReadStringAndDumpToStream<StringElementType::ASCII>(options);
    }
}

bool
lldb_private::formatters::NSAttributedStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    TargetSP target_sp(valobj.GetTargetSP());
    if (!target_sp)
        return false;
    uint32_t addr_size = target_sp->GetArchitecture().GetAddressByteSize();
    uint64_t pointer_value = valobj.GetValueAsUnsigned(0);
    if (!pointer_value)
        return false;
    pointer_value += addr_size;
    ClangASTType type(valobj.GetClangType());
    ExecutionContext exe_ctx(target_sp,false);
    ValueObjectSP child_ptr_sp(valobj.CreateValueObjectFromAddress("string_ptr", pointer_value, exe_ctx, type));
    if (!child_ptr_sp)
        return false;
    DataExtractor data;
    Error error;
    child_ptr_sp->GetData(data, error);
    if (error.Fail())
        return false;
    ValueObjectSP child_sp(child_ptr_sp->CreateValueObjectFromData("string_data", data, exe_ctx, type));
    child_sp->GetValueAsUnsigned(0);
    if (child_sp)
        return NSStringSummaryProvider(*child_sp, stream, options);
    return false;
}

bool
lldb_private::formatters::NSMutableAttributedStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    return NSAttributedStringSummaryProvider(valobj, stream, options);
}

bool
lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    stream.Printf("%s",valobj.GetObjectDescription());
    return true;
}

bool
lldb_private::formatters::ObjCBOOLSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    const uint32_t type_info = valobj.GetClangType().GetTypeInfo();
    
    ValueObjectSP real_guy_sp = valobj.GetSP();
    
    if (type_info & eTypeIsPointer)
    {
        Error err;
        real_guy_sp = valobj.Dereference(err);
        if (err.Fail() || !real_guy_sp)
            return false;
    }
    else if (type_info & eTypeIsReference)
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
lldb_private::formatters::ObjCSELSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    lldb::ValueObjectSP valobj_sp;

    ClangASTType charstar (valobj.GetClangType().GetBasicTypeFromAST(eBasicTypeChar).GetPointerType());
    
    if (!charstar)
        return false;

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
        Error error;
        valobj.GetData(data, error);
        if (error.Fail())
            return false;
        valobj_sp = ValueObject::CreateValueObjectFromData("text", data, exe_ctx, charstar);
    }
    
    if (!valobj_sp)
        return false;
    
    stream.Printf("%s",valobj_sp->GetSummaryAsCString());
    return true;
}

// POSIX has an epoch on Jan-1-1970, but Cocoa prefers Jan-1-2001
// this call gives the POSIX equivalent of the Cocoa epoch
time_t
lldb_private::formatters::GetOSXEpoch ()
{
    static time_t epoch = 0;
    if (!epoch)
    {
#ifndef _WIN32
        tzset();
        tm tm_epoch;
        tm_epoch.tm_sec = 0;
        tm_epoch.tm_hour = 0;
        tm_epoch.tm_min = 0;
        tm_epoch.tm_mon = 0;
        tm_epoch.tm_mday = 1;
        tm_epoch.tm_year = 2001-1900; // for some reason, we need to subtract 1900 from this field. not sure why.
        tm_epoch.tm_isdst = -1;
        tm_epoch.tm_gmtoff = 0;
        tm_epoch.tm_zone = NULL;
        epoch = timegm(&tm_epoch);
#endif
    }
    return epoch;
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
    m_item_sp.reset();

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
    m_item_sp = CreateValueObjectFromAddress("item", item_ptr->GetValueAsUnsigned(0), m_exe_ctx_ref, item_ptr->GetClangType().GetPointeeType());
    if (err.Fail())
        m_item_sp.reset();
    return false;
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
lldb_private::formatters::NSDataSummaryProvider<true> (ValueObject&, Stream&, const TypeSummaryOptions&) ;

template bool
lldb_private::formatters::NSDataSummaryProvider<false> (ValueObject&, Stream&, const TypeSummaryOptions&) ;

template bool
lldb_private::formatters::ObjCSELSummaryProvider<true> (ValueObject&, Stream&, const TypeSummaryOptions&) ;

template bool
lldb_private::formatters::ObjCSELSummaryProvider<false> (ValueObject&, Stream&, const TypeSummaryOptions&) ;
