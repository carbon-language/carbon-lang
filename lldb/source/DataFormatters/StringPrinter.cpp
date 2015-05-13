//===-- StringPrinter.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/StringPrinter.h"

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "llvm/Support/ConvertUTF.h"

#include <ctype.h>
#include <functional>
#include <locale>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

// I can't use a std::unique_ptr for this because the Deleter is a template argument there
// and I want the same type to represent both pointers I want to free and pointers I don't need
// to free - which is what this class essentially is
// It's very specialized to the needs of this file, and not suggested for general use
template <typename T = uint8_t, typename U = char, typename S = size_t>
struct StringPrinterBufferPointer
{
public:
    
    typedef std::function<void(const T*)> Deleter;
    
    StringPrinterBufferPointer (std::nullptr_t ptr) :
    m_data(nullptr),
    m_size(0),
    m_deleter()
    {}
    
    StringPrinterBufferPointer(const T* bytes, S size, Deleter deleter = nullptr) :
    m_data(bytes),
    m_size(size),
    m_deleter(deleter)
    {}
    
    StringPrinterBufferPointer(const U* bytes, S size, Deleter deleter = nullptr) :
    m_data((T*)bytes),
    m_size(size),
    m_deleter(deleter)
    {}
    
    StringPrinterBufferPointer(StringPrinterBufferPointer&& rhs) :
    m_data(rhs.m_data),
    m_size(rhs.m_size),
    m_deleter(rhs.m_deleter)
    {
        rhs.m_data = nullptr;
    }
    
    StringPrinterBufferPointer(const StringPrinterBufferPointer& rhs) :
    m_data(rhs.m_data),
    m_size(rhs.m_size),
    m_deleter(rhs.m_deleter)
    {
        rhs.m_data = nullptr; // this is why m_data has to be mutable
    }
    
    const T*
    GetBytes () const
    {
        return m_data;
    }
    
    const S
    GetSize () const
    {
        return m_size;
    }
    
    ~StringPrinterBufferPointer ()
    {
        if (m_data && m_deleter)
            m_deleter(m_data);
        m_data = nullptr;
    }
    
    StringPrinterBufferPointer&
    operator = (const StringPrinterBufferPointer& rhs)
    {
        if (m_data && m_deleter)
            m_deleter(m_data);
        m_data = rhs.m_data;
        m_size = rhs.m_size;
        m_deleter = rhs.m_deleter;
        rhs.m_data = nullptr;
        return *this;
    }
    
private:
    mutable const T* m_data;
    size_t m_size;
    Deleter m_deleter;
};

// we define this for all values of type but only implement it for those we care about
// that's good because we get linker errors for any unsupported type
template <StringElementType type>
static StringPrinterBufferPointer<>
GetPrintableImpl(uint8_t* buffer, uint8_t* buffer_end, uint8_t*& next);

// mimic isprint() for Unicode codepoints
static bool
isprint(char32_t codepoint)
{
    if (codepoint <= 0x1F || codepoint == 0x7F) // C0
    {
        return false;
    }
    if (codepoint >= 0x80 && codepoint <= 0x9F) // C1
    {
        return false;
    }
    if (codepoint == 0x2028 || codepoint == 0x2029) // line/paragraph separators
    {
        return false;
    }
    if (codepoint == 0x200E || codepoint == 0x200F || (codepoint >= 0x202A && codepoint <= 0x202E)) // bidirectional text control
    {
        return false;
    }
    if (codepoint >= 0xFFF9 && codepoint <= 0xFFFF) // interlinears and generally specials
    {
        return false;
    }
    return true;
}

template <>
StringPrinterBufferPointer<>
GetPrintableImpl<StringElementType::ASCII> (uint8_t* buffer, uint8_t* buffer_end, uint8_t*& next)
{
    StringPrinterBufferPointer<> retval = {nullptr};
    
    switch (*buffer)
    {
        case 0:
            retval = {"\\0",2};
            break;
        case '\a':
            retval = {"\\a",2};
            break;
        case '\b':
            retval = {"\\b",2};
            break;
        case '\f':
            retval = {"\\f",2};
            break;
        case '\n':
            retval = {"\\n",2};
            break;
        case '\r':
            retval = {"\\r",2};
            break;
        case '\t':
            retval = {"\\t",2};
            break;
        case '\v':
            retval = {"\\v",2};
            break;
        case '\"':
            retval = {"\\\"",2};
            break;
        case '\\':
            retval = {"\\\\",2};
            break;
        default:
          if (isprint(*buffer))
              retval = {buffer,1};
          else
          {
              uint8_t* data = new uint8_t[5];
              sprintf((char*)data,"\\x%02x",*buffer);
              retval = {data, 4, [] (const uint8_t* c) {delete[] c;} };
              break;
          }
    }
    
    next = buffer + 1;
    return retval;
}

static char32_t
ConvertUTF8ToCodePoint (unsigned char c0, unsigned char c1)
{
    return (c0-192)*64+(c1-128);
}
static char32_t
ConvertUTF8ToCodePoint (unsigned char c0, unsigned char c1, unsigned char c2)
{
    return (c0-224)*4096+(c1-128)*64+(c2-128);
}
static char32_t
ConvertUTF8ToCodePoint (unsigned char c0, unsigned char c1, unsigned char c2, unsigned char c3)
{
    return (c0-240)*262144+(c2-128)*4096+(c2-128)*64+(c3-128);
}

template <>
StringPrinterBufferPointer<>
GetPrintableImpl<StringElementType::UTF8> (uint8_t* buffer, uint8_t* buffer_end, uint8_t*& next)
{
    StringPrinterBufferPointer<> retval {nullptr};
    
    unsigned utf8_encoded_len = getNumBytesForUTF8(*buffer);
    
    if (1+buffer_end-buffer < utf8_encoded_len)
    {
        // I don't have enough bytes - print whatever I have left
        retval = {buffer,static_cast<size_t>(1+buffer_end-buffer)};
        next = buffer_end+1;
        return retval;
    }
    
    char32_t codepoint = 0;
    switch (utf8_encoded_len)
    {
        case 1:
            // this is just an ASCII byte - ask ASCII
            return GetPrintableImpl<StringElementType::ASCII>(buffer, buffer_end, next);
        case 2:
            codepoint = ConvertUTF8ToCodePoint((unsigned char)*buffer, (unsigned char)*(buffer+1));
            break;
        case 3:
            codepoint = ConvertUTF8ToCodePoint((unsigned char)*buffer, (unsigned char)*(buffer+1), (unsigned char)*(buffer+2));
            break;
        case 4:
            codepoint = ConvertUTF8ToCodePoint((unsigned char)*buffer, (unsigned char)*(buffer+1), (unsigned char)*(buffer+2), (unsigned char)*(buffer+3));
            break;
        default:
            // this is probably some bogus non-character thing
            // just print it as-is and hope to sync up again soon
            retval = {buffer,1};
            next = buffer+1;
            return retval;
    }
    
    if (codepoint)
    {
        switch (codepoint)
        {
            case 0:
                retval = {"\\0",2};
                break;
            case '\a':
                retval = {"\\a",2};
                break;
            case '\b':
                retval = {"\\b",2};
                break;
            case '\f':
                retval = {"\\f",2};
                break;
            case '\n':
                retval = {"\\n",2};
                break;
            case '\r':
                retval = {"\\r",2};
                break;
            case '\t':
                retval = {"\\t",2};
                break;
            case '\v':
                retval = {"\\v",2};
                break;
            case '\"':
                retval = {"\\\"",2};
                break;
            case '\\':
                retval = {"\\\\",2};
                break;
            default:
                if (isprint(codepoint))
                    retval = {buffer,utf8_encoded_len};
                else
                {
                    uint8_t* data = new uint8_t[11];
                    sprintf((char*)data,"\\U%08x",codepoint);
                    retval = { data,10,[] (const uint8_t* c) {delete[] c;} };
                    break;
                }
        }
        
        next = buffer + utf8_encoded_len;
        return retval;
    }
    
    // this should not happen - but just in case.. try to resync at some point
    retval = {buffer,1};
    next = buffer+1;
    return retval;
}

// Given a sequence of bytes, this function returns:
// a sequence of bytes to actually print out + a length
// the following unscanned position of the buffer is in next
static StringPrinterBufferPointer<>
GetPrintable(StringElementType type, uint8_t* buffer, uint8_t* buffer_end, uint8_t*& next)
{
    if (!buffer)
        return {nullptr};
    
    switch (type)
    {
        case StringElementType::ASCII:
            return GetPrintableImpl<StringElementType::ASCII>(buffer, buffer_end, next);
        case StringElementType::UTF8:
            return GetPrintableImpl<StringElementType::UTF8>(buffer, buffer_end, next);
        default:
            return {nullptr};
    }
}

// use this call if you already have an LLDB-side buffer for the data
template<typename SourceDataType>
static bool
DumpUTFBufferToStream (ConversionResult (*ConvertFunction) (const SourceDataType**,
                                                            const SourceDataType*,
                                                            UTF8**,
                                                            UTF8*,
                                                            ConversionFlags),
                       const DataExtractor& data,
                       Stream& stream,
                       char prefix_token,
                       char quote,
                       uint32_t sourceSize,
                       bool escapeNonPrintables)
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
        
        const SourceDataType *data_ptr = (const SourceDataType*)data.GetDataStart();
        const SourceDataType *data_end_ptr = data_ptr + sourceSize;
        
        while (data_ptr < data_end_ptr)
        {
            if (!*data_ptr)
            {
                data_end_ptr = data_ptr;
                break;
            }
            data_ptr++;
        }
        
        data_ptr = (const SourceDataType*)data.GetDataStart();
        
        lldb::DataBufferSP utf8_data_buffer_sp;
        UTF8* utf8_data_ptr = nullptr;
        UTF8* utf8_data_end_ptr = nullptr;
        
        if (ConvertFunction)
        {
            utf8_data_buffer_sp.reset(new DataBufferHeap(4*bufferSPSize,0));
            utf8_data_ptr = (UTF8*)utf8_data_buffer_sp->GetBytes();
            utf8_data_end_ptr = utf8_data_ptr + utf8_data_buffer_sp->GetByteSize();
            ConvertFunction ( &data_ptr, data_end_ptr, &utf8_data_ptr, utf8_data_end_ptr, lenientConversion );
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
        for (;utf8_data_ptr < utf8_data_end_ptr;)
        {
            if (!*utf8_data_ptr)
                break;
            
            if (escapeNonPrintables)
            {
                uint8_t* next_data = nullptr;
                auto printable = GetPrintable(StringElementType::UTF8, utf8_data_ptr, utf8_data_end_ptr, next_data);
                auto printable_bytes = printable.GetBytes();
                auto printable_size = printable.GetSize();
                if (!printable_bytes || !next_data)
                {
                    // GetPrintable() failed on us - print one byte in a desperate resync attempt
                    printable_bytes = utf8_data_ptr;
                    printable_size = 1;
                    next_data = utf8_data_ptr+1;
                }
                for (unsigned c = 0; c < printable_size; c++)
                    stream.Printf("%c", *(printable_bytes+c));
                utf8_data_ptr = (uint8_t*)next_data;
            }
            else
            {
                stream.Printf("%c",*utf8_data_ptr);
                utf8_data_ptr++;
            }
        }
    }
    if (quote != 0)
        stream.Printf("%c",quote);
    return true;
}

lldb_private::formatters::ReadStringAndDumpToStreamOptions::ReadStringAndDumpToStreamOptions (ValueObject& valobj) :
    ReadStringAndDumpToStreamOptions()
{
    SetEscapeNonPrintables(valobj.GetTargetSP()->GetDebugger().GetEscapeNonPrintables());
}

lldb_private::formatters::ReadBufferAndDumpToStreamOptions::ReadBufferAndDumpToStreamOptions (ValueObject& valobj) :
    ReadBufferAndDumpToStreamOptions()
{
    SetEscapeNonPrintables(valobj.GetTargetSP()->GetDebugger().GetEscapeNonPrintables());
}


namespace lldb_private
{

namespace formatters
{

template <>
bool
ReadStringAndDumpToStream<StringElementType::ASCII> (ReadStringAndDumpToStreamOptions options)
{
    assert(options.GetStream() && "need a Stream to print the string to");
    Error my_error;

    ProcessSP process_sp(options.GetProcessSP());

    if (process_sp.get() == nullptr || options.GetLocation() == 0)
        return false;

    size_t size;

    if (options.GetSourceSize() == 0)
        size = process_sp->GetTarget().GetMaximumSizeOfStringSummary();
    else if (!options.GetIgnoreMaxLength())
        size = std::min(options.GetSourceSize(),process_sp->GetTarget().GetMaximumSizeOfStringSummary());
    else
        size = options.GetSourceSize();

    lldb::DataBufferSP buffer_sp(new DataBufferHeap(size,0));

    process_sp->ReadCStringFromMemory(options.GetLocation(), (char*)buffer_sp->GetBytes(), size, my_error);

    if (my_error.Fail())
        return false;

    char prefix_token = options.GetPrefixToken();
    char quote = options.GetQuote();

    if (prefix_token != 0)
        options.GetStream()->Printf("%c%c",prefix_token,quote);
    else if (quote != 0)
        options.GetStream()->Printf("%c",quote);

    uint8_t* data_end = buffer_sp->GetBytes()+buffer_sp->GetByteSize();

    // since we tend to accept partial data (and even partially malformed data)
    // we might end up with no NULL terminator before the end_ptr
    // hence we need to take a slower route and ensure we stay within boundaries
    for (uint8_t* data = buffer_sp->GetBytes(); *data && (data < data_end);)
    {
        if (options.GetEscapeNonPrintables())
        {
            uint8_t* next_data = nullptr;
            auto printable = GetPrintable(StringElementType::ASCII, data, data_end, next_data);
            auto printable_bytes = printable.GetBytes();
            auto printable_size = printable.GetSize();
            if (!printable_bytes || !next_data)
            {
                // GetPrintable() failed on us - print one byte in a desperate resync attempt
                printable_bytes = data;
                printable_size = 1;
                next_data = data+1;
            }
            for (unsigned c = 0; c < printable_size; c++)
                options.GetStream()->Printf("%c", *(printable_bytes+c));
            data = (uint8_t*)next_data;
        }
        else
        {
            options.GetStream()->Printf("%c",*data);
            data++;
        }
    }

    if (quote != 0)
        options.GetStream()->Printf("%c",quote);

    return true;
}

template<typename SourceDataType>
static bool
ReadUTFBufferAndDumpToStream (const ReadStringAndDumpToStreamOptions& options,
                              ConversionResult (*ConvertFunction) (const SourceDataType**,
                                                                   const SourceDataType*,
                                                                   UTF8**,
                                                                   UTF8*,
                                                                   ConversionFlags))
{
    assert(options.GetStream() && "need a Stream to print the string to");

    if (options.GetLocation() == 0 || options.GetLocation() == LLDB_INVALID_ADDRESS)
        return false;

    lldb::ProcessSP process_sp(options.GetProcessSP());

    if (!process_sp)
        return false;

    const int type_width = sizeof(SourceDataType);
    const int origin_encoding = 8 * type_width ;
    if (origin_encoding != 8 && origin_encoding != 16 && origin_encoding != 32)
        return false;
    // if not UTF8, I need a conversion function to return proper UTF8
    if (origin_encoding != 8 && !ConvertFunction)
        return false;

    if (!options.GetStream())
        return false;

    uint32_t sourceSize = options.GetSourceSize();
    bool needs_zero_terminator = options.GetNeedsZeroTermination();

    if (!sourceSize)
    {
        sourceSize = process_sp->GetTarget().GetMaximumSizeOfStringSummary();
        needs_zero_terminator = true;
    }
    else if (!options.GetIgnoreMaxLength())
        sourceSize = std::min(sourceSize,process_sp->GetTarget().GetMaximumSizeOfStringSummary());

    const int bufferSPSize = sourceSize * type_width;

    lldb::DataBufferSP buffer_sp(new DataBufferHeap(bufferSPSize,0));

    if (!buffer_sp->GetBytes())
        return false;

    Error error;
    char *buffer = reinterpret_cast<char *>(buffer_sp->GetBytes());

    if (needs_zero_terminator)
        process_sp->ReadStringFromMemory(options.GetLocation(), buffer, bufferSPSize, error, type_width);
    else
        process_sp->ReadMemoryFromInferior(options.GetLocation(), (char*)buffer_sp->GetBytes(), bufferSPSize, error);

    if (error.Fail())
    {
        options.GetStream()->Printf("unable to read data");
        return true;
    }

    DataExtractor data(buffer_sp, process_sp->GetByteOrder(), process_sp->GetAddressByteSize());

    return DumpUTFBufferToStream(ConvertFunction, data, *options.GetStream(), options.GetPrefixToken(), options.GetQuote(), sourceSize, options.GetEscapeNonPrintables());
}

template <>
bool
ReadStringAndDumpToStream<StringElementType::UTF8> (ReadStringAndDumpToStreamOptions options)
{
    return ReadUTFBufferAndDumpToStream<UTF8>(options,
                                              nullptr);
}

template <>
bool
ReadStringAndDumpToStream<StringElementType::UTF16> (ReadStringAndDumpToStreamOptions options)
{
    return ReadUTFBufferAndDumpToStream<UTF16>(options,
                                               ConvertUTF16toUTF8);
}

template <>
bool
ReadStringAndDumpToStream<StringElementType::UTF32> (ReadStringAndDumpToStreamOptions options)
{
    return ReadUTFBufferAndDumpToStream<UTF32>(options,
                                               ConvertUTF32toUTF8);
}

template <>
bool
ReadBufferAndDumpToStream<StringElementType::UTF8> (ReadBufferAndDumpToStreamOptions options)
{
    assert(options.GetStream() && "need a Stream to print the string to");

    return DumpUTFBufferToStream<UTF8>(nullptr, options.GetData(), *options.GetStream(), options.GetPrefixToken(), options.GetQuote(), options.GetSourceSize(), options.GetEscapeNonPrintables());
}

template <>
bool
ReadBufferAndDumpToStream<StringElementType::ASCII> (ReadBufferAndDumpToStreamOptions options)
{
    // treat ASCII the same as UTF8
    // FIXME: can we optimize ASCII some more?
    return ReadBufferAndDumpToStream<StringElementType::UTF8>(options);
}

template <>
bool
ReadBufferAndDumpToStream<StringElementType::UTF16> (ReadBufferAndDumpToStreamOptions options)
{
    assert(options.GetStream() && "need a Stream to print the string to");

    return DumpUTFBufferToStream(ConvertUTF16toUTF8, options.GetData(), *options.GetStream(), options.GetPrefixToken(), options.GetQuote(), options.GetSourceSize(), options.GetEscapeNonPrintables());
}

template <>
bool
ReadBufferAndDumpToStream<StringElementType::UTF32> (ReadBufferAndDumpToStreamOptions options)
{
    assert(options.GetStream() && "need a Stream to print the string to");

    return DumpUTFBufferToStream(ConvertUTF32toUTF8, options.GetData(), *options.GetStream(), options.GetPrefixToken(), options.GetQuote(), options.GetSourceSize(), options.GetEscapeNonPrintables());
}

} // namespace formatters

} // namespace lldb_private
