//===-- StringPrinter.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StringPrinter_h_
#define liblldb_StringPrinter_h_

#include "lldb/lldb-forward.h"

#include "lldb/Core/DataExtractor.h"

namespace lldb_private {
    namespace formatters
    {
        
        enum class StringElementType {
            ASCII,
            UTF8,
            UTF16,
            UTF32
        };
        
        class ReadStringAndDumpToStreamOptions
        {
        public:
            
            ReadStringAndDumpToStreamOptions () :
            m_location(0),
            m_process_sp(),
            m_stream(NULL),
            m_prefix_token(0),
            m_quote('"'),
            m_source_size(0),
            m_needs_zero_termination(true),
            m_escape_non_printables(true),
            m_ignore_max_length(false)
            {
            }
            
            ReadStringAndDumpToStreamOptions (ValueObject& valobj);
            
            ReadStringAndDumpToStreamOptions&
            SetLocation (uint64_t l)
            {
                m_location = l;
                return *this;
            }
            
            uint64_t
            GetLocation () const
            {
                return m_location;
            }
            
            ReadStringAndDumpToStreamOptions&
            SetProcessSP (lldb::ProcessSP p)
            {
                m_process_sp = p;
                return *this;
            }
            
            lldb::ProcessSP
            GetProcessSP () const
            {
                return m_process_sp;
            }
            
            ReadStringAndDumpToStreamOptions&
            SetStream (Stream* s)
            {
                m_stream = s;
                return *this;
            }
            
            Stream*
            GetStream () const
            {
                return m_stream;
            }
            
            ReadStringAndDumpToStreamOptions&
            SetPrefixToken (char p)
            {
                m_prefix_token = p;
                return *this;
            }
            
            char
            GetPrefixToken () const
            {
                return m_prefix_token;
            }
            
            ReadStringAndDumpToStreamOptions&
            SetQuote (char q)
            {
                m_quote = q;
                return *this;
            }
            
            char
            GetQuote () const
            {
                return m_quote;
            }
            
            ReadStringAndDumpToStreamOptions&
            SetSourceSize (uint32_t s)
            {
                m_source_size = s;
                return *this;
            }
            
            uint32_t
            GetSourceSize () const
            {
                return m_source_size;
            }
            
            ReadStringAndDumpToStreamOptions&
            SetNeedsZeroTermination (bool z)
            {
                m_needs_zero_termination = z;
                return *this;
            }
            
            bool
            GetNeedsZeroTermination () const
            {
                return m_needs_zero_termination;
            }
            
            ReadStringAndDumpToStreamOptions&
            SetEscapeNonPrintables (bool e)
            {
                m_escape_non_printables = e;
                return *this;
            }
            
            bool
            GetEscapeNonPrintables () const
            {
                return m_escape_non_printables;
            }
            
            ReadStringAndDumpToStreamOptions&
            SetIgnoreMaxLength (bool e)
            {
                m_ignore_max_length = e;
                return *this;
            }
            
            bool
            GetIgnoreMaxLength () const
            {
                return m_ignore_max_length;
            }
            
        private:
            uint64_t m_location;
            lldb::ProcessSP m_process_sp;
            Stream* m_stream;
            char m_prefix_token;
            char m_quote;
            uint32_t m_source_size;
            bool m_needs_zero_termination;
            bool m_escape_non_printables;
            bool m_ignore_max_length;
        };
        
        class ReadBufferAndDumpToStreamOptions
        {
        public:
            
            ReadBufferAndDumpToStreamOptions () :
            m_data(),
            m_stream(NULL),
            m_prefix_token(0),
            m_quote('"'),
            m_source_size(0),
            m_escape_non_printables(true)
            {
            }
            
            ReadBufferAndDumpToStreamOptions (ValueObject& valobj);
            
            ReadBufferAndDumpToStreamOptions&
            SetData (DataExtractor d)
            {
                m_data = d;
                return *this;
            }
            
            lldb_private::DataExtractor
            GetData () const
            {
                return m_data;
            }
            
            ReadBufferAndDumpToStreamOptions&
            SetStream (Stream* s)
            {
                m_stream = s;
                return *this;
            }
            
            Stream*
            GetStream () const
            {
                return m_stream;
            }
            
            ReadBufferAndDumpToStreamOptions&
            SetPrefixToken (char p)
            {
                m_prefix_token = p;
                return *this;
            }
            
            char
            GetPrefixToken () const
            {
                return m_prefix_token;
            }
            
            ReadBufferAndDumpToStreamOptions&
            SetQuote (char q)
            {
                m_quote = q;
                return *this;
            }
            
            char
            GetQuote () const
            {
                return m_quote;
            }
            
            ReadBufferAndDumpToStreamOptions&
            SetSourceSize (uint32_t s)
            {
                m_source_size = s;
                return *this;
            }
            
            uint32_t
            GetSourceSize () const
            {
                return m_source_size;
            }
            
            ReadBufferAndDumpToStreamOptions&
            SetEscapeNonPrintables (bool e)
            {
                m_escape_non_printables = e;
                return *this;
            }
            
            bool
            GetEscapeNonPrintables () const
            {
                return m_escape_non_printables;
            }
            
        private:
            DataExtractor m_data;
            Stream* m_stream;
            char m_prefix_token;
            char m_quote;
            uint32_t m_source_size;
            bool m_escape_non_printables;
        };
        
        template <StringElementType element_type>
        bool
        ReadStringAndDumpToStream (ReadStringAndDumpToStreamOptions options);
        
        template <StringElementType element_type>
        bool
        ReadBufferAndDumpToStream (ReadBufferAndDumpToStreamOptions options);
        
    } // namespace formatters
} // namespace lldb_private

#endif // liblldb_StringPrinter_h_
