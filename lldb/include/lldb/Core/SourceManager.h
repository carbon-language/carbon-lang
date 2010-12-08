//===-- SourceManager.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SourceManager_h_
#define liblldb_SourceManager_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/FileSpec.h"

namespace lldb_private {

class SourceManager
{
public:
#ifndef SWIG
    class File
    {
    public:
        File (const FileSpec &file_spec);
        ~File();

        size_t
        DisplaySourceLines (uint32_t line,
                            uint32_t context_before,
                            uint32_t context_after,
                            Stream *s);

        uint32_t
        GetLineOffset (uint32_t line);

        bool
        LineIsValid (uint32_t line);

        bool
        FileSpecMatches (const FileSpec &file_spec);

    protected:

        bool
        CalculateLineOffsets (uint32_t line = UINT32_MAX);

        FileSpec m_file_spec;
        TimeValue m_mod_time;   // Keep the modification time that this file data is valid for
        lldb::DataBufferSP m_data_sp;
        typedef std::vector<uint32_t> LineOffsets;
        LineOffsets m_offsets;
    };
#endif


    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    SourceManager();

    ~SourceManager();

    typedef lldb::SharedPtr<File>::Type FileSP;

    FileSP
    GetFile (const FileSpec &file_spec);

    size_t
    DisplaySourceLines (const FileSpec &file,
                        uint32_t line,
                        uint32_t context_before,
                        uint32_t context_after,
                        Stream *s);

    size_t
    DisplaySourceLinesWithLineNumbers (const FileSpec &file,
                                       uint32_t line,
                                       uint32_t context_before,
                                       uint32_t context_after,
                                       const char* current_line_cstr,
                                       Stream *s);

    // This variant uses the last file we visited.
    size_t
    DisplaySourceLinesWithLineNumbersUsingLastFile (uint32_t line,
                                       uint32_t context_before,
                                       uint32_t context_after,
                                       const char* current_line_cstr,
                                       Stream *s);

    size_t
    DisplayMoreWithLineNumbers (Stream *s);

protected:
    //------------------------------------------------------------------
    // Classes that inherit from SourceManager can see and modify these
    //------------------------------------------------------------------
    typedef std::map <FileSpec, FileSP> FileCache;
    FileCache m_file_cache;
    FileSP m_last_file_sp;
    uint32_t m_last_file_line;
    uint32_t m_last_file_context_before;
    uint32_t m_last_file_context_after;
private:
    //------------------------------------------------------------------
    // For SourceManager only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (SourceManager);
};

} // namespace lldb_private

#endif  // liblldb_SourceManager_h_
