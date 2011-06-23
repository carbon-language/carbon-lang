//===-- FormatManager.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_FormatManager_h_
#define lldb_FormatManager_h_

// C Includes

#include <stdint.h>
#include <unistd.h>

// C++ Includes

#ifdef __GNUC__
#include <ext/hash_map>

namespace std
{
    using namespace __gnu_cxx;
}

#else
#include <hash_map>
#endif

#include <map>
#include <stack>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Core/Communication.h"
#include "lldb/Core/InputReaderStack.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/TargetList.h"

namespace lldb_private {

    
class FormatManager
{
public:

    typedef bool(*Callback)(void*, const char*, lldb::Format, bool);

    FormatManager() : 
        m_format_map(FormatMap()), 
        m_format_map_mutex(Mutex::eMutexTypeRecursive) 
    {
    }

    bool
    GetFormatForType (const ConstString &type, lldb::Format& format, bool& cascade);

    void
    AddFormatForType (const ConstString &type, lldb::Format format, bool cascade);

    bool
    DeleteFormatForType (const ConstString &type);

    void
    LoopThroughFormatList (Callback cback, void* param);

    static bool
    GetFormatFromCString (const char *format_cstr,
                          bool partial_match_ok,
                          lldb::Format &format);

    static char
    GetFormatAsFormatChar (lldb::Format format);

    static const char *
    GetFormatAsCString (lldb::Format format);

private:
    struct Entry 
    {
        lldb::Format format;
        bool cascades;
        Entry (lldb::Format f = lldb::eFormatInvalid, bool c = false) : 
            format (f), 
            cascades (c) 
        {
        }
    };
    
    typedef std::map<const char*, Entry> FormatMap;
    typedef FormatMap::iterator FormatIterator;

    FormatMap m_format_map;
    Mutex m_format_map_mutex;
};

} // namespace lldb_private

#endif	// lldb_FormatManager_h_
