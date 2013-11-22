//===-- FormatCache.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_FormatCache_h_
#define lldb_FormatCache_h_

// C Includes
// C++ Includes
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Host/Mutex.h"
#include "lldb/DataFormatters/FormatClasses.h"

namespace lldb_private {
class FormatCache
{
private:
    struct Entry
    {
    private:
        bool m_format_cached : 1;
        bool m_summary_cached : 1;
        bool m_synthetic_cached : 1;
        
        lldb::TypeFormatImplSP m_format_sp;
        lldb::TypeSummaryImplSP m_summary_sp;
        lldb::SyntheticChildrenSP m_synthetic_sp;
    public:
        Entry ();
        Entry (lldb::TypeFormatImplSP);
        Entry (lldb::TypeSummaryImplSP);
        Entry (lldb::SyntheticChildrenSP);
        Entry (lldb::TypeFormatImplSP,lldb::TypeSummaryImplSP,lldb::SyntheticChildrenSP);

        bool
        IsFormatCached ();
        
        bool
        IsSummaryCached ();
        
        bool
        IsSyntheticCached ();
        
        lldb::TypeFormatImplSP
        GetFormat ();
        
        lldb::TypeSummaryImplSP
        GetSummary ();
        
        lldb::SyntheticChildrenSP
        GetSynthetic ();
        
        void
        SetFormat (lldb::TypeFormatImplSP);
        
        void
        SetSummary (lldb::TypeSummaryImplSP);
        
        void
        SetSynthetic (lldb::SyntheticChildrenSP);
    };
    typedef std::map<ConstString,Entry> CacheMap;
    CacheMap m_map;
    Mutex m_mutex;
    
    uint64_t m_cache_hits;
    uint64_t m_cache_misses;
    
    Entry&
    GetEntry (const ConstString& type);
    
public:
    FormatCache ();
    
    bool
    GetFormat (const ConstString& type,lldb::TypeFormatImplSP& format_sp);
    
    bool
    GetSummary (const ConstString& type,lldb::TypeSummaryImplSP& summary_sp);

    bool
    GetSynthetic (const ConstString& type,lldb::SyntheticChildrenSP& synthetic_sp);
    
    void
    SetFormat (const ConstString& type,lldb::TypeFormatImplSP& format_sp);
    
    void
    SetSummary (const ConstString& type,lldb::TypeSummaryImplSP& summary_sp);
    
    void
    SetSynthetic (const ConstString& type,lldb::SyntheticChildrenSP& synthetic_sp);
    
    void
    Clear ();
    
    uint64_t
    GetCacheHits ()
    {
        return m_cache_hits;
    }
    
    uint64_t
    GetCacheMisses ()
    {
        return m_cache_misses;
    }
};
} // namespace lldb_private

#endif	// lldb_FormatCache_h_
