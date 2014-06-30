//===-- MemoryRegionInfo.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_MemoryRegionInfo_h
#define lldb_MemoryRegionInfo_h

#include "lldb/Core/RangeMap.h"
#include "lldb/Utility/Range.h"

namespace lldb_private
{
    class MemoryRegionInfo
    {
    public:
        typedef Range<lldb::addr_t, lldb::addr_t> RangeType;
        
        enum OptionalBool {
            eDontKnow  = -1,
            eNo         = 0,
            eYes        = 1
        };
        
        MemoryRegionInfo () :
        m_range (),
        m_read (eDontKnow),
        m_write (eDontKnow),
        m_execute (eDontKnow)
        {
        }
        
        ~MemoryRegionInfo ()
        {
        }
        
        RangeType &
        GetRange()
        {
            return m_range;
        }
        
        void
        Clear()
        {
            m_range.Clear();
            m_read = m_write = m_execute = eDontKnow;
        }
        
        const RangeType &
        GetRange() const
        {
            return m_range;
        }
        
        OptionalBool
        GetReadable () const
        {
            return m_read;
        }
        
        OptionalBool
        GetWritable () const
        {
            return m_write;
        }
        
        OptionalBool
        GetExecutable () const
        {
            return m_execute;
        }
        
        void
        SetReadable (OptionalBool val)
        {
            m_read = val;
        }
        
        void
        SetWritable (OptionalBool val)
        {
            m_write = val;
        }
        
        void
        SetExecutable (OptionalBool val)
        {
            m_execute = val;
        }
        
    protected:
        RangeType m_range;
        OptionalBool m_read;
        OptionalBool m_write;
        OptionalBool m_execute;
    };
}

#endif // #ifndef lldb_MemoryRegionInfo_h
