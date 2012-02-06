//===-- SBSection.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSection_h_
#define LLDB_SBSection_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBData.h"

namespace lldb {

class SBSection
{
public:

    SBSection ();

    SBSection (const lldb::SBSection &rhs);

    ~SBSection ();

    const lldb::SBSection &
    operator = (const lldb::SBSection &rhs);

    bool
    IsValid () const;

    const char *
    GetName ();
    
    lldb::SBSection
    FindSubSection (const char *sect_name);

    size_t
    GetNumSubSections ();

    lldb::SBSection
    GetSubSectionAtIndex (size_t idx);

    lldb::addr_t
    GetFileAddress ();

    lldb::addr_t
    GetByteSize ();

    uint64_t
    GetFileOffset ();

    uint64_t
    GetFileByteSize ();
    
    lldb::SBData
    GetSectionData ();

    lldb::SBData
    GetSectionData (uint64_t offset,
                    uint64_t size);
    
    SectionType
    GetSectionType ();

    bool
    operator == (const lldb::SBSection &rhs);

    bool
    operator != (const lldb::SBSection &rhs);

    bool
    GetDescription (lldb::SBStream &description);
    
private:

    friend class SBAddress;
    friend class SBModule;
    friend class SBTarget;
    
    SBSection (const lldb_private::Section *section);

    const lldb_private::Section *
    GetSection();
    
    void
    SetSection (const lldb_private::Section *section);
    
    std::auto_ptr<lldb_private::SectionImpl> m_opaque_ap;
};


} // namespace lldb

#endif // LLDB_SBSection_h_
