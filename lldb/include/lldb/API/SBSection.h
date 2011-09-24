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

#ifndef SWIG
    const lldb::SBSection &
    operator = (const lldb::SBSection &rhs);
#endif
    bool
    IsValid () const;

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
    GetSectionData (uint64_t offset = 0,
                    uint64_t size = UINT64_MAX);

    SectionType
    GetSectionType ();

#ifndef SWIG
    bool
    operator == (const lldb::SBSection &rhs);

    bool
    operator != (const lldb::SBSection &rhs);

#endif

    bool
    GetDescription (lldb::SBStream &description);
    
private:

#ifndef SWIG
    friend class SBAddress;
    friend class SBModule;
    friend class SBTarget;
    
    SBSection (const lldb_private::Section *section);

    const lldb_private::Section *
    GetSection();
    
    void
    SetSection (const lldb_private::Section *section);
#endif
    
    std::auto_ptr<lldb_private::SectionImpl> m_opaque_ap;
};


} // namespace lldb

#endif // LLDB_SBSection_h_
