//===-- SectionLoadList.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SectionLoadList_h_
#define liblldb_SectionLoadList_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-include.h"
#include "lldb/Core/ThreadSafeSTLMap.h"

namespace lldb_private {

class SectionLoadList
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    SectionLoadList () :
        m_section_load_info ()
    {
    }

    ~SectionLoadList()
    {
    }

    bool
    IsEmpty() const;

    void
    Clear ();

    lldb::addr_t
    GetSectionLoadAddress (const Section *section) const;

    bool
    ResolveLoadAddress (lldb::addr_t load_addr, Address &so_addr) const;

    bool
    SetSectionLoadAddress (const Section *section, lldb::addr_t load_addr);

    // The old load address should be specified when unloading to ensure we get
    // the correct instance of the section as a shared library could be loaded
    // at more than one location.
    bool
    SetSectionUnloaded (const Section *section, lldb::addr_t load_addr);

    // Unload all instances of a section. This function can be used on systems
    // that don't support multiple copies of the same shared library to be
    // loaded at the same time.
    size_t
    SetSectionUnloaded (const Section *section);


protected:
    typedef ThreadSafeSTLMap<lldb::addr_t, const Section *> collection;
    collection m_section_load_info;    ///< A mapping of all currently loaded sections.

private:
    DISALLOW_COPY_AND_ASSIGN (SectionLoadList);
};

} // namespace lldb_private

#endif  // liblldb_SectionLoadList_h_
