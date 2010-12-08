//===-- Section.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Section_h_
#define liblldb_Section_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/ModuleChild.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/VMRange.h"
#include <limits.h>

namespace lldb_private {

class SectionList
{
public:
    typedef std::vector<lldb::SectionSP>  collection;
    typedef collection::iterator        iterator;
    typedef collection::const_iterator  const_iterator;

    SectionList();

    virtual
    ~SectionList();

    uint32_t
    AddSection (lldb::SectionSP& sect_sp);

    uint32_t
    AddUniqueSection (lldb::SectionSP& sect_sp);

    uint32_t
    FindSectionIndex (const Section* sect);

    bool
    ContainsSection(lldb::user_id_t sect_id) const;

    void
    Dump (Stream *s, Target *target, bool show_header, uint32_t depth) const;

    lldb::SectionSP
    FindSectionByName (const ConstString &section_dstr) const;

    lldb::SectionSP
    FindSectionByID (lldb::user_id_t sect_id) const;

    lldb::SectionSP
    FindSectionByType (lldb::SectionType sect_type, bool check_children, uint32_t start_idx = 0) const;

    lldb::SectionSP
    GetSharedPointer (const Section *section, bool check_children) const;

    lldb::SectionSP
    FindSectionContainingFileAddress (lldb::addr_t addr, uint32_t depth = UINT32_MAX) const;

    lldb::SectionSP
    FindSectionContainingLinkedFileAddress (lldb::addr_t vm_addr, uint32_t depth) const;

    bool
    GetSectionData (const DataExtractor& module_data, DataExtractor& section_data) const;

    // Get the number of sections in this list only
    size_t
    GetSize () const
    {
        return m_sections.size();
    }

    // Get the number of sections in this list, and any contained child sections
    size_t
    GetNumSections (uint32_t depth) const;

    bool
    ReplaceSection (lldb::user_id_t sect_id, lldb::SectionSP& sect_sp, uint32_t depth = UINT32_MAX);

    lldb::SectionSP
    GetSectionAtIndex (uint32_t idx) const;

    size_t
    Slide (lldb::addr_t slide_amount, bool slide_children);

protected:
    collection  m_sections;
};


class Section :
    public ModuleChild,
    public UserID,
    public Flags
{
public:
    Section (
        Section *parent,    // NULL for top level sections, non-NULL for child sections
        Module* module,
        lldb::user_id_t sect_id,
        const ConstString &name,
        lldb::SectionType sect_type,
        lldb::addr_t file_vm_addr,
        lldb::addr_t vm_size,
        uint64_t file_offset,
        uint64_t file_size,
        uint32_t flags);

    ~Section ();

    static int
    Compare (const Section& a, const Section& b);

    // Get a valid shared pointer to this section object
    lldb::SectionSP
    GetSharedPointer() const;

    bool
    ContainsFileAddress (lldb::addr_t vm_addr) const;

    SectionList&
    GetChildren ()
    {
        return m_children;
    }

    const SectionList&
    GetChildren () const
    {
        return m_children;
    }

    void
    Dump (Stream *s, Target *target, uint32_t depth) const;

    void
    DumpName (Stream *s) const;

    lldb::addr_t
    GetLoadBaseAddress (Target *target) const;

    bool
    ResolveContainedAddress (lldb::addr_t offset, Address &so_addr) const;

    uint64_t
    GetFileOffset () const
    {
        return m_file_offset;
    }

    void
    SetFileOffset (uint64_t file_offset) 
    {
        m_file_offset = file_offset;
    }

    uint64_t
    GetFileSize () const
    {
        return m_file_size;
    }

    void
    SetFileSize (uint64_t file_size)
    {
        m_file_size = file_size;
    }

    lldb::addr_t
    GetFileAddress () const;

    lldb::addr_t
    GetOffset () const
    {
        // This section has a parent which means m_file_addr is an offset.
        if (m_parent)
            return m_file_addr;

        // This section has no parent, so there is no offset to be had
        return 0;
    }


    lldb::addr_t
    GetByteSize () const
    {
        return m_byte_size;
    }
    
    void
    SetByteSize (lldb::addr_t byte_size)
    {
        m_byte_size = byte_size;
    }
    
    size_t
    GetSectionDataFromImage (const DataExtractor& image_data, DataExtractor& section_data) const;

    bool
    IsFake() const
    {
        return m_fake;
    }

    void
    SetIsFake(bool fake)
    {
        m_fake = fake;
    }

    bool
    IsDescendant (const Section *section);

    size_t
    MemoryMapSectionDataFromObjectFile (const ObjectFile* file, DataExtractor& section_data) const;

    size_t
    ReadSectionDataFromObjectFile (const ObjectFile* objfile, off_t section_offset, void *dst, size_t dst_len) const;

    size_t
    ReadSectionDataFromObjectFile (const ObjectFile* file, DataExtractor& section_data) const;

    ConstString&
    GetName ();

    const ConstString&
    GetName () const;

    bool
    Slide (lldb::addr_t slide_amount, bool slide_children);

    void
    SetLinkedLocation (const Section *linked_section, uint64_t linked_offset);

    bool
    ContainsLinkedFileAddress (lldb::addr_t vm_addr) const;

    const Section *
    GetLinkedSection () const
    {
        return m_linked_section;
    }

    uint64_t
    GetLinkedOffset () const
    {
        return m_linked_offset;
    }

    lldb::addr_t
    GetLinkedFileAddress () const;

    lldb::SectionType
    GetType () const
    {
        return m_type;
    }

protected:

    Section *       m_parent;           // Parent section or NULL if no parent.
    ConstString     m_name;             // Name of this section
    lldb::SectionType m_type;           // The type of this section
    lldb::addr_t    m_file_addr;        // The absolute file virtual address range of this section if m_parent == NULL,
                                        // offset from parent file virtual address if m_parent != NULL
    lldb::addr_t    m_byte_size;        // Size in bytes that this section will occupy in memory at runtime
    uint64_t        m_file_offset;      // Object file offset (if any)
    uint64_t        m_file_size;        // Object file size (can be smaller than m_byte_size for zero filled sections...)
    SectionList     m_children;         // Child sections
    bool            m_fake;             // If true, then this section only can contain the address if one of its
                                        // children contains an address. This allows for gaps between the children
                                        // that are contained in the address range for this section, but do not produce
                                        // hits unless the children contain the address.
    const Section * m_linked_section;
    uint64_t        m_linked_offset;
private:
    DISALLOW_COPY_AND_ASSIGN (Section);
};


} // namespace lldb_private

#endif  // liblldb_Section_h_
