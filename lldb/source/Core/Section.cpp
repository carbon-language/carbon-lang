//===-- Section.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Section.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

Section::Section
(
    Section *parent,
    Module* module,
    user_id_t sect_id,
    const ConstString &name,
    SectionType sect_type,
    addr_t file_addr,
    addr_t byte_size,
    uint64_t file_offset,
    uint64_t file_size,
    uint32_t flags
) :
    ModuleChild     (module),
    UserID          (sect_id),
    Flags           (flags),
    m_parent        (parent),
    m_name          (name),
    m_type          (sect_type),
    m_file_addr     (file_addr),
    m_byte_size     (byte_size),
    m_file_offset   (file_offset),
    m_file_size     (file_size),
    m_children      (),
    m_fake          (false),
    m_linked_section(NULL),
    m_linked_offset (0)
{
}

Section::~Section()
{
}


// Get a valid shared pointer to this section object
SectionSP
Section::GetSharedPointer() const
{
    SectionSP this_sp;
    if (m_parent)
        this_sp = m_parent->GetChildren().GetSharedPointer (this, false);
    else
    {
        ObjectFile *objfile = m_module->GetObjectFile();
        if (objfile)
        {
            SectionList *section_list = objfile->GetSectionList();
            if (section_list)
                this_sp = section_list->GetSharedPointer (this, false);
        }
    }
    return this_sp;
}



ConstString&
Section::GetName()
{
    if (m_linked_section)
        return const_cast<Section *>(m_linked_section)->GetName();
    return m_name;
}

const ConstString&
Section::GetName() const
{
    if (m_linked_section)
        return m_linked_section->GetName();
    return m_name;
}

addr_t
Section::GetFileAddress () const
{
    if (m_parent)
    {
        // This section has a parent which means m_file_addr is an offset into
        // the parent section, so the file address for this section is the file
        // address of the parent plus the offset
        return m_parent->GetFileAddress() + m_file_addr;
    }
    // This section has no parent, so m_file_addr is the file base address
    return m_file_addr;
}

addr_t
Section::GetLinkedFileAddress () const
{
    if (m_linked_section)
        return m_linked_section->GetFileAddress() + m_linked_offset;
    return LLDB_INVALID_ADDRESS;
}


addr_t
Section::GetLoadBaseAddress (Target *target) const
{
    addr_t load_base_addr = LLDB_INVALID_ADDRESS;
    if (m_linked_section)
    {
        load_base_addr = m_linked_section->GetLoadBaseAddress(target) + m_linked_offset;
    }
    else
    if (m_parent)
    {
        load_base_addr = m_parent->GetLoadBaseAddress (target);
        if (load_base_addr != LLDB_INVALID_ADDRESS)
            load_base_addr += GetOffset();
    }
    else
    {
        load_base_addr = target->GetSectionLoadList().GetSectionLoadAddress (this);
    }

    return load_base_addr;
}

bool
Section::ResolveContainedAddress (addr_t offset, Address &so_addr) const
{
    const uint32_t num_children = m_children.GetSize();
    if (num_children > 0)
    {
        for (uint32_t i=0; i<num_children; i++)
        {
            Section* child_section = m_children.GetSectionAtIndex (i).get();

            addr_t child_offset = child_section->GetOffset();
            if (child_offset <= offset && offset - child_offset < child_section->GetByteSize())
                return child_section->ResolveContainedAddress (offset - child_offset, so_addr);
        }
    }
    if (m_linked_section)
    {
        so_addr.SetOffset(m_linked_offset + offset);
        so_addr.SetSection(m_linked_section);
    }
    else
    {
        so_addr.SetOffset(offset);
        so_addr.SetSection(this);
    }
    return true;
}

bool
Section::ContainsFileAddress (addr_t vm_addr) const
{
    const addr_t file_addr = GetFileAddress();
    if (file_addr != LLDB_INVALID_ADDRESS)
    {
        if (file_addr <= vm_addr)
        {
            const addr_t offset = vm_addr - file_addr;
            return offset < GetByteSize();
        }
    }
    return false;
}

bool
Section::ContainsLinkedFileAddress (addr_t vm_addr) const
{
    const addr_t linked_file_addr = GetLinkedFileAddress();
    if (linked_file_addr != LLDB_INVALID_ADDRESS)
    {
        if (linked_file_addr <= vm_addr)
        {
            const addr_t offset = vm_addr - linked_file_addr;
            return offset < GetByteSize();
        }
    }
    return false;
}

int
Section::Compare (const Section& a, const Section& b)
{
    if (&a == &b)
        return 0;

    const Module* a_module = a.GetModule();
    const Module* b_module = b.GetModule();
    if (a_module == b_module)
    {
        user_id_t a_sect_uid = a.GetID();
        user_id_t b_sect_uid = b.GetID();
        if (a_sect_uid < b_sect_uid)
            return -1;
        if (a_sect_uid > b_sect_uid)
            return 1;
        return 0;
    }
    else
    {
        // The modules are different, just compare the module pointers
        if (a_module < b_module)
            return -1;
        else
            return 1;   // We already know the modules aren't equal
    }
}


void
Section::Dump (Stream *s, Target *target) const
{
//    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    s->Printf("0x%8.8x %-14s ", GetID(), GetSectionTypeAsCString (m_type));
    bool resolved = true;
    addr_t addr = LLDB_INVALID_ADDRESS;

    if (GetByteSize() == 0)
        s->Printf("%39s", "");
    else
    {
        if (target && m_linked_section == NULL)
            addr = GetLoadBaseAddress (target);

        if (addr == LLDB_INVALID_ADDRESS)
        {
            if (target)
                resolved = false;
            addr = GetFileAddress();
        }

        VMRange range(addr, addr + m_byte_size);
        range.Dump (s, 0);
    }

    s->Printf("%c 0x%8.8llx 0x%8.8llx 0x%8.8x ", resolved ? ' ' : '*', m_file_offset, m_file_size, Get());

    DumpName (s);

    s->EOL();

    if (m_linked_section)
    {
        addr = LLDB_INVALID_ADDRESS;
        resolved = true;
        if (target)
        {
            addr = m_linked_section->GetLoadBaseAddress(target);
            if (addr != LLDB_INVALID_ADDRESS)
                addr += m_linked_offset;
        }

        if (addr == LLDB_INVALID_ADDRESS)
        {
            if (target)
                resolved = false;
            addr = m_linked_section->GetFileAddress() + m_linked_offset;
        }

        int indent = 26 + s->GetIndentLevel();
        s->Printf("%*.*s", indent, indent, "");
        VMRange linked_range(addr, addr + m_byte_size);
        linked_range.Dump (s, 0);
        indent = 3 * (sizeof(uint32_t) * 2 + 2 + 1) + 1;
        s->Printf("%c%*.*s", resolved ? ' ' : '*', indent, indent, "");

        m_linked_section->DumpName(s);
        s->Printf(" + 0x%llx\n", m_linked_offset);
    }

    m_children.Dump(s, target, false);
}

void
Section::DumpName (Stream *s) const
{
    if (m_parent == NULL)
    {
        // The top most section prints the module basename
        const char *module_basename = m_module->GetFileSpec().GetFilename().AsCString();
        if (module_basename && module_basename[0])
            s->Printf("%s.", module_basename);
    }
    else
    {
        m_parent->DumpName (s);
        s->PutChar('.');
    }
    m_name.Dump(s);
}

//----------------------------------------------------------------------
// Get the section data from a complete contiguous copy of the
// entire executable image.
//----------------------------------------------------------------------
size_t
Section::GetSectionDataFromImage (const DataExtractor& image_data, DataExtractor& section_data) const
{
    size_t file_size = GetByteSize();
    if (file_size > 0)
    {
        off_t file_offset = GetFileOffset();
        if (section_data.SetData (image_data, file_offset, file_size) == file_size)
            return true;
    }
    return false;
}

size_t
Section::ReadSectionDataFromObjectFile (const ObjectFile* objfile, off_t section_offset, void *dst, size_t dst_len) const
{
    if (objfile && dst && dst_len)
    {
        const FileSpec& file = objfile->GetFileSpec();

        if (file)
        {
            off_t section_file_offset = GetFileOffset() + objfile->GetOffset() + section_offset;        
            return file.ReadFileContents (section_file_offset, dst, dst_len);
        }
    }
    return 0;
}

//----------------------------------------------------------------------
// Get the section data the file on disk
//----------------------------------------------------------------------
size_t
Section::ReadSectionDataFromObjectFile(const ObjectFile* objfile, DataExtractor& section_data) const
{
    if (objfile == NULL)
        return 0;

    const FileSpec& file = objfile->GetFileSpec();

    if (file)
    {
        size_t section_file_size = GetByteSize();
        if (section_file_size > 0)
        {
            off_t section_file_offset = GetFileOffset() + objfile->GetOffset();
            DataBufferSP section_data_sp(file.ReadFileContents(section_file_offset, section_file_size));

            section_data.SetByteOrder(objfile->GetByteOrder());
            section_data.SetAddressByteSize(objfile->GetAddressByteSize());
            return section_data.SetData (section_data_sp);
        }
    }
    return 0;
}

size_t
Section::MemoryMapSectionDataFromObjectFile(const ObjectFile* objfile, DataExtractor& section_data) const
{
    if (objfile == NULL)
        return 0;

    const FileSpec& file = objfile->GetFileSpec();

    if (file)
    {
        size_t section_file_size = GetFileSize();
        if (section_file_size > 0)
        {
            off_t section_file_offset = GetFileOffset() + objfile->GetOffset();
            DataBufferSP section_data_sp(file.MemoryMapFileContents(section_file_offset, section_file_size));
            section_data.SetByteOrder(objfile->GetByteOrder());
            section_data.SetAddressByteSize(objfile->GetAddressByteSize());
            return section_data.SetData (section_data_sp);
        }
    }
    return 0;
}

bool
Section::IsDescendant (const Section *section)
{
    if (this == section)
        return true;
    if (m_parent)
        return m_parent->IsDescendant (section);
    return false;
}

bool
Section::Slide (addr_t slide_amount, bool slide_children)
{
    if (m_file_addr != LLDB_INVALID_ADDRESS)
    {
        if (slide_amount == 0)
            return true;

        m_file_addr += slide_amount;

        if (slide_children)
            m_children.Slide (slide_amount, slide_children);

        return true;
    }
    return false;
}

void
Section::SetLinkedLocation (const Section *linked_section, uint64_t linked_offset)
{
    if (linked_section)
        m_module = linked_section->GetModule();
    m_linked_section = linked_section;
    m_linked_offset  = linked_offset;
}

#pragma mark SectionList

SectionList::SectionList () :
    m_sections()
{
}


SectionList::~SectionList ()
{
}

uint32_t
SectionList::AddSection (SectionSP& sect_sp)
{
    uint32_t section_index = m_sections.size();
    m_sections.push_back(sect_sp);
    return section_index;
}

uint32_t
SectionList::FindSectionIndex (const Section* sect)
{
    iterator sect_iter;
    iterator begin = m_sections.begin();
    iterator end = m_sections.end();
    for (sect_iter = begin; sect_iter != end; ++sect_iter)
    {
        if (sect_iter->get() == sect)
        {
            // The secton was already in this section list
            return std::distance (begin, sect_iter);
        }
    }
    return UINT32_MAX;
}

uint32_t
SectionList::AddUniqueSection (SectionSP& sect_sp)
{
    uint32_t sect_idx = FindSectionIndex (sect_sp.get());
    if (sect_idx == UINT32_MAX)
        sect_idx = AddSection (sect_sp);
    return sect_idx;
}


bool
SectionList::ReplaceSection (user_id_t sect_id, SectionSP& sect_sp, uint32_t depth)
{
    iterator sect_iter, end = m_sections.end();
    for (sect_iter = m_sections.begin(); sect_iter != end; ++sect_iter)
    {
        if ((*sect_iter)->GetID() == sect_id)
        {
            *sect_iter = sect_sp;
            return true;
        }
        else if (depth > 0)
        {
            if ((*sect_iter)->GetChildren().ReplaceSection(sect_id, sect_sp, depth - 1))
                return true;
        }
    }
    return false;
}


size_t
SectionList::GetNumSections (uint32_t depth) const
{
    size_t count = m_sections.size();
    if (depth > 0)
    {
        const_iterator sect_iter, end = m_sections.end();
        for (sect_iter = m_sections.begin(); sect_iter != end; ++sect_iter)
        {
            count += (*sect_iter)->GetChildren().GetNumSections(depth - 1);
        }
    }
    return count;
}

SectionSP
SectionList::GetSectionAtIndex (uint32_t idx) const
{
    SectionSP sect_sp;
    if (idx < m_sections.size())
        sect_sp = m_sections[idx];
    return sect_sp;
}

SectionSP
SectionList::FindSectionByName (const ConstString &section_dstr) const
{
    SectionSP sect_sp;
    // Check if we have a valid section string
    if (section_dstr)
    {
        const_iterator sect_iter;
        const_iterator end = m_sections.end();
        for (sect_iter = m_sections.begin(); sect_iter != end && sect_sp.get() == NULL; ++sect_iter)
        {
            if ((*sect_iter)->GetName() == section_dstr)
            {
                sect_sp = *sect_iter;
            }
            else
            {
                sect_sp = (*sect_iter)->GetChildren().FindSectionByName(section_dstr);
            }
        }
    }
    return sect_sp;
}

SectionSP
SectionList::FindSectionByID (user_id_t sect_id) const
{
    SectionSP sect_sp;
    if (sect_id)
    {
        const_iterator sect_iter;
        const_iterator end = m_sections.end();
        for (sect_iter = m_sections.begin(); sect_iter != end && sect_sp.get() == NULL; ++sect_iter)
        {
            if ((*sect_iter)->GetID() == sect_id)
            {
                sect_sp = *sect_iter;
                break;
            }
            else
            {
                sect_sp = (*sect_iter)->GetChildren().FindSectionByID (sect_id);
            }
        }
    }
    return sect_sp;
}


SectionSP
SectionList::FindSectionByType (lldb::SectionType sect_type, bool check_children, uint32_t start_idx) const
{
    SectionSP sect_sp;
    uint32_t num_sections = m_sections.size();
    for (uint32_t idx = start_idx; idx < num_sections; ++idx)
    {
        if (m_sections[idx]->GetType() == sect_type)
        {
            sect_sp = m_sections[idx];
            break;
        }
        else if (check_children)
        {
            sect_sp = m_sections[idx]->GetChildren().FindSectionByType (sect_type, check_children, 0);
            if (sect_sp)
                break;
        }
    }
    return sect_sp;
}

SectionSP
SectionList::GetSharedPointer (const Section *section, bool check_children) const
{
    SectionSP sect_sp;
    if (section)
    {
        const_iterator sect_iter;
        const_iterator end = m_sections.end();
        for (sect_iter = m_sections.begin(); sect_iter != end && sect_sp.get() == NULL; ++sect_iter)
        {
            if (sect_iter->get() == section)
            {
                sect_sp = *sect_iter;
                break;
            }
            else if (check_children)
            {
                sect_sp = (*sect_iter)->GetChildren().GetSharedPointer (section, true);
            }
        }
    }
    return sect_sp;
}



SectionSP
SectionList::FindSectionContainingFileAddress (addr_t vm_addr, uint32_t depth) const
{
    SectionSP sect_sp;
    const_iterator sect_iter;
    const_iterator end = m_sections.end();
    for (sect_iter = m_sections.begin(); sect_iter != end && sect_sp.get() == NULL; ++sect_iter)
    {
        Section *sect = sect_iter->get();
        if (sect->ContainsFileAddress (vm_addr))
        {
            // The file address is in this section. We need to make sure one of our child
            // sections doesn't contain this address as well as obeying the depth limit
            // that was passed in.
            if (depth > 0)
                sect_sp = sect->GetChildren().FindSectionContainingFileAddress(vm_addr, depth - 1);

            if (sect_sp.get() == NULL && !sect->IsFake())
                sect_sp = *sect_iter;
        }
    }
    return sect_sp;
}


SectionSP
SectionList::FindSectionContainingLinkedFileAddress (addr_t vm_addr, uint32_t depth) const
{
    SectionSP sect_sp;
    const_iterator sect_iter;
    const_iterator end = m_sections.end();
    for (sect_iter = m_sections.begin(); sect_iter != end && sect_sp.get() == NULL; ++sect_iter)
    {
        Section *sect = sect_iter->get();
        if (sect->ContainsLinkedFileAddress (vm_addr))
        {
            sect_sp = *sect_iter;
        }
        else if (depth > 0)
        {
            sect_sp = sect->GetChildren().FindSectionContainingLinkedFileAddress (vm_addr, depth - 1);
        }
    }
    return sect_sp;
}

bool
SectionList::ContainsSection(user_id_t sect_id) const
{
    return FindSectionByID (sect_id).get() != NULL;
}

void
SectionList::Dump (Stream *s, Target *target, bool show_header) const
{
    bool target_has_loaded_sections = target && !target->GetSectionLoadList().IsEmpty();
    if (show_header && !m_sections.empty())
    {
//        s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
//        s->Indent();
//        s->PutCString(  "SectionList\n");
//        s->IndentMore();
//        s->Printf("%*s", 2*(sizeof(void *) + 2), "");
        s->Indent();
        s->Printf("SectID     Type           %s Address                             File Off.  File Size  Flags      Section Name\n", target_has_loaded_sections ? "Load" : "File");
//        s->Printf("%*s", 2*(sizeof(void *) + 2), "");
        s->Indent();
        s->PutCString("---------- -------------- ---------------------------------------  ---------- ---------- ---------- ----------------------------\n");
    }


    const_iterator sect_iter;
    const_iterator end = m_sections.end();
    for (sect_iter = m_sections.begin(); sect_iter != end; ++sect_iter)
    {
        (*sect_iter)->Dump(s, target_has_loaded_sections ? target : NULL);
    }

    if (show_header && !m_sections.empty())
        s->IndentLess();

}

size_t
SectionList::Slide (addr_t slide_amount, bool slide_children)
{
    size_t count = 0;
    const_iterator pos, end = m_sections.end();
    for (pos = m_sections.begin(); pos != end; ++pos)
    {
        if ((*pos)->Slide(slide_amount, slide_children))
            ++count;
    }
    return count;
}

