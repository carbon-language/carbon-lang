//===-- SBSection.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSection.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Module.h"

namespace lldb_private 
{
    // We need a section implementation to hold onto a reference to the module
    // since if the module goes away and we have anyone still holding onto a 
    // SBSection object, we could crash.
    class SectionImpl
    {
    public:
        SectionImpl (const lldb_private::Section *section = NULL) :
            m_module_sp (),
            m_section (section)
        {
            if (section)
                m_module_sp = section->GetModule();
        }
        
        SectionImpl (const SectionImpl &rhs) :
            m_module_sp (rhs.m_module_sp),
            m_section   (rhs.m_section)
        {
        }

        bool 
        IsValid () const
        {
            return m_section != NULL;
        }

        void
        operator = (const SectionImpl &rhs)
        {
            m_module_sp = rhs.m_module_sp;
            m_section = rhs.m_section;
        }

        void
        operator =(const lldb_private::Section *section)
        {
            m_section = section;
            if (section)
                m_module_sp.reset(section->GetModule());
            else
                m_module_sp.reset();
        }

        const lldb_private::Section *
        GetSection () const
        {
            return m_section;
        }

        Module *
        GetModule()
        {
            return m_module_sp.get();
        }

        const lldb::ModuleSP &
        GetModuleSP() const
        {
            return m_module_sp;
        }
    protected:
        lldb::ModuleSP m_module_sp;
        const lldb_private::Section *m_section;
    };
}

using namespace lldb;
using namespace lldb_private;


SBSection::SBSection () :
    m_opaque_ap ()
{
}

SBSection::SBSection (const SBSection &rhs) :
    m_opaque_ap ()
{
    if (rhs.IsValid())
        m_opaque_ap.reset (new SectionImpl (*rhs.m_opaque_ap));
}



SBSection::SBSection (const lldb_private::Section *section) :
    m_opaque_ap ()
{
    if (section)
        m_opaque_ap.reset (new SectionImpl(section));
}

const SBSection &
SBSection::operator = (const SBSection &rhs)
{
    if (this != &rhs && rhs.IsValid())
        m_opaque_ap.reset (new SectionImpl(*rhs.m_opaque_ap));
    else
        m_opaque_ap.reset ();
    return *this;
}

SBSection::~SBSection ()
{
}

bool
SBSection::IsValid () const
{
    return m_opaque_ap.get() != NULL && m_opaque_ap->IsValid();
}

const char *
SBSection::GetName ()
{
    if (IsValid())
        return m_opaque_ap->GetSection()->GetName().GetCString();
    return NULL;
}


lldb::SBSection
SBSection::FindSubSection (const char *sect_name)
{
    lldb::SBSection sb_section;
    if (IsValid())
    {
        ConstString const_sect_name(sect_name);
        sb_section.SetSection(m_opaque_ap->GetSection()->GetChildren ().FindSectionByName(const_sect_name).get());
    }
    return sb_section;
}

size_t
SBSection::GetNumSubSections ()
{
    if (IsValid())
        return m_opaque_ap->GetSection()->GetChildren ().GetSize();
    return 0;
}

lldb::SBSection
SBSection::GetSubSectionAtIndex (size_t idx)
{
    lldb::SBSection sb_section;
    if (IsValid())
        sb_section.SetSection(m_opaque_ap->GetSection()->GetChildren ().GetSectionAtIndex(idx).get());
    return sb_section;
}

const lldb_private::Section *
SBSection::GetSection()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetSection();
    return NULL;
}

void
SBSection::SetSection (const lldb_private::Section *section)
{
    m_opaque_ap.reset (new SectionImpl(section));
}




lldb::addr_t
SBSection::GetFileAddress ()
{
    lldb::addr_t file_addr = LLDB_INVALID_ADDRESS;
    if (IsValid())
        return m_opaque_ap->GetSection()->GetFileAddress();
    return file_addr;
}

lldb::addr_t
SBSection::GetByteSize ()
{
    if (IsValid())
    {
        const Section *section = m_opaque_ap->GetSection();
        if (section)
            return section->GetByteSize();
    }
    return 0;
}

uint64_t
SBSection::GetFileOffset ()
{
    if (IsValid())
    {
        const Section *section = m_opaque_ap->GetSection();
        if (section)
        {
            Module *module = m_opaque_ap->GetModule();
            if (module)
            {
                ObjectFile *objfile = module->GetObjectFile();
                if (objfile)
                    return objfile->GetOffset() + section->GetFileOffset();
            }
            return section->GetFileOffset();
        }
    }
    return 0;
}

uint64_t
SBSection::GetFileByteSize ()
{
    if (IsValid())
    {
        const Section *section = m_opaque_ap->GetSection();
        if (section)
            return section->GetFileSize();
    }
    return 0;
}

SBData
SBSection::GetSectionData (uint64_t offset, uint64_t size)
{
    SBData sb_data;
    if (IsValid())
    {
        const Section *section = m_opaque_ap->GetSection();
        if (section)
        {
            const uint64_t sect_file_size = section->GetFileSize();
            if (sect_file_size > 0)
            {
                Module *module = m_opaque_ap->GetModule();
                if (module)
                {
                    ObjectFile *objfile = module->GetObjectFile();
                    if (objfile)
                    {
                        const uint64_t sect_file_offset = objfile->GetOffset() + section->GetFileOffset();
                        const uint64_t file_offset = sect_file_offset + offset;
                        uint64_t file_size = size;
                        if (file_size == UINT64_MAX)
                        {
                            file_size = section->GetByteSize();
                            if (file_size > offset)
                                file_size -= offset;
                            else
                                file_size = 0;
                        }
                        DataBufferSP data_buffer_sp (objfile->GetFileSpec().ReadFileContents (file_offset, file_size));
                        if (data_buffer_sp && data_buffer_sp->GetByteSize() > 0)
                        {
                            DataExtractorSP data_extractor_sp (new DataExtractor (data_buffer_sp, 
                                                                                  objfile->GetByteOrder(), 
                                                                                  objfile->GetAddressByteSize()));
                            
                            sb_data.SetOpaque (data_extractor_sp);
                        }
                    }
                }
            }
        }
    }
    return sb_data;
}

SectionType
SBSection::GetSectionType ()
{
    if (m_opaque_ap.get())
    {
        const Section *section = m_opaque_ap->GetSection();
        if (section)
            return section->GetType();
    }
    return eSectionTypeInvalid;
}


bool
SBSection::operator == (const SBSection &rhs)
{
    SectionImpl *lhs_ptr = m_opaque_ap.get();
    SectionImpl *rhs_ptr = rhs.m_opaque_ap.get();
    if (lhs_ptr && rhs_ptr)
        return lhs_ptr->GetSection() == rhs_ptr->GetSection();
    return false;
}

bool
SBSection::operator != (const SBSection &rhs)
{
    SectionImpl *lhs_ptr = m_opaque_ap.get();
    SectionImpl *rhs_ptr = rhs.m_opaque_ap.get();
    if (lhs_ptr && rhs_ptr)
        return lhs_ptr->GetSection() != rhs_ptr->GetSection();
    return false;
}

bool
SBSection::GetDescription (SBStream &description)
{
    if (m_opaque_ap.get())
    {
        description.Printf ("SBSection");
    }
    else
    {
        description.Printf ("No value");
    }

    return true;
}

