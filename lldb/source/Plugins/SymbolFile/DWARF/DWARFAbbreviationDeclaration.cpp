//===-- DWARFAbbreviationDeclaration.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFAbbreviationDeclaration.h"

#include "lldb/Core/dwarf.h"

#include "DWARFFormValue.h"

using namespace lldb_private;

DWARFAbbreviationDeclaration::DWARFAbbreviationDeclaration() :
    m_code  (InvalidCode),
    m_tag   (0),
    m_has_children (0),
    m_attributes()
{
}

DWARFAbbreviationDeclaration::DWARFAbbreviationDeclaration(dw_tag_t tag, uint8_t has_children) :
    m_code  (InvalidCode),
    m_tag   (tag),
    m_has_children (has_children),
    m_attributes()
{
}

bool
DWARFAbbreviationDeclaration::Extract(const DWARFDataExtractor& data, lldb::offset_t* offset_ptr)
{
    return Extract(data, offset_ptr, data.GetULEB128(offset_ptr));
}

bool
DWARFAbbreviationDeclaration::Extract(const DWARFDataExtractor& data, lldb::offset_t *offset_ptr, dw_uleb128_t code)
{
    m_code = code;
    m_attributes.clear();
    if (m_code)
    {
        m_tag = data.GetULEB128(offset_ptr);
        m_has_children = data.GetU8(offset_ptr);

        while (data.ValidOffset(*offset_ptr))
        {
            dw_attr_t attr = data.GetULEB128(offset_ptr);
            dw_form_t form = data.GetULEB128(offset_ptr);

            if (attr && form)
                m_attributes.push_back(DWARFAttribute(attr, form));
            else
                break;
        }

        return m_tag != 0;
    }
    else
    {
        m_tag = 0;
        m_has_children = 0;
    }

    return false;
}


void
DWARFAbbreviationDeclaration::Dump(Stream *s)  const
{
//  *ostrm_ptr << std::setfill(' ') << std::dec << '[' << std::setw(3) << std::right << m_code << ']' << ' ' << std::setw(30) << std::left << DW_TAG_value_to_name(m_tag) << DW_CHILDREN_value_to_name(m_has_children) << std::endl;
//
//  DWARFAttribute::const_iterator pos;
//
//  for (pos = m_attributes.begin(); pos != m_attributes.end(); ++pos)
//      *ostrm_ptr << "      " << std::setw(29) << std::left << DW_AT_value_to_name(pos->attr()) << ' ' << DW_FORM_value_to_name(pos->form()) << std::endl;
//
//  *ostrm_ptr << std::endl;
}



bool
DWARFAbbreviationDeclaration::IsValid()
{
    return m_code != 0 && m_tag != 0;
}



uint32_t
DWARFAbbreviationDeclaration::FindAttributeIndex(dw_attr_t attr) const
{
    uint32_t i;
    const uint32_t kNumAttributes = m_attributes.size();
    for (i = 0; i < kNumAttributes; ++i)
    {
        if (m_attributes[i].get_attr() == attr)
            return i;
    }
    return DW_INVALID_INDEX;
}


bool
DWARFAbbreviationDeclaration::operator == (const DWARFAbbreviationDeclaration& rhs) const
{
    return Tag()            == rhs.Tag()
        && HasChildren()    == rhs.HasChildren()
        && Attributes()     == rhs.Attributes();
}

