//===-- DIERef.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DIERef.h"
#include "DWARFCompileUnit.h"
#include "DWARFFormValue.h"

DIERef::DIERef() :
    cu_offset(DW_INVALID_OFFSET),
    die_offset(DW_INVALID_OFFSET)
{}

DIERef::DIERef(dw_offset_t d) :
    cu_offset(DW_INVALID_OFFSET),
    die_offset(d)
{}

DIERef::DIERef(dw_offset_t c, dw_offset_t d) :
    cu_offset(c),
    die_offset(d)
{}

DIERef::DIERef(lldb::user_id_t uid) :
    cu_offset(uid>>32),
    die_offset(uid&0xffffffff)
{}

DIERef::DIERef(const DWARFFormValue& form_value) :
    cu_offset(DW_INVALID_OFFSET),
    die_offset(DW_INVALID_OFFSET)
{
    if (form_value.IsValid())
    {
        const DWARFCompileUnit* dwarf_cu = form_value.GetCompileUnit();
        if (dwarf_cu)
        {
            if (dwarf_cu->GetBaseObjOffset() != DW_INVALID_OFFSET)
                cu_offset = dwarf_cu->GetBaseObjOffset();
            else
                cu_offset = dwarf_cu->GetOffset();
        }
        die_offset = form_value.Reference();
    }
}

lldb::user_id_t
DIERef::GetUID() const
{
    return ((lldb::user_id_t)cu_offset) << 32 | die_offset;
}
