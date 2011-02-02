//===-- UniqueDWARFASTType.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UniqueDWARFASTType.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Symbol/Declaration.h"

#include "DWARFDebugInfoEntry.h"

bool
UniqueDWARFASTTypeList::Find 
(
    const DWARFDebugInfoEntry *die, 
    const lldb_private::Declaration &decl,
    UniqueDWARFASTType &entry
) const
{
    collection::const_iterator pos, end = m_collection.end();
    for (pos = m_collection.begin(); pos != end; ++pos)
    {
        if (pos->m_die->Tag() == die->Tag())
        {
            if (pos->m_declaration == decl)
            {
                entry = *pos;
                return true;
            }
        }
    }
    return false;
}
