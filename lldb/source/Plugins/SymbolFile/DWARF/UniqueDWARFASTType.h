//===-- UniqueDWARFASTType.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_UniqueDWARFASTType_h_
#define lldb_UniqueDWARFASTType_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
#include "llvm/ADT/DenseMap.h"

// Project includes
#include "lldb/Symbol/Declaration.h"

class DWARFCompileUnit;
class DWARFDebugInfoEntry;
class SymbolFileDWARF;

class UniqueDWARFASTType
{
public:
	//------------------------------------------------------------------
	// Constructors and Destructors
	//------------------------------------------------------------------
	UniqueDWARFASTType () :
        m_type_sp (),
        m_die (NULL),
        m_declaration ()
    {
    }

	UniqueDWARFASTType (lldb::TypeSP &type_sp,
                        DWARFDebugInfoEntry *die,
                        const lldb_private::Declaration &decl) :
        m_type_sp (type_sp),
        m_die (die),
        m_declaration (decl)
    {
    }
    
    UniqueDWARFASTType (const UniqueDWARFASTType &rhs) :
        m_type_sp (rhs.m_type_sp),
        m_die (rhs.m_die),
        m_declaration (rhs.m_declaration)
    {
    }

	~UniqueDWARFASTType()
    {
    }

    UniqueDWARFASTType &
    operator= (const UniqueDWARFASTType &rhs)
    {
        if (this != &rhs)
        {
            m_type_sp = rhs.m_type_sp;
            m_die = rhs.m_die;
            m_declaration = rhs.m_declaration;
        }
        return *this;
    }

    lldb::TypeSP m_type_sp;
    const DWARFDebugInfoEntry *m_die;
    lldb_private::Declaration m_declaration;    
};

class UniqueDWARFASTTypeList
{
public:
    UniqueDWARFASTTypeList () :
        m_collection()
    {
    }
    
    ~UniqueDWARFASTTypeList ()
    {
    }
    
    uint32_t
    GetSize()
    {
        return (uint32_t)m_collection.size();
    }
    
    void
    Append (const UniqueDWARFASTType &entry)
    {
        m_collection.push_back (entry);
    }
    
    bool
    Find (const DWARFDebugInfoEntry *die, 
          const lldb_private::Declaration &decl,
          UniqueDWARFASTType &entry) const;
    
protected:
    typedef std::vector<UniqueDWARFASTType> collection;
    collection m_collection;
};

class UniqueDWARFASTTypeMap
{
public:
    UniqueDWARFASTTypeMap () :
        m_collection ()
    {
    }
    
    ~UniqueDWARFASTTypeMap ()
    {
    }

    void
    Insert (const lldb_private::ConstString &name, 
            const UniqueDWARFASTType &entry)
    {
        m_collection[name.GetCString()].Append (entry);
    }

    bool
    Find (const lldb_private::ConstString &name, 
          const DWARFDebugInfoEntry *die, 
          const lldb_private::Declaration &decl,
          UniqueDWARFASTType &entry) const
    {
        const char *unique_name_cstr = name.GetCString();
        collection::const_iterator pos = m_collection.find (unique_name_cstr);
        if (pos != m_collection.end())
        {
            return pos->second.Find (die, decl, entry);
        }
        return false;
    }

protected:
    // A unique name string should be used
    typedef llvm::DenseMap<const char *, UniqueDWARFASTTypeList> collection;
    collection m_collection;
};

#endif	// lldb_UniqueDWARFASTType_h_
