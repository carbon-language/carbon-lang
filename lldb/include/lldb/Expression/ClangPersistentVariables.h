//===-- ClangPersistentVariables.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangPersistentVariables_h_
#define liblldb_ClangPersistentVariables_h_

#include "lldb-forward-rtti.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Symbol/TaggedASTType.h"

#include <map>
#include <string>

namespace lldb_private
{

class ClangPersistentVariable 
{
    friend class ClangPersistentVariables;
public:
    ClangPersistentVariable () :
        m_user_type(),
        m_data()
    {
    }
    
    ClangPersistentVariable (const ClangPersistentVariable &pv) :
        m_user_type(pv.m_user_type),
        m_data(pv.m_data)
    {
    }
    
    ClangPersistentVariable &operator=(const ClangPersistentVariable &pv)
    {
        m_user_type = pv.m_user_type;
        m_data = pv.m_data;
        return *this;
    }
    
    size_t Size ()
    {
        return (m_user_type.GetClangTypeBitWidth () + 7) / 8;
    }
    
    uint8_t *Data ()
    {
        return m_data->GetBytes();
    }
    
    TypeFromUser Type ()
    {
        return m_user_type;
    }
private:
    ClangPersistentVariable (TypeFromUser user_type)
    {
        m_user_type = user_type;
        m_data = lldb::DataBufferSP(new DataBufferHeap(Size(), 0));
    }
    TypeFromUser        m_user_type;
    lldb::DataBufferSP  m_data;
};
    
class ClangPersistentVariables
{
public:
    ClangPersistentVariable *CreateVariable (ConstString name, TypeFromUser user_type);
    ClangPersistentVariable *CreateResultVariable (TypeFromUser user_type);
    ClangPersistentVariable *GetVariable (ConstString name);
    
    ClangPersistentVariables ();
private:
    typedef std::map <ConstString, ClangPersistentVariable>    PVarMap;
    typedef PVarMap::iterator               PVarIterator;
    
    PVarMap                                 m_variables;
    uint64_t                                m_result_counter;
};

}

#endif