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

#include "lldb/lldb-forward-rtti.h"
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
        m_name(),
        m_user_type(),
        m_data()
    {
    }
    
    ClangPersistentVariable (const ClangPersistentVariable &pv) :
        m_name(pv.m_name),
        m_user_type(pv.m_user_type),
        m_data(pv.m_data)
    {
    }
    
    ClangPersistentVariable &operator=(const ClangPersistentVariable &pv)
    {
        m_name = pv.m_name;
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
    
    Error Print(Stream &output_stream,
                ExecutionContext &exe_ctx,
                lldb::Format format,
                bool show_types,
                bool show_summary,
                bool verbose);
private:
    ClangPersistentVariable (ConstString name, TypeFromUser user_type)
    {
        m_name = name;
        m_user_type = user_type;
        m_data = lldb::DataBufferSP(new DataBufferHeap(Size(), 0));
    }
    ConstString         m_name;
    TypeFromUser        m_user_type;
    lldb::DataBufferSP  m_data;
};
    
class ClangPersistentVariables
{
public:
    ClangPersistentVariable *CreateVariable (ConstString name, TypeFromUser user_type);
    ClangPersistentVariable *GetVariable (ConstString name);
    
    void GetNextResultName(std::string &name);
    
    ClangPersistentVariables ();
private:
    typedef std::map <ConstString, ClangPersistentVariable>    PVarMap;
    typedef PVarMap::iterator               PVarIterator;
    
    PVarMap                                 m_variables;
    uint64_t                                m_result_counter;
};

}

#endif
