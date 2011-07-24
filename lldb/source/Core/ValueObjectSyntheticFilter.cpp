//===-- ValueObjectSyntheticFilter.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/Core/ValueObjectSyntheticFilter.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/FormatClasses.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"

#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"


using namespace lldb_private;

ValueObjectSynthetic::ValueObjectSynthetic (ValueObject &parent, lldb::SyntheticChildrenSP filter) :
    ValueObject(parent),
    m_address (),
    m_type_sp(),
    m_use_synthetic (lldb::eUseSyntheticFilter),
    m_synth_filter(filter->GetFrontEnd(parent.GetSP())),
    m_children_byindex(),
    m_name_toindex()
{
    SetName (parent.GetName().AsCString());
}

ValueObjectSynthetic::~ValueObjectSynthetic()
{
    m_owning_valobj_sp.reset();
}

lldb::clang_type_t
ValueObjectSynthetic::GetClangType ()
{
    if (m_type_sp)
        return m_value.GetClangType();
    else
        return m_parent->GetClangType();
}

ConstString
ValueObjectSynthetic::GetTypeName()
{
    const bool success = UpdateValueIfNeeded();
    if (success && m_type_sp)
        return ClangASTType::GetConstTypeName (GetClangType());
    else
        return m_parent->GetTypeName();
}

uint32_t
ValueObjectSynthetic::CalculateNumChildren()
{
    return m_synth_filter->CalculateNumChildren();
}

clang::ASTContext *
ValueObjectSynthetic::GetClangAST ()
{
    const bool success = UpdateValueIfNeeded(false);
    if (success && m_type_sp)
        return m_type_sp->GetClangAST();
    else
        return m_parent->GetClangAST ();
}

size_t
ValueObjectSynthetic::GetByteSize()
{
    const bool success = UpdateValueIfNeeded();
    if (success && m_type_sp)
        return m_value.GetValueByteSize(GetClangAST(), NULL);
    else
        return m_parent->GetByteSize();
}

lldb::ValueType
ValueObjectSynthetic::GetValueType() const
{
    return m_parent->GetValueType();
}

bool
ValueObjectSynthetic::UpdateValue ()
{
    SetValueIsValid (false);
    m_error.Clear();

    if (!m_parent->UpdateValueIfNeeded())
    {
        // our parent could not update.. as we are meaningless without a parent, just stop
        if (m_error.Success() && m_parent->GetError().Fail())
            m_error = m_parent->GetError();
        return false;
    }

    m_children_byindex.clear();
    m_name_toindex.clear();
    
    SetValueIsValid(true);
    return true;
}

lldb::ValueObjectSP
ValueObjectSynthetic::GetChildAtIndex (uint32_t idx, bool can_create)
{
    ByIndexIterator iter = m_children_byindex.find(idx);
    
    if (iter == m_children_byindex.end())
    {
        if (can_create)
        {
            lldb::ValueObjectSP synth_guy = m_synth_filter->GetChildAtIndex (idx, can_create);
            m_children_byindex[idx]= synth_guy;
            return synth_guy;
        }
        else
            return lldb::ValueObjectSP();
    }
    else
        return iter->second;
}

lldb::ValueObjectSP
ValueObjectSynthetic::GetChildMemberWithName (const ConstString &name, bool can_create)
{
    
    uint32_t index = GetIndexOfChildWithName(name);
    
    if (index == UINT32_MAX)
        return lldb::ValueObjectSP();
    
    return GetChildAtIndex(index, can_create);
}

uint32_t
ValueObjectSynthetic::GetIndexOfChildWithName (const ConstString &name)
{
    NameToIndexIterator iter = m_name_toindex.find(name.GetCString());
    
    if (iter == m_name_toindex.end())
    {
        uint32_t index = m_synth_filter->GetIndexOfChildWithName (name);
        m_name_toindex[name.GetCString()] = index;
        return index;
    }
    return iter->second;
}

bool
ValueObjectSynthetic::IsInScope ()
{
    return m_parent->IsInScope();
}

