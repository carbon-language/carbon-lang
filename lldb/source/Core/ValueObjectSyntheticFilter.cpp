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

ValueObjectSyntheticFilter::ValueObjectSyntheticFilter (ValueObject &parent, lldb::SyntheticFilterSP filter) :
    ValueObject(parent),
    m_address (),
    m_type_sp(),
m_use_synthetic (lldb::eUseSyntheticFilter),
    m_synth_filter(filter)
{
    SetName (parent.GetName().AsCString());
}

ValueObjectSyntheticFilter::~ValueObjectSyntheticFilter()
{
    m_owning_valobj_sp.reset();
}

lldb::clang_type_t
ValueObjectSyntheticFilter::GetClangType ()
{
    if (m_type_sp)
        return m_value.GetClangType();
    else
        return m_parent->GetClangType();
}

ConstString
ValueObjectSyntheticFilter::GetTypeName()
{
    const bool success = UpdateValueIfNeeded();
    if (success && m_type_sp)
        return ClangASTType::GetConstTypeName (GetClangType());
    else
        return m_parent->GetTypeName();
}

uint32_t
ValueObjectSyntheticFilter::CalculateNumChildren()
{
    const bool success = UpdateValueIfNeeded();
    if (!success)
        return 0;
    if (m_synth_filter.get())
        return m_synth_filter->GetCount();
    return 0;
    if (success && m_type_sp)
        return ClangASTContext::GetNumChildren (GetClangAST (), GetClangType(), true);
    else
        return m_parent->GetNumChildren();
}

clang::ASTContext *
ValueObjectSyntheticFilter::GetClangAST ()
{
    const bool success = UpdateValueIfNeeded(false);
    if (success && m_type_sp)
        return m_type_sp->GetClangAST();
    else
        return m_parent->GetClangAST ();
}

size_t
ValueObjectSyntheticFilter::GetByteSize()
{
    const bool success = UpdateValueIfNeeded();
    if (success && m_type_sp)
        return m_value.GetValueByteSize(GetClangAST(), NULL);
    else
        return m_parent->GetByteSize();
}

lldb::ValueType
ValueObjectSyntheticFilter::GetValueType() const
{
    return m_parent->GetValueType();
}

bool
ValueObjectSyntheticFilter::UpdateValue ()
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

    SetValueIsValid(true);
    return true;
}

lldb::ValueObjectSP
ValueObjectSyntheticFilter::GetChildAtIndex (uint32_t idx, bool can_create)
{
    if (!m_synth_filter.get())
        return lldb::ValueObjectSP();
    if (idx >= m_synth_filter->GetCount())
        return lldb::ValueObjectSP();
    return m_parent->GetSyntheticExpressionPathChild(m_synth_filter->GetExpressionPathAtIndex(idx).c_str(), can_create);
}

lldb::ValueObjectSP
ValueObjectSyntheticFilter::GetChildMemberWithName (const ConstString &name, bool can_create)
{
    if (!m_synth_filter.get())
        return lldb::ValueObjectSP();
    uint32_t idx = GetIndexOfChildWithName(name);
    if (idx >= m_synth_filter->GetCount())
        return lldb::ValueObjectSP();
    return m_parent->GetSyntheticExpressionPathChild(name.GetCString(), can_create);
}

uint32_t
ValueObjectSyntheticFilter::GetIndexOfChildWithName (const ConstString &name)
{
    const char* name_cstr = name.GetCString();
    for (int i = 0; i < m_synth_filter->GetCount(); i++)
    {
        const char* expr_cstr = m_synth_filter->GetExpressionPathAtIndex(i).c_str();
        if (::strcmp(name_cstr, expr_cstr))
            return i;
    }
    return UINT32_MAX;
}

bool
ValueObjectSyntheticFilter::IsInScope ()
{
    return m_parent->IsInScope();
}

