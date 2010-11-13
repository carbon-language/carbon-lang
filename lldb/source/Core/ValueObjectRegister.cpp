//===-- ValueObjectRegister.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/Core/ValueObjectRegister.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Module.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

#pragma mark ValueObjectRegisterContext

ValueObjectRegisterContext::ValueObjectRegisterContext (ValueObject *parent, RegisterContext *reg_ctx) :
    ValueObject (parent),
    m_reg_ctx (reg_ctx)
{
    assert (reg_ctx);
    m_name.SetCString("Registers");
    SetValueIsValid (true);
}

ValueObjectRegisterContext::~ValueObjectRegisterContext()
{
}

void *
ValueObjectRegisterContext::GetClangType ()
{
    return NULL;
}

ConstString
ValueObjectRegisterContext::GetTypeName()
{
    ConstString empty_type_name;
    return empty_type_name;
}

uint32_t
ValueObjectRegisterContext::CalculateNumChildren()
{
    return m_reg_ctx->GetRegisterSetCount();
}

clang::ASTContext *
ValueObjectRegisterContext::GetClangAST ()
{
    return NULL;
}

size_t
ValueObjectRegisterContext::GetByteSize()
{
    return 0;
}

void
ValueObjectRegisterContext::UpdateValue (ExecutionContextScope *exe_scope)
{
    m_error.Clear();
    StackFrame *frame = exe_scope->CalculateStackFrame();
    if (frame)
        m_reg_ctx = frame->GetRegisterContext();
    else
        m_reg_ctx = NULL;

    SetValueIsValid (m_reg_ctx != NULL);
}

ValueObjectSP
ValueObjectRegisterContext::CreateChildAtIndex (uint32_t idx, bool synthetic_array_member, int32_t synthetic_index)
{
    ValueObjectSP valobj_sp;

    const uint32_t num_children = GetNumChildren();
    if (idx < num_children)
        valobj_sp.reset (new ValueObjectRegisterSet(this, m_reg_ctx, idx));
    return valobj_sp;
}


#pragma mark -
#pragma mark ValueObjectRegisterSet

ValueObjectRegisterSet::ValueObjectRegisterSet (ValueObject *parent, RegisterContext *reg_ctx, uint32_t reg_set_idx) :
    ValueObject (parent),
    m_reg_ctx (reg_ctx),
    m_reg_set (NULL),
    m_reg_set_idx (reg_set_idx)
{
    assert (reg_ctx);
    m_reg_set = reg_ctx->GetRegisterSet(m_reg_set_idx);
    if (m_reg_set)
    {
        m_name.SetCString (m_reg_set->name);
    }
}

ValueObjectRegisterSet::~ValueObjectRegisterSet()
{
}

void *
ValueObjectRegisterSet::GetClangType ()
{
    return NULL;
}

ConstString
ValueObjectRegisterSet::GetTypeName()
{
    return ConstString();
}

uint32_t
ValueObjectRegisterSet::CalculateNumChildren()
{
    const RegisterSet *reg_set = m_reg_ctx->GetRegisterSet(m_reg_set_idx);
    if (reg_set)
        return reg_set->num_registers;
    return 0;
}

clang::ASTContext *
ValueObjectRegisterSet::GetClangAST ()
{
    return NULL;
}

size_t
ValueObjectRegisterSet::GetByteSize()
{
    return 0;
}

void
ValueObjectRegisterSet::UpdateValue (ExecutionContextScope *exe_scope)
{
    m_error.Clear();
    SetValueDidChange (false);
    StackFrame *frame = exe_scope->CalculateStackFrame();
    if (frame == NULL)
        m_reg_ctx = NULL;
    else
    {
        m_reg_ctx = frame->GetRegisterContext ();
        if (m_reg_ctx)
        {
            const RegisterSet *reg_set = m_reg_ctx->GetRegisterSet (m_reg_set_idx);
            if (reg_set == NULL)
                m_reg_ctx = NULL;
            else if (m_reg_set != reg_set)
            {
                SetValueDidChange (true);
                m_name.SetCString(reg_set->name);
            }
        }
    }
    if (m_reg_ctx)
    {
        SetValueIsValid (true);
    }
    else
    {
        SetValueIsValid (false);
        m_children.clear();
    }
}


ValueObjectSP
ValueObjectRegisterSet::CreateChildAtIndex (uint32_t idx, bool synthetic_array_member, int32_t synthetic_index)
{
    ValueObjectSP valobj_sp;
    if (m_reg_ctx && m_reg_set)
    {
        const uint32_t num_children = GetNumChildren();
        if (idx < num_children)
            valobj_sp.reset (new ValueObjectRegister(this, m_reg_ctx, m_reg_set->registers[idx]));
    }
    return valobj_sp;
}


#pragma mark -
#pragma mark ValueObjectRegister

ValueObjectRegister::ValueObjectRegister (ValueObject *parent, RegisterContext *reg_ctx, uint32_t reg_num) :
    ValueObject (parent),
    m_reg_ctx (reg_ctx),
    m_reg_info (NULL),
    m_reg_num (reg_num),
    m_type_name (),
    m_clang_type (NULL)
{
    assert (reg_ctx);
    m_reg_info = reg_ctx->GetRegisterInfoAtIndex(reg_num);
    if (m_reg_info)
    {
        if (m_reg_info->name)
            m_name.SetCString(m_reg_info->name);
        else if (m_reg_info->alt_name)
            m_name.SetCString(m_reg_info->alt_name);
    }
}

ValueObjectRegister::~ValueObjectRegister()
{
}

void *
ValueObjectRegister::GetClangType ()
{
    if (m_clang_type == NULL && m_reg_info)
    {
        Process *process = m_reg_ctx->CalculateProcess ();
        if (process)
        {
            Module *exe_module = process->GetTarget().GetExecutableModule ().get();
            if (exe_module)
            {
                TypeList *type_list = exe_module->GetTypeList();
                if (type_list)
                    m_clang_type = type_list->GetClangASTContext().GetBuiltinTypeForEncodingAndBitSize (m_reg_info->encoding, m_reg_info->byte_size * 8);
            }
        }
    }
    return m_clang_type;
}

ConstString
ValueObjectRegister::GetTypeName()
{
    if (m_type_name.IsEmpty())
        m_type_name = ClangASTType::GetClangTypeName (GetClangType());
    return m_type_name;
}

uint32_t
ValueObjectRegister::CalculateNumChildren()
{
    return 0;
}

clang::ASTContext *
ValueObjectRegister::GetClangAST ()
{
    Process *process = m_reg_ctx->CalculateProcess ();
    if (process)
    {
        Module *exe_module = process->GetTarget().GetExecutableModule ().get();
        if (exe_module)
        {
            TypeList *type_list = exe_module->GetTypeList();
            if (type_list)
                return type_list->GetClangASTContext().getASTContext();
        }
    }
    return NULL;
}

size_t
ValueObjectRegister::GetByteSize()
{
    return m_reg_info->byte_size;
}

void
ValueObjectRegister::UpdateValue (ExecutionContextScope *exe_scope)
{
    m_error.Clear();
    StackFrame *frame = exe_scope->CalculateStackFrame();
    if (frame)
    {
        m_reg_ctx = frame->GetRegisterContext();
        if (m_reg_ctx)
        {
            const RegisterInfo *reg_info = m_reg_ctx->GetRegisterInfoAtIndex(m_reg_num);
            if (m_reg_info != reg_info)
            {
                m_reg_info = reg_info;
                if (m_reg_info)
                {
                    if (m_reg_info->name)
                        m_name.SetCString(m_reg_info->name);
                    else if (m_reg_info->alt_name)
                        m_name.SetCString(m_reg_info->alt_name);
                }
            }
        }
    }
    else
    {
        m_reg_ctx = NULL;
        m_reg_info = NULL;
    }


    if (m_reg_ctx && m_reg_info)
    {
        if (m_reg_ctx->ReadRegisterBytes (m_reg_num, m_data))
        {
            m_value.SetContext(Value::eContextTypeRegisterInfo, (void *)m_reg_info);
            m_value.SetValueType(Value::eValueTypeHostAddress);
            m_value.GetScalar() = (uintptr_t)m_data.GetDataStart();
            SetValueIsValid (true);
            return;
        }
    }
    SetValueIsValid (false);
}


