//===-- ValueObjectConstResult.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ValueObjectConstResult.h"

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ValueObjectList.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

ValueObjectConstResult::ValueObjectConstResult
(
    clang::ASTContext *clang_ast,
    void *clang_type,
    const ConstString &name,
    const lldb::DataBufferSP &data_sp,
    lldb::ByteOrder data_byte_order, 
    uint8_t data_addr_size
) :
    ValueObject (NULL),
    m_clang_ast (clang_ast),
    m_type_name ()
{
    m_data.SetByteOrder(data_byte_order);
    m_data.SetAddressByteSize(data_addr_size);
    m_data.SetData(data_sp);
    m_value.GetScalar() = (uintptr_t)data_sp->GetBytes();
    m_value.SetValueType(Value::eValueTypeHostAddress);
    m_value.SetContext(Value::eContextTypeClangType, clang_type);
    m_name = name;
    SetIsConstant ();
    SetValueIsValid(true);
}

ValueObjectConstResult::ValueObjectConstResult (const Error& error) :
    ValueObject (NULL),
    m_clang_ast (NULL),
    m_type_name ()
{
    m_error = error;
    SetIsConstant ();
}

ValueObjectConstResult::~ValueObjectConstResult()
{
}

void *
ValueObjectConstResult::GetClangType()
{
    return m_value.GetClangType();
}

lldb::ValueType
ValueObjectConstResult::GetValueType() const
{
    return eValueTypeConstResult;
}

size_t
ValueObjectConstResult::GetByteSize()
{
    // We stored all the data for this const object in our data
    return m_data.GetByteSize();
}

uint32_t
ValueObjectConstResult::CalculateNumChildren()
{
    return ClangASTContext::GetNumChildren (GetClangType(), true);
}

clang::ASTContext *
ValueObjectConstResult::GetClangAST ()
{
    return m_clang_ast;
}

ConstString
ValueObjectConstResult::GetTypeName()
{
    if (m_type_name.IsEmpty())
        m_type_name = ClangASTType::GetClangTypeName (GetClangType());
    return m_type_name;
}

void
ValueObjectConstResult::UpdateValue (ExecutionContextScope *exe_scope)
{
    m_error.Clear();
    // Const value is always valid
    SetValueIsValid (true);
}


bool
ValueObjectConstResult::IsInScope (StackFrame *frame)
{
    // A const result value is always in scope since it serializes all 
    // information needed to contain the constant value.
    return true;
}
