//===-- ExpressionVariable.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ExpressionVariable.h"

using namespace lldb_private;

ExpressionVariable::~ExpressionVariable()
{
}

uint8_t *
ExpressionVariable::GetValueBytes()
{
    const size_t byte_size = m_frozen_sp->GetByteSize();
    if (byte_size > 0)
    {
        if (m_frozen_sp->GetDataExtractor().GetByteSize() < byte_size)
        {
            m_frozen_sp->GetValue().ResizeData(byte_size);
            m_frozen_sp->GetValue().GetData (m_frozen_sp->GetDataExtractor());
        }
        return const_cast<uint8_t *>(m_frozen_sp->GetDataExtractor().GetDataStart());
    }
    return NULL;
}

PersistentExpressionState::~PersistentExpressionState ()
{
}
