//===-- StackID.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackID.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"

using namespace lldb_private;


void
StackID::Dump (Stream *s)
{
    s->Printf("StackID (start_pc = 0x%16.16llx, cfa = 0x%16.16llx, symbol_scope = %p", (uint64_t)m_start_pc, (uint64_t)m_cfa, m_symbol_scope);
    if (m_symbol_scope)
    {
        SymbolContext sc;
    
        m_symbol_scope->CalculateSymbolContext (&sc);
        if (sc.block)
            s->Printf(" (Block {0x%8.8x})", sc.block->GetID());
        else if (sc.symbol)
            s->Printf(" (Symbol{0x%8.8x})", sc.symbol->GetID());
    }
    s->PutCString(") ");
}

bool
lldb_private::operator== (const StackID& lhs, const StackID& rhs)
{
    return lhs.GetCallFrameAddress()    == rhs.GetCallFrameAddress()    && 
           lhs.GetSymbolContextScope()  == rhs.GetSymbolContextScope()  &&
           lhs.GetStartAddress()        == rhs.GetStartAddress();
}

bool
lldb_private::operator!= (const StackID& lhs, const StackID& rhs)
{
    return lhs.GetCallFrameAddress()    != rhs.GetCallFrameAddress()    || 
           lhs.GetSymbolContextScope()  != rhs.GetSymbolContextScope()  || 
           lhs.GetStartAddress()        != rhs.GetStartAddress();
}

bool
lldb_private::operator< (const StackID& lhs, const StackID& rhs)
{
    return lhs.GetCallFrameAddress()    < rhs.GetCallFrameAddress();
}
