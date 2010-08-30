//===-- StackID.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StackID_h_
#define liblldb_StackID_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/AddressRange.h"

namespace lldb_private {

class StackID
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    StackID () :
        m_start_pc (LLDB_INVALID_ADDRESS),
        m_cfa (LLDB_INVALID_ADDRESS),
        m_symbol_scope (NULL)
    {
    }

    explicit 
    StackID (lldb::addr_t start_pc, lldb::addr_t cfa, SymbolContextScope *symbol_scope) :
        m_start_pc (start_pc),
        m_cfa (cfa),
        m_symbol_scope (symbol_scope)
    {
    }

    StackID (const StackID& rhs) :
        m_start_pc (rhs.m_start_pc),
        m_cfa (rhs.m_cfa),
        m_symbol_scope (rhs.m_symbol_scope)
    {
    }

    ~StackID()
    {
    }

    const lldb::addr_t
    GetStartAddress() const
    {
        return m_start_pc;
    }

    void
    SetStartAddress(lldb::addr_t start_pc)
    {
        m_start_pc = start_pc;
    }

    lldb::addr_t
    GetCallFrameAddress() const
    {
        return m_cfa;
    }

    SymbolContextScope *
    GetSymbolContextScope () const
    {
        return m_symbol_scope;
    }
    
    void
    SetSymbolContextScope (SymbolContextScope *symbol_scope)
    {
        m_symbol_scope = symbol_scope;
    }

    void
    Dump (Stream *s);

    //------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------
    const StackID&
    operator=(const StackID& rhs)
    {
        if (this != &rhs)
        {
            m_start_pc = rhs.m_start_pc;
            m_cfa = rhs.m_cfa;
            m_symbol_scope = rhs.m_symbol_scope;
        }
        return *this;
    }

protected:
    //------------------------------------------------------------------
    // Classes that inherit from StackID can see and modify these
    //------------------------------------------------------------------
    lldb::addr_t m_start_pc;            // The start address for the function/symbol for this frame
    lldb::addr_t m_cfa;                 // The call frame address (stack pointer) value
                                        // at the beginning of the function that uniquely
                                        // identifies this frame (along with m_symbol_scope below)
    SymbolContextScope *m_symbol_scope; // If NULL, there is no block or symbol for this frame.
                                        // If not NULL, this will either be the scope for the 
                                        // lexical block for the frame, or the scope 
                                        // for the symbol. Symbol context scopes are 
                                        // always be unique pointers since the are part
                                        // of the Block and Symbol objects and can easily
                                        // be used to tell if a stack ID is the same as 
                                        // another.
};

bool operator== (const StackID& lhs, const StackID& rhs);
bool operator!= (const StackID& lhs, const StackID& rhs);
bool operator<  (const StackID& lhs, const StackID& rhs);

} // namespace lldb_private

#endif  // liblldb_StackID_h_
