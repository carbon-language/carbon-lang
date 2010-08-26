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
        m_start_address(),
        m_cfa (0),
        m_inline_block_id (0)
    {
    }

    explicit 
    StackID (lldb::addr_t cfa, lldb::user_id_t inline_block_id) :
        m_start_address (),
        m_cfa (cfa),
        m_inline_block_id (inline_block_id)
    {
    }

    StackID (const Address& start_address, lldb::addr_t cfa, uint32_t inline_block_id) : 
        m_start_address (start_address),
        m_cfa (cfa),
        m_inline_block_id (inline_block_id)
    {
    }

    StackID (const StackID& rhs) :
        m_start_address (rhs.m_start_address),
        m_cfa (rhs.m_cfa),
        m_inline_block_id (rhs.m_inline_block_id)
    {
    }

    ~StackID()
    {
    }

    const Address&
    GetStartAddress() const
    {
        return m_start_address;
    }

    void
    SetStartAddress(const Address& start_address)
    {
        m_start_address = start_address;
    }

    lldb::addr_t
    GetCallFrameAddress() const
    {
        return m_cfa;
    }

    lldb::user_id_t
    GetInlineBlockID () const
    {
        return m_inline_block_id;
    }
    
    void
    SetInlineBlockID (lldb::user_id_t inline_block_id)
    {
        m_inline_block_id = inline_block_id;
    }

    //------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------
    const StackID&
    operator=(const StackID& rhs)
    {
        if (this != &rhs)
        {
            m_start_address = rhs.m_start_address;
            m_cfa = rhs.m_cfa;
            m_inline_block_id = rhs.m_inline_block_id;
        }
        return *this;
    }

protected:
    //------------------------------------------------------------------
    // Classes that inherit from StackID can see and modify these
    //------------------------------------------------------------------
    Address m_start_address;            // The address range for the function for this frame
    lldb::addr_t m_cfa;                 // The call frame address (stack pointer) value
                                        // at the beginning of the function that uniquely
                                        // identifies this frame (along with m_inline_block_id below)
    lldb::user_id_t m_inline_block_id;  // The inline height of a stack frame. Zero is the actual
                                        // value for the place where a thread stops, 1 and above
                                        // are for the inlined frames above the concrete base frame.
};

bool operator== (const StackID& lhs, const StackID& rhs);
bool operator!= (const StackID& lhs, const StackID& rhs);
bool operator<  (const StackID& lhs, const StackID& rhs);

} // namespace lldb_private

#endif  // liblldb_StackID_h_
