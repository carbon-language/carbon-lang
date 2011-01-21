//===-- EmulateInstruction.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_EmulateInstruction_h_
#define lldb_EmulateInstruction_h_

#include "lldb/lldb-include.h"

namespace lldb_private {

class EmulateInstruction
{
public: 
    enum ContextType
    {
        eContextInvalid = 0,
        eContextReadOpcode,
        eContextImmediate,
        eContextPushRegisterOnStack,
        eContextAdjustStackPointer,
        eContextRegisterPlusOffset,
    };
    
    struct Context
    {
        ContextType type;
        lldb::addr_t arg0;
        lldb::addr_t arg1;
        lldb::addr_t arg2;
    };

    union Opcode
    {
        uint8_t inst8;
        uint16_t inst16;
        uint32_t inst32;
        uint64_t inst64;
        union inst
        {
            uint8_t bytes[16];
            uint8_t length;
        };
    };

    enum OpcodeType
    {
        eOpcode8,
        eOpcode16,
        eOpcode32,
        eOpcode64,
        eOpcodeBytes,
    };

    struct Instruction
    {
        OpcodeType opcode_type;
        Opcode opcode;
    };

    typedef size_t (*ReadMemory) (void *baton,
                                  const Context &context, 
                                  lldb::addr_t addr, 
                                  void *dst,
                                  size_t length);
    
    typedef size_t (*WriteMemory) (void *baton,
                                   const Context &context, 
                                   lldb::addr_t addr, 
                                   const void *dst,
                                   size_t length);
    
    typedef bool   (*ReadRegister)  (void *baton,
                                     uint32_t reg_kind, 
                                     uint32_t reg_num,
                                     uint64_t &reg_value);

    typedef bool   (*WriteRegister) (void *baton,
                                     const Context &context, 
                                     uint32_t reg_kind, 
                                     uint32_t reg_num,
                                     uint64_t reg_value);

    EmulateInstruction (lldb::ByteOrder byte_order,
                        uint32_t addr_byte_size,
                        void *baton,
                        ReadMemory read_mem_callback,
                        WriteMemory write_mem_callback,
                        ReadRegister read_reg_callback,
                        WriteRegister write_reg_callback);

    virtual ~EmulateInstruction()
    {
    }
    
    virtual bool 
    ReadInstruction () = 0;

    virtual bool
    EvaluateInstruction () = 0;
    
    // Create a mask that starts at bit zero and includes "bit"
    static uint64_t
    MaskUpToBit (const uint64_t bit)
    {
        return (1ull << (bit + 1ull)) - 1ull;
    }

    static bool
    BitIsSet (const uint64_t value, const uint64_t bit)
    {
        return (value & (1ull << bit)) != 0;
    }

    static bool
    BitIsClear (const uint64_t value, const uint64_t bit)
    {
        return (value & (1ull << bit)) == 0;
    }

    static int64_t
    SignedBits (const uint64_t value, const uint64_t msbit, const uint64_t lsbit)
    {
        uint64_t result = UnsignedBits (value, msbit, lsbit);
        if (BitIsSet(value, msbit))
        {
            // Sign extend
            result |= ~MaskUpToBit (msbit - lsbit);
        }
        return result;
    }

    static uint64_t
    UnsignedBits (const uint64_t value, const uint64_t msbit, const uint64_t lsbit)
    {
        uint64_t result = value >> lsbit;
        result &= MaskUpToBit (msbit - lsbit);
        return result;
    }

    uint64_t
    ReadRegisterUnsigned (uint32_t reg_kind, 
                          uint32_t reg_num, 
                          uint64_t fail_value, 
                          bool *success_ptr);

    bool
    WriteRegisterUnsigned (const Context &context, 
                           uint32_t reg_kind, 
                           uint32_t reg_num, 
                           uint64_t reg_value);

    uint64_t
    ReadMemoryUnsigned (const Context &context, 
                        lldb::addr_t addr, 
                        size_t byte_size, 
                        uint64_t fail_value, 
                        bool *success_ptr);

    bool
    WriteMemoryUnsigned (const Context &context, 
                         lldb::addr_t addr, 
                         uint64_t uval,
                         size_t uval_byte_size);

    static uint32_t
    BitCount (uint64_t value)
    {
        uint32_t set_bit_count = 0;
        while (value)
        {
            if (value & 1)
                ++set_bit_count;
            value >>= 1;
        }
        return set_bit_count;
    }
    
    uint32_t
    GetAddressByteSize () const
    {
        return m_addr_byte_size;
    }

    lldb::ByteOrder
    GetByteOrder () const
    {
        return m_byte_order;
    }

    uint64_t
    OpcodeAsUnsigned (bool *success_ptr)
    {
        if (success_ptr)
            *success_ptr = true;
        switch (m_inst.opcode_type)
        {
        eOpcode8:   return m_inst.opcode.inst8;
        eOpcode16:  return m_inst.opcode.inst16;
        eOpcode32:  return m_inst.opcode.inst32;
        eOpcode64:  return m_inst.opcode.inst64;
        eOpcodeBytes:
            break;
        }
        if (success_ptr)
            *success_ptr = false;
        return 0;
    }

protected:
    lldb::ByteOrder     m_byte_order;
    uint32_t            m_addr_byte_size;
    void *              m_baton;
    ReadMemory          m_read_mem_callback;
    WriteMemory         m_write_mem_callback;
    ReadRegister        m_read_reg_callback;
    WriteRegister       m_write_reg_callback;

    lldb::addr_t m_inst_pc;
    Instruction m_inst;
    //------------------------------------------------------------------
    // For EmulateInstruction only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (EmulateInstruction);
};

}   // namespace lldb_private

#endif  // lldb_EmulateInstruction_h_
