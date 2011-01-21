//===-- lldb_EmulateInstructionARM.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_EmulateInstructionARM_h_
#define lldb_EmulateInstructionARM_h_

#include "EmulateInstruction.h"
#include "ARM_DWARF_Registers.h"

namespace lldb_private {

class EmulateInstructionARM : public EmulateInstruction
{
public: 
    enum Mode
    {
        eModeInvalid,
        eModeARM,
        eModeThumb
    };

    EmulateInstructionARM (void *baton,
                           ReadMemory read_mem_callback,
                           WriteMemory write_mem_callback,
                           ReadRegister read_reg_callback,
                           WriteRegister write_reg_callback) :
        EmulateInstruction (lldb::eByteOrderLittle, // Byte order for ARM
                            4,                      // Address size in byte
                            baton,
                            read_mem_callback,
                            write_mem_callback,
                            read_reg_callback,
                            write_reg_callback),
        m_inst_mode (eModeInvalid)
    {
    }

    virtual bool 
    ReadInstruction ();

    virtual bool
    EvaluateInstruction ();

    bool
    ConditionPassed ();

    uint32_t
    CurrentCond ();

protected:

    Mode m_inst_mode;
    uint32_t m_inst_cpsr;

};

}   // namespace lldb_private

#endif  // lldb_EmulateInstructionARM_h_
