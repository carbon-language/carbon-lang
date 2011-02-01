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

#include "lldb/Core/EmulateInstruction.h"
#include "lldb/Core/Error.h"

namespace lldb_private {

class EmulateInstructionARM : public EmulateInstruction
{
public: 

    virtual const char *
    GetPluginName()
    {
        return "EmulateInstructionARM";
    }

    virtual const char *
    GetShortPluginName()
    {
        return "lldb.emulate-instruction.arm";
    }

    virtual uint32_t
    GetPluginVersion()
    {
        return 1;
    }

    virtual void
    GetPluginCommandHelp (const char *command, Stream *strm)
    {
    }

    virtual lldb_private::Error
    ExecutePluginCommand (Args &command, Stream *strm)
    {
        Error error;
        error.SetErrorString("no plug-in commands are supported");
        return error;
    }

    virtual Log *
    EnablePluginLogging (Stream *strm, Args &command)
    {
        return NULL;
    }

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
        m_arm_isa (0),
        m_inst_mode (eModeInvalid),
        m_inst_cpsr (0)
    {
    }
    
    
    virtual bool
    SetTargetTriple (const ConstString &triple);

    virtual bool 
    ReadInstruction ();

    virtual bool
    EvaluateInstruction ();

    bool
    ConditionPassed ();

    uint32_t
    CurrentCond ();

protected:
    uint32_t m_arm_isa;
    Mode m_inst_mode;
    uint32_t m_inst_cpsr;

};

}   // namespace lldb_private

#endif  // lldb_EmulateInstructionARM_h_
