//===-- EmulateInstructionMIPS64.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef EmulateInstructionMIPS64_h_
#define EmulateInstructionMIPS64_h_

namespace llvm
{
    class MCDisassembler;
    class MCSubtargetInfo;
    class MCRegisterInfo;
    class MCAsmInfo;
    class MCContext;
    class MCInstrInfo;
    class MCInst;
}

#include "lldb/Core/EmulateInstruction.h"
#include "lldb/Core/Error.h"
#include "lldb/Interpreter/OptionValue.h"

class EmulateInstructionMIPS64 : public lldb_private::EmulateInstruction
{
public: 
    static void
    Initialize ();

    static void
    Terminate ();

    static lldb_private::ConstString
    GetPluginNameStatic ();
    
    static const char *
    GetPluginDescriptionStatic ();
    
    static lldb_private::EmulateInstruction *
    CreateInstance (const lldb_private::ArchSpec &arch, 
                    lldb_private::InstructionType inst_type);

    static bool
    SupportsEmulatingInstructionsOfTypeStatic (lldb_private::InstructionType inst_type)
    {
        switch (inst_type)
        {
            case lldb_private::eInstructionTypeAny:
            case lldb_private::eInstructionTypePrologueEpilogue:
            case lldb_private::eInstructionTypePCModifying:
                return true;

            case lldb_private::eInstructionTypeAll:
                return false;
        }
        return false;
    }

    virtual lldb_private::ConstString
    GetPluginName();

    virtual lldb_private::ConstString
    GetShortPluginName()
    {
        return GetPluginNameStatic();
    }

    virtual uint32_t
    GetPluginVersion()
    {
        return 1;
    }

    bool
    SetTargetTriple (const lldb_private::ArchSpec &arch);
    
    EmulateInstructionMIPS64 (const lldb_private::ArchSpec &arch);

    virtual bool
    SupportsEmulatingInstructionsOfType (lldb_private::InstructionType inst_type)
    {
        return SupportsEmulatingInstructionsOfTypeStatic (inst_type);
    }

    virtual bool 
    ReadInstruction ();
    
    virtual bool
    EvaluateInstruction (uint32_t evaluate_options);
    
    virtual bool
    TestEmulation (lldb_private::Stream *out_stream, 
                   lldb_private::ArchSpec &arch, 
                   lldb_private::OptionValueDictionary *test_data)
    {
        return false;
    }

    virtual bool
    GetRegisterInfo (lldb::RegisterKind reg_kind,
                     uint32_t reg_num, 
                     lldb_private::RegisterInfo &reg_info);

    virtual bool
    CreateFunctionEntryUnwind (lldb_private::UnwindPlan &unwind_plan);


protected:

    typedef struct
    {
        const char *op_name;
        bool (EmulateInstructionMIPS64::*callback) (llvm::MCInst& insn);
        const char *insn_name;
    }  MipsOpcode;
    
    static MipsOpcode*
    GetOpcodeForInstruction (const char *op_name);

    bool
    Emulate_DADDiu (llvm::MCInst& insn);

    bool
    Emulate_SD (llvm::MCInst& insn);

    bool
    Emulate_LD (llvm::MCInst& insn);

    bool
    Emulate_B (llvm::MCInst& insn);
    
    bool
    Emulate_BAL (llvm::MCInst& insn);

    bool
    Emulate_BALC (llvm::MCInst& insn);

    bool
    nonvolatile_reg_p (uint64_t regnum);

    const char *
    GetRegisterName (unsigned reg_num, bool altnernate_name);

private:
    std::unique_ptr<llvm::MCDisassembler>   m_disasm;
    std::unique_ptr<llvm::MCSubtargetInfo>  m_subtype_info;
    std::unique_ptr<llvm::MCRegisterInfo>   m_reg_info;
    std::unique_ptr<llvm::MCAsmInfo>        m_asm_info;
    std::unique_ptr<llvm::MCContext>        m_context;
    std::unique_ptr<llvm::MCInstrInfo>      m_insn_info;
};

#endif  // EmulateInstructionMIPS64_h_
