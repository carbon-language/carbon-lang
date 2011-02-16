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

// ITSession - Keep track of the IT Block progression.
class ITSession
{
public:
    ITSession() : ITCounter(0), ITState(0) {}
    ~ITSession() {}

    // InitIT - Initializes ITCounter/ITState.
    bool InitIT(unsigned short bits7_0);

    // ITAdvance - Updates ITCounter/ITState as IT Block progresses.
    void ITAdvance();

    // InITBlock - Returns true if we're inside an IT Block.
    bool InITBlock();

    // LastInITBlock - Returns true if we're the last instruction inside an IT Block.
    bool LastInITBlock();

    // GetCond - Gets condition bits for the current thumb instruction.
    uint32_t GetCond();

private:
    uint32_t ITCounter; // Possible values: 0, 1, 2, 3, 4.
    uint32_t ITState;   // A2.5.2 Consists of IT[7:5] and IT[4:0] initially.
};

class EmulateInstructionARM : public EmulateInstruction
{
public: 
    typedef enum
    {
        eEncodingA1,
        eEncodingA2,
        eEncodingA3,
        eEncodingA4,
        eEncodingA5,
        eEncodingT1,
        eEncodingT2,
        eEncodingT3,
        eEncodingT4,
        eEncodingT5
    } ARMEncoding;
    

    static void
    Initialize ();
    
    static void
    Terminate ();

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
        m_inst_cpsr (0),
        m_it_session ()
    {
    }
    
    
    virtual bool
    SetArchitecture (const ArchSpec &arch);

    virtual bool 
    ReadInstruction ();

    virtual bool
    EvaluateInstruction ();

    uint32_t
    ArchVersion();

    bool
    ConditionPassed ();

    uint32_t
    CurrentCond ();

    // InITBlock - Returns true if we're in Thumb mode and inside an IT Block.
    bool InITBlock();

    // LastInITBlock - Returns true if we're in Thumb mode and the last instruction inside an IT Block.
    bool LastInITBlock();

    bool
    BranchWritePC(const Context &context, uint32_t addr);

    bool
    BXWritePC(Context &context, uint32_t addr);

    bool
    LoadWritePC(Context &context, uint32_t addr);

    bool
    ALUWritePC(Context &context, uint32_t addr);

    Mode
    CurrentInstrSet();

    bool
    SelectInstrSet(Mode arm_or_thumb);

    bool
    WriteBits32Unknown (int n);

    bool
    WriteBits32UnknownToMemory (lldb::addr_t address);
    
    bool
    UnalignedSupport();

    typedef struct
    {
        uint32_t result;
        uint8_t carry_out;
        uint8_t overflow;
    } AddWithCarryResult;

    AddWithCarryResult
    AddWithCarry(uint32_t x, uint32_t y, uint8_t carry_in);

protected:

    // Typedef for the callback function used during the emulation.
    // Pass along (ARMEncoding)encoding as the callback data.
    typedef enum
    {
        eSize16,
        eSize32
    } ARMInstrSize;

    typedef struct
    {
        uint32_t mask;
        uint32_t value;
        uint32_t variants;
        EmulateInstructionARM::ARMEncoding encoding;
        ARMInstrSize size;
        bool (EmulateInstructionARM::*callback) (EmulateInstructionARM::ARMEncoding encoding);
        const char *name;
    }  ARMOpcode;
    

    static ARMOpcode*
    GetARMOpcodeForInstruction (const uint32_t opcode);

    static ARMOpcode*
    GetThumbOpcodeForInstruction (const uint32_t opcode);

    // A8.6.123 PUSH
    bool
    EmulatePush (ARMEncoding encoding);
    
    // A8.6.122 POP
    bool
    EmulatePop (ARMEncoding encoding);
    
    // A8.6.8 ADD (SP plus immediate)
    bool
    EmulateAddRdSPImmediate (ARMEncoding encoding);

    // A8.6.97 MOV (register) -- Rd == r7|ip and Rm == sp
    bool
    EmulateMovRdSP (ARMEncoding encoding);

    // A8.6.97 MOV (register) -- move from r8-r15 to r0-r7
    bool
    EmulateMovLowHigh (ARMEncoding encoding);

    // A8.6.59 LDR (literal)
    bool
    EmulateLDRRtPCRelative (ARMEncoding encoding);

    // A8.6.8 ADD (SP plus immediate)
    bool
    EmulateAddSPImmediate (ARMEncoding encoding);

    // A8.6.9 ADD (SP plus register)
    bool
    EmulateAddSPRm (ARMEncoding encoding);

    // A8.6.23 BL, BLX (immediate)
    bool
    EmulateBLXImmediate (ARMEncoding encoding);

    // A8.6.24 BLX (register)
    bool
    EmulateBLXRm (ARMEncoding encoding);

    // A8.6.25 BX
    bool
    EmulateBXRm (ARMEncoding encoding);

    // A8.6.212 SUB (immediate, ARM) -- Rd == r7 and Rm == ip
    bool
    EmulateSubR7IPImmediate (ARMEncoding encoding);

    // A8.6.215 SUB (SP minus immediate) -- Rd == ip
    bool
    EmulateSubIPSPImmediate (ARMEncoding encoding);

    // A8.6.215 SUB (SP minus immediate)
    bool
    EmulateSubSPImmdiate (ARMEncoding encoding);

    // A8.6.194 STR (immediate, ARM) -- Rn == sp
    bool
    EmulateSTRRtSP (ARMEncoding encoding);

    // A8.6.355 VPUSH
    bool
    EmulateVPUSH (ARMEncoding encoding);

    // A8.6.354 VPOP
    bool
    EmulateVPOP (ARMEncoding encoding);

    // A8.6.218 SVC (previously SWI)
    bool
    EmulateSVC (ARMEncoding encoding);

    // A8.6.50 IT
    bool
    EmulateIT (ARMEncoding encoding);

    // A8.6.16 B
    bool
    EmulateB (ARMEncoding encoding);
    
    // A8.6.27 CBNZ, CBZ
    bool
    EmulateCB (ARMEncoding encoding);

    // A8.6.6 ADD (register) -- Encoding T2
    bool
    EmulateAddRdnRm (ARMEncoding encoding);

    // A8.6.97 MOV (register)
    bool
    EmulateMovRdRm (ARMEncoding encoding);

    // A8.6.96 MOV (immediate)
    bool
    EmulateMovRdImm (ARMEncoding encoding);

    // A8.6.106 MVN (immediate)
    bool
    EmulateMvnRdImm (ARMEncoding encoding);

    // A8.6.35 CMP (immediate)
    bool
    EmulateCmpRnImm (ARMEncoding encoding);

    // A8.6.36 CMP (register)
    bool
    EmulateCmpRnRm (ARMEncoding encoding);

    // A8.6.14 ASR (immediate)
    bool
    EmulateASRImm (ARMEncoding encoding);

    // A8.6.15 ASR (register)
    bool
    EmulateASRReg (ARMEncoding encoding);

    // A8.6.53 LDM/LDMIA/LDMFD
    bool
    EmulateLDM (ARMEncoding encoding);
    
    // A8.6.54 LDMDA/LDMFA
    bool
    EmulateLDMDA (ARMEncoding encoding);
    
    // A8.6.55 LDMDB/LDMEA
    bool 
    EmulateLDMDB (ARMEncoding encoding);
    
    // A8.6.56 LDMIB/LDMED
    bool
    EmulateLDMIB (ARMEncoding encoding);

    bool
    EmulateLDRRtRnImm (ARMEncoding encoding);

    // A8.6.188 STM/STMIA/STMEA
    bool
    EmulateSTM (ARMEncoding encoding);
    
    // A8.6.189 STMDA/STMED
    bool
    EmulateSTMDA (ARMEncoding encoding);
    
    // A8.6.190 STMDB/STMFD
    bool
    EmulateSTMDB (ARMEncoding encoding);
    
    // A8.6.191 STMIB/STMFA
    bool
    EmulateSTMIB (ARMEncoding encoding);
    
    // A8.6.192 STR (immediate, Thumb)
    bool
    EmulateSTRThumb(ARMEncoding encoding);
    
    // A8.6.194 STR (register)
    bool
    EmulateSTRRegister (ARMEncoding encoding);
    
    uint32_t m_arm_isa;
    Mode m_inst_mode;
    uint32_t m_inst_cpsr;
    uint32_t m_new_inst_cpsr; // This can get updated by the opcode.
    ITSession m_it_session;
};

}   // namespace lldb_private

#endif  // lldb_EmulateInstructionARM_h_
