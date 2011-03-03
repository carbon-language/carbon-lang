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
#include "Plugins/Process/Utility/ARMDefines.h"

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
    BadMode (uint32_t mode);
    
    bool
    CurrentModeIsPrivileged ();
    
    void
    CPSRWriteByInstr (uint32_t value, uint32_t bytemask, bool affect_execstate);
    
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

    // Helper method to read the content of an ARM core register.
    uint32_t
    ReadCoreReg (uint32_t regnum, bool *success);

    // See A8.6.96 MOV (immediate) Operation.
    // Default arguments are specified for carry and overflow parameters, which means
    // not to update the respective flags even if setflags is true.
    bool
    WriteCoreRegOptionalFlags (Context &context,
                               const uint32_t result,
                               const uint32_t Rd,
                               bool setflags,
                               const uint32_t carry = ~0u,
                               const uint32_t overflow = ~0u);

    bool
    WriteCoreReg (Context &context,
                  const uint32_t result,
                  const uint32_t Rd)
    {
        // Don't set the flags.
        return WriteCoreRegOptionalFlags(context, result, Rd, false);
    }

    // See A8.6.35 CMP (immediate) Operation.
    // Default arguments are specified for carry and overflow parameters, which means
    // not to update the respective flags.
    bool
    WriteFlags (Context &context,
                const uint32_t result,
                const uint32_t carry = ~0u,
                const uint32_t overflow = ~0u);

    inline uint64_t
    MemARead (EmulateInstruction::Context &context, 
              lldb::addr_t address, 
              uint32_t size, 
              uint64_t fail_value, 
              bool *success_ptr)
    {
        // This is a stub function corresponding to "MemA[]" in the ARM manual pseudocode, for 
        // aligned reads from memory.  Since we are not trying to write a full hardware simulator, and since
        // we are running in User mode (rather than Kernel mode) and therefore won't have access to many of the
        // system registers we would need in order to fully implement this function, we will just call
        // ReadMemoryUnsigned from here.  In the future, if we decide we do need to do more faithful emulation of
        // the hardware, we can update this function appropriately.
        
        return ReadMemoryUnsigned (context, address, size, fail_value, success_ptr);
    }
    
    inline bool
    MemAWrite (EmulateInstruction::Context &context,
               lldb::addr_t address,
               uint64_t data_val,
               uint32_t size)
    
    {
        // This is a stub function corresponding to "MemA[]" in the ARM manual pseudocode, for 
        // aligned writes to memory.  Since we are not trying to write a full hardware simulator, and since
        // we are running in User mode (rather than Kernel mode) and therefore won't have access to many of the
        // system registers we would need in order to fully implement this function, we will just call
        // WriteMemoryUnsigned from here.  In the future, if we decide we do need to do more faithful emulation of
        // the hardware, we can update this function appropriately.
        
        return WriteMemoryUnsigned (context, address, data_val, size);
    }
    
    
    inline uint64_t
    MemURead (EmulateInstruction::Context &context,
              lldb::addr_t address,
              uint32_t size,
              uint64_t fail_value,
              bool *success_ptr)
    {
        // This is a stub function corresponding to "MemU[]" in the ARM manual pseudocode, for 
        // unaligned reads from memory.  Since we are not trying to write a full hardware simulator, and since
        // we are running in User mode (rather than Kernel mode) and therefore won't have access to many of the
        // system registers we would need in order to fully implement this function, we will just call
        // ReadMemoryUnsigned from here.  In the future, if we decide we do need to do more faithful emulation of
        // the hardware, we can update this function appropriately.
        
        return ReadMemoryUnsigned (context, address, size, fail_value, success_ptr);
    }
    
    inline bool
    MemUWrite (EmulateInstruction::Context &context, 
               lldb::addr_t address,
               uint64_t data_val,
               uint32_t size)
    
    {
        // This is a stub function corresponding to "MemU[]" in the ARM manual pseudocode, for 
        // unaligned writes to memory.  Since we are not trying to write a full hardware simulator, and since
        // we are running in User mode (rather than Kernel mode) and therefore won't have access to many of the
        // system registers we would need in order to fully implement this function, we will just call
        // WriteMemoryUnsigned from here.  In the future, if we decide we do need to do more faithful emulation of
        // the hardware, we can update this function appropriately.
        
        return WriteMemoryUnsigned (context, address, data_val, size);
    }

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
    EmulatePUSH (ARMEncoding encoding);
    
    // A8.6.122 POP
    bool
    EmulatePOP (ARMEncoding encoding);
    
    // A8.6.8 ADD (SP plus immediate)
    bool
    EmulateADDRdSPImm (ARMEncoding encoding);

    // A8.6.97 MOV (register) -- Rd == r7|ip and Rm == sp
    bool
    EmulateMOVRdSP (ARMEncoding encoding);

    // A8.6.97 MOV (register) -- move from r8-r15 to r0-r7
    bool
    EmulateMOVLowHigh (ARMEncoding encoding);

    // A8.6.59 LDR (literal)
    bool
    EmulateLDRRtPCRelative (ARMEncoding encoding);

    // A8.6.8 ADD (SP plus immediate)
    bool
    EmulateADDSPImm (ARMEncoding encoding);

    // A8.6.9 ADD (SP plus register)
    bool
    EmulateADDSPRm (ARMEncoding encoding);

    // A8.6.23 BL, BLX (immediate)
    bool
    EmulateBLXImmediate (ARMEncoding encoding);

    // A8.6.24 BLX (register)
    bool
    EmulateBLXRm (ARMEncoding encoding);

    // A8.6.25 BX
    bool
    EmulateBXRm (ARMEncoding encoding);

    // A8.6.26 BXJ
    bool
    EmulateBXJRm (ARMEncoding encoding);

    // A8.6.212 SUB (immediate, ARM) -- Rd == r7 and Rm == ip
    bool
    EmulateSUBR7IPImm (ARMEncoding encoding);

    // A8.6.215 SUB (SP minus immediate) -- Rd == ip
    bool
    EmulateSUBIPSPImm (ARMEncoding encoding);

    // A8.6.215 SUB (SP minus immediate)
    bool
    EmulateSUBSPImm (ARMEncoding encoding);

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

    // A8.6.226 TBB, TBH
    bool
    EmulateTB (ARMEncoding encoding);

    // A8.6.4 ADD (immediate, Thumb)
    bool
    EmulateADDImmThumb (ARMEncoding encoding);

    // A8.6.5 ADD (immediate, ARM)
    bool
    EmulateADDImmARM (ARMEncoding encoding);
    
    // A8.6.6 ADD (register)
    bool
    EmulateADDReg (ARMEncoding encoding);

    // A8.6.97 MOV (register)
    bool
    EmulateMOVRdRm (ARMEncoding encoding);

    // A8.6.96 MOV (immediate)
    bool
    EmulateMOVRdImm (ARMEncoding encoding);

    // A8.6.35 CMP (immediate)
    bool
    EmulateCMPImm (ARMEncoding encoding);

    // A8.6.36 CMP (register)
    bool
    EmulateCMPReg (ARMEncoding encoding);

    // A8.6.14 ASR (immediate)
    bool
    EmulateASRImm (ARMEncoding encoding);

    // A8.6.15 ASR (register)
    bool
    EmulateASRReg (ARMEncoding encoding);

    // A8.6.88 LSL (immediate)
    bool
    EmulateLSLImm (ARMEncoding encoding);

    // A8.6.89 LSL (register)
    bool
    EmulateLSLReg (ARMEncoding encoding);

    // A8.6.90 LSR (immediate)
    bool
    EmulateLSRImm (ARMEncoding encoding);

    // A8.6.91 LSR (register)
    bool
    EmulateLSRReg (ARMEncoding encoding);

    // A8.6.139 ROR (immediate)
    bool
    EmulateRORImm (ARMEncoding encoding);

    // A8.6.140 ROR (register)
    bool
    EmulateRORReg (ARMEncoding encoding);

    // A8.6.141 RRX
    bool
    EmulateRRX (ARMEncoding encoding);

    // Helper method for ASR, LSL, LSR, ROR (immediate), and RRX
    bool
    EmulateShiftImm (ARMEncoding encoding, ARM_ShifterType shift_type);

    // Helper method for ASR, LSL, LSR, and ROR (register)
    bool
    EmulateShiftReg (ARMEncoding encoding, ARM_ShifterType shift_type);

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

    // A8.6.57 LDR (immediate, Thumb) -- Encoding T1
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
    
    // A8.6.195 STRB (immediate, Thumb)
    bool
    EmulateSTRBThumb (ARMEncoding encoding);
    
    // A8.6.1 ADC (immediate)
    bool
    EmulateADCImm (ARMEncoding encoding);
    
    // A8.6.2 ADC (Register)
    bool
    EmulateADCReg (ARMEncoding encoding);
    
    // A8.6.10 ADR
    bool
    EmulateADR (ARMEncoding encoding);
    
    // A8.6.11 AND (immediate)
    bool
    EmulateANDImm (ARMEncoding encoding);
    
    // A8.6.12 AND (register)
    bool
    EmulateANDReg (ARMEncoding encoding);
    
    // A8.6.19 BIC (immediate)
    bool
    EmulateBICImm (ARMEncoding encoding);
    
    // A8.6.20 BIC (register)
    bool
    EmulateBICReg (ARMEncoding encoding);
    
    // A8.6.26 BXJ
    bool
    EmulateBXJ (ARMEncoding encoding);
    
    // A8.6.32 CMN (immediate)
    bool
    EmulateCMNImm (ARMEncoding encoding);
    
    // A8.6.33 CMN (register)
    bool
    EmulateCMNReg (ARMEncoding encoding);
    
    // A8.6.44 EOR (immediate)
    bool
    EmulateEORImm (ARMEncoding encoding);
    
    // A8.6.45 EOR (register)
    bool
    EmulateEORReg (ARMEncoding encoding);
    
    // A8.6.58 LDR (immediate, ARM) - Encoding A1
    bool
    EmulateLDRImmediateARM (ARMEncoding encoding);
    
    // A8.6.60 LDR (register) - Encoding T1, T2, A1
    bool
    EmulateLDRRegister (ARMEncoding encoding);
    
    // A8.6.61 LDRB (immediate, Thumb) - Encoding T1, T2
    bool
    EmulateLDRBImmediate (ARMEncoding encoding);
    
    // A8.6.63 LDRB (literal) - Encoding T1
    bool
    EmulateLDRBLiteral (ARMEncoding encoding);
    
    // A8.6.64 LDRB (register) - Encoding T1
    bool
    EmulateLDRBRegister (ARMEncoding encoding);
    
    // A8.6.73 LDRH (immediate, Thumb) - Encoding T1, T2
    bool
    EmulateLDRHImmediate (ARMEncoding encoding);
    
    // A8.6.75 LDRH (literal) - Encoding T1
    bool
    EmulateLDRHLiteral (ARMEncoding encoding);
    
    // A8.6.76 LDRH (register) - Encoding T1, T2
    bool
    EmulateLDRHRegister (ARMEncoding encoding);
    
    // A8.6.78 LDRSB (immediate) - Encoding T1
    bool
    EmulateLDRSBImmediate (ARMEncoding encoding);
    
    // A8.6.79 LDRSB (literal) - Encoding T1
    bool
    EmulateLDRSBLiteral (ARMEncoding encoding);
    
    // A8.6.80 LDRSB (register) - Encoding T1, T2
    bool
    EmulateLDRSBRegister (ARMEncoding encoding);
    
    // A8.6.82 LDRSH (immediate) - Encoding T1
    bool
    EmulateLDRSHImmediate (ARMEncoding encoding);
    
    // A8.6.83 LDRSH (literal) - Encoding T1
    bool
    EmulateLDRSHLiteral (ARMEncoding encoding);
    
    // A8.6.84 LDRSH (register) - Encoding T1, T2
    bool
    EmulateLDRSHRegister (ARMEncoding encoding);
    
    // A8.6.105 MUL
    bool
    EmulateMUL (ARMEncoding encoding);
    
    // A8.6.106 MVN (immediate)
    bool
    EmulateMVNImm (ARMEncoding encoding);
    
    // A8.6.107 MVN (register)
    bool
    EmulateMVNReg (ARMEncoding encoding);
    
    // A8.6.113 ORR (immediate)
    bool
    EmulateORRImm (ARMEncoding encoding);

    // A8.6.114 ORR (register)
    bool
    EmulateORRReg (ARMEncoding encoding);

    // A8.6.117 PLD (immediate, literal) - Encoding T1, T2, T3, A1
    bool
    EmulatePLDImmediate (ARMEncoding encoding);
    
    // A8.6.119 PLI (immediate,literal) - Encoding T3, A1
    bool
    EmulatePLIImmediate (ARMEncoding encoding);
    
    // A8.6.120 PLI (register) - Encoding T1, A1
    bool
    EmulatePLIRegister (ARMEncoding encoding);
    
    // A8.6.141 RSB (immediate)
    bool
    EmulateRSBImm (ARMEncoding encoding);
    
    // A8.6.142 RSB (register)
    bool
    EmulateRSBReg (ARMEncoding encoding);
    
    // A8.6.144 RSC (immediate)
    bool
    EmulateRSCImm (ARMEncoding encoding);
    
    // A8.6.145 RSC (register)
    bool
    EmulateRSCReg (ARMEncoding encoding);
    
    // A8.6.150 SBC (immediate)
    bool
    EmulateSBCImm (ARMEncoding encoding);
    
    // A8.6.151 SBC (register)
    bool
    EmulateSBCReg (ARMEncoding encoding);
    
    // A8.6.211 SUB (immediate, Thumb)
    bool
    EmulateSUBImmThumb (ARMEncoding encoding);
    
    // A8.6.212 SUB (immediate, ARM)
    bool
    EmulateSUBImmARM (ARMEncoding encoding);
    
    // A8.6.222 SXTB  - Encoding T1
    bool
    EmulateSXTB (ARMEncoding encoding);
    
    // A8.6.224 SXTH  - EncodingT1
    bool
    EmulateSXTH (ARMEncoding encoding);
    
    // A8.6.227 TEQ (immediate) - Encoding A1
    bool
    EmulateTEQImm (ARMEncoding encoding);
    
    // A8.6.228 TEQ (register)  - Encoding A1
    bool
    EmulateTEQReg (ARMEncoding encoding);
    
    // A8.6.230 TST (immediate) - Encoding A1
    bool
    EmulateTSTImm (ARMEncoding encoding);
    
    // A8.6.231 TST (register)  - Encoding T1, A1
    bool
    EmulateTSTReg (ARMEncoding encoding);
    
    // A8.6.262 UXTB  - Encoding T1
    bool
    EmulateUXTB (ARMEncoding encoding);
    
    // A8.6.264 UXTH  - Encoding T1
    bool
    EmulateUXTH (ARMEncoding encoding);
    
    // B6.1.8  RFE
    bool
    EmulateRFE (ARMEncoding encoding);
     
    uint32_t m_arm_isa;
    Mode m_inst_mode;
    uint32_t m_inst_cpsr;
    uint32_t m_new_inst_cpsr; // This can get updated by the opcode.
    ITSession m_it_session;
};

}   // namespace lldb_private

#endif  // lldb_EmulateInstructionARM_h_
