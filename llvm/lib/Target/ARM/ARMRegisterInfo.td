//===- ARMRegisterInfo.td - ARM Register defs -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the ARM register file
//===----------------------------------------------------------------------===//

// Registers are identified with 4-bit ID numbers.
class ARMReg<bits<4> num, string n, list<Register> subregs = []> : Register<n> {
  field bits<4> Num;
  let Namespace = "ARM";
  let SubRegs = subregs;
}

class ARMFReg<bits<6> num, string n> : Register<n> {
  field bits<6> Num;
  let Namespace = "ARM";
}

// Integer registers
def R0  : ARMReg< 0, "r0">,  DwarfRegNum<[0]>;
def R1  : ARMReg< 1, "r1">,  DwarfRegNum<[1]>;
def R2  : ARMReg< 2, "r2">,  DwarfRegNum<[2]>;
def R3  : ARMReg< 3, "r3">,  DwarfRegNum<[3]>;
def R4  : ARMReg< 4, "r4">,  DwarfRegNum<[4]>;
def R5  : ARMReg< 5, "r5">,  DwarfRegNum<[5]>;
def R6  : ARMReg< 6, "r6">,  DwarfRegNum<[6]>;
def R7  : ARMReg< 7, "r7">,  DwarfRegNum<[7]>;
def R8  : ARMReg< 8, "r8">,  DwarfRegNum<[8]>;
def R9  : ARMReg< 9, "r9">,  DwarfRegNum<[9]>;
def R10 : ARMReg<10, "r10">, DwarfRegNum<[10]>;
def R11 : ARMReg<11, "r11">, DwarfRegNum<[11]>;
def R12 : ARMReg<12, "r12">, DwarfRegNum<[12]>;
def SP  : ARMReg<13, "sp">,  DwarfRegNum<[13]>;
def LR  : ARMReg<14, "lr">,  DwarfRegNum<[14]>;
def PC  : ARMReg<15, "pc">,  DwarfRegNum<[15]>;

// Float registers
def S0  : ARMFReg< 0, "s0">;  def S1  : ARMFReg< 1, "s1">;
def S2  : ARMFReg< 2, "s2">;  def S3  : ARMFReg< 3, "s3">;
def S4  : ARMFReg< 4, "s4">;  def S5  : ARMFReg< 5, "s5">;
def S6  : ARMFReg< 6, "s6">;  def S7  : ARMFReg< 7, "s7">;
def S8  : ARMFReg< 8, "s8">;  def S9  : ARMFReg< 9, "s9">;
def S10 : ARMFReg<10, "s10">; def S11 : ARMFReg<11, "s11">;
def S12 : ARMFReg<12, "s12">; def S13 : ARMFReg<13, "s13">;
def S14 : ARMFReg<14, "s14">; def S15 : ARMFReg<15, "s15">;
def S16 : ARMFReg<16, "s16">; def S17 : ARMFReg<17, "s17">;
def S18 : ARMFReg<18, "s18">; def S19 : ARMFReg<19, "s19">;
def S20 : ARMFReg<20, "s20">; def S21 : ARMFReg<21, "s21">;
def S22 : ARMFReg<22, "s22">; def S23 : ARMFReg<23, "s23">;
def S24 : ARMFReg<24, "s24">; def S25 : ARMFReg<25, "s25">;
def S26 : ARMFReg<26, "s26">; def S27 : ARMFReg<27, "s27">;
def S28 : ARMFReg<28, "s28">; def S29 : ARMFReg<29, "s29">;
def S30 : ARMFReg<30, "s30">; def S31 : ARMFReg<31, "s31">;
def SDummy : ARMFReg<63, "sINVALID">;

// Aliases of the F* registers used to hold 64-bit fp values (doubles)
def D0  : ARMReg< 0,  "d0", [S0,   S1]>;
def D1  : ARMReg< 1,  "d1", [S2,   S3]>;
def D2  : ARMReg< 2,  "d2", [S4,   S5]>;
def D3  : ARMReg< 3,  "d3", [S6,   S7]>;
def D4  : ARMReg< 4,  "d4", [S8,   S9]>;
def D5  : ARMReg< 5,  "d5", [S10, S11]>;
def D6  : ARMReg< 6,  "d6", [S12, S13]>;
def D7  : ARMReg< 7,  "d7", [S14, S15]>;
def D8  : ARMReg< 8,  "d8", [S16, S17]>;
def D9  : ARMReg< 9,  "d9", [S18, S19]>;
def D10 : ARMReg<10, "d10", [S20, S21]>;
def D11 : ARMReg<11, "d11", [S22, S23]>;
def D12 : ARMReg<12, "d12", [S24, S25]>;
def D13 : ARMReg<13, "d13", [S26, S27]>;
def D14 : ARMReg<14, "d14", [S28, S29]>;
def D15 : ARMReg<15, "d15", [S30, S31]>;

// VFP3 defines 16 additional double registers
def D16 : ARMFReg<16, "d16">; def D17 : ARMFReg<17, "d17">;
def D18 : ARMFReg<18, "d18">; def D19 : ARMFReg<19, "d19">;
def D20 : ARMFReg<20, "d20">; def D21 : ARMFReg<21, "d21">;
def D22 : ARMFReg<22, "d22">; def D23 : ARMFReg<23, "d23">;
def D24 : ARMFReg<24, "d24">; def D25 : ARMFReg<25, "d25">;
def D26 : ARMFReg<26, "d26">; def D27 : ARMFReg<27, "d27">;
def D28 : ARMFReg<28, "d28">; def D29 : ARMFReg<29, "d29">;
def D30 : ARMFReg<30, "d30">; def D31 : ARMFReg<31, "d31">;
def DDummy : ARMFReg<31, "dINVALID">;

// Advanced SIMD (NEON) defines 16 quad-word aliases
def Q0  : ARMReg< 0,  "q0", [D0,   D1]>;
def Q1  : ARMReg< 1,  "q1", [D2,   D3]>;
def Q2  : ARMReg< 2,  "q2", [D4,   D5]>;
def Q3  : ARMReg< 3,  "q3", [D6,   D7]>;
def Q4  : ARMReg< 4,  "q4", [D8,   D9]>;
def Q5  : ARMReg< 5,  "q5", [D10, D11]>;
def Q6  : ARMReg< 6,  "q6", [D12, D13]>;
def Q7  : ARMReg< 7,  "q7", [D14, D15]>;
def Q8  : ARMReg< 8,  "q8", [D16, D17]>;
def Q9  : ARMReg< 9,  "q9", [D18, D19]>;
def Q10 : ARMReg<10, "q10", [D20, D21]>;
def Q11 : ARMReg<11, "q11", [D22, D23]>;
def Q12 : ARMReg<12, "q12", [D24, D25]>;
def Q13 : ARMReg<13, "q13", [D26, D27]>;
def Q14 : ARMReg<14, "q14", [D28, D29]>;
def Q15 : ARMReg<15, "q15", [D30, D31]>;
def QDummy : ARMFReg<16, "qINVALID">;

// Pseudo 256-bit registers to represent pairs of Q registers. These should
// never be present in the emitted code.
// These are used for NEON load / store instructions, e.g. vld4, vst3.
// NOTE: It's possible to define more QQ registers since technical the
// starting D register number doesn't have to be multiple of 4. e.g. 
// D1, D2, D3, D4 would be a legal quad. But that would make the sub-register
// stuffs very messy.
def QQ0 : ARMReg<0, "qq0", [Q0,  Q1]>;
def QQ1 : ARMReg<1, "qq1", [Q2,  Q3]>;
def QQ2 : ARMReg<2, "qq2", [Q4,  Q5]>;
def QQ3 : ARMReg<3, "qq3", [Q6,  Q7]>;
def QQ4 : ARMReg<4, "qq4", [Q8,  Q9]>;
def QQ5 : ARMReg<5, "qq5", [Q10, Q11]>;
def QQ6 : ARMReg<6, "qq6", [Q12, Q13]>;
def QQ7 : ARMReg<7, "qq7", [Q14, Q15]>;

// Pseudo 512-bit registers to represent four consecutive Q registers.
def QQQQ0 : ARMReg<0, "qqqq0", [QQ0, QQ1]>;
def QQQQ1 : ARMReg<1, "qqqq1", [QQ2, QQ3]>;
def QQQQ2 : ARMReg<2, "qqqq2", [QQ4, QQ5]>;
def QQQQ3 : ARMReg<3, "qqqq3", [QQ6, QQ7]>;

// Current Program Status Register.
def CPSR  : ARMReg<0, "cpsr">;

def FPSCR : ARMReg<1, "fpscr">;

// Register classes.
//
// pc  == Program Counter
// lr  == Link Register
// sp  == Stack Pointer
// r12 == ip (scratch)
// r7  == Frame Pointer (thumb-style backtraces)
// r9  == May be reserved as Thread Register
// r11 == Frame Pointer (arm-style backtraces)
// r10 == Stack Limit
//
def GPR : RegisterClass<"ARM", [i32], 32, [R0, R1, R2, R3, R4, R5, R6,
                                           R7, R8, R9, R10, R11, R12,
                                           SP, LR, PC]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    // FP is R11, R9 is available.
    static const unsigned ARM_GPR_AO_1[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6, ARM::R7,
      ARM::R8, ARM::R9, ARM::R10,
      ARM::R11 };
    // FP is R11, R9 is not available.
    static const unsigned ARM_GPR_AO_2[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6, ARM::R7,
      ARM::R8, ARM::R10,
      ARM::R11 };
    // FP is R7, R9 is available as non-callee-saved register.
    // This is used by Darwin.
    static const unsigned ARM_GPR_AO_3[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R9, ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6,
      ARM::R8, ARM::R10,ARM::R11,ARM::R7 };
    // FP is R7, R9 is not available.
    static const unsigned ARM_GPR_AO_4[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6,
      ARM::R8, ARM::R10,ARM::R11,
      ARM::R7 };
    // FP is R7, R9 is available as callee-saved register.
    // This is used by non-Darwin platform in Thumb mode.
    static const unsigned ARM_GPR_AO_5[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6,
      ARM::R8, ARM::R9, ARM::R10,ARM::R11,ARM::R7 };

    // For Thumb1 mode, we don't want to allocate hi regs at all, as we
    // don't know how to spill them. If we make our prologue/epilogue code
    // smarter at some point, we can go back to using the above allocation
    // orders for the Thumb1 instructions that know how to use hi regs.
    static const unsigned THUMB_GPR_AO[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R4, ARM::R5, ARM::R6, ARM::R7 };

    GPRClass::iterator
    GPRClass::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      if (Subtarget.isThumb1Only())
        return THUMB_GPR_AO;
      if (Subtarget.isTargetDarwin()) {
        if (Subtarget.isR9Reserved())
          return ARM_GPR_AO_4;
        else
          return ARM_GPR_AO_3;
      } else {
        if (Subtarget.isR9Reserved())
          return ARM_GPR_AO_2;
        else if (Subtarget.isThumb())
          return ARM_GPR_AO_5;
        else
          return ARM_GPR_AO_1;
      }
    }

    GPRClass::iterator
    GPRClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      GPRClass::iterator I;

      if (Subtarget.isThumb1Only()) {
        I = THUMB_GPR_AO + (sizeof(THUMB_GPR_AO)/sizeof(unsigned));
        // Mac OS X requires FP not to be clobbered for backtracing purpose.
        return (Subtarget.isTargetDarwin() || RI->hasFP(MF)) ? I-1 : I;
      }

      if (Subtarget.isTargetDarwin()) {
        if (Subtarget.isR9Reserved())
          I = ARM_GPR_AO_4 + (sizeof(ARM_GPR_AO_4)/sizeof(unsigned));
        else
          I = ARM_GPR_AO_3 + (sizeof(ARM_GPR_AO_3)/sizeof(unsigned));
      } else {
        if (Subtarget.isR9Reserved())
          I = ARM_GPR_AO_2 + (sizeof(ARM_GPR_AO_2)/sizeof(unsigned));
        else if (Subtarget.isThumb())
          I = ARM_GPR_AO_5 + (sizeof(ARM_GPR_AO_5)/sizeof(unsigned));
        else
          I = ARM_GPR_AO_1 + (sizeof(ARM_GPR_AO_1)/sizeof(unsigned));
      }

      // Mac OS X requires FP not to be clobbered for backtracing purpose.
      return (Subtarget.isTargetDarwin() || RI->hasFP(MF)) ? I-1 : I;
    }
  }];
}

// Thumb registers are R0-R7 normally. Some instructions can still use
// the general GPR register class above (MOV, e.g.)
def tGPR : RegisterClass<"ARM", [i32], 32, [R0, R1, R2, R3, R4, R5, R6, R7]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned THUMB_tGPR_AO[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R4, ARM::R5, ARM::R6, ARM::R7 };

    // FP is R7, only low registers available.
    tGPRClass::iterator
    tGPRClass::allocation_order_begin(const MachineFunction &MF) const {
      return THUMB_tGPR_AO;
    }

    tGPRClass::iterator
    tGPRClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      tGPRClass::iterator I =
        THUMB_tGPR_AO + (sizeof(THUMB_tGPR_AO)/sizeof(unsigned));
      // Mac OS X requires FP not to be clobbered for backtracing purpose.
      return (Subtarget.isTargetDarwin() || RI->hasFP(MF)) ? I-1 : I;
    }
  }];
}

// Scalar single precision floating point register class..
def SPR : RegisterClass<"ARM", [f32], 32, [S0, S1, S2, S3, S4, S5, S6, S7, S8,
  S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22,
  S23, S24, S25, S26, S27, S28, S29, S30, S31]>;

// Subset of SPR which can be used as a source of NEON scalars for 16-bit
// operations
def SPR_8 : RegisterClass<"ARM", [f32], 32,
                          [S0, S1,  S2,  S3,  S4,  S5,  S6,  S7,
                           S8, S9, S10, S11, S12, S13, S14, S15]>;

// Dummy f32 regclass to represent impossible subreg indices.
def SPR_INVALID : RegisterClass<"ARM", [f32], 32, [SDummy]> {
  let CopyCost = -1;
}

// Scalar double precision floating point / generic 64-bit vector register
// class.
// ARM requires only word alignment for double. It's more performant if it
// is double-word alignment though.
def DPR : RegisterClass<"ARM", [f64, v8i8, v4i16, v2i32, v1i64, v2f32], 64,
                        [D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
                         D8,  D9,  D10, D11, D12, D13, D14, D15,
                         D16, D17, D18, D19, D20, D21, D22, D23,
                         D24, D25, D26, D27, D28, D29, D30, D31]> {
  let SubRegClassList = [SPR_INVALID, SPR_INVALID];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    // VFP2
    static const unsigned ARM_DPR_VFP2[] = {
      ARM::D0,  ARM::D1,  ARM::D2,  ARM::D3,
      ARM::D4,  ARM::D5,  ARM::D6,  ARM::D7,
      ARM::D8,  ARM::D9,  ARM::D10, ARM::D11,
      ARM::D12, ARM::D13, ARM::D14, ARM::D15 };
    // VFP3
    static const unsigned ARM_DPR_VFP3[] = {
      ARM::D0,  ARM::D1,  ARM::D2,  ARM::D3,
      ARM::D4,  ARM::D5,  ARM::D6,  ARM::D7,
      ARM::D8,  ARM::D9,  ARM::D10, ARM::D11,
      ARM::D12, ARM::D13, ARM::D14, ARM::D15,
      ARM::D16, ARM::D17, ARM::D18, ARM::D19,
      ARM::D20, ARM::D21, ARM::D22, ARM::D23,
      ARM::D24, ARM::D25, ARM::D26, ARM::D27,
      ARM::D28, ARM::D29, ARM::D30, ARM::D31 };
    DPRClass::iterator
    DPRClass::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      if (Subtarget.hasVFP3())
        return ARM_DPR_VFP3;
      return ARM_DPR_VFP2;
    }

    DPRClass::iterator
    DPRClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      if (Subtarget.hasVFP3())
        return ARM_DPR_VFP3 + (sizeof(ARM_DPR_VFP3)/sizeof(unsigned));
      else
        return ARM_DPR_VFP2 + (sizeof(ARM_DPR_VFP2)/sizeof(unsigned));
    }
  }];
}

// Subset of DPR that are accessible with VFP2 (and so that also have
// 32-bit SPR subregs).
def DPR_VFP2 : RegisterClass<"ARM", [f64, v8i8, v4i16, v2i32, v1i64, v2f32], 64,
                             [D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
                              D8,  D9,  D10, D11, D12, D13, D14, D15]> {
  let SubRegClassList = [SPR, SPR];
}

// Subset of DPR which can be used as a source of NEON scalars for 16-bit
// operations
def DPR_8 : RegisterClass<"ARM", [f64, v8i8, v4i16, v2i32, v1i64, v2f32], 64,
                          [D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7]> {
  let SubRegClassList = [SPR_8, SPR_8];
}

// Dummy 64-bit regclass to represent impossible subreg indices.
def DPR_INVALID : RegisterClass<"ARM",
                                [f64, v8i8, v4i16, v2i32, v1i64, v2f32], 64,
                                [DDummy]> {
  let CopyCost = -1;
}

// Generic 128-bit vector register class.
def QPR : RegisterClass<"ARM", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64], 128,
                        [Q0,  Q1,  Q2,  Q3,  Q4,  Q5,  Q6,  Q7,
                         Q8,  Q9,  Q10, Q11, Q12, Q13, Q14, Q15]> {
  let SubRegClassList = [SPR_INVALID, SPR_INVALID, SPR_INVALID, SPR_INVALID,
                         DPR, DPR, DPR_INVALID, DPR_INVALID,
                         DPR_INVALID, DPR_INVALID, DPR_INVALID, DPR_INVALID];
}

// Subset of QPR that have 32-bit SPR subregs.
def QPR_VFP2 : RegisterClass<"ARM", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
                             128,
                             [Q0,  Q1,  Q2,  Q3,  Q4,  Q5,  Q6,  Q7]> {
  let SubRegClassList = [SPR, SPR, SPR, SPR,
                         DPR_VFP2, DPR_VFP2, DPR_INVALID, DPR_INVALID,
                         DPR_INVALID, DPR_INVALID, DPR_INVALID, DPR_INVALID];
}

// Subset of QPR that have DPR_8 and SPR_8 subregs.
def QPR_8 : RegisterClass<"ARM", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
                           128,
                           [Q0,  Q1,  Q2,  Q3]> {
  let SubRegClassList = [SPR_8, SPR_8, SPR_8, SPR_8,
                         DPR_8, DPR_8, DPR_INVALID, DPR_INVALID,
                         DPR_INVALID, DPR_INVALID, DPR_INVALID, DPR_INVALID];
}

// Dummy 128-bit regclass to represent impossible subreg indices.
def QPR_INVALID : RegisterClass<"ARM",
                                [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64], 128,
                                [QDummy]> {
  let CopyCost = -1;
}

// Pseudo 256-bit vector register class to model pairs of Q registers
// (4 consecutive D registers).
def QQPR : RegisterClass<"ARM", [v4i64],
                         256,
                         [QQ0, QQ1, QQ2, QQ3, QQ4, QQ5, QQ6, QQ7]> {
  let SubRegClassList = [SPR_INVALID, SPR_INVALID, SPR_INVALID, SPR_INVALID,
                         DPR, DPR, DPR, DPR,
                         DPR_INVALID, DPR_INVALID, DPR_INVALID, DPR_INVALID,
                         QPR, QPR, QPR_INVALID, QPR_INVALID];
}

// Subset of QQPR that have 32-bit SPR subregs.
def QQPR_VFP2 : RegisterClass<"ARM", [v4i64],
                              256,
                              [QQ0, QQ1, QQ2, QQ3]> {
  let SubRegClassList = [SPR, SPR, SPR, SPR,
                         DPR_VFP2, DPR_VFP2, DPR_VFP2, DPR_VFP2,
                         DPR_INVALID, DPR_INVALID, DPR_INVALID, DPR_INVALID,
                         QPR_VFP2, QPR_VFP2, QPR_INVALID, QPR_INVALID];
}

// Pseudo 512-bit vector register class to model 4 consecutive Q registers
// (8 consecutive D registers).
def QQQQPR : RegisterClass<"ARM", [v8i64],
                         256,
                         [QQQQ0, QQQQ1, QQQQ2, QQQQ3]> {
  let SubRegClassList = [SPR_INVALID, SPR_INVALID, SPR_INVALID, SPR_INVALID,
                         DPR, DPR, DPR, DPR, DPR, DPR, DPR, DPR,
                         QPR, QPR, QPR, QPR];
}

// Condition code registers.
def CCR : RegisterClass<"ARM", [i32], 32, [CPSR]>;

//===----------------------------------------------------------------------===//
// Subregister Set Definitions... now that we have all of the pieces, define the
// sub registers for each register.
//

def arm_ssubreg_0 : PatLeaf<(i32 1)>;
def arm_ssubreg_1 : PatLeaf<(i32 2)>;
def arm_ssubreg_2 : PatLeaf<(i32 3)>;
def arm_ssubreg_3 : PatLeaf<(i32 4)>;

def arm_dsubreg_0 : PatLeaf<(i32 5)>;
def arm_dsubreg_1 : PatLeaf<(i32 6)>;
def arm_dsubreg_2 : PatLeaf<(i32 7)>;
def arm_dsubreg_3 : PatLeaf<(i32 8)>;
def arm_dsubreg_4 : PatLeaf<(i32 9)>;
def arm_dsubreg_5 : PatLeaf<(i32 10)>;
def arm_dsubreg_6 : PatLeaf<(i32 11)>;
def arm_dsubreg_7 : PatLeaf<(i32 12)>;

def arm_qsubreg_0 : PatLeaf<(i32 13)>;
def arm_qsubreg_1 : PatLeaf<(i32 14)>;
def arm_qsubreg_2 : PatLeaf<(i32 15)>;
def arm_qsubreg_3 : PatLeaf<(i32 16)>;

def arm_qqsubreg_0 : PatLeaf<(i32 17)>;
def arm_qqsubreg_1 : PatLeaf<(i32 18)>;


// S sub-registers of D registers.
def : SubRegSet<1, [D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
                    D8,  D9,  D10, D11, D12, D13, D14, D15],
                   [S0,  S2,  S4,  S6,  S8,  S10, S12, S14,
                    S16, S18, S20, S22, S24, S26, S28, S30]>;
def : SubRegSet<2, [D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
                    D8,  D9,  D10, D11, D12, D13, D14, D15],
                   [S1,  S3,  S5,  S7,  S9,  S11, S13, S15,
                    S17, S19, S21, S23, S25, S27, S29, S31]>;

// S sub-registers of Q registers.
def : SubRegSet<1, [Q0,  Q1,  Q2,  Q3,  Q4,  Q5,  Q6,  Q7],
                   [S0,  S4,  S8,  S12, S16, S20, S24, S28]>;
def : SubRegSet<2, [Q0,  Q1,  Q2,  Q3,  Q4,  Q5,  Q6,  Q7],
                   [S1,  S5,  S9,  S13, S17, S21, S25, S29]>;
def : SubRegSet<3, [Q0,  Q1,  Q2,  Q3,  Q4,  Q5,  Q6,  Q7],
                   [S2,  S6,  S10, S14, S18, S22, S26, S30]>;
def : SubRegSet<4, [Q0,  Q1,  Q2,  Q3,  Q4,  Q5,  Q6,  Q7],
                   [S3,  S7,  S11, S15, S19, S23, S27, S31]>;

// D sub-registers of Q registers.
def : SubRegSet<5, [Q0,  Q1,  Q2,  Q3,  Q4,  Q5,  Q6,  Q7,
                    Q8,  Q9,  Q10, Q11, Q12, Q13, Q14, Q15],
                   [D0,  D2,  D4,  D6,  D8,  D10, D12, D14,
                    D16, D18, D20, D22, D24, D26, D28, D30]>;
def : SubRegSet<6, [Q0,  Q1,  Q2,  Q3,  Q4,  Q5,  Q6,  Q7,
                    Q8,  Q9,  Q10, Q11, Q12, Q13, Q14, Q15],
                   [D1,  D3,  D5,  D7,  D9,  D11, D13, D15,
                    D17, D19, D21, D23, D25, D27, D29, D31]>;

// S sub-registers of QQ registers. Note there are no sub-indices
// for referencing S4 - S7, S12 - S15, and S20 - S23. It doesn't
// look like we need them.
def : SubRegSet<1, [QQ0, QQ1, QQ2, QQ3],
                   [S0,  S8,  S16, S24]>;
def : SubRegSet<2, [QQ0, QQ1, QQ2, QQ3],
                   [S1,  S9,  S17, S25]>;
def : SubRegSet<3, [QQ0, QQ1, QQ2, QQ3],
                   [S2,  S10, S18, S26]>;
def : SubRegSet<4, [QQ0, QQ1, QQ2, QQ3],
                   [S3,  S11, S19, S27]>;

// D sub-registers of QQ registers.
def : SubRegSet<5, [QQ0, QQ1, QQ2, QQ3, QQ4, QQ5, QQ6, QQ7],
                   [D0,  D4,  D8,  D12, D16, D20, D24, D28]>;
def : SubRegSet<6, [QQ0, QQ1, QQ2, QQ3, QQ4, QQ5, QQ6, QQ7],
                   [D1,  D5,  D9,  D13, D17, D21, D25, D29]>;
def : SubRegSet<7, [QQ0, QQ1, QQ2, QQ3, QQ4, QQ5, QQ6, QQ7],
                   [D2,  D6,  D10, D14, D18, D22, D26, D30]>;
def : SubRegSet<8, [QQ0, QQ1, QQ2, QQ3, QQ4, QQ5, QQ6, QQ7],
                   [D3,  D7,  D11, D15, D19, D23, D27, D31]>;

// Q sub-registers of QQ registers.
def : SubRegSet<13, [QQ0, QQ1, QQ2, QQ3, QQ4, QQ5, QQ6, QQ7],
                   [Q0,  Q2,  Q4,  Q6,  Q8,  Q10, Q12, Q14]>;
def : SubRegSet<14,[QQ0, QQ1, QQ2, QQ3, QQ4, QQ5, QQ6, QQ7],
                   [Q1,  Q3,  Q5,  Q7,  Q9,  Q11, Q13, Q15]>;


// D sub-registers of QQQQ registers.
def : SubRegSet<5,  [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [D0,    D8,    D16,   D24]>;
def : SubRegSet<6,  [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [D1,    D9,    D17,   D25]>;
def : SubRegSet<7,  [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [D2,    D10,   D18,   D26]>;
def : SubRegSet<8,  [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [D3,    D11,   D19,   D27]>;

def : SubRegSet<9,  [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [D4,    D12,   D20,   D28]>;
def : SubRegSet<10, [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [D5,    D13,   D21,   D29]>;
def : SubRegSet<11, [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [D6,    D14,   D22,   D30]>;
def : SubRegSet<12, [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [D7,    D15,   D23,   D31]>;

// Q sub-registers of QQQQQQQQ registers.
def : SubRegSet<13, [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [Q0,    Q4,    Q8,    Q12]>;
def : SubRegSet<14, [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [Q1,    Q5,    Q9,    Q13]>;
def : SubRegSet<15, [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [Q2,    Q6,    Q10,   Q14]>;
def : SubRegSet<16, [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [Q3,    Q7,    Q11,   Q15]>;

// QQ sub-registers of QQQQQQQQ registers.
def : SubRegSet<17, [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [QQ0,   QQ2,   QQ4,   QQ6]>;
def : SubRegSet<18, [QQQQ0, QQQQ1, QQQQ2, QQQQ3],
                    [QQ1,   QQ3,   QQ5,   QQ7]>;

