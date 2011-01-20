//===- ARMRegisterInfo.td - ARM Register defs --------------*- tablegen -*-===//
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

// Subregister indices.
let Namespace = "ARM" in {
// Note: Code depends on these having consecutive numbers.
def ssub_0  : SubRegIndex;
def ssub_1  : SubRegIndex;
def ssub_2  : SubRegIndex; // In a Q reg.
def ssub_3  : SubRegIndex;
def ssub_4  : SubRegIndex; // In a QQ reg.
def ssub_5  : SubRegIndex;
def ssub_6  : SubRegIndex;
def ssub_7  : SubRegIndex;
def ssub_8  : SubRegIndex; // In a QQQQ reg.
def ssub_9  : SubRegIndex;
def ssub_10 : SubRegIndex;
def ssub_11 : SubRegIndex;
def ssub_12 : SubRegIndex;
def ssub_13 : SubRegIndex;
def ssub_14 : SubRegIndex;
def ssub_15 : SubRegIndex;

def dsub_0 : SubRegIndex;
def dsub_1 : SubRegIndex;
def dsub_2 : SubRegIndex;
def dsub_3 : SubRegIndex;
def dsub_4 : SubRegIndex;
def dsub_5 : SubRegIndex;
def dsub_6 : SubRegIndex;
def dsub_7 : SubRegIndex;

def qsub_0 : SubRegIndex;
def qsub_1 : SubRegIndex;
def qsub_2 : SubRegIndex;
def qsub_3 : SubRegIndex;

def qqsub_0 : SubRegIndex;
def qqsub_1 : SubRegIndex;
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

// Aliases of the F* registers used to hold 64-bit fp values (doubles)
let SubRegIndices = [ssub_0, ssub_1] in {
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
}

// VFP3 defines 16 additional double registers
def D16 : ARMFReg<16, "d16">; def D17 : ARMFReg<17, "d17">;
def D18 : ARMFReg<18, "d18">; def D19 : ARMFReg<19, "d19">;
def D20 : ARMFReg<20, "d20">; def D21 : ARMFReg<21, "d21">;
def D22 : ARMFReg<22, "d22">; def D23 : ARMFReg<23, "d23">;
def D24 : ARMFReg<24, "d24">; def D25 : ARMFReg<25, "d25">;
def D26 : ARMFReg<26, "d26">; def D27 : ARMFReg<27, "d27">;
def D28 : ARMFReg<28, "d28">; def D29 : ARMFReg<29, "d29">;
def D30 : ARMFReg<30, "d30">; def D31 : ARMFReg<31, "d31">;

// Advanced SIMD (NEON) defines 16 quad-word aliases
let SubRegIndices = [dsub_0, dsub_1],
 CompositeIndices = [(ssub_2 dsub_1, ssub_0),
                     (ssub_3 dsub_1, ssub_1)] in {
def Q0  : ARMReg< 0,  "q0", [D0,   D1]>;
def Q1  : ARMReg< 1,  "q1", [D2,   D3]>;
def Q2  : ARMReg< 2,  "q2", [D4,   D5]>;
def Q3  : ARMReg< 3,  "q3", [D6,   D7]>;
def Q4  : ARMReg< 4,  "q4", [D8,   D9]>;
def Q5  : ARMReg< 5,  "q5", [D10, D11]>;
def Q6  : ARMReg< 6,  "q6", [D12, D13]>;
def Q7  : ARMReg< 7,  "q7", [D14, D15]>;
}
let SubRegIndices = [dsub_0, dsub_1] in {
def Q8  : ARMReg< 8,  "q8", [D16, D17]>;
def Q9  : ARMReg< 9,  "q9", [D18, D19]>;
def Q10 : ARMReg<10, "q10", [D20, D21]>;
def Q11 : ARMReg<11, "q11", [D22, D23]>;
def Q12 : ARMReg<12, "q12", [D24, D25]>;
def Q13 : ARMReg<13, "q13", [D26, D27]>;
def Q14 : ARMReg<14, "q14", [D28, D29]>;
def Q15 : ARMReg<15, "q15", [D30, D31]>;
}

// Pseudo 256-bit registers to represent pairs of Q registers. These should
// never be present in the emitted code.
// These are used for NEON load / store instructions, e.g., vld4, vst3.
// NOTE: It's possible to define more QQ registers since technically the
// starting D register number doesn't have to be multiple of 4, e.g.,
// D1, D2, D3, D4 would be a legal quad, but that would make the subregister
// stuff very messy.
let SubRegIndices = [qsub_0, qsub_1] in {
let CompositeIndices = [(dsub_2 qsub_1, dsub_0), (dsub_3 qsub_1, dsub_1),
                        (ssub_4 qsub_1, ssub_0), (ssub_5 qsub_1, ssub_1),
                        (ssub_6 qsub_1, ssub_2), (ssub_7 qsub_1, ssub_3)] in {
def QQ0 : ARMReg<0, "qq0", [Q0,  Q1]>;
def QQ1 : ARMReg<1, "qq1", [Q2,  Q3]>;
def QQ2 : ARMReg<2, "qq2", [Q4,  Q5]>;
def QQ3 : ARMReg<3, "qq3", [Q6,  Q7]>;
}
let CompositeIndices = [(dsub_2 qsub_1, dsub_0), (dsub_3 qsub_1, dsub_1)] in {
def QQ4 : ARMReg<4, "qq4", [Q8,  Q9]>;
def QQ5 : ARMReg<5, "qq5", [Q10, Q11]>;
def QQ6 : ARMReg<6, "qq6", [Q12, Q13]>;
def QQ7 : ARMReg<7, "qq7", [Q14, Q15]>;
}
}

// Pseudo 512-bit registers to represent four consecutive Q registers.
let SubRegIndices = [qqsub_0, qqsub_1] in {
let CompositeIndices = [(qsub_2  qqsub_1, qsub_0), (qsub_3  qqsub_1, qsub_1),
                        (dsub_4  qqsub_1, dsub_0), (dsub_5  qqsub_1, dsub_1),
                        (dsub_6  qqsub_1, dsub_2), (dsub_7  qqsub_1, dsub_3),
                        (ssub_8  qqsub_1, ssub_0), (ssub_9  qqsub_1, ssub_1),
                        (ssub_10 qqsub_1, ssub_2), (ssub_11 qqsub_1, ssub_3),
                        (ssub_12 qqsub_1, ssub_4), (ssub_13 qqsub_1, ssub_5),
                        (ssub_14 qqsub_1, ssub_6), (ssub_15 qqsub_1, ssub_7)] in
{
def QQQQ0 : ARMReg<0, "qqqq0", [QQ0, QQ1]>;
def QQQQ1 : ARMReg<1, "qqqq1", [QQ2, QQ3]>;
}
let CompositeIndices = [(qsub_2 qqsub_1, qsub_0), (qsub_3 qqsub_1, qsub_1),
                        (dsub_4 qqsub_1, dsub_0), (dsub_5 qqsub_1, dsub_1),
                        (dsub_6 qqsub_1, dsub_2), (dsub_7 qqsub_1, dsub_3)] in {
def QQQQ2 : ARMReg<2, "qqqq2", [QQ4, QQ5]>;
def QQQQ3 : ARMReg<3, "qqqq3", [QQ6, QQ7]>;
}
}

// Current Program Status Register.
def CPSR    : ARMReg<0, "cpsr">;
def FPSCR   : ARMReg<1, "fpscr">;
def ITSTATE : ARMReg<2, "itstate">;

// Special Registers - only available in privileged mode.
def FPSID   : ARMReg<0, "fpsid">;
def FPEXC   : ARMReg<8, "fpexc">;

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
    static const unsigned ARM_GPR_AO[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6, ARM::R7,
      ARM::R8, ARM::R9, ARM::R10, ARM::R11 };

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
      return ARM_GPR_AO;
    }

    GPRClass::iterator
    GPRClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      if (Subtarget.isThumb1Only())
        return THUMB_GPR_AO + (sizeof(THUMB_GPR_AO)/sizeof(unsigned));
      return ARM_GPR_AO + (sizeof(ARM_GPR_AO)/sizeof(unsigned));
    }
  }];
}

// restricted GPR register class. Many Thumb2 instructions allow the full
// register range for operands, but have undefined behaviours when PC
// or SP (R13 or R15) are used. The ARM ISA refers to these operands
// via the BadReg() pseudo-code description.
def rGPR : RegisterClass<"ARM", [i32], 32, [R0, R1, R2, R3, R4, R5, R6,
                                            R7, R8, R9, R10, R11, R12, LR]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned ARM_rGPR_AO[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6, ARM::R7,
      ARM::R8, ARM::R9, ARM::R10,
      ARM::R11 };

    // For Thumb1 mode, we don't want to allocate hi regs at all, as we
    // don't know how to spill them. If we make our prologue/epilogue code
    // smarter at some point, we can go back to using the above allocation
    // orders for the Thumb1 instructions that know how to use hi regs.
    static const unsigned THUMB_rGPR_AO[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R4, ARM::R5, ARM::R6, ARM::R7 };

    rGPRClass::iterator
    rGPRClass::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      if (Subtarget.isThumb1Only())
        return THUMB_rGPR_AO;
      return ARM_rGPR_AO;
    }

    rGPRClass::iterator
    rGPRClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();

      if (Subtarget.isThumb1Only())
        return THUMB_rGPR_AO + (sizeof(THUMB_rGPR_AO)/sizeof(unsigned));
      return ARM_rGPR_AO + (sizeof(ARM_rGPR_AO)/sizeof(unsigned));
    }
  }];
}

// Thumb registers are R0-R7 normally. Some instructions can still use
// the general GPR register class above (MOV, e.g.)
def tGPR : RegisterClass<"ARM", [i32], 32, [R0, R1, R2, R3, R4, R5, R6, R7]> {}

// For tail calls, we can't use callee-saved registers, as they are restored
// to the saved value before the tail call, which would clobber a call address.
// Note, getMinimalPhysRegClass(R0) returns tGPR because of the names of
// this class and the preceding one(!)  This is what we want.
def tcGPR : RegisterClass<"ARM", [i32], 32, [R0, R1, R2, R3, R9, R12]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    // R9 is available.
    static const unsigned ARM_GPR_R9_TC[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R9, ARM::R12 };
    // R9 is not available.
    static const unsigned ARM_GPR_NOR9_TC[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3,
      ARM::R12 };

    // For Thumb1 mode, we don't want to allocate hi regs at all, as we
    // don't know how to spill them. If we make our prologue/epilogue code
    // smarter at some point, we can go back to using the above allocation
    // orders for the Thumb1 instructions that know how to use hi regs.
    static const unsigned THUMB_GPR_AO_TC[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3 };

    tcGPRClass::iterator
    tcGPRClass::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      if (Subtarget.isThumb1Only())
        return THUMB_GPR_AO_TC;
      return Subtarget.isTargetDarwin() ? ARM_GPR_R9_TC : ARM_GPR_NOR9_TC;
    }

    tcGPRClass::iterator
    tcGPRClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();

      if (Subtarget.isThumb1Only())
        return THUMB_GPR_AO_TC + (sizeof(THUMB_GPR_AO_TC)/sizeof(unsigned));

      return Subtarget.isTargetDarwin() ?
        ARM_GPR_R9_TC + (sizeof(ARM_GPR_R9_TC)/sizeof(unsigned)) :
        ARM_GPR_NOR9_TC + (sizeof(ARM_GPR_NOR9_TC)/sizeof(unsigned));
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

// Scalar double precision floating point / generic 64-bit vector register
// class.
// ARM requires only word alignment for double. It's more performant if it
// is double-word alignment though.
def DPR : RegisterClass<"ARM", [f64, v8i8, v4i16, v2i32, v1i64, v2f32], 64,
                        [D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
                         D8,  D9,  D10, D11, D12, D13, D14, D15,
                         D16, D17, D18, D19, D20, D21, D22, D23,
                         D24, D25, D26, D27, D28, D29, D30, D31]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    // VFP2 / VFPv3-D16
    static const unsigned ARM_DPR_VFP2[] = {
      ARM::D0,  ARM::D1,  ARM::D2,  ARM::D3,
      ARM::D4,  ARM::D5,  ARM::D6,  ARM::D7,
      ARM::D8,  ARM::D9,  ARM::D10, ARM::D11,
      ARM::D12, ARM::D13, ARM::D14, ARM::D15 };
    // VFP3: D8-D15 are callee saved and should be allocated last.
    // Save other low registers for use as DPR_VFP2 and DPR_8 classes.
    static const unsigned ARM_DPR_VFP3[] = {
      ARM::D16, ARM::D17, ARM::D18, ARM::D19,
      ARM::D20, ARM::D21, ARM::D22, ARM::D23,
      ARM::D24, ARM::D25, ARM::D26, ARM::D27,
      ARM::D28, ARM::D29, ARM::D30, ARM::D31,
      ARM::D0,  ARM::D1,  ARM::D2,  ARM::D3,
      ARM::D4,  ARM::D5,  ARM::D6,  ARM::D7,
      ARM::D8,  ARM::D9,  ARM::D10, ARM::D11,
      ARM::D12, ARM::D13, ARM::D14, ARM::D15 };

    DPRClass::iterator
    DPRClass::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      if (Subtarget.hasVFP3() && !Subtarget.hasD16())
        return ARM_DPR_VFP3;
      return ARM_DPR_VFP2;
    }

    DPRClass::iterator
    DPRClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      if (Subtarget.hasVFP3() && !Subtarget.hasD16())
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
  let SubRegClasses = [(SPR ssub_0, ssub_1)];
}

// Subset of DPR which can be used as a source of NEON scalars for 16-bit
// operations
def DPR_8 : RegisterClass<"ARM", [f64, v8i8, v4i16, v2i32, v1i64, v2f32], 64,
                          [D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7]> {
  let SubRegClasses = [(SPR_8 ssub_0, ssub_1)];
}

// Generic 128-bit vector register class.
def QPR : RegisterClass<"ARM", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64], 128,
                        [Q0,  Q1,  Q2,  Q3,  Q4,  Q5,  Q6,  Q7,
                         Q8,  Q9,  Q10, Q11, Q12, Q13, Q14, Q15]> {
  let SubRegClasses = [(DPR dsub_0, dsub_1)];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    // Q4-Q7 are callee saved and should be allocated last.
    // Save other low registers for use as QPR_VFP2 and QPR_8 classes.
    static const unsigned ARM_QPR[] = {
      ARM::Q8,  ARM::Q9,  ARM::Q10, ARM::Q11,
      ARM::Q12, ARM::Q13, ARM::Q14, ARM::Q15,
      ARM::Q0,  ARM::Q1,  ARM::Q2,  ARM::Q3,
      ARM::Q4,  ARM::Q5,  ARM::Q6,  ARM::Q7 };

    QPRClass::iterator
    QPRClass::allocation_order_begin(const MachineFunction &MF) const {
      return ARM_QPR;
    }

    QPRClass::iterator
    QPRClass::allocation_order_end(const MachineFunction &MF) const {
      return ARM_QPR + (sizeof(ARM_QPR)/sizeof(unsigned));
    }
  }];
}

// Subset of QPR that have 32-bit SPR subregs.
def QPR_VFP2 : RegisterClass<"ARM", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
                             128,
                             [Q0,  Q1,  Q2,  Q3,  Q4,  Q5,  Q6,  Q7]> {
  let SubRegClasses = [(SPR      ssub_0, ssub_1, ssub_2, ssub_3),
                       (DPR_VFP2 dsub_0, dsub_1)];
}

// Subset of QPR that have DPR_8 and SPR_8 subregs.
def QPR_8 : RegisterClass<"ARM", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
                           128,
                           [Q0,  Q1,  Q2,  Q3]> {
  let SubRegClasses = [(SPR_8 ssub_0, ssub_1, ssub_2, ssub_3),
                       (DPR_8 dsub_0, dsub_1)];
}

// Pseudo 256-bit vector register class to model pairs of Q registers
// (4 consecutive D registers).
def QQPR : RegisterClass<"ARM", [v4i64],
                         256,
                         [QQ0, QQ1, QQ2, QQ3, QQ4, QQ5, QQ6, QQ7]> {
  let SubRegClasses = [(DPR dsub_0, dsub_1, dsub_2, dsub_3),
                       (QPR qsub_0, qsub_1)];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    // QQ2-QQ3 are callee saved and should be allocated last.
    // Save other low registers for use as QPR_VFP2 and QPR_8 classes.
    static const unsigned ARM_QQPR[] = {
      ARM::QQ4, ARM::QQ5, ARM::QQ6, ARM::QQ7,
      ARM::QQ0, ARM::QQ1, ARM::QQ2, ARM::QQ3 };

    QQPRClass::iterator
    QQPRClass::allocation_order_begin(const MachineFunction &MF) const {
      return ARM_QQPR;
    }

    QQPRClass::iterator
    QQPRClass::allocation_order_end(const MachineFunction &MF) const {
      return ARM_QQPR + (sizeof(ARM_QQPR)/sizeof(unsigned));
    }
  }];
}

// Subset of QQPR that have 32-bit SPR subregs.
def QQPR_VFP2 : RegisterClass<"ARM", [v4i64],
                              256,
                              [QQ0, QQ1, QQ2, QQ3]> {
  let SubRegClasses = [(SPR      ssub_0, ssub_1, ssub_2, ssub_3),
                       (DPR_VFP2 dsub_0, dsub_1, dsub_2, dsub_3),
                       (QPR_VFP2 qsub_0, qsub_1)];

}

// Pseudo 512-bit vector register class to model 4 consecutive Q registers
// (8 consecutive D registers).
def QQQQPR : RegisterClass<"ARM", [v8i64],
                         256,
                         [QQQQ0, QQQQ1, QQQQ2, QQQQ3]> {
  let SubRegClasses = [(DPR dsub_0, dsub_1, dsub_2, dsub_3,
                            dsub_4, dsub_5, dsub_6, dsub_7),
                       (QPR qsub_0, qsub_1, qsub_2, qsub_3)];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    // QQQQ1 is callee saved and should be allocated last.
    // Save QQQQ0 for use as QPR_VFP2 and QPR_8 classes.
    static const unsigned ARM_QQQQPR[] = {
      ARM::QQQQ2, ARM::QQQQ3, ARM::QQQQ0, ARM::QQQQ1 };

    QQQQPRClass::iterator
    QQQQPRClass::allocation_order_begin(const MachineFunction &MF) const {
      return ARM_QQQQPR;
    }

    QQQQPRClass::iterator
    QQQQPRClass::allocation_order_end(const MachineFunction &MF) const {
      return ARM_QQQQPR + (sizeof(ARM_QQQQPR)/sizeof(unsigned));
    }
  }];
}

// Condition code registers.
def CCR : RegisterClass<"ARM", [i32], 32, [CPSR]>;
