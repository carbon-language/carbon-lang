//===-- MipsRegisterInfo.td - Mips Register defs -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the MIPS register file
//===----------------------------------------------------------------------===//
let Namespace = "Mips" in {
def sub_fpeven : SubRegIndex<32>;
def sub_fpodd  : SubRegIndex<32, 32>;
def sub_32     : SubRegIndex<32>;
def sub_lo     : SubRegIndex<32>;
def sub_hi     : SubRegIndex<32, 32>;
def sub_dsp16_19 : SubRegIndex<4, 16>;
def sub_dsp20    : SubRegIndex<1, 20>;
def sub_dsp21    : SubRegIndex<1, 21>;
def sub_dsp22    : SubRegIndex<1, 22>;
def sub_dsp23    : SubRegIndex<1, 23>;
}

class Unallocatable {
  bit isAllocatable = 0;
}

// We have banks of 32 registers each.
class MipsReg<bits<16> Enc, string n> : Register<n> {
  let HWEncoding = Enc;
  let Namespace = "Mips";
}

class MipsRegWithSubRegs<bits<16> Enc, string n, list<Register> subregs>
  : RegisterWithSubRegs<n, subregs> {
  let HWEncoding = Enc;
  let Namespace = "Mips";
}

// Mips CPU Registers
class MipsGPRReg<bits<16> Enc, string n> : MipsReg<Enc, n>;

// Mips 64-bit CPU Registers
class Mips64GPRReg<bits<16> Enc, string n, list<Register> subregs>
  : MipsRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_32];
}

// Mips 32-bit FPU Registers
class FPR<bits<16> Enc, string n> : MipsReg<Enc, n>;

// Mips 64-bit (aliased) FPU Registers
class AFPR<bits<16> Enc, string n, list<Register> subregs>
  : MipsRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_fpeven, sub_fpodd];
  let CoveredBySubRegs = 1;
}

class AFPR64<bits<16> Enc, string n, list<Register> subregs>
  : MipsRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_32];
}

// Accumulator Registers
class ACC<bits<16> Enc, string n, list<Register> subregs>
  : MipsRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_lo, sub_hi];
  let CoveredBySubRegs = 1;
}

// Mips Hardware Registers
class HWR<bits<16> Enc, string n> : MipsReg<Enc, n>;

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

let Namespace = "Mips" in {
  // General Purpose Registers
  def ZERO : MipsGPRReg< 0, "zero">, DwarfRegNum<[0]>;
  def AT   : MipsGPRReg< 1, "1">,    DwarfRegNum<[1]>;
  def V0   : MipsGPRReg< 2, "2">,    DwarfRegNum<[2]>;
  def V1   : MipsGPRReg< 3, "3">,    DwarfRegNum<[3]>;
  def A0   : MipsGPRReg< 4, "4">,    DwarfRegNum<[4]>;
  def A1   : MipsGPRReg< 5, "5">,    DwarfRegNum<[5]>;
  def A2   : MipsGPRReg< 6, "6">,    DwarfRegNum<[6]>;
  def A3   : MipsGPRReg< 7, "7">,    DwarfRegNum<[7]>;
  def T0   : MipsGPRReg< 8, "8">,    DwarfRegNum<[8]>;
  def T1   : MipsGPRReg< 9, "9">,    DwarfRegNum<[9]>;
  def T2   : MipsGPRReg< 10, "10">,  DwarfRegNum<[10]>;
  def T3   : MipsGPRReg< 11, "11">,  DwarfRegNum<[11]>;
  def T4   : MipsGPRReg< 12, "12">,  DwarfRegNum<[12]>;
  def T5   : MipsGPRReg< 13, "13">,  DwarfRegNum<[13]>;
  def T6   : MipsGPRReg< 14, "14">,  DwarfRegNum<[14]>;
  def T7   : MipsGPRReg< 15, "15">,  DwarfRegNum<[15]>;
  def S0   : MipsGPRReg< 16, "16">,  DwarfRegNum<[16]>;
  def S1   : MipsGPRReg< 17, "17">,  DwarfRegNum<[17]>;
  def S2   : MipsGPRReg< 18, "18">,  DwarfRegNum<[18]>;
  def S3   : MipsGPRReg< 19, "19">,  DwarfRegNum<[19]>;
  def S4   : MipsGPRReg< 20, "20">,  DwarfRegNum<[20]>;
  def S5   : MipsGPRReg< 21, "21">,  DwarfRegNum<[21]>;
  def S6   : MipsGPRReg< 22, "22">,  DwarfRegNum<[22]>;
  def S7   : MipsGPRReg< 23, "23">,  DwarfRegNum<[23]>;
  def T8   : MipsGPRReg< 24, "24">,  DwarfRegNum<[24]>;
  def T9   : MipsGPRReg< 25, "25">,  DwarfRegNum<[25]>;
  def K0   : MipsGPRReg< 26, "26">,  DwarfRegNum<[26]>;
  def K1   : MipsGPRReg< 27, "27">,  DwarfRegNum<[27]>;
  def GP   : MipsGPRReg< 28, "gp">,  DwarfRegNum<[28]>;
  def SP   : MipsGPRReg< 29, "sp">,  DwarfRegNum<[29]>;
  def FP   : MipsGPRReg< 30, "fp">,  DwarfRegNum<[30]>;
  def RA   : MipsGPRReg< 31, "ra">,  DwarfRegNum<[31]>;

  // General Purpose 64-bit Registers
  def ZERO_64 : Mips64GPRReg< 0, "zero", [ZERO]>, DwarfRegNum<[0]>;
  def AT_64   : Mips64GPRReg< 1, "1",    [AT]>, DwarfRegNum<[1]>;
  def V0_64   : Mips64GPRReg< 2, "2",    [V0]>, DwarfRegNum<[2]>;
  def V1_64   : Mips64GPRReg< 3, "3",    [V1]>, DwarfRegNum<[3]>;
  def A0_64   : Mips64GPRReg< 4, "4",    [A0]>, DwarfRegNum<[4]>;
  def A1_64   : Mips64GPRReg< 5, "5",    [A1]>, DwarfRegNum<[5]>;
  def A2_64   : Mips64GPRReg< 6, "6",    [A2]>, DwarfRegNum<[6]>;
  def A3_64   : Mips64GPRReg< 7, "7",    [A3]>, DwarfRegNum<[7]>;
  def T0_64   : Mips64GPRReg< 8, "8",    [T0]>, DwarfRegNum<[8]>;
  def T1_64   : Mips64GPRReg< 9, "9",    [T1]>, DwarfRegNum<[9]>;
  def T2_64   : Mips64GPRReg< 10, "10",  [T2]>, DwarfRegNum<[10]>;
  def T3_64   : Mips64GPRReg< 11, "11",  [T3]>, DwarfRegNum<[11]>;
  def T4_64   : Mips64GPRReg< 12, "12",  [T4]>, DwarfRegNum<[12]>;
  def T5_64   : Mips64GPRReg< 13, "13",  [T5]>, DwarfRegNum<[13]>;
  def T6_64   : Mips64GPRReg< 14, "14",  [T6]>, DwarfRegNum<[14]>;
  def T7_64   : Mips64GPRReg< 15, "15",  [T7]>, DwarfRegNum<[15]>;
  def S0_64   : Mips64GPRReg< 16, "16",  [S0]>, DwarfRegNum<[16]>;
  def S1_64   : Mips64GPRReg< 17, "17",  [S1]>, DwarfRegNum<[17]>;
  def S2_64   : Mips64GPRReg< 18, "18",  [S2]>, DwarfRegNum<[18]>;
  def S3_64   : Mips64GPRReg< 19, "19",  [S3]>, DwarfRegNum<[19]>;
  def S4_64   : Mips64GPRReg< 20, "20",  [S4]>, DwarfRegNum<[20]>;
  def S5_64   : Mips64GPRReg< 21, "21",  [S5]>, DwarfRegNum<[21]>;
  def S6_64   : Mips64GPRReg< 22, "22",  [S6]>, DwarfRegNum<[22]>;
  def S7_64   : Mips64GPRReg< 23, "23",  [S7]>, DwarfRegNum<[23]>;
  def T8_64   : Mips64GPRReg< 24, "24",  [T8]>, DwarfRegNum<[24]>;
  def T9_64   : Mips64GPRReg< 25, "25",  [T9]>, DwarfRegNum<[25]>;
  def K0_64   : Mips64GPRReg< 26, "26",  [K0]>, DwarfRegNum<[26]>;
  def K1_64   : Mips64GPRReg< 27, "27",  [K1]>, DwarfRegNum<[27]>;
  def GP_64   : Mips64GPRReg< 28, "gp",  [GP]>, DwarfRegNum<[28]>;
  def SP_64   : Mips64GPRReg< 29, "sp",  [SP]>, DwarfRegNum<[29]>;
  def FP_64   : Mips64GPRReg< 30, "fp",  [FP]>, DwarfRegNum<[30]>;
  def RA_64   : Mips64GPRReg< 31, "ra",  [RA]>, DwarfRegNum<[31]>;

  /// Mips Single point precision FPU Registers
  foreach I = 0-31 in
  def F#I : FPR<I, "f"#I>, DwarfRegNum<[!add(I, 32)]>;

  /// Mips Double point precision FPU Registers (aliased
  /// with the single precision to hold 64 bit values)
  foreach I = 0-15 in
  def D#I : AFPR<!shl(I, 1), "f"#!shl(I, 1),
                 [!cast<FPR>("F"#!shl(I, 1)),
                  !cast<FPR>("F"#!add(!shl(I, 1), 1))]>;

  /// Mips Double point precision FPU Registers in MFP64 mode.
  foreach I = 0-31 in
  def D#I#_64 : AFPR64<I, "f"#I, [!cast<FPR>("F"#I)]>,
                DwarfRegNum<[!add(I, 32)]>;

  // Hi/Lo registers
  def HI  : Register<"ac0">, DwarfRegNum<[64]>;
  def HI1 : Register<"ac1">, DwarfRegNum<[176]>;
  def HI2 : Register<"ac2">, DwarfRegNum<[178]>;
  def HI3 : Register<"ac3">, DwarfRegNum<[180]>;
  def LO  : Register<"ac0">, DwarfRegNum<[65]>;
  def LO1 : Register<"ac1">, DwarfRegNum<[177]>;
  def LO2 : Register<"ac2">, DwarfRegNum<[179]>;
  def LO3 : Register<"ac3">, DwarfRegNum<[181]>;

  let SubRegIndices = [sub_32] in {
  def HI64  : RegisterWithSubRegs<"hi", [HI]>;
  def LO64  : RegisterWithSubRegs<"lo", [LO]>;
  }

  // FP control registers.
  foreach I = 0-31 in
  def FCR#I : MipsReg<#I, ""#I>;

  // FP condition code registers.
  foreach I = 0-7 in
  def FCC#I : MipsReg<#I, "fcc"#I>;

  // PC register
  def PC : Register<"pc">;

  // Hardware register $29
  def HWR29 : MipsReg<29, "29">;
  def HWR29_64 : MipsReg<29, "29">;

  // Accum registers
  def AC0 : ACC<0, "ac0", [LO, HI]>;
  def AC1 : ACC<1, "ac1", [LO1, HI1]>;
  def AC2 : ACC<2, "ac2", [LO2, HI2]>;
  def AC3 : ACC<3, "ac3", [LO3, HI3]>;

  def AC0_64 : ACC<0, "ac0", [LO64, HI64]>;

  // DSP-ASE control register fields.
  def DSPPos : Register<"">;
  def DSPSCount : Register<"">;
  def DSPCarry : Register<"">;
  def DSPEFI : Register<"">;
  def DSPOutFlag16_19 : Register<"">;
  def DSPOutFlag20 : Register<"">;
  def DSPOutFlag21 : Register<"">;
  def DSPOutFlag22 : Register<"">;
  def DSPOutFlag23 : Register<"">;
  def DSPCCond : Register<"">;

  let SubRegIndices = [sub_dsp16_19, sub_dsp20, sub_dsp21, sub_dsp22,
                       sub_dsp23] in
  def DSPOutFlag : RegisterWithSubRegs<"", [DSPOutFlag16_19, DSPOutFlag20,
                                            DSPOutFlag21, DSPOutFlag22,
                                            DSPOutFlag23]>;
}

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

class GPR32Class<list<ValueType> regTypes> :
  RegisterClass<"Mips", regTypes, 32, (add
  // Reserved
  ZERO, AT,
  // Return Values and Arguments
  V0, V1, A0, A1, A2, A3,
  // Not preserved across procedure calls
  T0, T1, T2, T3, T4, T5, T6, T7,
  // Callee save
  S0, S1, S2, S3, S4, S5, S6, S7,
  // Not preserved across procedure calls
  T8, T9,
  // Reserved
  K0, K1, GP, SP, FP, RA)>;

def GPR32 : GPR32Class<[i32]>;
def DSPRegs : GPR32Class<[v4i8, v2i16]>;

def GPR64 : RegisterClass<"Mips", [i64], 64, (add
// Reserved
  ZERO_64, AT_64,
  // Return Values and Arguments
  V0_64, V1_64, A0_64, A1_64, A2_64, A3_64,
  // Not preserved across procedure calls
  T0_64, T1_64, T2_64, T3_64, T4_64, T5_64, T6_64, T7_64,
  // Callee save
  S0_64, S1_64, S2_64, S3_64, S4_64, S5_64, S6_64, S7_64,
  // Not preserved across procedure calls
  T8_64, T9_64,
  // Reserved
  K0_64, K1_64, GP_64, SP_64, FP_64, RA_64)>;

def CPU16Regs : RegisterClass<"Mips", [i32], 32, (add
  // Return Values and Arguments
  V0, V1, A0, A1, A2, A3,
  // Callee save
  S0, S1)>;

def CPU16RegsPlusSP : RegisterClass<"Mips", [i32], 32, (add
  // Return Values and Arguments
  V0, V1, A0, A1, A2, A3,
  // Callee save
  S0, S1,
  SP)>;

def CPURAReg : RegisterClass<"Mips", [i32], 32, (add RA)>, Unallocatable;

def CPUSPReg : RegisterClass<"Mips", [i32], 32, (add SP)>, Unallocatable;

// 64bit fp:
// * FGR64  - 32 64-bit registers
// * AFGR64 - 16 32-bit even registers (32-bit FP Mode)
//
// 32bit fp:
// * FGR32 - 16 32-bit even registers
// * FGR32 - 32 32-bit registers (single float only mode)
def FGR32 : RegisterClass<"Mips", [f32], 32, (sequence "F%u", 0, 31)>;

def AFGR64 : RegisterClass<"Mips", [f64], 64, (add
  // Return Values and Arguments
  D0, D1,
  // Not preserved across procedure calls
  D2, D3, D4, D5,
  // Return Values and Arguments
  D6, D7,
  // Not preserved across procedure calls
  D8, D9,
  // Callee save
  D10, D11, D12, D13, D14, D15)>;

def FGR64 : RegisterClass<"Mips", [f64], 64, (sequence "D%u_64", 0, 31)>;

// FP control registers.
def CCR : RegisterClass<"Mips", [i32], 32, (sequence "FCR%u", 0, 31)>,
          Unallocatable;

// FP condition code registers.
def FCC : RegisterClass<"Mips", [i32], 32, (sequence "FCC%u", 0, 7)>,
          Unallocatable;

// Hi/Lo Registers
def LORegs : RegisterClass<"Mips", [i32], 32, (add LO)>;
def HIRegs : RegisterClass<"Mips", [i32], 32, (add HI)>;
def LORegsDSP : RegisterClass<"Mips", [i32], 32, (add LO, LO1, LO2, LO3)>;
def HIRegsDSP : RegisterClass<"Mips", [i32], 32, (add HI, HI1, HI2, HI3)>;
def LORegs64 : RegisterClass<"Mips", [i64], 64, (add LO64)>;
def HIRegs64 : RegisterClass<"Mips", [i64], 64, (add HI64)>;

// Hardware registers
def HWRegs : RegisterClass<"Mips", [i32], 32, (add HWR29)>, Unallocatable;
def HWRegs64 : RegisterClass<"Mips", [i64], 64, (add HWR29_64)>, Unallocatable;

// Accumulator Registers
def ACRegs : RegisterClass<"Mips", [untyped], 64, (add AC0)> {
  let Size = 64;
}

def ACRegs128 : RegisterClass<"Mips", [untyped], 128, (add AC0_64)> {
  let Size = 128;
}

def ACRegsDSP : RegisterClass<"Mips", [untyped], 64, (sequence "AC%u", 0, 3)> {
  let Size = 64;
}

def DSPCC : RegisterClass<"Mips", [v4i8, v2i16], 32, (add DSPCCond)>;

// Register Operands.

class MipsAsmRegOperand : AsmOperandClass {
  let RenderMethod = "addRegAsmOperands";
}
def GPR32AsmOperand : MipsAsmRegOperand {
  let Name = "GPR32Asm";
  let ParserMethod = "parseGPR32";
}

def GPR64AsmOperand : MipsAsmRegOperand {
  let Name = "GPR64Asm";
  let ParserMethod = "parseGPR64";
}

def ACRegsDSPAsmOperand : MipsAsmRegOperand {
  let Name = "ACRegsDSPAsm";
  let ParserMethod = "parseACRegsDSP";
}

def CCRAsmOperand : MipsAsmRegOperand {
  let Name = "CCRAsm";
  let ParserMethod = "parseCCRRegs";
}

def AFGR64AsmOperand : MipsAsmRegOperand {
  let Name = "AFGR64Asm";
  let ParserMethod = "parseAFGR64Regs";
}

def FGR64AsmOperand : MipsAsmRegOperand {
  let Name = "FGR64Asm";
  let ParserMethod = "parseFGR64Regs";
}

def FGR32AsmOperand : MipsAsmRegOperand {
  let Name = "FGR32Asm";
  let ParserMethod = "parseFGR32Regs";
}

def FCCRegsAsmOperand : MipsAsmRegOperand {
  let Name = "FCCRegsAsm";
  let ParserMethod = "parseFCCRegs";
}

def GPR32Opnd : RegisterOperand<GPR32> {
  let ParserMatchClass = GPR32AsmOperand;
}

def GPR64Opnd : RegisterOperand<GPR64> {
  let ParserMatchClass = GPR64AsmOperand;
}

def CCROpnd : RegisterOperand<CCR> {
  let ParserMatchClass = CCRAsmOperand;
}

def HWRegsAsmOperand : MipsAsmRegOperand {
  let Name = "HWRegsAsm";
  let ParserMethod = "parseHWRegs";
}

def HW64RegsAsmOperand : MipsAsmRegOperand {
  let Name = "HW64RegsAsm";
  let ParserMethod = "parseHW64Regs";
}

def HWRegsOpnd : RegisterOperand<HWRegs> {
  let ParserMatchClass = HWRegsAsmOperand;
}

def HW64RegsOpnd : RegisterOperand<HWRegs64> {
  let ParserMatchClass = HW64RegsAsmOperand;
}

def AFGR64RegsOpnd : RegisterOperand<AFGR64> {
  let ParserMatchClass = AFGR64AsmOperand;
}

def FGR64RegsOpnd : RegisterOperand<FGR64> {
  let ParserMatchClass = FGR64AsmOperand;
}

def FGR32RegsOpnd : RegisterOperand<FGR32> {
  let ParserMatchClass = FGR32AsmOperand;
}

def FCCRegsOpnd : RegisterOperand<FCC> {
  let ParserMatchClass = FCCRegsAsmOperand;
}

def ACRegsDSPOpnd : RegisterOperand<ACRegsDSP> {
  let ParserMatchClass = ACRegsDSPAsmOperand;
}
