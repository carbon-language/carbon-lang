//===- X86RegisterInfo.td - Describe the X86 Register File --*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 Register file, defining the registers themselves,
// aliases between the registers, and the register classes built out of the
// registers.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Register definitions...
//
let Namespace = "X86" in {

  // Subregister indices.
  def sub_8bit    : SubRegIndex;
  def sub_8bit_hi : SubRegIndex;
  def sub_16bit   : SubRegIndex;
  def sub_32bit   : SubRegIndex;

  def sub_ss  : SubRegIndex;
  def sub_sd  : SubRegIndex;
  def sub_xmm : SubRegIndex;


  // In the register alias definitions below, we define which registers alias
  // which others.  We only specify which registers the small registers alias,
  // because the register file generator is smart enough to figure out that
  // AL aliases AX if we tell it that AX aliased AL (for example).

  // Dwarf numbering is different for 32-bit and 64-bit, and there are
  // variations by target as well. Currently the first entry is for X86-64,
  // second - for EH on X86-32/Darwin and third is 'generic' one (X86-32/Linux
  // and debug information on X86-32/Darwin)

  // 8-bit registers
  // Low registers
  def AL : Register<"al">;
  def DL : Register<"dl">;
  def CL : Register<"cl">;
  def BL : Register<"bl">;

  // X86-64 only, requires REX.
  let CostPerUse = 1 in {
  def SIL : Register<"sil">;
  def DIL : Register<"dil">;
  def BPL : Register<"bpl">;
  def SPL : Register<"spl">;
  def R8B  : Register<"r8b">;
  def R9B  : Register<"r9b">;
  def R10B : Register<"r10b">;
  def R11B : Register<"r11b">;
  def R12B : Register<"r12b">;
  def R13B : Register<"r13b">;
  def R14B : Register<"r14b">;
  def R15B : Register<"r15b">;
  }

  // High registers. On x86-64, these cannot be used in any instruction
  // with a REX prefix.
  def AH : Register<"ah">;
  def DH : Register<"dh">;
  def CH : Register<"ch">;
  def BH : Register<"bh">;

  // 16-bit registers
  let SubRegIndices = [sub_8bit, sub_8bit_hi] in {
  def AX : RegisterWithSubRegs<"ax", [AL,AH]>;
  def DX : RegisterWithSubRegs<"dx", [DL,DH]>;
  def CX : RegisterWithSubRegs<"cx", [CL,CH]>;
  def BX : RegisterWithSubRegs<"bx", [BL,BH]>;
  }
  let SubRegIndices = [sub_8bit] in {
  def SI : RegisterWithSubRegs<"si", [SIL]>;
  def DI : RegisterWithSubRegs<"di", [DIL]>;
  def BP : RegisterWithSubRegs<"bp", [BPL]>;
  def SP : RegisterWithSubRegs<"sp", [SPL]>;
  }
  def IP : Register<"ip">;

  // X86-64 only, requires REX.
  let SubRegIndices = [sub_8bit], CostPerUse = 1 in {
  def R8W  : RegisterWithSubRegs<"r8w", [R8B]>;
  def R9W  : RegisterWithSubRegs<"r9w", [R9B]>;
  def R10W : RegisterWithSubRegs<"r10w", [R10B]>;
  def R11W : RegisterWithSubRegs<"r11w", [R11B]>;
  def R12W : RegisterWithSubRegs<"r12w", [R12B]>;
  def R13W : RegisterWithSubRegs<"r13w", [R13B]>;
  def R14W : RegisterWithSubRegs<"r14w", [R14B]>;
  def R15W : RegisterWithSubRegs<"r15w", [R15B]>;
  }
  // 32-bit registers
  let SubRegIndices = [sub_16bit] in {
  def EAX : RegisterWithSubRegs<"eax", [AX]>, DwarfRegNum<[-2, 0, 0]>;
  def EDX : RegisterWithSubRegs<"edx", [DX]>, DwarfRegNum<[-2, 2, 2]>;
  def ECX : RegisterWithSubRegs<"ecx", [CX]>, DwarfRegNum<[-2, 1, 1]>;
  def EBX : RegisterWithSubRegs<"ebx", [BX]>, DwarfRegNum<[-2, 3, 3]>;
  def ESI : RegisterWithSubRegs<"esi", [SI]>, DwarfRegNum<[-2, 6, 6]>;
  def EDI : RegisterWithSubRegs<"edi", [DI]>, DwarfRegNum<[-2, 7, 7]>;
  def EBP : RegisterWithSubRegs<"ebp", [BP]>, DwarfRegNum<[-2, 4, 5]>;
  def ESP : RegisterWithSubRegs<"esp", [SP]>, DwarfRegNum<[-2, 5, 4]>;
  def EIP : RegisterWithSubRegs<"eip", [IP]>, DwarfRegNum<[-2, 8, 8]>;

  // X86-64 only, requires REX
  let CostPerUse = 1 in {
  def R8D  : RegisterWithSubRegs<"r8d", [R8W]>;
  def R9D  : RegisterWithSubRegs<"r9d", [R9W]>;
  def R10D : RegisterWithSubRegs<"r10d", [R10W]>;
  def R11D : RegisterWithSubRegs<"r11d", [R11W]>;
  def R12D : RegisterWithSubRegs<"r12d", [R12W]>;
  def R13D : RegisterWithSubRegs<"r13d", [R13W]>;
  def R14D : RegisterWithSubRegs<"r14d", [R14W]>;
  def R15D : RegisterWithSubRegs<"r15d", [R15W]>;
  }}

  // 64-bit registers, X86-64 only
  let SubRegIndices = [sub_32bit] in {
  def RAX : RegisterWithSubRegs<"rax", [EAX]>, DwarfRegNum<[0, -2, -2]>;
  def RDX : RegisterWithSubRegs<"rdx", [EDX]>, DwarfRegNum<[1, -2, -2]>;
  def RCX : RegisterWithSubRegs<"rcx", [ECX]>, DwarfRegNum<[2, -2, -2]>;
  def RBX : RegisterWithSubRegs<"rbx", [EBX]>, DwarfRegNum<[3, -2, -2]>;
  def RSI : RegisterWithSubRegs<"rsi", [ESI]>, DwarfRegNum<[4, -2, -2]>;
  def RDI : RegisterWithSubRegs<"rdi", [EDI]>, DwarfRegNum<[5, -2, -2]>;
  def RBP : RegisterWithSubRegs<"rbp", [EBP]>, DwarfRegNum<[6, -2, -2]>;
  def RSP : RegisterWithSubRegs<"rsp", [ESP]>, DwarfRegNum<[7, -2, -2]>;

  // These also require REX.
  let CostPerUse = 1 in {
  def R8  : RegisterWithSubRegs<"r8", [R8D]>, DwarfRegNum<[8, -2, -2]>;
  def R9  : RegisterWithSubRegs<"r9", [R9D]>, DwarfRegNum<[9, -2, -2]>;
  def R10 : RegisterWithSubRegs<"r10", [R10D]>, DwarfRegNum<[10, -2, -2]>;
  def R11 : RegisterWithSubRegs<"r11", [R11D]>, DwarfRegNum<[11, -2, -2]>;
  def R12 : RegisterWithSubRegs<"r12", [R12D]>, DwarfRegNum<[12, -2, -2]>;
  def R13 : RegisterWithSubRegs<"r13", [R13D]>, DwarfRegNum<[13, -2, -2]>;
  def R14 : RegisterWithSubRegs<"r14", [R14D]>, DwarfRegNum<[14, -2, -2]>;
  def R15 : RegisterWithSubRegs<"r15", [R15D]>, DwarfRegNum<[15, -2, -2]>;
  def RIP : RegisterWithSubRegs<"rip", [EIP]>,  DwarfRegNum<[16, -2, -2]>;
  }}

  // MMX Registers. These are actually aliased to ST0 .. ST7
  def MM0 : Register<"mm0">, DwarfRegNum<[41, 29, 29]>;
  def MM1 : Register<"mm1">, DwarfRegNum<[42, 30, 30]>;
  def MM2 : Register<"mm2">, DwarfRegNum<[43, 31, 31]>;
  def MM3 : Register<"mm3">, DwarfRegNum<[44, 32, 32]>;
  def MM4 : Register<"mm4">, DwarfRegNum<[45, 33, 33]>;
  def MM5 : Register<"mm5">, DwarfRegNum<[46, 34, 34]>;
  def MM6 : Register<"mm6">, DwarfRegNum<[47, 35, 35]>;
  def MM7 : Register<"mm7">, DwarfRegNum<[48, 36, 36]>;

  // Pseudo Floating Point registers
  def FP0 : Register<"fp0">;
  def FP1 : Register<"fp1">;
  def FP2 : Register<"fp2">;
  def FP3 : Register<"fp3">;
  def FP4 : Register<"fp4">;
  def FP5 : Register<"fp5">;
  def FP6 : Register<"fp6">;

  // XMM Registers, used by the various SSE instruction set extensions.
  // The sub_ss and sub_sd subregs are the same registers with another regclass.
  let CompositeIndices = [(sub_ss), (sub_sd)] in {
  def XMM0: Register<"xmm0">, DwarfRegNum<[17, 21, 21]>;
  def XMM1: Register<"xmm1">, DwarfRegNum<[18, 22, 22]>;
  def XMM2: Register<"xmm2">, DwarfRegNum<[19, 23, 23]>;
  def XMM3: Register<"xmm3">, DwarfRegNum<[20, 24, 24]>;
  def XMM4: Register<"xmm4">, DwarfRegNum<[21, 25, 25]>;
  def XMM5: Register<"xmm5">, DwarfRegNum<[22, 26, 26]>;
  def XMM6: Register<"xmm6">, DwarfRegNum<[23, 27, 27]>;
  def XMM7: Register<"xmm7">, DwarfRegNum<[24, 28, 28]>;

  // X86-64 only
  let CostPerUse = 1 in {
  def XMM8:  Register<"xmm8">,  DwarfRegNum<[25, -2, -2]>;
  def XMM9:  Register<"xmm9">,  DwarfRegNum<[26, -2, -2]>;
  def XMM10: Register<"xmm10">, DwarfRegNum<[27, -2, -2]>;
  def XMM11: Register<"xmm11">, DwarfRegNum<[28, -2, -2]>;
  def XMM12: Register<"xmm12">, DwarfRegNum<[29, -2, -2]>;
  def XMM13: Register<"xmm13">, DwarfRegNum<[30, -2, -2]>;
  def XMM14: Register<"xmm14">, DwarfRegNum<[31, -2, -2]>;
  def XMM15: Register<"xmm15">, DwarfRegNum<[32, -2, -2]>;
  }}

  // YMM Registers, used by AVX instructions
  let SubRegIndices = [sub_xmm] in {
  def YMM0: RegisterWithSubRegs<"ymm0", [XMM0]>, DwarfRegAlias<XMM0>;
  def YMM1: RegisterWithSubRegs<"ymm1", [XMM1]>, DwarfRegAlias<XMM1>;
  def YMM2: RegisterWithSubRegs<"ymm2", [XMM2]>, DwarfRegAlias<XMM2>;
  def YMM3: RegisterWithSubRegs<"ymm3", [XMM3]>, DwarfRegAlias<XMM3>;
  def YMM4: RegisterWithSubRegs<"ymm4", [XMM4]>, DwarfRegAlias<XMM4>;
  def YMM5: RegisterWithSubRegs<"ymm5", [XMM5]>, DwarfRegAlias<XMM5>;
  def YMM6: RegisterWithSubRegs<"ymm6", [XMM6]>, DwarfRegAlias<XMM6>;
  def YMM7: RegisterWithSubRegs<"ymm7", [XMM7]>, DwarfRegAlias<XMM7>;
  def YMM8:  RegisterWithSubRegs<"ymm8", [XMM8]>, DwarfRegAlias<XMM8>;
  def YMM9:  RegisterWithSubRegs<"ymm9", [XMM9]>, DwarfRegAlias<XMM9>;
  def YMM10: RegisterWithSubRegs<"ymm10", [XMM10]>, DwarfRegAlias<XMM10>;
  def YMM11: RegisterWithSubRegs<"ymm11", [XMM11]>, DwarfRegAlias<XMM11>;
  def YMM12: RegisterWithSubRegs<"ymm12", [XMM12]>, DwarfRegAlias<XMM12>;
  def YMM13: RegisterWithSubRegs<"ymm13", [XMM13]>, DwarfRegAlias<XMM13>;
  def YMM14: RegisterWithSubRegs<"ymm14", [XMM14]>, DwarfRegAlias<XMM14>;
  def YMM15: RegisterWithSubRegs<"ymm15", [XMM15]>, DwarfRegAlias<XMM15>;
  }

  class STRegister<string Name, list<Register> A> : Register<Name> {
    let Aliases = A;
  }

  // Floating point stack registers. These don't map one-to-one to the FP
  // pseudo registers, but we still mark them as aliasing FP registers. That
  // way both kinds can be live without exceeding the stack depth. ST registers
  // are only live around inline assembly.
  def ST0 : STRegister<"st(0)", []>, DwarfRegNum<[33, 12, 11]>;
  def ST1 : STRegister<"st(1)", [FP6]>, DwarfRegNum<[34, 13, 12]>;
  def ST2 : STRegister<"st(2)", [FP5]>, DwarfRegNum<[35, 14, 13]>;
  def ST3 : STRegister<"st(3)", [FP4]>, DwarfRegNum<[36, 15, 14]>;
  def ST4 : STRegister<"st(4)", [FP3]>, DwarfRegNum<[37, 16, 15]>;
  def ST5 : STRegister<"st(5)", [FP2]>, DwarfRegNum<[38, 17, 16]>;
  def ST6 : STRegister<"st(6)", [FP1]>, DwarfRegNum<[39, 18, 17]>;
  def ST7 : STRegister<"st(7)", [FP0]>, DwarfRegNum<[40, 19, 18]>;

  // Status flags register
  def EFLAGS : Register<"flags">;

  // Segment registers
  def CS : Register<"cs">;
  def DS : Register<"ds">;
  def SS : Register<"ss">;
  def ES : Register<"es">;
  def FS : Register<"fs">;
  def GS : Register<"gs">;

  // Debug registers
  def DR0 : Register<"dr0">;
  def DR1 : Register<"dr1">;
  def DR2 : Register<"dr2">;
  def DR3 : Register<"dr3">;
  def DR4 : Register<"dr4">;
  def DR5 : Register<"dr5">;
  def DR6 : Register<"dr6">;
  def DR7 : Register<"dr7">;

  // Control registers
  def CR0 : Register<"cr0">;
  def CR1 : Register<"cr1">;
  def CR2 : Register<"cr2">;
  def CR3 : Register<"cr3">;
  def CR4 : Register<"cr4">;
  def CR5 : Register<"cr5">;
  def CR6 : Register<"cr6">;
  def CR7 : Register<"cr7">;
  def CR8 : Register<"cr8">;
  def CR9 : Register<"cr9">;
  def CR10 : Register<"cr10">;
  def CR11 : Register<"cr11">;
  def CR12 : Register<"cr12">;
  def CR13 : Register<"cr13">;
  def CR14 : Register<"cr14">;
  def CR15 : Register<"cr15">;

  // Pseudo index registers
  def EIZ : Register<"eiz">;
  def RIZ : Register<"riz">;
}


//===----------------------------------------------------------------------===//
// Register Class Definitions... now that we have all of the pieces, define the
// top-level register classes.  The order specified in the register list is
// implicitly defined to be the register allocation order.
//

// List call-clobbered registers before callee-save registers. RBX, RBP, (and
// R12, R13, R14, and R15 for X86-64) are callee-save registers.
// In 64-mode, there are 12 additional i8 registers, SIL, DIL, BPL, SPL, and
// R8B, ... R15B.
// Allocate R12 and R13 last, as these require an extra byte when
// encoded in x86_64 instructions.
// FIXME: Allow AH, CH, DH, BH to be used as general-purpose registers in
// 64-bit mode. The main complication is that they cannot be encoded in an
// instruction requiring a REX prefix, while SIL, DIL, BPL, R8D, etc.
// require a REX prefix. For example, "addb %ah, %dil" and "movzbl %ah, %r8d"
// cannot be encoded.
def GR8 : RegisterClass<"X86", [i8],  8,
                        (add AL, CL, DL, AH, CH, DH, BL, BH, SIL, DIL, BPL, SPL,
                             R8B, R9B, R10B, R11B, R14B, R15B, R12B, R13B)> {
  let AltOrders = [(sub GR8, AH, BH, CH, DH)];
  let AltOrderSelect = [{
    return MF.getTarget().getSubtarget<X86Subtarget>().is64Bit();
  }];
}

def GR16 : RegisterClass<"X86", [i16], 16,
                         (add AX, CX, DX, SI, DI, BX, BP, SP,
                              R8W, R9W, R10W, R11W, R14W, R15W, R12W, R13W)> {
  let SubRegClasses = [(GR8 sub_8bit, sub_8bit_hi)];
}

def GR32 : RegisterClass<"X86", [i32], 32,
                         (add EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP,
                              R8D, R9D, R10D, R11D, R14D, R15D, R12D, R13D)> {
  let SubRegClasses = [(GR8 sub_8bit, sub_8bit_hi), (GR16 sub_16bit)];
}

// GR64 - 64-bit GPRs. This oddly includes RIP, which isn't accurate, since
// RIP isn't really a register and it can't be used anywhere except in an
// address, but it doesn't cause trouble.
def GR64 : RegisterClass<"X86", [i64], 64,
                         (add RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
                              RBX, R14, R15, R12, R13, RBP, RSP, RIP)> {
  let SubRegClasses = [(GR8 sub_8bit, sub_8bit_hi),
                       (GR16 sub_16bit),
                       (GR32 sub_32bit)];
}

// Segment registers for use by MOV instructions (and others) that have a
//   segment register as one operand.  Always contain a 16-bit segment
//   descriptor.
def SEGMENT_REG : RegisterClass<"X86", [i16], 16, (add CS, DS, SS, ES, FS, GS)>;

// Debug registers.
def DEBUG_REG : RegisterClass<"X86", [i32], 32, (sequence "DR%u", 0, 7)>;

// Control registers.
def CONTROL_REG : RegisterClass<"X86", [i64], 64, (sequence "CR%u", 0, 15)>;

// GR8_ABCD_L, GR8_ABCD_H, GR16_ABCD, GR32_ABCD, GR64_ABCD - Subclasses of
// GR8, GR16, GR32, and GR64 which contain just the "a" "b", "c", and "d"
// registers. On x86-32, GR16_ABCD and GR32_ABCD are classes for registers
// that support 8-bit subreg operations. On x86-64, GR16_ABCD, GR32_ABCD,
// and GR64_ABCD are classes for registers that support 8-bit h-register
// operations.
def GR8_ABCD_L : RegisterClass<"X86", [i8], 8, (add AL, CL, DL, BL)>;
def GR8_ABCD_H : RegisterClass<"X86", [i8], 8, (add AH, CH, DH, BH)>;
def GR16_ABCD : RegisterClass<"X86", [i16], 16, (add AX, CX, DX, BX)> {
  let SubRegClasses = [(GR8_ABCD_L sub_8bit), (GR8_ABCD_H sub_8bit_hi)];
}
def GR32_ABCD : RegisterClass<"X86", [i32], 32, (add EAX, ECX, EDX, EBX)> {
  let SubRegClasses = [(GR8_ABCD_L sub_8bit),
                       (GR8_ABCD_H sub_8bit_hi),
                       (GR16_ABCD sub_16bit)];
}
def GR64_ABCD : RegisterClass<"X86", [i64], 64, (add RAX, RCX, RDX, RBX)> {
  let SubRegClasses = [(GR8_ABCD_L sub_8bit),
                       (GR8_ABCD_H sub_8bit_hi),
                       (GR16_ABCD sub_16bit),
                       (GR32_ABCD sub_32bit)];
}
def GR32_TC   : RegisterClass<"X86", [i32], 32, (add EAX, ECX, EDX)> {
  let SubRegClasses = [(GR8 sub_8bit, sub_8bit_hi), (GR16 sub_16bit)];
}
def GR64_TC   : RegisterClass<"X86", [i64], 64, (add RAX, RCX, RDX, RSI, RDI,
                                                     R8, R9, R11, RIP)> {
  let SubRegClasses = [(GR8 sub_8bit, sub_8bit_hi),
                       (GR16 sub_16bit),
                       (GR32_TC sub_32bit)];
}

def GR64_TCW64 : RegisterClass<"X86", [i64], 64, (add RAX, RCX, RDX,
                                                      R8, R9, R11)>;

// GR8_NOREX - GR8 registers which do not require a REX prefix.
def GR8_NOREX : RegisterClass<"X86", [i8], 8,
                              (add AL, CL, DL, AH, CH, DH, BL, BH)> {
  let AltOrders = [(sub GR8_NOREX, AH, BH, CH, DH)];
  let AltOrderSelect = [{
    return MF.getTarget().getSubtarget<X86Subtarget>().is64Bit();
  }];
}
// GR16_NOREX - GR16 registers which do not require a REX prefix.
def GR16_NOREX : RegisterClass<"X86", [i16], 16,
                               (add AX, CX, DX, SI, DI, BX, BP, SP)> {
  let SubRegClasses = [(GR8_NOREX sub_8bit, sub_8bit_hi)];
}
// GR32_NOREX - GR32 registers which do not require a REX prefix.
def GR32_NOREX : RegisterClass<"X86", [i32], 32,
                               (add EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP)> {
  let SubRegClasses = [(GR8_NOREX sub_8bit, sub_8bit_hi),
                       (GR16_NOREX sub_16bit)];
}
// GR64_NOREX - GR64 registers which do not require a REX prefix.
def GR64_NOREX : RegisterClass<"X86", [i64], 64,
                            (add RAX, RCX, RDX, RSI, RDI, RBX, RBP, RSP, RIP)> {
  let SubRegClasses = [(GR8_NOREX sub_8bit, sub_8bit_hi),
                       (GR16_NOREX sub_16bit),
                       (GR32_NOREX sub_32bit)];
}

// GR32_NOAX - GR32 registers except EAX. Used by AddRegFrm of XCHG32 in 64-bit
// mode to prevent encoding using the 0x90 NOP encoding. xchg %eax, %eax needs
// to clear upper 32-bits of RAX so is not a NOP.
def GR32_NOAX : RegisterClass<"X86", [i32], 32, (sub GR32, EAX)> {
  let SubRegClasses = [(GR8 sub_8bit, sub_8bit_hi), (GR16 sub_16bit)];
}

// GR32_NOSP - GR32 registers except ESP.
def GR32_NOSP : RegisterClass<"X86", [i32], 32, (sub GR32, ESP)> {
  let SubRegClasses = [(GR8 sub_8bit, sub_8bit_hi), (GR16 sub_16bit)];
}

// GR64_NOSP - GR64 registers except RSP (and RIP).
def GR64_NOSP : RegisterClass<"X86", [i64], 64, (sub GR64, RSP, RIP)> {
  let SubRegClasses = [(GR8 sub_8bit, sub_8bit_hi),
                       (GR16 sub_16bit),
                       (GR32_NOSP sub_32bit)];
}

// GR32_NOREX_NOSP - GR32 registers which do not require a REX prefix except
// ESP.
def GR32_NOREX_NOSP : RegisterClass<"X86", [i32], 32,
                                    (and GR32_NOREX, GR32_NOSP)> {
  let SubRegClasses = [(GR8_NOREX sub_8bit, sub_8bit_hi),
                       (GR16_NOREX sub_16bit)];
}

// GR64_NOREX_NOSP - GR64_NOREX registers except RSP.
def GR64_NOREX_NOSP : RegisterClass<"X86", [i64], 64,
                                    (and GR64_NOREX, GR64_NOSP)> {
  let SubRegClasses = [(GR8_NOREX sub_8bit, sub_8bit_hi),
                       (GR16_NOREX sub_16bit),
                       (GR32_NOREX_NOSP sub_32bit)];
}

// A class to support the 'A' assembler constraint: EAX then EDX.
def GR32_AD : RegisterClass<"X86", [i32], 32, (add EAX, EDX)> {
  let SubRegClasses = [(GR8_ABCD_L sub_8bit),
                       (GR8_ABCD_H sub_8bit_hi),
                       (GR16_ABCD sub_16bit)];
}

// Scalar SSE2 floating point registers.
def FR32 : RegisterClass<"X86", [f32], 32, (sequence "XMM%u", 0, 15)>;

def FR64 : RegisterClass<"X86", [f64], 64, (add FR32)>;


// FIXME: This sets up the floating point register files as though they are f64
// values, though they really are f80 values.  This will cause us to spill
// values as 64-bit quantities instead of 80-bit quantities, which is much much
// faster on common hardware.  In reality, this should be controlled by a
// command line option or something.

def RFP32 : RegisterClass<"X86",[f32], 32, (sequence "FP%u", 0, 6)>;
def RFP64 : RegisterClass<"X86",[f64], 32, (add RFP32)>;
def RFP80 : RegisterClass<"X86",[f80], 32, (add RFP32)>;

// Floating point stack registers (these are not allocatable by the
// register allocator - the floating point stackifier is responsible
// for transforming FPn allocations to STn registers)
def RST : RegisterClass<"X86", [f80, f64, f32], 32, (sequence "ST%u", 0, 7)> {
  let isAllocatable = 0;
}

// Generic vector registers: VR64 and VR128.
def VR64: RegisterClass<"X86", [x86mmx], 64, (sequence "MM%u", 0, 7)>;
def VR128 : RegisterClass<"X86", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
                          128, (add FR32)> {
  let SubRegClasses = [(FR32 sub_ss), (FR64 sub_sd)];
}

def VR256 : RegisterClass<"X86", [v32i8, v16i16, v8i32, v4i64, v8f32, v4f64],
                          256, (sequence "YMM%u", 0, 15)> {
  let SubRegClasses = [(FR32 sub_ss), (FR64 sub_sd), (VR128 sub_xmm)];
}

// Status flags registers.
def CCR : RegisterClass<"X86", [i32], 32, (add EFLAGS)> {
  let CopyCost = -1;  // Don't allow copying of status registers.
  let isAllocatable = 0;
}
