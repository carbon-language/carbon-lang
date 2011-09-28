//===- MipsInstrFPU.td - Mips FPU Instruction Information --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Mips FPU instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Floating Point Instructions
// ------------------------
// * 64bit fp:
//    - 32 64-bit registers (default mode)
//    - 16 even 32-bit registers (32-bit compatible mode) for
//      single and double access.
// * 32bit fp:
//    - 16 even 32-bit registers - single and double (aliased)
//    - 32 32-bit registers (within single-only mode)
//===----------------------------------------------------------------------===//

// Floating Point Compare and Branch
def SDT_MipsFPBrcond : SDTypeProfile<0, 2, [SDTCisInt<0>,
                                            SDTCisVT<1, OtherVT>]>;
def SDT_MipsFPCmp : SDTypeProfile<0, 3, [SDTCisSameAs<0, 1>, SDTCisFP<1>,
                                         SDTCisVT<2, i32>]>;
def SDT_MipsCMovFP : SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>,
                                          SDTCisSameAs<1, 2>]>;
def SDT_MipsBuildPairF64 : SDTypeProfile<1, 2, [SDTCisVT<0, f64>,
                                                SDTCisVT<1, i32>,
                                                SDTCisSameAs<1, 2>]>;
def SDT_MipsExtractElementF64 : SDTypeProfile<1, 2, [SDTCisVT<0, i32>,
                                                     SDTCisVT<1, f64>,
                                                     SDTCisVT<2, i32>]>;

def MipsFPCmp : SDNode<"MipsISD::FPCmp", SDT_MipsFPCmp, [SDNPOutGlue]>;
def MipsCMovFP_T : SDNode<"MipsISD::CMovFP_T", SDT_MipsCMovFP, [SDNPInGlue]>;
def MipsCMovFP_F : SDNode<"MipsISD::CMovFP_F", SDT_MipsCMovFP, [SDNPInGlue]>;
def MipsFPBrcond : SDNode<"MipsISD::FPBrcond", SDT_MipsFPBrcond,
                          [SDNPHasChain, SDNPOptInGlue]>;
def MipsBuildPairF64 : SDNode<"MipsISD::BuildPairF64", SDT_MipsBuildPairF64>;
def MipsExtractElementF64 : SDNode<"MipsISD::ExtractElementF64",
                                   SDT_MipsExtractElementF64>;

// Operand for printing out a condition code.
let PrintMethod = "printFCCOperand" in
  def condcode : Operand<i32>;

//===----------------------------------------------------------------------===//
// Feature predicates.
//===----------------------------------------------------------------------===//

def IsFP64bit        : Predicate<"Subtarget.isFP64bit()">;
def NotFP64bit       : Predicate<"!Subtarget.isFP64bit()">;
def IsSingleFloat    : Predicate<"Subtarget.isSingleFloat()">;
def IsNotSingleFloat : Predicate<"!Subtarget.isSingleFloat()">;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//
// A set of multiclasses is used to address the register usage.
//
// S   - single precision in 16 32bit even fp registers
//       single precision in 32 32bit fp registers in SingleOnly mode
// D32 - double precision in 16 32bit even fp registers
// D64 - double precision in 32 64bit fp registers (In64BitMode)
//
// Only S and D32 are supported right now.
//===----------------------------------------------------------------------===//

// Unary instruction without pattern.
class FFR1<bits<6> funct, bits<5> fmt, RegisterClass RCDst,
           RegisterClass RCSrc, string asmstr>:
  FFR<0x11, funct, fmt, (outs RCDst:$fd), (ins RCSrc:$fs),
      !strconcat(asmstr, "\t$fd, $fs"), []> {
  let ft = 0;
}

// Unary instruction with pattern.
// All operands belong to the same register class. 
class FFR1P<bits<6> funct, bits<5> fmt, RegisterClass RC, string asmstr,
            SDNode FOp>:
  FFR1<funct, fmt, RC, RC, asmstr> {
  let Pattern = [(set RC:$fd, (FOp RC:$fs))];
}

// Binary instruction with pattern.
// All operands belong to the same register class. 
class FFR2<bits<6> funct, bits<5> fmt, RegisterClass RC, string asmstr,
          SDNode FOp, bit isComm = 0>:
  FFR<0x11, funct, fmt, (outs RC:$fd), (ins RC:$fs, RC:$ft),
      !strconcat(asmstr, "\t$fd, $fs, $ft"),
      [(set RC:$fd, (FOp RC:$fs, RC:$ft))]> {
  let isCommutable = isComm;
}

// Multiclasses.
// Unary instruction without pattern.
multiclass FFR1_1<bits<6> funct, string asmstr>
{
  def _S   : FFR1<funct, 16, FGR32, FGR32, asmstr>;
  def _D32 : FFR1<funct, 17, FGR32, AFGR64, asmstr>, Requires<[NotFP64bit]>;
}

// Unary instruction with pattern.
// All operands belong to the same register class. 
multiclass FFR1_2<bits<6> funct, string asmstr, SDNode FOp>
{
  def _S   : FFR1P<funct, 16, FGR32, asmstr, FOp>;
  def _D32 : FFR1P<funct, 17, AFGR64, asmstr, FOp>, Requires<[NotFP64bit]>;
}

// Binary instruction with pattern.
// All operands belong to the same register class. 
multiclass FFR1_4<bits<6> funct, string asmstr, SDNode FOp, bit isComm = 0> {
  def _S   : FFR2<funct, 16, FGR32, asmstr, FOp, isComm>;
  def _D32 : FFR2<funct, 17, AFGR64, asmstr, FOp, isComm>,
             Requires<[NotFP64bit]>;
}

//===----------------------------------------------------------------------===//
// Floating Point Instructions
//===----------------------------------------------------------------------===//

let ft = 0 in {
  defm FLOOR_W : FFR1_1<0b001111, "floor.w">;
  defm CEIL_W  : FFR1_1<0b001110, "ceil.w">;
  defm ROUND_W : FFR1_1<0b001100, "round.w">;
  defm TRUNC_W : FFR1_1<0b001101, "trunc.w">;
  defm CVTW    : FFR1_1<0b100100, "cvt.w">;

  defm FABS    : FFR1_2<0b000101, "abs",  fabs>;
  defm FNEG    : FFR1_2<0b000111, "neg",  fneg>;
  defm FSQRT   : FFR1_2<0b000100, "sqrt", fsqrt>;

  /// Convert to Single Precison
  def CVTS_W32 : FFR1<0b100000, 0x2, FGR32,  FGR32,  "cvt.s.w">;

  let Predicates = [IsNotSingleFloat] in {
    /// Ceil to long signed integer
    def CEIL_LS   : FFR1<0b001010, 0x0, FGR32, FGR32, "ceil.l">;
    def CEIL_LD   : FFR1<0b001010, 0x1, AFGR64, AFGR64, "ceil.l">;

    /// Round to long signed integer
    def ROUND_LS  : FFR1<0b001000, 0x0, FGR32, FGR32, "round.l">;
    def ROUND_LD  : FFR1<0b001000, 0x1, AFGR64, AFGR64, "round.l">;

    /// Floor to long signed integer
    def FLOOR_LS  : FFR1<0b001011, 0x0, FGR32, FGR32, "floor.l">;
    def FLOOR_LD  : FFR1<0b001011, 0x1, AFGR64, AFGR64, "floor.l">;

    /// Trunc to long signed integer
    def TRUNC_LS  : FFR1<0b001001, 0x0, FGR32, FGR32, "trunc.l">;
    def TRUNC_LD  : FFR1<0b001001, 0x1, AFGR64, AFGR64, "trunc.l">;

    /// Convert to long signed integer
    def CVTL_S    : FFR1<0b100101, 0x0, FGR32, FGR32, "cvt.l">;
    def CVTL_D    : FFR1<0b100101, 0x1, AFGR64, AFGR64, "cvt.l">;

    /// Convert to Double Precison
    def CVTD_S : FFR1<0b100001, 0x0, AFGR64, FGR32, "cvt.d.s">;
    def CVTD_W32 : FFR1<0b100001, 0x2, AFGR64, FGR32, "cvt.d.w">;
    def CVTD_L32 : FFR1<0b100001, 0x3, AFGR64, AFGR64, "cvt.d.l">;

    /// Convert to Single Precison
    def CVTS_D32 : FFR1<0b100000, 0x1, FGR32, AFGR64, "cvt.s.d">;
    def CVTS_L32 : FFR1<0b100000, 0x3, FGR32, AFGR64, "cvt.s.l">;
  }
}

// The odd-numbered registers are only referenced when doing loads,
// stores, and moves between floating-point and integer registers.
// When defining instructions, we reference all 32-bit registers,
// regardless of register aliasing.
let fd = 0 in {
  /// Move Control Registers From/To CPU Registers
  def CFC1  : FFR<0x11, 0x0, 0x2, (outs CPURegs:$rt), (ins CCR:$fs),
                  "cfc1\t$rt, $fs", []>;

  def CTC1  : FFR<0x11, 0x0, 0x6, (outs CCR:$rt), (ins CPURegs:$fs),
                  "ctc1\t$fs, $rt", []>;

  def MFC1  : FFR<0x11, 0x00, 0x00, (outs CPURegs:$rt), (ins FGR32:$fs),
                  "mfc1\t$rt, $fs",
                  [(set CPURegs:$rt, (bitconvert FGR32:$fs))]>;

  def MTC1  : FFR<0x11, 0x00, 0x04, (outs FGR32:$fs), (ins CPURegs:$rt),
                  "mtc1\t$rt, $fs",
                  [(set FGR32:$fs, (bitconvert CPURegs:$rt))]>;
}

def FMOV_S : FFR<0x11, 0b000110, 0x0, (outs FGR32:$fd), (ins FGR32:$fs),
                   "mov.s\t$fd, $fs", []>;
def FMOV_D32 : FFR<0x11, 0b000110, 0x1, (outs AFGR64:$fd), (ins AFGR64:$fs),
                   "mov.d\t$fd, $fs", []>;

/// Floating Point Memory Instructions
let Predicates = [IsNotSingleFloat] in {
  def LDC1 : FFI<0b110101, (outs AFGR64:$ft), (ins mem:$addr),
                 "ldc1\t$ft, $addr", [(set AFGR64:$ft, (load addr:$addr))]>;

  def SDC1 : FFI<0b111101, (outs), (ins AFGR64:$ft, mem:$addr),
                 "sdc1\t$ft, $addr", [(store AFGR64:$ft, addr:$addr)]>;
}

// LWC1 and SWC1 can always be emitted with odd registers.
def LWC1  : FFI<0b110001, (outs FGR32:$ft), (ins mem:$addr), "lwc1\t$ft, $addr",
               [(set FGR32:$ft, (load addr:$addr))]>;
def SWC1  : FFI<0b111001, (outs), (ins FGR32:$ft, mem:$addr),
               "swc1\t$ft, $addr", [(store FGR32:$ft, addr:$addr)]>;

/// Floating-point Aritmetic
defm FADD : FFR1_4<0x10, "add", fadd, 1>;
defm FDIV : FFR1_4<0x03, "div", fdiv>;
defm FMUL : FFR1_4<0x02, "mul", fmul, 1>;
defm FSUB : FFR1_4<0x01, "sub", fsub>;

//===----------------------------------------------------------------------===//
// Floating Point Branch Codes
//===----------------------------------------------------------------------===//
// Mips branch codes. These correspond to condcode in MipsInstrInfo.h.
// They must be kept in synch.
def MIPS_BRANCH_F  : PatLeaf<(i32 0)>;
def MIPS_BRANCH_T  : PatLeaf<(i32 1)>;

/// Floating Point Branch of False/True (Likely)
let isBranch=1, isTerminator=1, hasDelaySlot=1, base=0x8, Uses=[FCR31] in
  class FBRANCH<PatLeaf op, string asmstr> : FFI<0x11, (outs),
        (ins brtarget:$dst), !strconcat(asmstr, "\t$dst"),
        [(MipsFPBrcond op, bb:$dst)]>;

def BC1F  : FBRANCH<MIPS_BRANCH_F,  "bc1f">;
def BC1T  : FBRANCH<MIPS_BRANCH_T,  "bc1t">;

//===----------------------------------------------------------------------===//
// Floating Point Flag Conditions
//===----------------------------------------------------------------------===//
// Mips condition codes. They must correspond to condcode in MipsInstrInfo.h.
// They must be kept in synch.
def MIPS_FCOND_F    : PatLeaf<(i32 0)>;
def MIPS_FCOND_UN   : PatLeaf<(i32 1)>;
def MIPS_FCOND_OEQ  : PatLeaf<(i32 2)>;
def MIPS_FCOND_UEQ  : PatLeaf<(i32 3)>;
def MIPS_FCOND_OLT  : PatLeaf<(i32 4)>;
def MIPS_FCOND_ULT  : PatLeaf<(i32 5)>;
def MIPS_FCOND_OLE  : PatLeaf<(i32 6)>;
def MIPS_FCOND_ULE  : PatLeaf<(i32 7)>;
def MIPS_FCOND_SF   : PatLeaf<(i32 8)>;
def MIPS_FCOND_NGLE : PatLeaf<(i32 9)>;
def MIPS_FCOND_SEQ  : PatLeaf<(i32 10)>;
def MIPS_FCOND_NGL  : PatLeaf<(i32 11)>;
def MIPS_FCOND_LT   : PatLeaf<(i32 12)>;
def MIPS_FCOND_NGE  : PatLeaf<(i32 13)>;
def MIPS_FCOND_LE   : PatLeaf<(i32 14)>;
def MIPS_FCOND_NGT  : PatLeaf<(i32 15)>;

/// Floating Point Compare
let Defs=[FCR31] in {
  def FCMP_S : FCC<0x0, (outs), (ins FGR32:$fs, FGR32:$ft, condcode:$cc),
                     "c.$cc.s\t$fs, $ft",
                     [(MipsFPCmp FGR32:$fs, FGR32:$ft, imm:$cc)]>;

  def FCMP_D32 : FCC<0x1, (outs), (ins AFGR64:$fs, AFGR64:$ft, condcode:$cc),
                     "c.$cc.d\t$fs, $ft",
                     [(MipsFPCmp AFGR64:$fs, AFGR64:$ft, imm:$cc)]>,
                     Requires<[NotFP64bit]>;
}


// Conditional moves:
// These instructions are expanded in
// MipsISelLowering::EmitInstrWithCustomInserter if target does not have
// conditional move instructions.
// flag:int, data:float
let usesCustomInserter = 1, Constraints = "$F = $dst" in
class CondMovIntFP<RegisterClass RC, bits<5> fmt, bits<6> func,
                   string instr_asm> :
  FFR<0x11, func, fmt, (outs RC:$dst), (ins RC:$T, CPURegs:$cond, RC:$F),
      !strconcat(instr_asm, "\t$dst, $T, $cond"), []>;

def MOVZ_S : CondMovIntFP<FGR32, 16, 18, "movz.s">;
def MOVN_S : CondMovIntFP<FGR32, 16, 19, "movn.s">;

let Predicates = [NotFP64bit] in {
  def MOVZ_D : CondMovIntFP<AFGR64, 17, 18, "movz.d">;
  def MOVN_D : CondMovIntFP<AFGR64, 17, 19, "movn.d">;
}

defm : MovzPats<FGR32, MOVZ_S>;
defm : MovnPats<FGR32, MOVN_S>;

let Predicates = [NotFP64bit] in {
  defm : MovzPats<AFGR64, MOVZ_D>;
  defm : MovnPats<AFGR64, MOVN_D>;
}

let usesCustomInserter = 1, Uses = [FCR31], Constraints = "$F = $dst" in {
// flag:float, data:int
class CondMovFPInt<SDNode cmov, bits<1> tf, string instr_asm> :
  FCMOV<tf, (outs CPURegs:$dst), (ins CPURegs:$T, CPURegs:$F),
        !strconcat(instr_asm, "\t$dst, $T, $$fcc0"),
        [(set CPURegs:$dst, (cmov CPURegs:$T, CPURegs:$F))]>;

// flag:float, data:float
class CondMovFPFP<RegisterClass RC, SDNode cmov, bits<5> fmt, bits<1> tf,
                  string instr_asm> :
  FFCMOV<fmt, tf, (outs RC:$dst), (ins RC:$T, RC:$F),
         !strconcat(instr_asm, "\t$dst, $T, $$fcc0"),
         [(set RC:$dst, (cmov RC:$T, RC:$F))]>;
}

def MOVT : CondMovFPInt<MipsCMovFP_T, 1, "movt">;
def MOVF : CondMovFPInt<MipsCMovFP_F, 0, "movf">;
def MOVT_S : CondMovFPFP<FGR32, MipsCMovFP_T, 16, 1, "movt.s">;
def MOVF_S : CondMovFPFP<FGR32, MipsCMovFP_F, 16, 0, "movf.s">;

let Predicates = [NotFP64bit] in {
  def MOVT_D : CondMovFPFP<AFGR64, MipsCMovFP_T, 17, 1, "movt.d">;
  def MOVF_D : CondMovFPFP<AFGR64, MipsCMovFP_F, 17, 0, "movf.d">;
}

//===----------------------------------------------------------------------===//
// Floating Point Pseudo-Instructions
//===----------------------------------------------------------------------===//
def MOVCCRToCCR : MipsPseudo<(outs CCR:$dst), (ins CCR:$src),
                             "# MOVCCRToCCR", []>;

// This pseudo instr gets expanded into 2 mtc1 instrs after register
// allocation.
def BuildPairF64 :
  MipsPseudo<(outs AFGR64:$dst),
             (ins CPURegs:$lo, CPURegs:$hi), "",
             [(set AFGR64:$dst, (MipsBuildPairF64 CPURegs:$lo, CPURegs:$hi))]>;

// This pseudo instr gets expanded into 2 mfc1 instrs after register
// allocation.
// if n is 0, lower part of src is extracted.
// if n is 1, higher part of src is extracted.
def ExtractElementF64 :
  MipsPseudo<(outs CPURegs:$dst),
             (ins AFGR64:$src, i32imm:$n), "",
             [(set CPURegs:$dst,
               (MipsExtractElementF64 AFGR64:$src, imm:$n))]>;

//===----------------------------------------------------------------------===//
// Floating Point Patterns
//===----------------------------------------------------------------------===//
def fpimm0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fpimm0neg : PatLeaf<(fpimm), [{
  return N->isExactlyValue(-0.0);
}]>;

def : Pat<(f32 fpimm0), (MTC1 ZERO)>;
def : Pat<(f32 fpimm0neg), (FNEG_S (MTC1 ZERO))>;

def : Pat<(f32 (sint_to_fp CPURegs:$src)), (CVTS_W32 (MTC1 CPURegs:$src))>;
def : Pat<(f64 (sint_to_fp CPURegs:$src)), (CVTD_W32 (MTC1 CPURegs:$src))>;

def : Pat<(i32 (fp_to_sint FGR32:$src)), (MFC1 (TRUNC_W_S FGR32:$src))>;
def : Pat<(i32 (fp_to_sint AFGR64:$src)), (MFC1 (TRUNC_W_D32 AFGR64:$src))>;

let Predicates = [NotFP64bit] in {
  def : Pat<(f32 (fround AFGR64:$src)), (CVTS_D32 AFGR64:$src)>;
  def : Pat<(f64 (fextend FGR32:$src)), (CVTD_S FGR32:$src)>;
}

