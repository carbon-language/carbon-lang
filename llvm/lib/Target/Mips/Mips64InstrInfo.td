//===- Mips64InstrInfo.td - Mips64 Instruction Information -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips64 instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mips Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

// Instruction operand types
def shamt_64       : Operand<i64>;

// Unsigned Operand
def uimm16_64      : Operand<i64> {
  let PrintMethod = "printUnsignedImm";
}

// Transformation Function - get Imm - 32.
def Subtract32 : SDNodeXForm<imm, [{
  return getImm(N, (unsigned)N->getZExtValue() - 32);
}]>;

// shamt must fit in 6 bits.
def immZExt6 : ImmLeaf<i32, [{return Imm == (Imm & 0x3f);}]>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//
let DecoderNamespace = "Mips64" in {

multiclass Atomic2Ops64<PatFrag Op> {
  def NAME : Atomic2Ops<Op, CPU64Regs, CPURegs>,
             Requires<[NotN64, HasStdEnc]>;
  def _P8  : Atomic2Ops<Op, CPU64Regs, CPU64Regs>,
             Requires<[IsN64, HasStdEnc]> {
    let isCodeGenOnly = 1;
  }
}

multiclass AtomicCmpSwap64<PatFrag Op>  {
  def NAME : AtomicCmpSwap<Op, CPU64Regs, CPURegs>,
             Requires<[NotN64, HasStdEnc]>;
  def _P8  : AtomicCmpSwap<Op, CPU64Regs, CPU64Regs>,
             Requires<[IsN64, HasStdEnc]> {
    let isCodeGenOnly = 1;
  }
}
}
let usesCustomInserter = 1, Predicates = [HasStdEnc],
  DecoderNamespace = "Mips64" in {
  defm ATOMIC_LOAD_ADD_I64  : Atomic2Ops64<atomic_load_add_64>;
  defm ATOMIC_LOAD_SUB_I64  : Atomic2Ops64<atomic_load_sub_64>;
  defm ATOMIC_LOAD_AND_I64  : Atomic2Ops64<atomic_load_and_64>;
  defm ATOMIC_LOAD_OR_I64   : Atomic2Ops64<atomic_load_or_64>;
  defm ATOMIC_LOAD_XOR_I64  : Atomic2Ops64<atomic_load_xor_64>;
  defm ATOMIC_LOAD_NAND_I64 : Atomic2Ops64<atomic_load_nand_64>;
  defm ATOMIC_SWAP_I64      : Atomic2Ops64<atomic_swap_64>;
  defm ATOMIC_CMP_SWAP_I64  : AtomicCmpSwap64<atomic_cmp_swap_64>;
}

/// Pseudo instructions for loading and storing accumulator registers.
let isPseudo = 1 in {
  defm LOAD_AC128  : LoadM<"load_ac128", ACRegs128>;
  defm STORE_AC128 : StoreM<"store_ac128", ACRegs128>;
}

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//
let DecoderNamespace = "Mips64" in {
/// Arithmetic Instructions (ALU Immediate)
def DADDi   : ArithLogicI<"daddi", simm16_64, CPU64RegsOpnd>, ADDI_FM<0x18>;
def DADDiu  : ArithLogicI<"daddiu", simm16_64, CPU64RegsOpnd, immSExt16, add>,
              ADDI_FM<0x19>, IsAsCheapAsAMove;
def DANDi   : ArithLogicI<"andi", uimm16_64, CPU64RegsOpnd, immZExt16, and>,
              ADDI_FM<0xc>;
def SLTi64  : SetCC_I<"slti", setlt, simm16_64, immSExt16, CPU64Regs>,
              SLTI_FM<0xa>;
def SLTiu64 : SetCC_I<"sltiu", setult, simm16_64, immSExt16, CPU64Regs>,
              SLTI_FM<0xb>;
def ORi64   : ArithLogicI<"ori", uimm16_64, CPU64RegsOpnd, immZExt16, or>,
              ADDI_FM<0xd>;
def XORi64  : ArithLogicI<"xori", uimm16_64, CPU64RegsOpnd, immZExt16, xor>,
              ADDI_FM<0xe>;
def LUi64   : LoadUpper<"lui", CPU64Regs, uimm16_64>, LUI_FM;

/// Arithmetic Instructions (3-Operand, R-Type)
def DADD   : ArithLogicR<"dadd", CPU64RegsOpnd>, ADD_FM<0, 0x2c>;
def DADDu  : ArithLogicR<"daddu", CPU64RegsOpnd, 1, IIAlu, add>,
                              ADD_FM<0, 0x2d>;
def DSUBu  : ArithLogicR<"dsubu", CPU64RegsOpnd, 0, IIAlu, sub>,
                              ADD_FM<0, 0x2f>;
def SLT64  : SetCC_R<"slt", setlt, CPU64Regs>, ADD_FM<0, 0x2a>;
def SLTu64 : SetCC_R<"sltu", setult, CPU64Regs>, ADD_FM<0, 0x2b>;
def AND64  : ArithLogicR<"and", CPU64RegsOpnd, 1, IIAlu, and>, ADD_FM<0, 0x24>;
def OR64   : ArithLogicR<"or", CPU64RegsOpnd, 1, IIAlu, or>, ADD_FM<0, 0x25>;
def XOR64  : ArithLogicR<"xor", CPU64RegsOpnd, 1, IIAlu, xor>, ADD_FM<0, 0x26>;
def NOR64  : LogicNOR<"nor", CPU64RegsOpnd>, ADD_FM<0, 0x27>;

/// Shift Instructions
def DSLL   : shift_rotate_imm<"dsll", shamt, CPU64RegsOpnd, shl, immZExt6>,
             SRA_FM<0x38, 0>;
def DSRL   : shift_rotate_imm<"dsrl", shamt, CPU64RegsOpnd, srl, immZExt6>,
             SRA_FM<0x3a, 0>;
def DSRA   : shift_rotate_imm<"dsra", shamt, CPU64RegsOpnd, sra, immZExt6>,
             SRA_FM<0x3b, 0>;
def DSLLV  : shift_rotate_reg<"dsllv", CPU64RegsOpnd, shl>, SRLV_FM<0x14, 0>;
def DSRLV  : shift_rotate_reg<"dsrlv", CPU64RegsOpnd, srl>, SRLV_FM<0x16, 0>;
def DSRAV  : shift_rotate_reg<"dsrav", CPU64RegsOpnd, sra>, SRLV_FM<0x17, 0>;
def DSLL32 : shift_rotate_imm<"dsll32", shamt, CPU64RegsOpnd>, SRA_FM<0x3c, 0>;
def DSRL32 : shift_rotate_imm<"dsrl32", shamt, CPU64RegsOpnd>, SRA_FM<0x3e, 0>;
def DSRA32 : shift_rotate_imm<"dsra32", shamt, CPU64RegsOpnd>, SRA_FM<0x3f, 0>;
}
// Rotate Instructions
let Predicates = [HasMips64r2, HasStdEnc],
    DecoderNamespace = "Mips64" in {
  def DROTR  : shift_rotate_imm<"drotr", shamt, CPU64RegsOpnd, rotr, immZExt6>,
               SRA_FM<0x3a, 1>;
  def DROTRV : shift_rotate_reg<"drotrv", CPU64RegsOpnd, rotr>,
               SRLV_FM<0x16, 1>;
}

let DecoderNamespace = "Mips64" in {
/// Load and Store Instructions
///  aligned
defm LB64  : LoadM<"lb", CPU64Regs, sextloadi8, IILoad>, LW_FM<0x20>;
defm LBu64 : LoadM<"lbu", CPU64Regs, zextloadi8, IILoad>, LW_FM<0x24>;
defm LH64  : LoadM<"lh", CPU64Regs, sextloadi16, IILoad>, LW_FM<0x21>;
defm LHu64 : LoadM<"lhu", CPU64Regs, zextloadi16, IILoad>, LW_FM<0x25>;
defm LW64  : LoadM<"lw", CPU64Regs, sextloadi32, IILoad>, LW_FM<0x23>;
defm LWu64 : LoadM<"lwu", CPU64Regs, zextloadi32, IILoad>, LW_FM<0x27>;
defm SB64  : StoreM<"sb", CPU64Regs, truncstorei8, IIStore>, LW_FM<0x28>;
defm SH64  : StoreM<"sh", CPU64Regs, truncstorei16, IIStore>, LW_FM<0x29>;
defm SW64  : StoreM<"sw", CPU64Regs, truncstorei32, IIStore>, LW_FM<0x2b>;
defm LD    : LoadM<"ld", CPU64Regs, load, IILoad>, LW_FM<0x37>;
defm SD    : StoreM<"sd", CPU64Regs, store, IIStore>, LW_FM<0x3f>;

/// load/store left/right
defm LWL64 : LoadLeftRightM<"lwl", MipsLWL, CPU64Regs>, LW_FM<0x22>;
defm LWR64 : LoadLeftRightM<"lwr", MipsLWR, CPU64Regs>, LW_FM<0x26>;
defm SWL64 : StoreLeftRightM<"swl", MipsSWL, CPU64Regs>, LW_FM<0x2a>;
defm SWR64 : StoreLeftRightM<"swr", MipsSWR, CPU64Regs>, LW_FM<0x2e>;

defm LDL   : LoadLeftRightM<"ldl", MipsLDL, CPU64Regs>, LW_FM<0x1a>;
defm LDR   : LoadLeftRightM<"ldr", MipsLDR, CPU64Regs>, LW_FM<0x1b>;
defm SDL   : StoreLeftRightM<"sdl", MipsSDL, CPU64Regs>, LW_FM<0x2c>;
defm SDR   : StoreLeftRightM<"sdr", MipsSDR, CPU64Regs>, LW_FM<0x2d>;

/// Load-linked, Store-conditional
let Predicates = [NotN64, HasStdEnc] in {
  def LLD : LLBase<"lld", CPU64RegsOpnd, mem>, LW_FM<0x34>;
  def SCD : SCBase<"scd", CPU64RegsOpnd, mem>, LW_FM<0x3c>;
}

let Predicates = [IsN64, HasStdEnc], isCodeGenOnly = 1 in {
  def LLD_P8 : LLBase<"lld", CPU64RegsOpnd, mem64>, LW_FM<0x34>;
  def SCD_P8 : SCBase<"scd", CPU64RegsOpnd, mem64>, LW_FM<0x3c>;
}

/// Jump and Branch Instructions
def JR64   : IndirectBranch<CPU64Regs>, MTLO_FM<8>;
def BEQ64  : CBranch<"beq", seteq, CPU64RegsOpnd>, BEQ_FM<4>;
def BNE64  : CBranch<"bne", setne, CPU64RegsOpnd>, BEQ_FM<5>;
def BGEZ64 : CBranchZero<"bgez", setge, CPU64RegsOpnd>, BGEZ_FM<1, 1>;
def BGTZ64 : CBranchZero<"bgtz", setgt, CPU64RegsOpnd>, BGEZ_FM<7, 0>;
def BLEZ64 : CBranchZero<"blez", setle, CPU64RegsOpnd>, BGEZ_FM<6, 0>;
def BLTZ64 : CBranchZero<"bltz", setlt, CPU64RegsOpnd>, BGEZ_FM<1, 0>;
}
let DecoderNamespace = "Mips64" in
def JALR64 : JumpLinkReg<"jalr", CPU64Regs>, JALR_FM;
def JALR64Pseudo : JumpLinkRegPseudo<CPU64Regs, JALR64, RA_64>;
def TAILCALL64_R : JumpFR<CPU64Regs, MipsTailCall>, MTLO_FM<8>, IsTailCall;

let DecoderNamespace = "Mips64" in {
/// Multiply and Divide Instructions.
def DMULT  : Mult<"dmult", IIImul, CPU64RegsOpnd, [HI64, LO64]>,
             MULT_FM<0, 0x1c>;
def DMULTu : Mult<"dmultu", IIImul, CPU64RegsOpnd, [HI64, LO64]>,
             MULT_FM<0, 0x1d>;
def PseudoDMULT  : MultDivPseudo<DMULT, ACRegs128, CPU64RegsOpnd, MipsMult,
                                 IIImul>;
def PseudoDMULTu : MultDivPseudo<DMULTu, ACRegs128, CPU64RegsOpnd, MipsMultu,
                                 IIImul>;
def DSDIV : Div<"ddiv", IIIdiv, CPU64RegsOpnd, [HI64, LO64]>, MULT_FM<0, 0x1e>;
def DUDIV : Div<"ddivu", IIIdiv, CPU64RegsOpnd, [HI64, LO64]>, MULT_FM<0, 0x1f>;
def PseudoDSDIV : MultDivPseudo<DSDIV, ACRegs128, CPU64RegsOpnd, MipsDivRem,
                                IIIdiv, 0, 1, 1>;
def PseudoDUDIV : MultDivPseudo<DUDIV, ACRegs128, CPU64RegsOpnd, MipsDivRemU,
                                IIIdiv, 0, 1, 1>;

def MTHI64 : MoveToLOHI<"mthi", CPU64Regs, [HI64]>, MTLO_FM<0x11>;
def MTLO64 : MoveToLOHI<"mtlo", CPU64Regs, [LO64]>, MTLO_FM<0x13>;
def MFHI64 : MoveFromLOHI<"mfhi", CPU64Regs, [HI64]>, MFLO_FM<0x10>;
def MFLO64 : MoveFromLOHI<"mflo", CPU64Regs, [LO64]>, MFLO_FM<0x12>;

/// Sign Ext In Register Instructions.
def SEB64 : SignExtInReg<"seb", i8, CPU64Regs>, SEB_FM<0x10, 0x20>;
def SEH64 : SignExtInReg<"seh", i16, CPU64Regs>, SEB_FM<0x18, 0x20>;

/// Count Leading
def DCLZ : CountLeading0<"dclz", CPU64RegsOpnd>, CLO_FM<0x24>;
def DCLO : CountLeading1<"dclo", CPU64RegsOpnd>, CLO_FM<0x25>;

/// Double Word Swap Bytes/HalfWords
def DSBH : SubwordSwap<"dsbh", CPU64RegsOpnd>, SEB_FM<2, 0x24>;
def DSHD : SubwordSwap<"dshd", CPU64RegsOpnd>, SEB_FM<5, 0x24>;

def LEA_ADDiu64 : EffectiveAddress<"daddiu", CPU64Regs, mem_ea_64>, LW_FM<0x19>;

}
let DecoderNamespace = "Mips64" in {
def RDHWR64 : ReadHardware<CPU64Regs, HW64RegsOpnd>, RDHWR_FM;

def DEXT : ExtBase<"dext", CPU64RegsOpnd>, EXT_FM<3>;
let Pattern = []<dag> in {
  def DEXTU : ExtBase<"dextu", CPU64RegsOpnd>, EXT_FM<2>;
  def DEXTM : ExtBase<"dextm", CPU64RegsOpnd>, EXT_FM<1>;
}
def DINS : InsBase<"dins", CPU64RegsOpnd>, EXT_FM<7>;
let Pattern = []<dag> in {
  def DINSU : InsBase<"dinsu", CPU64RegsOpnd>, EXT_FM<6>;
  def DINSM : InsBase<"dinsm", CPU64RegsOpnd>, EXT_FM<5>;
}

let isCodeGenOnly = 1, rs = 0, shamt = 0 in {
  def DSLL64_32 : FR<0x00, 0x3c, (outs CPU64Regs:$rd), (ins CPURegs:$rt),
                     "dsll\t$rd, $rt, 32", [], IIAlu>;
  def SLL64_32 : FR<0x0, 0x00, (outs CPU64Regs:$rd), (ins CPURegs:$rt),
                    "sll\t$rd, $rt, 0", [], IIAlu>;
  def SLL64_64 : FR<0x0, 0x00, (outs CPU64Regs:$rd), (ins CPU64Regs:$rt),
                    "sll\t$rd, $rt, 0", [], IIAlu>;
}
}
//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// extended loads
let Predicates = [NotN64, HasStdEnc] in {
  def : MipsPat<(i64 (extloadi1  addr:$src)), (LB64 addr:$src)>;
  def : MipsPat<(i64 (extloadi8  addr:$src)), (LB64 addr:$src)>;
  def : MipsPat<(i64 (extloadi16 addr:$src)), (LH64 addr:$src)>;
  def : MipsPat<(i64 (extloadi32 addr:$src)), (LW64 addr:$src)>;
}
let Predicates = [IsN64, HasStdEnc] in {
  def : MipsPat<(i64 (extloadi1  addr:$src)), (LB64_P8 addr:$src)>;
  def : MipsPat<(i64 (extloadi8  addr:$src)), (LB64_P8 addr:$src)>;
  def : MipsPat<(i64 (extloadi16 addr:$src)), (LH64_P8 addr:$src)>;
  def : MipsPat<(i64 (extloadi32 addr:$src)), (LW64_P8 addr:$src)>;
}

// hi/lo relocs
def : MipsPat<(MipsHi tglobaladdr:$in), (LUi64 tglobaladdr:$in)>;
def : MipsPat<(MipsHi tblockaddress:$in), (LUi64 tblockaddress:$in)>;
def : MipsPat<(MipsHi tjumptable:$in), (LUi64 tjumptable:$in)>;
def : MipsPat<(MipsHi tconstpool:$in), (LUi64 tconstpool:$in)>;
def : MipsPat<(MipsHi tglobaltlsaddr:$in), (LUi64 tglobaltlsaddr:$in)>;
def : MipsPat<(MipsHi texternalsym:$in), (LUi64 texternalsym:$in)>;

def : MipsPat<(MipsLo tglobaladdr:$in), (DADDiu ZERO_64, tglobaladdr:$in)>;
def : MipsPat<(MipsLo tblockaddress:$in), (DADDiu ZERO_64, tblockaddress:$in)>;
def : MipsPat<(MipsLo tjumptable:$in), (DADDiu ZERO_64, tjumptable:$in)>;
def : MipsPat<(MipsLo tconstpool:$in), (DADDiu ZERO_64, tconstpool:$in)>;
def : MipsPat<(MipsLo tglobaltlsaddr:$in),
              (DADDiu ZERO_64, tglobaltlsaddr:$in)>;
def : MipsPat<(MipsLo texternalsym:$in), (DADDiu ZERO_64, texternalsym:$in)>;

def : MipsPat<(add CPU64Regs:$hi, (MipsLo tglobaladdr:$lo)),
              (DADDiu CPU64Regs:$hi, tglobaladdr:$lo)>;
def : MipsPat<(add CPU64Regs:$hi, (MipsLo tblockaddress:$lo)),
              (DADDiu CPU64Regs:$hi, tblockaddress:$lo)>;
def : MipsPat<(add CPU64Regs:$hi, (MipsLo tjumptable:$lo)),
              (DADDiu CPU64Regs:$hi, tjumptable:$lo)>;
def : MipsPat<(add CPU64Regs:$hi, (MipsLo tconstpool:$lo)),
              (DADDiu CPU64Regs:$hi, tconstpool:$lo)>;
def : MipsPat<(add CPU64Regs:$hi, (MipsLo tglobaltlsaddr:$lo)),
              (DADDiu CPU64Regs:$hi, tglobaltlsaddr:$lo)>;

def : WrapperPat<tglobaladdr, DADDiu, CPU64Regs>;
def : WrapperPat<tconstpool, DADDiu, CPU64Regs>;
def : WrapperPat<texternalsym, DADDiu, CPU64Regs>;
def : WrapperPat<tblockaddress, DADDiu, CPU64Regs>;
def : WrapperPat<tjumptable, DADDiu, CPU64Regs>;
def : WrapperPat<tglobaltlsaddr, DADDiu, CPU64Regs>;

defm : BrcondPats<CPU64Regs, BEQ64, BNE64, SLT64, SLTu64, SLTi64, SLTiu64,
                  ZERO_64>;

def : MipsPat<(brcond (i32 (setlt i64:$lhs, 1)), bb:$dst),
              (BLEZ64 i64:$lhs, bb:$dst)>;
def : MipsPat<(brcond (i32 (setgt i64:$lhs, -1)), bb:$dst),
              (BGEZ64 i64:$lhs, bb:$dst)>;

// setcc patterns
defm : SeteqPats<CPU64Regs, SLTiu64, XOR64, SLTu64, ZERO_64>;
defm : SetlePats<CPU64Regs, SLT64, SLTu64>;
defm : SetgtPats<CPU64Regs, SLT64, SLTu64>;
defm : SetgePats<CPU64Regs, SLT64, SLTu64>;
defm : SetgeImmPats<CPU64Regs, SLTi64, SLTiu64>;

// truncate
def : MipsPat<(i32 (trunc CPU64Regs:$src)),
              (SLL (EXTRACT_SUBREG CPU64Regs:$src, sub_32), 0)>,
      Requires<[IsN64, HasStdEnc]>;

// 32-to-64-bit extension
def : MipsPat<(i64 (anyext CPURegs:$src)), (SLL64_32 CPURegs:$src)>;
def : MipsPat<(i64 (zext CPURegs:$src)), (DSRL (DSLL64_32 CPURegs:$src), 32)>;
def : MipsPat<(i64 (sext CPURegs:$src)), (SLL64_32 CPURegs:$src)>;

// Sign extend in register
def : MipsPat<(i64 (sext_inreg CPU64Regs:$src, i32)),
              (SLL64_64 CPU64Regs:$src)>;

// bswap MipsPattern
def : MipsPat<(bswap CPU64Regs:$rt), (DSHD (DSBH CPU64Regs:$rt))>;

// mflo/hi patterns.
def : MipsPat<(i64 (ExtractLOHI ACRegs128:$ac, imm:$lohi_idx)),
              (EXTRACT_SUBREG ACRegs128:$ac, imm:$lohi_idx)>;

//===----------------------------------------------------------------------===//
// Instruction aliases
//===----------------------------------------------------------------------===//
def : InstAlias<"move $dst, $src",
                (DADDu CPU64RegsOpnd:$dst,  CPU64RegsOpnd:$src, ZERO_64), 1>,
      Requires<[HasMips64]>;
def : InstAlias<"move $dst, $src",
                (OR64 CPU64RegsOpnd:$dst, CPU64RegsOpnd:$src, ZERO_64), 1>,
      Requires<[HasMips64]>;
def : InstAlias<"and $rs, $rt, $imm",
                (DANDi CPU64RegsOpnd:$rs, CPU64RegsOpnd:$rt, uimm16_64:$imm),
                1>,
      Requires<[HasMips64]>;
def : InstAlias<"slt $rs, $rt, $imm",
                (SLTi64 CPURegsOpnd:$rs, CPU64Regs:$rt, simm16_64:$imm), 1>,
      Requires<[HasMips64]>;
def : InstAlias<"xor $rs, $rt, $imm",
                (XORi64 CPU64RegsOpnd:$rs, CPU64RegsOpnd:$rt, uimm16_64:$imm),
                1>,
      Requires<[HasMips64]>;
def : InstAlias<"not $rt, $rs",
                (NOR64 CPU64RegsOpnd:$rt, CPU64RegsOpnd:$rs, ZERO_64), 1>,
      Requires<[HasMips64]>;
def : InstAlias<"j $rs", (JR64 CPU64Regs:$rs), 0>, Requires<[HasMips64]>;
def : InstAlias<"jalr $rs", (JALR64 RA_64, CPU64Regs:$rs)>,
      Requires<[HasMips64]>;
def : InstAlias<"jal $rs", (JALR64 RA_64, CPU64Regs:$rs), 0>,
                 Requires<[HasMips64]>;
def : InstAlias<"jal $rd,$rs", (JALR64 CPU64Regs:$rd, CPU64Regs:$rs), 0>,
                 Requires<[HasMips64]>;
def : InstAlias<"daddu $rs, $rt, $imm",
                (DADDiu CPU64RegsOpnd:$rs, CPU64RegsOpnd:$rt, simm16_64:$imm),
                1>;
def : InstAlias<"dadd $rs, $rt, $imm",
                (DADDi CPU64RegsOpnd:$rs, CPU64RegsOpnd:$rt, simm16_64:$imm),
                1>;
def : InstAlias<"or $rs, $rt, $imm",
                (ORi64 CPU64RegsOpnd:$rs, CPU64RegsOpnd:$rt, uimm16_64:$imm),
                1>, Requires<[HasMips64]>;
def : InstAlias<"bnez $rs,$offset",
                 (BNE64 CPU64RegsOpnd:$rs, ZERO_64, brtarget:$offset), 1>,
                 Requires<[HasMips64]>;
def : InstAlias<"beqz $rs,$offset",
                 (BEQ64 CPU64RegsOpnd:$rs, ZERO_64, brtarget:$offset), 1>,
                 Requires<[HasMips64]>;

/// Move between CPU and coprocessor registers
let DecoderNamespace = "Mips64" in {
def DMFC0_3OP64 : MFC3OP<(outs CPU64RegsOpnd:$rt),
                         (ins CPU64RegsOpnd:$rd, uimm16:$sel),
                         "dmfc0\t$rt, $rd, $sel">, MFC3OP_FM<0x10, 1>;
def DMTC0_3OP64 : MFC3OP<(outs CPU64RegsOpnd:$rd, uimm16:$sel),
                         (ins CPU64RegsOpnd:$rt),
                         "dmtc0\t$rt, $rd, $sel">, MFC3OP_FM<0x10, 5>;
def DMFC2_3OP64 : MFC3OP<(outs CPU64RegsOpnd:$rt),
                         (ins CPU64RegsOpnd:$rd, uimm16:$sel),
                         "dmfc2\t$rt, $rd, $sel">, MFC3OP_FM<0x12, 1>;
def DMTC2_3OP64 : MFC3OP<(outs CPU64RegsOpnd:$rd, uimm16:$sel),
                         (ins CPU64RegsOpnd:$rt),
                         "dmtc2\t$rt, $rd, $sel">, MFC3OP_FM<0x12, 5>;
}

// Two operand (implicit 0 selector) versions:
def : InstAlias<"dmfc0 $rt, $rd",
                (DMFC0_3OP64 CPU64RegsOpnd:$rt, CPU64RegsOpnd:$rd, 0), 0>;
def : InstAlias<"dmtc0 $rt, $rd",
                (DMTC0_3OP64 CPU64RegsOpnd:$rd, 0, CPU64RegsOpnd:$rt), 0>;
def : InstAlias<"dmfc2 $rt, $rd",
                (DMFC2_3OP64 CPU64RegsOpnd:$rt, CPU64RegsOpnd:$rd, 0), 0>;
def : InstAlias<"dmtc2 $rt, $rd",
                (DMTC2_3OP64 CPU64RegsOpnd:$rd, 0, CPU64RegsOpnd:$rt), 0>;

