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
  def NAME : Atomic2Ops<Op, GPR64, GPR32>, Requires<[NotN64, HasStdEnc]>;
  def _P8  : Atomic2Ops<Op, GPR64, GPR64>, Requires<[IsN64, HasStdEnc]>;
}

multiclass AtomicCmpSwap64<PatFrag Op>  {
  def NAME : AtomicCmpSwap<Op, GPR64, GPR32>,
             Requires<[NotN64, HasStdEnc]>;
  def _P8  : AtomicCmpSwap<Op, GPR64, GPR64>,
             Requires<[IsN64, HasStdEnc]>;
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
let isPseudo = 1, isCodeGenOnly = 1 in {
  defm LOAD_AC128  : LoadM<"", ACRegs128>;
  defm STORE_AC128 : StoreM<"", ACRegs128>;
}

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//
let DecoderNamespace = "Mips64" in {
/// Arithmetic Instructions (ALU Immediate)
def DADDi   : ArithLogicI<"daddi", simm16_64, GPR64Opnd>, ADDI_FM<0x18>;
def DADDiu  : ArithLogicI<"daddiu", simm16_64, GPR64Opnd, IIArith,
                          immSExt16, add>,
              ADDI_FM<0x19>, IsAsCheapAsAMove;

let isCodeGenOnly = 1 in {
def SLTi64  : SetCC_I<"slti", setlt, simm16_64, immSExt16, GPR64Opnd>,
              SLTI_FM<0xa>;
def SLTiu64 : SetCC_I<"sltiu", setult, simm16_64, immSExt16, GPR64Opnd>,
              SLTI_FM<0xb>;
def ANDi64 : ArithLogicI<"andi", uimm16_64, GPR64Opnd, IILogic, immZExt16,
                         and>,
             ADDI_FM<0xc>;
def ORi64   : ArithLogicI<"ori", uimm16_64, GPR64Opnd, IILogic, immZExt16,
                          or>,
              ADDI_FM<0xd>;
def XORi64  : ArithLogicI<"xori", uimm16_64, GPR64Opnd, IILogic, immZExt16,
                          xor>,
              ADDI_FM<0xe>;
def LUi64   : LoadUpper<"lui", GPR64Opnd, uimm16_64>, LUI_FM;
}

/// Arithmetic Instructions (3-Operand, R-Type)
def DADD   : ArithLogicR<"dadd", GPR64Opnd>, ADD_FM<0, 0x2c>;
def DADDu  : ArithLogicR<"daddu", GPR64Opnd, 1, IIArith, add>,
                              ADD_FM<0, 0x2d>;
def DSUBu  : ArithLogicR<"dsubu", GPR64Opnd, 0, IIArith, sub>,
                              ADD_FM<0, 0x2f>;

let isCodeGenOnly = 1 in {
def SLT64  : SetCC_R<"slt", setlt, GPR64Opnd>, ADD_FM<0, 0x2a>;
def SLTu64 : SetCC_R<"sltu", setult, GPR64Opnd>, ADD_FM<0, 0x2b>;
def AND64  : ArithLogicR<"and", GPR64Opnd, 1, IIArith, and>, ADD_FM<0, 0x24>;
def OR64   : ArithLogicR<"or", GPR64Opnd, 1, IIArith, or>, ADD_FM<0, 0x25>;
def XOR64  : ArithLogicR<"xor", GPR64Opnd, 1, IIArith, xor>, ADD_FM<0, 0x26>;
def NOR64  : LogicNOR<"nor", GPR64Opnd>, ADD_FM<0, 0x27>;
}

/// Shift Instructions
def DSLL   : shift_rotate_imm<"dsll", shamt, GPR64Opnd, shl, immZExt6>,
             SRA_FM<0x38, 0>;
def DSRL   : shift_rotate_imm<"dsrl", shamt, GPR64Opnd, srl, immZExt6>,
             SRA_FM<0x3a, 0>;
def DSRA   : shift_rotate_imm<"dsra", shamt, GPR64Opnd, sra, immZExt6>,
             SRA_FM<0x3b, 0>;
def DSLLV  : shift_rotate_reg<"dsllv", GPR64Opnd, shl>, SRLV_FM<0x14, 0>;
def DSRLV  : shift_rotate_reg<"dsrlv", GPR64Opnd, srl>, SRLV_FM<0x16, 0>;
def DSRAV  : shift_rotate_reg<"dsrav", GPR64Opnd, sra>, SRLV_FM<0x17, 0>;
def DSLL32 : shift_rotate_imm<"dsll32", shamt, GPR64Opnd>, SRA_FM<0x3c, 0>;
def DSRL32 : shift_rotate_imm<"dsrl32", shamt, GPR64Opnd>, SRA_FM<0x3e, 0>;
def DSRA32 : shift_rotate_imm<"dsra32", shamt, GPR64Opnd>, SRA_FM<0x3f, 0>;

// Rotate Instructions
let Predicates = [HasMips64r2, HasStdEnc] in {
  def DROTR  : shift_rotate_imm<"drotr", shamt, GPR64Opnd, rotr, immZExt6>,
               SRA_FM<0x3a, 1>;
  def DROTRV : shift_rotate_reg<"drotrv", GPR64Opnd, rotr>,
               SRLV_FM<0x16, 1>;
}

/// Load and Store Instructions
///  aligned
let isCodeGenOnly = 1 in {
defm LB64  : LoadM<"lb", GPR64Opnd, sextloadi8, IILoad>, LW_FM<0x20>;
defm LBu64 : LoadM<"lbu", GPR64Opnd, zextloadi8, IILoad>, LW_FM<0x24>;
defm LH64  : LoadM<"lh", GPR64Opnd, sextloadi16, IILoad>, LW_FM<0x21>;
defm LHu64 : LoadM<"lhu", GPR64Opnd, zextloadi16, IILoad>, LW_FM<0x25>;
defm LW64  : LoadM<"lw", GPR64Opnd, sextloadi32, IILoad>, LW_FM<0x23>;
defm SB64  : StoreM<"sb", GPR64Opnd, truncstorei8, IIStore>, LW_FM<0x28>;
defm SH64  : StoreM<"sh", GPR64Opnd, truncstorei16, IIStore>, LW_FM<0x29>;
defm SW64  : StoreM<"sw", GPR64Opnd, truncstorei32, IIStore>, LW_FM<0x2b>;
}

defm LWu   : LoadM<"lwu", GPR64Opnd, zextloadi32, IILoad>, LW_FM<0x27>;
defm LD    : LoadM<"ld", GPR64Opnd, load, IILoad>, LW_FM<0x37>;
defm SD    : StoreM<"sd", GPR64Opnd, store, IIStore>, LW_FM<0x3f>;

/// load/store left/right
let isCodeGenOnly = 1 in {
defm LWL64 : LoadLeftRightM<"lwl", MipsLWL, GPR64Opnd>, LW_FM<0x22>;
defm LWR64 : LoadLeftRightM<"lwr", MipsLWR, GPR64Opnd>, LW_FM<0x26>;
defm SWL64 : StoreLeftRightM<"swl", MipsSWL, GPR64Opnd>, LW_FM<0x2a>;
defm SWR64 : StoreLeftRightM<"swr", MipsSWR, GPR64Opnd>, LW_FM<0x2e>;
}

defm LDL   : LoadLeftRightM<"ldl", MipsLDL, GPR64Opnd>, LW_FM<0x1a>;
defm LDR   : LoadLeftRightM<"ldr", MipsLDR, GPR64Opnd>, LW_FM<0x1b>;
defm SDL   : StoreLeftRightM<"sdl", MipsSDL, GPR64Opnd>, LW_FM<0x2c>;
defm SDR   : StoreLeftRightM<"sdr", MipsSDR, GPR64Opnd>, LW_FM<0x2d>;

/// Load-linked, Store-conditional
let Predicates = [NotN64, HasStdEnc] in {
  def LLD : LLBase<"lld", GPR64Opnd, mem>, LW_FM<0x34>;
  def SCD : SCBase<"scd", GPR64Opnd, mem>, LW_FM<0x3c>;
}

let Predicates = [IsN64, HasStdEnc], isCodeGenOnly = 1 in {
  def LLD_P8 : LLBase<"lld", GPR64Opnd, mem64>, LW_FM<0x34>;
  def SCD_P8 : SCBase<"scd", GPR64Opnd, mem64>, LW_FM<0x3c>;
}

/// Jump and Branch Instructions
let isCodeGenOnly = 1 in {
def JR64   : IndirectBranch<GPR64Opnd>, MTLO_FM<8>;
def BEQ64  : CBranch<"beq", seteq, GPR64Opnd>, BEQ_FM<4>;
def BNE64  : CBranch<"bne", setne, GPR64Opnd>, BEQ_FM<5>;
def BGEZ64 : CBranchZero<"bgez", setge, GPR64Opnd>, BGEZ_FM<1, 1>;
def BGTZ64 : CBranchZero<"bgtz", setgt, GPR64Opnd>, BGEZ_FM<7, 0>;
def BLEZ64 : CBranchZero<"blez", setle, GPR64Opnd>, BGEZ_FM<6, 0>;
def BLTZ64 : CBranchZero<"bltz", setlt, GPR64Opnd>, BGEZ_FM<1, 0>;
def JALR64 : JumpLinkReg<"jalr", GPR64Opnd>, JALR_FM;
def JALR64Pseudo : JumpLinkRegPseudo<GPR64Opnd, JALR, RA, GPR32Opnd>;
def TAILCALL64_R : JumpFR<GPR64Opnd, MipsTailCall>, MTLO_FM<8>, IsTailCall;
}

/// Multiply and Divide Instructions.
def DMULT  : Mult<"dmult", IIImult, GPR64Opnd, [HI64, LO64]>,
             MULT_FM<0, 0x1c>;
def DMULTu : Mult<"dmultu", IIImult, GPR64Opnd, [HI64, LO64]>,
             MULT_FM<0, 0x1d>;
def PseudoDMULT  : MultDivPseudo<DMULT, ACRegs128, GPR64Opnd, MipsMult,
                                 IIImult>;
def PseudoDMULTu : MultDivPseudo<DMULTu, ACRegs128, GPR64Opnd, MipsMultu,
                                 IIImult>;
def DSDIV : Div<"ddiv", IIIdiv, GPR64Opnd, [HI64, LO64]>, MULT_FM<0, 0x1e>;
def DUDIV : Div<"ddivu", IIIdiv, GPR64Opnd, [HI64, LO64]>, MULT_FM<0, 0x1f>;
def PseudoDSDIV : MultDivPseudo<DSDIV, ACRegs128, GPR64Opnd, MipsDivRem,
                                IIIdiv, 0, 1, 1>;
def PseudoDUDIV : MultDivPseudo<DUDIV, ACRegs128, GPR64Opnd, MipsDivRemU,
                                IIIdiv, 0, 1, 1>;

let isCodeGenOnly = 1 in {
def MTHI64 : MoveToLOHI<"mthi", GPR64Opnd, [HI64]>, MTLO_FM<0x11>;
def MTLO64 : MoveToLOHI<"mtlo", GPR64Opnd, [LO64]>, MTLO_FM<0x13>;
def MFHI64 : MoveFromLOHI<"mfhi", GPR64Opnd, [HI64]>, MFLO_FM<0x10>;
def MFLO64 : MoveFromLOHI<"mflo", GPR64Opnd, [LO64]>, MFLO_FM<0x12>;

/// Sign Ext In Register Instructions.
def SEB64 : SignExtInReg<"seb", i8, GPR64Opnd>, SEB_FM<0x10, 0x20>;
def SEH64 : SignExtInReg<"seh", i16, GPR64Opnd>, SEB_FM<0x18, 0x20>;
}

/// Count Leading
def DCLZ : CountLeading0<"dclz", GPR64Opnd>, CLO_FM<0x24>;
def DCLO : CountLeading1<"dclo", GPR64Opnd>, CLO_FM<0x25>;

/// Double Word Swap Bytes/HalfWords
def DSBH : SubwordSwap<"dsbh", GPR64Opnd>, SEB_FM<2, 0x24>;
def DSHD : SubwordSwap<"dshd", GPR64Opnd>, SEB_FM<5, 0x24>;

def LEA_ADDiu64 : EffectiveAddress<"daddiu", GPR64Opnd, mem_ea_64>, LW_FM<0x19>;

let isCodeGenOnly = 1 in
def RDHWR64 : ReadHardware<GPR64Opnd, HW64RegsOpnd>, RDHWR_FM;

def DEXT : ExtBase<"dext", GPR64Opnd>, EXT_FM<3>;
let Pattern = []<dag> in {
  def DEXTU : ExtBase<"dextu", GPR64Opnd>, EXT_FM<2>;
  def DEXTM : ExtBase<"dextm", GPR64Opnd>, EXT_FM<1>;
}
def DINS : InsBase<"dins", GPR64Opnd>, EXT_FM<7>;
let Pattern = []<dag> in {
  def DINSU : InsBase<"dinsu", GPR64Opnd>, EXT_FM<6>;
  def DINSM : InsBase<"dinsm", GPR64Opnd>, EXT_FM<5>;
}

let isCodeGenOnly = 1, rs = 0, shamt = 0 in {
  def DSLL64_32 : FR<0x00, 0x3c, (outs GPR64:$rd), (ins GPR32:$rt),
                     "dsll\t$rd, $rt, 32", [], IIArith>;
  def SLL64_32 : FR<0x0, 0x00, (outs GPR64:$rd), (ins GPR32:$rt),
                    "sll\t$rd, $rt, 0", [], IIArith>;
  def SLL64_64 : FR<0x0, 0x00, (outs GPR64:$rd), (ins GPR64:$rt),
                    "sll\t$rd, $rt, 0", [], IIArith>;
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

def : MipsPat<(add GPR64:$hi, (MipsLo tglobaladdr:$lo)),
              (DADDiu GPR64:$hi, tglobaladdr:$lo)>;
def : MipsPat<(add GPR64:$hi, (MipsLo tblockaddress:$lo)),
              (DADDiu GPR64:$hi, tblockaddress:$lo)>;
def : MipsPat<(add GPR64:$hi, (MipsLo tjumptable:$lo)),
              (DADDiu GPR64:$hi, tjumptable:$lo)>;
def : MipsPat<(add GPR64:$hi, (MipsLo tconstpool:$lo)),
              (DADDiu GPR64:$hi, tconstpool:$lo)>;
def : MipsPat<(add GPR64:$hi, (MipsLo tglobaltlsaddr:$lo)),
              (DADDiu GPR64:$hi, tglobaltlsaddr:$lo)>;

def : WrapperPat<tglobaladdr, DADDiu, GPR64>;
def : WrapperPat<tconstpool, DADDiu, GPR64>;
def : WrapperPat<texternalsym, DADDiu, GPR64>;
def : WrapperPat<tblockaddress, DADDiu, GPR64>;
def : WrapperPat<tjumptable, DADDiu, GPR64>;
def : WrapperPat<tglobaltlsaddr, DADDiu, GPR64>;

defm : BrcondPats<GPR64, BEQ64, BNE64, SLT64, SLTu64, SLTi64, SLTiu64,
                  ZERO_64>;

def : MipsPat<(brcond (i32 (setlt i64:$lhs, 1)), bb:$dst),
              (BLEZ64 i64:$lhs, bb:$dst)>;
def : MipsPat<(brcond (i32 (setgt i64:$lhs, -1)), bb:$dst),
              (BGEZ64 i64:$lhs, bb:$dst)>;

// setcc patterns
defm : SeteqPats<GPR64, SLTiu64, XOR64, SLTu64, ZERO_64>;
defm : SetlePats<GPR64, SLT64, SLTu64>;
defm : SetgtPats<GPR64, SLT64, SLTu64>;
defm : SetgePats<GPR64, SLT64, SLTu64>;
defm : SetgeImmPats<GPR64, SLTi64, SLTiu64>;

// truncate
def : MipsPat<(i32 (trunc GPR64:$src)),
              (SLL (EXTRACT_SUBREG GPR64:$src, sub_32), 0)>,
      Requires<[IsN64, HasStdEnc]>;

// 32-to-64-bit extension
def : MipsPat<(i64 (anyext GPR32:$src)), (SLL64_32 GPR32:$src)>;
def : MipsPat<(i64 (zext GPR32:$src)), (DSRL (DSLL64_32 GPR32:$src), 32)>;
def : MipsPat<(i64 (sext GPR32:$src)), (SLL64_32 GPR32:$src)>;

// Sign extend in register
def : MipsPat<(i64 (sext_inreg GPR64:$src, i32)),
              (SLL64_64 GPR64:$src)>;

// bswap MipsPattern
def : MipsPat<(bswap GPR64:$rt), (DSHD (DSBH GPR64:$rt))>;

// mflo/hi patterns.
def : MipsPat<(i64 (ExtractLOHI ACRegs128:$ac, imm:$lohi_idx)),
              (EXTRACT_SUBREG ACRegs128:$ac, imm:$lohi_idx)>;

//===----------------------------------------------------------------------===//
// Instruction aliases
//===----------------------------------------------------------------------===//
def : InstAlias<"move $dst, $src",
                (DADDu GPR64Opnd:$dst,  GPR64Opnd:$src, ZERO_64), 1>,
      Requires<[HasMips64]>;
def : InstAlias<"daddu $rs, $rt, $imm",
                (DADDiu GPR64Opnd:$rs, GPR64Opnd:$rt, simm16_64:$imm),
                0>;
def : InstAlias<"dadd $rs, $rt, $imm",
                (DADDi GPR64Opnd:$rs, GPR64Opnd:$rt, simm16_64:$imm),
                0>;

/// Move between CPU and coprocessor registers
let DecoderNamespace = "Mips64" in {
def DMFC0_3OP64 : MFC3OP<(outs GPR64Opnd:$rt),
                         (ins GPR64Opnd:$rd, uimm16:$sel),
                         "dmfc0\t$rt, $rd, $sel">, MFC3OP_FM<0x10, 1>;
def DMTC0_3OP64 : MFC3OP<(outs GPR64Opnd:$rd, uimm16:$sel),
                         (ins GPR64Opnd:$rt),
                         "dmtc0\t$rt, $rd, $sel">, MFC3OP_FM<0x10, 5>;
def DMFC2_3OP64 : MFC3OP<(outs GPR64Opnd:$rt),
                         (ins GPR64Opnd:$rd, uimm16:$sel),
                         "dmfc2\t$rt, $rd, $sel">, MFC3OP_FM<0x12, 1>;
def DMTC2_3OP64 : MFC3OP<(outs GPR64Opnd:$rd, uimm16:$sel),
                         (ins GPR64Opnd:$rt),
                         "dmtc2\t$rt, $rd, $sel">, MFC3OP_FM<0x12, 5>;
}

// Two operand (implicit 0 selector) versions:
def : InstAlias<"dmfc0 $rt, $rd",
                (DMFC0_3OP64 GPR64Opnd:$rt, GPR64Opnd:$rd, 0), 0>;
def : InstAlias<"dmtc0 $rt, $rd",
                (DMTC0_3OP64 GPR64Opnd:$rd, 0, GPR64Opnd:$rt), 0>;
def : InstAlias<"dmfc2 $rt, $rd",
                (DMFC2_3OP64 GPR64Opnd:$rt, GPR64Opnd:$rd, 0), 0>;
def : InstAlias<"dmtc2 $rt, $rd",
                (DMTC2_3OP64 GPR64Opnd:$rd, 0, GPR64Opnd:$rt), 0>;

