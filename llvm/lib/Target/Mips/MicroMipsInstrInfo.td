def addrimm12 : ComplexPattern<iPTR, 2, "selectIntAddrMM", [frameindex]>;

def simm12 : Operand<i32> {
  let DecoderMethod = "DecodeSimm12";
}

def mem_mm_12 : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops GPR32, simm12);
  let EncoderMethod = "getMemEncodingMMImm12";
  let ParserMatchClass = MipsMemAsmOperand;
  let OperandType = "OPERAND_MEMORY";
}

def jmptarget_mm : Operand<OtherVT> {
  let EncoderMethod = "getJumpTargetOpValueMM";
}

def calltarget_mm : Operand<iPTR> {
  let EncoderMethod = "getJumpTargetOpValueMM";
}

def brtarget_mm : Operand<OtherVT> {
  let EncoderMethod = "getBranchTargetOpValueMM";
  let OperandType   = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTargetMM";
}

let canFoldAsLoad = 1 in
class LoadLeftRightMM<string opstr, SDNode OpNode, RegisterOperand RO,
                      Operand MemOpnd> :
  InstSE<(outs RO:$rt), (ins MemOpnd:$addr, RO:$src),
         !strconcat(opstr, "\t$rt, $addr"),
         [(set RO:$rt, (OpNode addrimm12:$addr, RO:$src))],
         NoItinerary, FrmI> {
  let DecoderMethod = "DecodeMemMMImm12";
  string Constraints = "$src = $rt";
}

class StoreLeftRightMM<string opstr, SDNode OpNode, RegisterOperand RO,
                       Operand MemOpnd>:
  InstSE<(outs), (ins RO:$rt, MemOpnd:$addr),
         !strconcat(opstr, "\t$rt, $addr"),
         [(OpNode RO:$rt, addrimm12:$addr)], NoItinerary, FrmI> {
  let DecoderMethod = "DecodeMemMMImm12";
}

class LLBaseMM<string opstr, RegisterOperand RO> :
  InstSE<(outs RO:$rt), (ins mem_mm_12:$addr),
         !strconcat(opstr, "\t$rt, $addr"), [], NoItinerary, FrmI> {
  let DecoderMethod = "DecodeMemMMImm12";
  let mayLoad = 1;
}

class SCBaseMM<string opstr, RegisterOperand RO> :
  InstSE<(outs RO:$dst), (ins RO:$rt, mem_mm_12:$addr),
         !strconcat(opstr, "\t$rt, $addr"), [], NoItinerary, FrmI> {
  let DecoderMethod = "DecodeMemMMImm12";
  let mayStore = 1;
  let Constraints = "$rt = $dst";
}

class LoadMM<string opstr, DAGOperand RO, SDPatternOperator OpNode = null_frag,
             InstrItinClass Itin = NoItinerary> :
  InstSE<(outs RO:$rt), (ins mem_mm_12:$addr),
         !strconcat(opstr, "\t$rt, $addr"),
         [(set RO:$rt, (OpNode addrimm12:$addr))], Itin, FrmI> {
  let DecoderMethod = "DecodeMemMMImm12";
  let canFoldAsLoad = 1;
  let mayLoad = 1;
}

class MoveFromHILOMM<string opstr, RegisterOperand RO, Register UseReg> :
      MicroMipsInst16<(outs RO:$rd), (ins), !strconcat(opstr, "\t$rd"),
  [], II_MFHI_MFLO, FrmR> {
  let Uses = [UseReg];
  let hasSideEffects = 0;
}

class MoveMM16<string opstr, RegisterOperand RO, bit isComm = 0,
               InstrItinClass Itin = NoItinerary> :
  MicroMipsInst16<(outs RO:$rd), (ins RO:$rs),
                  !strconcat(opstr, "\t$rd, $rs"), [], Itin, FrmR> {
  let isCommutable = isComm;
  let isReMaterializable = 1;
}

// 16-bit Jump and Link (Call)
class JumpLinkRegMM16<string opstr, RegisterOperand RO> :
  MicroMipsInst16<(outs), (ins RO:$rs), !strconcat(opstr, "\t$rs"),
           [(MipsJmpLink RO:$rs)], IIBranch, FrmR> {
  let isCall = 1;
  let hasDelaySlot = 1;
  let Defs = [RA];
}

def MFHI16_MM : MoveFromHILOMM<"mfhi", GPR32Opnd, AC0>, MFHILO_FM_MM16<0x10>;
def MFLO16_MM : MoveFromHILOMM<"mflo", GPR32Opnd, AC0>, MFHILO_FM_MM16<0x12>;
def MOVE16_MM : MoveMM16<"move", GPR32Opnd>, MOVE_FM_MM16<0x03>;
def JALR16_MM : JumpLinkRegMM16<"jalr", GPR32Opnd>, JALR_FM_MM16<0x0e>;

class WaitMM<string opstr> :
  InstSE<(outs), (ins uimm10:$code_), !strconcat(opstr, "\t$code_"), [],
         NoItinerary, FrmOther, opstr>;

let DecoderNamespace = "MicroMips", Predicates = [InMicroMips] in {
  /// Arithmetic Instructions (ALU Immediate)
  def ADDiu_MM : MMRel, ArithLogicI<"addiu", simm16, GPR32Opnd>,
                 ADDI_FM_MM<0xc>;
  def ADDi_MM  : MMRel, ArithLogicI<"addi", simm16, GPR32Opnd>,
                 ADDI_FM_MM<0x4>;
  def SLTi_MM  : MMRel, SetCC_I<"slti", setlt, simm16, immSExt16, GPR32Opnd>,
                 SLTI_FM_MM<0x24>;
  def SLTiu_MM : MMRel, SetCC_I<"sltiu", setult, simm16, immSExt16, GPR32Opnd>,
                 SLTI_FM_MM<0x2c>;
  def ANDi_MM  : MMRel, ArithLogicI<"andi", uimm16, GPR32Opnd>,
                 ADDI_FM_MM<0x34>;
  def ORi_MM   : MMRel, ArithLogicI<"ori", uimm16, GPR32Opnd>,
                 ADDI_FM_MM<0x14>;
  def XORi_MM  : MMRel, ArithLogicI<"xori", uimm16, GPR32Opnd>,
                 ADDI_FM_MM<0x1c>;
  def LUi_MM   : MMRel, LoadUpper<"lui", GPR32Opnd, uimm16>, LUI_FM_MM;

  def LEA_ADDiu_MM : MMRel, EffectiveAddress<"addiu", GPR32Opnd>,
                     LW_FM_MM<0xc>;

  /// Arithmetic Instructions (3-Operand, R-Type)
  def ADDu_MM  : MMRel, ArithLogicR<"addu", GPR32Opnd>, ADD_FM_MM<0, 0x150>;
  def SUBu_MM  : MMRel, ArithLogicR<"subu", GPR32Opnd>, ADD_FM_MM<0, 0x1d0>;
  def MUL_MM   : MMRel, ArithLogicR<"mul", GPR32Opnd>, ADD_FM_MM<0, 0x210>;
  def ADD_MM   : MMRel, ArithLogicR<"add", GPR32Opnd>, ADD_FM_MM<0, 0x110>;
  def SUB_MM   : MMRel, ArithLogicR<"sub", GPR32Opnd>, ADD_FM_MM<0, 0x190>;
  def SLT_MM   : MMRel, SetCC_R<"slt", setlt, GPR32Opnd>, ADD_FM_MM<0, 0x350>;
  def SLTu_MM  : MMRel, SetCC_R<"sltu", setult, GPR32Opnd>,
                 ADD_FM_MM<0, 0x390>;
  def AND_MM   : MMRel, ArithLogicR<"and", GPR32Opnd, 1, II_AND, and>,
                 ADD_FM_MM<0, 0x250>;
  def OR_MM    : MMRel, ArithLogicR<"or", GPR32Opnd, 1, II_OR, or>,
                 ADD_FM_MM<0, 0x290>;
  def XOR_MM   : MMRel, ArithLogicR<"xor", GPR32Opnd, 1, II_XOR, xor>,
                 ADD_FM_MM<0, 0x310>;
  def NOR_MM   : MMRel, LogicNOR<"nor", GPR32Opnd>, ADD_FM_MM<0, 0x2d0>;
  def MULT_MM  : MMRel, Mult<"mult", II_MULT, GPR32Opnd, [HI0, LO0]>,
                 MULT_FM_MM<0x22c>;
  def MULTu_MM : MMRel, Mult<"multu", II_MULTU, GPR32Opnd, [HI0, LO0]>,
                 MULT_FM_MM<0x26c>;
  def SDIV_MM  : MMRel, Div<"div", II_DIV, GPR32Opnd, [HI0, LO0]>,
                 MULT_FM_MM<0x2ac>;
  def UDIV_MM  : MMRel, Div<"divu", II_DIVU, GPR32Opnd, [HI0, LO0]>,
                 MULT_FM_MM<0x2ec>;

  /// Shift Instructions
  def SLL_MM   : MMRel, shift_rotate_imm<"sll", uimm5, GPR32Opnd, II_SLL>,
                 SRA_FM_MM<0, 0>;
  def SRL_MM   : MMRel, shift_rotate_imm<"srl", uimm5, GPR32Opnd, II_SRL>,
                 SRA_FM_MM<0x40, 0>;
  def SRA_MM   : MMRel, shift_rotate_imm<"sra", uimm5, GPR32Opnd, II_SRA>,
                 SRA_FM_MM<0x80, 0>;
  def SLLV_MM  : MMRel, shift_rotate_reg<"sllv", GPR32Opnd, II_SLLV>,
                 SRLV_FM_MM<0x10, 0>;
  def SRLV_MM  : MMRel, shift_rotate_reg<"srlv", GPR32Opnd, II_SRLV>,
                 SRLV_FM_MM<0x50, 0>;
  def SRAV_MM  : MMRel, shift_rotate_reg<"srav", GPR32Opnd, II_SRAV>,
                 SRLV_FM_MM<0x90, 0>;
  def ROTR_MM  : MMRel, shift_rotate_imm<"rotr", uimm5, GPR32Opnd, II_ROTR>,
                 SRA_FM_MM<0xc0, 0>;
  def ROTRV_MM : MMRel, shift_rotate_reg<"rotrv", GPR32Opnd, II_ROTRV>,
                 SRLV_FM_MM<0xd0, 0>;

  /// Load and Store Instructions - aligned
  let DecoderMethod = "DecodeMemMMImm16" in {
    def LB_MM  : Load<"lb", GPR32Opnd>, MMRel, LW_FM_MM<0x7>;
    def LBu_MM : Load<"lbu", GPR32Opnd>, MMRel, LW_FM_MM<0x5>;
    def LH_MM  : Load<"lh", GPR32Opnd>, MMRel, LW_FM_MM<0xf>;
    def LHu_MM : Load<"lhu", GPR32Opnd>, MMRel, LW_FM_MM<0xd>;
    def LW_MM  : Load<"lw", GPR32Opnd>, MMRel, LW_FM_MM<0x3f>;
    def SB_MM  : Store<"sb", GPR32Opnd>, MMRel, LW_FM_MM<0x6>;
    def SH_MM  : Store<"sh", GPR32Opnd>, MMRel, LW_FM_MM<0xe>;
    def SW_MM  : Store<"sw", GPR32Opnd>, MMRel, LW_FM_MM<0x3e>;
  }

  def LWU_MM : LoadMM<"lwu", GPR32Opnd, zextloadi32, II_LWU>, LL_FM_MM<0xe>;

  /// Load and Store Instructions - unaligned
  def LWL_MM : LoadLeftRightMM<"lwl", MipsLWL, GPR32Opnd, mem_mm_12>,
               LWL_FM_MM<0x0>;
  def LWR_MM : LoadLeftRightMM<"lwr", MipsLWR, GPR32Opnd, mem_mm_12>,
               LWL_FM_MM<0x1>;
  def SWL_MM : StoreLeftRightMM<"swl", MipsSWL, GPR32Opnd, mem_mm_12>,
               LWL_FM_MM<0x8>;
  def SWR_MM : StoreLeftRightMM<"swr", MipsSWR, GPR32Opnd, mem_mm_12>,
               LWL_FM_MM<0x9>;

  /// Move Conditional
  def MOVZ_I_MM : MMRel, CMov_I_I_FT<"movz", GPR32Opnd, GPR32Opnd,
                  NoItinerary>, ADD_FM_MM<0, 0x58>;
  def MOVN_I_MM : MMRel, CMov_I_I_FT<"movn", GPR32Opnd, GPR32Opnd,
                  NoItinerary>, ADD_FM_MM<0, 0x18>;
  def MOVT_I_MM : MMRel, CMov_F_I_FT<"movt", GPR32Opnd, II_MOVT>,
                  CMov_F_I_FM_MM<0x25>;
  def MOVF_I_MM : MMRel, CMov_F_I_FT<"movf", GPR32Opnd, II_MOVF>,
                  CMov_F_I_FM_MM<0x5>;

  /// Move to/from HI/LO
  def MTHI_MM : MMRel, MoveToLOHI<"mthi", GPR32Opnd, [HI0]>,
                MTLO_FM_MM<0x0b5>;
  def MTLO_MM : MMRel, MoveToLOHI<"mtlo", GPR32Opnd, [LO0]>,
                MTLO_FM_MM<0x0f5>;
  def MFHI_MM : MMRel, MoveFromLOHI<"mfhi", GPR32Opnd, AC0>,
                MFLO_FM_MM<0x035>;
  def MFLO_MM : MMRel, MoveFromLOHI<"mflo", GPR32Opnd, AC0>,
                MFLO_FM_MM<0x075>;

  /// Multiply Add/Sub Instructions
  def MADD_MM  : MMRel, MArithR<"madd", II_MADD, 1>, MULT_FM_MM<0x32c>;
  def MADDU_MM : MMRel, MArithR<"maddu", II_MADDU, 1>, MULT_FM_MM<0x36c>;
  def MSUB_MM  : MMRel, MArithR<"msub", II_MSUB>, MULT_FM_MM<0x3ac>;
  def MSUBU_MM : MMRel, MArithR<"msubu", II_MSUBU>, MULT_FM_MM<0x3ec>;

  /// Count Leading
  def CLZ_MM : MMRel, CountLeading0<"clz", GPR32Opnd>, CLO_FM_MM<0x16c>;
  def CLO_MM : MMRel, CountLeading1<"clo", GPR32Opnd>, CLO_FM_MM<0x12c>;

  /// Sign Ext In Register Instructions.
  def SEB_MM : MMRel, SignExtInReg<"seb", i8, GPR32Opnd, II_SEB>, SEB_FM_MM<0x0ac>;
  def SEH_MM : MMRel, SignExtInReg<"seh", i16, GPR32Opnd, II_SEH>, SEB_FM_MM<0x0ec>;

  /// Word Swap Bytes Within Halfwords
  def WSBH_MM : MMRel, SubwordSwap<"wsbh", GPR32Opnd>, SEB_FM_MM<0x1ec>;

  def EXT_MM : MMRel, ExtBase<"ext", GPR32Opnd, uimm5, MipsExt>,
               EXT_FM_MM<0x2c>;
  def INS_MM : MMRel, InsBase<"ins", GPR32Opnd, uimm5, MipsIns>,
               EXT_FM_MM<0x0c>;

  /// Jump Instructions
  let DecoderMethod = "DecodeJumpTargetMM" in {
    def J_MM        : MMRel, JumpFJ<jmptarget_mm, "j", br, bb, "j">,
                      J_FM_MM<0x35>;
    def JAL_MM      : MMRel, JumpLink<"jal", calltarget_mm>, J_FM_MM<0x3d>;
  }
  def JR_MM   : MMRel, IndirectBranch<"jr", GPR32Opnd>, JR_FM_MM<0x3c>;
  def JALR_MM : JumpLinkReg<"jalr", GPR32Opnd>, JALR_FM_MM<0x03c>;
  def RET_MM : MMRel, RetBase<"ret", GPR32Opnd>, JR_FM_MM<0x3c>;

  /// Branch Instructions
  def BEQ_MM  : MMRel, CBranch<"beq", brtarget_mm, seteq, GPR32Opnd>,
                BEQ_FM_MM<0x25>;
  def BNE_MM  : MMRel, CBranch<"bne", brtarget_mm, setne, GPR32Opnd>,
                BEQ_FM_MM<0x2d>;
  def BGEZ_MM : MMRel, CBranchZero<"bgez", brtarget_mm, setge, GPR32Opnd>,
                BGEZ_FM_MM<0x2>;
  def BGTZ_MM : MMRel, CBranchZero<"bgtz", brtarget_mm, setgt, GPR32Opnd>,
                BGEZ_FM_MM<0x6>;
  def BLEZ_MM : MMRel, CBranchZero<"blez", brtarget_mm, setle, GPR32Opnd>,
                BGEZ_FM_MM<0x4>;
  def BLTZ_MM : MMRel, CBranchZero<"bltz", brtarget_mm, setlt, GPR32Opnd>,
                BGEZ_FM_MM<0x0>;
  def BGEZAL_MM : MMRel, BGEZAL_FT<"bgezal", brtarget_mm, GPR32Opnd>,
                  BGEZAL_FM_MM<0x03>;
  def BLTZAL_MM : MMRel, BGEZAL_FT<"bltzal", brtarget_mm, GPR32Opnd>,
                  BGEZAL_FM_MM<0x01>;

  /// Control Instructions
  def SYNC_MM    : MMRel, SYNC_FT<"sync">, SYNC_FM_MM;
  def BREAK_MM   : MMRel, BRK_FT<"break">, BRK_FM_MM;
  def SYSCALL_MM : MMRel, SYS_FT<"syscall">, SYS_FM_MM;
  def WAIT_MM    : WaitMM<"wait">, WAIT_FM_MM;
  def ERET_MM    : MMRel, ER_FT<"eret">, ER_FM_MM<0x3cd>;
  def DERET_MM   : MMRel, ER_FT<"deret">, ER_FM_MM<0x38d>;
  def EI_MM      : MMRel, DEI_FT<"ei", GPR32Opnd>, EI_FM_MM<0x15d>;
  def DI_MM      : MMRel, DEI_FT<"di", GPR32Opnd>, EI_FM_MM<0x11d>;

  /// Trap Instructions
  def TEQ_MM  : MMRel, TEQ_FT<"teq", GPR32Opnd>, TEQ_FM_MM<0x0>;
  def TGE_MM  : MMRel, TEQ_FT<"tge", GPR32Opnd>, TEQ_FM_MM<0x08>;
  def TGEU_MM : MMRel, TEQ_FT<"tgeu", GPR32Opnd>, TEQ_FM_MM<0x10>;
  def TLT_MM  : MMRel, TEQ_FT<"tlt", GPR32Opnd>, TEQ_FM_MM<0x20>;
  def TLTU_MM : MMRel, TEQ_FT<"tltu", GPR32Opnd>, TEQ_FM_MM<0x28>;
  def TNE_MM  : MMRel, TEQ_FT<"tne", GPR32Opnd>, TEQ_FM_MM<0x30>;

  def TEQI_MM  : MMRel, TEQI_FT<"teqi", GPR32Opnd>, TEQI_FM_MM<0x0e>;
  def TGEI_MM  : MMRel, TEQI_FT<"tgei", GPR32Opnd>, TEQI_FM_MM<0x09>;
  def TGEIU_MM : MMRel, TEQI_FT<"tgeiu", GPR32Opnd>, TEQI_FM_MM<0x0b>;
  def TLTI_MM  : MMRel, TEQI_FT<"tlti", GPR32Opnd>, TEQI_FM_MM<0x08>;
  def TLTIU_MM : MMRel, TEQI_FT<"tltiu", GPR32Opnd>, TEQI_FM_MM<0x0a>;
  def TNEI_MM  : MMRel, TEQI_FT<"tnei", GPR32Opnd>, TEQI_FM_MM<0x0c>;

  /// Load-linked, Store-conditional
  def LL_MM : LLBaseMM<"ll", GPR32Opnd>, LL_FM_MM<0x3>;
  def SC_MM : SCBaseMM<"sc", GPR32Opnd>, LL_FM_MM<0xb>;
}

//===----------------------------------------------------------------------===//
// MicroMips instruction aliases
//===----------------------------------------------------------------------===//

let Predicates = [InMicroMips] in {
  def : MipsInstAlias<"wait", (WAIT_MM 0x0), 1>;
}
