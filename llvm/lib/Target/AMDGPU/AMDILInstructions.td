//===-- AMDILInstructions.td - AMDIL Instruction definitions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//

let Predicates = [Has32BitPtr] in {
  let isCodeGenOnly=1 in {
    //===----------------------------------------------------------------------===//
    // Store Memory Operations
    //===----------------------------------------------------------------------===//
    defm GLOBALTRUNCSTORE   : GTRUNCSTORE<"!global trunc store">;
    defm LOCALTRUNCSTORE    : LTRUNCSTORE<"!local trunc store">;
    defm LOCALSTORE         : STORE<"!local store"          , local_store>;
    defm PRIVATETRUNCSTORE  : PTRUNCSTORE<"!private trunc store">;
    defm PRIVATESTORE       : STORE<"!private store"        , private_store>;
    defm REGIONTRUNCSTORE   : RTRUNCSTORE<"!region trunc store">;
    defm REGIONSTORE        : STORE<"!region hw store"      , region_store>;


    //===---------------------------------------------------------------------===//
    // Load Memory Operations
    //===---------------------------------------------------------------------===//
    defm GLOBALZEXTLOAD     : LOAD<"!global zext load"       , global_zext_load>;
    defm GLOBALSEXTLOAD     : LOAD<"!global sext load"       , global_sext_load>;
    defm GLOBALAEXTLOAD     : LOAD<"!global aext load"       , global_aext_load>;
    defm PRIVATELOAD        : LOAD<"!private load"           , private_load>;
    defm PRIVATEZEXTLOAD    : LOAD<"!private zext load"      , private_zext_load>;
    defm PRIVATESEXTLOAD    : LOAD<"!private sext load"      , private_sext_load>;
    defm PRIVATEAEXTLOAD    : LOAD<"!private aext load"      , private_aext_load>;
    defm CPOOLLOAD          : LOAD<"!constant pool load"     , cp_load>;
    defm CPOOLZEXTLOAD      : LOAD<"!constant pool zext load", cp_zext_load>;
    defm CPOOLSEXTLOAD      : LOAD<"!constant pool sext load", cp_sext_load>;
    defm CPOOLAEXTLOAD      : LOAD<"!constant aext pool load", cp_aext_load>;
    defm CONSTANTLOAD       : LOAD<"!constant load"          , constant_load>;
    defm CONSTANTZEXTLOAD   : LOAD<"!constant zext load"     , constant_zext_load>;
    defm CONSTANTSEXTLOAD   : LOAD<"!constant sext load"     , constant_sext_load>;
    defm CONSTANTAEXTLOAD   : LOAD<"!constant aext load"     , constant_aext_load>;
    defm LOCALLOAD          : LOAD<"!local load"             , local_load>;
    defm LOCALZEXTLOAD      : LOAD<"!local zext load"        , local_zext_load>;
    defm LOCALSEXTLOAD      : LOAD<"!local sext load"        , local_sext_load>;
    defm LOCALAEXTLOAD      : LOAD<"!local aext load"        , local_aext_load>;
    defm REGIONLOAD         : LOAD<"!region load"            , region_load>;
    defm REGIONZEXTLOAD     : LOAD<"!region zext load"       , region_zext_load>;
    defm REGIONSEXTLOAD     : LOAD<"!region sext load"       , region_sext_load>;
    defm REGIONAEXTLOAD     : LOAD<"!region aext load"       , region_aext_load>;
  }
}

//===---------------------------------------------------------------------===//
// Custom Inserter for Branches and returns, this eventually will be a
// seperate pass
//===---------------------------------------------------------------------===//
let isTerminator = 1 in {
  def BRANCH : ILFormat<IL_PSEUDO_INST, (outs), (ins brtarget:$target),
      "; Pseudo unconditional branch instruction",
      [(br bb:$target)]>;
  defm BRANCH_COND : BranchConditional<IL_brcond>;
}
//===---------------------------------------------------------------------===//
// return instructions
//===---------------------------------------------------------------------===//
let isTerminator = 1, isReturn = 1, isBarrier = 1, hasCtrlDep = 1 in {
  def RETURN          : ILFormat<IL_OP_RET,(outs), (ins variable_ops),
      IL_OP_RET.Text, [(IL_retflag)]>;
}

//===---------------------------------------------------------------------===//
// Handle a function call
//===---------------------------------------------------------------------===//
let isCall = 1,
    Defs = [
    R1, R2, R3, R4, R5, R6, R7, R8, R9, R10
    ]
    ,
    Uses = [
    R11, R12, R13, R14, R15, R16, R17, R18, R19, R20
    ]
    in {
      def CALL : UnaryOpNoRet<IL_OP_CALL, (outs),
      (ins calltarget:$dst), 
      !strconcat(IL_OP_CALL.Text, " $dst"), []>;
    }


//===---------------------------------------------------------------------===//
// Flow and Program control Instructions
//===---------------------------------------------------------------------===//
let isTerminator=1 in {
  def SWITCH      : ILFormat<IL_OP_SWITCH, (outs), (ins GPRI32:$src),
  !strconcat(IL_OP_SWITCH.Text, " $src"), []>;
  def CASE        : ILFormat<IL_OP_CASE, (outs), (ins GPRI32:$src),
      !strconcat(IL_OP_CASE.Text, " $src"), []>;
  def BREAK       : ILFormat<IL_OP_BREAK, (outs), (ins),
      IL_OP_BREAK.Text, []>;
  def CONTINUE    : ILFormat<IL_OP_CONTINUE, (outs), (ins),
      IL_OP_CONTINUE.Text, []>;
  def DEFAULT     : ILFormat<IL_OP_DEFAULT, (outs), (ins),
      IL_OP_DEFAULT.Text, []>;
  def ELSE        : ILFormat<IL_OP_ELSE, (outs), (ins),
      IL_OP_ELSE.Text, []>;
  def ENDSWITCH   : ILFormat<IL_OP_ENDSWITCH, (outs), (ins),
      IL_OP_ENDSWITCH.Text, []>;
  def ENDMAIN     : ILFormat<IL_OP_ENDMAIN, (outs), (ins),
      IL_OP_ENDMAIN.Text, []>;
  def END         : ILFormat<IL_OP_END, (outs), (ins),
      IL_OP_END.Text, []>;
  def ENDFUNC     : ILFormat<IL_OP_ENDFUNC, (outs), (ins),
      IL_OP_ENDFUNC.Text, []>;
  def ENDIF       : ILFormat<IL_OP_ENDIF, (outs), (ins),
      IL_OP_ENDIF.Text, []>;
  def WHILELOOP   : ILFormat<IL_OP_WHILE, (outs), (ins),
      IL_OP_WHILE.Text, []>;
  def ENDLOOP     : ILFormat<IL_OP_ENDLOOP, (outs), (ins),
      IL_OP_ENDLOOP.Text, []>;
  def FUNC        : ILFormat<IL_OP_FUNC, (outs), (ins),
      IL_OP_FUNC.Text, []>;
  def RETDYN      : ILFormat<IL_OP_RET_DYN, (outs), (ins),
      IL_OP_RET_DYN.Text, []>;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm IF_LOGICALNZ  : BranchInstr<IL_OP_IF_LOGICALNZ>;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm IF_LOGICALZ   : BranchInstr<IL_OP_IF_LOGICALZ>;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm BREAK_LOGICALNZ : BranchInstr<IL_OP_BREAK_LOGICALNZ>;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm BREAK_LOGICALZ : BranchInstr<IL_OP_BREAK_LOGICALZ>;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm CONTINUE_LOGICALNZ : BranchInstr<IL_OP_CONTINUE_LOGICALNZ>;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm CONTINUE_LOGICALZ : BranchInstr<IL_OP_CONTINUE_LOGICALZ>;
  defm IFC         : BranchInstr2<IL_OP_IFC>;
  defm BREAKC      : BranchInstr2<IL_OP_BREAKC>;
  defm CONTINUEC   : BranchInstr2<IL_OP_CONTINUEC>;
}
let isTerminator = 1, isBarrier = 1, hasCtrlDep = 1 in {
  def TRAP : ILFormat<IL_OP_NOP, (outs), (ins),
      IL_OP_NOP.Text, [(trap)]>;
}

