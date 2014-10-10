//===-- SIInstrInfo.td - SI Instruction Infos -------------*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class vop {
  field bits<9> SI3;
}

class vopc <bits<8> si> : vop {
  field bits<8> SI = si;

  field bits<9> SI3 = {0, si{7-0}};
}

class vop1 <bits<8> si> : vop {
  field bits<8> SI  = si;

  field bits<9> SI3 = {1, 1, si{6-0}};
}

class vop2 <bits<6> si> : vop {
  field bits<6> SI = si;

  field bits<9> SI3 = {1, 0, 0, si{5-0}};
}

class vop3 <bits<9> si> : vop {
  field bits<9> SI3 = si;
}

// Execpt for the NONE field, this must be kept in sync with the SISubtarget enum
// in AMDGPUMCInstLower.h
def SISubtarget {
  int NONE = -1;
  int SI = 0;
}

//===----------------------------------------------------------------------===//
// SI DAG Nodes
//===----------------------------------------------------------------------===//

def SIload_constant : SDNode<"AMDGPUISD::LOAD_CONSTANT",
  SDTypeProfile<1, 2, [SDTCisVT<0, f32>, SDTCisVT<1, v4i32>, SDTCisVT<2, i32>]>,
                      [SDNPMayLoad, SDNPMemOperand]
>;

def SItbuffer_store : SDNode<"AMDGPUISD::TBUFFER_STORE_FORMAT",
  SDTypeProfile<0, 13,
    [SDTCisVT<0, v4i32>,   // rsrc(SGPR)
     SDTCisVT<1, iAny>,   // vdata(VGPR)
     SDTCisVT<2, i32>,    // num_channels(imm)
     SDTCisVT<3, i32>,    // vaddr(VGPR)
     SDTCisVT<4, i32>,    // soffset(SGPR)
     SDTCisVT<5, i32>,    // inst_offset(imm)
     SDTCisVT<6, i32>,    // dfmt(imm)
     SDTCisVT<7, i32>,    // nfmt(imm)
     SDTCisVT<8, i32>,    // offen(imm)
     SDTCisVT<9, i32>,    // idxen(imm)
     SDTCisVT<10, i32>,   // glc(imm)
     SDTCisVT<11, i32>,   // slc(imm)
     SDTCisVT<12, i32>    // tfe(imm)
    ]>,
  [SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def SIload_input : SDNode<"AMDGPUISD::LOAD_INPUT",
  SDTypeProfile<1, 3, [SDTCisVT<0, v4f32>, SDTCisVT<1, v4i32>, SDTCisVT<2, i16>,
                       SDTCisVT<3, i32>]>
>;

class SDSample<string opcode> : SDNode <opcode,
  SDTypeProfile<1, 4, [SDTCisVT<0, v4f32>, SDTCisVT<2, v32i8>,
                       SDTCisVT<3, v4i32>, SDTCisVT<4, i32>]>
>;

def SIsample : SDSample<"AMDGPUISD::SAMPLE">;
def SIsampleb : SDSample<"AMDGPUISD::SAMPLEB">;
def SIsampled : SDSample<"AMDGPUISD::SAMPLED">;
def SIsamplel : SDSample<"AMDGPUISD::SAMPLEL">;

def SIconstdata_ptr : SDNode<
  "AMDGPUISD::CONST_DATA_PTR", SDTypeProfile <1, 0, [SDTCisVT<0, i64>]>
>;

// Transformation function, extract the lower 32bit of a 64bit immediate
def LO32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 0xffffffff, MVT::i32);
}]>;

def LO32f : SDNodeXForm<fpimm, [{
  APInt V = N->getValueAPF().bitcastToAPInt().trunc(32);
  return CurDAG->getTargetConstantFP(APFloat(APFloat::IEEEsingle, V), MVT::f32);
}]>;

// Transformation function, extract the upper 32bit of a 64bit immediate
def HI32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() >> 32, MVT::i32);
}]>;

def HI32f : SDNodeXForm<fpimm, [{
  APInt V = N->getValueAPF().bitcastToAPInt().lshr(32).trunc(32);
  return CurDAG->getTargetConstantFP(APFloat(APFloat::IEEEsingle, V), MVT::f32);
}]>;

def IMM8bitDWORD : PatLeaf <(imm),
  [{return (N->getZExtValue() & ~0x3FC) == 0;}]
>;

def as_dword_i32imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() >> 2, MVT::i32);
}]>;

def as_i1imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), MVT::i1);
}]>;

def as_i8imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), MVT::i8);
}]>;

def as_i16imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), MVT::i16);
}]>;

def as_i32imm: SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), MVT::i32);
}]>;

def IMM8bit : PatLeaf <(imm),
  [{return isUInt<8>(N->getZExtValue());}]
>;

def IMM12bit : PatLeaf <(imm),
  [{return isUInt<12>(N->getZExtValue());}]
>;

def IMM16bit : PatLeaf <(imm),
  [{return isUInt<16>(N->getZExtValue());}]
>;

def IMM32bit : PatLeaf <(imm),
  [{return isUInt<32>(N->getZExtValue());}]
>;

def mubuf_vaddr_offset : PatFrag<
  (ops node:$ptr, node:$offset, node:$imm_offset),
  (add (add node:$ptr, node:$offset), node:$imm_offset)
>;

class InlineImm <ValueType vt> : PatLeaf <(vt imm), [{
  return isInlineImmediate(N);
}]>;

class SGPRImm <dag frag> : PatLeaf<frag, [{
  if (TM.getSubtarget<AMDGPUSubtarget>().getGeneration() <
      AMDGPUSubtarget::SOUTHERN_ISLANDS) {
    return false;
  }
  const SIRegisterInfo *SIRI =
                       static_cast<const SIRegisterInfo*>(TM.getSubtargetImpl()->getRegisterInfo());
  for (SDNode::use_iterator U = N->use_begin(), E = SDNode::use_end();
                                                U != E; ++U) {
    if (SIRI->isSGPRClass(getOperandRegClass(*U, U.getOperandNo()))) {
      return true;
    }
  }
  return false;
}]>;

//===----------------------------------------------------------------------===//
// Custom Operands
//===----------------------------------------------------------------------===//

def FRAMEri32 : Operand<iPTR> {
  let MIOperandInfo = (ops i32:$ptr, i32imm:$index);
}

def sopp_brtarget : Operand<OtherVT> {
  let EncoderMethod = "getSOPPBrEncoding";
  let OperandType = "OPERAND_PCREL";
}

include "SIInstrFormats.td"

let OperandType = "OPERAND_IMMEDIATE" in {

def offen : Operand<i1> {
  let PrintMethod = "printOffen";
}
def idxen : Operand<i1> {
  let PrintMethod = "printIdxen";
}
def addr64 : Operand<i1> {
  let PrintMethod = "printAddr64";
}
def mbuf_offset : Operand<i16> {
  let PrintMethod = "printMBUFOffset";
}
def ds_offset : Operand<i16> {
  let PrintMethod = "printDSOffset";
}
def ds_offset0 : Operand<i8> {
  let PrintMethod = "printDSOffset0";
}
def ds_offset1 : Operand<i8> {
  let PrintMethod = "printDSOffset1";
}
def glc : Operand <i1> {
  let PrintMethod = "printGLC";
}
def slc : Operand <i1> {
  let PrintMethod = "printSLC";
}
def tfe : Operand <i1> {
  let PrintMethod = "printTFE";
}

def omod : Operand <i32> {
  let PrintMethod = "printOModSI";
}

def ClampMod : Operand <i1> {
  let PrintMethod = "printClampSI";
}

} // End OperandType = "OPERAND_IMMEDIATE"

//===----------------------------------------------------------------------===//
// Complex patterns
//===----------------------------------------------------------------------===//

def DS1Addr1Offset : ComplexPattern<i32, 2, "SelectDS1Addr1Offset">;
def DS64Bit4ByteAligned : ComplexPattern<i32, 3, "SelectDS64Bit4ByteAligned">;

def MUBUFAddr32 : ComplexPattern<i64, 9, "SelectMUBUFAddr32">;
def MUBUFAddr64 : ComplexPattern<i64, 3, "SelectMUBUFAddr64">;
def MUBUFAddr64Atomic : ComplexPattern<i64, 4, "SelectMUBUFAddr64">;
def MUBUFScratch : ComplexPattern<i64, 4, "SelectMUBUFScratch">;
def MUBUFOffset : ComplexPattern<i64, 6, "SelectMUBUFOffset">;
def MUBUFOffsetAtomic : ComplexPattern<i64, 4, "SelectMUBUFOffset">;

def VOP3Mods0 : ComplexPattern<untyped, 4, "SelectVOP3Mods0">;
def VOP3Mods  : ComplexPattern<untyped, 2, "SelectVOP3Mods">;

//===----------------------------------------------------------------------===//
// SI assembler operands
//===----------------------------------------------------------------------===//

def SIOperand {
  int ZERO = 0x80;
  int VCC = 0x6A;
  int FLAT_SCR = 0x68;
}

def SRCMODS {
  int NONE = 0;
}

def DSTCLAMP {
  int NONE = 0;
}

def DSTOMOD {
  int NONE = 0;
}

//===----------------------------------------------------------------------===//
//
// SI Instruction multiclass helpers.
//
// Instructions with _32 take 32-bit operands.
// Instructions with _64 take 64-bit operands.
//
// VOP_* instructions can use either a 32-bit or 64-bit encoding.  The 32-bit
// encoding is the standard encoding, but instruction that make use of
// any of the instruction modifiers must use the 64-bit encoding.
//
// Instructions with _e32 use the 32-bit encoding.
// Instructions with _e64 use the 64-bit encoding.
//
//===----------------------------------------------------------------------===//

class SIMCInstr <string pseudo, int subtarget> {
  string PseudoInstr = pseudo;
  int Subtarget = subtarget;
}

//===----------------------------------------------------------------------===//
// EXP classes
//===----------------------------------------------------------------------===//

class EXPCommon : InstSI<
  (outs),
  (ins i32imm:$en, i32imm:$tgt, i32imm:$compr, i32imm:$done, i32imm:$vm,
       VReg_32:$src0, VReg_32:$src1, VReg_32:$src2, VReg_32:$src3),
  "EXP $en, $tgt, $compr, $done, $vm, $src0, $src1, $src2, $src3",
  [] > {

  let EXP_CNT = 1;
  let Uses = [EXEC];
}

multiclass EXP_m {

  let isPseudo = 1 in {
    def "" : EXPCommon, SIMCInstr <"EXP", SISubtarget.NONE> ;
  }

  def _si : EXPCommon, SIMCInstr <"EXP", SISubtarget.SI>, EXPe;
}

//===----------------------------------------------------------------------===//
// Scalar classes
//===----------------------------------------------------------------------===//

class SOP1_32 <bits<8> op, string opName, list<dag> pattern> : SOP1 <
  op, (outs SReg_32:$dst), (ins SSrc_32:$src0),
  opName#" $dst, $src0", pattern
>;

class SOP1_64 <bits<8> op, string opName, list<dag> pattern> : SOP1 <
  op, (outs SReg_64:$dst), (ins SSrc_64:$src0),
  opName#" $dst, $src0", pattern
>;

// 64-bit input, 32-bit output.
class SOP1_32_64 <bits<8> op, string opName, list<dag> pattern> : SOP1 <
  op, (outs SReg_32:$dst), (ins SSrc_64:$src0),
  opName#" $dst, $src0", pattern
>;

class SOP2_32 <bits<7> op, string opName, list<dag> pattern> : SOP2 <
  op, (outs SReg_32:$dst), (ins SSrc_32:$src0, SSrc_32:$src1),
  opName#" $dst, $src0, $src1", pattern
>;

class SOP2_64 <bits<7> op, string opName, list<dag> pattern> : SOP2 <
  op, (outs SReg_64:$dst), (ins SSrc_64:$src0, SSrc_64:$src1),
  opName#" $dst, $src0, $src1", pattern
>;

class SOP2_SHIFT_64 <bits<7> op, string opName, list<dag> pattern> : SOP2 <
  op, (outs SReg_64:$dst), (ins SSrc_64:$src0, SSrc_32:$src1),
  opName#" $dst, $src0, $src1", pattern
>;


class SOPC_Helper <bits<7> op, RegisterClass rc, ValueType vt,
                    string opName, PatLeaf cond> : SOPC <
  op, (outs SCCReg:$dst), (ins rc:$src0, rc:$src1),
  opName#" $dst, $src0, $src1", []>;

class SOPC_32<bits<7> op, string opName, PatLeaf cond = COND_NULL>
  : SOPC_Helper<op, SSrc_32, i32, opName, cond>;

class SOPC_64<bits<7> op, string opName, PatLeaf cond = COND_NULL>
  : SOPC_Helper<op, SSrc_64, i64, opName, cond>;

class SOPK_32 <bits<5> op, string opName, list<dag> pattern> : SOPK <
  op, (outs SReg_32:$dst), (ins i16imm:$src0),
  opName#" $dst, $src0", pattern
>;

class SOPK_64 <bits<5> op, string opName, list<dag> pattern> : SOPK <
  op, (outs SReg_64:$dst), (ins i16imm:$src0),
  opName#" $dst, $src0", pattern
>;

//===----------------------------------------------------------------------===//
// SMRD classes
//===----------------------------------------------------------------------===//

class SMRD_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  SMRD <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
}

class SMRD_Real_si <bits<5> op, string opName, bit imm, dag outs, dag ins,
                    string asm> :
  SMRD <outs, ins, asm, []>,
  SMRDe <op, imm>,
  SIMCInstr<opName, SISubtarget.SI>;

multiclass SMRD_m <bits<5> op, string opName, bit imm, dag outs, dag ins,
                   string asm, list<dag> pattern> {

  def "" : SMRD_Pseudo <opName, outs, ins, pattern>;

  def _si : SMRD_Real_si <op, opName, imm, outs, ins, asm>;

}

multiclass SMRD_Helper <bits<5> op, string opName, RegisterClass baseClass,
                        RegisterClass dstClass> {
  defm _IMM : SMRD_m <
    op, opName#"_IMM", 1, (outs dstClass:$dst),
    (ins baseClass:$sbase, u32imm:$offset),
    opName#" $dst, $sbase, $offset", []
  >;

  defm _SGPR : SMRD_m <
    op, opName#"_SGPR", 0, (outs dstClass:$dst),
    (ins baseClass:$sbase, SReg_32:$soff),
    opName#" $dst, $sbase, $soff", []
  >;
}

//===----------------------------------------------------------------------===//
// Vector ALU classes
//===----------------------------------------------------------------------===//

// This must always be right before the operand being input modified.
def InputMods : OperandWithDefaultOps <i32, (ops (i32 0))> {
  let PrintMethod = "printOperandAndMods";
}
def InputModsNoDefault : Operand <i32> {
  let PrintMethod = "printOperandAndMods";
}

class getNumSrcArgs<ValueType Src1, ValueType Src2> {
  int ret =
    !if (!eq(Src1.Value, untyped.Value),      1,   // VOP1
         !if (!eq(Src2.Value, untyped.Value), 2,   // VOP2
                                              3)); // VOP3
}

// Returns the register class to use for the destination of VOP[123C]
// instructions for the given VT.
class getVALUDstForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 32), VReg_32, VReg_64);
}

// Returns the register class to use for source 0 of VOP[12C]
// instructions for the given VT.
class getVOPSrc0ForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 32), VSrc_32, VSrc_64);
}

// Returns the register class to use for source 1 of VOP[12C] for the
// given VT.
class getVOPSrc1ForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 32), VReg_32, VReg_64);
}

// Returns the register classes for the source arguments of a VOP[12C]
// instruction for the given SrcVTs.
class getInRC32 <list<ValueType> SrcVT> {
  list<RegisterClass> ret = [
    getVOPSrc0ForVT<SrcVT[0]>.ret,
    getVOPSrc1ForVT<SrcVT[1]>.ret
  ];
}

// Returns the register class to use for sources of VOP3 instructions for the
// given VT.
class getVOP3SrcForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 32), VCSrc_32, VCSrc_64);
}

// Returns the register classes for the source arguments of a VOP3
// instruction for the given SrcVTs.
class getInRC64 <list<ValueType> SrcVT> {
  list<RegisterClass> ret = [
    getVOP3SrcForVT<SrcVT[0]>.ret,
    getVOP3SrcForVT<SrcVT[1]>.ret,
    getVOP3SrcForVT<SrcVT[2]>.ret
  ];
}

// Returns 1 if the source arguments have modifiers, 0 if they do not.
class hasModifiers<ValueType SrcVT> {
  bit ret = !if(!eq(SrcVT.Value, f32.Value), 1,
            !if(!eq(SrcVT.Value, f64.Value), 1, 0));
}

// Returns the input arguments for VOP[12C] instructions for the given SrcVT.
class getIns32 <RegisterClass Src0RC, RegisterClass Src1RC, int NumSrcArgs> {
  dag ret = !if(!eq(NumSrcArgs, 1), (ins Src0RC:$src0),               // VOP1
            !if(!eq(NumSrcArgs, 2), (ins Src0RC:$src0, Src1RC:$src1), // VOP2
                                    (ins)));
}

// Returns the input arguments for VOP3 instructions for the given SrcVT.
class getIns64 <RegisterClass Src0RC, RegisterClass Src1RC,
                RegisterClass Src2RC, int NumSrcArgs,
                bit HasModifiers> {

  dag ret =
    !if (!eq(NumSrcArgs, 1),
      !if (!eq(HasModifiers, 1),
        // VOP1 with modifiers
        (ins InputModsNoDefault:$src0_modifiers, Src0RC:$src0,
             ClampMod:$clamp, omod:$omod)
      /* else */,
        // VOP1 without modifiers
        (ins Src0RC:$src0)
      /* endif */ ),
    !if (!eq(NumSrcArgs, 2),
      !if (!eq(HasModifiers, 1),
        // VOP 2 with modifiers
        (ins InputModsNoDefault:$src0_modifiers, Src0RC:$src0,
             InputModsNoDefault:$src1_modifiers, Src1RC:$src1,
             ClampMod:$clamp, omod:$omod)
      /* else */,
        // VOP2 without modifiers
        (ins Src0RC:$src0, Src1RC:$src1)
      /* endif */ )
    /* NumSrcArgs == 3 */,
      !if (!eq(HasModifiers, 1),
        // VOP3 with modifiers
        (ins InputModsNoDefault:$src0_modifiers, Src0RC:$src0,
             InputModsNoDefault:$src1_modifiers, Src1RC:$src1,
             InputModsNoDefault:$src2_modifiers, Src2RC:$src2,
             ClampMod:$clamp, omod:$omod)
      /* else */,
        // VOP3 without modifiers
        (ins Src0RC:$src0, Src1RC:$src1, Src2RC:$src2)
      /* endif */ )));
}

// Returns the assembly string for the inputs and outputs of a VOP[12C]
// instruction.  This does not add the _e32 suffix, so it can be reused
// by getAsm64.
class getAsm32 <int NumSrcArgs> {
  string src1 = ", $src1";
  string src2 = ", $src2";
  string ret = " $dst, $src0"#
               !if(!eq(NumSrcArgs, 1), "", src1)#
               !if(!eq(NumSrcArgs, 3), src2, "");
}

// Returns the assembly string for the inputs and outputs of a VOP3
// instruction.
class getAsm64 <int NumSrcArgs, bit HasModifiers> {
  string src0 = "$src0_modifiers,";
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                           " $src1_modifiers,"));
  string src2 = !if(!eq(NumSrcArgs, 3), " $src2_modifiers", "");
  string ret =
  !if(!eq(HasModifiers, 0),
      getAsm32<NumSrcArgs>.ret,
      " $dst, "#src0#src1#src2#"$clamp"#"$omod");
}


class VOPProfile <list<ValueType> _ArgVT> {

  field list<ValueType> ArgVT = _ArgVT;

  field ValueType DstVT = ArgVT[0];
  field ValueType Src0VT = ArgVT[1];
  field ValueType Src1VT = ArgVT[2];
  field ValueType Src2VT = ArgVT[3];
  field RegisterClass DstRC = getVALUDstForVT<DstVT>.ret;
  field RegisterClass Src0RC32 = getVOPSrc0ForVT<Src0VT>.ret;
  field RegisterClass Src1RC32 = getVOPSrc1ForVT<Src1VT>.ret;
  field RegisterClass Src0RC64 = getVOP3SrcForVT<Src0VT>.ret;
  field RegisterClass Src1RC64 = getVOP3SrcForVT<Src1VT>.ret;
  field RegisterClass Src2RC64 = getVOP3SrcForVT<Src2VT>.ret;

  field int NumSrcArgs = getNumSrcArgs<Src1VT, Src2VT>.ret;
  field bit HasModifiers = hasModifiers<Src0VT>.ret;

  field dag Outs = (outs DstRC:$dst);

  field dag Ins32 = getIns32<Src0RC32, Src1RC32, NumSrcArgs>.ret;
  field dag Ins64 = getIns64<Src0RC64, Src1RC64, Src2RC64, NumSrcArgs,
                             HasModifiers>.ret;

  field string Asm32 = "_e32"#getAsm32<NumSrcArgs>.ret;
  field string Asm64 = getAsm64<NumSrcArgs, HasModifiers>.ret;
}

def VOP_F32_F32 : VOPProfile <[f32, f32, untyped, untyped]>;
def VOP_F32_F64 : VOPProfile <[f32, f64, untyped, untyped]>;
def VOP_F32_I32 : VOPProfile <[f32, i32, untyped, untyped]>;
def VOP_F64_F32 : VOPProfile <[f64, f32, untyped, untyped]>;
def VOP_F64_F64 : VOPProfile <[f64, f64, untyped, untyped]>;
def VOP_F64_I32 : VOPProfile <[f64, i32, untyped, untyped]>;
def VOP_I32_F32 : VOPProfile <[i32, f32, untyped, untyped]>;
def VOP_I32_F64 : VOPProfile <[i32, f64, untyped, untyped]>;
def VOP_I32_I32 : VOPProfile <[i32, i32, untyped, untyped]>;

def VOP_F32_F32_F32 : VOPProfile <[f32, f32, f32, untyped]>;
def VOP_F32_F32_I32 : VOPProfile <[f32, f32, i32, untyped]>;
def VOP_F64_F64_F64 : VOPProfile <[f64, f64, f64, untyped]>;
def VOP_F64_F64_I32 : VOPProfile <[f64, f64, i32, untyped]>;
def VOP_I32_F32_F32 : VOPProfile <[i32, f32, f32, untyped]>;
def VOP_I32_I32_I32 : VOPProfile <[i32, i32, i32, untyped]>;
def VOP_I32_I32_I32_VCC : VOPProfile <[i32, i32, i32, untyped]> {
  let Src0RC32 = VCSrc_32;
}
def VOP_I64_I64_I32 : VOPProfile <[i64, i64, i32, untyped]>;
def VOP_I64_I64_I64 : VOPProfile <[i64, i64, i64, untyped]>;

def VOP_F32_F32_F32_F32 : VOPProfile <[f32, f32, f32, f32]>;
def VOP_F64_F64_F64_F64 : VOPProfile <[f64, f64, f64, f64]>;
def VOP_I32_I32_I32_I32 : VOPProfile <[i32, i32, i32, i32]>;
def VOP_I64_I32_I32_I64 : VOPProfile <[i64, i32, i32, i64]>;


class VOP <string opName> {
  string OpName = opName;
}

class VOP2_REV <string revOp, bit isOrig> {
  string RevOp = revOp;
  bit IsOrig = isOrig;
}

class AtomicNoRet <string noRetOp, bit isRet> {
  string NoRetOp = noRetOp;
  bit IsRet = isRet;
}

class VOP1_Pseudo <dag outs, dag ins, list<dag> pattern, string opName> :
  VOP1Common <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
}

multiclass VOP1_m <vop1 op, dag outs, dag ins, string asm, list<dag> pattern,
                   string opName> {
  def "" : VOP1_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP1<op.SI, outs, ins, asm, []>,
            SIMCInstr <opName, SISubtarget.SI>;
}

class VOP3DisableFields <bit HasSrc1, bit HasSrc2, bit HasModifiers> {

  bits<2> src0_modifiers = !if(HasModifiers, ?, 0);
  bits<2> src1_modifiers = !if(HasModifiers, !if(HasSrc1, ?, 0), 0);
  bits<2> src2_modifiers = !if(HasModifiers, !if(HasSrc2, ? ,0) ,0);
  bits<2> omod = !if(HasModifiers, ?, 0);
  bits<1> clamp = !if(HasModifiers, ?, 0);
  bits<9> src1 = !if(HasSrc1, ?, 0);
  bits<9> src2 = !if(HasSrc2, ?, 0);
}

class VOP3_Pseudo <dag outs, dag ins, list<dag> pattern, string opName> :
  VOP3Common <outs, ins, "", pattern>,
  VOP <opName>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
}

class VOP3_Real_si <bits<9> op, dag outs, dag ins, string asm, string opName> :
  VOP3 <op, outs, ins, asm, []>,
  SIMCInstr<opName, SISubtarget.SI>;

multiclass VOP3_m <vop3 op, dag outs, dag ins, string asm, list<dag> pattern,
                   string opName, int NumSrcArgs, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<!if(!eq(NumSrcArgs, 1), 0, 1),
                              !if(!eq(NumSrcArgs, 2), 0, 1),
                              HasMods>;

}

multiclass VOP3_1_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<0, 0, HasMods>;
}

multiclass VOP3_2_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, string revOp,
                     bit HasMods = 1, bit UseFullOp = 0> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)>;

  def _si : VOP3_Real_si <op.SI3,
              outs, ins, asm, opName>,
            VOP2_REV<revOp#"_e64_si", !eq(revOp, opName)>,
            VOP3DisableFields<1, 0, HasMods>;
}

multiclass VOP3b_2_m <vop op, dag outs, dag ins, string asm,
                      list<dag> pattern, string opName, string revOp,
                      bit HasMods = 1, bit UseFullOp = 0> {
  def "" : VOP3_Pseudo <outs, ins, pattern, opName>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)>;

  // The VOP2 variant puts the carry out into VCC, the VOP3 variant
  // can write it into any SGPR. We currently don't use the carry out,
  // so for now hardcode it to VCC as well.
  let sdst = SIOperand.VCC, Defs = [VCC] in {
    def _si : VOP3b <op.SI3, outs, ins, asm, pattern>,
              VOP3DisableFields<1, 0, HasMods>,
              SIMCInstr<opName, SISubtarget.SI>,
              VOP2_REV<revOp#"_e64_si", !eq(revOp, opName)>;
  } // End sdst = SIOperand.VCC, Defs = [VCC]
}

multiclass VOP3_C_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName,
                     bit HasMods, bit defExec> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
              VOP3DisableFields<1, 0, HasMods> {
    let Defs = !if(defExec, [EXEC], []);
  }
}

multiclass VOP1_Helper <vop1 op, string opName, dag outs,
                        dag ins32, string asm32, list<dag> pat32,
                        dag ins64, string asm64, list<dag> pat64,
                        bit HasMods> {

  def _e32 : VOP1 <op.SI, outs, ins32, opName#asm32, pat32>, VOP<opName>;

  defm _e64 : VOP3_1_m <op, outs, ins64, opName#"_e64"#asm64, pat64, opName, HasMods>;
}

multiclass VOP1Inst <vop1 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag> : VOP1_Helper <
  op, opName, P.Outs,
  P.Ins32, P.Asm32, [],
  P.Ins64, P.Asm64,
  !if(P.HasModifiers,
      [(set P.DstVT:$dst, (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0,
                                i32:$src0_modifiers, i1:$clamp, i32:$omod))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0))]),
  P.HasModifiers
>;

class VOP2_e32 <bits<6> op, string opName, dag outs, dag ins, string asm,
                list<dag> pattern, string revOp> :
  VOP2 <op, outs, ins, opName#asm, pattern>,
  VOP <opName>,
  VOP2_REV<revOp#"_e32", !eq(revOp, opName)>;

multiclass VOP2_Helper <vop2 op, string opName, dag outs,
                        dag ins32, string asm32, list<dag> pat32,
                        dag ins64, string asm64, list<dag> pat64,
                        string revOp, bit HasMods> {
  def _e32 : VOP2_e32 <op.SI, opName, outs, ins32, asm32, pat32, revOp>;

  defm _e64 : VOP3_2_m <op,
    outs, ins64, opName#"_e64"#asm64, pat64, opName, revOp, HasMods
  >;
}

multiclass VOP2Inst <vop2 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag,
                     string revOp = opName> : VOP2_Helper <
  op, opName, P.Outs,
  P.Ins32, P.Asm32, [],
  P.Ins64, P.Asm64,
  !if(P.HasModifiers,
      [(set P.DstVT:$dst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp, P.HasModifiers
>;

multiclass VOP2b_Helper <vop2 op, string opName, dag outs,
                         dag ins32, string asm32, list<dag> pat32,
                         dag ins64, string asm64, list<dag> pat64,
                         string revOp, bit HasMods> {

  def _e32 : VOP2_e32 <op.SI, opName, outs, ins32, asm32, pat32, revOp>;

  defm _e64 : VOP3b_2_m <op,
    outs, ins64, opName#"_e64"#asm64, pat64, opName, revOp, HasMods
  >;
}

multiclass VOP2bInst <vop2 op, string opName, VOPProfile P,
                      SDPatternOperator node = null_frag,
                      string revOp = opName> : VOP2b_Helper <
  op, opName, P.Outs,
  P.Ins32, P.Asm32, [],
  P.Ins64, P.Asm64,
  !if(P.HasModifiers,
      [(set P.DstVT:$dst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp, P.HasModifiers
>;

multiclass VOPC_Helper <vopc op, string opName,
                        dag ins32, string asm32, list<dag> pat32,
                        dag out64, dag ins64, string asm64, list<dag> pat64,
                        bit HasMods, bit DefExec> {
  def _e32 : VOPC <op.SI, ins32, opName#asm32, pat32>, VOP <opName> {
    let Defs = !if(DefExec, [EXEC], []);
  }

  defm _e64 : VOP3_C_m <op, out64, ins64, opName#"_e64"#asm64, pat64, opName,
                        HasMods, DefExec>;
}

multiclass VOPCInst <vopc op, string opName,
                     VOPProfile P, PatLeaf cond = COND_NULL,
                     bit DefExec = 0> : VOPC_Helper <
  op, opName,
  P.Ins32, P.Asm32, [],
  (outs SReg_64:$dst), P.Ins64, P.Asm64,
  !if(P.HasModifiers,
      [(set i1:$dst,
          (setcc (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                 cond))],
      [(set i1:$dst, (setcc P.Src0VT:$src0, P.Src1VT:$src1, cond))]),
  P.HasModifiers, DefExec
>;

multiclass VOPC_F32 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCInst <op, opName, VOP_F32_F32_F32, cond>;

multiclass VOPC_F64 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCInst <op, opName, VOP_F64_F64_F64, cond>;

multiclass VOPC_I32 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCInst <op, opName, VOP_I32_I32_I32, cond>;

multiclass VOPC_I64 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCInst <op, opName, VOP_I64_I64_I64, cond>;


multiclass VOPCX <vopc op, string opName, VOPProfile P,
                  PatLeaf cond = COND_NULL>
  : VOPCInst <op, opName, P, cond, 1>;

multiclass VOPCX_F32 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCX <op, opName, VOP_F32_F32_F32, cond>;

multiclass VOPCX_F64 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCX <op, opName, VOP_F64_F64_F64, cond>;

multiclass VOPCX_I32 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCX <op, opName, VOP_I32_I32_I32, cond>;

multiclass VOPCX_I64 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCX <op, opName, VOP_I64_I64_I64, cond>;

multiclass VOP3_Helper <vop3 op, string opName, dag outs, dag ins, string asm,
                        list<dag> pat, int NumSrcArgs, bit HasMods> : VOP3_m <
    op, outs, ins, opName#asm, pat, opName, NumSrcArgs, HasMods
>;

multiclass VOP3Inst <vop3 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag> : VOP3_Helper <
  op, opName, P.Outs, P.Ins64, P.Asm64,
  !if(!eq(P.NumSrcArgs, 3),
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                  (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1,
                                  P.Src2VT:$src2))]),
  !if(!eq(P.NumSrcArgs, 2),
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))])
  /* P.NumSrcArgs == 1 */,
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0))]))),
  P.NumSrcArgs, P.HasModifiers
>;

multiclass VOP3b_Helper <vop op, RegisterClass vrc, RegisterClass arc,
                    string opName, list<dag> pattern> :
  VOP3b_2_m <
  op, (outs vrc:$dst0, SReg_64:$dst1),
      (ins InputModsNoDefault:$src0_modifiers, arc:$src0,
           InputModsNoDefault:$src1_modifiers, arc:$src1,
           InputModsNoDefault:$src2_modifiers, arc:$src2,
           ClampMod:$clamp, i32imm:$omod),
  opName#" $dst0, $dst1, $src0_modifiers, $src1_modifiers, $src2_modifiers"#"$clamp"#"$omod", pattern,
  opName, opName, 1, 1
>;

multiclass VOP3b_64 <vop3 op, string opName, list<dag> pattern> :
  VOP3b_Helper <op, VReg_64, VSrc_64, opName, pattern>;

multiclass VOP3b_32 <vop3 op, string opName, list<dag> pattern> :
  VOP3b_Helper <op, VReg_32, VSrc_32, opName, pattern>;


class Vop3ModPat<Instruction Inst, VOPProfile P, SDPatternOperator node> : Pat<
  (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod)),
        (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
        (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers))),
  (Inst i32:$src0_modifiers, P.Src0VT:$src0,
        i32:$src1_modifiers, P.Src1VT:$src1,
        i32:$src2_modifiers, P.Src2VT:$src2,
        i1:$clamp,
        i32:$omod)>;

//===----------------------------------------------------------------------===//
// Vector I/O classes
//===----------------------------------------------------------------------===//

class DS_1A <bits<8> op, dag outs, dag ins, string asm, list<dag> pat> :
    DS <op, outs, ins, asm, pat> {
  bits<16> offset;

  // Single load interpret the 2 i8imm operands as a single i16 offset.
  let offset0 = offset{7-0};
  let offset1 = offset{15-8};
}

class DS_Load_Helper <bits<8> op, string asm, RegisterClass regClass> : DS_1A <
  op,
  (outs regClass:$vdst),
  (ins i1imm:$gds, VReg_32:$addr, ds_offset:$offset),
  asm#" $vdst, $addr"#"$offset"#" [M0]",
  []> {
  let data0 = 0;
  let data1 = 0;
  let mayLoad = 1;
  let mayStore = 0;
}

class DS_Load2_Helper <bits<8> op, string asm, RegisterClass regClass> : DS <
  op,
  (outs regClass:$vdst),
  (ins i1imm:$gds, VReg_32:$addr, ds_offset0:$offset0, ds_offset1:$offset1),
  asm#" $vdst, $addr"#"$offset0"#"$offset1 [M0]",
  []> {
  let data0 = 0;
  let data1 = 0;
  let mayLoad = 1;
  let mayStore = 0;
}

class DS_Store_Helper <bits<8> op, string asm, RegisterClass regClass> : DS_1A <
  op,
  (outs),
  (ins i1imm:$gds, VReg_32:$addr, regClass:$data0, ds_offset:$offset),
  asm#" $addr, $data0"#"$offset"#" [M0]",
  []> {
  let data1 = 0;
  let mayStore = 1;
  let mayLoad = 0;
  let vdst = 0;
}

class DS_Store2_Helper <bits<8> op, string asm, RegisterClass regClass> : DS <
  op,
  (outs),
  (ins i1imm:$gds, VReg_32:$addr, regClass:$data0, regClass:$data1,
       ds_offset0:$offset0, ds_offset1:$offset1),
  asm#" $addr, $data0, $data1"#"$offset0"#"$offset1 [M0]",
  []> {
  let mayStore = 1;
  let mayLoad = 0;
  let vdst = 0;
}

// 1 address, 1 data.
class DS_1A1D_RET <bits<8> op, string asm, RegisterClass rc, string noRetOp = ""> : DS_1A <
  op,
  (outs rc:$vdst),
  (ins i1imm:$gds, VReg_32:$addr, rc:$data0, ds_offset:$offset),
  asm#" $vdst, $addr, $data0"#"$offset"#" [M0]", []>,
  AtomicNoRet<noRetOp, 1> {

  let data1 = 0;
  let mayStore = 1;
  let mayLoad = 1;

  let hasPostISelHook = 1; // Adjusted to no return version.
}

// 1 address, 2 data.
class DS_1A2D_RET <bits<8> op, string asm, RegisterClass rc, string noRetOp = ""> : DS_1A <
  op,
  (outs rc:$vdst),
  (ins i1imm:$gds, VReg_32:$addr, rc:$data0, rc:$data1, ds_offset:$offset),
  asm#" $vdst, $addr, $data0, $data1"#"$offset"#" [M0]",
  []>,
  AtomicNoRet<noRetOp, 1> {
  let mayStore = 1;
  let mayLoad = 1;

  let hasPostISelHook = 1; // Adjusted to no return version.
}

// 1 address, 2 data.
class DS_1A2D_NORET <bits<8> op, string asm, RegisterClass rc, string noRetOp = asm> : DS_1A <
  op,
  (outs),
  (ins i1imm:$gds, VReg_32:$addr, rc:$data0, rc:$data1, ds_offset:$offset),
  asm#" $addr, $data0, $data1"#"$offset"#" [M0]",
  []>,
  AtomicNoRet<noRetOp, 0> {
  let mayStore = 1;
  let mayLoad = 1;
}

// 1 address, 1 data.
class DS_1A1D_NORET <bits<8> op, string asm, RegisterClass rc, string noRetOp = asm> : DS_1A <
  op,
  (outs),
  (ins i1imm:$gds, VReg_32:$addr, rc:$data0, ds_offset:$offset),
  asm#" $addr, $data0"#"$offset"#" [M0]",
  []>,
  AtomicNoRet<noRetOp, 0> {

  let data1 = 0;
  let mayStore = 1;
  let mayLoad = 1;
}

//===----------------------------------------------------------------------===//
// MTBUF classes
//===----------------------------------------------------------------------===//

class MTBUF_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  MTBUF <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
}

class MTBUF_Real_si <bits<3> op, string opName, dag outs, dag ins,
                    string asm> :
  MTBUF <outs, ins, asm, []>,
  MTBUFe <op>,
  SIMCInstr<opName, SISubtarget.SI>;

multiclass MTBUF_m <bits<3> op, string opName, dag outs, dag ins, string asm,
                    list<dag> pattern> {

  def "" : MTBUF_Pseudo <opName, outs, ins, pattern>;

  def _si : MTBUF_Real_si <op, opName, outs, ins, asm>;

}

let mayStore = 1, mayLoad = 0 in {

multiclass MTBUF_Store_Helper <bits<3> op, string opName,
                               RegisterClass regClass> : MTBUF_m <
  op, opName, (outs),
  (ins regClass:$vdata, u16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc,
   i1imm:$addr64, i8imm:$dfmt, i8imm:$nfmt, VReg_32:$vaddr,
   SReg_128:$srsrc, i1imm:$slc, i1imm:$tfe, SSrc_32:$soffset),
  opName#" $vdata, $offset, $offen, $idxen, $glc, $addr64, $dfmt,"
        #" $nfmt, $vaddr, $srsrc, $slc, $tfe, $soffset", []
>;

} // mayStore = 1, mayLoad = 0

let mayLoad = 1, mayStore = 0 in {

multiclass MTBUF_Load_Helper <bits<3> op, string opName,
                              RegisterClass regClass> : MTBUF_m <
  op, opName, (outs regClass:$dst),
  (ins u16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc, i1imm:$addr64,
       i8imm:$dfmt, i8imm:$nfmt, VReg_32:$vaddr, SReg_128:$srsrc,
       i1imm:$slc, i1imm:$tfe, SSrc_32:$soffset),
  opName#" $dst, $offset, $offen, $idxen, $glc, $addr64, $dfmt,"
        #" $nfmt, $vaddr, $srsrc, $slc, $tfe, $soffset", []
>;

} // mayLoad = 1, mayStore = 0

class MUBUFAddr64Table <bit is_addr64, string suffix = ""> {

  bit IsAddr64 = is_addr64;
  string OpName = NAME # suffix;
}

class MUBUFAtomicAddr64 <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern>
    : MUBUF <op, outs, ins, asm, pattern> {

  let offen = 0;
  let idxen = 0;
  let addr64 = 1;
  let tfe = 0;
  let lds = 0;
  let soffset = 128;
}

class MUBUFAtomicOffset <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern>
    : MUBUF <op, outs, ins, asm, pattern> {

  let offen = 0;
  let idxen = 0;
  let addr64 = 0;
  let tfe = 0;
  let lds = 0;
  let vaddr = 0;
}

multiclass MUBUF_Atomic <bits<7> op, string name, RegisterClass rc,
                         ValueType vt, SDPatternOperator atomic> {

  let mayStore = 1, mayLoad = 1, hasPostISelHook = 1 in {

    // No return variants
    let glc = 0 in {

      def _ADDR64 : MUBUFAtomicAddr64 <
        op, (outs),
        (ins rc:$vdata, SReg_128:$srsrc, VReg_64:$vaddr,
             mbuf_offset:$offset, slc:$slc),
        name#" $vdata, $vaddr, $srsrc, 0 addr64"#"$offset"#"$slc", []
      >, MUBUFAddr64Table<1>, AtomicNoRet<NAME#"_ADDR64", 0>;

      def _OFFSET : MUBUFAtomicOffset <
        op, (outs),
        (ins rc:$vdata, SReg_128:$srsrc, mbuf_offset:$offset,
             SSrc_32:$soffset, slc:$slc),
        name#" $vdata, $srsrc, $soffset"#"$offset"#"$slc", []
      >, MUBUFAddr64Table<0>, AtomicNoRet<NAME#"_OFFSET", 0>;
    } // glc = 0

    // Variant that return values
    let glc = 1, Constraints = "$vdata = $vdata_in",
        DisableEncoding = "$vdata_in"  in {

      def _RTN_ADDR64 : MUBUFAtomicAddr64 <
        op, (outs rc:$vdata),
        (ins rc:$vdata_in, SReg_128:$srsrc, VReg_64:$vaddr,
             mbuf_offset:$offset, slc:$slc),
        name#" $vdata, $vaddr, $srsrc, 0 addr64"#"$offset"#" glc"#"$slc",
        [(set vt:$vdata,
         (atomic (MUBUFAddr64Atomic v4i32:$srsrc, i64:$vaddr, i16:$offset,
                                    i1:$slc), vt:$vdata_in))]
      >, MUBUFAddr64Table<1, "_RTN">, AtomicNoRet<NAME#"_ADDR64", 1>;

      def _RTN_OFFSET : MUBUFAtomicOffset <
        op, (outs rc:$vdata),
        (ins rc:$vdata_in, SReg_128:$srsrc, mbuf_offset:$offset,
             SSrc_32:$soffset, slc:$slc),
        name#" $vdata, $srsrc, $soffset"#"$offset"#" glc $slc",
        [(set vt:$vdata,
         (atomic (MUBUFOffsetAtomic v4i32:$srsrc, i32:$soffset, i16:$offset,
                                    i1:$slc), vt:$vdata_in))]
      >, MUBUFAddr64Table<0, "_RTN">, AtomicNoRet<NAME#"_OFFSET", 1>;

    } // glc = 1

  } // mayStore = 1, mayLoad = 1, hasPostISelHook = 1
}

multiclass MUBUF_Load_Helper <bits<7> op, string asm, RegisterClass regClass,
                              ValueType load_vt = i32,
                              SDPatternOperator ld = null_frag> {

  let lds = 0, mayLoad = 1 in {

    let addr64 = 0 in {

      let offen = 0, idxen = 0, vaddr = 0 in {
        def _OFFSET : MUBUF <op, (outs regClass:$vdata),
                             (ins SReg_128:$srsrc,
                             mbuf_offset:$offset, SSrc_32:$soffset, glc:$glc,
                             slc:$slc, tfe:$tfe),
                             asm#" $vdata, $srsrc, $soffset"#"$offset"#"$glc"#"$slc"#"$tfe",
                             [(set load_vt:$vdata, (ld (MUBUFOffset v4i32:$srsrc,
                                                       i32:$soffset, i16:$offset,
                                                       i1:$glc, i1:$slc, i1:$tfe)))]>,
                     MUBUFAddr64Table<0>;
      }

      let offen = 1, idxen = 0  in {
        def _OFFEN  : MUBUF <op, (outs regClass:$vdata),
                             (ins SReg_128:$srsrc, VReg_32:$vaddr,
                             SSrc_32:$soffset, mbuf_offset:$offset, glc:$glc, slc:$slc,
                             tfe:$tfe),
                             asm#" $vdata, $vaddr, $srsrc, $soffset offen"#"$offset"#"$glc"#"$slc"#"$tfe", []>;
      }

      let offen = 0, idxen = 1 in {
        def _IDXEN  : MUBUF <op, (outs regClass:$vdata),
                             (ins SReg_128:$srsrc, VReg_32:$vaddr,
                             mbuf_offset:$offset, SSrc_32:$soffset, glc:$glc,
                             slc:$slc, tfe:$tfe),
                             asm#" $vdata, $vaddr, $srsrc, $soffset idxen"#"$offset"#"$glc"#"$slc"#"$tfe", []>;
      }

      let offen = 1, idxen = 1 in {
        def _BOTHEN : MUBUF <op, (outs regClass:$vdata),
                             (ins SReg_128:$srsrc, VReg_64:$vaddr,
                             SSrc_32:$soffset, glc:$glc, slc:$slc, tfe:$tfe),
                             asm#" $vdata, $vaddr, $srsrc, $soffset, idxen offen"#"$glc"#"$slc"#"$tfe", []>;
      }
    }

    let offen = 0, idxen = 0, addr64 = 1, glc = 0, slc = 0, tfe = 0, soffset = 128 /* ZERO */ in {
      def _ADDR64 : MUBUF <op, (outs regClass:$vdata),
                           (ins SReg_128:$srsrc, VReg_64:$vaddr, mbuf_offset:$offset),
                           asm#" $vdata, $vaddr, $srsrc, 0 addr64"#"$offset",
                           [(set load_vt:$vdata, (ld (MUBUFAddr64 v4i32:$srsrc,
                                                  i64:$vaddr, i16:$offset)))]>, MUBUFAddr64Table<1>;
    }
  }
}

multiclass MUBUF_Store_Helper <bits<7> op, string name, RegisterClass vdataClass,
                          ValueType store_vt, SDPatternOperator st> {

  let addr64 = 0, lds = 0 in {

    def "" : MUBUF <
      op, (outs),
      (ins vdataClass:$vdata, SReg_128:$srsrc, VReg_32:$vaddr, SSrc_32:$soffset,
           mbuf_offset:$offset, offen:$offen, idxen:$idxen, glc:$glc, slc:$slc,
           tfe:$tfe),
      name#" $vdata, $vaddr, $srsrc, $soffset"#"$offen"#"$idxen"#"$offset"#
           "$glc"#"$slc"#"$tfe",
      []
    >;

    let offen = 0, idxen = 0, vaddr = 0 in {
      def _OFFSET : MUBUF <
        op, (outs),
        (ins vdataClass:$vdata, SReg_128:$srsrc, mbuf_offset:$offset,
              SSrc_32:$soffset, glc:$glc, slc:$slc, tfe:$tfe),
        name#" $vdata, $srsrc, $soffset"#"$offset"#"$glc"#"$slc"#"$tfe",
        [(st store_vt:$vdata, (MUBUFOffset v4i32:$srsrc, i32:$soffset,
                                           i16:$offset, i1:$glc, i1:$slc,
                                           i1:$tfe))]
      >, MUBUFAddr64Table<0>;
    } // offen = 0, idxen = 0, vaddr = 0

    let offen = 1, idxen = 0  in {
      def _OFFEN  : MUBUF <
        op, (outs),
        (ins vdataClass:$vdata, SReg_128:$srsrc, VReg_32:$vaddr, SSrc_32:$soffset,
             mbuf_offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
        name#" $vdata, $vaddr, $srsrc, $soffset offen"#"$offset"#
            "$glc"#"$slc"#"$tfe",
        []
      >;
    } // end offen = 1, idxen = 0

  } // End addr64 = 0, lds = 0

  def _ADDR64 : MUBUF <
    op, (outs),
    (ins vdataClass:$vdata, SReg_128:$srsrc, VReg_64:$vaddr, mbuf_offset:$offset),
    name#" $vdata, $vaddr, $srsrc, 0 addr64"#"$offset",
    [(st store_vt:$vdata,
     (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i16:$offset))]>, MUBUFAddr64Table<1>
     {

      let mayLoad = 0;
      let mayStore = 1;

      // Encoding
      let offen = 0;
      let idxen = 0;
      let glc = 0;
      let addr64 = 1;
      let lds = 0;
      let slc = 0;
      let tfe = 0;
      let soffset = 128; // ZERO
   }
}

class FLAT_Load_Helper <bits<7> op, string asm, RegisterClass regClass> :
      FLAT <op, (outs regClass:$data),
                (ins VReg_64:$addr),
            asm#" $data, $addr, [M0, FLAT_SCRATCH]", []> {
  let glc = 0;
  let slc = 0;
  let tfe = 0;
  let mayLoad = 1;
}

class FLAT_Store_Helper <bits<7> op, string name, RegisterClass vdataClass> :
      FLAT <op, (outs), (ins vdataClass:$data, VReg_64:$addr),
          name#" $data, $addr, [M0, FLAT_SCRATCH]",
         []> {

  let mayLoad = 0;
  let mayStore = 1;

  // Encoding
  let glc = 0;
  let slc = 0;
  let tfe = 0;
}

class MIMG_Mask <string op, int channels> {
  string Op = op;
  int Channels = channels;
}

class MIMG_NoSampler_Helper <bits<7> op, string asm,
                             RegisterClass dst_rc,
                             RegisterClass src_rc> : MIMG <
  op,
  (outs dst_rc:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, src_rc:$vaddr,
       SReg_256:$srsrc),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc",
  []> {
  let SSAMP = 0;
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
}

multiclass MIMG_NoSampler_Src_Helper <bits<7> op, string asm,
                                      RegisterClass dst_rc,
                                      int channels> {
  def _V1 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_32>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_64>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_128>,
            MIMG_Mask<asm#"_V4", channels>;
}

multiclass MIMG_NoSampler <bits<7> op, string asm> {
  defm _V1 : MIMG_NoSampler_Src_Helper <op, asm, VReg_32, 1>;
  defm _V2 : MIMG_NoSampler_Src_Helper <op, asm, VReg_64, 2>;
  defm _V3 : MIMG_NoSampler_Src_Helper <op, asm, VReg_96, 3>;
  defm _V4 : MIMG_NoSampler_Src_Helper <op, asm, VReg_128, 4>;
}

class MIMG_Sampler_Helper <bits<7> op, string asm,
                           RegisterClass dst_rc,
                           RegisterClass src_rc> : MIMG <
  op,
  (outs dst_rc:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, src_rc:$vaddr,
       SReg_256:$srsrc, SReg_128:$ssamp),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc, $ssamp",
  []> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
}

multiclass MIMG_Sampler_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels> {
  def _V1 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_32>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_64>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_128>,
            MIMG_Mask<asm#"_V4", channels>;
  def _V8 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_256>,
            MIMG_Mask<asm#"_V8", channels>;
  def _V16 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_512>,
            MIMG_Mask<asm#"_V16", channels>;
}

multiclass MIMG_Sampler <bits<7> op, string asm> {
  defm _V1 : MIMG_Sampler_Src_Helper<op, asm, VReg_32, 1>;
  defm _V2 : MIMG_Sampler_Src_Helper<op, asm, VReg_64, 2>;
  defm _V3 : MIMG_Sampler_Src_Helper<op, asm, VReg_96, 3>;
  defm _V4 : MIMG_Sampler_Src_Helper<op, asm, VReg_128, 4>;
}

class MIMG_Gather_Helper <bits<7> op, string asm,
                          RegisterClass dst_rc,
                          RegisterClass src_rc> : MIMG <
  op,
  (outs dst_rc:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, src_rc:$vaddr,
       SReg_256:$srsrc, SReg_128:$ssamp),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc, $ssamp",
  []> {
  let mayLoad = 1;
  let mayStore = 0;

  // DMASK was repurposed for GATHER4. 4 components are always
  // returned and DMASK works like a swizzle - it selects
  // the component to fetch. The only useful DMASK values are
  // 1=red, 2=green, 4=blue, 8=alpha. (e.g. 1 returns
  // (red,red,red,red) etc.) The ISA document doesn't mention
  // this.
  // Therefore, disable all code which updates DMASK by setting these two:
  let MIMG = 0;
  let hasPostISelHook = 0;
}

multiclass MIMG_Gather_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels> {
  def _V1 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_32>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_64>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_128>,
            MIMG_Mask<asm#"_V4", channels>;
  def _V8 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_256>,
            MIMG_Mask<asm#"_V8", channels>;
  def _V16 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_512>,
            MIMG_Mask<asm#"_V16", channels>;
}

multiclass MIMG_Gather <bits<7> op, string asm> {
  defm _V1 : MIMG_Gather_Src_Helper<op, asm, VReg_32, 1>;
  defm _V2 : MIMG_Gather_Src_Helper<op, asm, VReg_64, 2>;
  defm _V3 : MIMG_Gather_Src_Helper<op, asm, VReg_96, 3>;
  defm _V4 : MIMG_Gather_Src_Helper<op, asm, VReg_128, 4>;
}

//===----------------------------------------------------------------------===//
// Vector instruction mappings
//===----------------------------------------------------------------------===//

// Maps an opcode in e32 form to its e64 equivalent
def getVOPe64 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size"];
  let KeyCol = ["4"];
  let ValueCols = [["8"]];
}

// Maps an opcode in e64 form to its e32 equivalent
def getVOPe32 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size"];
  let KeyCol = ["8"];
  let ValueCols = [["4"]];
}

// Maps an original opcode to its commuted version
def getCommuteRev : InstrMapping {
  let FilterClass = "VOP2_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

def getMaskedMIMGOp : InstrMapping {
  let FilterClass = "MIMG_Mask";
  let RowFields = ["Op"];
  let ColFields = ["Channels"];
  let KeyCol = ["4"];
  let ValueCols = [["1"], ["2"], ["3"] ];
}

// Maps an commuted opcode to its original version
def getCommuteOrig : InstrMapping {
  let FilterClass = "VOP2_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

def isDS : InstrMapping {
  let FilterClass = "DS";
  let RowFields = ["Inst"];
  let ColFields = ["Size"];
  let KeyCol = ["8"];
  let ValueCols = [["8"]];
}

def getMCOpcode : InstrMapping {
  let FilterClass = "SIMCInstr";
  let RowFields = ["PseudoInstr"];
  let ColFields = ["Subtarget"];
  let KeyCol = [!cast<string>(SISubtarget.NONE)];
  let ValueCols = [[!cast<string>(SISubtarget.SI)]];
}

def getAddr64Inst : InstrMapping {
  let FilterClass = "MUBUFAddr64Table";
  let RowFields = ["OpName"];
  let ColFields = ["IsAddr64"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an atomic opcode to its version with a return value.
def getAtomicRetOp : InstrMapping {
  let FilterClass = "AtomicNoRet";
  let RowFields = ["NoRetOp"];
  let ColFields = ["IsRet"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an atomic opcode to its returnless version.
def getAtomicNoRetOp : InstrMapping {
  let FilterClass = "AtomicNoRet";
  let RowFields = ["NoRetOp"];
  let ColFields = ["IsRet"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

include "SIInstructions.td"
