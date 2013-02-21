//===-- SIInstrInfo.td - SI Instruction Infos -------------*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SI DAG Nodes
//===----------------------------------------------------------------------===//

// SMRD takes a 64bit memory address and can only add an 32bit offset
def SIadd64bit32bit : SDNode<"ISD::ADD",
  SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>, SDTCisVT<0, i64>, SDTCisVT<2, i32>]>
>;

// Transformation function, extract the lower 32bit of a 64bit immediate
def LO32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 0xffffffff, MVT::i32);
}]>;

// Transformation function, extract the upper 32bit of a 64bit immediate
def HI32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() >> 32, MVT::i32);
}]>;

def IMM8bitDWORD : ImmLeaf <
  i32, [{
    return (Imm & ~0x3FC) == 0;
  }], SDNodeXForm<imm, [{
    return CurDAG->getTargetConstant(
      N->getZExtValue() >> 2, MVT::i32);
  }]>
>;

def IMM12bit : ImmLeaf <
  i16,
  [{return isUInt<12>(Imm);}]
>;

class InlineImm <ValueType vt> : ImmLeaf <vt, [{
  return -16 <= Imm && Imm <= 64;
}]>;


//===----------------------------------------------------------------------===//
// SI assembler operands
//===----------------------------------------------------------------------===//

def SIOperand {
  int ZERO = 0x80;
}

class GPR4Align <RegisterClass rc> : Operand <vAny> {
  let EncoderMethod = "GPR4AlignEncode";
  let MIOperandInfo = (ops rc:$reg); 
}

class GPR2Align <RegisterClass rc> : Operand <iPTR> {
  let EncoderMethod = "GPR2AlignEncode";
  let MIOperandInfo = (ops rc:$reg);
}

include "SIInstrFormats.td"

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

//===----------------------------------------------------------------------===//
// Scalar classes
//===----------------------------------------------------------------------===//

class SOP1_32 <bits<8> op, string opName, list<dag> pattern>
  : SOP1 <op, (outs SReg_32:$dst), (ins SSrc_32:$src0), opName, pattern>;

class SOP1_64 <bits<8> op, string opName, list<dag> pattern>
  : SOP1 <op, (outs SReg_64:$dst), (ins SSrc_64:$src0), opName, pattern>;

class SOP2_32 <bits<7> op, string opName, list<dag> pattern>
  : SOP2 <op, (outs SReg_32:$dst), (ins SSrc_32:$src0, SSrc_32:$src1), opName, pattern>;

class SOP2_64 <bits<7> op, string opName, list<dag> pattern>
  : SOP2 <op, (outs SReg_64:$dst), (ins SSrc_64:$src0, SSrc_64:$src1), opName, pattern>;

class SOPC_32 <bits<7> op, string opName, list<dag> pattern>
  : SOPC <op, (outs SCCReg:$dst), (ins SSrc_32:$src0, SSrc_32:$src1), opName, pattern>;

class SOPC_64 <bits<7> op, string opName, list<dag> pattern>
  : SOPC <op, (outs SCCReg:$dst), (ins SSrc_64:$src0, SSrc_64:$src1), opName, pattern>;

class SOPK_32 <bits<5> op, string opName, list<dag> pattern>
  : SOPK <op, (outs SReg_32:$dst), (ins i16imm:$src0), opName, pattern>;

class SOPK_64 <bits<5> op, string opName, list<dag> pattern>
  : SOPK <op, (outs SReg_64:$dst), (ins i16imm:$src0), opName, pattern>;

multiclass SMRD_Helper <bits<5> op, string asm, RegisterClass dstClass> {
  def _IMM : SMRD <
    op, 1, (outs dstClass:$dst),
    (ins GPR2Align<SReg_64>:$sbase, i32imm:$offset),
    asm, []
  >;

  def _SGPR : SMRD <
    op, 0, (outs dstClass:$dst),
    (ins GPR2Align<SReg_64>:$sbase, SReg_32:$soff),
    asm, []
  >;
}

//===----------------------------------------------------------------------===//
// Vector ALU classes
//===----------------------------------------------------------------------===//

class VOP3_32 <bits<9> op, string opName, list<dag> pattern> : VOP3 <
  op, (outs VReg_32:$dst),
  (ins VSrc_32:$src0, VReg_32:$src1, VReg_32:$src2, i32imm:$src3,
   i32imm:$src4, i32imm:$src5, i32imm:$src6),
  opName, pattern
>;

class VOP3_64 <bits<9> op, string opName, list<dag> pattern> : VOP3 <
  op, (outs VReg_64:$dst),
  (ins VSrc_64:$src0, VReg_64:$src1, VReg_64:$src2,
   i32imm:$src3, i32imm:$src4, i32imm:$src5, i32imm:$src6),
  opName, pattern
>;

class VOP1_Helper <bits<8> op, RegisterClass vrc, RegisterClass arc,
                   string opName, list<dag> pattern> : 
  VOP1 <
    op, (outs vrc:$dst), (ins arc:$src0), opName, pattern
  >;

multiclass VOP1_32 <bits<8> op, string opName, list<dag> pattern> {
  def _e32: VOP1_Helper <op, VReg_32, VSrc_32, opName, pattern>;
  def _e64 : VOP3_32 <{1, 1, op{6}, op{5}, op{4}, op{3}, op{2}, op{1}, op{0}},
                      opName, []
  >;
}

multiclass VOP1_64 <bits<8> op, string opName, list<dag> pattern> {

  def _e32 : VOP1_Helper <op, VReg_64, VSrc_64, opName, pattern>;

  def _e64 : VOP3_64 <
    {1, 1, op{6}, op{5}, op{4}, op{3}, op{2}, op{1}, op{0}},
    opName, []
  >;
}

class VOP2_Helper <bits<6> op, RegisterClass vrc, RegisterClass arc,
                   string opName, list<dag> pattern> :
  VOP2 <
    op, (outs vrc:$dst), (ins arc:$src0, vrc:$src1), opName, pattern
  >;

multiclass VOP2_32 <bits<6> op, string opName, list<dag> pattern> {

  def _e32 : VOP2_Helper <op, VReg_32, VSrc_32, opName, pattern>;

  def _e64 : VOP3_32 <{1, 0, 0, op{5}, op{4}, op{3}, op{2}, op{1}, op{0}},
                      opName, []
  >;
}

multiclass VOP2_64 <bits<6> op, string opName, list<dag> pattern> {
  def _e32: VOP2_Helper <op, VReg_64, VSrc_64, opName, pattern>;

  def _e64 : VOP3_64 <
    {1, 0, 0, op{5}, op{4}, op{3}, op{2}, op{1}, op{0}},
    opName, []
  >;
}

multiclass VOPC_Helper <bits<8> op, RegisterClass vrc, RegisterClass arc,
                        string opName, list<dag> pattern> {

  def _e32 : VOPC <op, (ins arc:$src0, vrc:$src1), opName, pattern>;
  def _e64 : VOP3 <
    {0, op{7}, op{6}, op{5}, op{4}, op{3}, op{2}, op{1}, op{0}},
    (outs SReg_64:$dst),
    (ins arc:$src0, vrc:$src1,
         InstFlag:$abs, InstFlag:$clamp,
         InstFlag:$omod, InstFlag:$neg),
    opName, pattern
  > {
    let SRC2 = SIOperand.ZERO;
  }
}

multiclass VOPC_32 <bits<8> op, string opName, list<dag> pattern>
  : VOPC_Helper <op, VReg_32, VSrc_32, opName, pattern>;

multiclass VOPC_64 <bits<8> op, string opName, list<dag> pattern>
  : VOPC_Helper <op, VReg_64, VSrc_64, opName, pattern>;

//===----------------------------------------------------------------------===//
// Vector I/O classes
//===----------------------------------------------------------------------===//

class MTBUF_Store_Helper <bits<3> op, string asm, RegisterClass regClass> : MTBUF <
  op,
  (outs),
  (ins regClass:$vdata, i16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc,
   i1imm:$addr64, i8imm:$dfmt, i8imm:$nfmt, VReg_32:$vaddr,
   GPR4Align<SReg_128>:$srsrc, i1imm:$slc, i1imm:$tfe, SSrc_32:$soffset),
  asm,
  []> {
  let mayStore = 1;
  let mayLoad = 0;
}

class MUBUF_Load_Helper <bits<7> op, string asm, RegisterClass regClass> : MUBUF <
  op,
  (outs regClass:$dst),
  (ins i16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc, i1imm:$addr64,
       i1imm:$lds, VReg_32:$vaddr, GPR4Align<SReg_128>:$srsrc, i1imm:$slc,
       i1imm:$tfe, SSrc_32:$soffset),
  asm,
  []> {
  let mayLoad = 1;
  let mayStore = 0;
}

class MTBUF_Load_Helper <bits<3> op, string asm, RegisterClass regClass> : MTBUF <
  op,
  (outs regClass:$dst),
  (ins i16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc, i1imm:$addr64,
       i8imm:$dfmt, i8imm:$nfmt, VReg_32:$vaddr, GPR4Align<SReg_128>:$srsrc,
       i1imm:$slc, i1imm:$tfe, SSrc_32:$soffset),
  asm,
  []> {
  let mayLoad = 1;
  let mayStore = 0;
}

class MIMG_Load_Helper <bits<7> op, string asm> : MIMG <
  op,
  (outs VReg_128:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, VReg_32:$vaddr,
       GPR4Align<SReg_256>:$srsrc, GPR4Align<SReg_128>:$ssamp),
  asm,
  []> {
  let mayLoad = 1;
  let mayStore = 0;
}

include "SIInstructions.td"
