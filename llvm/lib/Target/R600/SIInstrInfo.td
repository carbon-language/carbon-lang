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

def as_i16imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), MVT::i16);
}]>;

def IMM12bit : PatLeaf <(imm),
  [{return isUInt<12>(N->getZExtValue());}]
>;

class InlineImm <ValueType vt> : PatLeaf <(vt imm), [{
  return
    (*(const SITargetLowering *)getTargetLowering()).analyzeImmediate(N) == 0;
}]>;

//===----------------------------------------------------------------------===//
// SI assembler operands
//===----------------------------------------------------------------------===//

def SIOperand {
  int ZERO = 0x80;
  int VCC = 0x6A;
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

class SOP1_32 <bits<8> op, string opName, list<dag> pattern> : SOP1 <
  op, (outs SReg_32:$dst), (ins SSrc_32:$src0),
  opName#" $dst, $src0", pattern
>;

class SOP1_64 <bits<8> op, string opName, list<dag> pattern> : SOP1 <
  op, (outs SReg_64:$dst), (ins SSrc_64:$src0),
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

class SOPC_32 <bits<7> op, string opName, list<dag> pattern> : SOPC <
  op, (outs SCCReg:$dst), (ins SSrc_32:$src0, SSrc_32:$src1),
  opName#" $dst, $src0, $src1", pattern
>;

class SOPC_64 <bits<7> op, string opName, list<dag> pattern> : SOPC <
  op, (outs SCCReg:$dst), (ins SSrc_64:$src0, SSrc_64:$src1),
  opName#" $dst, $src0, $src1", pattern
>;

class SOPK_32 <bits<5> op, string opName, list<dag> pattern> : SOPK <
  op, (outs SReg_32:$dst), (ins i16imm:$src0),
  opName#" $dst, $src0", pattern
>;

class SOPK_64 <bits<5> op, string opName, list<dag> pattern> : SOPK <
  op, (outs SReg_64:$dst), (ins i16imm:$src0),
  opName#" $dst, $src0", pattern
>;

multiclass SMRD_Helper <bits<5> op, string asm, RegisterClass baseClass,
                        RegisterClass dstClass> {
  def _IMM : SMRD <
    op, 1, (outs dstClass:$dst),
    (ins baseClass:$sbase, i32imm:$offset),
    asm#" $dst, $sbase, $offset", []
  >;

  def _SGPR : SMRD <
    op, 0, (outs dstClass:$dst),
    (ins baseClass:$sbase, SReg_32:$soff),
    asm#" $dst, $sbase, $soff", []
  >;
}

//===----------------------------------------------------------------------===//
// Vector ALU classes
//===----------------------------------------------------------------------===//

class VOP <string opName> {
  string OpName = opName;
}

class VOP2_REV <string revOp, bit isOrig> {
  string RevOp = revOp;
  bit IsOrig = isOrig;
}

multiclass VOP1_Helper <bits<8> op, RegisterClass drc, RegisterClass src,
                        string opName, list<dag> pattern> {

  def _e32 : VOP1 <
    op, (outs drc:$dst), (ins src:$src0),
    opName#"_e32 $dst, $src0", pattern
  >, VOP <opName>;

  def _e64 : VOP3 <
    {1, 1, op{6}, op{5}, op{4}, op{3}, op{2}, op{1}, op{0}},
    (outs drc:$dst),
    (ins src:$src0,
         i32imm:$abs, i32imm:$clamp,
         i32imm:$omod, i32imm:$neg),
    opName#"_e64 $dst, $src0, $abs, $clamp, $omod, $neg", []
  >, VOP <opName> {
    let src1 = SIOperand.ZERO;
    let src2 = SIOperand.ZERO;
  }
}

multiclass VOP1_32 <bits<8> op, string opName, list<dag> pattern>
  : VOP1_Helper <op, VReg_32, VSrc_32, opName, pattern>;

multiclass VOP1_64 <bits<8> op, string opName, list<dag> pattern>
  : VOP1_Helper <op, VReg_64, VSrc_64, opName, pattern>;

multiclass VOP2_Helper <bits<6> op, RegisterClass vrc, RegisterClass arc,
                        string opName, list<dag> pattern, string revOp> {
  def _e32 : VOP2 <
    op, (outs vrc:$dst), (ins arc:$src0, vrc:$src1),
    opName#"_e32 $dst, $src0, $src1", pattern
  >, VOP <opName>, VOP2_REV<revOp#"_e32", !eq(revOp, opName)>;

  def _e64 : VOP3 <
    {1, 0, 0, op{5}, op{4}, op{3}, op{2}, op{1}, op{0}},
    (outs vrc:$dst),
    (ins arc:$src0, arc:$src1,
         i32imm:$abs, i32imm:$clamp,
         i32imm:$omod, i32imm:$neg),
    opName#"_e64 $dst, $src0, $src1, $abs, $clamp, $omod, $neg", []
  >, VOP <opName>, VOP2_REV<revOp#"_e64", !eq(revOp, opName)> {
    let src2 = SIOperand.ZERO;
  }
}

multiclass VOP2_32 <bits<6> op, string opName, list<dag> pattern,
                    string revOp = opName>
  : VOP2_Helper <op, VReg_32, VSrc_32, opName, pattern, revOp>;

multiclass VOP2_64 <bits<6> op, string opName, list<dag> pattern,
                    string revOp = opName>
  : VOP2_Helper <op, VReg_64, VSrc_64, opName, pattern, revOp>;

multiclass VOP2b_32 <bits<6> op, string opName, list<dag> pattern,
                     string revOp = opName> {

  def _e32 : VOP2 <
    op, (outs VReg_32:$dst), (ins VSrc_32:$src0, VReg_32:$src1),
    opName#"_e32 $dst, $src0, $src1", pattern
  >, VOP <opName>, VOP2_REV<revOp#"_e32", !eq(revOp, opName)>;

  def _e64 : VOP3b <
    {1, 0, 0, op{5}, op{4}, op{3}, op{2}, op{1}, op{0}},
    (outs VReg_32:$dst),
    (ins VSrc_32:$src0, VSrc_32:$src1,
         i32imm:$abs, i32imm:$clamp,
         i32imm:$omod, i32imm:$neg),
    opName#"_e64 $dst, $src0, $src1, $abs, $clamp, $omod, $neg", []
  >, VOP <opName>, VOP2_REV<revOp#"_e64", !eq(revOp, opName)> {
    let src2 = SIOperand.ZERO;
    /* the VOP2 variant puts the carry out into VCC, the VOP3 variant
       can write it into any SGPR. We currently don't use the carry out,
       so for now hardcode it to VCC as well */
    let sdst = SIOperand.VCC;
  }
}

multiclass VOPC_Helper <bits<8> op, RegisterClass vrc, RegisterClass arc,
                        string opName, ValueType vt, PatLeaf cond> {

  def _e32 : VOPC <
    op, (ins arc:$src0, vrc:$src1),
    opName#"_e32 $dst, $src0, $src1", []
  >, VOP <opName>;

  def _e64 : VOP3 <
    {0, op{7}, op{6}, op{5}, op{4}, op{3}, op{2}, op{1}, op{0}},
    (outs SReg_64:$dst),
    (ins arc:$src0, arc:$src1,
         InstFlag:$abs, InstFlag:$clamp,
         InstFlag:$omod, InstFlag:$neg),
    opName#"_e64 $dst, $src0, $src1, $abs, $clamp, $omod, $neg",
    !if(!eq(!cast<string>(cond), "COND_NULL"), []<dag>,
      [(set SReg_64:$dst, (i1 (setcc (vt arc:$src0), arc:$src1, cond)))]
    )
  >, VOP <opName> {
    let src2 = SIOperand.ZERO;
  }
}

multiclass VOPC_32 <bits<8> op, string opName,
  ValueType vt = untyped, PatLeaf cond = COND_NULL>
  : VOPC_Helper <op, VReg_32, VSrc_32, opName, vt, cond>;

multiclass VOPC_64 <bits<8> op, string opName,
  ValueType vt = untyped, PatLeaf cond = COND_NULL>
  : VOPC_Helper <op, VReg_64, VSrc_64, opName, vt, cond>;

class VOP3_32 <bits<9> op, string opName, list<dag> pattern> : VOP3 <
  op, (outs VReg_32:$dst),
  (ins VSrc_32:$src0, VSrc_32:$src1, VSrc_32:$src2,
   InstFlag:$abs, InstFlag:$clamp, InstFlag:$omod, InstFlag:$neg),
  opName#" $dst, $src0, $src1, $src2, $abs, $clamp, $omod, $neg", pattern
>, VOP <opName>;

class VOP3_64_Shift <bits <9> op, string opName, list<dag> pattern> : VOP3 <
  op, (outs VReg_64:$dst),
  (ins VSrc_64:$src0, VSrc_32:$src1),
  opName#" $dst, $src0, $src1", pattern
>, VOP <opName> {

  let src2 = SIOperand.ZERO;
  let abs = 0;
  let clamp = 0;
  let omod = 0;
  let neg = 0;
}

class VOP3_64 <bits<9> op, string opName, list<dag> pattern> : VOP3 <
  op, (outs VReg_64:$dst),
  (ins VSrc_64:$src0, VSrc_64:$src1, VSrc_64:$src2,
   InstFlag:$abs, InstFlag:$clamp, InstFlag:$omod, InstFlag:$neg),
  opName#" $dst, $src0, $src1, $src2, $abs, $clamp, $omod, $neg", pattern
>, VOP <opName>;

//===----------------------------------------------------------------------===//
// Vector I/O classes
//===----------------------------------------------------------------------===//

class MTBUF_Store_Helper <bits<3> op, string asm, RegisterClass regClass> : MTBUF <
  op,
  (outs),
  (ins regClass:$vdata, i16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc,
   i1imm:$addr64, i8imm:$dfmt, i8imm:$nfmt, VReg_32:$vaddr,
   SReg_128:$srsrc, i1imm:$slc, i1imm:$tfe, SSrc_32:$soffset),
  asm#" $vdata, $offset, $offen, $idxen, $glc, $addr64, $dfmt,"
     #" $nfmt, $vaddr, $srsrc, $slc, $tfe, $soffset",
  []> {
  let mayStore = 1;
  let mayLoad = 0;
}

multiclass MUBUF_Load_Helper <bits<7> op, string asm, RegisterClass regClass> {

  let glc = 0, lds = 0, slc = 0, tfe = 0, soffset = 128 /* ZERO */,
                                          mayLoad = 1 in {

  let offen = 1, idxen = 0, addr64 = 0, offset = 0 in {
    def _OFFEN  : MUBUF <op, (outs regClass:$vdata),
                         (ins SReg_128:$srsrc, VReg_32:$vaddr),
                         asm#" $vdata, $srsrc + $vaddr", []>;
  }

  let offen = 0, idxen = 1, addr64 = 0 in {
    def _IDXEN  : MUBUF <op, (outs regClass:$vdata),
                         (ins SReg_128:$srsrc, VReg_32:$vaddr, i16imm:$offset),
                         asm#" $vdata, $srsrc[$vaddr] + $offset", []>;
  }

  let offen = 0, idxen = 0, addr64 = 1 in {
    def _ADDR64 : MUBUF <op, (outs regClass:$vdata),
                         (ins SReg_128:$srsrc, VReg_64:$vaddr, i16imm:$offset),
                         asm#" $vdata, $srsrc + $vaddr + $offset", []>;
  }
  }
}

class MUBUF_Store_Helper <bits<7> op, string name, RegisterClass vdataClass,
                         ValueType VT> :
    MUBUF <op, (outs), (ins vdataClass:$vdata, SReg_128:$srsrc, VReg_64:$vaddr, i16imm:$offset),
          name#" $vdata, $srsrc + $vaddr + $offset",
         []> {

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

class MTBUF_Load_Helper <bits<3> op, string asm, RegisterClass regClass> : MTBUF <
  op,
  (outs regClass:$dst),
  (ins i16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc, i1imm:$addr64,
       i8imm:$dfmt, i8imm:$nfmt, VReg_32:$vaddr, SReg_128:$srsrc,
       i1imm:$slc, i1imm:$tfe, SSrc_32:$soffset),
  asm#" $dst, $offset, $offen, $idxen, $glc, $addr64, $dfmt,"
     #" $nfmt, $vaddr, $srsrc, $slc, $tfe, $soffset",
  []> {
  let mayLoad = 1;
  let mayStore = 0;
}

class MIMG_NoSampler_Helper <bits<7> op, string asm> : MIMG <
  op,
  (outs VReg_128:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, unknown:$vaddr,
       SReg_256:$srsrc),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc",
  []> {
  let SSAMP = 0;
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
}

class MIMG_Sampler_Helper <bits<7> op, string asm> : MIMG <
  op,
  (outs VReg_128:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, unknown:$vaddr,
       SReg_256:$srsrc, SReg_128:$ssamp),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc, $ssamp",
  []> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
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

// Maps an original opcode to its commuted version
def getCommuteRev : InstrMapping {
  let FilterClass = "VOP2_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

// Maps an commuted opcode to its original version
def getCommuteOrig : InstrMapping {
  let FilterClass = "VOP2_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Test if the supplied opcode is an MIMG instruction
def isMIMG : InstrMapping {
  let FilterClass = "MIMG";
  let RowFields = ["Inst"];
  let ColFields = ["Size"];
  let KeyCol = ["8"];
  let ValueCols = [["8"]];
}

include "SIInstructions.td"
