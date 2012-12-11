//===-- SIInstrInfo.td - SI Instruction Encodings ---------*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SI DAG Profiles
//===----------------------------------------------------------------------===//
def SDTVCCBinaryOp : SDTypeProfile<1, 2, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisSameAs<1, 2>
]>;

//===----------------------------------------------------------------------===//
// SI DAG Nodes
//===----------------------------------------------------------------------===//

// and operation on 64-bit wide vcc
def SIsreg1_and : SDNode<"SIISD::VCC_AND", SDTVCCBinaryOp,
  [SDNPCommutative, SDNPAssociative]
>;

// Special bitcast node for sharing VCC register between VALU and SALU
def SIsreg1_bitcast : SDNode<"SIISD::VCC_BITCAST",
  SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisInt<1>]>
>;

// and operation on 64-bit wide vcc
def SIvcc_and : SDNode<"SIISD::VCC_AND", SDTVCCBinaryOp,
  [SDNPCommutative, SDNPAssociative]
>;

// Special bitcast node for sharing VCC register between VALU and SALU
def SIvcc_bitcast : SDNode<"SIISD::VCC_BITCAST",
  SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisInt<1>]>
>;

class InstSI <dag outs, dag ins, string asm, list<dag> pattern> :
    AMDGPUInst<outs, ins, asm, pattern> {

  field bits<4> EncodingType = 0;
  field bits<1> NeedWait = 0;

  let TSFlags{3-0} = EncodingType;
  let TSFlags{4} = NeedWait;

}

class Enc32 <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> {

  field bits<32> Inst;
}

class Enc64 <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> {

  field bits<64> Inst;
}

class SIOperand <ValueType vt, dag opInfo>: Operand <vt> {
  let EncoderMethod = "encodeOperand";
  let MIOperandInfo = opInfo;
}

def IMM16bit : ImmLeaf <
  i16,
  [{return isInt<16>(Imm);}]
>;

def IMM8bit : ImmLeaf <
  i32,
  [{return (int32_t)Imm >= 0 && (int32_t)Imm <= 0xff;}]
>;

def IMM12bit : ImmLeaf <
  i16,
  [{return (int16_t)Imm >= 0 && (int16_t)Imm <= 0xfff;}]
>;

def IMM32bitIn64bit : ImmLeaf <
  i64,
  [{return isInt<32>(Imm);}]
>;

class GPR4Align <RegisterClass rc> : Operand <vAny> {
  let EncoderMethod = "GPR4AlignEncode";
  let MIOperandInfo = (ops rc:$reg); 
}

class GPR2Align <RegisterClass rc, ValueType vt> : Operand <vt> {
  let EncoderMethod = "GPR2AlignEncode";
  let MIOperandInfo = (ops rc:$reg);
}

def SMRDmemrr : Operand<iPTR> {
  let MIOperandInfo = (ops SReg_64, SReg_32);
  let EncoderMethod = "GPR2AlignEncode";
}

def SMRDmemri : Operand<iPTR> {
  let MIOperandInfo = (ops SReg_64, i32imm);
  let EncoderMethod = "SMRDmemriEncode";
}

def ADDR_Reg     : ComplexPattern<i64, 2, "SelectADDRReg", [], []>;
def ADDR_Offset8 : ComplexPattern<i64, 2, "SelectADDR8BitOffset", [], []>;

let Uses = [EXEC] in {

def EXP : Enc64<
  (outs),
  (ins i32imm:$en, i32imm:$tgt, i32imm:$compr, i32imm:$done, i32imm:$vm,
       VReg_32:$src0, VReg_32:$src1, VReg_32:$src2, VReg_32:$src3),
  "EXP $en, $tgt, $compr, $done, $vm, $src0, $src1, $src2, $src3",
  [] > {

  bits<4> EN;
  bits<6> TGT;
  bits<1> COMPR;
  bits<1> DONE;
  bits<1> VM;
  bits<8> VSRC0;
  bits<8> VSRC1;
  bits<8> VSRC2;
  bits<8> VSRC3;

  let Inst{3-0} = EN;
  let Inst{9-4} = TGT;
  let Inst{10} = COMPR;
  let Inst{11} = DONE;
  let Inst{12} = VM;
  let Inst{31-26} = 0x3e;
  let Inst{39-32} = VSRC0;
  let Inst{47-40} = VSRC1;
  let Inst{55-48} = VSRC2;
  let Inst{63-56} = VSRC3;
  let EncodingType = 0; //SIInstrEncodingType::EXP

  let NeedWait = 1;
  let usesCustomInserter = 1;
}

class MIMG <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc64 <outs, ins, asm, pattern> {

  bits<8> VDATA;
  bits<4> DMASK;
  bits<1> UNORM;
  bits<1> GLC;
  bits<1> DA;
  bits<1> R128;
  bits<1> TFE;
  bits<1> LWE;
  bits<1> SLC;
  bits<8> VADDR;
  bits<5> SRSRC;
  bits<5> SSAMP; 

  let Inst{11-8} = DMASK;
  let Inst{12} = UNORM;
  let Inst{13} = GLC;
  let Inst{14} = DA;
  let Inst{15} = R128;
  let Inst{16} = TFE;
  let Inst{17} = LWE;
  let Inst{24-18} = op;
  let Inst{25} = SLC;
  let Inst{31-26} = 0x3c;
  let Inst{39-32} = VADDR;
  let Inst{47-40} = VDATA;
  let Inst{52-48} = SRSRC;
  let Inst{57-53} = SSAMP;

  let EncodingType = 2; //SIInstrEncodingType::MIMG

  let NeedWait = 1;
  let usesCustomInserter = 1;
}

class MTBUF <bits<3> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc64<outs, ins, asm, pattern> {

  bits<8> VDATA;
  bits<12> OFFSET;
  bits<1> OFFEN;
  bits<1> IDXEN;
  bits<1> GLC;
  bits<1> ADDR64;
  bits<4> DFMT;
  bits<3> NFMT;
  bits<8> VADDR;
  bits<5> SRSRC;
  bits<1> SLC;
  bits<1> TFE;
  bits<8> SOFFSET;

  let Inst{11-0} = OFFSET;
  let Inst{12} = OFFEN;
  let Inst{13} = IDXEN;
  let Inst{14} = GLC;
  let Inst{15} = ADDR64;
  let Inst{18-16} = op;
  let Inst{22-19} = DFMT;
  let Inst{25-23} = NFMT;
  let Inst{31-26} = 0x3a; //encoding
  let Inst{39-32} = VADDR;
  let Inst{47-40} = VDATA;
  let Inst{52-48} = SRSRC;
  let Inst{54} = SLC;
  let Inst{55} = TFE;
  let Inst{63-56} = SOFFSET;
  let EncodingType = 3; //SIInstrEncodingType::MTBUF

  let NeedWait = 1;
  let usesCustomInserter = 1;
  let neverHasSideEffects = 1;
}

class MUBUF <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc64<outs, ins, asm, pattern> {

  bits<8> VDATA;
  bits<12> OFFSET;
  bits<1> OFFEN;
  bits<1> IDXEN;
  bits<1> GLC;
  bits<1> ADDR64;
  bits<1> LDS;
  bits<8> VADDR;
  bits<5> SRSRC;
  bits<1> SLC;
  bits<1> TFE;
  bits<8> SOFFSET;

  let Inst{11-0} = OFFSET;
  let Inst{12} = OFFEN;
  let Inst{13} = IDXEN;
  let Inst{14} = GLC;
  let Inst{15} = ADDR64;
  let Inst{16} = LDS;
  let Inst{24-18} = op;
  let Inst{31-26} = 0x38; //encoding
  let Inst{39-32} = VADDR;
  let Inst{47-40} = VDATA;
  let Inst{52-48} = SRSRC;
  let Inst{54} = SLC;
  let Inst{55} = TFE;
  let Inst{63-56} = SOFFSET;
  let EncodingType = 4; //SIInstrEncodingType::MUBUF

  let NeedWait = 1;
  let usesCustomInserter = 1;
  let neverHasSideEffects = 1;
}

} // End Uses = [EXEC]

class SMRD <bits<5> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc32<outs, ins, asm, pattern> {

  bits<7> SDST;
  bits<15> PTR;
  bits<8> OFFSET = PTR{7-0};
  bits<1> IMM    = PTR{8};
  bits<6> SBASE  = PTR{14-9};
  
  let Inst{7-0} = OFFSET;
  let Inst{8} = IMM;
  let Inst{14-9} = SBASE;
  let Inst{21-15} = SDST;
  let Inst{26-22} = op;
  let Inst{31-27} = 0x18; //encoding
  let EncodingType = 5; //SIInstrEncodingType::SMRD

  let NeedWait = 1;
  let usesCustomInserter = 1;
}

class SOP1 <bits<8> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc32<outs, ins, asm, pattern> {

  bits<7> SDST;
  bits<8> SSRC0;

  let Inst{7-0} = SSRC0;
  let Inst{15-8} = op;
  let Inst{22-16} = SDST;
  let Inst{31-23} = 0x17d; //encoding;
  let EncodingType = 6; //SIInstrEncodingType::SOP1

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

class SOP2 <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc32 <outs, ins, asm, pattern> {
  
  bits<7> SDST;
  bits<8> SSRC0;
  bits<8> SSRC1;

  let Inst{7-0} = SSRC0;
  let Inst{15-8} = SSRC1;
  let Inst{22-16} = SDST;
  let Inst{29-23} = op;
  let Inst{31-30} = 0x2; // encoding
  let EncodingType = 7; // SIInstrEncodingType::SOP2  

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

class SOPC <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
  Enc32<outs, ins, asm, pattern> {

  bits<8> SSRC0;
  bits<8> SSRC1;

  let Inst{7-0} = SSRC0;
  let Inst{15-8} = SSRC1;
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17e;
  let EncodingType = 8; // SIInstrEncodingType::SOPC

  let DisableEncoding = "$dst";
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

class SOPK <bits<5> op, dag outs, dag ins, string asm, list<dag> pattern> :
   Enc32 <outs, ins , asm, pattern> {

  bits <7> SDST;
  bits <16> SIMM16;
  
  let Inst{15-0} = SIMM16;
  let Inst{22-16} = SDST;
  let Inst{27-23} = op;
  let Inst{31-28} = 0xb; //encoding
  let EncodingType = 9; // SIInstrEncodingType::SOPK

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

class SOPP <bits<7> op, dag ins, string asm, list<dag> pattern> : Enc32 <
  (outs),
  ins,
  asm,
  pattern > {

  bits <16> SIMM16;

  let Inst{15-0} = SIMM16;
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17f; // encoding
  let EncodingType = 10; // SIInstrEncodingType::SOPP

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}
    
let Uses = [EXEC] in {

class VINTRP <bits <2> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc32 <outs, ins, asm, pattern> {

  bits<8> VDST;
  bits<8> VSRC;
  bits<2> ATTRCHAN;
  bits<6> ATTR;

  let Inst{7-0} = VSRC;
  let Inst{9-8} = ATTRCHAN;
  let Inst{15-10} = ATTR;
  let Inst{17-16} = op;
  let Inst{25-18} = VDST;
  let Inst{31-26} = 0x32; // encoding
  let EncodingType = 11; // SIInstrEncodingType::VINTRP

  let neverHasSideEffects = 1;
  let mayLoad = 1;
  let mayStore = 0;
}

class VOP1 <bits<8> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc32 <outs, ins, asm, pattern> {

  bits<8> VDST;
  bits<9> SRC0;
  
  let Inst{8-0} = SRC0;
  let Inst{16-9} = op;
  let Inst{24-17} = VDST;
  let Inst{31-25} = 0x3f; //encoding
  
  let EncodingType = 12; // SIInstrEncodingType::VOP1
  let PostEncoderMethod = "VOPPostEncode";

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

class VOP2 <bits<6> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc32 <outs, ins, asm, pattern> {

  bits<8> VDST;
  bits<9> SRC0;
  bits<8> VSRC1;
  
  let Inst{8-0} = SRC0;
  let Inst{16-9} = VSRC1;
  let Inst{24-17} = VDST;
  let Inst{30-25} = op;
  let Inst{31} = 0x0; //encoding
  
  let EncodingType = 13; // SIInstrEncodingType::VOP2
  let PostEncoderMethod = "VOPPostEncode";

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

class VOP3 <bits<9> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc64 <outs, ins, asm, pattern> {

  bits<8> VDST;
  bits<9> SRC0;
  bits<9> SRC1;
  bits<9> SRC2;
  bits<3> ABS; 
  bits<1> CLAMP;
  bits<2> OMOD;
  bits<3> NEG;

  let Inst{7-0} = VDST;
  let Inst{10-8} = ABS;
  let Inst{11} = CLAMP;
  let Inst{25-17} = op;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = SRC0;
  let Inst{49-41} = SRC1;
  let Inst{58-50} = SRC2;
  let Inst{60-59} = OMOD;
  let Inst{63-61} = NEG;
  
  let EncodingType = 14; // SIInstrEncodingType::VOP3
  let PostEncoderMethod = "VOPPostEncode";

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

class VOP3b <bits<9> op, dag outs, dag ins, string asm, list<dag> pattern> :
    Enc64 <outs, ins, asm, pattern> {

  bits<8> VDST;
  bits<9> SRC0;
  bits<9> SRC1;
  bits<9> SRC2;
  bits<7> SDST;
  bits<2> OMOD;
  bits<3> NEG;

  let Inst{7-0} = VDST;
  let Inst{14-8} = SDST;
  let Inst{25-17} = op;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = SRC0;
  let Inst{49-41} = SRC1;
  let Inst{58-50} = SRC2;
  let Inst{60-59} = OMOD;
  let Inst{63-61} = NEG;

  let EncodingType = 14; // SIInstrEncodingType::VOP3
  let PostEncoderMethod = "VOPPostEncode";

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

class VOPC <bits<8> op, dag ins, string asm, list<dag> pattern> :
    Enc32 <(outs VCCReg:$dst), ins, asm, pattern> {

  bits<9> SRC0;
  bits<8> VSRC1;

  let Inst{8-0} = SRC0;
  let Inst{16-9} = VSRC1;
  let Inst{24-17} = op;
  let Inst{31-25} = 0x3e;
 
  let EncodingType = 15; //SIInstrEncodingType::VOPC
  let PostEncoderMethod = "VOPPostEncode";
  let DisableEncoding = "$dst";
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

} // End Uses = [EXEC]

class MIMG_Load_Helper <bits<7> op, string asm> : MIMG <
  op,
  (outs VReg_128:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, VReg_128:$vaddr,
       GPR4Align<SReg_256>:$srsrc, GPR4Align<SReg_128>:$ssamp),
  asm,
  []> {
  let mayLoad = 1;
  let mayStore = 0;
}

class MUBUF_Load_Helper <bits<7> op, string asm, RegisterClass regClass> : MUBUF <
  op,
  (outs regClass:$dst),
  (ins i16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc, i1imm:$addr64,
       i1imm:$lds, VReg_32:$vaddr, GPR4Align<SReg_128>:$srsrc, i1imm:$slc,
       i1imm:$tfe, SReg_32:$soffset),
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
       i1imm:$slc, i1imm:$tfe, SReg_32:$soffset),
  asm,
  []> {
  let mayLoad = 1;
  let mayStore = 0;
}

class MTBUF_Store_Helper <bits<3> op, string asm, RegisterClass regClass> : MTBUF <
  op,
  (outs),
  (ins regClass:$vdata, i16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc,
   i1imm:$addr64, i8imm:$dfmt, i8imm:$nfmt, VReg_32:$vaddr,
   GPR4Align<SReg_128>:$srsrc, i1imm:$slc, i1imm:$tfe, SReg_32:$soffset),
  asm,
  []> {
  let mayStore = 1;
  let mayLoad = 0;
}

multiclass SMRD_Helper <bits<5> op, string asm, RegisterClass dstClass,
                        ValueType vt> {
  def _IMM : SMRD <
              op,
              (outs dstClass:$dst),
              (ins SMRDmemri:$src0),
              asm,
              [(set (vt dstClass:$dst), (constant_load ADDR_Offset8:$src0))]
  >;

  def _SGPR : SMRD <
              op,
              (outs dstClass:$dst),
              (ins SMRDmemrr:$src0),
              asm,
              [(set (vt dstClass:$dst), (constant_load ADDR_Reg:$src0))]
  >;
}

multiclass SMRD_32 <bits<5> op, string asm, RegisterClass dstClass> {
  defm _F32 : SMRD_Helper <op, asm, dstClass, f32>;
  defm _I32 : SMRD_Helper <op, asm, dstClass, i32>;
}

include "SIInstrFormats.td"
include "SIInstructions.td"
