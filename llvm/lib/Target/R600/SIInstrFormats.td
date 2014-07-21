//===-- SIInstrFormats.td - SI Instruction Encodings ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SI Instruction format definitions.
//
//===----------------------------------------------------------------------===//

class InstSI <dag outs, dag ins, string asm, list<dag> pattern> :
    AMDGPUInst<outs, ins, asm, pattern>, PredicateControl {

  field bits<1> VM_CNT = 0;
  field bits<1> EXP_CNT = 0;
  field bits<1> LGKM_CNT = 0;
  field bits<1> MIMG = 0;
  field bits<1> SMRD = 0;
  field bits<1> VOP1 = 0;
  field bits<1> VOP2 = 0;
  field bits<1> VOP3 = 0;
  field bits<1> VOPC = 0;
  field bits<1> SALU = 0;

  let TSFlags{0} = VM_CNT;
  let TSFlags{1} = EXP_CNT;
  let TSFlags{2} = LGKM_CNT;
  let TSFlags{3} = MIMG;
  let TSFlags{4} = SMRD;
  let TSFlags{5} = VOP1;
  let TSFlags{6} = VOP2;
  let TSFlags{7} = VOP3;
  let TSFlags{8} = VOPC;
  let TSFlags{9} = SALU;
}

class Enc32 {

  field bits<32> Inst;
  int Size = 4;
}

class Enc64 {

  field bits<64> Inst;
  int Size = 8;
}

class VOP3Common <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let VOP3 = 1;

  int Size = 8;
}

//===----------------------------------------------------------------------===//
// Scalar operations
//===----------------------------------------------------------------------===//

class SOP1e <bits<8> op> : Enc32 {

  bits<7> SDST;
  bits<8> SSRC0;

  let Inst{7-0} = SSRC0;
  let Inst{15-8} = op;
  let Inst{22-16} = SDST;
  let Inst{31-23} = 0x17d; //encoding;
}

class SOP2e <bits<7> op> : Enc32 {

  bits<7> SDST;
  bits<8> SSRC0;
  bits<8> SSRC1;

  let Inst{7-0} = SSRC0;
  let Inst{15-8} = SSRC1;
  let Inst{22-16} = SDST;
  let Inst{29-23} = op;
  let Inst{31-30} = 0x2; // encoding
}

class SOPCe <bits<7> op> : Enc32 {

  bits<8> SSRC0;
  bits<8> SSRC1;

  let Inst{7-0} = SSRC0;
  let Inst{15-8} = SSRC1;
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17e;
}

class SOPKe <bits<5> op> : Enc32 {

  bits <7> SDST;
  bits <16> SIMM16;

  let Inst{15-0} = SIMM16;
  let Inst{22-16} = SDST;
  let Inst{27-23} = op;
  let Inst{31-28} = 0xb; //encoding
}

class SOPPe <bits<7> op> : Enc32 {

  bits <16> simm16;

  let Inst{15-0} = simm16;
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17f; // encoding
}

class SMRDe <bits<5> op, bits<1> imm> : Enc32 {

  bits<7> SDST;
  bits<7> SBASE;
  bits<8> OFFSET;

  let Inst{7-0} = OFFSET;
  let Inst{8} = imm;
  let Inst{14-9} = SBASE{6-1};
  let Inst{21-15} = SDST;
  let Inst{26-22} = op;
  let Inst{31-27} = 0x18; //encoding
}

class SOP1 <bits<8> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI<outs, ins, asm, pattern>, SOP1e <op> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
}

class SOP2 <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern>, SOP2e<op> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
}

class SOPC <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
  InstSI<outs, ins, asm, pattern>, SOPCe <op> {

  let DisableEncoding = "$dst";
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
}

class SOPK <bits<5> op, dag outs, dag ins, string asm, list<dag> pattern> :
   InstSI <outs, ins , asm, pattern>, SOPKe<op> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
}

class SOPP <bits<7> op, dag ins, string asm, list<dag> pattern> :
		InstSI <(outs), ins, asm, pattern >, SOPPe <op> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
}

class SMRD <bits<5> op, bits<1> imm, dag outs, dag ins, string asm,
            list<dag> pattern> : InstSI<outs, ins, asm, pattern>, SMRDe<op, imm> {

  let LGKM_CNT = 1;
  let SMRD = 1;
}

//===----------------------------------------------------------------------===//
// Vector ALU operations
//===----------------------------------------------------------------------===//

class VOP1e <bits<8> op> : Enc32 {

  bits<8> VDST;
  bits<9> SRC0;

  let Inst{8-0} = SRC0;
  let Inst{16-9} = op;
  let Inst{24-17} = VDST;
  let Inst{31-25} = 0x3f; //encoding
}

class VOP2e <bits<6> op> : Enc32 {

  bits<8> VDST;
  bits<9> SRC0;
  bits<8> VSRC1;

  let Inst{8-0} = SRC0;
  let Inst{16-9} = VSRC1;
  let Inst{24-17} = VDST;
  let Inst{30-25} = op;
  let Inst{31} = 0x0; //encoding
}

class VOP3e <bits<9> op> : Enc64 {

  bits<8> dst;
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
  bits<9> src2;
  bits<1> clamp;
  bits<2> omod;

  let Inst{7-0} = dst;
  let Inst{8} = src0_modifiers{1};
  let Inst{9} = src1_modifiers{1};
  let Inst{10} = src2_modifiers{1};
  let Inst{11} = clamp;
  let Inst{25-17} = op;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = src0;
  let Inst{49-41} = src1;
  let Inst{58-50} = src2;
  let Inst{60-59} = omod;
  let Inst{61} = src0_modifiers{0};
  let Inst{62} = src1_modifiers{0};
  let Inst{63} = src2_modifiers{0};
}

class VOP3be <bits<9> op> : Enc64 {

  bits<8> dst;
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
  bits<9> src2;
  bits<7> sdst;
  bits<2> omod;

  let Inst{7-0} = dst;
  let Inst{14-8} = sdst;
  let Inst{25-17} = op;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = src0;
  let Inst{49-41} = src1;
  let Inst{58-50} = src2;
  let Inst{60-59} = omod;
  let Inst{61} = src0_modifiers{0};
  let Inst{62} = src1_modifiers{0};
  let Inst{63} = src2_modifiers{0};
}

class VOPCe <bits<8> op> : Enc32 {

  bits<9> SRC0;
  bits<8> VSRC1;

  let Inst{8-0} = SRC0;
  let Inst{16-9} = VSRC1;
  let Inst{24-17} = op;
  let Inst{31-25} = 0x3e;
}

class VINTRPe <bits<2> op> : Enc32 {

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
}

class DSe <bits<8> op> : Enc64 {

  bits<8> vdst;
  bits<1> gds;
  bits<8> addr;
  bits<8> data0;
  bits<8> data1;
  bits<8> offset0;
  bits<8> offset1;

  let Inst{7-0} = offset0;
  let Inst{15-8} = offset1;
  let Inst{17} = gds;
  let Inst{25-18} = op;
  let Inst{31-26} = 0x36; //encoding
  let Inst{39-32} = addr;
  let Inst{47-40} = data0;
  let Inst{55-48} = data1;
  let Inst{63-56} = vdst;
}

class MUBUFe <bits<7> op> : Enc64 {

  bits<12> offset;
  bits<1> offen;
  bits<1> idxen;
  bits<1> glc;
  bits<1> addr64;
  bits<1> lds;
  bits<8> vaddr;
  bits<8> vdata;
  bits<7> srsrc;
  bits<1> slc;
  bits<1> tfe;
  bits<8> soffset;

  let Inst{11-0} = offset;
  let Inst{12} = offen;
  let Inst{13} = idxen;
  let Inst{14} = glc;
  let Inst{15} = addr64;
  let Inst{16} = lds;
  let Inst{24-18} = op;
  let Inst{31-26} = 0x38; //encoding
  let Inst{39-32} = vaddr;
  let Inst{47-40} = vdata;
  let Inst{52-48} = srsrc{6-2};
  let Inst{54} = slc;
  let Inst{55} = tfe;
  let Inst{63-56} = soffset;
}

class MTBUFe <bits<3> op> : Enc64 {

  bits<8> VDATA;
  bits<12> OFFSET;
  bits<1> OFFEN;
  bits<1> IDXEN;
  bits<1> GLC;
  bits<1> ADDR64;
  bits<4> DFMT;
  bits<3> NFMT;
  bits<8> VADDR;
  bits<7> SRSRC;
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
  let Inst{52-48} = SRSRC{6-2};
  let Inst{54} = SLC;
  let Inst{55} = TFE;
  let Inst{63-56} = SOFFSET;
}

class MIMGe <bits<7> op> : Enc64 {

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
  bits<7> SRSRC;
  bits<7> SSAMP;

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
  let Inst{52-48} = SRSRC{6-2};
  let Inst{57-53} = SSAMP{6-2};
}

class EXPe : Enc64 {

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
}

let Uses = [EXEC] in {

class VOP1 <bits<8> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern>, VOP1e<op> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let VOP1 = 1;
}

class VOP2 <bits<6> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern>, VOP2e<op> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let VOP2 = 1;
}

class VOP3 <bits<9> op, dag outs, dag ins, string asm, list<dag> pattern> :
    VOP3Common <outs, ins, asm, pattern>, VOP3e<op>;

class VOP3b <bits<9> op, dag outs, dag ins, string asm, list<dag> pattern> :
    VOP3Common <outs, ins, asm, pattern>, VOP3be<op>;

class VOPC <bits<8> op, dag ins, string asm, list<dag> pattern> :
    InstSI <(outs VCCReg:$dst), ins, asm, pattern>, VOPCe <op> {

  let DisableEncoding = "$dst";
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let VOPC = 1;
}

class VINTRP <bits <2> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern>, VINTRPe<op> {

  let neverHasSideEffects = 1;
  let mayLoad = 1;
  let mayStore = 0;
}

} // End Uses = [EXEC]

//===----------------------------------------------------------------------===//
// Vector I/O operations
//===----------------------------------------------------------------------===//

let Uses = [EXEC] in {

class DS <bits<8> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> , DSe<op> {

  let LGKM_CNT = 1;
}

class MUBUF <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI<outs, ins, asm, pattern>, MUBUFe <op> {

  let VM_CNT = 1;
  let EXP_CNT = 1;

  let neverHasSideEffects = 1;
  let UseNamedOperandTable = 1;
}

class MTBUF <bits<3> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI<outs, ins, asm, pattern>, MTBUFe <op> {

  let VM_CNT = 1;
  let EXP_CNT = 1;

  let neverHasSideEffects = 1;
}

class MIMG <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern>, MIMGe <op> {

  let VM_CNT = 1;
  let EXP_CNT = 1;
  let MIMG = 1;
}

def EXP : InstSI<
  (outs),
  (ins i32imm:$en, i32imm:$tgt, i32imm:$compr, i32imm:$done, i32imm:$vm,
       VReg_32:$src0, VReg_32:$src1, VReg_32:$src2, VReg_32:$src3),
  "EXP $en, $tgt, $compr, $done, $vm, $src0, $src1, $src2, $src3",
  [] >, EXPe {

  let EXP_CNT = 1;
}

} // End Uses = [EXEC]
