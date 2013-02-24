//===- AArch64InstrFormats.td - AArch64 Instruction Formats --*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file describes AArch64 instruction formats, down to the level of the
// instruction's overall class.
// ===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// A64 Instruction Format Definitions.
//===----------------------------------------------------------------------===//

// A64 is currently the only instruction set supported by the AArch64
// architecture.
class A64Inst<dag outs, dag ins, string asmstr, list<dag> patterns,
              InstrItinClass itin>
    : Instruction {
  // All A64 instructions are 32-bit. This field will be filled in
  // gradually going down the hierarchy.
  field bits<32> Inst;

  field bits<32> Unpredictable = 0;
  // SoftFail is the generic name for this field, but we alias it so
  // as to make it more obvious what it means in ARM-land.
  field bits<32> SoftFail = Unpredictable;

  // LLVM-level model of the AArch64/A64 distinction.
  let Namespace = "AArch64";
  let DecoderNamespace = "A64";
  let Size = 4;

  // Set the templated fields
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = patterns;
  let Itinerary = itin;
}

class PseudoInst<dag outs, dag ins, list<dag> patterns> : Instruction {
  let Namespace = "AArch64";

  let OutOperandList = outs;
  let InOperandList= ins;
  let Pattern = patterns;
  let isCodeGenOnly = 1;
  let isPseudo = 1;
}

// Represents a pseudo-instruction that represents a single A64 instruction for
// whatever reason, the eventual result will be a 32-bit real instruction.
class A64PseudoInst<dag outs, dag ins, list<dag> patterns>
  : PseudoInst<outs, ins, patterns> {
  let Size = 4;
}

// As above, this will be a single A64 instruction, but we can actually give the
// expansion in TableGen.
class A64PseudoExpand<dag outs, dag ins, list<dag> patterns, dag Result>
  : A64PseudoInst<outs, ins, patterns>,
    PseudoInstExpansion<Result>;


// First, some common cross-hierarchy register formats.

class A64InstRd<dag outs, dag ins, string asmstr,
                list<dag> patterns, InstrItinClass itin>
  : A64Inst<outs, ins, asmstr, patterns, itin> {
  bits<5> Rd;

  let Inst{4-0} = Rd;
}

class A64InstRt<dag outs, dag ins, string asmstr,
                list<dag> patterns, InstrItinClass itin>
  : A64Inst<outs, ins, asmstr, patterns, itin> {
  bits<5> Rt;

  let Inst{4-0} = Rt;
}


class A64InstRdn<dag outs, dag ins, string asmstr,
                 list<dag> patterns, InstrItinClass itin>
    : A64InstRd<outs, ins, asmstr, patterns, itin> {
  // Inherit rdt
  bits<5> Rn;

  let Inst{9-5} = Rn;
}

class A64InstRtn<dag outs, dag ins, string asmstr,
                list<dag> patterns, InstrItinClass itin>
    : A64InstRt<outs, ins, asmstr, patterns, itin> {
  // Inherit rdt
  bits<5> Rn;

  let Inst{9-5} = Rn;
}

// Instructions taking Rt,Rt2,Rn
class A64InstRtt2n<dag outs, dag ins, string asmstr,
                   list<dag> patterns, InstrItinClass itin>
  : A64InstRtn<outs, ins, asmstr, patterns, itin> {
  bits<5> Rt2;

  let Inst{14-10} = Rt2;
}

class A64InstRdnm<dag outs, dag ins, string asmstr,
                  list<dag> patterns, InstrItinClass itin>
  : A64InstRdn<outs, ins, asmstr, patterns, itin> {
  bits<5> Rm;

  let Inst{20-16} = Rm;
}

//===----------------------------------------------------------------------===//
//
// Actual A64 Instruction Formats
//

// Format for Add-subtract (extended register) instructions.
class A64I_addsubext<bit sf, bit op, bit S, bits<2> opt, bits<3> option,
                     dag outs, dag ins, string asmstr, list<dag> patterns,
                     InstrItinClass itin>
    : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
    bits<3> Imm3;

    let Inst{31} = sf;
    let Inst{30} = op;
    let Inst{29} = S;
    let Inst{28-24} = 0b01011;
    let Inst{23-22} = opt;
    let Inst{21} = 0b1;
    // Rm inherited in 20-16
    let Inst{15-13} = option;
    let Inst{12-10} = Imm3;
    // Rn inherited in 9-5
    // Rd inherited in 4-0
}

// Format for Add-subtract (immediate) instructions.
class A64I_addsubimm<bit sf, bit op, bit S, bits<2> shift,
                     dag outs, dag ins, string asmstr,
                     list<dag> patterns, InstrItinClass itin>
  : A64InstRdn<outs, ins, asmstr, patterns, itin> {
  bits<12> Imm12;

  let Inst{31} = sf;
  let Inst{30} = op;
  let Inst{29} = S;
  let Inst{28-24} = 0b10001;
  let Inst{23-22} = shift;
  let Inst{21-10} = Imm12;
}

// Format for Add-subtract (shifted register) instructions.
class A64I_addsubshift<bit sf, bit op, bit S, bits<2> shift,
                       dag outs, dag ins, string asmstr, list<dag> patterns,
                       InstrItinClass itin>
    : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
    bits<6> Imm6;

    let Inst{31} = sf;
    let Inst{30} = op;
    let Inst{29} = S;
    let Inst{28-24} = 0b01011;
    let Inst{23-22} = shift;
    let Inst{21} = 0b0;
    // Rm inherited in 20-16
    let Inst{15-10} = Imm6;
    // Rn inherited in 9-5
    // Rd inherited in 4-0
}

// Format for Add-subtract (with carry) instructions.
class A64I_addsubcarry<bit sf, bit op, bit S, bits<6> opcode2,
                       dag outs, dag ins, string asmstr, list<dag> patterns,
                       InstrItinClass itin>
    : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
    let Inst{31} = sf;
    let Inst{30} = op;
    let Inst{29} = S;
    let Inst{28-21} = 0b11010000;
    // Rm inherited in 20-16
    let Inst{15-10} = opcode2;
    // Rn inherited in 9-5
    // Rd inherited in 4-0
}


// Format for Bitfield instructions
class A64I_bitfield<bit sf, bits<2> opc, bit n,
                    dag outs, dag ins, string asmstr,
                    list<dag> patterns, InstrItinClass itin>
  : A64InstRdn<outs, ins, asmstr, patterns, itin> {
  bits<6> ImmR;
  bits<6> ImmS;

  let Inst{31} = sf;
  let Inst{30-29} = opc;
  let Inst{28-23} = 0b100110;
  let Inst{22} = n;
  let Inst{21-16} = ImmR;
  let Inst{15-10} = ImmS;
  // Inherit Rn in 9-5
  // Inherit Rd in 4-0
}

// Format for compare and branch (immediate) instructions.
class A64I_cmpbr<bit sf, bit op,
                  dag outs, dag ins, string asmstr,
                  list<dag> patterns, InstrItinClass itin>
  : A64InstRt<outs, ins, asmstr, patterns, itin> {
  bits<19> Label;

  let Inst{31} = sf;
  let Inst{30-25} = 0b011010;
  let Inst{24} = op;
  let Inst{23-5} = Label;
  // Inherit Rt in 4-0
}

// Format for conditional branch (immediate) instructions.
class A64I_condbr<bit o1, bit o0,
                  dag outs, dag ins, string asmstr,
                  list<dag> patterns, InstrItinClass itin>
  : A64Inst<outs, ins, asmstr, patterns, itin> {
  bits<19> Label;
  bits<4> Cond;

  let Inst{31-25} = 0b0101010;
  let Inst{24} = o1;
  let Inst{23-5} = Label;
  let Inst{4} = o0;
  let Inst{3-0} = Cond;
}

// Format for conditional compare (immediate) instructions.
class A64I_condcmpimm<bit sf, bit op, bit o2, bit o3, bit s,
                      dag outs, dag ins, string asmstr,
                      list<dag> patterns, InstrItinClass itin>
  : A64Inst<outs, ins, asmstr, patterns, itin> {
  bits<5> Rn;
  bits<5> UImm5;
  bits<4> NZCVImm;
  bits<4> Cond;

  let Inst{31} = sf;
  let Inst{30} = op;
  let Inst{29} = s;
  let Inst{28-21} = 0b11010010;
  let Inst{20-16} = UImm5;
  let Inst{15-12} = Cond;
  let Inst{11} = 0b1;
  let Inst{10} = o2;
  let Inst{9-5} = Rn;
  let Inst{4} = o3;
  let Inst{3-0} = NZCVImm;
}

// Format for conditional compare (register) instructions.
class A64I_condcmpreg<bit sf, bit op, bit o2, bit o3, bit s,
                      dag outs, dag ins, string asmstr,
                      list<dag> patterns, InstrItinClass itin>
  : A64Inst<outs, ins, asmstr, patterns, itin> {
  bits<5> Rn;
  bits<5> Rm;
  bits<4> NZCVImm;
  bits<4> Cond;


  let Inst{31} = sf;
  let Inst{30} = op;
  let Inst{29} = s;
  let Inst{28-21} = 0b11010010;
  let Inst{20-16} = Rm;
  let Inst{15-12} = Cond;
  let Inst{11} = 0b0;
  let Inst{10} = o2;
  let Inst{9-5} = Rn;
  let Inst{4} = o3;
  let Inst{3-0} = NZCVImm;
}

// Format for conditional select instructions.
class A64I_condsel<bit sf, bit op, bit s, bits<2> op2,
                   dag outs, dag ins, string asmstr,
                   list<dag> patterns, InstrItinClass itin>
  : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
  bits<4> Cond;

  let Inst{31} = sf;
  let Inst{30} = op;
  let Inst{29} = s;
  let Inst{28-21} = 0b11010100;
  // Inherit Rm in 20-16
  let Inst{15-12} = Cond;
  let Inst{11-10} = op2;
  // Inherit Rn in 9-5
  // Inherit Rd in 4-0
}

// Format for data processing (1 source) instructions
class A64I_dp_1src<bit sf, bit S, bits<5> opcode2, bits<6> opcode,
                string asmstr, dag outs, dag ins,
                list<dag> patterns, InstrItinClass itin>
  : A64InstRdn<outs, ins, asmstr, patterns, itin> {
  let Inst{31} = sf;
  let Inst{30} = 0b1;
  let Inst{29} = S;
  let Inst{28-21} = 0b11010110;
  let Inst{20-16} = opcode2;
  let Inst{15-10} = opcode;
}

// Format for data processing (2 source) instructions
class A64I_dp_2src<bit sf, bits<6> opcode, bit S,
                string asmstr, dag outs, dag ins,
                list<dag> patterns, InstrItinClass itin>
  : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
  let Inst{31} = sf;
  let Inst{30} = 0b0;
  let Inst{29} = S;
  let Inst{28-21} = 0b11010110;
  let Inst{15-10} = opcode;
}

// Format for data-processing (3 source) instructions

class A64I_dp3<bit sf, bits<6> opcode,
               dag outs, dag ins, string asmstr,
               list<dag> patterns, InstrItinClass itin>
  : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
  bits<5> Ra;

  let Inst{31} = sf;
  let Inst{30-29} = opcode{5-4};
  let Inst{28-24} = 0b11011;
  let Inst{23-21} = opcode{3-1};
  // Inherits Rm in 20-16
  let Inst{15} = opcode{0};
  let Inst{14-10} = Ra;
  // Inherits Rn in 9-5
  // Inherits Rd in 4-0
}

// Format for exception generation instructions
class A64I_exception<bits<3> opc, bits<3> op2, bits<2> ll,
                     dag outs, dag ins, string asmstr,
                     list<dag> patterns, InstrItinClass itin>
  : A64Inst<outs, ins, asmstr, patterns, itin> {
  bits<16> UImm16;

  let Inst{31-24} = 0b11010100;
  let Inst{23-21} = opc;
  let Inst{20-5} = UImm16;
  let Inst{4-2} = op2;
  let Inst{1-0} = ll;
}

// Format for extract (immediate) instructions
class A64I_extract<bit sf, bits<3> op, bit n,
                   dag outs, dag ins, string asmstr,
                   list<dag> patterns, InstrItinClass itin>
  : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
  bits<6> LSB;

  let Inst{31} = sf;
  let Inst{30-29} = op{2-1};
  let Inst{28-23} = 0b100111;
  let Inst{22} = n;
  let Inst{21} = op{0};
  // Inherits Rm in bits 20-16
  let Inst{15-10} = LSB;
  // Inherits Rn in 9-5
  // Inherits Rd in 4-0
}

// Format for floating-point compare instructions.
class A64I_fpcmp<bit m, bit s, bits<2> type, bits<2> op, bits<5> opcode2,
                dag outs, dag ins, string asmstr,
                list<dag> patterns, InstrItinClass itin>
  : A64Inst<outs, ins, asmstr, patterns, itin> {
  bits<5> Rn;
  bits<5> Rm;

  let Inst{31} = m;
  let Inst{30} = 0b0;
  let Inst{29} = s;
  let Inst{28-24} = 0b11110;
  let Inst{23-22} = type;
  let Inst{21} = 0b1;
  let Inst{20-16} = Rm;
  let Inst{15-14} = op;
  let Inst{13-10} = 0b1000;
  let Inst{9-5} = Rn;
  let Inst{4-0} = opcode2;
}

// Format for floating-point conditional compare instructions.
class A64I_fpccmp<bit m, bit s, bits<2> type, bit op,
                 dag outs, dag ins, string asmstr,
                 list<dag> patterns, InstrItinClass itin>
  : A64InstRdn<outs, ins, asmstr, patterns, itin> {
  bits<5> Rn;
  bits<5> Rm;
  bits<4> NZCVImm;
  bits<4> Cond;

  let Inst{31} = m;
  let Inst{30} = 0b0;
  let Inst{29} = s;
  let Inst{28-24} = 0b11110;
  let Inst{23-22} = type;
  let Inst{21} = 0b1;
  let Inst{20-16} = Rm;
  let Inst{15-12} = Cond;
  let Inst{11-10} = 0b01;
  let Inst{9-5} = Rn;
  let Inst{4} = op;
  let Inst{3-0} = NZCVImm;
}

// Format for floating-point conditional select instructions.
class A64I_fpcondsel<bit m, bit s, bits<2> type,
                     dag outs, dag ins, string asmstr,
                     list<dag> patterns, InstrItinClass itin>
  : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
  bits<4> Cond;

  let Inst{31} = m;
  let Inst{30} = 0b0;
  let Inst{29} = s;
  let Inst{28-24} = 0b11110;
  let Inst{23-22} = type;
  let Inst{21} = 0b1;
  // Inherit Rm in 20-16
  let Inst{15-12} = Cond;
  let Inst{11-10} = 0b11;
  // Inherit Rn in 9-5
  // Inherit Rd in 4-0
}


// Format for floating-point data-processing (1 source) instructions.
class A64I_fpdp1<bit m, bit s, bits<2> type, bits<6> opcode,
                 dag outs, dag ins, string asmstr,
                 list<dag> patterns, InstrItinClass itin>
  : A64InstRdn<outs, ins, asmstr, patterns, itin> {
  let Inst{31} = m;
  let Inst{30} = 0b0;
  let Inst{29} = s;
  let Inst{28-24} = 0b11110;
  let Inst{23-22} = type;
  let Inst{21} = 0b1;
  let Inst{20-15} = opcode;
  let Inst{14-10} = 0b10000;
  // Inherit Rn in 9-5
  // Inherit Rd in 4-0
}

// Format for floating-point data-processing (2 sources) instructions.
class A64I_fpdp2<bit m, bit s, bits<2> type, bits<4> opcode,
                 dag outs, dag ins, string asmstr,
                 list<dag> patterns, InstrItinClass itin>
  : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
  let Inst{31} = m;
  let Inst{30} = 0b0;
  let Inst{29} = s;
  let Inst{28-24} = 0b11110;
  let Inst{23-22} = type;
  let Inst{21} = 0b1;
  // Inherit Rm in 20-16
  let Inst{15-12} = opcode;
  let Inst{11-10} = 0b10;
  // Inherit Rn in 9-5
  // Inherit Rd in 4-0
}

// Format for floating-point data-processing (3 sources) instructions.
class A64I_fpdp3<bit m, bit s, bits<2> type, bit o1, bit o0,
                 dag outs, dag ins, string asmstr,
                 list<dag> patterns, InstrItinClass itin>
  : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
  bits<5> Ra;

  let Inst{31} = m;
  let Inst{30} = 0b0;
  let Inst{29} = s;
  let Inst{28-24} = 0b11111;
  let Inst{23-22} = type;
  let Inst{21} = o1;
  // Inherit Rm in 20-16
  let Inst{15} = o0;
  let Inst{14-10} = Ra;
  // Inherit Rn in 9-5
  // Inherit Rd in 4-0
}

// Format for floating-point <-> fixed-point conversion instructions.
class A64I_fpfixed<bit sf, bit s, bits<2> type, bits<2> mode, bits<3> opcode,
                 dag outs, dag ins, string asmstr,
                 list<dag> patterns, InstrItinClass itin>
  : A64InstRdn<outs, ins, asmstr, patterns, itin> {
  bits<6> Scale;

  let Inst{31} = sf;
  let Inst{30} = 0b0;
  let Inst{29} = s;
  let Inst{28-24} = 0b11110;
  let Inst{23-22} = type;
  let Inst{21} = 0b0;
  let Inst{20-19} = mode;
  let Inst{18-16} = opcode;
  let Inst{15-10} = Scale;
  // Inherit Rn in 9-5
  // Inherit Rd in 4-0
}

// Format for floating-point <-> integer conversion instructions.
class A64I_fpint<bit sf, bit s, bits<2> type, bits<2> rmode, bits<3> opcode,
                 dag outs, dag ins, string asmstr,
                 list<dag> patterns, InstrItinClass itin>
  : A64InstRdn<outs, ins, asmstr, patterns, itin> {
  let Inst{31} = sf;
  let Inst{30} = 0b0;
  let Inst{29} = s;
  let Inst{28-24} = 0b11110;
  let Inst{23-22} = type;
  let Inst{21} = 0b1;
  let Inst{20-19} = rmode;
  let Inst{18-16} = opcode;
  let Inst{15-10} = 0b000000;
  // Inherit Rn in 9-5
  // Inherit Rd in 4-0
}


// Format for floating-point immediate instructions.
class A64I_fpimm<bit m, bit s, bits<2> type, bits<5> imm5,
                 dag outs, dag ins, string asmstr,
                 list<dag> patterns, InstrItinClass itin>
  : A64InstRd<outs, ins, asmstr, patterns, itin> {
  bits<8> Imm8;

  let Inst{31} = m;
  let Inst{30} = 0b0;
  let Inst{29} = s;
  let Inst{28-24} = 0b11110;
  let Inst{23-22} = type;
  let Inst{21} = 0b1;
  let Inst{20-13} = Imm8;
  let Inst{12-10} = 0b100;
  let Inst{9-5} = imm5;
  // Inherit Rd in 4-0
}

// Format for load-register (literal) instructions.
class A64I_LDRlit<bits<2> opc, bit v,
                  dag outs, dag ins, string asmstr,
                  list<dag> patterns, InstrItinClass itin>
  : A64InstRt<outs, ins, asmstr, patterns, itin> {
  bits<19> Imm19;

  let Inst{31-30} = opc;
  let Inst{29-27} = 0b011;
  let Inst{26} = v;
  let Inst{25-24} = 0b00;
  let Inst{23-5} = Imm19;
  // Inherit Rt in 4-0
}

// Format for load-store exclusive instructions.
class A64I_LDSTex_tn<bits<2> size, bit o2, bit L, bit o1, bit o0,
                 dag outs, dag ins, string asmstr,
                 list <dag> patterns, InstrItinClass itin>
  : A64InstRtn<outs, ins, asmstr, patterns, itin> {
  let Inst{31-30} = size;
  let Inst{29-24} = 0b001000;
  let Inst{23} = o2;
  let Inst{22} = L;
  let Inst{21} = o1;
  let Inst{15} = o0;
}

class A64I_LDSTex_tt2n<bits<2> size, bit o2, bit L, bit o1, bit o0,
                     dag outs, dag ins, string asmstr,
                     list <dag> patterns, InstrItinClass itin>:
      A64I_LDSTex_tn<size, o2, L, o1, o0, outs, ins, asmstr, patterns, itin>{
   bits<5> Rt2;
   let Inst{14-10} = Rt2;
}

class A64I_LDSTex_stn<bits<2> size, bit o2, bit L, bit o1, bit o0,
                     dag outs, dag ins, string asmstr,
                     list <dag> patterns, InstrItinClass itin>:
      A64I_LDSTex_tn<size, o2, L, o1, o0, outs, ins, asmstr, patterns, itin>{
   bits<5> Rs;
   let Inst{20-16} = Rs;
}

class A64I_LDSTex_stt2n<bits<2> size, bit o2, bit L, bit o1, bit o0,
                     dag outs, dag ins, string asmstr,
                     list <dag> patterns, InstrItinClass itin>:
      A64I_LDSTex_stn<size, o2, L, o1, o0, outs, ins, asmstr, patterns, itin>{
   bits<5> Rt2;
   let Inst{14-10} = Rt2;
}

// Format for load-store register (immediate post-indexed) instructions
class A64I_LSpostind<bits<2> size, bit v, bits<2> opc,
                     dag outs, dag ins, string asmstr,
                     list<dag> patterns, InstrItinClass itin>
  : A64InstRtn<outs, ins, asmstr, patterns, itin> {
  bits<9> SImm9;

  let Inst{31-30} = size;
  let Inst{29-27} = 0b111;
  let Inst{26} = v;
  let Inst{25-24} = 0b00;
  let Inst{23-22} = opc;
  let Inst{21} = 0b0;
  let Inst{20-12} = SImm9;
  let Inst{11-10} = 0b01;
  // Inherit Rn in 9-5
  // Inherit Rt in 4-0
}

// Format for load-store register (immediate pre-indexed) instructions
class A64I_LSpreind<bits<2> size, bit v, bits<2> opc,
                    dag outs, dag ins, string asmstr,
                    list<dag> patterns, InstrItinClass itin>
  : A64InstRtn<outs, ins, asmstr, patterns, itin> {
  bits<9> SImm9;


  let Inst{31-30} = size;
  let Inst{29-27} = 0b111;
  let Inst{26} = v;
  let Inst{25-24} = 0b00;
  let Inst{23-22} = opc;
  let Inst{21} = 0b0;
  let Inst{20-12} = SImm9;
  let Inst{11-10} = 0b11;
  // Inherit Rn in 9-5
  // Inherit Rt in 4-0
}

// Format for load-store register (unprivileged) instructions
class A64I_LSunpriv<bits<2> size, bit v, bits<2> opc,
                    dag outs, dag ins, string asmstr,
                    list<dag> patterns, InstrItinClass itin>
  : A64InstRtn<outs, ins, asmstr, patterns, itin> {
  bits<9> SImm9;


  let Inst{31-30} = size;
  let Inst{29-27} = 0b111;
  let Inst{26} = v;
  let Inst{25-24} = 0b00;
  let Inst{23-22} = opc;
  let Inst{21} = 0b0;
  let Inst{20-12} = SImm9;
  let Inst{11-10} = 0b10;
  // Inherit Rn in 9-5
  // Inherit Rt in 4-0
}

// Format for load-store (unscaled immediate) instructions.
class A64I_LSunalimm<bits<2> size, bit v, bits<2> opc,
                     dag outs, dag ins, string asmstr,
                     list<dag> patterns, InstrItinClass itin>
  : A64InstRtn<outs, ins, asmstr, patterns, itin> {
  bits<9> SImm9;

  let Inst{31-30} = size;
  let Inst{29-27} = 0b111;
  let Inst{26} = v;
  let Inst{25-24} = 0b00;
  let Inst{23-22} = opc;
  let Inst{21} = 0b0;
  let Inst{20-12} = SImm9;
  let Inst{11-10} = 0b00;
  // Inherit Rn in 9-5
  // Inherit Rt in 4-0
}


// Format for load-store (unsigned immediate) instructions.
class A64I_LSunsigimm<bits<2> size, bit v, bits<2> opc,
                      dag outs, dag ins, string asmstr,
                      list<dag> patterns, InstrItinClass itin>
  : A64InstRtn<outs, ins, asmstr, patterns, itin> {
  bits<12> UImm12;

  let Inst{31-30} = size;
  let Inst{29-27} = 0b111;
  let Inst{26} = v;
  let Inst{25-24} = 0b01;
  let Inst{23-22} = opc;
  let Inst{21-10} = UImm12;
}

// Format for load-store register (register offset) instructions.
class A64I_LSregoff<bits<2> size, bit v, bits<2> opc, bit optionlo,
                    dag outs, dag ins, string asmstr,
                    list<dag> patterns, InstrItinClass itin>
  : A64InstRtn<outs, ins, asmstr, patterns, itin> {
  bits<5> Rm;

  // Complex operand selection needed for these instructions, so they
  // need an "addr" field for encoding/decoding to be generated.
  bits<3> Ext;
  // OptionHi = Ext{2-1}
  // S = Ext{0}

  let Inst{31-30} = size;
  let Inst{29-27} = 0b111;
  let Inst{26} = v;
  let Inst{25-24} = 0b00;
  let Inst{23-22} = opc;
  let Inst{21} = 0b1;
  let Inst{20-16} = Rm;
  let Inst{15-14} = Ext{2-1};
  let Inst{13} = optionlo;
  let Inst{12} = Ext{0};
  let Inst{11-10} = 0b10;
  // Inherits Rn in 9-5
  // Inherits Rt in 4-0

  let AddedComplexity = 50;
}

// Format for Load-store register pair (offset) instructions
class A64I_LSPoffset<bits<2> opc, bit v, bit l,
                      dag outs, dag ins, string asmstr,
                      list<dag> patterns, InstrItinClass itin>
  : A64InstRtt2n<outs, ins, asmstr, patterns, itin> {
  bits<7> SImm7;

  let Inst{31-30} = opc;
  let Inst{29-27} = 0b101;
  let Inst{26} = v;
  let Inst{25-23} = 0b010;
  let Inst{22} = l;
  let Inst{21-15} = SImm7;
  // Inherit Rt2 in 14-10
  // Inherit Rn in 9-5
  // Inherit Rt in 4-0
}

// Format for Load-store register pair (post-indexed) instructions
class A64I_LSPpostind<bits<2> opc, bit v, bit l,
                      dag outs, dag ins, string asmstr,
                      list<dag> patterns, InstrItinClass itin>
  : A64InstRtt2n<outs, ins, asmstr, patterns, itin> {
  bits<7> SImm7;

  let Inst{31-30} = opc;
  let Inst{29-27} = 0b101;
  let Inst{26} = v;
  let Inst{25-23} = 0b001;
  let Inst{22} = l;
  let Inst{21-15} = SImm7;
  // Inherit Rt2 in 14-10
  // Inherit Rn in 9-5
  // Inherit Rt in 4-0
}

// Format for Load-store register pair (pre-indexed) instructions
class A64I_LSPpreind<bits<2> opc, bit v, bit l,
                      dag outs, dag ins, string asmstr,
                      list<dag> patterns, InstrItinClass itin>
  : A64InstRtt2n<outs, ins, asmstr, patterns, itin> {
  bits<7> SImm7;

  let Inst{31-30} = opc;
  let Inst{29-27} = 0b101;
  let Inst{26} = v;
  let Inst{25-23} = 0b011;
  let Inst{22} = l;
  let Inst{21-15} = SImm7;
  // Inherit Rt2 in 14-10
  // Inherit Rn in 9-5
  // Inherit Rt in 4-0
}

// Format for Load-store non-temporal register pair (offset) instructions
class A64I_LSPnontemp<bits<2> opc, bit v, bit l,
                      dag outs, dag ins, string asmstr,
                      list<dag> patterns, InstrItinClass itin>
  : A64InstRtt2n<outs, ins, asmstr, patterns, itin> {
  bits<7> SImm7;

  let Inst{31-30} = opc;
  let Inst{29-27} = 0b101;
  let Inst{26} = v;
  let Inst{25-23} = 0b000;
  let Inst{22} = l;
  let Inst{21-15} = SImm7;
  // Inherit Rt2 in 14-10
  // Inherit Rn in 9-5
  // Inherit Rt in 4-0
}

// Format for Logical (immediate) instructions
class A64I_logicalimm<bit sf, bits<2> opc,
                      dag outs, dag ins, string asmstr,
                      list<dag> patterns, InstrItinClass itin>
  : A64InstRdn<outs, ins, asmstr, patterns, itin> {
  bit N;
  bits<6> ImmR;
  bits<6> ImmS;

  // N, ImmR and ImmS have no separate existence in any assembly syntax (or for
  // selection), so we'll combine them into a single field here.
  bits<13> Imm;
  // N = Imm{12};
  // ImmR = Imm{11-6};
  // ImmS = Imm{5-0};

  let Inst{31} = sf;
  let Inst{30-29} = opc;
  let Inst{28-23} = 0b100100;
  let Inst{22} = Imm{12};
  let Inst{21-16} = Imm{11-6};
  let Inst{15-10} = Imm{5-0};
  // Rn inherited in 9-5
  // Rd inherited in 4-0
}

// Format for Logical (shifted register) instructions
class A64I_logicalshift<bit sf, bits<2> opc, bits<2> shift, bit N,
                        dag outs, dag ins, string asmstr,
                        list<dag> patterns, InstrItinClass itin>
  : A64InstRdnm<outs, ins, asmstr, patterns, itin> {
  bits<6> Imm6;

  let Inst{31} = sf;
  let Inst{30-29} = opc;
  let Inst{28-24} = 0b01010;
  let Inst{23-22} = shift;
  let Inst{21} = N;
  // Rm inherited
  let Inst{15-10} = Imm6;
  // Rn inherited
  // Rd inherited
}

// Format for Move wide (immediate)
class A64I_movw<bit sf, bits<2> opc,
                dag outs, dag ins, string asmstr,
                list<dag> patterns, InstrItinClass itin>
  : A64InstRd<outs, ins, asmstr, patterns, itin> {
  bits<16> UImm16;
  bits<2> Shift; // Called "hw" officially

  let Inst{31} = sf;
  let Inst{30-29} = opc;
  let Inst{28-23} = 0b100101;
  let Inst{22-21} = Shift;
  let Inst{20-5} = UImm16;
  // Inherits Rd in 4-0
}

// Format for PC-relative addressing instructions, ADR and ADRP.
class A64I_PCADR<bit op,
                 dag outs, dag ins, string asmstr,
                 list<dag> patterns, InstrItinClass itin>
  : A64InstRd<outs, ins, asmstr, patterns, itin> {
  bits<21> Label;

  let Inst{31} = op;
  let Inst{30-29} = Label{1-0};
  let Inst{28-24} = 0b10000;
  let Inst{23-5} = Label{20-2};
}

// Format for system instructions
class A64I_system<bit l,
                  dag outs, dag ins, string asmstr,
                  list<dag> patterns, InstrItinClass itin>
  : A64Inst<outs, ins, asmstr, patterns, itin> {
  bits<2> Op0;
  bits<3> Op1;
  bits<4> CRn;
  bits<4> CRm;
  bits<3> Op2;
  bits<5> Rt;

  let Inst{31-22} = 0b1101010100;
  let Inst{21} = l;
  let Inst{20-19} = Op0;
  let Inst{18-16} = Op1;
  let Inst{15-12} = CRn;
  let Inst{11-8} = CRm;
  let Inst{7-5} = Op2;
  let Inst{4-0} = Rt;

  // These instructions can do horrible things.
  let hasSideEffects = 1;
}

// Format for unconditional branch (immediate) instructions
class A64I_Bimm<bit op,
                dag outs, dag ins, string asmstr,
                list<dag> patterns, InstrItinClass itin>
  : A64Inst<outs, ins, asmstr, patterns, itin> {
  // Doubly special in not even sharing register fields with other
  // instructions, so we create our own Rn here.
  bits<26> Label;

  let Inst{31} = op;
  let Inst{30-26} = 0b00101;
  let Inst{25-0} = Label;
}

// Format for Test & branch (immediate) instructions
class A64I_TBimm<bit op,
                dag outs, dag ins, string asmstr,
                list<dag> patterns, InstrItinClass itin>
  : A64InstRt<outs, ins, asmstr, patterns, itin> {
  // Doubly special in not even sharing register fields with other
  // instructions, so we create our own Rn here.
  bits<6> Imm;
  bits<14> Label;

  let Inst{31} = Imm{5};
  let Inst{30-25} = 0b011011;
  let Inst{24} = op;
  let Inst{23-19} = Imm{4-0};
  let Inst{18-5} = Label;
  // Inherit Rt in 4-0
}

// Format for Unconditional branch (register) instructions, including
// RET.  Shares no fields with instructions further up the hierarchy
// so top-level.
class A64I_Breg<bits<4> opc, bits<5> op2, bits<6> op3, bits<5> op4,
                dag outs, dag ins, string asmstr,
                list<dag> patterns, InstrItinClass itin>
  : A64Inst<outs, ins, asmstr, patterns, itin> {
  // Doubly special in not even sharing register fields with other
  // instructions, so we create our own Rn here.
  bits<5> Rn;

  let Inst{31-25} = 0b1101011;
  let Inst{24-21} = opc;
  let Inst{20-16} = op2;
  let Inst{15-10} = op3;
  let Inst{9-5}   = Rn;
  let Inst{4-0}   = op4;
}

