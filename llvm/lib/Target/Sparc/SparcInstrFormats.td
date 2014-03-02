//===-- SparcInstrFormats.td - Sparc Instruction Formats ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class InstSP<dag outs, dag ins, string asmstr, list<dag> pattern>
          : Instruction {
  field bits<32> Inst;

  let Namespace = "SP";
  let Size = 4;

  bits<2> op;
  let Inst{31-30} = op;               // Top two bits are the 'op' field

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;

  let DecoderNamespace = "Sparc";
  field bits<32> SoftFail = 0;
}

//===----------------------------------------------------------------------===//
// Format #2 instruction classes in the Sparc
//===----------------------------------------------------------------------===//

// Format 2 instructions
class F2<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSP<outs, ins, asmstr, pattern> {
  bits<3>  op2;
  bits<22> imm22;
  let op          = 0;    // op = 0
  let Inst{24-22} = op2;
  let Inst{21-0}  = imm22;
}

// Specific F2 classes: SparcV8 manual, page 44
//
class F2_1<bits<3> op2Val, dag outs, dag ins, string asmstr, list<dag> pattern>
   : F2<outs, ins, asmstr, pattern> {
  bits<5>  rd;

  let op2         = op2Val;

  let Inst{29-25} = rd;
}

class F2_2<bits<3> op2Val, bit annul, dag outs, dag ins, string asmstr,
           list<dag> pattern> : F2<outs, ins, asmstr, pattern> {
  bits<4>   cond;
  let op2         = op2Val;

  let Inst{29}    = annul;
  let Inst{28-25} = cond;
}

class F2_3<bits<3> op2Val, bit annul, bit pred,
           dag outs, dag ins, string asmstr, list<dag> pattern>
      : InstSP<outs, ins, asmstr, pattern> {
  bits<2>  cc;
  bits<4>  cond;
  bits<19> imm19;

  let op          = 0;    // op = 0

  let Inst{29}    = annul;
  let Inst{28-25} = cond;
  let Inst{24-22} = op2Val;
  let Inst{21-20} = cc;
  let Inst{19}    = pred;
  let Inst{18-0}  = imm19;
}

//===----------------------------------------------------------------------===//
// Format #3 instruction classes in the Sparc
//===----------------------------------------------------------------------===//

class F3<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstSP<outs, ins, asmstr, pattern> {
  bits<5> rd;
  bits<6> op3;
  bits<5> rs1;
  let op{1} = 1;   // Op = 2 or 3
  let Inst{29-25} = rd;
  let Inst{24-19} = op3;
  let Inst{18-14} = rs1;
}

// Specific F3 classes: SparcV8 manual, page 44
//
class F3_1_asi<bits<2> opVal, bits<6> op3val, bits<8> asi, dag outs, dag ins,
           string asmstr, list<dag> pattern> : F3<outs, ins, asmstr, pattern> {
  bits<5> rs2;

  let op         = opVal;
  let op3        = op3val;

  let Inst{13}   = 0;     // i field = 0
  let Inst{12-5} = asi;   // address space identifier
  let Inst{4-0}  = rs2;
}

class F3_1<bits<2> opVal, bits<6> op3val, dag outs, dag ins, string asmstr,
       list<dag> pattern> : F3_1_asi<opVal, op3val, 0, outs, ins,
                                                     asmstr, pattern>;

class F3_2<bits<2> opVal, bits<6> op3val, dag outs, dag ins,
           string asmstr, list<dag> pattern> : F3<outs, ins, asmstr, pattern> {
  bits<13> simm13;

  let op         = opVal;
  let op3        = op3val;

  let Inst{13}   = 1;     // i field = 1
  let Inst{12-0} = simm13;
}

// floating-point
class F3_3<bits<2> opVal, bits<6> op3val, bits<9> opfval, dag outs, dag ins,
           string asmstr, list<dag> pattern> : F3<outs, ins, asmstr, pattern> {
  bits<5> rs2;

  let op         = opVal;
  let op3        = op3val;

  let Inst{13-5} = opfval;   // fp opcode
  let Inst{4-0}  = rs2;
}

// floating-point unary operations.
class F3_3u<bits<2> opVal, bits<6> op3val, bits<9> opfval, dag outs, dag ins,
           string asmstr, list<dag> pattern> : F3<outs, ins, asmstr, pattern> {
  bits<5> rs2;

  let op         = opVal;
  let op3        = op3val;
  let rs1        = 0;

  let Inst{13-5} = opfval;   // fp opcode
  let Inst{4-0}  = rs2;
}

// floating-point compares.
class F3_3c<bits<2> opVal, bits<6> op3val, bits<9> opfval, dag outs, dag ins,
           string asmstr, list<dag> pattern> : F3<outs, ins, asmstr, pattern> {
  bits<5> rs2;

  let op         = opVal;
  let op3        = op3val;

  let Inst{13-5} = opfval;   // fp opcode
  let Inst{4-0}  = rs2;
}

// Shift by register rs2.
class F3_Sr<bits<2> opVal, bits<6> op3val, bit xVal, dag outs, dag ins,
            string asmstr, list<dag> pattern> : F3<outs, ins, asmstr, pattern> {
  bit x = xVal;           // 1 for 64-bit shifts.
  bits<5> rs2;

  let op         = opVal;
  let op3        = op3val;

  let Inst{13}   = 0;     // i field = 0
  let Inst{12}   = x;     // extended registers.
  let Inst{4-0}  = rs2;
}

// Shift by immediate.
class F3_Si<bits<2> opVal, bits<6> op3val, bit xVal, dag outs, dag ins,
            string asmstr, list<dag> pattern> : F3<outs, ins, asmstr, pattern> {
  bit x = xVal;           // 1 for 64-bit shifts.
  bits<6> shcnt;          // shcnt32 / shcnt64.

  let op         = opVal;
  let op3        = op3val;

  let Inst{13}   = 1;     // i field = 1
  let Inst{12}   = x;     // extended registers.
  let Inst{5-0}  = shcnt;
}

// Define rr and ri shift instructions with patterns.
multiclass F3_S<string OpcStr, bits<6> Op3Val, bit XVal, SDNode OpNode,
                ValueType VT, RegisterClass RC> {
  def rr : F3_Sr<2, Op3Val, XVal, (outs RC:$rd), (ins RC:$rs1, IntRegs:$rs2),
                 !strconcat(OpcStr, " $rs1, $rs2, $rd"),
                 [(set VT:$rd, (OpNode VT:$rs1, i32:$rs2))]>;
  def ri : F3_Si<2, Op3Val, XVal, (outs RC:$rd), (ins RC:$rs1, i32imm:$shcnt),
                 !strconcat(OpcStr, " $rs1, $shcnt, $rd"),
                 [(set VT:$rd, (OpNode VT:$rs1, (i32 imm:$shcnt)))]>;
}

class F4<bits<6> op3, dag outs, dag ins, string asmstr, list<dag> pattern>
      : InstSP<outs, ins, asmstr, pattern> {
  bits<5> rd;

  let op          = 2;
  let Inst{29-25} = rd;
  let Inst{24-19} = op3;
}


class F4_1<bits<6> op3, dag outs, dag ins,
            string asmstr, list<dag> pattern>
      : F4<op3, outs, ins, asmstr, pattern> {

  bits<3> cc;
  bits<4> cond;
  bits<5> rs2;

  let Inst{4-0}   = rs2;
  let Inst{11}    = cc{0};
  let Inst{12}    = cc{1};
  let Inst{13}    = 0;
  let Inst{17-14} = cond;
  let Inst{18}    = cc{2};

}

class F4_2<bits<6> op3, dag outs, dag ins,
            string asmstr, list<dag> pattern>
      : F4<op3, outs, ins, asmstr, pattern> {
  bits<3>  cc;
  bits<4>  cond;
  bits<11> simm11;

  let Inst{10-0}  = simm11;
  let Inst{11}    = cc{0};
  let Inst{12}    = cc{1};
  let Inst{13}    = 1;
  let Inst{17-14} = cond;
  let Inst{18}    = cc{2};
}

class F4_3<bits<6> op3, bits<6> opf_low, dag outs, dag ins,
           string asmstr, list<dag> pattern>
      : F4<op3, outs, ins, asmstr, pattern> {
  bits<4> cond;
  bits<3> opf_cc;
  bits<5> rs2;

  let Inst{18}     = 0;
  let Inst{17-14}  = cond;
  let Inst{13-11}  = opf_cc;
  let Inst{10-5}   = opf_low;
  let Inst{4-0}    = rs2;
}
