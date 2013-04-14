//===-- SparcInstrFormats.td - Sparc Instruction Formats ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class InstSP<dag outs, dag ins, string asmstr, list<dag> pattern> : Instruction {
  field bits<32> Inst;

  let Namespace = "SP";

  bits<2> op;
  let Inst{31-30} = op;               // Top two bits are the 'op' field
  
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;
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

class F2_2<bits<4> condVal, bits<3> op2Val, dag outs, dag ins, string asmstr, 
           list<dag> pattern> : F2<outs, ins, asmstr, pattern> {
  bits<4>   cond;
  bit       annul = 0;     // currently unused

  let cond        = condVal;
  let op2         = op2Val;

  let Inst{29}    = annul;
  let Inst{28-25} = cond;
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
class F3_1<bits<2> opVal, bits<6> op3val, dag outs, dag ins,
           string asmstr, list<dag> pattern> : F3<outs, ins, asmstr, pattern> {
  bits<8> asi = 0; // asi not currently used
  bits<5> rs2;

  let op         = opVal;
  let op3        = op3val;

  let Inst{13}   = 0;     // i field = 0
  let Inst{12-5} = asi;   // address space identifier
  let Inst{4-0}  = rs2;
}

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
  def rr : F3_Sr<2, Op3Val, XVal, (outs RC:$rd), (ins RC:$rs, IntRegs:$rs2),
                 !strconcat(OpcStr, " $rs, $rs2, $rd"),
                 [(set VT:$rd, (OpNode VT:$rs, i32:$rs2))]>;
  def ri : F3_Si<2, Op3Val, XVal, (outs RC:$rd), (ins RC:$rs, i32imm:$shcnt),
                 !strconcat(OpcStr, " $rs, $shcnt, $rd"),
                 [(set VT:$rd, (OpNode VT:$rs, (i32 imm:$shcnt)))]>;
}
