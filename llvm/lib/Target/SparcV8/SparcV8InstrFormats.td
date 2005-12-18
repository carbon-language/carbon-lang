//===- SparcV8InstrFormats.td - SparcV8 Instr Formats ------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Format #2 instruction classes in the SparcV8
//===----------------------------------------------------------------------===//

class F2 : InstV8 {                   // Format 2 instructions
  bits<3>  op2;
  bits<22> imm22;
  let op          = 0;    // op = 0
  let Inst{24-22} = op2;
  let Inst{21-0}  = imm22;
}

// Specific F2 classes: SparcV8 manual, page 44
//
class F2_1<bits<3> op2Val, dag ops, string asmstr, list<dag> pattern> : F2 {
  bits<5>  rd;

  dag OperandList = ops;
  let AsmString   = asmstr;
  let Pattern = pattern;

  let op2         = op2Val;

  let Inst{29-25} = rd;
}

class F2_2<bits<4> condVal, bits<3> op2Val, dag ops, string asmstr, 
           list<dag> pattern> : F2 {
  bits<4>   cond;
  bit       annul = 0;     // currently unused

  dag OperandList = ops;
  let AsmString   = asmstr;
  let Pattern = pattern;

  let cond        = condVal;
  let op2         = op2Val;

  let Inst{29}    = annul;
  let Inst{28-25} = cond;
}

//===----------------------------------------------------------------------===//
// Format #3 instruction classes in the SparcV8
//===----------------------------------------------------------------------===//

class F3 : InstV8 {
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
class F3_1<bits<2> opVal, bits<6> op3val, dag ops,
           string asmstr, list<dag> pattern> : F3 {
  bits<8> asi = 0; // asi not currently used in SparcV8
  bits<5> rs2;

  dag OperandList = ops;
  let AsmString   = asmstr;
  let Pattern = pattern;

  let op         = opVal;
  let op3        = op3val;

  let Inst{13}   = 0;     // i field = 0
  let Inst{12-5} = asi;   // address space identifier
  let Inst{4-0}  = rs2;
}

class F3_2<bits<2> opVal, bits<6> op3val, dag ops, 
           string asmstr, list<dag> pattern> : F3 {
  bits<13> simm13;

  dag OperandList = ops;
  let AsmString   = asmstr;
  let Pattern = pattern;

  let op         = opVal;
  let op3        = op3val;

  let Inst{13}   = 1;     // i field = 1
  let Inst{12-0} = simm13;
}

// floating-point
class F3_3<bits<2> opVal, bits<6> op3val, bits<9> opfval, dag ops,
           string asmstr, list<dag> pattern> : F3 {
  bits<5> rs2;

  dag OperandList = ops;
  let AsmString   = asmstr;
  let Pattern = pattern;

  let op         = opVal;
  let op3        = op3val;

  let Inst{13-5} = opfval;   // fp opcode
  let Inst{4-0}  = rs2;
}
