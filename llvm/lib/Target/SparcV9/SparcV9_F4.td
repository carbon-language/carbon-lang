//===- SparcV9_F4.td - Format 4 instructions: SparcV9 V9 Target -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//----------------------- F4 classes -----------------------------------------

// F4 - Common superclass of all F4 instructions.  All instructions have an op3
// field.
class F4 : InstV9 {
  bits<6> op3;
  let Inst{24-19} = op3;
}

// F4_rs1 - Common class of instructions that use an rs1 field
class F4_rs1 : F4 {
  bits<5> rs1;
  let Inst{18-14} = rs1;
}

// F4_rs1rs2 - Common class of instructions that have rs1 and rs2 fields
class F4_rs1rs2 : F4_rs1 {
  bits<5> rs2;
  let Inst{4-0} = rs2;
}

// F4_rs1rs2rd - Common class of instructions that have 3 register operands
class F4_rs1rs2rd : F4_rs1rs2 {
  bits<5> rd;
  let Inst{29-25} = rd;
}

// F4_rs1rs2rd - Common class of instructions that have 2 reg and 1 imm operand
class F4_rs1simm11rd : F4_rs1 {
  bits<11> simm11;
  bits<5>  rd;

  let Inst{10-0}  = simm11;
  let Inst{29-25} = rd;
}

// F4_cc - Common class of instructions that have a cond field
class F4_cond : F4 {
  bits<4> cond;
  let Inst{17-14} = cond;
}

// F4_cc - Common class of instructions that have cc register as first operand
class F4_condcc : F4_cond {
  bits<3> cc;
  let Inst{18} = cc{2};
  let Inst{12} = cc{1};
  let Inst{11} = cc{0};
}

// Actual F4 instruction classes
//
class F4_1<bits<2> opVal, bits<6> op3Val, string name> : F4_rs1rs2rd {
  bits<2> cc;

  let op = opVal;
  let op3 = op3Val;
  let Name = name;
  let Inst{13} = 0; // i bit
  let Inst{12-11} = cc;
  let Inst{10-5} = 0; // don't care
}

class F4_2<bits<2> opVal, bits<6> op3Val, string name> : F4_rs1simm11rd {
  bits<2> cc;

  let op = opVal;
  let op3 = op3Val;
  let Name = name;
  let Inst{13} = 1; // i bit
  let Inst{12-11} = cc;
}

class F4_3<bits<2> opVal, bits<6> op3Val, bits<4> condVal,
           string name> : F4_condcc {
  bits<5> rs2;
  bits<5> rd;

  let op = opVal;
  let op3 = op3Val;
  let cond = condVal;
  let Name = name;
  let Inst{29-25} = rd;
  let Inst{13} = 0; // i bit
  let Inst{10-5} = 0; // don't care
  let Inst{4-0} = rs2;
}

class F4_4<bits<2> opVal, bits<6> op3Val, bits<4> condVal,
           string name> : F4_condcc {
  bits<11> simm11;
  bits<5>  rd;

  let op   = opVal;
  let op3  = op3Val;
  let cond = condVal;
  let Name = name;
  let Inst{29-25} = rd;
  let Inst{13} = 1; // i bit
  let Inst{10-0} = simm11;
}  
  
// FIXME: class F4_5

class F4_6<bits<2> opVal, bits<6> op3Val, bits<3> rcondVal,
           bits<5> opf_lowVal, string name> : F4_rs1rs2rd {
  let op  = opVal;
  let op3 = op3Val;
  let Name = name;
  let Inst{13} = 0;
  let Inst{12-10} = rcondVal;
  let Inst{9-5} = opf_lowVal;
}

class F4_7<bits<2> opVal, bits<6> op3Val, bits<4> condVal,
           bits<6> opf_lowVal, string name> : F4_cond {
  bits<3> cc;
  bits<5> rs2;
  bits<5> rd;

  let op   = opVal;
  let op3  = op3Val;
  let cond = condVal;
  let Name = name;
  let Inst{29-25} = rd;
  let Inst{18}    = 0;
  let Inst{13-11} = cc;
  let Inst{10-5}  = opf_lowVal;
  let Inst{4-0}   = rs2;
}

// FIXME: F4 classes 8-9
