//===- Sparc.td - Target Description for Sparc V9 Target --------*- C++ -*-===//
// vim:ft=cpp
//===----------------------------------------------------------------------===//

//----------------------- F4 classes -----------------------------------------

// F4 - Common superclass of all F4 instructions.  All instructions have an op3
// field.
class F4 : InstV9 {
  bits<6> op3;
  set Inst{24-19} = op3;
}

// F4_rs1 - Common class of instructions that use an rs1 field
class F4_rs1 : F4 {
  bits<5> rs1;
  set Inst{18-14} = rs1;
}

// F4_rs1rs2 - Common class of instructions that have rs1 and rs2 fields
class F4_rs1rs2 : F4_rs1 {
  bits<5> rs2;
  set Inst{4-0} = rs2;
}

// F4_rs1rs2rd - Common class of instructions that have 3 register operands
class F4_rs1rs2rd : F4_rs1rs2 {
  bits<5> rd;
  set Inst{29-25} = rd;
}

// F4_rs1rs2rd - Common class of instructions that have 2 reg and 1 imm operand
class F4_rs1simm11rd : F4_rs1 {
  bits<11> simm11;
  bits<5>  rd;

  set Inst{10-0}  = simm11;
  set Inst{29-25} = rd;
}

// F4_cc - Common class of instructions that have a cond field
class F4_cond : F4 {
  bits<4> cond;
  set Inst{17-14} = cond;
}

// F4_cc - Common class of instructions that have cc register as first operand
class F4_condcc : F4_cond {
  bits<3> cc;
  set Inst{18} = cc{2};
  set Inst{12} = cc{1};
  set Inst{11} = cc{0};
}

// Actual F4 instruction classes
//
class F4_1<bits<2> opVal, bits<6> op3Val, string name> : F4_rs1rs2rd {
  bits<2> cc;

  set op = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13} = 0; // i bit
  set Inst{12-11} = cc;
  set Inst{10-5} = 0; // don't care
}

class F4_2<bits<2> opVal, bits<6> op3Val, string name> : F4_rs1simm11rd {
  bits<2> cc;

  set op = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13} = 1; // i bit
  set Inst{12-11} = cc;
}

class F4_3<bits<2> opVal, bits<6> op3Val, bits<4> condVal,
           string name> : F4_condcc {
  bits<5> rs2;
  bits<5> rd;

  set op = opVal;
  set op3 = op3Val;
  set cond = condVal;
  set Name = name;
  set Inst{29-25} = rd;
  set Inst{13} = 0; // i bit
  set Inst{10-5} = 0; // don't care
  set Inst{4-0} = rs2;
}

class F4_4<bits<2> opVal, bits<6> op3Val, bits<4> condVal,
           string name> : F4_condcc {
  bits<11> simm11;
  bits<5>  rd;

  set op   = opVal;
  set op3  = op3Val;
  set cond = condVal;
  set Name = name;
  set Inst{29-25} = rd;
  set Inst{13} = 1; // i bit
  set Inst{10-0} = simm11;
}  
  
// FIXME: class F4_5

class F4_6<bits<2> opVal, bits<6> op3Val, bits<3> rcondVal,
           bits<5> opf_lowVal, string name> : F4_rs1rs2rd {
  set op  = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13} = 0;
  set Inst{12-10} = rcondVal;
  set Inst{9-5} = opf_lowVal;
}

class F4_7<bits<2> opVal, bits<6> op3Val, bits<4> condVal,
           bits<6> opf_lowVal, string name> : F4_cond {
  bits<3> cc;
  bits<5> rs2;
  bits<5> rd;

  set op   = opVal;
  set op3  = op3Val;
  set cond = condVal;
  set Name = name;
  set Inst{29-25} = rd;
  set Inst{18}    = 0;
  set Inst{13-11} = cc;
  set Inst{10-5}  = opf_lowVal;
  set Inst{4-0}   = rs2;
}

// FIXME: F4 classes 8-9
