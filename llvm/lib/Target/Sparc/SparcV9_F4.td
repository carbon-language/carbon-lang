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

class F4_rd : F4 {
  bits<5> rd;
  set Inst{29-25} = rd;
}

class F4_rdsimm11 : F4_rd {
  bits<11> simm11;
  set Inst{10-0} = simm11;
}

class F4_rdsimm11rs1 : F4_rdsimm11 {
  bits<5> rs1;
  set Inst{18-14} = rs1;
}

// F4_rdrs1 - Common superclass of instructions that use rd & rs1
class F4_rdrs1 : F4_rd {
  bits<5> rs1;
  set Inst{18-14} = rs1;
}

// F4_rs1rdrs2 - Common superclass of instructions with rd, rs1, & rs2 fields
class F4_rdrs1rs2 : F4_rdrs1 {
  bits<5> rs2;
  set Inst{4-0} = rs2;
}

// F4_rs1 - Common class of instructions that do not have an rd field,
// but start at rs1
class F4_rs1 : F4 {
  bits<5> rs1;
  //set Inst{29-25} = dontcare;
  set Inst{18-14} = rs1;
}

// F4_rs1rs2 - Common class of instructions that only have rs1 and rs2 fields
class F4_rs1rs2 : F4_rs1 {
  bits<5> rs2;
  //set Inst{12-5} = dontcare;
  set Inst{4-0} = rs2;
}

// Actual F4 instruction classes

class F4_1<bits<2> opVal, bits<6> op3Val, string name> : F4_rdrs1rs2 {
  bits<2> cc;

  set op = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13} = 0; // i bit
  set Inst{12-11} = cc;
  //set Inst{10-5} = dontcare;
}

class F4_2<bits<2> opVal, bits<6> op3Val, string name> : F4_rdsimm11rs1 {
  bits<2> cc;

  set op = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13} = 1; // i bit
  set Inst{12-11} = cc;
}

class F4_3<bits<2> opVal, bits<6> op3Val, string name> : F3_rd {
  bits<5> rs2;
  bits<2> cc;

  set op = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13} = 0; // i bit
  set Inst{12-11} = cc;
  //set Inst{10-5} = dontcare;
  set Inst{4-0} = rs2;
}
