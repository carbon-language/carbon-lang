//===- Sparc.td - Target Description for Sparc V9 Target --------*- C++ -*-===//
// vim:ft=cpp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Format #3 classes
//

// F3 - Common superclass of all F3 instructions.  All instructions have an op3
// field.
class F3 : InstV9 {
  bits<6> op3;
  set op{1} = 1;   // Op = 2 or 3
  set Inst{24-19} = op3;
}

// F3_rs1 - Common class of instructions that have an rs1 field
class F3_rs1 : F3 {
  bits<5> rs1;
  set Inst{18-14} = rs1;
}

// F3_rs1rs2 - Common class of instructions that only have rs1 and rs2 fields
class F3_rs1rs2 : F3_rs1 {
  bits<5> rs2;
  set Inst{4-0} = rs2;
}

// F3_rs1rs2 - Common class of instructions that only have rs1 and rs2 fields
class F3_rs1rs2rd : F3_rs1rs2 {
  bits<5> rd;
  set Inst{29-25} = rd;
  set Inst{4-0} = rs2;
}

// F3_rs1simm13 - Common class of instructions that only have rs1 and simm13
class F3_rs1simm13 : F3_rs1 {
  bits<13> simm13;
  set Inst{12-0} = simm13;
}

class F3_rs1simm13rd : F3_rs1simm13 {
  bits<5> rd;
  set Inst{29-25} = rd;
}

// F3_rs1rd - Common class of instructions that have an rs1 and rd fields
class F3_rs1rd : F3_rs1 {
  bits<5> rd;
  set Inst{29-25} = rd;
}

// F3_rs2 - Common class of instructions that don't use an rs1
class F3_rs2 : F3 {
  bits<5> rs2;
  set Inst{4-0} = rs2;
}

// F3_rs2rd - Common class of instructions use rs2 and rd, but not rs1
class F3_rs2rd : F3_rs2 {
  bits<5> rd;
  set Inst{29-25} = rd;
}

// F3_rd - Common class of instructions that have an rd field
class F3_rd : F3 {
  bits<5> rd;
  set Inst{29-25} = rd;
}

// F3_rdrs1 - Common class of instructions that have rd and rs1 fields
class F3_rdrs1 : F3_rd {
  bits<5> rs1;
  set Inst{18-14} = rs1;
}

// F3_rdrs1simm13 - Common class of instructions that have rd, rs1, and simm13
class F3_rdrs1simm13 : F3_rd {
  bits<13> simm13;
  set Inst{12-0} = simm13;
}


// F3_rdrs1rs2 - Common class of instructions that have rd, rs1, and rs2 fields
class F3_rdrs1rs2 : F3_rs1 {
  bits<5> rs2;
  set Inst{4-0} = rs2;
}


// Specific F3 classes...
//

class F3_1<bits<2> opVal, bits<6> op3val, string name> : F3_rs1rs2rd {
  set op  = opVal;
  set op3 = op3val;
  set Name = name;
  set Inst{13} = 0;   // i field = 0
  //set Inst{12-5} = dontcare;
}

// The store instructions seem to like to see rd first, then rs1 and rs2
class F3_1rd<bits<2> opVal, bits<6> op3val, string name> : F3_rdrs1rs2 {
  set op  = opVal;
  set op3 = op3val;
  set Name = name;
  set Inst{13} = 0;   // i field = 0
  //set Inst{12-5} = dontcare;
}

class F3_2<bits<2> opVal, bits<6> op3val, string name> : F3_rs1simm13rd {
  set op  = opVal;
  set op3 = op3val;
  set Name = name;
  set Inst{13} = 1;   // i field = 1
}

// The store instructions seem to like to see rd first, then rs1 and imm
class F3_2rd<bits<2> opVal, bits<6> op3val, string name> : F3_rdrs1simm13 {
  set op  = opVal;
  set op3 = op3val;
  set Name = name;
  set Inst{13} = 1;   // i field = 1
}

class F3_3<bits<2> opVal, bits<6> op3val, string name> : F3_rs1rs2 {
  set op   = opVal;
  set op3  = op3val;
  set Name = name;
  set Inst{13}  = 0;
}

class F3_4<bits<2> opVal, bits<6> op3Val, string name> : F3_rs1simm13 {
  bits<13> simm;
  set op   = opVal;
  set op3  = op3Val;
  set Name = name;
  //set Inst{29-25} = dontcare;
  set Inst{13}  = 1;
  set Inst{12-0} = simm;
}

class F3_5<bits<2> opVal, bits<6> op3Val, bits<3> rcondVal,
           string name> : F3_rs1rs2rd {
  set op  = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13} = 0;           // i field = 0
  set Inst{12-10} = rcondVal; // rcond field
}

class F3_6<bits<2> opVal, bits<6> op3Val, bits<3> rcondVal,
           string name> : F3_rs1 {
  bits<10> simm10;
  bits<5>  rd;

  set op  = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13} = 1;           // i field = 1
  set Inst{12-10} = rcondVal; // rcond field
}

//FIXME: classes 7-10 not defined!!

class F3_11<bits<2> opVal, bits<6> op3Val, string name> : F3_rs1rs2rd {
  bit x;
  set op = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13} = 0; // i field = 0
  set Inst{12} = x;
  //set Inst{11-5} = dontcare;
}

class F3_12<bits<2> opVal, bits<6> op3Val, string name> : F3_rs1 {
  bits<5> shcnt;
  bits<5> rd;

  set op  = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{29-25} = rd;
  set Inst{13} = 1; // i field = 1
  set Inst{12} = 0; // x field = 0
  //set Inst{11-5} = dontcare;
  set Inst{4-0} = shcnt;
}

class F3_13<bits<2> opVal, bits<6> op3Val, string name> : F3_rs1 {
  bits<6> shcnt;
  bits<5> rd;

  set op  = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{29-25} = rd;
  set Inst{13} = 1; // i field = 1
  set Inst{12} = 1; // x field = 1
  //set Inst{11-6} = dontcare;
  set Inst{5-0} = shcnt;
}

class F3_14<bits<2> opVal, bits<6> op3Val,
            bits<9> opfval, string name> : F3_rs2rd {
  set op  = opVal;
  set op3 = op3Val;
  set Name = name;
  //set Inst{18-14} = dontcare;
  set Inst{13-5} = opfval;
}
  
class F3_16<bits<2> opVal,  bits<6> op3Val,
            bits<9> opfval, string name> : F3_rs1rs2rd {
  set op  = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13-5} = opfval;
}

class F3_17<bits<2> opVal, bits<6> op3Val, string name> : F3_rs1rd {
  set op = opVal;
  set op3 = op3Val;
  set Name = name;
  //Inst{13-0} = dontcare;
}

class F3_18<bits<5> fcn, string name> : F3 {
  set op = 2;
  set op3 = 0b111110;
  set Name = name;
  set Inst{29-25} = fcn;
  //set Inst{18-0 } = dontcare;
}

class F3_19<bits<2> opVal, bits<6> op3Val, string name> : F3_rd {
  set op = opVal;
  set op3 = op3Val;
  set Name = name;
  //Inst{18-0} = dontcare;
}

// FIXME: class F3_20
// FIXME: class F3_21
