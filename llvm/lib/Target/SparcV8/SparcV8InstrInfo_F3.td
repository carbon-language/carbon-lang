//===- SparcV8Instrs_F3.td - Format 3 Instructions: SparcV8 Target --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Format #3 instruction classes in the SparcV8
//
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
class F3_1<bits<2> opVal, bits<6> op3val, string name> : F3 {
  bits<8> asi;
  bits<5> rs2;

  let op         = opVal;
  let op3        = op3val;
  let Name       = name;

  let Inst{13}   = 0;     // i field = 0
  let Inst{12-5} = asi;   // address space identifier
  let Inst{4-0}  = rs2;
}

class F3_2<bits<2> opVal, bits<6> op3val, string name> : F3 {
  bits<13> simm13;

  let op         = opVal;
  let op3        = op3val;
  let Name       = name;

  let Inst{13}   = 1;     // i field = 1
  let Inst{12-0} = simm13;
}

/*
class F3_3<bits<2> opVal, bits<6> op3val, bits<9> opfVal, string name> 
  : F3_rs1rs2 {
  bits<5> rs2;

  let op         = opVal;
  let op3        = op3val;
  let Name       = name;

  let Inst{13-5} = opfVal;
  let Inst{4-0}  = rs2;
}
*/