//===- SparcV8Instrs_F2.td - Format 2 instructions: SparcV8 Target --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Format #2 instruction classes in the SparcV8
//
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
class F2_1<bits<3> op2Val, string name> : F2 {
  bits<5>  rd;
  bits<22> imm;

  let op2         = op2Val;
  let Name        = name;

  let Inst{29-25} = rd;
}

class F2_2<bits<4> cond, bits<3> op2Val, string name> : F2 {
  bits<4>   cond;
  bit       annul = 0;     // currently unused

  let cond        = condVal;
  let op2         = op2Val;
  let Name        = name;

  let Inst{29}    = annul;
  let Inst{28-25} = cond;
}
