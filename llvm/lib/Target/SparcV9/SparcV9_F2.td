//===- SparcV9_F2.td - Format 2 instructions: SparcV9 V9 Target -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Format #2 classes
//
class F2 : InstV9 {                   // Format 2 instructions
  bits<3> op2;
  let op          = 0;    // Op = 0
  let Inst{24-22} = op2;
}

// Format 2.1 instructions
class F2_1<string name> : F2 {
  bits<22> imm;
  bits<5>  rd;

  let Name        = name;
  let Inst{29-25} = rd;
  let Inst{21-0}  = imm;
}

class F2_br : F2 {                    // Format 2 Branch instruction
  let isBranch    = 1;      // All instances are branch instructions
}

class F2_2<bits<4> cond, string name> : F2_br { // Format 2.2 instructions
  bits<22> disp;
  bit      annul = 0; // currently unused by SparcV9 backend

  let Name        = name;
  let Inst{29}    = annul;
  let Inst{28-25} = cond;
  let Inst{21-0}  = disp;
}

class F2_3<bits<4> cond, string name> : F2_br { // Format 2.3 instructions
  bits<2>  cc;
  bits<19> disp;
  bit      predict = 1;
  bit      annul = 0; // currently unused by SparcV9 backend

  let Name        = name;
  let Inst{29}    = annul;
  let Inst{28-25} = cond;
  let Inst{21-20} = cc;
  let Inst{19}    = predict;
  let Inst{18-0}  = disp;
}

class F2_4<bits<3> rcond, string name> : F2_br { // Format 2.4 instructions
  bits<5>  rs1;
  bits<16> disp;
  bit      predict = 1;
  bit      annul = 0; // currently unused by SparcV9 backend

  let Name        = name;
  let Inst{29}    = annul;
  let Inst{28}    = 0;
  let Inst{27-25} = rcond;
  let Inst{21-20} = disp{15-14};
  let Inst{19}    = predict;
  let Inst{18-14} = rs1;
  let Inst{13-0 } = disp{13-0};
}
