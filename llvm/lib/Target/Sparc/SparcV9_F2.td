//===- Sparc.td - Target Description for Sparc V9 Target --------*- C++ -*-===//
// vim:ft=cpp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Format #2 classes
//
class F2 : InstV9 {                   // Format 2 instructions
  bits<3> op2;
  set op          = 0;    // Op = 0
  set Inst{24-22} = op2;
}

// Format 2.1 instructions
class F2_1<string name> : F2 {
  bits<22> imm;
  bits<5>  rd;

  set Name        = name;
  set Inst{29-25} = rd;
  set Inst{21-0}  = imm;
}

class F2_br : F2 {                    // Format 2 Branch instruction
  set isBranch    = 1;      // All instances are branch instructions
}

class F2_2<bits<4> cond, string name> : F2_br { // Format 2.2 instructions
  bits<22> disp;
  bit      annul;

  set Name        = name;
  set Inst{29}    = annul;
  set Inst{28-25} = cond;
  set Inst{21-0}  = disp;
}

class F2_3<bits<4> cond, string name> : F2_br { // Format 2.3 instructions
  bits<2>  cc;
  bits<19> disp;
  bit      predict = 1;
  bit      annul;

  set Name        = name;
  set Inst{29}    = annul;
  set Inst{28-25} = cond;
  set Inst{21-20} = cc;
  set Inst{19}    = 1; // predict;
  set Inst{18-0}  = disp;
}

class F2_4<bits<3> rcond, string name> : F2_br { // Format 2.4 instructions
  bits<5>  rs1;
  bits<16> disp;
  bit      predict = 1;
  bit      annul;

  set Name        = name;
  set Inst{29}    = annul;
  set Inst{28}    = 0;
  set Inst{27-25} = rcond;
  set Inst{21-20} = disp{15-14};
  set Inst{19}    = 1; // predict;
  set Inst{18-14} = rs1;
  set Inst{13-0 } = disp{13-0};
}
