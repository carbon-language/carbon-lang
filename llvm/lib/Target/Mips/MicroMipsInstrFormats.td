class MMArch {
  string Arch = "micromips";
  list<dag> Pattern = [];
}

class ADD_FM_MM<bits<6> op, bits<10> funct> : MMArch {
  bits<5> rt;
  bits<5> rs;
  bits<5> rd;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10}    = 0;
  let Inst{9-0}   = funct;
}

class ADDI_FM_MM<bits<6> op> : MMArch {
  bits<5>  rs;
  bits<5>  rt;
  bits<16> imm16;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-0}  = imm16;
}

class SLTI_FM_MM<bits<6> op> : MMArch {
  bits<5> rt;
  bits<5> rs;
  bits<16> imm16;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-0}  = imm16;
}

class LUI_FM_MM : MMArch {
  bits<5> rt;
  bits<16> imm16;

  bits<32> Inst;

  let Inst{31-26} = 0x10;
  let Inst{25-21} = 0xd;
  let Inst{20-16} = rt;
  let Inst{15-0}  = imm16;
}

class MULT_FM_MM<bits<10> funct> : MMArch {
  bits<5>  rs;
  bits<5>  rt;

  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0x3c;
}
