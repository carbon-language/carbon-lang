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
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
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

class SRA_FM_MM<bits<10> funct, bit rotate> : MMArch {
  bits<5> rd;
  bits<5> rt;
  bits<5> shamt;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{25-21} = rd;
  let Inst{20-16} = rt;
  let Inst{15-11} = shamt;
  let Inst{10}    = rotate;
  let Inst{9-0}   = funct;
}

class SRLV_FM_MM<bits<10> funct, bit rotate> : MMArch {
  bits<5> rd;
  bits<5> rt;
  bits<5> rs;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10}    = rotate;
  let Inst{9-0}   = funct;
}

class LW_FM_MM<bits<6> op> : MMArch {
  bits<5> rt;
  bits<21> addr;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rt;
  let Inst{20-16} = addr{20-16};
  let Inst{15-0}  = addr{15-0};
}

class LWL_FM_MM<bits<4> funct> {
  bits<5> rt;
  bits<21> addr;

  bits<32> Inst;

  let Inst{31-26} = 0x18;
  let Inst{25-21} = rt;
  let Inst{20-16} = addr{20-16};
  let Inst{15-12} = funct;
  let Inst{11-0}  = addr{11-0};
}

class CMov_F_I_FM_MM<bits<7> func> : MMArch {
  bits<5> rd;
  bits<5> rs;
  bits<3> fcc;

  bits<32> Inst;

  let Inst{31-26} = 0x15;
  let Inst{25-21} = rd;
  let Inst{20-16} = rs;
  let Inst{15-13} = fcc;
  let Inst{12-6}  = func;
  let Inst{5-0}   = 0x3b;
}

class MTLO_FM_MM<bits<10> funct> : MMArch {
  bits<5> rs;

  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-21} = 0x00;
  let Inst{20-16} = rs;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0x3c;
}

class MFLO_FM_MM<bits<10> funct> : MMArch {
  bits<5> rd;

  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-21} = 0x00;
  let Inst{20-16} = rd;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0x3c;
}

class CLO_FM_MM<bits<10> funct> : MMArch {
  bits<5> rd;
  bits<5> rs;

  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-21} = rd;
  let Inst{20-16} = rs;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0x3c;
}

class SEB_FM_MM<bits<10> funct> : MMArch {
  bits<5> rd;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-21} = rd;
  let Inst{20-16} = rt;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0x3c;
}

class EXT_FM_MM<bits<6> funct> : MMArch {
  bits<5> rt;
  bits<5> rs;
  bits<5> pos;
  bits<5> size;

  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = size;
  let Inst{10-6}  = pos;
  let Inst{5-0}   = funct;
}

class J_FM_MM<bits<6> op> : MMArch {
  bits<26> target;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-0}  = target;
}

class JR_FM_MM<bits<8> funct> : MMArch {
  bits<5> rs;

  bits<32> Inst;

  let Inst{31-21} = 0x00;
  let Inst{20-16} = rs;
  let Inst{15-14} = 0x0;
  let Inst{13-6}  = funct;
  let Inst{5-0}   = 0x3c;
}

class JALR_FM_MM<bits<10> funct> : MMArch {
  bits<5> rs;
  bits<5> rd;

  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-21} = rd;
  let Inst{20-16} = rs;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0x3c;
}

class BEQ_FM_MM<bits<6> op> : MMArch {
  bits<5>  rs;
  bits<5>  rt;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-0}  = offset;
}

class BGEZ_FM_MM<bits<5> funct> : MMArch {
  bits<5>  rs;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = 0x10;
  let Inst{25-21} = funct;
  let Inst{20-16} = rs;
  let Inst{15-0}  = offset;
}

class BGEZAL_FM_MM<bits<5> funct> : MMArch {
  bits<5>  rs;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = 0x10;
  let Inst{25-21} = funct;
  let Inst{20-16} = rs;
  let Inst{15-0}  = offset;
}

class SYNC_FM_MM : MMArch {
  bits<5> stype;

  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-21} = 0x0;
  let Inst{20-16} = stype;
  let Inst{15-6}  = 0x1ad;
  let Inst{5-0}   = 0x3c;
}

class BRK_FM_MM : MMArch {
  bits<10> code_1;
  bits<10> code_2;
  bits<32> Inst;
  let Inst{31-26} = 0x0;
  let Inst{25-16} = code_1;
  let Inst{15-6}  = code_2;
  let Inst{5-0}   = 0x07;
}

class SYS_FM_MM : MMArch {
  bits<10> code_;
  bits<32> Inst;
  let Inst{31-26} = 0x0;
  let Inst{25-16} = code_;
  let Inst{15-6}  = 0x22b;
  let Inst{5-0}   = 0x3c;
}

class WAIT_FM_MM : MMArch {
  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-16} = 0x00;
  let Inst{15-6}  = 0x24d;
  let Inst{5-0}   = 0x3c;
}

class ER_FM_MM<bits<10> funct> : MMArch {
  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-16} = 0x00;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0x3c;
}

class EI_FM_MM<bits<10> funct> : MMArch {
  bits<32> Inst;
  bits<5> rt;

  let Inst{31-26} = 0x00;
  let Inst{25-21} = 0x00;
  let Inst{20-16} = rt;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0x3c;
}

class TEQ_FM_MM<bits<6> funct> : MMArch {
  bits<5> rs;
  bits<5> rt;
  bits<4> code_;

  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-12} = code_;
  let Inst{11-6}  = funct;
  let Inst{5-0}   = 0x3c;
}

class TEQI_FM_MM<bits<5> funct> : MMArch {
  bits<5> rs;
  bits<16> imm16;

  bits<32> Inst;

  let Inst{31-26} = 0x10;
  let Inst{25-21} = funct;
  let Inst{20-16} = rs;
  let Inst{15-0}  = imm16;
}

class LL_FM_MM<bits<4> funct> {
  bits<5> rt;
  bits<21> addr;

  bits<32> Inst;

  let Inst{31-26} = 0x18;
  let Inst{25-21} = rt;
  let Inst{20-16} = addr{20-16};
  let Inst{15-12} = funct;
  let Inst{11-0}  = addr{11-0};
}
