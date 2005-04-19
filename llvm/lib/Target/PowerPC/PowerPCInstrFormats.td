//===- PowerPCInstrFormats.td - PowerPC Instruction Formats --*- tablegen -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

class Format<bits<5> val> {
  bits<5> Value = val;
}

def Pseudo: Format<0>;
def Gpr : Format<1>;
def Gpr0 : Format<2>;
def Simm16 : Format<3>;
def PCRelimm24 : Format<5>;
def Imm24 : Format<6>;
def Imm5 : Format<7>;
def PCRelimm14 : Format<8>;
def Imm14 : Format<9>;
def Imm2 : Format<10>;
def Crf : Format<11>;
def Imm3 : Format<12>;
def Imm1 : Format<13>;
def Fpr : Format<14>;
def Imm4 : Format<15>;
def Imm8 : Format<16>;
def Disimm16 : Format<17>;
def Disimm14 : Format<18>;
def Spr : Format<19>;
def Sgr : Format<20>;
def Imm15 : Format<21>;
def Vpr : Format<22>;
def Imm6 : Format<23>;

//===----------------------------------------------------------------------===//
//
// PowerPC instruction formats

class I<bits<6> opcode, bit ppc64, bit vmx, dag OL, string asmstr> 
  : Instruction {
  field bits<32> Inst;

  bit PPC64 = ppc64;
  bit VMX = vmx;

  let Name = "";
  let Namespace = "PPC";
  let Inst{0-5} = opcode;
  let OperandList = OL;
  let AsmString = asmstr;
}

// 1.7.1 I-Form
class IForm<bits<6> opcode, bit aa, bit lk, bit ppc64, bit vmx,
            dag OL, string asmstr> : I<opcode, ppc64, vmx, OL, asmstr> {
  bits<24> LI;

  let Inst{6-29}  = LI;
  let Inst{30}    = aa;
  let Inst{31}    = lk;
}

// 1.7.2 B-Form
class BForm<bits<6> opcode, bit aa, bit lk, bit ppc64, bit vmx,
            dag OL, string asmstr> : I<opcode, ppc64, vmx, OL, asmstr> {
  bits<5>  BO;
  bits<3>  CRNum;
  bits<2>  BICode;
  bits<14> BD;

  let Inst{6-10}  = BO;
  let Inst{11-13} = CRNum;
  let Inst{14-15} = BICode;
  let Inst{16-29} = BD;
  let Inst{30}    = aa;
  let Inst{31}    = lk;
}

class BForm_ext<bits<6> opcode, bit aa, bit lk, bits<5> bo, bits<2> bicode, 
                bit ppc64, bit vmx, dag OL, string asmstr>
  : BForm<opcode, aa, lk, ppc64, vmx, OL, asmstr> {
  let BO = bo;
  let BICode = bicode;
}

// 1.7.4 D-Form
class DForm_base<bits<6> opcode, dag OL, string asmstr>
  : I<opcode, 0, 0, OL, asmstr> {
  bits<5>  A;
  bits<5>  B;
  bits<16> C;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = B;
  let Inst{16-31} = C;
}

class DForm_1<bits<6> opcode, dag OL, string asmstr>
         : I<opcode, 0, 0, OL, asmstr> {
  bits<5>  A;
  bits<16> C;
  bits<5>  B;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = B;
  let Inst{16-31} = C;
}

class DForm_2<bits<6> opcode, dag OL, string asmstr>
  : DForm_base<opcode, OL, asmstr>;

class DForm_2_r0<bits<6> opcode, dag OL, string asmstr>
  : I<opcode, 0, 0, OL, asmstr> {
  bits<5>  A;
  bits<16> B;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = 0;
  let Inst{16-31} = B;
}

// Currently we make the use/def reg distinction in ISel, not tablegen
class DForm_3<bits<6> opcode, dag OL, string asmstr>
  : DForm_1<opcode, OL, asmstr>;

class DForm_4<bits<6> opcode, dag OL, string asmstr> 
  : I<opcode, 0, 0, OL, asmstr> {
  bits<5>  B;
  bits<5>  A;
  bits<16> C;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = B;
  let Inst{16-31} = C;
}
              
class DForm_4_zero<bits<6> opcode, dag OL, string asmstr>
  : DForm_1<opcode, OL, asmstr> {
  let A = 0;
  let B = 0;
  let C = 0;
}

class DForm_5<bits<6> opcode, dag OL, string asmstr>
         : I<opcode, 0, 0, OL, asmstr> {
  bits<3>  BF;
  bits<1>  L;
  bits<5>  RA;
  bits<16> I;

  let Inst{6-8}   = BF;
  let Inst{9}     = 0;
  let Inst{10}    = L;
  let Inst{11-15} = RA;
  let Inst{16-31} = I;
}

class DForm_5_ext<bits<6> opcode, dag OL, string asmstr>
  : DForm_5<opcode, OL, asmstr> {
  let L = PPC64;
}

class DForm_6<bits<6> opcode, dag OL, string asmstr> 
  : DForm_5<opcode, OL, asmstr>;

class DForm_6_ext<bits<6> opcode, dag OL, string asmstr>
  : DForm_6<opcode, OL, asmstr> {
  let L = PPC64;
}

class DForm_8<bits<6> opcode, dag OL, string asmstr>
  : DForm_1<opcode, OL, asmstr> {
}

class DForm_9<bits<6> opcode, dag OL, string asmstr>
  : DForm_1<opcode, OL, asmstr> {
}

// 1.7.5 DS-Form
class DSForm_1<bits<6> opcode, bits<2> xo, dag OL, string asmstr>
         : I<opcode, 0, 0, OL, asmstr> {
  bits<5>  RST;
  bits<14> DS;
  bits<5>  RA;

  let Inst{6-10}  = RST;
  let Inst{11-15} = RA;
  let Inst{16-29} = DS;
  let Inst{30-31} = xo;
}

class DSForm_2<bits<6> opcode, bits<2> xo, dag OL, string asmstr>
  : DSForm_1<opcode, xo, OL, asmstr>;

// 1.7.6 X-Form
class XForm_base_r3xo<bits<6> opcode, bits<10> xo, bit rc,
                      dag OL, string asmstr> 
  : I<opcode, 0, 0, OL, asmstr> {
  bits<5> RST;
  bits<5> A;
  bits<5> B;

  let Inst{6-10}  = RST;
  let Inst{11-15} = A;
  let Inst{16-20} = B;
  let Inst{21-30} = xo;
  let Inst{31}    = rc;
}

// This is the same as XForm_base_r3xo, but the first two operands are swapped
// when code is emitted.
class XForm_base_r3xo_swapped
        <bits<6> opcode, bits<10> xo, bit rc, dag OL, string asmstr> 
  : I<opcode, 0, 0, OL, asmstr> {
  bits<5> A;
  bits<5> RST;
  bits<5> B;

  let Inst{6-10}  = RST;
  let Inst{11-15} = A;
  let Inst{16-20} = B;
  let Inst{21-30} = xo;
  let Inst{31}    = rc;
}


class XForm_1<bits<6> opcode, bits<10> xo, dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, 0, OL, asmstr>;

class XForm_6<bits<6> opcode, bits<10> xo, bit rc, dag OL, string asmstr> 
  : XForm_base_r3xo_swapped<opcode, xo, rc, OL, asmstr>;

class XForm_8<bits<6> opcode, bits<10> xo, dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, 0, OL, asmstr>;

class XForm_10<bits<6> opcode, bits<10> xo, bit rc, dag OL, string asmstr> 
  : XForm_base_r3xo_swapped<opcode, xo, rc, OL, asmstr> {
}

class XForm_11<bits<6> opcode, bits<10> xo, bit rc, dag OL, string asmstr> 
  : XForm_base_r3xo_swapped<opcode, xo, rc, OL, asmstr> {
  let B = 0;
}

class XForm_16<bits<6> opcode, bits<10> xo, dag OL, string asmstr>
         : I<opcode, 0, 0, OL, asmstr> {
  bits<3> BF;
  bits<1> L; 
  bits<5> RA;
  bits<5> RB;
  
  let Inst{6-8}   = BF;
  let Inst{9}     = 0;
  let Inst{10}    = L;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XForm_16_ext<bits<6> opcode, bits<10> xo, dag OL, string asmstr>
  : XForm_16<opcode, xo, OL, asmstr> {
  let L = PPC64;
}

class XForm_17<bits<6> opcode, bits<10> xo, dag OL, string asmstr>
         : I<opcode, 0, 0, OL, asmstr> {
  bits<3> BF;
  bits<5> FRA;
  bits<5> FRB;
  
  let Inst{6-8}   = BF;
  let Inst{9-10}  = 0;
  let Inst{11-15} = FRA;
  let Inst{16-20} = FRB;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XForm_25<bits<6> opcode, bits<10> xo, dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, 0, OL, asmstr> {
}

class XForm_26<bits<6> opcode, bits<10> xo, bit rc, dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, rc, OL, asmstr> {
  let A = 0;
}

class XForm_28<bits<6> opcode, bits<10> xo, dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, 0, OL, asmstr> {
}

// 1.7.7 XL-Form
class XLForm_1<bits<6> opcode, bits<10> xo, dag OL, string asmstr>
         : I<opcode, 0, 0, OL, asmstr> {
  bits<3> CRD;
  bits<2> CRDb;
  bits<3> CRA;
  bits<2> CRAb;
  bits<3> CRB;
  bits<2> CRBb;
  
  let Inst{6-8}   = CRD;
  let Inst{9-10}  = CRDb;
  let Inst{11-13} = CRA;
  let Inst{14-15} = CRAb;
  let Inst{16-18} = CRB;
  let Inst{19-20} = CRBb;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XLForm_2<bits<6> opcode, bits<10> xo, bit lk, 
               dag OL, string asmstr> : I<opcode, 0, 0, OL, asmstr> {
  bits<5> BO;
  bits<5> BI;
  bits<2> BH;
  
  let Inst{6-10}  = BO;
  let Inst{11-15} = BI;
  let Inst{16-18} = 0;
  let Inst{19-20} = BH;
  let Inst{21-30} = xo;
  let Inst{31}    = lk;
}

class XLForm_2_ext<bits<6> opcode, bits<10> xo, bits<5> bo, 
                   bits<5> bi, bit lk, dag OL, string asmstr>
  : XLForm_2<opcode, xo, lk, OL, asmstr> {
  let BO = bo;
  let BI = bi;
  let BH = 0;
}

class XLForm_3<bits<6> opcode, bits<10> xo, dag OL, string asmstr>
         : I<opcode, 0, 0, OL, asmstr> {
  bits<3> BF;
  bits<3> BFA;
  
  let Inst{6-8}   = BF;
  let Inst{9-10}  = 0;
  let Inst{11-13} = BFA;
  let Inst{14-15} = 0;
  let Inst{16-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

// 1.7.8 XFX-Form
class XFXForm_1<bits<6> opcode, bits<10> xo, dag OL, string asmstr>
         : I<opcode, 0, 0, OL, asmstr> {
  bits<5>  RT;
  bits<10> SPR;

  let Inst{6-10}  = RT;
  let Inst{11-20} = SPR;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XFXForm_1_ext<bits<6> opcode, bits<10> xo, bits<10> spr, 
                   dag OL, string asmstr> 
  : XFXForm_1<opcode, xo, OL, asmstr> {
  let SPR = spr;
}

class XFXForm_3<bits<6> opcode, bits<10> xo, 
                dag OL, string asmstr> : I<opcode, 0, 0, OL, asmstr> {
  bits<5>  RT;
   
  let Inst{6-10}  = RT;
  let Inst{11-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XFXForm_5<bits<6> opcode, bit mfcrf, bits<10> xo, 
                dag OL, string asmstr> : I<opcode, 0, 0, OL, asmstr> {
  bits<8>  FXM;
  bits<5>  ST;
   
  let Inst{6-10}  = ST;
  let Inst{11}    = mfcrf;
  let Inst{12-19} = FXM;
  let Inst{20}    = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XFXForm_7<bits<6> opcode, bits<10> xo, dag OL, string asmstr>
  : XFXForm_1<opcode, xo, OL, asmstr>;

class XFXForm_7_ext<bits<6> opcode, bits<10> xo, bits<10> spr, 
                    dag OL, string asmstr> 
  : XFXForm_7<opcode, xo, OL, asmstr> {
  let SPR = spr;
}

// 1.7.10 XS-Form
class XSForm_1<bits<6> opcode, bits<9> xo, bit rc, 
               dag OL, string asmstr> : I<opcode, 0, 0, OL, asmstr> {
  bits<5> RS;
  bits<5> A;
  bits<6> SH;

  let Inst{6-10}  = RS;
  let Inst{11-15} = A;
  let Inst{16-20} = SH{1-5};
  let Inst{21-29} = xo;
  let Inst{30}    = SH{0};
  let Inst{31}    = rc;
}

// 1.7.11 XO-Form
class XOForm_1<bits<6> opcode, bits<9> xo, bit oe, bit rc, 
               dag OL, string asmstr> : I<opcode, 0, 0, OL, asmstr> {
  bits<5> RT;
  bits<5> RA;
  bits<5> RB;

  let Inst{6-10}  = RT;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21}    = oe;
  let Inst{22-30} = xo;
  let Inst{31}    = rc;  
}

class XOForm_1r<bits<6> opcode, bits<9> xo, bit oe, bit rc,
               dag OL, string asmstr>
  : XOForm_1<opcode, xo, oe, rc, OL, asmstr> {
  let Inst{11-15} = RB;
  let Inst{16-20} = RA;
}

class XOForm_3<bits<6> opcode, bits<9> xo, bit oe, bit rc, 
               dag OL, string asmstr>
  : XOForm_1<opcode, xo, oe, rc, OL, asmstr> {
  let RB = 0;
}

// 1.7.12 A-Form
class AForm_1<bits<6> opcode, bits<5> xo, bit rc, dag OL, string asmstr>
         : I<opcode, 0, 0, OL, asmstr> {
  bits<5> FRT;
  bits<5> FRA;
  bits<5> FRC;
  bits<5> FRB;

  let Inst{6-10}  = FRT;
  let Inst{11-15} = FRA;
  let Inst{16-20} = FRB;
  let Inst{21-25} = FRC;
  let Inst{26-30} = xo;
  let Inst{31}    = rc;
}

class AForm_2<bits<6> opcode, bits<5> xo, bit rc, dag OL, string asmstr> 
  : AForm_1<opcode, xo, rc, OL, asmstr> {
  let FRC = 0;
}

class AForm_3<bits<6> opcode, bits<5> xo, bit rc, dag OL,
              string asmstr> 
  : AForm_1<opcode, xo, rc, OL, asmstr> {
  let FRB = 0;
}

// 1.7.13 M-Form
class MForm_1<bits<6> opcode, bit rc, dag OL, string asmstr>
         : I<opcode, 0, 0, OL, asmstr> {
  bits<5> RA;
  bits<5> RS;
  bits<5> RB;
  bits<5> MB;
  bits<5> ME;

  let Inst{6-10}  = RS;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-25} = MB;
  let Inst{26-30} = ME;
  let Inst{31}    = rc;
}

class MForm_2<bits<6> opcode, bit rc, dag OL, string asmstr>
  : MForm_1<opcode, rc, OL, asmstr> {
}

// 1.7.14 MD-Form
class MDForm_1<bits<6> opcode, bits<3> xo, bit rc,
               dag OL, string asmstr> : I<opcode, 0, 0, OL, asmstr> {
  bits<5> RS;
  bits<5> RA;
  bits<6> SH;
  bits<6> MBE;

  let Inst{6-10}  = RS;
  let Inst{11-15} = RA;
  let Inst{16-20} = SH{1-5};
  let Inst{21-26} = MBE;
  let Inst{27-29} = xo;
  let Inst{30}    = SH{0};
  let Inst{31}    = rc;
}

//===----------------------------------------------------------------------===//

class Pseudo<dag OL, string asmstr> : I<0, 0, 0, OL, asmstr> {
  let PPC64 = 0;
  let VMX = 0;

  let Inst{31-0} = 0;
}
