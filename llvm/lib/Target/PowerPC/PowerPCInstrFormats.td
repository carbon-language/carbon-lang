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

class I<string name, bits<6> opcode, bit ppc64, bit vmx> : Instruction {
  field bits<32> Inst;

  bits<3> ArgCount;
  bits<5> Arg0Type;
  bits<5> Arg1Type;
  bits<5> Arg2Type;
  bits<5> Arg3Type;
  bits<5> Arg4Type;
  bit PPC64 = ppc64;
  bit VMX = vmx;

  let Name = name;
  let Namespace = "PPC";
  let Inst{0-5} = opcode;
}

// 1.7.1 I-Form
class IForm<string name, bits<6> opcode, bit aa, bit lk, bit ppc64, bit vmx>
  : I<name, opcode, ppc64, vmx> {
  field bits<24> LI;

  let ArgCount = 1;
  let Arg0Type = Imm24.Value;
  let Arg1Type = 0;
  let Arg2Type = 0;
  let Arg3Type = 0;
  let Arg4Type = 0;

  let Inst{6-29}  = LI;
  let Inst{30}    = aa;
  let Inst{31}    = lk;
}

// 1.7.2 B-Form
class BForm<string name, bits<6> opcode, bit aa, bit lk, bit ppc64, bit vmx>
  : I<name, opcode, ppc64, vmx> {
  field bits<5>  BO;
  field bits<5>  BI;
  field bits<14> BD;

  let ArgCount = 3;
  let Arg0Type = Imm5.Value;
  let Arg1Type = Imm5.Value;
  let Arg2Type = PCRelimm14.Value;
  let Arg3Type = 0;
  let Arg4Type = 0;

  let Inst{6-10}  = BO;
  let Inst{11-15} = BI;
  let Inst{16-29} = BD;
  let Inst{30}    = aa;
  let Inst{31}    = lk;
}

class BForm_ext<string name, bits<6> opcode, bit aa, bit lk, bits<5> bo, 
                bits<5> bi, bit ppc64, bit vmx>
  : BForm<name, opcode, aa, lk, ppc64, vmx> {
  let ArgCount = 2;
  let Arg2Type = Imm5.Value;
  let Arg1Type = PCRelimm14.Value;
  let Arg2Type = 0;
  let BO = bo;
  let BI = bi;
}

// 1.7.4 D-Form
class DForm_base<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : I<name, opcode, ppc64, vmx> {
  field bits<5>  A;
  field bits<5>  B;
  field bits<16> C;
  
  let ArgCount = 3;
  let Arg0Type = Gpr.Value;
  let Arg1Type = Gpr.Value;
  let Arg2Type = Simm16.Value;
  let Arg3Type = 0;
  let Arg4Type = 0;
 
  let Inst{6-10}  = A;
  let Inst{11-15} = B;
  let Inst{16-31} = C;
}

class DForm_1<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : DForm_base<name, opcode, ppc64, vmx> {
  let Arg1Type = Disimm16.Value;
  let Arg2Type = Gpr0.Value;
}

class DForm_2<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : DForm_base<name, opcode, ppc64, vmx>;

class DForm_2_r0<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : DForm_base<name, opcode, ppc64, vmx> {
  let Arg1Type = Gpr0.Value;
}

// Currently we make the use/def reg distinction in ISel, not tablegen
class DForm_3<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : DForm_1<name, opcode, ppc64, vmx>;

class DForm_4<bits<6> opcode, bit ppc64, bit vmx,
              dag OL, string asmstr> : DForm_base<"", opcode, ppc64, vmx> {
  let OperandList = OL;
  let AsmString = asmstr;
}

class DForm_4_zero<string name, bits<6> opcode, bit ppc64, bit vmx,
                   dag OL, string asmstr> : DForm_1<"", opcode, ppc64, vmx> {
  let ArgCount = 0;
  let Arg0Type = 0;
  let Arg1Type = 0;
  let Arg2Type = 0;
  let A = 0;
  let B = 0;
  let C = 0;
  let OperandList = OL;
  let AsmString = asmstr;
}

class DForm_5<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : I<name, opcode, ppc64, vmx> {
  field bits<3>  BF;
  field bits<1>  L;
  field bits<5>  RA;
  field bits<16> I;

  let ArgCount = 4;
  let Arg0Type = Imm3.Value;
  let Arg1Type = Imm1.Value;
  let Arg2Type = Gpr.Value;
  let Arg3Type = Simm16.Value;
  let Arg4Type = 0;

  let Inst{6-8}   = BF;
  let Inst{9}     = 0;
  let Inst{10}    = L;
  let Inst{11-15} = RA;
  let Inst{16-31} = I;
}

class DForm_5_ext<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : DForm_5<name, opcode, ppc64, vmx> {
  let L = ppc64;
  let ArgCount = 3;
  let Arg0Type = Imm3.Value;
  let Arg1Type = Gpr.Value;
  let Arg2Type = Simm16.Value;
  let Arg3Type = 0;
}

class DForm_6<bits<6> opcode, bit ppc64, bit vmx,
              dag OL, string asmstr> 
  : DForm_5<"", opcode, ppc64, vmx> {
  let OperandList = OL;
  let AsmString = asmstr;
}

class DForm_6_ext<bits<6> opcode, bit ppc64, bit vmx,
                  dag OL, string asmstr> 
  : DForm_6<opcode, ppc64, vmx, OL, asmstr> {
  let L = ppc64;
  let ArgCount = 3;
  let Arg0Type = Imm3.Value;
  let Arg1Type = Gpr.Value;
  let Arg2Type = Simm16.Value;
  let Arg3Type = 0;
}

class DForm_7<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : DForm_base<name, opcode, ppc64, vmx> {
  let Arg1Type = Imm5.Value;
}

class DForm_8<string name, bits<6> opcode, bit ppc64, bit vmx>
  : DForm_1<name, opcode, ppc64, vmx> {
  let Arg0Type = Fpr.Value;
}

class DForm_9<string name, bits<6> opcode, bit ppc64, bit vmx>
  : DForm_1<name, opcode, ppc64, vmx> {
  let Arg0Type = Fpr.Value;
}

// 1.7.5 DS-Form
class DSForm_1<string name, bits<6> opcode, bits<2> xo, bit ppc64, bit vmx> 
  : I<name, opcode, ppc64, vmx> {
  field bits<5>  RST;
  field bits<14> DS;
  field bits<5>  RA;

  let ArgCount = 3;
  let Arg0Type = Gpr.Value;
  let Arg1Type = Disimm14.Value;
  let Arg2Type = Gpr.Value;
  let Arg3Type = 0;
  let Arg4Type = 0;

  let Inst{6-10}  = RST;
  let Inst{11-15} = RA;
  let Inst{16-29} = DS;
  let Inst{30-31} = xo;
}

class DSForm_2<string name, bits<6> opcode, bits<2> xo, bit ppc64, bit vmx>
  : DSForm_1<name, opcode, xo, ppc64, vmx>;

// 1.7.6 X-Form
class XForm_base_r3xo<bits<6> opcode, bits<10> xo, bit rc, bit ppc64, bit vmx,
                      dag OL, string asmstr> : I<"", opcode, ppc64, vmx> {
  field bits<5> RST;
  field bits<5> A;
  field bits<5> B;

  let ArgCount = 3;
  let Arg0Type = Gpr.Value;
  let Arg1Type = Gpr.Value;
  let Arg2Type = Gpr.Value;
  let Arg3Type = 0;
  let Arg4Type = 0;

  let Inst{6-10}  = RST;
  let Inst{11-15} = A;
  let Inst{16-20} = B;
  let Inst{21-30} = xo;
  let Inst{31}    = rc;
  let OperandList = OL;
  let AsmString = asmstr;
}


class XForm_1<bits<6> opcode, bits<10> xo, bit ppc64, bit vmx,
              dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, 0, ppc64, vmx, OL, asmstr>;

class XForm_5<bits<6> opcode, bits<10> xo, bit ppc64, bit vmx,
              dag OL, string asmstr>
  : XForm_base_r3xo<opcode, xo, 0, ppc64, vmx, OL, asmstr> {
  let ArgCount = 1;
  let Arg1Type = 0;
  let Arg2Type = 0;
  let A = 0;
  let B = 0;
}

class XForm_6<bits<6> opcode, bits<10> xo, bit rc, bit ppc64, bit vmx,
              dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, rc, ppc64, vmx, OL, asmstr>;

class XForm_8<bits<6> opcode, bits<10> xo, bit ppc64, bit vmx,
              dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, 0, ppc64, vmx, OL, asmstr>;

class XForm_10<bits<6> opcode, bits<10> xo, bit rc, bit ppc64, bit vmx,
               dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, rc, ppc64, vmx, OL, asmstr> {
  let Arg2Type = Imm5.Value;
}

class XForm_11<bits<6> opcode, bits<10> xo, bit rc, bit ppc64, bit vmx,
               dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, rc, ppc64, vmx, OL, asmstr> {
  let ArgCount = 2;
  let Arg2Type = 0;
  let B = 0;
}

class XForm_16<string name, bits<6> opcode, bits<10> xo, bit ppc64, bit vmx>
  : I<name, opcode, ppc64, vmx> {
  field bits<3>  BF;
  field bits<1>  L; 
  field bits<5>  RA;
  field bits<5>  RB;
  
  let ArgCount = 4;
  let Arg0Type = Imm3.Value;
  let Arg1Type = Imm1.Value;
  let Arg2Type = Gpr.Value;
  let Arg3Type = Gpr.Value;
  let Arg4Type = 0;
  
  let Inst{6-8}   = BF;
  let Inst{9}     = 0;
  let Inst{10}    = L;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XForm_16_ext<string name, bits<6> opcode, bits<10> xo, bit ppc64, bit vmx>
  : XForm_16<name, opcode, xo, ppc64, vmx> {
  let L = ppc64;
}

class XForm_17<bits<6> opcode, bits<10> xo, bit ppc64, bit vmx,
               dag OL, string asmstr> : I<"", opcode, ppc64, vmx> {
  field bits<3> BF;
  field bits<5> FRA;
  field bits<5> FRB;
  
  let ArgCount = 3;
  let Arg0Type = Imm3.Value;
  let Arg1Type = Fpr.Value;
  let Arg2Type = Fpr.Value;
  let Arg3Type = 0;
  let Arg4Type = 0;

  let Inst{6-8}   = BF;
  let Inst{9-10}  = 0;
  let Inst{11-15} = FRA;
  let Inst{16-20} = FRB;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
  let OperandList = OL;
  let AsmString = asmstr;
}

class XForm_25<bits<6> opcode, bits<10> xo, bit ppc64, bit vmx,
               dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, 0, ppc64, vmx, OL, asmstr> {
  let Arg0Type = Fpr.Value;
  let Arg1Type = Gpr0.Value;
}

class XForm_26<bits<6> opcode, bits<10> xo, bit rc, bit ppc64, bit vmx,
               dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, rc, ppc64, vmx, OL, asmstr> {
  let ArgCount = 2;
  let Arg0Type = Fpr.Value;
  let Arg1Type = Fpr.Value;
  let Arg2Type = 0;
  let A = 0;
}

class XForm_28<bits<6> opcode, bits<10> xo, bit ppc64, bit vmx,
               dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, 0, ppc64, vmx, OL, asmstr> {
  let Arg0Type = Fpr.Value;
  let Arg1Type = Gpr0.Value;
}

// 1.7.7 XL-Form
class XLForm_1<bits<6> opcode, bits<10> xo, bit ppc64, bit vmx,
               dag OL, string asmstr> 
  : XForm_base_r3xo<opcode, xo, 0, ppc64, vmx, OL, asmstr> {
  let Arg0Type = Imm5.Value;
  let Arg1Type = Imm5.Value;
  let Arg2Type = Imm5.Value;
}

class XLForm_2<string name, bits<6> opcode, bits<10> xo, bit lk, bit ppc64, 
               bit vmx> : I<name, opcode, ppc64, vmx> {
  field bits<5>  BO;
  field bits<5>  BI;
  field bits<2>  BH;
  
  let ArgCount = 3;
  let Arg0Type = Imm5.Value;
  let Arg1Type = Imm5.Value;
  let Arg2Type = Imm2.Value;
  let Arg3Type = 0;
  let Arg4Type = 0;

  let Inst{6-10}  = BO;
  let Inst{11-15} = BI;
  let Inst{16-18} = 0;
  let Inst{19-20} = BH;
  let Inst{21-30} = xo;
  let Inst{31}    = lk;
}

class XLForm_2_ext<string name, bits<6> opcode, bits<10> xo, bits<5> bo, 
                   bits<5> bi, bit lk, bit ppc64, bit vmx>
  : XLForm_2<name, opcode, xo, lk, ppc64, vmx> {
  let ArgCount = 0;
  let Arg0Type = 0;
  let Arg1Type = 0;
  let Arg2Type = 0;
  let BO = bo;
  let BI = bi;
  let BH = 0;
}

// 1.7.8 XFX-Form
class XFXForm_1<bits<6> opcode, bits<10> xo, bit ppc64, bit vmx, 
                dag OL, string asmstr> : I<"", opcode, ppc64, vmx> {
  field bits<5>  ST;
  field bits<10> SPR;

  let ArgCount = 2;
  let Arg0Type = Imm5.Value;
  let Arg1Type = Gpr.Value;
  let Arg2Type = 0;
  let Arg3Type = 0;
  let Arg4Type = 0;

  let Inst{6-10}  = ST;
  let Inst{11-20} = SPR;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
  let OperandList = OL;
  let AsmString = asmstr;
}

class XFXForm_1_ext<bits<6> opcode, bits<10> xo, bits<10> spr, bit ppc64, 
                    bit vmx, dag OL, string asmstr> 
  : XFXForm_1<opcode, xo, ppc64, vmx, OL, asmstr> {
  let ArgCount = 1;
  let Arg0Type = Gpr.Value;
  let Arg1Type = 0;
  let SPR = spr;
}

class XFXForm_7<bits<6> opcode, bits<10> xo, bit ppc64, bit vmx,
                dag OL, string asmstr>
  : XFXForm_1<opcode, xo, ppc64, vmx, OL, asmstr>;

class XFXForm_7_ext<bits<6> opcode, bits<10> xo, bits<10> spr, 
                    bit ppc64, bit vmx, dag OL, string asmstr> 
  : XFXForm_7<opcode, xo, ppc64, vmx, OL, asmstr> {
  let ArgCount = 1;
  let Arg0Type = Gpr.Value;
  let Arg1Type = 0;
  let SPR = spr;
}

// 1.7.10 XS-Form
class XSForm_1<bits<6> opcode, bits<9> xo, bit rc, bit ppc64, bit vmx,
               dag OL, string asmstr> : I<"", opcode, ppc64, vmx> {
  field bits<5> RS;
  field bits<5> A;
  field bits<6> SH;

  let ArgCount = 3;
  let Arg0Type = Gpr.Value;
  let Arg1Type = Gpr.Value;
  let Arg2Type = Imm6.Value;
  let Arg3Type = 0;
  let Arg4Type = 0;

  let Inst{6-10}  = RS;
  let Inst{11-15} = A;
  let Inst{16-20} = SH{1-5};
  let Inst{21-29} = xo;
  let Inst{30}    = SH{0};
  let Inst{31}    = rc;
  let OperandList = OL;
  let AsmString = asmstr;
}

// 1.7.11 XO-Form
class XOForm_1<bits<6> opcode, bits<9> xo, bit oe, bit rc, bit ppc64, bit vmx,
               dag OL, string asmstr> : I<"", opcode, ppc64, vmx> {
  field bits<5>  RT;
  field bits<5>  RA;
  field bits<5>  RB;

  let ArgCount = 3;
  let Arg0Type = Gpr.Value;
  let Arg1Type = Gpr.Value;
  let Arg2Type = Gpr.Value;
  let Arg3Type = 0;
  let Arg4Type = 0;

  let Inst{6-10}  = RT;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21}    = oe;
  let Inst{22-30} = xo;
  let Inst{31}    = rc;  
  let OperandList = OL;
  let AsmString = asmstr;
}

class XOForm_1r<bits<6> opcode, bits<9> xo, bit oe, bit rc, bit ppc64, bit vmx,
               dag OL, string asmstr>
  : XOForm_1<opcode, xo, oe, rc, ppc64, vmx, OL, asmstr> {
  let Inst{11-15} = RB;
  let Inst{16-20} = RA;
}

class XOForm_3<bits<6> opcode, bits<9> xo, bit oe, bit rc, bit ppc64, bit vmx,
               dag OL, string asmstr>
  : XOForm_1<opcode, xo, oe, rc, ppc64, vmx, OL, asmstr> {
  let ArgCount = 2;
  let RB = 0;
}

// 1.7.12 A-Form
class AForm_1<bits<6> opcode, bits<5> xo, bit rc, bit ppc64, bit vmx, 
              dag OL, string asmstr> : I<"", opcode, ppc64, vmx> {
  let ArgCount = 4;
  field bits<5>  FRT;
  field bits<5>  FRA;
  field bits<5>  FRB;
  field bits<5>  FRC;

  let Arg0Type = Fpr.Value;
  let Arg1Type = Fpr.Value;
  let Arg2Type = Fpr.Value;
  let Arg3Type = Fpr.Value;
  let Arg4Type = 0;

  let Inst{6-10}  = FRT;
  let Inst{11-15} = FRA;
  let Inst{16-20} = FRB;
  let Inst{21-25} = FRC;
  let Inst{26-30} = xo;
  let Inst{31}    = rc;
  let OperandList = OL;
  let AsmString = asmstr;
}

class AForm_2<bits<6> opcode, bits<5> xo, bit rc, bit ppc64, bit vmx, dag OL,
              string asmstr> 
  : AForm_1<opcode, xo, rc, ppc64, vmx, OL, asmstr> {
  let ArgCount = 3;
  let Arg3Type = 0;
  let FRC = 0;
}

class AForm_3<bits<6> opcode, bits<5> xo, bit rc, bit ppc64, bit vmx, dag OL,
              string asmstr> 
  : AForm_1<opcode, xo, rc, ppc64, vmx, OL, asmstr> {
  let ArgCount = 3;
  let Arg3Type = 0;
  let FRB = 0;
}

// 1.7.13 M-Form
class MForm_1<bits<6> opcode, bit rc, bit ppc64, bit vmx,
              dag OL, string asmstr> : I<"", opcode, ppc64, vmx> {
  let ArgCount = 5;
  field bits<5>  RS;
  field bits<5>  RA;
  field bits<5>  RB;
  field bits<5>  MB;
  field bits<5>  ME;

  let Arg0Type = Gpr.Value;
  let Arg1Type = Gpr.Value;
  let Arg2Type = Gpr.Value;
  let Arg3Type = Imm5.Value;
  let Arg4Type = Imm5.Value;

  let Inst{6-10}  = RS;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-25} = MB;
  let Inst{26-30} = ME;
  let Inst{31}    = rc;
  let OperandList = OL;
  let AsmString = asmstr;
}

class MForm_2<bits<6> opcode, bit rc, bit ppc64, bit vmx, 
              dag OL, string asmstr>
  : MForm_1<opcode, rc, ppc64, vmx, OL, asmstr> {
  let Arg2Type = Imm5.Value;
}

// 1.7.14 MD-Form
class MDForm_1<bits<6> opcode, bits<3> xo, bit rc, bit ppc64, bit vmx,
               dag OL, string asmstr> : I<"", opcode, ppc64, vmx> {
  let ArgCount = 4;
  field bits<5>  RS;
  field bits<5>  RA;
  field bits<6>  SH;
  field bits<6>  MBE;

  let Arg0Type = Gpr.Value;
  let Arg1Type = Gpr.Value;
  let Arg2Type = Imm6.Value;
  let Arg3Type = Imm6.Value;
  let Arg4Type = 0;

  let Inst{6-10}  = RS;
  let Inst{11-15} = RA;
  let Inst{16-20} = SH{1-5};
  let Inst{21-26} = MBE;
  let Inst{27-29} = xo;
  let Inst{30}    = SH{0};
  let Inst{31}    = rc;
  let OperandList = OL;
  let AsmString = asmstr;
}

//===----------------------------------------------------------------------===//

class Pseudo<string name> : I<name, 0, 0, 0> {
	let Name = name;
	let ArgCount = 0;
	let PPC64 = 0;
	let VMX = 0;

	let Arg0Type = Pseudo.Value;
	let Arg1Type = Pseudo.Value;
	let Arg2Type = Pseudo.Value;
	let Arg3Type = Pseudo.Value;
	let Arg4Type = 0;

	let Inst{31-0} = 0;
}
