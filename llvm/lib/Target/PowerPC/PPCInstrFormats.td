//===- PowerPCInstrFormats.td - PowerPC Instruction Formats --*- tablegen -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// PowerPC instruction formats

class I<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin>
        : Instruction {
  field bits<32> Inst;

  bit PPC64 = 0;  // Default value, override with isPPC64

  let Name = "";
  let Namespace = "PPC";
  let Inst{0-5} = opcode;
  let OperandList = OL;
  let AsmString = asmstr;
  let Itinerary = itin;
  
  /// These fields correspond to the fields in PPCInstrInfo.h.  Any changes to
  /// these must be reflected there!  See comments there for what these are.
  bits<1> PPC970_First = 0;
  bits<1> PPC970_Single = 0;
  bits<1> PPC970_Cracked = 0;
  bits<3> PPC970_Unit = 0;
}

class PPC970_DGroup_First   { bits<1> PPC970_First = 1;  }
class PPC970_DGroup_Single  { bits<1> PPC970_Single = 1; }
class PPC970_DGroup_Cracked { bits<1> PPC970_Cracked = 1; }
class PPC970_MicroCode;

class PPC970_Unit_Pseudo   { bits<3> PPC970_Unit = 0;   }
class PPC970_Unit_FXU      { bits<3> PPC970_Unit = 1;   }
class PPC970_Unit_LSU      { bits<3> PPC970_Unit = 2;   }
class PPC970_Unit_FPU      { bits<3> PPC970_Unit = 3;   }
class PPC970_Unit_CRU      { bits<3> PPC970_Unit = 4;   }
class PPC970_Unit_VALU     { bits<3> PPC970_Unit = 5;   }
class PPC970_Unit_VPERM    { bits<3> PPC970_Unit = 6;   }
class PPC970_Unit_BRU      { bits<3> PPC970_Unit = 7;   }


// 1.7.1 I-Form
class IForm<bits<6> opcode, bit aa, bit lk, dag OL, string asmstr,
            InstrItinClass itin, list<dag> pattern>
         : I<opcode, OL, asmstr, itin> {
  let Pattern = pattern;
  bits<24> LI;

  let Inst{6-29}  = LI;
  let Inst{30}    = aa;
  let Inst{31}    = lk;
}

// 1.7.2 B-Form
class BForm<bits<6> opcode, bit aa, bit lk, bits<5> bo, bits<2> bicode, dag OL, 
            string asmstr, InstrItinClass itin>
  : I<opcode, OL, asmstr, itin> {
  bits<3>  CR;
  bits<14> BD;

  let Inst{6-10}  = bo;
  let Inst{11-13} = CR;
  let Inst{14-15} = bicode;
  let Inst{16-29} = BD;
  let Inst{30}    = aa;
  let Inst{31}    = lk;
}

// 1.7.4 D-Form
class DForm_base<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin,
                 list<dag> pattern> 
  : I<opcode, OL, asmstr, itin> {
  bits<5>  A;
  bits<5>  B;
  bits<16> C;

  let Pattern = pattern;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = B;
  let Inst{16-31} = C;
}

class DForm_1<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin,
              list<dag> pattern>
  : I<opcode, OL, asmstr, itin> {
  bits<5>  A;
  bits<16> C;
  bits<5>  B;

  let Pattern = pattern;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = B;
  let Inst{16-31} = C;
}

class DForm_2<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin,
              list<dag> pattern>
  : DForm_base<opcode, OL, asmstr, itin, pattern>;

class DForm_2_r0<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin,
                 list<dag> pattern>
  : I<opcode, OL, asmstr, itin> {
  bits<5>  A;
  bits<16> B;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = 0;
  let Inst{16-31} = B;
}

// Currently we make the use/def reg distinction in ISel, not tablegen
class DForm_3<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin,
              list<dag> pattern>
  : DForm_1<opcode, OL, asmstr, itin, pattern>;

class DForm_4<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin,
              list<dag> pattern>
  : I<opcode, OL, asmstr, itin> {
  bits<5>  B;
  bits<5>  A;
  bits<16> C;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = B;
  let Inst{16-31} = C;
}
              
class DForm_4_zero<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin,
                   list<dag> pattern>
  : DForm_1<opcode, OL, asmstr, itin, pattern> {
  let A = 0;
  let B = 0;
  let C = 0;
}

class DForm_5<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin>
  : I<opcode, OL, asmstr, itin> {
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

class DForm_5_ext<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin>
  : DForm_5<opcode, OL, asmstr, itin> {
  let L = PPC64;
}

class DForm_6<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin> 
  : DForm_5<opcode, OL, asmstr, itin>;

class DForm_6_ext<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin>
  : DForm_6<opcode, OL, asmstr, itin> {
  let L = PPC64;
}

class DForm_8<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin,
              list<dag> pattern>
  : DForm_1<opcode, OL, asmstr, itin, pattern> {
}

class DForm_9<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin,
              list<dag> pattern>
  : DForm_1<opcode, OL, asmstr, itin, pattern> {
}

// 1.7.5 DS-Form
class DSForm_1<bits<6> opcode, bits<2> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
         : I<opcode, OL, asmstr, itin> {
  bits<5>  RST;
  bits<14> DS;
  bits<5>  RA;

  let Pattern = pattern;
  
  let Inst{6-10}  = RST;
  let Inst{11-15} = RA;
  let Inst{16-29} = DS;
  let Inst{30-31} = xo;
}

class DSForm_2<bits<6> opcode, bits<2> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
  : DSForm_1<opcode, xo, OL, asmstr, itin, pattern>;

// 1.7.6 X-Form
class XForm_base_r3xo<bits<6> opcode, bits<10> xo, dag OL, string asmstr, 
                      InstrItinClass itin, list<dag> pattern>
  : I<opcode, OL, asmstr, itin> {
  bits<5> RST;
  bits<5> A;
  bits<5> B;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RST;
  let Inst{11-15} = A;
  let Inst{16-20} = B;
  let Inst{21-30} = xo;
  let Inst{31}    = RC;
}

// This is the same as XForm_base_r3xo, but the first two operands are swapped
// when code is emitted.
class XForm_base_r3xo_swapped
        <bits<6> opcode, bits<10> xo, dag OL, string asmstr,
        InstrItinClass itin> 
  : I<opcode, OL, asmstr, itin> {
  bits<5> A;
  bits<5> RST;
  bits<5> B;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RST;
  let Inst{11-15} = A;
  let Inst{16-20} = B;
  let Inst{21-30} = xo;
  let Inst{31}    = RC;
}


class XForm_1<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
              InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo<opcode, xo, OL, asmstr, itin, pattern>;

class XForm_6<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
              InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo_swapped<opcode, xo, OL, asmstr, itin> {
  let Pattern = pattern;
}

class XForm_8<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
              InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo<opcode, xo, OL, asmstr, itin, pattern>;

class XForm_10<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo_swapped<opcode, xo, OL, asmstr, itin> {
    let Pattern = pattern;
}

class XForm_11<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo_swapped<opcode, xo, OL, asmstr, itin> {
  let B = 0;
  let Pattern = pattern;
}

class XForm_16<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
               InstrItinClass itin>
         : I<opcode, OL, asmstr, itin> {
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

class XForm_16_ext<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
                   InstrItinClass itin>
  : XForm_16<opcode, xo, OL, asmstr, itin> {
  let L = PPC64;
}

class XForm_17<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
               InstrItinClass itin>
         : I<opcode, OL, asmstr, itin> {
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

class XForm_25<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo<opcode, xo, OL, asmstr, itin, pattern> {
}

class XForm_26<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OL, asmstr, itin, pattern> {
  let A = 0;
}

class XForm_28<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo<opcode, xo, OL, asmstr, itin, pattern> {
}

// DCB_Form - Form X instruction, used for dcb* instructions.
class DCB_Form<bits<10> xo, bits<5> immfield, dag OL, string asmstr, 
                      InstrItinClass itin, list<dag> pattern>
  : I<31, OL, asmstr, itin> {
  bits<5> A;
  bits<5> B;

  let Pattern = pattern;

  let Inst{6-10}  = immfield;
  let Inst{11-15} = A;
  let Inst{16-20} = B;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}


// DSS_Form - Form X instruction, used for altivec dss* instructions.
class DSS_Form<bits<10> xo, dag OL, string asmstr, 
                      InstrItinClass itin, list<dag> pattern>
  : I<31, OL, asmstr, itin> {
  bits<1> T;
  bits<2> STRM;
  bits<5> A;
  bits<5> B;

  let Pattern = pattern;

  let Inst{6}     = T;
  let Inst{7-8}   = 0;
  let Inst{9-10}  = STRM;
  let Inst{11-15} = A;
  let Inst{16-20} = B;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

// 1.7.7 XL-Form
class XLForm_1<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
               InstrItinClass itin>
    : I<opcode, OL, asmstr, itin> {
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

class XLForm_2<bits<6> opcode, bits<10> xo, bit lk, dag OL, string asmstr, 
               InstrItinClass itin, list<dag> pattern>
    : I<opcode, OL, asmstr, itin> {
  bits<5> BO;
  bits<5> BI;
  bits<2> BH;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = BO;
  let Inst{11-15} = BI;
  let Inst{16-18} = 0;
  let Inst{19-20} = BH;
  let Inst{21-30} = xo;
  let Inst{31}    = lk;
}

class XLForm_2_ext<bits<6> opcode, bits<10> xo, bits<5> bo,  bits<5> bi, bit lk,
                  dag OL, string asmstr, InstrItinClass itin, list<dag> pattern>
  : XLForm_2<opcode, xo, lk, OL, asmstr, itin, pattern> {
  let BO = bo;
  let BI = bi;
  let BH = 0;
}

class XLForm_3<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
               InstrItinClass itin>
         : I<opcode, OL, asmstr, itin> {
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
class XFXForm_1<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
                InstrItinClass itin>
         : I<opcode, OL, asmstr, itin> {
  bits<5>  RT;
  bits<10> SPR;

  let Inst{6-10}  = RT;
  let Inst{11}    = SPR{4};
  let Inst{12}    = SPR{3};
  let Inst{13}    = SPR{2};
  let Inst{14}    = SPR{1};
  let Inst{15}    = SPR{0};
  let Inst{16}    = SPR{9};
  let Inst{17}    = SPR{8};
  let Inst{18}    = SPR{7};
  let Inst{19}    = SPR{6};
  let Inst{20}    = SPR{5};
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XFXForm_1_ext<bits<6> opcode, bits<10> xo, bits<10> spr, 
                   dag OL, string asmstr, InstrItinClass itin> 
  : XFXForm_1<opcode, xo, OL, asmstr, itin> {
  let SPR = spr;
}

class XFXForm_3<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
                InstrItinClass itin>
         : I<opcode, OL, asmstr, itin> {
  bits<5>  RT;
   
  let Inst{6-10}  = RT;
  let Inst{11-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XFXForm_5<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
                InstrItinClass itin> 
  : I<opcode, OL, asmstr, itin> {
  bits<8>  FXM;
  bits<5>  ST;
   
  let Inst{6-10}  = ST;
  let Inst{11}    = 0;
  let Inst{12-19} = FXM;
  let Inst{20}    = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XFXForm_5a<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
                 InstrItinClass itin> 
  : I<opcode, OL, asmstr, itin> {
  bits<5>  ST;
  bits<8>  FXM;
   
  let Inst{6-10}  = ST;
  let Inst{11}    = 1;
  let Inst{12-19} = FXM;
  let Inst{20}    = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XFXForm_7<bits<6> opcode, bits<10> xo, dag OL, string asmstr,
                InstrItinClass itin>
  : XFXForm_1<opcode, xo, OL, asmstr, itin>;

class XFXForm_7_ext<bits<6> opcode, bits<10> xo, bits<10> spr, 
                    dag OL, string asmstr, InstrItinClass itin> 
  : XFXForm_7<opcode, xo, OL, asmstr, itin> {
  let SPR = spr;
}

// 1.7.10 XS-Form
class XSForm_1<bits<6> opcode, bits<9> xo, dag OL, string asmstr,
               InstrItinClass itin>
         : I<opcode, OL, asmstr, itin> {
  bits<5> RS;
  bits<5> A;
  bits<6> SH;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RS;
  let Inst{11-15} = A;
  let Inst{16-20} = SH{1-5};
  let Inst{21-29} = xo;
  let Inst{30}    = SH{0};
  let Inst{31}    = RC;
}

// 1.7.11 XO-Form
class XOForm_1<bits<6> opcode, bits<9> xo, bit oe, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
         : I<opcode, OL, asmstr, itin> {
  bits<5> RT;
  bits<5> RA;
  bits<5> RB;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RT;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21}    = oe;
  let Inst{22-30} = xo;
  let Inst{31}    = RC;  
}

class XOForm_3<bits<6> opcode, bits<9> xo, bit oe, 
               dag OL, string asmstr, InstrItinClass itin, list<dag> pattern>
  : XOForm_1<opcode, xo, oe, OL, asmstr, itin, pattern> {
  let RB = 0;
}

// 1.7.12 A-Form
class AForm_1<bits<6> opcode, bits<5> xo, dag OL, string asmstr, 
              InstrItinClass itin, list<dag> pattern>
         : I<opcode, OL, asmstr, itin> {
  bits<5> FRT;
  bits<5> FRA;
  bits<5> FRC;
  bits<5> FRB;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = FRT;
  let Inst{11-15} = FRA;
  let Inst{16-20} = FRB;
  let Inst{21-25} = FRC;
  let Inst{26-30} = xo;
  let Inst{31}    = RC;
}

class AForm_2<bits<6> opcode, bits<5> xo, dag OL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : AForm_1<opcode, xo, OL, asmstr, itin, pattern> {
  let FRC = 0;
}

class AForm_3<bits<6> opcode, bits<5> xo, dag OL, string asmstr,
              InstrItinClass itin, list<dag> pattern> 
  : AForm_1<opcode, xo, OL, asmstr, itin, pattern> {
  let FRB = 0;
}

// 1.7.13 M-Form
class MForm_1<bits<6> opcode, dag OL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
    : I<opcode, OL, asmstr, itin> {
  bits<5> RA;
  bits<5> RS;
  bits<5> RB;
  bits<5> MB;
  bits<5> ME;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RS;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-25} = MB;
  let Inst{26-30} = ME;
  let Inst{31}    = RC;
}

class MForm_2<bits<6> opcode, dag OL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : MForm_1<opcode, OL, asmstr, itin, pattern> {
}

// 1.7.14 MD-Form
class MDForm_1<bits<6> opcode, bits<3> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<opcode, OL, asmstr, itin> {
  bits<5> RS;
  bits<5> RA;
  bits<6> SH;
  bits<6> MBE;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RS;
  let Inst{11-15} = RA;
  let Inst{16-20} = SH{1-5};
  let Inst{21-26} = MBE;
  let Inst{27-29} = xo;
  let Inst{30}    = SH{0};
  let Inst{31}    = RC;
}



// E-1 VA-Form

// VAForm_1 - DACB ordering.
class VAForm_1<bits<6> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<4, OL, asmstr, itin> {
  bits<5> VD;
  bits<5> VA;
  bits<5> VC;
  bits<5> VB;

  let Pattern = pattern;
  
  let Inst{6-10}  = VD;
  let Inst{11-15} = VA;
  let Inst{16-20} = VB;
  let Inst{21-25} = VC;
  let Inst{26-31} = xo;
}

// VAForm_1a - DABC ordering.
class VAForm_1a<bits<6> xo, dag OL, string asmstr,
                InstrItinClass itin, list<dag> pattern>
    : I<4, OL, asmstr, itin> {
  bits<5> VD;
  bits<5> VA;
  bits<5> VB;
  bits<5> VC;

  let Pattern = pattern;
  
  let Inst{6-10}  = VD;
  let Inst{11-15} = VA;
  let Inst{16-20} = VB;
  let Inst{21-25} = VC;
  let Inst{26-31} = xo;
}

class VAForm_2<bits<6> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<4, OL, asmstr, itin> {
  bits<5> VD;
  bits<5> VA;
  bits<5> VB;
  bits<4> SH;

  let Pattern = pattern;
  
  let Inst{6-10}  = VD;
  let Inst{11-15} = VA;
  let Inst{16-20} = VB;
  let Inst{21}    = 0;
  let Inst{22-25} = SH;
  let Inst{26-31} = xo;
}

// E-2 VX-Form
class VXForm_1<bits<11> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<4, OL, asmstr, itin> {
  bits<5> VD;
  bits<5> VA;
  bits<5> VB;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = VD;
  let Inst{11-15} = VA;
  let Inst{16-20} = VB;
  let Inst{21-31} = xo;
}

class VXForm_setzero<bits<11> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : VXForm_1<xo, OL, asmstr, itin, pattern> {
  let VA = VD;
  let VB = VD;
}


class VXForm_2<bits<11> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<4, OL, asmstr, itin> {
  bits<5> VD;
  bits<5> VB;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = VD;
  let Inst{11-15} = 0;
  let Inst{16-20} = VB;
  let Inst{21-31} = xo;
}

class VXForm_3<bits<11> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<4, OL, asmstr, itin> {
  bits<5> VD;
  bits<5> IMM;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = VD;
  let Inst{11-15} = IMM;
  let Inst{16-20} = 0;
  let Inst{21-31} = xo;
}

/// VXForm_4 - VX instructions with "VD,0,0" register fields, like mfvscr.
class VXForm_4<bits<11> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<4, OL, asmstr, itin> {
  bits<5> VD;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = VD;
  let Inst{11-15} = 0;
  let Inst{16-20} = 0;
  let Inst{21-31} = xo;
}

/// VXForm_5 - VX instructions with "0,0,VB" register fields, like mtvscr.
class VXForm_5<bits<11> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<4, OL, asmstr, itin> {
  bits<5> VB;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = 0;
  let Inst{11-15} = 0;
  let Inst{16-20} = VB;
  let Inst{21-31} = xo;
}

// E-4 VXR-Form
class VXRForm_1<bits<10> xo, dag OL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<4, OL, asmstr, itin> {
  bits<5> VD;
  bits<5> VA;
  bits<5> VB;
  bit RC = 0;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = VD;
  let Inst{11-15} = VA;
  let Inst{16-20} = VB;
  let Inst{21}    = RC;
  let Inst{22-31} = xo;
}

//===----------------------------------------------------------------------===//
class Pseudo<dag OL, string asmstr, list<dag> pattern>
    : I<0, OL, asmstr, NoItinerary> {
  let PPC64 = 0;
  let VMX = 0;
  let Pattern = pattern;
  let Inst{31-0} = 0;
}
