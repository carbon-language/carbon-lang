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
def Zimm16 : Format<4>;
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

class PPC32Inst : Instruction {
	field bits<32> Inst;
	bits<3> ArgCount;
	bits<5> Arg0Type;
	bits<5> Arg1Type;
	bits<5> Arg2Type;
	bits<5> Arg3Type;
	bits<5> Arg4Type;
	bit PPC64;
	bit VMX;

	let Namespace = "PPC32";
}

//===----------------------------------------------------------------------===//
//
// PowerPC instruction formats

class PPC32I<string name, bits<6> opcode, bit ppc64, bit vmx> : Instruction {
  field bits<32> Inst;
  field bits<6> Opcode = opcode;

  bits<3> ArgCount;
  bits<5> Arg0Type;
  bits<5> Arg1Type;
  bits<5> Arg2Type;
  bits<5> Arg3Type;
  bits<5> Arg4Type;
  bit PPC64 = ppc64;
  bit VMX = vmx;

  let Name = name;
  let Inst{0-5} = Opcode;
}

class XForm_base_r3xo<string name, bits<6> opcode, bits<10> xo, bit ppc64, 
                      bit vmx> : PPC32I<name, opcode, ppc64, vmx> {
  let ArgCount = 3;
  field bits<5>  A;
  field bits<5>  B;
  field bits<5>  C;
  field bits<10> D = xo;
  field bit     Rc = 0;

  let ArgCount = 3;
  let Arg0Type = Gpr.Value;
  let Arg1Type = Gpr.Value;
  let Arg2Type = Gpr.Value;
  let Arg3Type = 0;
  let Arg4Type = 0;

  let Inst{6-10}  = A;
  let Inst{11-15} = B;
  let Inst{16-20} = C;
  let Inst{21-30} = D;
  let Inst{31}    = Rc;
}

class XForm_6<string name, bits<6> opcode, bits<10> xo, bit rc, bit ppc64, 
              bit vmx> : XForm_base_r3xo<name, opcode, xo, ppc64, vmx> {
  let Rc = rc;
}

class XForm_7<string name, bits<6> opcode, bits<10> xo, bit ppc64, bit vmx> 
  : XForm_6<name, opcode, xo, 1, ppc64, vmx>;

class XForm_10<string name, bits<6> opcode, bits<10> xo, bit rc, bit ppc64, 
               bit vmx> : XForm_base_r3xo<name, opcode, xo, ppc64, vmx> {
  let Rc = rc;
  let Arg2Type = Imm5.Value;
}

class DForm_base<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : PPC32I<name, opcode, ppc64, vmx> {
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
  let Arg2Type = Zimm16.Value;
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

class DForm_4<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : DForm_1<name, opcode, ppc64, vmx>;

class DForm_7<string name, bits<6> opcode, bit ppc64, bit vmx> 
  : DForm_base<name, opcode, ppc64, vmx> {
  let Arg1Type = Imm5.Value;
}

//===----------------------------------------------------------------------===//

class PPC32InstPattern1 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<5> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<16> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {20-16} = opconstant1;
	let Inst {25-21} = operand0;
	let Inst {15-0} = operand1;
}

class PPC32InstPattern2 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<5> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {10-0} = opconstant1;
	let Inst {25-21} = operand0;
	let Inst {20-16} = operand1;
	let Inst {15-11} = operand2;
}

class PPC32InstPattern3 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<16> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {15-0} = opconstant1;
	let Inst {25-21} = operand0;
	let Inst {20-16} = operand1;
}

class PPC32InstPattern6 <string name, Format OperandType0, bits<6> opconstant0, bits<2> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 1;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = 0;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<24> operand0;


	let Inst {31-26} = opconstant0;
	let Inst {1-0} = opconstant1;
	let Inst {25-2} = operand0;
}

class PPC32InstPattern7 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bits<2> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<14> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {1-0} = opconstant1;
	let Inst {25-21} = operand0;
	let Inst {20-16} = operand1;
	let Inst {15-2} = operand2;
}

class PPC32InstPattern8 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bits<3> opconstant1, bits<11> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<2> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {15-13} = opconstant1;
	let Inst {10-0} = opconstant2;
	let Inst {25-21} = operand0;
	let Inst {20-16} = operand1;
	let Inst {12-11} = operand2;
}

class PPC32InstPattern9 <string name, Format OperandType0, Format OperandType1, bits<11> opconstant0, bits<2> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<14> operand1;


	let Inst {31-21} = opconstant0;
	let Inst {1-0} = opconstant1;
	let Inst {20-16} = operand0;
	let Inst {15-2} = operand1;
}

class PPC32InstPattern10 <string name, Format OperandType0, bits<16> opconstant0, bits<2> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 1;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = 0;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<14> operand0;


	let Inst {31-16} = opconstant0;
	let Inst {1-0} = opconstant1;
	let Inst {15-2} = operand0;
}

class PPC32InstPattern11 <string name, Format OperandType0, bits<19> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 1;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = 0;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<2> operand0;


	let Inst {31-13} = opconstant0;
	let Inst {10-0} = opconstant1;
	let Inst {12-11} = operand0;
}

class PPC32InstPattern12 <string name, Format OperandType0, Format OperandType1, bits<11> opconstant0, bits<3> opconstant1, bits<11> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<2> operand1;


	let Inst {31-21} = opconstant0;
	let Inst {15-13} = opconstant1;
	let Inst {10-0} = opconstant2;
	let Inst {20-16} = operand0;
	let Inst {12-11} = operand1;
}

class PPC32InstPattern13 <string name, Format OperandType0, Format OperandType1, Format OperandType2, Format OperandType3, bits<6> opconstant0, bits<1> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 4;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = OperandType3.Value;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<3> operand0;
	bits<1> operand1;
	bits<5> operand2;
	bits<16> operand3;


	let Inst {31-26} = opconstant0;
	let Inst {22} = opconstant1;
	let Inst {25-23} = operand0;
	let Inst {21} = operand1;
	let Inst {20-16} = operand2;
	let Inst {15-0} = operand3;
}

class PPC32InstPattern14 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bits<2> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<3> operand0;
	bits<5> operand1;
	bits<16> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {22-21} = opconstant1;
	let Inst {25-23} = operand0;
	let Inst {20-16} = operand1;
	let Inst {15-0} = operand2;
}

class PPC32InstPattern15 <string name, Format OperandType0, Format OperandType1, Format OperandType2, Format OperandType3, bits<6> opconstant0, bits<1> opconstant1, bits<11> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 4;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = OperandType3.Value;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<3> operand0;
	bits<1> operand1;
	bits<5> operand2;
	bits<5> operand3;


	let Inst {31-26} = opconstant0;
	let Inst {22} = opconstant1;
	let Inst {10-0} = opconstant2;
	let Inst {25-23} = operand0;
	let Inst {21} = operand1;
	let Inst {20-16} = operand2;
	let Inst {15-11} = operand3;
}

class PPC32InstPattern16 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bits<2> opconstant1, bits<11> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<3> operand0;
	bits<5> operand1;
	bits<5> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {22-21} = opconstant1;
	let Inst {10-0} = opconstant2;
	let Inst {25-23} = operand0;
	let Inst {20-16} = operand1;
	let Inst {15-11} = operand2;
}

class PPC32InstPattern17 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<16> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {15-0} = opconstant1;
	let Inst {20-16} = operand0;
	let Inst {25-21} = operand1;
}

class PPC32InstPattern18 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bits<5> opconstant1, bits<6> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<5> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {15-11} = opconstant1;
	let Inst {5-0} = opconstant2;
	let Inst {25-21} = operand0;
	let Inst {20-16} = operand1;
	let Inst {10-6} = operand2;
}

class PPC32InstPattern19 <string name, Format OperandType0, Format OperandType1, Format OperandType2, Format OperandType3, bits<6> opconstant0, bits<6> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 4;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = OperandType3.Value;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<5> operand2;
	bits<5> operand3;


	let Inst {31-26} = opconstant0;
	let Inst {5-0} = opconstant1;
	let Inst {25-21} = operand0;
	let Inst {20-16} = operand1;
	let Inst {10-6} = operand2;
	let Inst {15-11} = operand3;
}

class PPC32InstPattern20 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<5> opconstant1, bits<11> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {20-16} = opconstant1;
	let Inst {10-0} = opconstant2;
	let Inst {25-21} = operand0;
	let Inst {15-11} = operand1;
}

class PPC32InstPattern21 <string name, Format OperandType0, bits<6> opconstant0, bits<21> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 1;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = 0;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;


	let Inst {31-26} = opconstant0;
	let Inst {20-0} = opconstant1;
	let Inst {25-21} = operand0;
}

class PPC32InstPattern22 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<18> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<3> operand0;
	bits<5> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {17-0} = opconstant1;
	let Inst {25-23} = operand0;
	let Inst {22-18} = operand1;
}

class PPC32InstPattern23 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<7> opconstant1, bits<12> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<3> operand0;
	bits<4> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {22-16} = opconstant1;
	let Inst {11-0} = opconstant2;
	let Inst {25-23} = operand0;
	let Inst {15-12} = operand1;
}

class PPC32InstPattern24 <string name, Format OperandType0, Format OperandType1, bits<7> opconstant0, bits<1> opconstant1, bits<11> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<8> operand0;
	bits<5> operand1;


	let Inst {31-25} = opconstant0;
	let Inst {16} = opconstant1;
	let Inst {10-0} = opconstant2;
	let Inst {24-17} = operand0;
	let Inst {15-11} = operand1;
}

class PPC32InstPattern25 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<16> operand1;
	bits<5> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {25-21} = operand0;
	let Inst {15-0} = operand1;
	let Inst {20-16} = operand2;
}

class PPC32InstPattern26 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bits<2> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<14> operand1;
	bits<5> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {1-0} = opconstant1;
	let Inst {25-21} = operand0;
	let Inst {15-2} = operand1;
	let Inst {20-16} = operand2;
}

class PPC32InstPattern27 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<2> opconstant1, bits<18> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<3> operand0;
	bits<3> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {22-21} = opconstant1;
	let Inst {17-0} = opconstant2;
	let Inst {25-23} = operand0;
	let Inst {20-18} = operand1;
}

class PPC32InstPattern28 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<10> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {10-0} = opconstant1;
	let Inst {25-21} = operand0;
	let Inst {20-11} = operand1;
}

class PPC32InstPattern29 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<10> operand0;
	bits<5> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {10-0} = opconstant1;
	let Inst {20-11} = operand0;
	let Inst {25-21} = operand1;
}

class PPC32InstPattern30 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<1> opconstant1, bits<12> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<8> operand0;
	bits<5> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {20} = opconstant1;
	let Inst {11-0} = opconstant2;
	let Inst {19-12} = operand0;
	let Inst {25-21} = operand1;
}

class PPC32InstPattern31 <string name, Format OperandType0, bits<6> opconstant0, bits<23> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 1;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = 0;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<3> operand0;


	let Inst {31-26} = opconstant0;
	let Inst {22-0} = opconstant1;
	let Inst {25-23} = operand0;
}

class PPC32InstPattern32 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<1> opconstant1, bits<12> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<8> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {20} = opconstant1;
	let Inst {11-0} = opconstant2;
	let Inst {25-21} = operand0;
	let Inst {19-12} = operand1;
}

class PPC32InstPattern33 <string name, bits<32> opconstant0, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 0;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = 0;
	let Arg1Type = 0;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;


	let Inst {31-0} = opconstant0;
}

class PPC32InstPattern34 <string name, Format OperandType0, Format OperandType1, Format OperandType2, Format OperandType3, Format OperandType4, bits<6> opconstant0, bits<1> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 5;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = OperandType3.Value;
	let Arg4Type = OperandType4.Value;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<5> operand2;
	bits<5> operand3;
	bits<5> operand4;


	let Inst {31-26} = opconstant0;
	let Inst {0} = opconstant1;
	let Inst {20-16} = operand0;
	let Inst {25-21} = operand1;
	let Inst {15-11} = operand2;
	let Inst {10-6} = operand3;
	let Inst {5-1} = operand4;
}

class PPC32InstPattern35 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<5> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {10-0} = opconstant1;
	let Inst {25-21} = operand0;
	let Inst {15-11} = operand1;
	let Inst {20-16} = operand2;
}

class PPC32InstPattern36 <string name, Format OperandType0, bits<9> opconstant0, bits<21> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 1;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = 0;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<2> operand0;


	let Inst {31-23} = opconstant0;
	let Inst {20-0} = opconstant1;
	let Inst {22-21} = operand0;
}

class PPC32InstPattern37 <string name, Format OperandType0, Format OperandType1, bits<11> opconstant0, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<16> operand1;


	let Inst {31-21} = opconstant0;
	let Inst {20-16} = operand0;
	let Inst {15-0} = operand1;
}

class PPC32InstPattern38 <string name, Format OperandType0, Format OperandType1, bits<11> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;


	let Inst {31-21} = opconstant0;
	let Inst {10-0} = opconstant1;
	let Inst {20-16} = operand0;
	let Inst {15-11} = operand1;
}

class PPC32InstPattern39 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<7> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<4> operand2;


	let Inst {31-25} = opconstant0;
	let Inst {10-0} = opconstant1;
	let Inst {20-16} = operand0;
	let Inst {15-11} = operand1;
	let Inst {24-21} = operand2;
}

class PPC32InstPattern40 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<4> opconstant1, bits<16> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<1> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {20-17} = opconstant1;
	let Inst {15-0} = opconstant2;
	let Inst {25-21} = operand0;
	let Inst {16} = operand1;
}

class PPC32InstPattern41 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<1> opconstant1, bits<16> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<4> operand0;
	bits<5> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {20} = opconstant1;
	let Inst {15-0} = opconstant2;
	let Inst {19-16} = operand0;
	let Inst {25-21} = operand1;
}

class PPC32InstPattern42 <string name, Format OperandType0, Format OperandType1, bits<6> opconstant0, bits<1> opconstant1, bits<16> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<4> operand1;


	let Inst {31-26} = opconstant0;
	let Inst {20} = opconstant1;
	let Inst {15-0} = opconstant2;
	let Inst {25-21} = operand0;
	let Inst {19-16} = operand1;
}

class PPC32InstPattern43 <string name, Format OperandType0, bits<16> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 1;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = 0;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;


	let Inst {31-16} = opconstant0;
	let Inst {10-0} = opconstant1;
	let Inst {15-11} = operand0;
}

class PPC32InstPattern44 <string name, Format OperandType0, Format OperandType1, bits<10> opconstant0, bits<5> opconstant1, bits<11> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 2;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<1> operand1;


	let Inst {31-22} = opconstant0;
	let Inst {20-16} = opconstant1;
	let Inst {10-0} = opconstant2;
	let Inst {15-11} = operand0;
	let Inst {21} = operand1;
}

class PPC32InstPattern45 <string name, Format OperandType0, bits<6> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 1;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = 0;
	let Arg2Type = 0;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<15> operand0;


	let Inst {31-26} = opconstant0;
	let Inst {10-0} = opconstant1;
	let Inst {25-11} = operand0;
}

class PPC32InstPattern46 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<9> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 3;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = 0;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<2> operand2;


	let Inst {31-23} = opconstant0;
	let Inst {10-0} = opconstant1;
	let Inst {20-16} = operand0;
	let Inst {15-11} = operand1;
	let Inst {22-21} = operand2;
}

class PPC32InstPattern47 <string name, Format OperandType0, Format OperandType1, Format OperandType2, Format OperandType3, bits<6> opconstant0, bits<6> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 4;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = OperandType3.Value;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<5> operand2;
	bits<5> operand3;


	let Inst {31-26} = opconstant0;
	let Inst {5-0} = opconstant1;
	let Inst {25-21} = operand0;
	let Inst {20-16} = operand1;
	let Inst {15-11} = operand2;
	let Inst {10-6} = operand3;
}

class PPC32InstPattern48 <string name, Format OperandType0, Format OperandType1, Format OperandType2, Format OperandType3, bits<6> opconstant0, bits<1> opconstant1, bits<6> opconstant2, bit ppc64, bit vmx> : PPC32Inst {
	let Name = name;
	let ArgCount = 4;
	let PPC64 = ppc64;
	let VMX =vmx;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType1.Value;
	let Arg2Type = OperandType2.Value;
	let Arg3Type = OperandType3.Value;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<5> operand2;
	bits<4> operand3;


	let Inst {31-26} = opconstant0;
	let Inst {10} = opconstant1;
	let Inst {5-0} = opconstant2;
	let Inst {25-21} = operand0;
	let Inst {20-16} = operand1;
	let Inst {15-11} = operand2;
	let Inst {9-6} = operand3;
}

class PPC32InstPatternPseudo <string name, Format OperandType0> : PPC32Inst {
	let Name = name;
	let ArgCount = 0;
	let PPC64 = 0;
	let VMX = 0;

	let Arg0Type = OperandType0.Value;
	let Arg1Type = OperandType0.Value;
	let Arg2Type = OperandType0.Value;
	let Arg3Type = OperandType0.Value;
	let Arg4Type = 0;
	let PPC64 = 0;
	let VMX = 0;
	bits<5> operand0;
	bits<5> operand1;
	bits<5> operand2;
	bits<4> operand3;


	let Inst {31-0} = 0;
}
