//===- PowerPCInstrInfo.td - Describe the PowerPC Instruction Set -*- C++ -*-=//
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

class PPC32InstPattern0 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bit ppc64, bit vmx> : PPC32Inst {
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
	bits<16> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {25-21} = operand0;
	let Inst {20-16} = operand1;
	let Inst {15-0} = operand2;
}

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

class PPC32InstPattern4 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bit ppc64, bit vmx> : PPC32Inst {
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
	bits<16> operand2;


	let Inst {31-26} = opconstant0;
	let Inst {20-16} = operand0;
	let Inst {25-21} = operand1;
	let Inst {15-0} = operand2;
}

class PPC32InstPattern5 <string name, Format OperandType0, Format OperandType1, Format OperandType2, bits<6> opconstant0, bits<11> opconstant1, bit ppc64, bit vmx> : PPC32Inst {
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
	let Inst {20-16} = operand0;
	let Inst {25-21} = operand1;
	let Inst {15-11} = operand2;
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


let isCall = 1 in
  // All calls clobber the non-callee saved registers...
  let Defs = [R0, R2, R3, R4, R5, R6, R7, R8, R9, R10] in {
    def CALLpcrel : PPC32InstPattern6 <"bl", PCRelimm24, 18, 1, 0, 0>;
    def CALLindirect : PPC32InstPattern3 <"bctrl", Imm5, Imm5, 19, 1057, 0, 0>;
 }

let isTerminator = 1, isReturn = 1 in
	def BLR : PPC32InstPattern11 <"blr", Imm2, 160768, 32, 0, 0>;

// Pseudo-instructions:
def PHI : PPC32InstPatternPseudo<"PHI", Pseudo>;          // PHI node...
def ADJCALLSTACKDOWN : PPC32InstPatternPseudo<"ADJCALLSTACKDOWN", Pseudo>;
def ADJCALLSTACKUP : PPC32InstPatternPseudo<"ADJCALLSTACKUP", Pseudo>;
def MovePCtoLR : PPC32InstPatternPseudo<"MovePCtoLR", Pseudo>;

def LOADLoAddr : PPC32InstPattern0 <"addi", Gpr, Gpr0, Simm16, 14, 0, 0>;
def LOADHiAddr : PPC32InstPattern0 <"addis", Gpr, Gpr0, Simm16, 15, 0, 0>;

def ADDI : PPC32InstPattern0 <"addi", Gpr, Gpr0, Simm16, 14, 0, 0>;
def LI : PPC32InstPattern1 <"li", Gpr, Simm16, 14, 0, 0, 0>;
def ADDIS : PPC32InstPattern0 <"addis", Gpr, Gpr0, Simm16, 15, 0, 0>;
def LIS : PPC32InstPattern1 <"lis", Gpr, Simm16, 15, 0, 0, 0>;
def ADDIC : PPC32InstPattern0 <"addic", Gpr, Gpr, Simm16, 12, 0, 0>;
def ADDICo : PPC32InstPattern0 <"addic.", Gpr, Gpr, Simm16, 13, 0, 0>;
def ADD : PPC32InstPattern2 <"add", Gpr, Gpr, Gpr, 31, 532, 0, 0>;
def ADDo : PPC32InstPattern2 <"add.", Gpr, Gpr, Gpr, 31, 533, 0, 0>;
def ADDO : PPC32InstPattern2 <"addo", Gpr, Gpr, Gpr, 31, 532, 0, 0>;
def ADDOo : PPC32InstPattern2 <"addo.", Gpr, Gpr, Gpr, 31, 533, 0, 0>;
def ADDC : PPC32InstPattern2 <"addc", Gpr, Gpr, Gpr, 31, 20, 0, 0>;
def ADDCo : PPC32InstPattern2 <"addc.", Gpr, Gpr, Gpr, 31, 21, 0, 0>;
def ADDCO : PPC32InstPattern2 <"addco", Gpr, Gpr, Gpr, 31, 20, 0, 0>;
def ADDCOo : PPC32InstPattern2 <"addco.", Gpr, Gpr, Gpr, 31, 21, 0, 0>;
def ADDE : PPC32InstPattern2 <"adde", Gpr, Gpr, Gpr, 31, 276, 0, 0>;
def ADDEo : PPC32InstPattern2 <"adde.", Gpr, Gpr, Gpr, 31, 277, 0, 0>;
def ADDEO : PPC32InstPattern2 <"addeo", Gpr, Gpr, Gpr, 31, 276, 0, 0>;
def ADDEOo : PPC32InstPattern2 <"addeo.", Gpr, Gpr, Gpr, 31, 277, 0, 0>;
def ADDME : PPC32InstPattern3 <"addme", Gpr, Gpr, 31, 468, 0, 0>;
def ADDMEo : PPC32InstPattern3 <"addme.", Gpr, Gpr, 31, 469, 0, 0>;
def ADDMEO : PPC32InstPattern3 <"addmeo", Gpr, Gpr, 31, 1492, 0, 0>;
def ADDMEOo : PPC32InstPattern3 <"addmeo.", Gpr, Gpr, 31, 1493, 0, 0>;
def ADDZE : PPC32InstPattern3 <"addze", Gpr, Gpr, 31, 404, 0, 0>;
def ADDZEo : PPC32InstPattern3 <"addze.", Gpr, Gpr, 31, 405, 0, 0>;
def ADDZEO : PPC32InstPattern3 <"addzeo", Gpr, Gpr, 31, 1428, 0, 0>;
def ADDZEOo : PPC32InstPattern3 <"addzeo.", Gpr, Gpr, 31, 1429, 0, 0>;
def ANDIo : PPC32InstPattern4 <"andi.", Gpr, Gpr, Zimm16, 28, 0, 0>;
def ANDISo : PPC32InstPattern4 <"andis.", Gpr, Gpr, Zimm16, 29, 0, 0>;
def AND : PPC32InstPattern5 <"and", Gpr, Gpr, Gpr, 31, 56, 0, 0>;
def ANDo : PPC32InstPattern5 <"and.", Gpr, Gpr, Gpr, 31, 57, 0, 0>;
def ANDC : PPC32InstPattern5 <"andc", Gpr, Gpr, Gpr, 31, 120, 0, 0>;
def ANDCo : PPC32InstPattern5 <"andc.", Gpr, Gpr, Gpr, 31, 121, 0, 0>;
def B : PPC32InstPattern6 <"b", PCRelimm24, 18, 0, 0, 0>;
def BA : PPC32InstPattern6 <"ba", Imm24, 18, 0, 0, 0>;
def BL : PPC32InstPattern6 <"bl", PCRelimm24, 18, 1, 0, 0>;
def BLA : PPC32InstPattern6 <"bla", Imm24, 18, 1, 0, 0>;
def BC : PPC32InstPattern7 <"bc", Imm5, Imm5, PCRelimm14, 16, 0, 0, 0>;
def BCA : PPC32InstPattern7 <"bca", Imm5, Imm5, Imm14, 16, 0, 0, 0>;
def BCL : PPC32InstPattern7 <"bcl", Imm5, Imm5, PCRelimm14, 16, 1, 0, 0>;
def BCLA : PPC32InstPattern7 <"bcla", Imm5, Imm5, Imm14, 16, 1, 0, 0>;
def BCCTR : PPC32InstPattern8 <"bcctr", Imm5, Imm5, Imm2, 19, 0, 32, 0, 0>;
def BCCTRL : PPC32InstPattern8 <"bcctrl", Imm5, Imm5, Imm2, 19, 0, 33, 0, 0>;
def BCLR : PPC32InstPattern8 <"bclr", Imm5, Imm5, Imm2, 19, 0, 32, 0, 0>;
def BCLRL : PPC32InstPattern8 <"bclrl", Imm5, Imm5, Imm2, 19, 0, 33, 0, 0>;
def BT : PPC32InstPattern9 <"bt", Imm5, PCRelimm14, 524, 0, 0, 0>;
def BTL : PPC32InstPattern9 <"btl", Imm5, PCRelimm14, 524, 1, 0, 0>;
def BF : PPC32InstPattern9 <"bf", Imm5, PCRelimm14, 516, 0, 0, 0>;
def BFL : PPC32InstPattern9 <"bfl", Imm5, PCRelimm14, 516, 1, 0, 0>;
def BDNZ : PPC32InstPattern10 <"bdnz", PCRelimm14, 16896, 0, 0, 0>;
def BDNZL : PPC32InstPattern10 <"bdnzl", PCRelimm14, 16896, 1, 0, 0>;
def BDNZT : PPC32InstPattern9 <"bdnzt", Imm5, PCRelimm14, 520, 0, 0, 0>;
def BDNZTL : PPC32InstPattern9 <"bdnztl", Imm5, PCRelimm14, 520, 1, 0, 0>;
def BDNZF : PPC32InstPattern9 <"bdnzf", Imm5, PCRelimm14, 512, 0, 0, 0>;
def BDNZFL : PPC32InstPattern9 <"bdnzfl", Imm5, PCRelimm14, 512, 1, 0, 0>;
def BDZ : PPC32InstPattern10 <"bdz", PCRelimm14, 16960, 0, 0, 0>;
def BDZL : PPC32InstPattern10 <"bdzl", PCRelimm14, 16960, 1, 0, 0>;
def BDZT : PPC32InstPattern9 <"bdzt", Imm5, PCRelimm14, 522, 0, 0, 0>;
def BDZTL : PPC32InstPattern9 <"bdztl", Imm5, PCRelimm14, 522, 1, 0, 0>;
def BDZF : PPC32InstPattern9 <"bdzf", Imm5, PCRelimm14, 514, 0, 0, 0>;
def BDZFL : PPC32InstPattern9 <"bdzfl", Imm5, PCRelimm14, 514, 1, 0, 0>;
def BTA : PPC32InstPattern9 <"bta", Imm5, Imm14, 524, 0, 0, 0>;
def BTLA : PPC32InstPattern9 <"btla", Imm5, Imm14, 524, 1, 0, 0>;
def BFA : PPC32InstPattern9 <"bfa", Imm5, Imm14, 516, 0, 0, 0>;
def BFLA : PPC32InstPattern9 <"bfla", Imm5, Imm14, 516, 1, 0, 0>;
def BDNZA : PPC32InstPattern10 <"bdnza", Imm14, 16896, 0, 0, 0>;
def BDNZLA : PPC32InstPattern10 <"bdnzla", Imm14, 16896, 1, 0, 0>;
def BDNZTA : PPC32InstPattern9 <"bdnzta", Imm5, Imm14, 520, 0, 0, 0>;
def BDNZTLA : PPC32InstPattern9 <"bdnztla", Imm5, Imm14, 520, 1, 0, 0>;
def BDNZFA : PPC32InstPattern9 <"bdnzfa", Imm5, Imm14, 512, 0, 0, 0>;
def BDNZFLA : PPC32InstPattern9 <"bdnzfla", Imm5, Imm14, 512, 1, 0, 0>;
def BDZA : PPC32InstPattern10 <"bdza", Imm14, 16960, 0, 0, 0>;
def BDZLA : PPC32InstPattern10 <"bdzla", Imm14, 16960, 1, 0, 0>;
def BDZTA : PPC32InstPattern9 <"bdzta", Imm5, Imm14, 522, 0, 0, 0>;
def BDZTLA : PPC32InstPattern9 <"bdztla", Imm5, Imm14, 522, 1, 0, 0>;
def BDZFA : PPC32InstPattern9 <"bdzfa", Imm5, Imm14, 514, 0, 0, 0>;
def BDZFLA : PPC32InstPattern9 <"bdzfla", Imm5, Imm14, 514, 1, 0, 0>;
def BLRL : PPC32InstPattern11 <"blrl", Imm2, 160768, 33, 0, 0>;
def BTLR : PPC32InstPattern12 <"btlr", Imm5, Imm2, 620, 0, 32, 0, 0>;
def BTLRL : PPC32InstPattern12 <"btlrl", Imm5, Imm2, 620, 0, 33, 0, 0>;
def BFLR : PPC32InstPattern12 <"bflr", Imm5, Imm2, 612, 0, 32, 0, 0>;
def BFLRL : PPC32InstPattern12 <"bflrl", Imm5, Imm2, 612, 0, 33, 0, 0>;
def BDNZLR : PPC32InstPattern11 <"bdnzlr", Imm2, 159744, 32, 0, 0>;
def BDNZLRL : PPC32InstPattern11 <"bdnzlrl", Imm2, 159744, 33, 0, 0>;
def BDNZTLR : PPC32InstPattern12 <"bdnztlr", Imm5, Imm2, 616, 0, 32, 0, 0>;
def BDNZTLRL : PPC32InstPattern12 <"bdnztlrl", Imm5, Imm2, 616, 0, 33, 0, 0>;
def BDNZFLR : PPC32InstPattern12 <"bdnzflr", Imm5, Imm2, 608, 0, 32, 0, 0>;
def BDNZFLRL : PPC32InstPattern12 <"bdnzflrl", Imm5, Imm2, 608, 0, 33, 0, 0>;
def BDZLR : PPC32InstPattern11 <"bdzlr", Imm2, 160256, 32, 0, 0>;
def BDZLRL : PPC32InstPattern11 <"bdzlrl", Imm2, 160256, 33, 0, 0>;
def BDZTLR : PPC32InstPattern12 <"bdztlr", Imm5, Imm2, 618, 0, 32, 0, 0>;
def BDZTLRL : PPC32InstPattern12 <"bdztlrl", Imm5, Imm2, 618, 0, 33, 0, 0>;
def BDZFLR : PPC32InstPattern12 <"bdzflr", Imm5, Imm2, 610, 0, 32, 0, 0>;
def BDZFLRL : PPC32InstPattern12 <"bdzflrl", Imm5, Imm2, 610, 0, 33, 0, 0>;
def BCTR : PPC32InstPattern3 <"bctr", Imm5, Imm5, 19, 1056, 0, 0>;
def BCTRL : PPC32InstPattern3 <"bctrl", Imm5, Imm5, 19, 1057, 0, 0>;
def BTCTR : PPC32InstPattern12 <"btctr", Imm5, Imm2, 620, 0, 32, 0, 0>;
def BTCTRL : PPC32InstPattern12 <"btctrl", Imm5, Imm2, 620, 0, 33, 0, 0>;
def BFCTR : PPC32InstPattern12 <"bfctr", Imm5, Imm2, 612, 0, 32, 0, 0>;
def BFCTRL : PPC32InstPattern12 <"bfctrl", Imm5, Imm2, 612, 0, 33, 0, 0>;
def BLT : PPC32InstPattern9 <"blt", Crf, PCRelimm14, 524, 0, 0, 0>;
def BLTL : PPC32InstPattern9 <"bltl", Crf, PCRelimm14, 524, 1, 0, 0>;
def BLE : PPC32InstPattern9 <"ble", Crf, PCRelimm14, 516, 0, 0, 0>;
def BLEL : PPC32InstPattern9 <"blel", Crf, PCRelimm14, 516, 1, 0, 0>;
def BEQ : PPC32InstPattern9 <"beq", Crf, PCRelimm14, 524, 0, 0, 0>;
def BEQL : PPC32InstPattern9 <"beql", Crf, PCRelimm14, 524, 1, 0, 0>;
def BGE : PPC32InstPattern9 <"bge", Crf, PCRelimm14, 516, 0, 0, 0>;
def BGEL : PPC32InstPattern9 <"bgel", Crf, PCRelimm14, 516, 1, 0, 0>;
def BGT : PPC32InstPattern9 <"bgt", Crf, PCRelimm14, 524, 0, 0, 0>;
def BGTL : PPC32InstPattern9 <"bgtl", Crf, PCRelimm14, 524, 1, 0, 0>;
def BNL : PPC32InstPattern9 <"bnl", Crf, PCRelimm14, 516, 0, 0, 0>;
def BNLL : PPC32InstPattern9 <"bnll", Crf, PCRelimm14, 516, 1, 0, 0>;
def BNE : PPC32InstPattern9 <"bne", Crf, PCRelimm14, 516, 0, 0, 0>;
def BNEL : PPC32InstPattern9 <"bnel", Crf, PCRelimm14, 516, 1, 0, 0>;
def BNG : PPC32InstPattern9 <"bng", Crf, PCRelimm14, 516, 0, 0, 0>;
def BNGL : PPC32InstPattern9 <"bngl", Crf, PCRelimm14, 516, 1, 0, 0>;
def BSO : PPC32InstPattern9 <"bso", Crf, PCRelimm14, 524, 0, 0, 0>;
def BSOL : PPC32InstPattern9 <"bsol", Crf, PCRelimm14, 524, 1, 0, 0>;
def BNS : PPC32InstPattern9 <"bns", Crf, PCRelimm14, 516, 0, 0, 0>;
def BNSL : PPC32InstPattern9 <"bnsl", Crf, PCRelimm14, 516, 1, 0, 0>;
def BUN : PPC32InstPattern9 <"bun", Crf, PCRelimm14, 524, 0, 0, 0>;
def BUNL : PPC32InstPattern9 <"bunl", Crf, PCRelimm14, 524, 1, 0, 0>;
def BNU : PPC32InstPattern9 <"bnu", Crf, PCRelimm14, 516, 0, 0, 0>;
def BNUL : PPC32InstPattern9 <"bnul", Crf, PCRelimm14, 516, 1, 0, 0>;
def BLTA : PPC32InstPattern9 <"blta", Crf, Imm14, 524, 0, 0, 0>;
def BLTLA : PPC32InstPattern9 <"bltla", Crf, Imm14, 524, 1, 0, 0>;
def BLEA : PPC32InstPattern9 <"blea", Crf, Imm14, 516, 0, 0, 0>;
def BLELA : PPC32InstPattern9 <"blela", Crf, Imm14, 516, 1, 0, 0>;
def BEQA : PPC32InstPattern9 <"beqa", Crf, Imm14, 524, 0, 0, 0>;
def BEQLA : PPC32InstPattern9 <"beqla", Crf, Imm14, 524, 1, 0, 0>;
def BGEA : PPC32InstPattern9 <"bgea", Crf, Imm14, 516, 0, 0, 0>;
def BGELA : PPC32InstPattern9 <"bgela", Crf, Imm14, 516, 1, 0, 0>;
def BGTA : PPC32InstPattern9 <"bgta", Crf, Imm14, 524, 0, 0, 0>;
def BGTLA : PPC32InstPattern9 <"bgtla", Crf, Imm14, 524, 1, 0, 0>;
def BNLA : PPC32InstPattern9 <"bnla", Crf, Imm14, 516, 0, 0, 0>;
def BNLLA : PPC32InstPattern9 <"bnlla", Crf, Imm14, 516, 1, 0, 0>;
def BNEA : PPC32InstPattern9 <"bnea", Crf, Imm14, 516, 0, 0, 0>;
def BNELA : PPC32InstPattern9 <"bnela", Crf, Imm14, 516, 1, 0, 0>;
def BNGA : PPC32InstPattern9 <"bnga", Crf, Imm14, 516, 0, 0, 0>;
def BNGLA : PPC32InstPattern9 <"bngla", Crf, Imm14, 516, 1, 0, 0>;
def BSOA : PPC32InstPattern9 <"bsoa", Crf, Imm14, 524, 0, 0, 0>;
def BSOLA : PPC32InstPattern9 <"bsola", Crf, Imm14, 524, 1, 0, 0>;
def BNSA : PPC32InstPattern9 <"bnsa", Crf, Imm14, 516, 0, 0, 0>;
def BNSLA : PPC32InstPattern9 <"bnsla", Crf, Imm14, 516, 1, 0, 0>;
def BUNA : PPC32InstPattern9 <"buna", Crf, Imm14, 524, 0, 0, 0>;
def BUNLA : PPC32InstPattern9 <"bunla", Crf, Imm14, 524, 1, 0, 0>;
def BNUA : PPC32InstPattern9 <"bnua", Crf, Imm14, 516, 0, 0, 0>;
def BNULA : PPC32InstPattern9 <"bnula", Crf, Imm14, 516, 1, 0, 0>;
def BLTLR : PPC32InstPattern12 <"bltlr", Crf, Imm2, 620, 0, 32, 0, 0>;
def BLTLRL : PPC32InstPattern12 <"bltlrl", Crf, Imm2, 620, 0, 33, 0, 0>;
def BLELR : PPC32InstPattern12 <"blelr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BLELRL : PPC32InstPattern12 <"blelrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BEQLR : PPC32InstPattern12 <"beqlr", Crf, Imm2, 620, 0, 32, 0, 0>;
def BEQLRL : PPC32InstPattern12 <"beqlrl", Crf, Imm2, 620, 0, 33, 0, 0>;
def BGELR : PPC32InstPattern12 <"bgelr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BGELRL : PPC32InstPattern12 <"bgelrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BGTLR : PPC32InstPattern12 <"bgtlr", Crf, Imm2, 620, 0, 32, 0, 0>;
def BGTLRL : PPC32InstPattern12 <"bgtlrl", Crf, Imm2, 620, 0, 33, 0, 0>;
def BNLLR : PPC32InstPattern12 <"bnllr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BNLLRL : PPC32InstPattern12 <"bnllrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BNELR : PPC32InstPattern12 <"bnelr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BNELRL : PPC32InstPattern12 <"bnelrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BNGLR : PPC32InstPattern12 <"bnglr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BNGLRL : PPC32InstPattern12 <"bnglrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BSOLR : PPC32InstPattern12 <"bsolr", Crf, Imm2, 620, 0, 32, 0, 0>;
def BSOLRL : PPC32InstPattern12 <"bsolrl", Crf, Imm2, 620, 0, 33, 0, 0>;
def BNSLR : PPC32InstPattern12 <"bnslr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BNSLRL : PPC32InstPattern12 <"bnslrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BUNLR : PPC32InstPattern12 <"bunlr", Crf, Imm2, 620, 0, 32, 0, 0>;
def BUNLRL : PPC32InstPattern12 <"bunlrl", Crf, Imm2, 620, 0, 33, 0, 0>;
def BNULR : PPC32InstPattern12 <"bnulr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BNULRL : PPC32InstPattern12 <"bnulrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BLTCTR : PPC32InstPattern12 <"bltctr", Crf, Imm2, 620, 0, 32, 0, 0>;
def BLTCTRL : PPC32InstPattern12 <"bltctrl", Crf, Imm2, 620, 0, 33, 0, 0>;
def BLECTR : PPC32InstPattern12 <"blectr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BLECTRL : PPC32InstPattern12 <"blectrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BEQCTR : PPC32InstPattern12 <"beqctr", Crf, Imm2, 620, 0, 32, 0, 0>;
def BEQCTRL : PPC32InstPattern12 <"beqctrl", Crf, Imm2, 620, 0, 33, 0, 0>;
def BGECTR : PPC32InstPattern12 <"bgectr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BGECTRL : PPC32InstPattern12 <"bgectrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BGTCTR : PPC32InstPattern12 <"bgtctr", Crf, Imm2, 620, 0, 32, 0, 0>;
def BGTCTRL : PPC32InstPattern12 <"bgtctrl", Crf, Imm2, 620, 0, 33, 0, 0>;
def BNLCTR : PPC32InstPattern12 <"bnlctr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BNLCTRL : PPC32InstPattern12 <"bnlctrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BNECTR : PPC32InstPattern12 <"bnectr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BNECTRL : PPC32InstPattern12 <"bnectrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BNGCTR : PPC32InstPattern12 <"bngctr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BNGCTRL : PPC32InstPattern12 <"bngctrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BSOCTR : PPC32InstPattern12 <"bsoctr", Crf, Imm2, 620, 0, 32, 0, 0>;
def BSOCTRL : PPC32InstPattern12 <"bsoctrl", Crf, Imm2, 620, 0, 33, 0, 0>;
def BNSCTR : PPC32InstPattern12 <"bnsctr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BNSCTRL : PPC32InstPattern12 <"bnsctrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def BUNCTR : PPC32InstPattern12 <"bunctr", Crf, Imm2, 620, 0, 32, 0, 0>;
def BUNCTRL : PPC32InstPattern12 <"bunctrl", Crf, Imm2, 620, 0, 33, 0, 0>;
def BNUCTR : PPC32InstPattern12 <"bnuctr", Crf, Imm2, 612, 0, 32, 0, 0>;
def BNUCTRL : PPC32InstPattern12 <"bnuctrl", Crf, Imm2, 612, 0, 33, 0, 0>;
def CMPI : PPC32InstPattern13 <"cmpi", Imm3, Imm1, Gpr, Simm16, 11, 0, 0, 0>;
def CMPWI : PPC32InstPattern14 <"cmpwi", Imm3, Gpr, Simm16, 11, 0, 0, 0>;
def CMPDI : PPC32InstPattern14 <"cmpdi", Imm3, Gpr, Simm16, 11, 1, 0, 0>;
def CMP : PPC32InstPattern15 <"cmp", Imm3, Imm1, Gpr, Gpr, 31, 0, 0, 0, 0>;
def CMPW : PPC32InstPattern16 <"cmpw", Imm3, Gpr, Gpr, 31, 0, 0, 0, 0>;
def CMPD : PPC32InstPattern16 <"cmpd", Imm3, Gpr, Gpr, 31, 1, 0, 0, 0>;
def CMPLI : PPC32InstPattern13 <"cmpli", Imm3, Imm1, Gpr, Zimm16, 10, 0, 0, 0>;
def CMPLWI : PPC32InstPattern14 <"cmplwi", Imm3, Gpr, Zimm16, 10, 0, 0, 0>;
def CMPLDI : PPC32InstPattern14 <"cmpldi", Imm3, Gpr, Zimm16, 10, 1, 0, 0>;
def CMPL : PPC32InstPattern15 <"cmpl", Imm3, Imm1, Gpr, Gpr, 31, 0, 64, 0, 0>;
def CMPLW : PPC32InstPattern16 <"cmplw", Imm3, Gpr, Gpr, 31, 0, 64, 0, 0>;
def CMPLD : PPC32InstPattern16 <"cmpld", Imm3, Gpr, Gpr, 31, 1, 64, 0, 0>;
def CNTLZW : PPC32InstPattern17 <"cntlzw", Gpr, Gpr, 31, 52, 0, 0>;
def CNTLZWo : PPC32InstPattern17 <"cntlzw.", Gpr, Gpr, 31, 53, 0, 0>;
def CNTLZD : PPC32InstPattern17 <"cntlzd", Gpr, Gpr, 31, 116, 1, 0>;
def CNTLZDo : PPC32InstPattern17 <"cntlzd.", Gpr, Gpr, 31, 117, 1, 0>;
def CRAND : PPC32InstPattern2 <"crand", Imm5, Imm5, Imm5, 19, 514, 0, 0>;
def CRANDC : PPC32InstPattern2 <"crandc", Imm5, Imm5, Imm5, 19, 258, 0, 0>;
def CREQV : PPC32InstPattern2 <"creqv", Imm5, Imm5, Imm5, 19, 578, 0, 0>;
def CRNAND : PPC32InstPattern2 <"crnand", Imm5, Imm5, Imm5, 19, 450, 0, 0>;
def CRNOR : PPC32InstPattern2 <"crnor", Imm5, Imm5, Imm5, 19, 66, 0, 0>;
def CROR : PPC32InstPattern2 <"cror", Imm5, Imm5, Imm5, 19, 898, 0, 0>;
def CRORC : PPC32InstPattern2 <"crorc", Imm5, Imm5, Imm5, 19, 834, 0, 0>;
def CRXOR : PPC32InstPattern2 <"crxor", Imm5, Imm5, Imm5, 19, 386, 0, 0>;
def DIVD : PPC32InstPattern2 <"divd", Gpr, Gpr, Gpr, 31, 978, 1, 0>;
def DIVDo : PPC32InstPattern2 <"divd.", Gpr, Gpr, Gpr, 31, 979, 1, 0>;
def DIVDO : PPC32InstPattern2 <"divdo", Gpr, Gpr, Gpr, 31, 978, 1, 0>;
def DIVDOo : PPC32InstPattern2 <"divdo.", Gpr, Gpr, Gpr, 31, 979, 1, 0>;
def DIVDU : PPC32InstPattern2 <"divdu", Gpr, Gpr, Gpr, 31, 914, 1, 0>;
def DIVDUo : PPC32InstPattern2 <"divdu.", Gpr, Gpr, Gpr, 31, 915, 1, 0>;
def DIVDUO : PPC32InstPattern2 <"divduo", Gpr, Gpr, Gpr, 31, 914, 1, 0>;
def DIVDUOo : PPC32InstPattern2 <"divduo.", Gpr, Gpr, Gpr, 31, 915, 1, 0>;
def DIVW : PPC32InstPattern2 <"divw", Gpr, Gpr, Gpr, 31, 982, 0, 0>;
def DIVWo : PPC32InstPattern2 <"divw.", Gpr, Gpr, Gpr, 31, 983, 0, 0>;
def DIVWO : PPC32InstPattern2 <"divwo", Gpr, Gpr, Gpr, 31, 982, 0, 0>;
def DIVWOo : PPC32InstPattern2 <"divwo.", Gpr, Gpr, Gpr, 31, 983, 0, 0>;
def DIVWU : PPC32InstPattern2 <"divwu", Gpr, Gpr, Gpr, 31, 918, 0, 0>;
def DIVWUo : PPC32InstPattern2 <"divwu.", Gpr, Gpr, Gpr, 31, 919, 0, 0>;
def DIVWUO : PPC32InstPattern2 <"divwuo", Gpr, Gpr, Gpr, 31, 918, 0, 0>;
def DIVWUOo : PPC32InstPattern2 <"divwuo.", Gpr, Gpr, Gpr, 31, 919, 0, 0>;
def EQV : PPC32InstPattern5 <"eqv", Gpr, Gpr, Gpr, 31, 568, 0, 0>;
def EQVo : PPC32InstPattern5 <"eqv.", Gpr, Gpr, Gpr, 31, 569, 0, 0>;
def EXTSB : PPC32InstPattern17 <"extsb", Gpr, Gpr, 31, 1908, 0, 0>;
def EXTSBo : PPC32InstPattern17 <"extsb.", Gpr, Gpr, 31, 1909, 0, 0>;
def EXTSH : PPC32InstPattern17 <"extsh", Gpr, Gpr, 31, 1844, 0, 0>;
def EXTSHo : PPC32InstPattern17 <"extsh.", Gpr, Gpr, 31, 1845, 0, 0>;
def EXTSW : PPC32InstPattern17 <"extsw", Gpr, Gpr, 31, 1972, 1, 0>;
def EXTSWo : PPC32InstPattern17 <"extsw.", Gpr, Gpr, 31, 1973, 1, 0>;
def FADD : PPC32InstPattern2 <"fadd", Fpr, Fpr, Fpr, 63, 42, 0, 0>;
def FADDo : PPC32InstPattern2 <"fadd.", Fpr, Fpr, Fpr, 63, 43, 0, 0>;
def FADDS : PPC32InstPattern2 <"fadds", Fpr, Fpr, Fpr, 59, 42, 0, 0>;
def FADDSo : PPC32InstPattern2 <"fadds.", Fpr, Fpr, Fpr, 59, 43, 0, 0>;
def FSUB : PPC32InstPattern2 <"fsub", Fpr, Fpr, Fpr, 63, 40, 0, 0>;
def FSUBo : PPC32InstPattern2 <"fsub.", Fpr, Fpr, Fpr, 63, 41, 0, 0>;
def FSUBS : PPC32InstPattern2 <"fsubs", Fpr, Fpr, Fpr, 59, 40, 0, 0>;
def FSUBSo : PPC32InstPattern2 <"fsubs.", Fpr, Fpr, Fpr, 59, 41, 0, 0>;
def FMUL : PPC32InstPattern18 <"fmul", Fpr, Fpr, Fpr, 63, 0, 18, 0, 0>;
def FMULo : PPC32InstPattern18 <"fmul.", Fpr, Fpr, Fpr, 63, 0, 19, 0, 0>;
def FMULS : PPC32InstPattern18 <"fmuls", Fpr, Fpr, Fpr, 59, 0, 18, 0, 0>;
def FMULSo : PPC32InstPattern18 <"fmuls.", Fpr, Fpr, Fpr, 59, 0, 19, 0, 0>;
def FDIV : PPC32InstPattern2 <"fdiv", Fpr, Fpr, Fpr, 63, 36, 0, 0>;
def FDIVo : PPC32InstPattern2 <"fdiv.", Fpr, Fpr, Fpr, 63, 37, 0, 0>;
def FDIVS : PPC32InstPattern2 <"fdivs", Fpr, Fpr, Fpr, 59, 36, 0, 0>;
def FDIVSo : PPC32InstPattern2 <"fdivs.", Fpr, Fpr, Fpr, 59, 37, 0, 0>;
def FMADD : PPC32InstPattern19 <"fmadd", Fpr, Fpr, Fpr, Fpr, 63, 26, 0, 0>;
def FMADDo : PPC32InstPattern19 <"fmadd.", Fpr, Fpr, Fpr, Fpr, 63, 27, 0, 0>;
def FMADDS : PPC32InstPattern19 <"fmadds", Fpr, Fpr, Fpr, Fpr, 59, 26, 0, 0>;
def FMADDSo : PPC32InstPattern19 <"fmadds.", Fpr, Fpr, Fpr, Fpr, 59, 27, 0, 0>;
def FMSUB : PPC32InstPattern19 <"fmsub", Fpr, Fpr, Fpr, Fpr, 63, 24, 0, 0>;
def FMSUBo : PPC32InstPattern19 <"fmsub.", Fpr, Fpr, Fpr, Fpr, 63, 25, 0, 0>;
def FMSUBS : PPC32InstPattern19 <"fmsubs", Fpr, Fpr, Fpr, Fpr, 59, 24, 0, 0>;
def FMSUBSo : PPC32InstPattern19 <"fmsubs.", Fpr, Fpr, Fpr, Fpr, 59, 25, 0, 0>;
def FNMADD : PPC32InstPattern19 <"fnmadd", Fpr, Fpr, Fpr, Fpr, 63, 30, 0, 0>;
def FNMADDo : PPC32InstPattern19 <"fnmadd.", Fpr, Fpr, Fpr, Fpr, 63, 31, 0, 0>;
def FNMADDS : PPC32InstPattern19 <"fnmadds", Fpr, Fpr, Fpr, Fpr, 59, 30, 0, 0>;
def FNMADDSo : PPC32InstPattern19 <"fnmadds.", Fpr, Fpr, Fpr, Fpr, 59, 31, 0, 0>;
def FNMSUB : PPC32InstPattern19 <"fnmsub", Fpr, Fpr, Fpr, Fpr, 63, 28, 0, 0>;
def FNMSUBo : PPC32InstPattern19 <"fnmsub.", Fpr, Fpr, Fpr, Fpr, 63, 29, 0, 0>;
def FNMSUBS : PPC32InstPattern19 <"fnmsubs", Fpr, Fpr, Fpr, Fpr, 59, 28, 0, 0>;
def FNMSUBSo : PPC32InstPattern19 <"fnmsubs.", Fpr, Fpr, Fpr, Fpr, 59, 29, 0, 0>;
def FMR : PPC32InstPattern20 <"fmr", Fpr, Fpr, 63, 0, 144, 0, 0>;
def FMRo : PPC32InstPattern20 <"fmr.", Fpr, Fpr, 63, 0, 145, 0, 0>;
def FABS : PPC32InstPattern20 <"fabs", Fpr, Fpr, 63, 0, 528, 0, 0>;
def FABSo : PPC32InstPattern20 <"fabs.", Fpr, Fpr, 63, 0, 529, 0, 0>;
def FNEG : PPC32InstPattern20 <"fneg", Fpr, Fpr, 63, 0, 80, 0, 0>;
def FNEGo : PPC32InstPattern20 <"fneg.", Fpr, Fpr, 63, 0, 81, 0, 0>;
def FNABS : PPC32InstPattern20 <"fnabs", Fpr, Fpr, 63, 0, 272, 0, 0>;
def FNABSo : PPC32InstPattern20 <"fnabs.", Fpr, Fpr, 63, 0, 273, 0, 0>;
def FRES : PPC32InstPattern20 <"fres", Fpr, Fpr, 59, 0, 48, 0, 0>;
def FRESo : PPC32InstPattern20 <"fres.", Fpr, Fpr, 59, 0, 49, 0, 0>;
def FRSP : PPC32InstPattern20 <"frsp", Fpr, Fpr, 63, 0, 24, 0, 0>;
def FRSPo : PPC32InstPattern20 <"frsp.", Fpr, Fpr, 63, 0, 25, 0, 0>;
def FRSQRTE : PPC32InstPattern20 <"frsqrte", Fpr, Fpr, 63, 0, 52, 0, 0>;
def FRSQRTEo : PPC32InstPattern20 <"frsqrte.", Fpr, Fpr, 63, 0, 53, 0, 0>;
def FSEL : PPC32InstPattern19 <"fsel", Fpr, Fpr, Fpr, Fpr, 63, 14, 0, 0>;
def FSELo : PPC32InstPattern19 <"fsel.", Fpr, Fpr, Fpr, Fpr, 63, 15, 0, 0>;
def FSQRT : PPC32InstPattern20 <"fsqrt", Fpr, Fpr, 63, 0, 44, 0, 0>;
def FSQRTo : PPC32InstPattern20 <"fsqrt.", Fpr, Fpr, 63, 0, 45, 0, 0>;
def FSQRTS : PPC32InstPattern20 <"fsqrts", Fpr, Fpr, 59, 0, 44, 0, 0>;
def FSQRTSo : PPC32InstPattern20 <"fsqrts.", Fpr, Fpr, 59, 0, 45, 0, 0>;
def FCTID : PPC32InstPattern20 <"fctid", Fpr, Fpr, 63, 0, 604, 1, 0>;
def FCTIDo : PPC32InstPattern20 <"fctid.", Fpr, Fpr, 63, 0, 605, 1, 0>;
def FCTIDZ : PPC32InstPattern20 <"fctidz", Fpr, Fpr, 63, 0, 606, 1, 0>;
def FCTIDZo : PPC32InstPattern20 <"fctidz.", Fpr, Fpr, 63, 0, 607, 1, 0>;
def FCTIW : PPC32InstPattern20 <"fctiw", Fpr, Fpr, 63, 0, 28, 0, 0>;
def FCTIWo : PPC32InstPattern20 <"fctiw.", Fpr, Fpr, 63, 0, 29, 0, 0>;
def FCTIWZ : PPC32InstPattern20 <"fctiwz", Fpr, Fpr, 63, 0, 30, 0, 0>;
def FCTIWZo : PPC32InstPattern20 <"fctiwz.", Fpr, Fpr, 63, 0, 31, 0, 0>;
def FCFID : PPC32InstPattern20 <"fcfid", Fpr, Fpr, 63, 0, 668, 1, 0>;
def FCFIDo : PPC32InstPattern20 <"fcfid.", Fpr, Fpr, 63, 0, 669, 1, 0>;
def FCMPU : PPC32InstPattern16 <"fcmpu", Imm3, Fpr, Fpr, 63, 0, 0, 0, 0>;
def FCMPO : PPC32InstPattern16 <"fcmpo", Imm3, Fpr, Fpr, 63, 0, 64, 0, 0>;
def MFFS : PPC32InstPattern21 <"mffs", Fpr, 63, 1166, 0, 0>;
def MFFSo : PPC32InstPattern21 <"mffs.", Fpr, 63, 1167, 0, 0>;
def MCRFS : PPC32InstPattern22 <"mcrfs", Imm3, Imm5, 63, 128, 0, 0>;
def MTFSFI : PPC32InstPattern23 <"mtfsfi", Imm3, Imm4, 63, 0, 268, 0, 0>;
def MTFSFIo : PPC32InstPattern23 <"mtfsfi.", Imm3, Imm4, 63, 0, 269, 0, 0>;
def MTFSF : PPC32InstPattern24 <"mtfsf", Imm8, Fpr, 126, 0, 398, 0, 0>;
def MTFSFo : PPC32InstPattern24 <"mtfsf.", Imm8, Fpr, 126, 0, 399, 0, 0>;
def MTFSB0 : PPC32InstPattern21 <"mtfsb0", Imm5, 63, 140, 0, 0>;
def MTFSB0o : PPC32InstPattern21 <"mtfsb0.", Imm5, 63, 141, 0, 0>;
def MTFSB1 : PPC32InstPattern21 <"mtfsb1", Imm5, 63, 76, 0, 0>;
def MTFSB1o : PPC32InstPattern21 <"mtfsb1.", Imm5, 63, 77, 0, 0>;
def LBZ : PPC32InstPattern25 <"lbz", Gpr, Disimm16, Gpr0, 34, 0, 0>;
def LBZX : PPC32InstPattern2 <"lbzx", Gpr, Gpr0, Gpr, 31, 174, 0, 0>;
def LBZU : PPC32InstPattern25 <"lbzu", Gpr, Disimm16, Gpr0, 35, 0, 0>;
def LBZUX : PPC32InstPattern2 <"lbzux", Gpr, Gpr, Gpr, 31, 238, 0, 0>;
def LHZ : PPC32InstPattern25 <"lhz", Gpr, Disimm16, Gpr0, 40, 0, 0>;
def LHZX : PPC32InstPattern2 <"lhzx", Gpr, Gpr0, Gpr, 31, 558, 0, 0>;
def LHZU : PPC32InstPattern25 <"lhzu", Gpr, Disimm16, Gpr0, 41, 0, 0>;
def LHZUX : PPC32InstPattern2 <"lhzux", Gpr, Gpr, Gpr, 31, 622, 0, 0>;
def LHA : PPC32InstPattern25 <"lha", Gpr, Disimm16, Gpr0, 42, 0, 0>;
def LHAX : PPC32InstPattern2 <"lhax", Gpr, Gpr0, Gpr, 31, 686, 0, 0>;
def LHAU : PPC32InstPattern25 <"lhau", Gpr, Disimm16, Gpr, 43, 0, 0>;
def LHAUX : PPC32InstPattern2 <"lhaux", Gpr, Gpr, Gpr, 31, 750, 0, 0>;
def LWZ : PPC32InstPattern25 <"lwz", Gpr, Disimm16, Gpr0, 32, 0, 0>;
def LWZX : PPC32InstPattern2 <"lwzx", Gpr, Gpr0, Gpr, 31, 46, 0, 0>;
def LWZU : PPC32InstPattern25 <"lwzu", Gpr, Disimm16, Gpr, 33, 0, 0>;
def LWZUX : PPC32InstPattern2 <"lwzux", Gpr, Gpr, Gpr, 31, 110, 0, 0>;
def LWA : PPC32InstPattern26 <"lwa", Gpr, Disimm14, Gpr0, 58, 0, 1, 0>;
def LWAX : PPC32InstPattern2 <"lwax", Gpr, Gpr0, Gpr, 31, 682, 1, 0>;
def LWAUX : PPC32InstPattern2 <"lwaux", Gpr, Gpr, Gpr, 31, 746, 1, 0>;
def LD : PPC32InstPattern26 <"ld", Gpr, Disimm14, Gpr0, 58, 0, 1, 0>;
def LDX : PPC32InstPattern2 <"ldx", Gpr, Gpr0, Gpr, 31, 42, 1, 0>;
def LDU : PPC32InstPattern26 <"ldu", Gpr, Disimm14, Gpr, 58, 1, 1, 0>;
def LDUX : PPC32InstPattern2 <"ldux", Gpr, Gpr, Gpr, 31, 106, 1, 0>;
def LMW : PPC32InstPattern25 <"lmw", Gpr, Disimm16, Gpr0, 46, 0, 0>;
def STMW : PPC32InstPattern25 <"stmw", Gpr, Disimm16, Gpr0, 47, 0, 0>;
def LHBRX : PPC32InstPattern2 <"lhbrx", Gpr, Gpr0, Gpr, 31, 556, 0, 0>;
def LWBRX : PPC32InstPattern2 <"lwbrx", Gpr, Gpr0, Gpr, 31, 44, 0, 0>;
def LSWX : PPC32InstPattern2 <"lswx", Gpr, Gpr0, Gpr, 31, 42, 0, 0>;
def LWARX : PPC32InstPattern2 <"lwarx", Gpr, Gpr0, Gpr, 31, 40, 0, 0>;
def LDARX : PPC32InstPattern2 <"ldarx", Gpr, Gpr0, Gpr, 31, 168, 1, 0>;
def LSWI : PPC32InstPattern2 <"lswi", Gpr, Gpr0, Imm5, 31, 170, 0, 0>;
def LFS : PPC32InstPattern25 <"lfs", Fpr, Disimm16, Gpr0, 48, 0, 0>;
def LFSU : PPC32InstPattern25 <"lfsu", Fpr, Disimm16, Gpr, 49, 0, 0>;
def LFSX : PPC32InstPattern2 <"lfsx", Fpr, Gpr0, Gpr, 31, 46, 0, 0>;
def LFSUX : PPC32InstPattern2 <"lfsux", Fpr, Gpr, Gpr, 31, 110, 0, 0>;
def LFD : PPC32InstPattern25 <"lfd", Fpr, Disimm16, Gpr0, 50, 0, 0>;
def LFDU : PPC32InstPattern25 <"lfdu", Fpr, Disimm16, Gpr, 51, 0, 0>;
def LFDX : PPC32InstPattern2 <"lfdx", Fpr, Gpr0, Gpr, 31, 174, 0, 0>;
def LFDUX : PPC32InstPattern2 <"lfdux", Fpr, Gpr, Gpr, 31, 238, 0, 0>;
def LA : PPC32InstPattern25 <"la", Gpr, Disimm16, Gpr0, 14, 0, 0>;
def MCRF : PPC32InstPattern27 <"mcrf", Imm3, Imm3, 19, 0, 0, 0, 0>;
def MFSPR : PPC32InstPattern28 <"mfspr", Gpr, Spr, 31, 678, 0, 0>;
def MTSPR : PPC32InstPattern29 <"mtspr", Spr, Gpr, 31, 934, 0, 0>;
def MTCRF : PPC32InstPattern30 <"mtcrf", Imm8, Gpr, 31, 0, 288, 0, 0>;
def MCRXR : PPC32InstPattern31 <"mcrxr", Imm3, 31, 1024, 0, 0>;
def MFCR : PPC32InstPattern32 <"mfcr", Gpr, Imm8, 31, 0, 38, 0, 0>;
def MFXER : PPC32InstPattern21 <"mfxer", Gpr, 31, 66214, 0, 0>;
def MFLR : PPC32InstPattern21 <"mflr", Gpr, 31, 524966, 0, 0>;
def MFCTR : PPC32InstPattern21 <"mfctr", Gpr, 31, 590502, 0, 0>;
def MTXER : PPC32InstPattern21 <"mtxer", Gpr, 31, 66470, 0, 0>;
def MTLR : PPC32InstPattern21 <"mtlr", Gpr, 31, 525222, 0, 0>;
def MTCTR : PPC32InstPattern21 <"mtctr", Gpr, 31, 590758, 0, 0>;
def MFMQ : PPC32InstPattern21 <"mfmq", Gpr, 31, 678, 0, 0>;
def MFRTCL : PPC32InstPattern21 <"mfrtcl", Gpr, 31, 328358, 0, 0>;
def MFRTCU : PPC32InstPattern21 <"mfrtcu", Gpr, 31, 262822, 0, 0>;
def MTMQ : PPC32InstPattern21 <"mtmq", Gpr, 31, 934, 0, 0>;
def MTRTCL : PPC32InstPattern21 <"mtrtcl", Gpr, 31, 328614, 0, 0>;
def MTRTCU : PPC32InstPattern21 <"mtrtcu", Gpr, 31, 263078, 0, 0>;
def MULLW : PPC32InstPattern2 <"mullw", Gpr, Gpr, Gpr, 31, 470, 0, 0>;
def MULLWo : PPC32InstPattern2 <"mullw.", Gpr, Gpr, Gpr, 31, 471, 0, 0>;
def MULLWO : PPC32InstPattern2 <"mullwo", Gpr, Gpr, Gpr, 31, 470, 0, 0>;
def MULLWOo : PPC32InstPattern2 <"mullwo.", Gpr, Gpr, Gpr, 31, 471, 0, 0>;
def MULHD : PPC32InstPattern2 <"mulhd", Gpr, Gpr, Gpr, 31, 146, 1, 0>;
def MULHDo : PPC32InstPattern2 <"mulhd.", Gpr, Gpr, Gpr, 31, 147, 1, 0>;
def MULHW : PPC32InstPattern2 <"mulhw", Gpr, Gpr, Gpr, 31, 150, 0, 0>;
def MULHWo : PPC32InstPattern2 <"mulhw.", Gpr, Gpr, Gpr, 31, 151, 0, 0>;
def MULHDU : PPC32InstPattern2 <"mulhdu", Gpr, Gpr, Gpr, 31, 18, 1, 0>;
def MULHDUo : PPC32InstPattern2 <"mulhdu.", Gpr, Gpr, Gpr, 31, 19, 1, 0>;
def MULHWU : PPC32InstPattern2 <"mulhwu", Gpr, Gpr, Gpr, 31, 22, 0, 0>;
def MULHWUo : PPC32InstPattern2 <"mulhwu.", Gpr, Gpr, Gpr, 31, 23, 0, 0>;
def MULLD : PPC32InstPattern2 <"mulld", Gpr, Gpr, Gpr, 31, 466, 1, 0>;
def MULLDo : PPC32InstPattern2 <"mulld.", Gpr, Gpr, Gpr, 31, 467, 1, 0>;
def MULLDO : PPC32InstPattern2 <"mulldo", Gpr, Gpr, Gpr, 31, 466, 1, 0>;
def MULLDOo : PPC32InstPattern2 <"mulldo.", Gpr, Gpr, Gpr, 31, 467, 1, 0>;
def NAND : PPC32InstPattern5 <"nand", Gpr, Gpr, Gpr, 31, 952, 0, 0>;
def NANDo : PPC32InstPattern5 <"nand.", Gpr, Gpr, Gpr, 31, 953, 0, 0>;
def NEG : PPC32InstPattern3 <"neg", Gpr, Gpr, 31, 208, 0, 0>;
def NEGo : PPC32InstPattern3 <"neg.", Gpr, Gpr, 31, 209, 0, 0>;
def NEGO : PPC32InstPattern3 <"nego", Gpr, Gpr, 31, 1232, 0, 0>;
def NEGOo : PPC32InstPattern3 <"nego.", Gpr, Gpr, 31, 1233, 0, 0>;
def NOR : PPC32InstPattern5 <"nor", Gpr, Gpr, Gpr, 31, 248, 0, 0>;
def NORo : PPC32InstPattern5 <"nor.", Gpr, Gpr, Gpr, 31, 249, 0, 0>;
def NOP : PPC32InstPattern33 <"nop", 1610612736, 0, 0>;
def ORI : PPC32InstPattern4 <"ori", Gpr, Gpr, Zimm16, 24, 0, 0>;
def ORIS : PPC32InstPattern4 <"oris", Gpr, Gpr, Zimm16, 25, 0, 0>;
def OR : PPC32InstPattern5 <"or", Gpr, Gpr, Gpr, 31, 888, 0, 0>;
def ORo : PPC32InstPattern5 <"or.", Gpr, Gpr, Gpr, 31, 889, 0, 0>;
def ORC : PPC32InstPattern5 <"orc", Gpr, Gpr, Gpr, 31, 824, 0, 0>;
def ORCo : PPC32InstPattern5 <"orc.", Gpr, Gpr, Gpr, 31, 825, 0, 0>;
def RLDICL : PPC32InstPattern17 <"rldicl", Gpr, Gpr, 30, 0, 1, 0>;
def RLDICLo : PPC32InstPattern17 <"rldicl.", Gpr, Gpr, 30, 1, 1, 0>;
def RLDICR : PPC32InstPattern17 <"rldicr", Gpr, Gpr, 30, 4, 1, 0>;
def RLDICRo : PPC32InstPattern17 <"rldicr.", Gpr, Gpr, 30, 5, 1, 0>;
def RLDIC : PPC32InstPattern17 <"rldic", Gpr, Gpr, 30, 8, 1, 0>;
def RLDICo : PPC32InstPattern17 <"rldic.", Gpr, Gpr, 30, 9, 1, 0>;
def RLDIMI : PPC32InstPattern17 <"rldimi", Gpr, Gpr, 30, 12, 1, 0>;
def RLDIMIo : PPC32InstPattern17 <"rldimi.", Gpr, Gpr, 30, 13, 1, 0>;
def RLDCL : PPC32InstPattern5 <"rldcl", Gpr, Gpr, Gpr, 30, 16, 1, 0>;
def RLDCLo : PPC32InstPattern5 <"rldcl.", Gpr, Gpr, Gpr, 30, 17, 1, 0>;
def RLDCR : PPC32InstPattern5 <"rldcr", Gpr, Gpr, Gpr, 30, 18, 1, 0>;
def RLDCRo : PPC32InstPattern5 <"rldcr.", Gpr, Gpr, Gpr, 30, 19, 1, 0>;
def RLWINM : PPC32InstPattern34 <"rlwinm", Gpr, Gpr, Imm5, Imm5, Imm5, 21, 0, 0, 0>;
def RLWINMo : PPC32InstPattern34 <"rlwinm.", Gpr, Gpr, Imm5, Imm5, Imm5, 21, 0, 0, 0>;
def RLWNM : PPC32InstPattern34 <"rlwnm", Gpr, Gpr, Gpr, Imm5, Imm5, 23, 0, 0, 0>;
def RLWNMo : PPC32InstPattern34 <"rlwnm.", Gpr, Gpr, Gpr, Imm5, Imm5, 23, 0, 0, 0>;
def RLWIMI : PPC32InstPattern34 <"rlwimi", Gpr, Gpr, Imm5, Imm5, Imm5, 20, 0, 0, 0>;
def RLWIMIo : PPC32InstPattern34 <"rlwimi.", Gpr, Gpr, Imm5, Imm5, Imm5, 20, 0, 0, 0>;
def SC : PPC32InstPattern33 <"sc", 1140850690, 0, 0>;
def RFID : PPC32InstPattern33 <"rfid", 1275068452, 1, 0>;
def SLW : PPC32InstPattern5 <"slw", Gpr, Gpr, Gpr, 31, 48, 0, 0>;
def SLWo : PPC32InstPattern5 <"slw.", Gpr, Gpr, Gpr, 31, 49, 0, 0>;
def SLD : PPC32InstPattern5 <"sld", Gpr, Gpr, Gpr, 31, 54, 1, 0>;
def SLDo : PPC32InstPattern5 <"sld.", Gpr, Gpr, Gpr, 31, 55, 1, 0>;
def SRW : PPC32InstPattern5 <"srw", Gpr, Gpr, Gpr, 31, 48, 0, 0>;
def SRWo : PPC32InstPattern5 <"srw.", Gpr, Gpr, Gpr, 31, 49, 0, 0>;
def SRD : PPC32InstPattern5 <"srd", Gpr, Gpr, Gpr, 31, 54, 1, 0>;
def SRDo : PPC32InstPattern5 <"srd.", Gpr, Gpr, Gpr, 31, 55, 1, 0>;
def SRAWI : PPC32InstPattern5 <"srawi", Gpr, Gpr, Imm5, 31, 624, 0, 0>;
def SRAWIo : PPC32InstPattern5 <"srawi.", Gpr, Gpr, Imm5, 31, 625, 0, 0>;
def SRADI : PPC32InstPattern17 <"sradi", Gpr, Gpr, 31, 1652, 1, 0>;
def SRADIo : PPC32InstPattern17 <"sradi.", Gpr, Gpr, 31, 1653, 1, 0>;
def SRAW : PPC32InstPattern5 <"sraw", Gpr, Gpr, Gpr, 31, 560, 0, 0>;
def SRAWo : PPC32InstPattern5 <"sraw.", Gpr, Gpr, Gpr, 31, 561, 0, 0>;
def SRAD : PPC32InstPattern5 <"srad", Gpr, Gpr, Gpr, 31, 564, 1, 0>;
def SRADo : PPC32InstPattern5 <"srad.", Gpr, Gpr, Gpr, 31, 565, 1, 0>;
def STB : PPC32InstPattern25 <"stb", Gpr, Disimm16, Gpr0, 38, 0, 0>;
def STBU : PPC32InstPattern25 <"stbu", Gpr, Disimm16, Gpr, 39, 0, 0>;
def STBX : PPC32InstPattern2 <"stbx", Gpr, Gpr0, Gpr, 31, 430, 0, 0>;
def STBUX : PPC32InstPattern2 <"stbux", Gpr, Gpr, Gpr, 31, 494, 0, 0>;
def STH : PPC32InstPattern25 <"sth", Gpr, Disimm16, Gpr0, 44, 0, 0>;
def STHU : PPC32InstPattern25 <"sthu", Gpr, Disimm16, Gpr, 45, 0, 0>;
def STHX : PPC32InstPattern2 <"sthx", Gpr, Gpr0, Gpr, 31, 814, 0, 0>;
def STHUX : PPC32InstPattern2 <"sthux", Gpr, Gpr, Gpr, 31, 878, 0, 0>;
def STW : PPC32InstPattern25 <"stw", Gpr, Disimm16, Gpr0, 36, 0, 0>;
def STWU : PPC32InstPattern25 <"stwu", Gpr, Disimm16, Gpr, 37, 0, 0>;
def STWX : PPC32InstPattern2 <"stwx", Gpr, Gpr0, Gpr, 31, 302, 0, 0>;
def STWUX : PPC32InstPattern2 <"stwux", Gpr, Gpr, Gpr, 31, 366, 0, 0>;
def STD : PPC32InstPattern26 <"std", Gpr, Disimm14, Gpr0, 62, 0, 1, 0>;
def STDU : PPC32InstPattern26 <"stdu", Gpr, Disimm14, Gpr, 62, 1, 1, 0>;
def STDX : PPC32InstPattern2 <"stdx", Gpr, Gpr0, Gpr, 31, 298, 1, 0>;
def STDUX : PPC32InstPattern2 <"stdux", Gpr, Gpr, Gpr, 31, 362, 1, 0>;
def STHBRX : PPC32InstPattern2 <"sthbrx", Gpr, Gpr0, Gpr, 31, 812, 0, 0>;
def STWBRX : PPC32InstPattern2 <"stwbrx", Gpr, Gpr0, Gpr, 31, 300, 0, 0>;
def STSWX : PPC32InstPattern2 <"stswx", Gpr, Gpr0, Gpr, 31, 298, 0, 0>;
def STWCXo : PPC32InstPattern2 <"stwcx.", Gpr, Gpr0, Gpr, 31, 301, 0, 0>;
def STDCXo : PPC32InstPattern2 <"stdcx.", Gpr, Gpr0, Gpr, 31, 429, 1, 0>;
def STSWI : PPC32InstPattern2 <"stswi", Gpr, Gpr0, Imm5, 31, 426, 0, 0>;
def STFIWX : PPC32InstPattern2 <"stfiwx", Fpr, Gpr0, Gpr, 31, 942, 0, 0>;
def STFS : PPC32InstPattern25 <"stfs", Fpr, Disimm16, Gpr0, 52, 0, 0>;
def STFSU : PPC32InstPattern25 <"stfsu", Fpr, Disimm16, Gpr, 53, 0, 0>;
def STFSX : PPC32InstPattern2 <"stfsx", Fpr, Gpr0, Gpr, 31, 302, 0, 0>;
def STFSUX : PPC32InstPattern2 <"stfsux", Fpr, Gpr, Gpr, 31, 366, 0, 0>;
def STFD : PPC32InstPattern25 <"stfd", Fpr, Disimm16, Gpr0, 54, 0, 0>;
def STFDU : PPC32InstPattern25 <"stfdu", Fpr, Disimm16, Gpr, 55, 0, 0>;
def STFDX : PPC32InstPattern2 <"stfdx", Fpr, Gpr0, Gpr, 31, 430, 0, 0>;
def STFDUX : PPC32InstPattern2 <"stfdux", Fpr, Gpr, Gpr, 31, 494, 0, 0>;
def SUBFIC : PPC32InstPattern0 <"subfic", Gpr, Gpr, Simm16, 8, 0, 0>;
def SUB : PPC32InstPattern35 <"sub", Gpr, Gpr, Gpr, 31, 80, 0, 0>;
def SUBo : PPC32InstPattern35 <"sub.", Gpr, Gpr, Gpr, 31, 81, 0, 0>;
def SUBO : PPC32InstPattern35 <"subo", Gpr, Gpr, Gpr, 31, 80, 0, 0>;
def SUBOo : PPC32InstPattern35 <"subo.", Gpr, Gpr, Gpr, 31, 81, 0, 0>;
def SUBF : PPC32InstPattern2 <"subf", Gpr, Gpr, Gpr, 31, 80, 0, 0>;
def SUBFo : PPC32InstPattern2 <"subf.", Gpr, Gpr, Gpr, 31, 81, 0, 0>;
def SUBFO : PPC32InstPattern2 <"subfo", Gpr, Gpr, Gpr, 31, 80, 0, 0>;
def SUBFOo : PPC32InstPattern2 <"subfo.", Gpr, Gpr, Gpr, 31, 81, 0, 0>;
def SUBC : PPC32InstPattern35 <"subc", Gpr, Gpr, Gpr, 31, 16, 0, 0>;
def SUBCo : PPC32InstPattern35 <"subc.", Gpr, Gpr, Gpr, 31, 17, 0, 0>;
def SUBCO : PPC32InstPattern35 <"subco", Gpr, Gpr, Gpr, 31, 16, 0, 0>;
def SUBCOo : PPC32InstPattern35 <"subco.", Gpr, Gpr, Gpr, 31, 17, 0, 0>;
def SUBFC : PPC32InstPattern2 <"subfc", Gpr, Gpr, Gpr, 31, 16, 0, 0>;
def SUBFCo : PPC32InstPattern2 <"subfc.", Gpr, Gpr, Gpr, 31, 17, 0, 0>;
def SUBFCO : PPC32InstPattern2 <"subfco", Gpr, Gpr, Gpr, 31, 16, 0, 0>;
def SUBFCOo : PPC32InstPattern2 <"subfco.", Gpr, Gpr, Gpr, 31, 17, 0, 0>;
def SUBFE : PPC32InstPattern2 <"subfe", Gpr, Gpr, Gpr, 31, 272, 0, 0>;
def SUBFEo : PPC32InstPattern2 <"subfe.", Gpr, Gpr, Gpr, 31, 273, 0, 0>;
def SUBFEO : PPC32InstPattern2 <"subfeo", Gpr, Gpr, Gpr, 31, 272, 0, 0>;
def SUBFEOo : PPC32InstPattern2 <"subfeo.", Gpr, Gpr, Gpr, 31, 273, 0, 0>;
def SUBFME : PPC32InstPattern3 <"subfme", Gpr, Gpr, 31, 464, 0, 0>;
def SUBFMEo : PPC32InstPattern3 <"subfme.", Gpr, Gpr, 31, 465, 0, 0>;
def SUBFMEO : PPC32InstPattern3 <"subfmeo", Gpr, Gpr, 31, 1488, 0, 0>;
def SUBFMEOo : PPC32InstPattern3 <"subfmeo.", Gpr, Gpr, 31, 1489, 0, 0>;
def SUBFZE : PPC32InstPattern3 <"subfze", Gpr, Gpr, 31, 400, 0, 0>;
def SUBFZEo : PPC32InstPattern3 <"subfze.", Gpr, Gpr, 31, 401, 0, 0>;
def SUBFZEO : PPC32InstPattern3 <"subfzeo", Gpr, Gpr, 31, 1424, 0, 0>;
def SUBFZEOo : PPC32InstPattern3 <"subfzeo.", Gpr, Gpr, 31, 1425, 0, 0>;
def SYNC : PPC32InstPattern36 <"sync", Imm2, 248, 1196, 0, 0>;
def LWSYNC : PPC32InstPattern33 <"lwsync", 2082473132, 0, 0>;
def PTESYNC : PPC32InstPattern33 <"ptesync", 2084570284, 0, 0>;
def TDI : PPC32InstPattern0 <"tdi", Imm5, Gpr, Simm16, 2, 1, 0>;
def TDLTI : PPC32InstPattern37 <"tdlti", Gpr, Simm16, 80, 1, 0>;
def TDLEI : PPC32InstPattern37 <"tdlei", Gpr, Simm16, 84, 1, 0>;
def TDEQI : PPC32InstPattern37 <"tdeqi", Gpr, Simm16, 68, 1, 0>;
def TDGEI : PPC32InstPattern37 <"tdgei", Gpr, Simm16, 76, 1, 0>;
def TDGTI : PPC32InstPattern37 <"tdgti", Gpr, Simm16, 72, 1, 0>;
def TDNLI : PPC32InstPattern37 <"tdnli", Gpr, Simm16, 76, 1, 0>;
def TDNEI : PPC32InstPattern37 <"tdnei", Gpr, Simm16, 88, 1, 0>;
def TDNGI : PPC32InstPattern37 <"tdngi", Gpr, Simm16, 84, 1, 0>;
def TDLLTI : PPC32InstPattern37 <"tdllti", Gpr, Simm16, 66, 1, 0>;
def TDLLEI : PPC32InstPattern37 <"tdllei", Gpr, Simm16, 70, 1, 0>;
def TDLGEI : PPC32InstPattern37 <"tdlgei", Gpr, Simm16, 69, 1, 0>;
def TDLGTI : PPC32InstPattern37 <"tdlgti", Gpr, Simm16, 65, 1, 0>;
def TDLNLI : PPC32InstPattern37 <"tdlnli", Gpr, Simm16, 69, 1, 0>;
def TDLNGI : PPC32InstPattern37 <"tdlngi", Gpr, Simm16, 70, 1, 0>;
def TD : PPC32InstPattern2 <"td", Imm5, Gpr, Gpr, 31, 136, 1, 0>;
def TDLT : PPC32InstPattern38 <"tdlt", Gpr, Gpr, 1008, 136, 1, 0>;
def TDLE : PPC32InstPattern38 <"tdle", Gpr, Gpr, 1012, 136, 1, 0>;
def TDEQ : PPC32InstPattern38 <"tdeq", Gpr, Gpr, 996, 136, 1, 0>;
def TDGE : PPC32InstPattern38 <"tdge", Gpr, Gpr, 1004, 136, 1, 0>;
def TDGT : PPC32InstPattern38 <"tdgt", Gpr, Gpr, 1000, 136, 1, 0>;
def TDNL : PPC32InstPattern38 <"tdnl", Gpr, Gpr, 1004, 136, 1, 0>;
def TDNE : PPC32InstPattern38 <"tdne", Gpr, Gpr, 1016, 136, 1, 0>;
def TDNG : PPC32InstPattern38 <"tdng", Gpr, Gpr, 1012, 136, 1, 0>;
def TDLLT : PPC32InstPattern38 <"tdllt", Gpr, Gpr, 994, 136, 1, 0>;
def TDLLE : PPC32InstPattern38 <"tdlle", Gpr, Gpr, 998, 136, 1, 0>;
def TDLGE : PPC32InstPattern38 <"tdlge", Gpr, Gpr, 997, 136, 1, 0>;
def TDLGT : PPC32InstPattern38 <"tdlgt", Gpr, Gpr, 993, 136, 1, 0>;
def TDLNL : PPC32InstPattern38 <"tdlnl", Gpr, Gpr, 997, 136, 1, 0>;
def TDLNG : PPC32InstPattern38 <"tdlng", Gpr, Gpr, 998, 136, 1, 0>;
def TWI : PPC32InstPattern0 <"twi", Imm5, Gpr, Simm16, 3, 0, 0>;
def TWLTI : PPC32InstPattern37 <"twlti", Gpr, Simm16, 112, 0, 0>;
def TWLEI : PPC32InstPattern37 <"twlei", Gpr, Simm16, 116, 0, 0>;
def TWEQI : PPC32InstPattern37 <"tweqi", Gpr, Simm16, 100, 0, 0>;
def TWGEI : PPC32InstPattern37 <"twgei", Gpr, Simm16, 108, 0, 0>;
def TWGTI : PPC32InstPattern37 <"twgti", Gpr, Simm16, 104, 0, 0>;
def TWNLI : PPC32InstPattern37 <"twnli", Gpr, Simm16, 108, 0, 0>;
def TWNEI : PPC32InstPattern37 <"twnei", Gpr, Simm16, 120, 0, 0>;
def TWNGI : PPC32InstPattern37 <"twngi", Gpr, Simm16, 116, 0, 0>;
def TWLLTI : PPC32InstPattern37 <"twllti", Gpr, Simm16, 98, 0, 0>;
def TWLLEI : PPC32InstPattern37 <"twllei", Gpr, Simm16, 102, 0, 0>;
def TWLGEI : PPC32InstPattern37 <"twlgei", Gpr, Simm16, 101, 0, 0>;
def TWLGTI : PPC32InstPattern37 <"twlgti", Gpr, Simm16, 97, 0, 0>;
def TWLNLI : PPC32InstPattern37 <"twlnli", Gpr, Simm16, 101, 0, 0>;
def TWLNGI : PPC32InstPattern37 <"twlngi", Gpr, Simm16, 102, 0, 0>;
def TW : PPC32InstPattern2 <"tw", Imm5, Gpr, Gpr, 31, 8, 0, 0>;
def TWLT : PPC32InstPattern38 <"twlt", Gpr, Gpr, 1008, 8, 0, 0>;
def TWLE : PPC32InstPattern38 <"twle", Gpr, Gpr, 1012, 8, 0, 0>;
def TWEQ : PPC32InstPattern38 <"tweq", Gpr, Gpr, 996, 8, 0, 0>;
def TWGE : PPC32InstPattern38 <"twge", Gpr, Gpr, 1004, 8, 0, 0>;
def TWGT : PPC32InstPattern38 <"twgt", Gpr, Gpr, 1000, 8, 0, 0>;
def TWNL : PPC32InstPattern38 <"twnl", Gpr, Gpr, 1004, 8, 0, 0>;
def TWNE : PPC32InstPattern38 <"twne", Gpr, Gpr, 1016, 8, 0, 0>;
def TWNG : PPC32InstPattern38 <"twng", Gpr, Gpr, 1012, 8, 0, 0>;
def TWLLT : PPC32InstPattern38 <"twllt", Gpr, Gpr, 994, 8, 0, 0>;
def TWLLE : PPC32InstPattern38 <"twlle", Gpr, Gpr, 998, 8, 0, 0>;
def TWLGE : PPC32InstPattern38 <"twlge", Gpr, Gpr, 997, 8, 0, 0>;
def TWLGT : PPC32InstPattern38 <"twlgt", Gpr, Gpr, 993, 8, 0, 0>;
def TWLNL : PPC32InstPattern38 <"twlnl", Gpr, Gpr, 997, 8, 0, 0>;
def TWLNG : PPC32InstPattern38 <"twlng", Gpr, Gpr, 998, 8, 0, 0>;
def TRAP : PPC32InstPattern33 <"trap", 2145386504, 0, 0>;
def XORI : PPC32InstPattern4 <"xori", Gpr, Gpr, Zimm16, 26, 0, 0>;
def XORIS : PPC32InstPattern4 <"xoris", Gpr, Gpr, Zimm16, 27, 0, 0>;
def XOR : PPC32InstPattern5 <"xor", Gpr, Gpr, Gpr, 31, 632, 0, 0>;
def XORo : PPC32InstPattern5 <"xor.", Gpr, Gpr, Gpr, 31, 633, 0, 0>;
def ICBI : PPC32InstPattern38 <"icbi", Gpr0, Gpr, 992, 940, 0, 0>;
def ISYNC : PPC32InstPattern33 <"isync", 1275068716, 0, 0>;
def DCBT : PPC32InstPattern39 <"dcbt", Gpr0, Gpr, Imm4, 62, 556, 1, 0>;
def DCBTST : PPC32InstPattern38 <"dcbtst", Gpr0, Gpr, 992, 492, 0, 0>;
def DCBT128 : PPC32InstPattern39 <"dcbt128", Gpr0, Gpr, Imm4, 62, 556, 1, 0>;
def DCBZ : PPC32InstPattern38 <"dcbz", Gpr0, Gpr, 992, 1004, 0, 0>;
def DCBZL : PPC32InstPattern38 <"dcbzl", Gpr0, Gpr, 993, 1004, 1, 0>;
def DCBZ128 : PPC32InstPattern38 <"dcbz128", Gpr0, Gpr, 993, 1004, 1, 0>;
def DCBST : PPC32InstPattern38 <"dcbst", Gpr0, Gpr, 992, 108, 0, 0>;
def DCBF : PPC32InstPattern38 <"dcbf", Gpr0, Gpr, 992, 172, 0, 0>;
def ECIWX : PPC32InstPattern2 <"eciwx", Gpr, Gpr0, Gpr, 31, 620, 0, 0>;
def ECOWX : PPC32InstPattern2 <"ecowx", Gpr, Gpr0, Gpr, 31, 876, 0, 0>;
def EIEIO : PPC32InstPattern33 <"eieio", 2080376492, 0, 0>;
def RFI : PPC32InstPattern33 <"rfi", 1275068516, 0, 0>;
def MTMSR : PPC32InstPattern21 <"mtmsr", Gpr, 31, 292, 0, 0>;
def MTMSRD : PPC32InstPattern40 <"mtmsrd", Gpr, Imm1, 31, 0, 356, 1, 0>;
def MFMSR : PPC32InstPattern21 <"mfmsr", Gpr, 31, 166, 0, 0>;
def DCBA : PPC32InstPattern38 <"dcba", Gpr0, Gpr, 992, 492, 0, 0>;
def DCBI : PPC32InstPattern38 <"dcbi", Gpr0, Gpr, 992, 940, 0, 0>;
def MTSR : PPC32InstPattern41 <"mtsr", Sgr, Gpr, 31, 0, 420, 0, 0>;
def MFSR : PPC32InstPattern42 <"mfsr", Gpr, Sgr, 31, 0, 1190, 0, 0>;
def MTSRIN : PPC32InstPattern20 <"mtsrin", Gpr, Gpr, 31, 0, 484, 0, 0>;
def MFSRIN : PPC32InstPattern20 <"mfsrin", Gpr, Gpr, 31, 0, 294, 0, 0>;
def SLBIE : PPC32InstPattern43 <"slbie", Gpr, 31744, 868, 1, 0>;
def SLBIA : PPC32InstPattern33 <"slbia", 2080375780, 1, 0>;
def SLBMTE : PPC32InstPattern20 <"slbmte", Gpr, Gpr, 31, 0, 804, 1, 0>;
def SLBMFEV : PPC32InstPattern20 <"slbmfev", Gpr, Gpr, 31, 0, 678, 1, 0>;
def SLBMFEE : PPC32InstPattern20 <"slbmfee", Gpr, Gpr, 31, 0, 806, 1, 0>;
def TLBIE : PPC32InstPattern44 <"tlbie", Gpr, Imm1, 496, 0, 612, 1, 0>;
def TLBIEL : PPC32InstPattern43 <"tlbiel", Gpr, 31744, 548, 1, 0>;
def TLBIA : PPC32InstPattern33 <"tlbia", 2080375524, 0, 0>;
def TLBSYNC : PPC32InstPattern33 <"tlbsync", 2080375916, 0, 0>;
def MTTBL : PPC32InstPattern21 <"mttbl", Gpr, 31, 803750, 0, 0>;
def MTTBU : PPC32InstPattern21 <"mttbu", Gpr, 31, 869286, 0, 0>;
def MFTB : PPC32InstPattern28 <"mftb", Gpr, Spr, 31, 742, 0, 0>;
def MFTBU : PPC32InstPattern21 <"mftbu", Gpr, 31, 869094, 0, 0>;
def ATTN : PPC32InstPattern45 <"attn", Imm15, 0, 512, 0, 0>;
def MULLI : PPC32InstPattern0 <"mulli", Gpr, Gpr, Simm16, 7, 0, 0>;
def TLBLD : PPC32InstPattern43 <"tlbld", Gpr, 31744, 932, 0, 0>;
def TLBLI : PPC32InstPattern43 <"tlbli", Gpr, 31744, 996, 0, 0>;
def LVEBX : PPC32InstPattern2 <"lvebx", Vpr, Gpr0, Gpr, 31, 14, 0, 1>;
def LVEHX : PPC32InstPattern2 <"lvehx", Vpr, Gpr0, Gpr, 31, 78, 0, 1>;
def LVEWX : PPC32InstPattern2 <"lvewx", Vpr, Gpr0, Gpr, 31, 142, 0, 1>;
def LVX : PPC32InstPattern2 <"lvx", Vpr, Gpr0, Gpr, 31, 206, 0, 1>;
def LVXL : PPC32InstPattern2 <"lvxl", Vpr, Gpr0, Gpr, 31, 718, 0, 1>;
def STVEBX : PPC32InstPattern2 <"stvebx", Vpr, Gpr0, Gpr, 31, 270, 0, 1>;
def STVEHX : PPC32InstPattern2 <"stvehx", Vpr, Gpr0, Gpr, 31, 334, 0, 1>;
def STVEWX : PPC32InstPattern2 <"stvewx", Vpr, Gpr0, Gpr, 31, 398, 0, 1>;
def STVX : PPC32InstPattern2 <"stvx", Vpr, Gpr0, Gpr, 31, 462, 0, 1>;
def STVXL : PPC32InstPattern2 <"stvxl", Vpr, Gpr0, Gpr, 31, 974, 0, 1>;
def LVSL : PPC32InstPattern2 <"lvsl", Vpr, Gpr0, Gpr, 31, 12, 0, 1>;
def LVSR : PPC32InstPattern2 <"lvsr", Vpr, Gpr0, Gpr, 31, 76, 0, 1>;
def MTVSCR : PPC32InstPattern43 <"mtvscr", Vpr, 4096, 580, 0, 1>;
def MFVSCR : PPC32InstPattern21 <"mfvscr", Vpr, 4, 1540, 0, 1>;
def DST : PPC32InstPattern46 <"dst", Gpr, Gpr, Imm2, 248, 684, 0, 1>;
def DSTT : PPC32InstPattern46 <"dstt", Gpr, Gpr, Imm2, 252, 684, 0, 1>;
def DSTST : PPC32InstPattern46 <"dstst", Gpr, Gpr, Imm2, 248, 748, 0, 1>;
def DSTSTT : PPC32InstPattern46 <"dststt", Gpr, Gpr, Imm2, 252, 748, 0, 1>;
def DSS : PPC32InstPattern36 <"dss", Imm2, 248, 1644, 0, 1>;
def DSSALL : PPC32InstPattern33 <"dssall", 2113930860, 0, 1>;
def VADDUBM : PPC32InstPattern2 <"vaddubm", Vpr, Vpr, Vpr, 4, 0, 0, 1>;
def VADDUBS : PPC32InstPattern2 <"vaddubs", Vpr, Vpr, Vpr, 4, 512, 0, 1>;
def VADDSBS : PPC32InstPattern2 <"vaddsbs", Vpr, Vpr, Vpr, 4, 768, 0, 1>;
def VADDUHM : PPC32InstPattern2 <"vadduhm", Vpr, Vpr, Vpr, 4, 64, 0, 1>;
def VADDUHS : PPC32InstPattern2 <"vadduhs", Vpr, Vpr, Vpr, 4, 576, 0, 1>;
def VADDSHS : PPC32InstPattern2 <"vaddshs", Vpr, Vpr, Vpr, 4, 832, 0, 1>;
def VADDUWM : PPC32InstPattern2 <"vadduwm", Vpr, Vpr, Vpr, 4, 128, 0, 1>;
def VADDUWS : PPC32InstPattern2 <"vadduws", Vpr, Vpr, Vpr, 4, 640, 0, 1>;
def VADDSWS : PPC32InstPattern2 <"vaddsws", Vpr, Vpr, Vpr, 4, 896, 0, 1>;
def VADDFP : PPC32InstPattern2 <"vaddfp", Vpr, Vpr, Vpr, 4, 10, 0, 1>;
def VADDCUW : PPC32InstPattern2 <"vaddcuw", Vpr, Vpr, Vpr, 4, 384, 0, 1>;
def VSUBUBM : PPC32InstPattern2 <"vsububm", Vpr, Vpr, Vpr, 4, 0, 0, 1>;
def VSUBUBS : PPC32InstPattern2 <"vsububs", Vpr, Vpr, Vpr, 4, 512, 0, 1>;
def VSUBSBS : PPC32InstPattern2 <"vsubsbs", Vpr, Vpr, Vpr, 4, 768, 0, 1>;
def VSUBUHM : PPC32InstPattern2 <"vsubuhm", Vpr, Vpr, Vpr, 4, 64, 0, 1>;
def VSUBUHS : PPC32InstPattern2 <"vsubuhs", Vpr, Vpr, Vpr, 4, 576, 0, 1>;
def VSUBSHS : PPC32InstPattern2 <"vsubshs", Vpr, Vpr, Vpr, 4, 832, 0, 1>;
def VSUBUWM : PPC32InstPattern2 <"vsubuwm", Vpr, Vpr, Vpr, 4, 128, 0, 1>;
def VSUBUWS : PPC32InstPattern2 <"vsubuws", Vpr, Vpr, Vpr, 4, 640, 0, 1>;
def VSUBSWS : PPC32InstPattern2 <"vsubsws", Vpr, Vpr, Vpr, 4, 896, 0, 1>;
def VSUBFP : PPC32InstPattern2 <"vsubfp", Vpr, Vpr, Vpr, 4, 74, 0, 1>;
def VSUBCUW : PPC32InstPattern2 <"vsubcuw", Vpr, Vpr, Vpr, 4, 384, 0, 1>;
def VMULOUB : PPC32InstPattern2 <"vmuloub", Vpr, Vpr, Vpr, 4, 8, 0, 1>;
def VMULOSB : PPC32InstPattern2 <"vmulosb", Vpr, Vpr, Vpr, 4, 264, 0, 1>;
def VMULOUH : PPC32InstPattern2 <"vmulouh", Vpr, Vpr, Vpr, 4, 72, 0, 1>;
def VMULOSH : PPC32InstPattern2 <"vmulosh", Vpr, Vpr, Vpr, 4, 328, 0, 1>;
def VMULEUB : PPC32InstPattern2 <"vmuleub", Vpr, Vpr, Vpr, 4, 520, 0, 1>;
def VMULESB : PPC32InstPattern2 <"vmulesb", Vpr, Vpr, Vpr, 4, 776, 0, 1>;
def VMULEUH : PPC32InstPattern2 <"vmuleuh", Vpr, Vpr, Vpr, 4, 584, 0, 1>;
def VMULESH : PPC32InstPattern2 <"vmulesh", Vpr, Vpr, Vpr, 4, 840, 0, 1>;
def VMHADDSHS : PPC32InstPattern47 <"vmhaddshs", Vpr, Vpr, Vpr, Vpr, 4, 0, 0, 1>;
def VMHRADDSHS : PPC32InstPattern47 <"vmhraddshs", Vpr, Vpr, Vpr, Vpr, 4, 1, 0, 1>;
def VMLADDUHM : PPC32InstPattern47 <"vmladduhm", Vpr, Vpr, Vpr, Vpr, 4, 2, 0, 1>;
def VMADDFP : PPC32InstPattern19 <"vmaddfp", Vpr, Vpr, Vpr, Vpr, 4, 14, 0, 1>;
def VMSUMUBM : PPC32InstPattern47 <"vmsumubm", Vpr, Vpr, Vpr, Vpr, 4, 4, 0, 1>;
def VMSUMMBM : PPC32InstPattern47 <"vmsummbm", Vpr, Vpr, Vpr, Vpr, 4, 5, 0, 1>;
def VMSUMUHM : PPC32InstPattern47 <"vmsumuhm", Vpr, Vpr, Vpr, Vpr, 4, 6, 0, 1>;
def VMSUMUHS : PPC32InstPattern47 <"vmsumuhs", Vpr, Vpr, Vpr, Vpr, 4, 7, 0, 1>;
def VMSUMSHM : PPC32InstPattern47 <"vmsumshm", Vpr, Vpr, Vpr, Vpr, 4, 8, 0, 1>;
def VMSUMSHS : PPC32InstPattern47 <"vmsumshs", Vpr, Vpr, Vpr, Vpr, 4, 9, 0, 1>;
def VSUMSWS : PPC32InstPattern2 <"vsumsws", Vpr, Vpr, Vpr, 4, 904, 0, 1>;
def VSUM2SWS : PPC32InstPattern2 <"vsum2sws", Vpr, Vpr, Vpr, 4, 648, 0, 1>;
def VSUM4UBS : PPC32InstPattern2 <"vsum4ubs", Vpr, Vpr, Vpr, 4, 520, 0, 1>;
def VSUM4SBS : PPC32InstPattern2 <"vsum4sbs", Vpr, Vpr, Vpr, 4, 776, 0, 1>;
def VSUM4SHS : PPC32InstPattern2 <"vsum4shs", Vpr, Vpr, Vpr, 4, 584, 0, 1>;
def VAVGUB : PPC32InstPattern2 <"vavgub", Vpr, Vpr, Vpr, 4, 2, 0, 1>;
def VAVGUH : PPC32InstPattern2 <"vavguh", Vpr, Vpr, Vpr, 4, 66, 0, 1>;
def VAVGUW : PPC32InstPattern2 <"vavguw", Vpr, Vpr, Vpr, 4, 130, 0, 1>;
def VAVGSB : PPC32InstPattern2 <"vavgsb", Vpr, Vpr, Vpr, 4, 258, 0, 1>;
def VAVGSH : PPC32InstPattern2 <"vavgsh", Vpr, Vpr, Vpr, 4, 322, 0, 1>;
def VAVGSW : PPC32InstPattern2 <"vavgsw", Vpr, Vpr, Vpr, 4, 386, 0, 1>;
def VAND : PPC32InstPattern2 <"vand", Vpr, Vpr, Vpr, 4, 4, 0, 1>;
def VOR : PPC32InstPattern2 <"vor", Vpr, Vpr, Vpr, 4, 132, 0, 1>;
def VXOR : PPC32InstPattern2 <"vxor", Vpr, Vpr, Vpr, 4, 196, 0, 1>;
def VANDC : PPC32InstPattern2 <"vandc", Vpr, Vpr, Vpr, 4, 68, 0, 1>;
def VNOR : PPC32InstPattern2 <"vnor", Vpr, Vpr, Vpr, 4, 260, 0, 1>;
def VRLB : PPC32InstPattern2 <"vrlb", Vpr, Vpr, Vpr, 4, 4, 0, 1>;
def VRLH : PPC32InstPattern2 <"vrlh", Vpr, Vpr, Vpr, 4, 68, 0, 1>;
def VRLW : PPC32InstPattern2 <"vrlw", Vpr, Vpr, Vpr, 4, 132, 0, 1>;
def VSLB : PPC32InstPattern2 <"vslb", Vpr, Vpr, Vpr, 4, 260, 0, 1>;
def VSLH : PPC32InstPattern2 <"vslh", Vpr, Vpr, Vpr, 4, 324, 0, 1>;
def VSLW : PPC32InstPattern2 <"vslw", Vpr, Vpr, Vpr, 4, 388, 0, 1>;
def VSL : PPC32InstPattern2 <"vsl", Vpr, Vpr, Vpr, 4, 452, 0, 1>;
def VSRB : PPC32InstPattern2 <"vsrb", Vpr, Vpr, Vpr, 4, 516, 0, 1>;
def VSRAB : PPC32InstPattern2 <"vsrab", Vpr, Vpr, Vpr, 4, 772, 0, 1>;
def VSRH : PPC32InstPattern2 <"vsrh", Vpr, Vpr, Vpr, 4, 580, 0, 1>;
def VSRAH : PPC32InstPattern2 <"vsrah", Vpr, Vpr, Vpr, 4, 836, 0, 1>;
def VSRW : PPC32InstPattern2 <"vsrw", Vpr, Vpr, Vpr, 4, 644, 0, 1>;
def VSRAW : PPC32InstPattern2 <"vsraw", Vpr, Vpr, Vpr, 4, 900, 0, 1>;
def VSR : PPC32InstPattern2 <"vsr", Vpr, Vpr, Vpr, 4, 708, 0, 1>;
def VCMPGTUB : PPC32InstPattern2 <"vcmpgtub", Vpr, Vpr, Vpr, 4, 518, 0, 1>;
def VCMPGTUBo : PPC32InstPattern2 <"vcmpgtub.", Vpr, Vpr, Vpr, 4, 518, 0, 1>;
def VCMPGTSB : PPC32InstPattern2 <"vcmpgtsb", Vpr, Vpr, Vpr, 4, 774, 0, 1>;
def VCMPGTSBo : PPC32InstPattern2 <"vcmpgtsb.", Vpr, Vpr, Vpr, 4, 774, 0, 1>;
def VCMPGTUH : PPC32InstPattern2 <"vcmpgtuh", Vpr, Vpr, Vpr, 4, 582, 0, 1>;
def VCMPGTUHo : PPC32InstPattern2 <"vcmpgtuh.", Vpr, Vpr, Vpr, 4, 582, 0, 1>;
def VCMPGTSH : PPC32InstPattern2 <"vcmpgtsh", Vpr, Vpr, Vpr, 4, 838, 0, 1>;
def VCMPGTSHo : PPC32InstPattern2 <"vcmpgtsh.", Vpr, Vpr, Vpr, 4, 838, 0, 1>;
def VCMPGTUW : PPC32InstPattern2 <"vcmpgtuw", Vpr, Vpr, Vpr, 4, 646, 0, 1>;
def VCMPGTUWo : PPC32InstPattern2 <"vcmpgtuw.", Vpr, Vpr, Vpr, 4, 646, 0, 1>;
def VCMPGTSW : PPC32InstPattern2 <"vcmpgtsw", Vpr, Vpr, Vpr, 4, 902, 0, 1>;
def VCMPGTSWo : PPC32InstPattern2 <"vcmpgtsw.", Vpr, Vpr, Vpr, 4, 902, 0, 1>;
def VCMPGTFP : PPC32InstPattern2 <"vcmpgtfp", Vpr, Vpr, Vpr, 4, 710, 0, 1>;
def VCMPGTFPo : PPC32InstPattern2 <"vcmpgtfp.", Vpr, Vpr, Vpr, 4, 710, 0, 1>;
def VCMPEQUB : PPC32InstPattern2 <"vcmpequb", Vpr, Vpr, Vpr, 4, 6, 0, 1>;
def VCMPEQUBo : PPC32InstPattern2 <"vcmpequb.", Vpr, Vpr, Vpr, 4, 6, 0, 1>;
def VCMPEQUH : PPC32InstPattern2 <"vcmpequh", Vpr, Vpr, Vpr, 4, 70, 0, 1>;
def VCMPEQUHo : PPC32InstPattern2 <"vcmpequh.", Vpr, Vpr, Vpr, 4, 70, 0, 1>;
def VCMPEQUW : PPC32InstPattern2 <"vcmpequw", Vpr, Vpr, Vpr, 4, 134, 0, 1>;
def VCMPEQUWo : PPC32InstPattern2 <"vcmpequw.", Vpr, Vpr, Vpr, 4, 134, 0, 1>;
def VCMPEQFP : PPC32InstPattern2 <"vcmpeqfp", Vpr, Vpr, Vpr, 4, 198, 0, 1>;
def VCMPEQFPo : PPC32InstPattern2 <"vcmpeqfp.", Vpr, Vpr, Vpr, 4, 198, 0, 1>;
def VCMPGEFP : PPC32InstPattern2 <"vcmpgefp", Vpr, Vpr, Vpr, 4, 454, 0, 1>;
def VCMPGEFPo : PPC32InstPattern2 <"vcmpgefp.", Vpr, Vpr, Vpr, 4, 454, 0, 1>;
def VCMPBFP : PPC32InstPattern2 <"vcmpbfp", Vpr, Vpr, Vpr, 4, 966, 0, 1>;
def VCMPBFPo : PPC32InstPattern2 <"vcmpbfp.", Vpr, Vpr, Vpr, 4, 966, 0, 1>;
def VSEL : PPC32InstPattern47 <"vsel", Vpr, Vpr, Vpr, Vpr, 4, 10, 0, 1>;
def VPKUHUM : PPC32InstPattern2 <"vpkuhum", Vpr, Vpr, Vpr, 4, 14, 0, 1>;
def VPKUHUS : PPC32InstPattern2 <"vpkuhus", Vpr, Vpr, Vpr, 4, 142, 0, 1>;
def VPKSHUS : PPC32InstPattern2 <"vpkshus", Vpr, Vpr, Vpr, 4, 270, 0, 1>;
def VPKSHSS : PPC32InstPattern2 <"vpkshss", Vpr, Vpr, Vpr, 4, 398, 0, 1>;
def VPKUWUM : PPC32InstPattern2 <"vpkuwum", Vpr, Vpr, Vpr, 4, 78, 0, 1>;
def VPKUWUS : PPC32InstPattern2 <"vpkuwus", Vpr, Vpr, Vpr, 4, 206, 0, 1>;
def VPKSWUS : PPC32InstPattern2 <"vpkswus", Vpr, Vpr, Vpr, 4, 334, 0, 1>;
def VPKSWSS : PPC32InstPattern2 <"vpkswss", Vpr, Vpr, Vpr, 4, 462, 0, 1>;
def VPKPX : PPC32InstPattern2 <"vpkpx", Vpr, Vpr, Vpr, 4, 782, 0, 1>;
def VUPKHSB : PPC32InstPattern20 <"vupkhsb", Vpr, Vpr, 4, 0, 526, 0, 1>;
def VUPKHSH : PPC32InstPattern20 <"vupkhsh", Vpr, Vpr, 4, 0, 590, 0, 1>;
def VUPKHPX : PPC32InstPattern20 <"vupkhpx", Vpr, Vpr, 4, 0, 846, 0, 1>;
def VUPKLSB : PPC32InstPattern20 <"vupklsb", Vpr, Vpr, 4, 0, 654, 0, 1>;
def VUPKLSH : PPC32InstPattern20 <"vupklsh", Vpr, Vpr, 4, 0, 718, 0, 1>;
def VUPKLPX : PPC32InstPattern20 <"vupklpx", Vpr, Vpr, 4, 0, 974, 0, 1>;
def VMRGHB : PPC32InstPattern2 <"vmrghb", Vpr, Vpr, Vpr, 4, 12, 0, 1>;
def VMRGHH : PPC32InstPattern2 <"vmrghh", Vpr, Vpr, Vpr, 4, 76, 0, 1>;
def VMRGHW : PPC32InstPattern2 <"vmrghw", Vpr, Vpr, Vpr, 4, 140, 0, 1>;
def VMRGLB : PPC32InstPattern2 <"vmrglb", Vpr, Vpr, Vpr, 4, 268, 0, 1>;
def VMRGLH : PPC32InstPattern2 <"vmrglh", Vpr, Vpr, Vpr, 4, 332, 0, 1>;
def VMRGLW : PPC32InstPattern2 <"vmrglw", Vpr, Vpr, Vpr, 4, 396, 0, 1>;
def VSPLTB : PPC32InstPattern35 <"vspltb", Vpr, Vpr, Imm5, 4, 524, 0, 1>;
def VSPLTH : PPC32InstPattern35 <"vsplth", Vpr, Vpr, Imm5, 4, 588, 0, 1>;
def VSPLTW : PPC32InstPattern35 <"vspltw", Vpr, Vpr, Imm5, 4, 652, 0, 1>;
def VSPLTISB : PPC32InstPattern3 <"vspltisb", Vpr, Imm5, 4, 780, 0, 1>;
def VSPLTISH : PPC32InstPattern3 <"vspltish", Vpr, Imm5, 4, 844, 0, 1>;
def VSPLTISW : PPC32InstPattern3 <"vspltisw", Vpr, Imm5, 4, 908, 0, 1>;
def VPERM : PPC32InstPattern47 <"vperm", Vpr, Vpr, Vpr, Vpr, 4, 11, 0, 1>;
def VSLDOI : PPC32InstPattern48 <"vsldoi", Vpr, Vpr, Vpr, Imm4, 4, 0, 12, 0, 1>;
def VSLO : PPC32InstPattern2 <"vslo", Vpr, Vpr, Vpr, 4, 12, 0, 1>;
def VSRO : PPC32InstPattern2 <"vsro", Vpr, Vpr, Vpr, 4, 76, 0, 1>;
def VMAXUB : PPC32InstPattern2 <"vmaxub", Vpr, Vpr, Vpr, 4, 2, 0, 1>;
def VMAXSB : PPC32InstPattern2 <"vmaxsb", Vpr, Vpr, Vpr, 4, 258, 0, 1>;
def VMAXUH : PPC32InstPattern2 <"vmaxuh", Vpr, Vpr, Vpr, 4, 66, 0, 1>;
def VMAXSH : PPC32InstPattern2 <"vmaxsh", Vpr, Vpr, Vpr, 4, 322, 0, 1>;
def VMAXUW : PPC32InstPattern2 <"vmaxuw", Vpr, Vpr, Vpr, 4, 130, 0, 1>;
def VMAXSW : PPC32InstPattern2 <"vmaxsw", Vpr, Vpr, Vpr, 4, 386, 0, 1>;
def VMAXFP : PPC32InstPattern2 <"vmaxfp", Vpr, Vpr, Vpr, 4, 10, 0, 1>;
def VMINUB : PPC32InstPattern2 <"vminub", Vpr, Vpr, Vpr, 4, 514, 0, 1>;
def VMINSB : PPC32InstPattern2 <"vminsb", Vpr, Vpr, Vpr, 4, 770, 0, 1>;
def VMINUH : PPC32InstPattern2 <"vminuh", Vpr, Vpr, Vpr, 4, 578, 0, 1>;
def VMINSH : PPC32InstPattern2 <"vminsh", Vpr, Vpr, Vpr, 4, 834, 0, 1>;
def VMINUW : PPC32InstPattern2 <"vminuw", Vpr, Vpr, Vpr, 4, 642, 0, 1>;
def VMINSW : PPC32InstPattern2 <"vminsw", Vpr, Vpr, Vpr, 4, 898, 0, 1>;
def VMINFP : PPC32InstPattern2 <"vminfp", Vpr, Vpr, Vpr, 4, 74, 0, 1>;
def VREFP : PPC32InstPattern20 <"vrefp", Vpr, Vpr, 4, 0, 266, 0, 1>;
def VRSQRTEFP : PPC32InstPattern20 <"vrsqrtefp", Vpr, Vpr, 4, 0, 330, 0, 1>;
def VLOGEFP : PPC32InstPattern20 <"vlogefp", Vpr, Vpr, 4, 0, 458, 0, 1>;
def VEXPTEFP : PPC32InstPattern20 <"vexptefp", Vpr, Vpr, 4, 0, 394, 0, 1>;
def VNMSUBFP : PPC32InstPattern19 <"vnmsubfp", Vpr, Vpr, Vpr, Vpr, 4, 15, 0, 1>;
def VRFIN : PPC32InstPattern20 <"vrfin", Vpr, Vpr, 4, 0, 522, 0, 1>;
def VRFIZ : PPC32InstPattern20 <"vrfiz", Vpr, Vpr, 4, 0, 586, 0, 1>;
def VRFIP : PPC32InstPattern20 <"vrfip", Vpr, Vpr, 4, 0, 650, 0, 1>;
def VRFIM : PPC32InstPattern20 <"vrfim", Vpr, Vpr, 4, 0, 714, 0, 1>;
def VCTUXS : PPC32InstPattern35 <"vctuxs", Vpr, Vpr, Imm5, 4, 906, 0, 1>;
def VCTSXS : PPC32InstPattern35 <"vctsxs", Vpr, Vpr, Imm5, 4, 970, 0, 1>;
def VCFUX : PPC32InstPattern35 <"vcfux", Vpr, Vpr, Imm5, 4, 778, 0, 1>;
def VCFSX : PPC32InstPattern35 <"vcfsx", Vpr, Vpr, Imm5, 4, 842, 0, 1>;
