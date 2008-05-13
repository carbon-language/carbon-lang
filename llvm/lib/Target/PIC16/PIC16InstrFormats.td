//===- PIC16RegisterInfo.td - PIC16 Register defs ------------*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe PIC16 instructions format
//
//  All the possible PIC16 fields are:
//
//  opcode  - operation code.
//  f       - 7-bit register file address.
//  d       - 1-bit direction specifier
//  k       - 8/11 bit literals
//  b       - 3 bits bit num specifier
//
//===----------------------------------------------------------------------===//

// Generic PIC16 Format
class PIC16Inst<dag outs, dag ins, string asmstr, list<dag> pattern>
	: Instruction 
{
  field bits<14> Inst;

  let Namespace = "PIC16";

  dag OutOperandList = outs;
  dag InOperandList  = ins;

  let AsmString   = asmstr;
  let Pattern     = pattern;
}


//===----------------------------------------------------------------------===//
// Byte Oriented instruction class in PIC16 : <|opcode|d|f|>
//===----------------------------------------------------------------------===//

class ByteFormat<bits<6> op, dag outs, dag ins, string asmstr,
         	 list<dag> pattern>
	:PIC16Inst<outs, ins, asmstr, pattern> 
{
  bits<1>  d;
  bits<7>  f;

  let Inst{13-8} = op;

  let Inst{7} = d;
  let Inst{6-0} = f; 
}

//===----------------------------------------------------------------------===//
// Bit Oriented instruction class in PIC16 : <|opcode|b|f|>
//===----------------------------------------------------------------------===//

class BitFormat<bits<4> op, dag outs, dag ins, string asmstr, list<dag> pattern>
         : PIC16Inst<outs, ins, asmstr, pattern> 
{
  bits<3>  b;
  bits<7>  f;

  let Inst{13-10} = op;

  let Inst{9-7} = b;
  let Inst{6-0} = f; 
}

//===----------------------------------------------------------------------===//
// Literal Format instruction class in PIC16 : <|opcode|k|>
//===----------------------------------------------------------------------===//

class LiteralFormat<bits<6> op, dag outs, dag ins, string asmstr, 
		    list<dag> pattern>
	: PIC16Inst<outs, ins, asmstr, pattern> 
{
  bits<8> k;

  
  let Inst{13-8} = op;

  let Inst{7-0} = k; 
}

//===----------------------------------------------------------------------===//
// Control Format instruction class in PIC16 : <|opcode|k|>
//===----------------------------------------------------------------------===//

class ControlFormat<bits<3> op, dag outs, dag ins, string asmstr, 
		    list<dag> pattern>
	:PIC16Inst<outs, ins, asmstr, pattern> 
{
  bits<11> k;

  
  let Inst{13-11} = op;

  let Inst{10-0} = k; 
}

//===----------------------------------------------------------------------===//
// Pseudo instruction class in PIC16
//===----------------------------------------------------------------------===//

class Pseudo<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>:
      PIC16Inst<outs, ins, asmstr, pattern>
{
   let Inst{13-6} = op;
}
