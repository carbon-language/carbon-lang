//===- MSP430InstrFormats.td - MSP430 Instruction Formats-----*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe MSP430 instructions format here
//

// Generic MSP430 Format
class MSP430Inst<dag outs, dag ins, string asmstr> : Instruction {
  field bits<16> Inst;

  let Namespace = "MSP430";

  dag OutOperandList = outs;
  dag InOperandList  = ins;

  let AsmString   = asmstr;
}

// FIXME: Create different classes for different addressing modes.

// MSP430 Double Operand (Format I) Instructions
class IForm<bits<4> opcode, bit ad, bit bw, bits<2> as,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, asmstr> {
  let Pattern = pattern;
  
  let Inst{12-15} = opcode;
  let Inst{7}     = ad;
  let Inst{6}     = bw;
  let Inst{4-5}   = as;
}

// MSP430 Single Operand (Format II) Instructions
class IIForm<bits<9> opcode, bit bw, bits<2> ad,
             dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, asmstr> {
  let Pattern = pattern;
  
  let Inst{7-15} = opcode;
  let Inst{6}    = bw;
  let Inst{4-5}  = ad;
}

// MSP430 Conditional Jumps Instructions
class CJForm<bits<3> opcode, bits<3> cond, bit s,
             dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, asmstr> {
  let Pattern = pattern;
  
  let Inst{13-15} = opcode;
  let Inst{10-12} = cond;
  let Inst{9}     = s;
}

// Pseudo instructions
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, asmstr> {
  let Pattern = pattern;
  let Inst{15-0} = 0;
}
