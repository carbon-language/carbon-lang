//===- MipsRegisterInfo.td - Mips Register defs -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe MIPS instructions format
//
//  All the possible Mips fields are:
//
//  opcode  - operation code.
//  rs      - src reg.
//  rt      - dst reg (on a 2 regs instr) or src reg (on a 3 reg instr).
//  rd      - dst reg, only used on 3 regs instr.
//  shamt   - only used on shift instructions, contains the shift amount.
//  funct   - combined with opcode field give us an operation code.
//
//===----------------------------------------------------------------------===//

// Generic Mips Format
class MipsInst<dag outs, dag ins, string asmstr, list<dag> pattern, 
               InstrItinClass itin>: Instruction 
{
  field bits<32> Inst;

  let Namespace = "Mips";

  bits<6> opcode;

  // Top 5 bits are the 'opcode' field
  let Inst{31-26} = opcode;   
  
  dag OutOperandList = outs;
  dag InOperandList  = ins;

  let AsmString   = asmstr;
  let Pattern     = pattern;
  let Itinerary   = itin;
}

// Mips Pseudo Instructions Format
class MipsPseudo<dag outs, dag ins, string asmstr, list<dag> pattern>:
      MipsInst<outs, ins, asmstr, pattern, IIPseudo>;

//===----------------------------------------------------------------------===//
// Format R instruction class in Mips : <|opcode|rs|rt|rd|shamt|funct|>
//===----------------------------------------------------------------------===//

class FR<bits<6> op, bits<6> _funct, dag outs, dag ins, string asmstr,
         list<dag> pattern, InstrItinClass itin>:
      MipsInst<outs, ins, asmstr, pattern, itin> 
{
  bits<5>  rd;
  bits<5>  rs;
  bits<5>  rt;
  bits<5>  shamt;
  bits<6>  funct;

  let opcode = op;
  let funct  = _funct;

  let Inst{25-21} = rs;
  let Inst{20-16} = rt; 
  let Inst{15-11} = rd;
  let Inst{10-6}  = shamt;
  let Inst{5-0}   = funct;
}

//===----------------------------------------------------------------------===//
// Format I instruction class in Mips : <|opcode|rs|rt|immediate|>
//===----------------------------------------------------------------------===//

class FI<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
         InstrItinClass itin>: MipsInst<outs, ins, asmstr, pattern, itin> 
{
  bits<5>  rt;
  bits<5>  rs;
  bits<16> imm16;

  let opcode = op;

  let Inst{25-21} = rs;
  let Inst{20-16} = rt; 
  let Inst{15-0}  = imm16;
}

//===----------------------------------------------------------------------===//
// Format J instruction class in Mips : <|opcode|address|>
//===----------------------------------------------------------------------===//

class FJ<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
         InstrItinClass itin>: MipsInst<outs, ins, asmstr, pattern, itin> 
{
  bits<26> addr;

  let opcode = op;
  
  let Inst{25-0} = addr;
}

