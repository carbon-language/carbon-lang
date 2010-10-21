//===- MBlazeInstrFormats.td - MB Instruction defs ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe MBlaze instructions format
//
//  CPU INSTRUCTION FORMATS
//
//  opcode  - operation code.
//  rd      - dst reg.
//  ra      - first src. reg.
//  rb      - second src. reg.
//  imm16   - 16-bit immediate value.
//
//===----------------------------------------------------------------------===//

// Generic MBlaze Format
class MBlazeInst<dag outs, dag ins, string asmstr, list<dag> pattern, 
               InstrItinClass itin> : Instruction 
{
  field bits<32> Inst;

  let Namespace = "MBlaze";

  bits<6> opcode;

  // Top 6 bits are the 'opcode' field
  let Inst{0-5} = opcode;   
  
  dag OutOperandList = outs;
  dag InOperandList  = ins;

  let AsmString   = asmstr;
  let Pattern     = pattern;
  let Itinerary   = itin;
}

//===----------------------------------------------------------------------===//
// Pseudo instruction class
//===----------------------------------------------------------------------===//
class MBlazePseudo<dag outs, dag ins, string asmstr, list<dag> pattern>:
      MBlazeInst<outs, ins, asmstr, pattern, IIPseudo>;

//===----------------------------------------------------------------------===//
// Type A instruction class in MBlaze : <|opcode|rd|ra|rb|flags|>
//===----------------------------------------------------------------------===//

class TA<bits<6> op, bits<11> flags, dag outs, dag ins, string asmstr,
         list<dag> pattern, InstrItinClass itin> : 
         MBlazeInst<outs, ins, asmstr, pattern, itin> 
{
  bits<5> rd;
  bits<5> ra;
  bits<5> rb;

  let opcode = op;

  let Inst{6-10}  = rd;
  let Inst{11-15} = ra; 
  let Inst{16-20} = rb;
  let Inst{21-31} = flags;
}

class TAI<bits<6> op, dag outs, dag ins, string asmstr,
          list<dag> pattern, InstrItinClass itin> :
          MBlazeInst<outs, ins, asmstr, pattern, itin>
{
  bits<5> rd;
  bits<5> ra;
  bits<16> imm16;

  let opcode = op;

  let Inst{6-10}  = rd;
  let Inst{11-15} = ra; 
  let Inst{16-31} = imm16;
}

class TIMM<bits<6> op, dag outs, dag ins, string asmstr,
           list<dag> pattern, InstrItinClass itin> :
           MBlazeInst<outs, ins, asmstr, pattern, itin>
{
  bits<5> ra;
  bits<16> imm16;

  let opcode = op;

  let Inst{6-15}  = 0;
  let Inst{16-31} = imm16;
}

class TADDR<bits<6> op, dag outs, dag ins, string asmstr,
            list<dag> pattern, InstrItinClass itin> :
            MBlazeInst<outs, ins, asmstr, pattern, itin>
{
  bits<26> addr;

  let opcode = op;

  let Inst{6-31} = addr;
}

//===----------------------------------------------------------------------===//
// Type B instruction class in MBlaze : <|opcode|rd|ra|immediate|>
//===----------------------------------------------------------------------===//

class TB<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
         InstrItinClass itin> : 
         MBlazeInst<outs, ins, asmstr, pattern, itin> 
{
  bits<5>  rd;
  bits<5>  ra;
  bits<16> imm16;

  let opcode = op;

  let Inst{6-10}  = rd;
  let Inst{11-15} = ra; 
  let Inst{16-31} = imm16;
}

//===----------------------------------------------------------------------===//
// Float instruction class in MBlaze : <|opcode|rd|ra|flags|>
//===----------------------------------------------------------------------===//

class TF<bits<6> op, bits<11> flags, dag outs, dag ins, string asmstr,
         list<dag> pattern, InstrItinClass itin> : 
         MBlazeInst<outs, ins, asmstr, pattern, itin> 
{
  bits<5>  rd;
  bits<5>  ra;

  let opcode = op;

  let Inst{6-10}  = rd;
  let Inst{11-15} = ra; 
  let Inst{16-20} = 0;
  let Inst{21-31} = flags;
}

//===----------------------------------------------------------------------===//
// Branch instruction class in MBlaze : <|opcode|rd|br|ra|flags|>
//===----------------------------------------------------------------------===//

class TBR<bits<6> op, bits<5> br, bits<11> flags, dag outs, dag ins,
          string asmstr, list<dag> pattern, InstrItinClass itin> :
          MBlazeInst<outs, ins, asmstr, pattern, itin>
{
  bits<5> ra;

  let opcode = op;

  let Inst{6-10}  = 0;
  let Inst{11-15} = br; 
  let Inst{16-20} = ra;
  let Inst{21-31} = flags;
}

class TBRC<bits<6> op, bits<5> br, bits<11> flags, dag outs, dag ins,
           string asmstr, list<dag> pattern, InstrItinClass itin> :
           MBlazeInst<outs, ins, asmstr, pattern, itin>
{
  bits<5> ra;
  bits<5> rb;

  let opcode = op;

  let Inst{6-10}  = br;
  let Inst{11-15} = ra; 
  let Inst{16-20} = rb;
  let Inst{21-31} = flags;
}

class TBRL<bits<6> op, bits<5> br, bits<11> flags, dag outs, dag ins,
           string asmstr, list<dag> pattern, InstrItinClass itin> :
           MBlazeInst<outs, ins, asmstr, pattern, itin>
{
  bits<5> ra;

  let opcode = op;

  let Inst{6-10}  = 0xF;
  let Inst{11-15} = br; 
  let Inst{16-20} = ra;
  let Inst{21-31} = flags;
}

class TBRI<bits<6> op, bits<5> br, dag outs, dag ins,
           string asmstr, list<dag> pattern, InstrItinClass itin> :
           MBlazeInst<outs, ins, asmstr, pattern, itin>
{
  bits<16> imm16;

  let opcode = op;

  let Inst{6-10}  = 0;
  let Inst{11-15} = br; 
  let Inst{16-31} = imm16;
}

class TBRLI<bits<6> op, bits<5> br, dag outs, dag ins,
            string asmstr, list<dag> pattern, InstrItinClass itin> :
            MBlazeInst<outs, ins, asmstr, pattern, itin>
{
  bits<16> imm16;

  let opcode = op;

  let Inst{6-10}  = 0xF;
  let Inst{11-15} = br; 
  let Inst{16-31} = imm16;
}

class TBRCI<bits<6> op, bits<5> br, dag outs, dag ins,
            string asmstr, list<dag> pattern, InstrItinClass itin> :
            MBlazeInst<outs, ins, asmstr, pattern, itin>
{
  bits<5> ra;
  bits<16> imm16;

  let opcode = op;

  let Inst{6-10}  = br;
  let Inst{11-15} = ra; 
  let Inst{16-31} = imm16;
}

class TRET<bits<6> op, dag outs, dag ins,
            string asmstr, list<dag> pattern, InstrItinClass itin> :
            MBlazeInst<outs, ins, asmstr, pattern, itin>
{
  bits<5>  ra;
  bits<16> imm16;

  let opcode = op;

  let Inst{6-10}  = 0x10;
  let Inst{11-15} = ra; 
  let Inst{16-31} = imm16;
}
