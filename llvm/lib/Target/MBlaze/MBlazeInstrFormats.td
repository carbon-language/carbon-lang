//===- MBlazeInstrFormats.td - MB Instruction defs ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<6> val> {
      bits<6> Value = val;
}

def FPseudo : Format<0>;
def FRRR    : Format<1>;  // ADD, RSUB, OR, etc.
def FRRI    : Format<2>;  // ADDI, RSUBI, ORI, etc.
def FCRR    : Format<3>;  // PUTD, WDC, WIC, BEQ, BNE, BGE, etc.
def FCRI    : Format<4>;  // RTID, RTED, RTSD, BEQI, BNEI, BGEI, etc.
def FRCR    : Format<5>;  // BRLD, BRALD, GETD
def FRCI    : Format<6>;  // BRLID, BRALID, MSRCLR, MSRSET
def FCCR    : Format<7>;  // BR, BRA, BRD, etc.
def FCCI    : Format<8>;  // IMM, BRI, BRAI, BRID, etc.
def FRRCI   : Format<9>;  // BSRLI, BSRAI, BSLLI
def FRRC    : Format<10>; // SEXT8, SEXT16, SRA, SRC, SRL, FLT, FINT, FSQRT
def FRCX    : Format<11>; // GET
def FRCS    : Format<12>; // MFS
def FCRCS   : Format<13>; // MTS
def FCRCX   : Format<14>; // PUT
def FCX     : Format<15>; // TPUT
def FCR     : Format<16>; // TPUTD
def FRIR    : Format<17>; // RSUBI
def FC      : Format<18>; // NOP

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
class MBlazeInst<bits<6> op, Format form, dag outs, dag ins, string asmstr, 
                 list<dag> pattern, InstrItinClass itin> : Instruction {
  let Namespace = "MBlaze";
  field bits<32> Inst;

  bits<6> opcode = op;
  Format Form = form;
  bits<6> FormBits = Form.Value;

  // Top 6 bits are the 'opcode' field
  let Inst{0-5} = opcode;

  // If the instruction is marked as a pseudo, set isCodeGenOnly so that the
  // assembler and disassmbler ignore it.
  let isCodeGenOnly = !eq(!cast<string>(form), "FPseudo");
  
  dag OutOperandList = outs;
  dag InOperandList  = ins;

  let AsmString   = asmstr;
  let Pattern     = pattern;
  let Itinerary   = itin;

  // TSFlags layout should be kept in sync with MBlazeInstrInfo.h.
  let TSFlags{5-0}   = FormBits;
}

//===----------------------------------------------------------------------===//
// Pseudo instruction class
//===----------------------------------------------------------------------===//
class MBlazePseudo<dag outs, dag ins, string asmstr, list<dag> pattern>:
      MBlazeInst<0x0, FPseudo, outs, ins, asmstr, pattern, IIPseudo>;

//===----------------------------------------------------------------------===//
// Type A instruction class in MBlaze : <|opcode|rd|ra|rb|flags|>
//===----------------------------------------------------------------------===//

class TA<bits<6> op, bits<11> flags, dag outs, dag ins, string asmstr,
         list<dag> pattern, InstrItinClass itin> :
         MBlazeInst<op,FRRR,outs, ins, asmstr, pattern, itin>
{
  bits<5> rd;
  bits<5> ra;
  bits<5> rb;

  let Inst{6-10}  = rd;
  let Inst{11-15} = ra;
  let Inst{16-20} = rb;
  let Inst{21-31} = flags;
}

//===----------------------------------------------------------------------===//
// Type B instruction class in MBlaze : <|opcode|rd|ra|immediate|>
//===----------------------------------------------------------------------===//

class TB<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
         InstrItinClass itin> :
         MBlazeInst<op, FRRI, outs, ins, asmstr, pattern, itin>
{
  bits<5>  rd;
  bits<5>  ra;
  bits<16> imm16;

  let Inst{6-10}  = rd;
  let Inst{11-15} = ra;
  let Inst{16-31} = imm16;
}

//===----------------------------------------------------------------------===//
// Type B instruction class in MBlaze but with the operands reversed in
// the LLVM DAG : <|opcode|rd|ra|immediate|>
//===----------------------------------------------------------------------===//
class TBR<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
         InstrItinClass itin> :
         TB<op, outs, ins, asmstr, pattern, itin> {
  bits<5>  rrd;
  bits<16> rimm16;
  bits<5>  rra;

  let Form = FRIR;

  let rd = rrd;
  let ra = rra;
  let imm16 = rimm16;
}
