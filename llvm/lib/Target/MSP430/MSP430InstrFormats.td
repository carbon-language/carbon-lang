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

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<2> val> {
  bits<2> Value = val;
}

class SourceMode<bits<2> val> {
  bits<2> Value = val;
}

class DestMode<bit val> {
  bit Value = val;
}

def PseudoFrm   : Format<0>;
def SingleOpFrm : Format<1>;
def DoubleOpFrm : Format<2>;
def CondJumpFrm : Format<3>;

def DstReg      : DestMode<0>;
def DstMem      : DestMode<1>;

def SrcReg      : SourceMode<0>;
def SrcMem      : SourceMode<1>;
def SrcIndReg   : SourceMode<2>;
def SrcPostInc  : SourceMode<3>;
def SrcImm      : SourceMode<3>;

// Generic MSP430 Format
class MSP430Inst<dag outs, dag ins, Format f, string asmstr> : Instruction {
  field bits<16> Inst;

  let Namespace = "MSP430";

  dag OutOperandList = outs;
  dag InOperandList  = ins;

  Format Form = f;
  bits<2> FormBits = Form.Value;

  let AsmString   = asmstr;
}

// FIXME: Create different classes for different addressing modes.

// MSP430 Double Operand (Format I) Instructions
class IForm<bits<4> opcode, DestMode dest, bit bw, SourceMode src,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, DoubleOpFrm, asmstr> {
  let Pattern = pattern;

  DestMode ad = dest;
  SourceMode  as = src;
  
  let Inst{12-15} = opcode;
  let Inst{7}     = ad.Value;
  let Inst{6}     = bw;
  let Inst{4-5}   = as.Value;
}

// 8 bit IForm instructions
class IForm8<bits<4> opcode, DestMode dest, SourceMode src,
             dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm<opcode, dest, 1, src, outs, ins, asmstr, pattern>;

class I8rr<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstReg, SrcReg, outs, ins, asmstr, pattern>;

class I8ri<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstReg, SrcImm, outs, ins, asmstr, pattern>;

class I8rm<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstReg, SrcMem, outs, ins, asmstr, pattern>;

class I8mr<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstMem, SrcReg, outs, ins, asmstr, pattern>;

class I8mi<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstMem, SrcImm, outs, ins, asmstr, pattern>;

class I8mm<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstMem, SrcMem, outs, ins, asmstr, pattern>;

// 16 bit IForm instructions
class IForm16<bits<4> opcode, DestMode dest, SourceMode src,
              dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm<opcode, dest, 0, src, outs, ins, asmstr, pattern>;

class I16rr<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstReg, SrcReg, outs, ins, asmstr, pattern>;

class I16ri<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstReg, SrcImm, outs, ins, asmstr, pattern>;

class I16rm<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstReg, SrcMem, outs, ins, asmstr, pattern>;

class I16mr<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstMem, SrcReg, outs, ins, asmstr, pattern>;

class I16mi<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstMem, SrcImm, outs, ins, asmstr, pattern>;

class I16mm<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstMem, SrcMem, outs, ins, asmstr, pattern>;

// MSP430 Single Operand (Format II) Instructions
class IIForm<bits<9> opcode, bit bw, SourceMode src,
             dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, SingleOpFrm, asmstr> {
  let Pattern = pattern;
  
  SourceMode as = src;

  let Inst{7-15} = opcode;
  let Inst{6}    = bw;
  let Inst{4-5}  = as.Value;
}

// 8 bit IIForm instructions
class IIForm8<bits<9> opcode, SourceMode src,
              dag outs, dag ins, string asmstr, list<dag> pattern>
  : IIForm<opcode, 1, src, outs, ins, asmstr, pattern>;

// 16 bit IIForm instructions
class IIForm16<bits<9> opcode, SourceMode src,
               dag outs, dag ins, string asmstr, list<dag> pattern>
  : IIForm<opcode, 0, src, outs, ins, asmstr, pattern>;

// MSP430 Conditional Jumps Instructions
class CJForm<bits<3> opcode, bits<3> cond, bit s,
             dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, CondJumpFrm, asmstr> {
  let Pattern = pattern;
  
  let Inst{13-15} = opcode;
  let Inst{10-12} = cond;
  let Inst{9}     = s;
}

// Pseudo instructions
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, PseudoFrm, asmstr> {
  let Pattern = pattern;
  let Inst{15-0} = 0;
}
