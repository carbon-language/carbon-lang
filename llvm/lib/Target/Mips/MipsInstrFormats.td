//===-- MipsInstrFormats.td - Mips Instruction Formats -----*- tablegen -*-===//
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
//  CPU INSTRUCTION FORMATS
//
//  opcode  - operation code.
//  rs      - src reg.
//  rt      - dst reg (on a 2 regs instr) or src reg (on a 3 reg instr).
//  rd      - dst reg, only used on 3 regs instr.
//  shamt   - only used on shift instructions, contains the shift amount.
//  funct   - combined with opcode field give us an operation code.
//
//===----------------------------------------------------------------------===//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<4> val> {
  bits<4> Value = val;
}

def Pseudo    : Format<0>;
def FrmR      : Format<1>;
def FrmI      : Format<2>;
def FrmJ      : Format<3>;
def FrmFR     : Format<4>;
def FrmFI     : Format<5>;
def FrmOther  : Format<6>; // Instruction w/ a custom format

class MMRel;

def Std2MicroMips : InstrMapping {
  let FilterClass = "MMRel";
  // Instructions with the same BaseOpcode and isNVStore values form a row.
  let RowFields = ["BaseOpcode"];
  // Instructions with the same predicate sense form a column.
  let ColFields = ["Arch"];
  // The key column is the unpredicated instructions.
  let KeyCol = ["se"];
  // Value columns are PredSense=true and PredSense=false
  let ValueCols = [["se"], ["micromips"]];
}

class StdArch {
  string Arch = "se";
}

// Generic Mips Format
class MipsInst<dag outs, dag ins, string asmstr, list<dag> pattern,
               InstrItinClass itin, Format f>: Instruction
{
  field bits<32> Inst;
  Format Form = f;

  let Namespace = "Mips";

  let Size = 4;

  bits<6> Opcode = 0;

  // Top 6 bits are the 'opcode' field
  let Inst{31-26} = Opcode;

  let OutOperandList = outs;
  let InOperandList  = ins;

  let AsmString   = asmstr;
  let Pattern     = pattern;
  let Itinerary   = itin;

  //
  // Attributes specific to Mips instructions...
  //
  bits<4> FormBits = Form.Value;

  // TSFlags layout should be kept in sync with MipsInstrInfo.h.
  let TSFlags{3-0}   = FormBits;

  let DecoderNamespace = "Mips";

  field bits<32> SoftFail = 0;
}

// Mips32/64 Instruction Format
class InstSE<dag outs, dag ins, string asmstr, list<dag> pattern,
             InstrItinClass itin, Format f, string opstr = ""> :
  MipsInst<outs, ins, asmstr, pattern, itin, f> {
  let Predicates = [HasStdEnc];
  string BaseOpcode = opstr;
  string Arch;
}

// Mips Pseudo Instructions Format
class MipsPseudo<dag outs, dag ins, list<dag> pattern,
                 InstrItinClass itin = IIPseudo> :
  MipsInst<outs, ins, "", pattern, itin, Pseudo> {
  let isCodeGenOnly = 1;
  let isPseudo = 1;
}

// Mips32/64 Pseudo Instruction Format
class PseudoSE<dag outs, dag ins, list<dag> pattern,
               InstrItinClass itin = IIPseudo>:
  MipsPseudo<outs, ins, pattern, itin> {
  let Predicates = [HasStdEnc];
}

// Pseudo-instructions for alternate assembly syntax (never used by codegen).
// These are aliases that require C++ handling to convert to the target
// instruction, while InstAliases can be handled directly by tblgen.
class MipsAsmPseudoInst<dag outs, dag ins, string asmstr>:
  MipsInst<outs, ins, asmstr, [], IIPseudo, Pseudo> {
  let isPseudo = 1;
  let Pattern = [];
}
//===----------------------------------------------------------------------===//
// Format R instruction class in Mips : <|opcode|rs|rt|rd|shamt|funct|>
//===----------------------------------------------------------------------===//

class FR<bits<6> op, bits<6> _funct, dag outs, dag ins, string asmstr,
         list<dag> pattern, InstrItinClass itin>:
  InstSE<outs, ins, asmstr, pattern, itin, FrmR>
{
  bits<5>  rd;
  bits<5>  rs;
  bits<5>  rt;
  bits<5>  shamt;
  bits<6>  funct;

  let Opcode = op;
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
         InstrItinClass itin>: InstSE<outs, ins, asmstr, pattern, itin, FrmI>
{
  bits<5>  rt;
  bits<5>  rs;
  bits<16> imm16;

  let Opcode = op;

  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-0}  = imm16;
}

class BranchBase<bits<6> op, dag outs, dag ins, string asmstr,
                  list<dag> pattern, InstrItinClass itin>:
  InstSE<outs, ins, asmstr, pattern, itin, FrmI>
{
  bits<5>  rs;
  bits<5>  rt;
  bits<16> imm16;

  let Opcode = op;

  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-0}  = imm16;
}

//===----------------------------------------------------------------------===//
// Format J instruction class in Mips : <|opcode|address|>
//===----------------------------------------------------------------------===//

class FJ<bits<6> op>
{
  bits<26> target;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-0}  = target;
}

//===----------------------------------------------------------------------===//
// MFC instruction class in Mips : <|op|mf|rt|rd|0000000|sel|>
//===----------------------------------------------------------------------===//
class MFC3OP_FM<bits<6> op, bits<5> mfmt>
{
  bits<5> rt;
  bits<5> rd;
  bits<3> sel;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = mfmt;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-3}  = 0;
  let Inst{2-0}   = sel;
}

class ADD_FM<bits<6> op, bits<6> funct> : StdArch {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-6}  = 0;
  let Inst{5-0}   = funct;
}

class ADDI_FM<bits<6> op> : StdArch {
  bits<5>  rs;
  bits<5>  rt;
  bits<16> imm16;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-0}  = imm16;
}

class SRA_FM<bits<6> funct, bit rotate> : StdArch {
  bits<5> rd;
  bits<5> rt;
  bits<5> shamt;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{25-22} = 0;
  let Inst{21}    = rotate;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-6}  = shamt;
  let Inst{5-0}   = funct;
}

class SRLV_FM<bits<6> funct, bit rotate> : StdArch {
  bits<5> rd;
  bits<5> rt;
  bits<5> rs;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-7}  = 0;
  let Inst{6}     = rotate;
  let Inst{5-0}   = funct;
}

class BEQ_FM<bits<6> op> {
  bits<5>  rs;
  bits<5>  rt;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-0}  = offset;
}

class BGEZ_FM<bits<6> op, bits<5> funct> {
  bits<5>  rs;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rs;
  let Inst{20-16} = funct;
  let Inst{15-0}  = offset;
}

class B_FM {
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = 4;
  let Inst{25-21} = 0;
  let Inst{20-16} = 0;
  let Inst{15-0}  = offset;
}

class SLTI_FM<bits<6> op> : StdArch {
  bits<5> rt;
  bits<5> rs;
  bits<16> imm16;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-0}  = imm16;
}

class MFLO_FM<bits<6> funct> {
  bits<5> rd;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{25-16} = 0;
  let Inst{15-11} = rd;
  let Inst{10-6}  = 0;
  let Inst{5-0}   = funct;
}

class MTLO_FM<bits<6> funct> {
  bits<5> rs;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{25-21} = rs;
  let Inst{20-6}  = 0;
  let Inst{5-0}   = funct;
}

class SEB_FM<bits<5> funct, bits<6> funct2> {
  bits<5> rd;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = 0x1f;
  let Inst{25-21} = 0;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-6}  = funct;
  let Inst{5-0}   = funct2;
}

class CLO_FM<bits<6> funct> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = 0x1c;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-6}  = 0;
  let Inst{5-0}   = funct;
  let rt = rd;
}

class LUI_FM {
  bits<5> rt;
  bits<16> imm16;

  bits<32> Inst;

  let Inst{31-26} = 0xf;
  let Inst{25-21} = 0;
  let Inst{20-16} = rt;
  let Inst{15-0}  = imm16;
}

class JALR_FM {
  bits<5> rd;
  bits<5> rs;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{25-21} = rs;
  let Inst{20-16} = 0;
  let Inst{15-11} = rd;
  let Inst{10-6}  = 0;
  let Inst{5-0}   = 9;
}

class BAL_FM {
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = 1;
  let Inst{25-21} = 0;
  let Inst{20-16} = 0x11;
  let Inst{15-0}  = offset;
}

class BGEZAL_FM<bits<5> funct> {
  bits<5>  rs;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = 1;
  let Inst{25-21} = rs;
  let Inst{20-16} = funct;
  let Inst{15-0}  = offset;
}

class SYNC_FM {
  bits<5> stype;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{10-6}  = stype;
  let Inst{5-0}   = 0xf;
}

class MULT_FM<bits<6> op, bits<6> funct> : StdArch {
  bits<5>  rs;
  bits<5>  rt;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-6}  = 0;
  let Inst{5-0}   = funct;
}

class EXT_FM<bits<6> funct> {
  bits<5> rt;
  bits<5> rs;
  bits<5> pos;
  bits<5> size;

  bits<32> Inst;

  let Inst{31-26} = 0x1f;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = size;
  let Inst{10-6}  = pos;
  let Inst{5-0}   = funct;
}

class RDHWR_FM {
  bits<5> rt;
  bits<5> rd;

  bits<32> Inst;

  let Inst{31-26} = 0x1f;
  let Inst{25-21} = 0;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-6}  = 0;
  let Inst{5-0}   = 0x3b;
}

//===----------------------------------------------------------------------===//
//
//  FLOATING POINT INSTRUCTION FORMATS
//
//  opcode  - operation code.
//  fs      - src reg.
//  ft      - dst reg (on a 2 regs instr) or src reg (on a 3 reg instr).
//  fd      - dst reg, only used on 3 regs instr.
//  fmt     - double or single precision.
//  funct   - combined with opcode field give us an operation code.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Format FI instruction class in Mips : <|opcode|base|ft|immediate|>
//===----------------------------------------------------------------------===//

class FFI<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern>:
  InstSE<outs, ins, asmstr, pattern, NoItinerary, FrmFI>
{
  bits<5>  ft;
  bits<5>  base;
  bits<16> imm16;

  let Opcode = op;

  let Inst{25-21} = base;
  let Inst{20-16} = ft;
  let Inst{15-0}  = imm16;
}

class ADDS_FM<bits<6> funct, bits<5> fmt> {
  bits<5> fd;
  bits<5> fs;
  bits<5> ft;

  bits<32> Inst;

  let Inst{31-26} = 0x11;
  let Inst{25-21} = fmt;
  let Inst{20-16} = ft;
  let Inst{15-11} = fs;
  let Inst{10-6}  = fd;
  let Inst{5-0}   = funct;
}

class ABSS_FM<bits<6> funct, bits<5> fmt> {
  bits<5> fd;
  bits<5> fs;

  bits<32> Inst;

  let Inst{31-26} = 0x11;
  let Inst{25-21} = fmt;
  let Inst{20-16} = 0;
  let Inst{15-11} = fs;
  let Inst{10-6}  = fd;
  let Inst{5-0}   = funct;
}

class MFC1_FM<bits<5> funct> {
  bits<5> rt;
  bits<5> fs;

  bits<32> Inst;

  let Inst{31-26} = 0x11;
  let Inst{25-21} = funct;
  let Inst{20-16} = rt;
  let Inst{15-11} = fs;
  let Inst{10-0}  = 0;
}

class LW_FM<bits<6> op> {
  bits<5> rt;
  bits<21> addr;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = addr{20-16};
  let Inst{20-16} = rt;
  let Inst{15-0}  = addr{15-0};
}

class MADDS_FM<bits<3> funct, bits<3> fmt> {
  bits<5> fd;
  bits<5> fr;
  bits<5> fs;
  bits<5> ft;

  bits<32> Inst;

  let Inst{31-26} = 0x13;
  let Inst{25-21} = fr;
  let Inst{20-16} = ft;
  let Inst{15-11} = fs;
  let Inst{10-6}  = fd;
  let Inst{5-3}   = funct;
  let Inst{2-0}   = fmt;
}

class LWXC1_FM<bits<6> funct> {
  bits<5> fd;
  bits<5> base;
  bits<5> index;

  bits<32> Inst;

  let Inst{31-26} = 0x13;
  let Inst{25-21} = base;
  let Inst{20-16} = index;
  let Inst{15-11} = 0;
  let Inst{10-6}  = fd;
  let Inst{5-0}   = funct;
}

class SWXC1_FM<bits<6> funct> {
  bits<5> fs;
  bits<5> base;
  bits<5> index;

  bits<32> Inst;

  let Inst{31-26} = 0x13;
  let Inst{25-21} = base;
  let Inst{20-16} = index;
  let Inst{15-11} = fs;
  let Inst{10-6}  = 0;
  let Inst{5-0}   = funct;
}

class BC1F_FM<bit nd, bit tf> {
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = 0x11;
  let Inst{25-21} = 0x8;
  let Inst{20-18} = 0; // cc
  let Inst{17} = nd;
  let Inst{16} = tf;
  let Inst{15-0} = offset;
}

class CEQS_FM<bits<5> fmt> {
  bits<5> fs;
  bits<5> ft;
  bits<4> cond;

  bits<32> Inst;

  let Inst{31-26} = 0x11;
  let Inst{25-21} = fmt;
  let Inst{20-16} = ft;
  let Inst{15-11} = fs;
  let Inst{10-8} = 0; // cc
  let Inst{7-4} = 0x3;
  let Inst{3-0} = cond;
}

class CMov_I_F_FM<bits<6> funct, bits<5> fmt> {
  bits<5> fd;
  bits<5> fs;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = 0x11;
  let Inst{25-21} = fmt;
  let Inst{20-16} = rt;
  let Inst{15-11} = fs;
  let Inst{10-6} = fd;
  let Inst{5-0} = funct;
}

class CMov_F_I_FM<bit tf> {
  bits<5> rd;
  bits<5> rs;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{25-21} = rs;
  let Inst{20-18} = 0; // cc
  let Inst{17} = 0;
  let Inst{16} = tf;
  let Inst{15-11} = rd;
  let Inst{10-6} = 0;
  let Inst{5-0} = 1;
}

class CMov_F_F_FM<bits<5> fmt, bit tf> {
  bits<5> fd;
  bits<5> fs;

  bits<32> Inst;

  let Inst{31-26} = 0x11;
  let Inst{25-21} = fmt;
  let Inst{20-18} = 0; // cc
  let Inst{17} = 0;
  let Inst{16} = tf;
  let Inst{15-11} = fs;
  let Inst{10-6} = fd;
  let Inst{5-0} = 0x11;
}
