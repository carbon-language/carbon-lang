//===- MBlazeInstrFPU.td - MBlaze FPU Instruction defs ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MBlaze profiles and nodes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MBlaze Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Memory Access Instructions
//===----------------------------------------------------------------------===//
class LoadFM<bits<6> op, string instr_asm, PatFrag OpNode> :
             TA<op, 0x000, (outs FGR32:$dst), (ins memrr:$addr),
                !strconcat(instr_asm, "   $dst, $addr"),
                [(set FGR32:$dst, (OpNode xaddr:$addr))], IILoad>;

class LoadFMI<bits<6> op, string instr_asm, PatFrag OpNode> :
              TAI<op, (outs FGR32:$dst), (ins memri:$addr),
                  !strconcat(instr_asm, "   $dst, $addr"),
                  [(set FGR32:$dst, (OpNode iaddr:$addr))], IILoad>;

class StoreFM<bits<6> op, string instr_asm, PatFrag OpNode> :
              TA<op, 0x000, (outs), (ins FGR32:$dst, memrr:$addr),
                 !strconcat(instr_asm, "   $dst, $addr"),
                 [(OpNode FGR32:$dst, xaddr:$addr)], IIStore>;

class StoreFMI<bits<6> op, string instr_asm, PatFrag OpNode> :
               TAI<op, (outs), (ins FGR32:$dst, memrr:$addr),
                   !strconcat(instr_asm, "   $dst, $addr"),
                   [(OpNode FGR32:$dst, iaddr:$addr)], IIStore>;

class ArithF<bits<6> op, bits<11> flags, string instr_asm, SDNode OpNode,
             InstrItinClass itin> :
             TA<op, flags, (outs FGR32:$dst), (ins FGR32:$b, FGR32:$c),
                !strconcat(instr_asm, "   $dst, $b, $c"),
                [(set FGR32:$dst, (OpNode FGR32:$b, FGR32:$c))], itin>;

class CmpFN<bits<6> op, bits<11> flags, string instr_asm,
            InstrItinClass itin> :
            TA<op, flags, (outs CPURegs:$dst), (ins FGR32:$b, FGR32:$c),
               !strconcat(instr_asm, "   $dst, $b, $c"),
               [], itin>;

class ArithFR<bits<6> op, bits<11> flags, string instr_asm, SDNode OpNode,
             InstrItinClass itin> :
             TA<op, flags, (outs FGR32:$dst), (ins FGR32:$b, FGR32:$c),
                !strconcat(instr_asm, "   $dst, $c, $b"),
                [(set FGR32:$dst, (OpNode FGR32:$b, FGR32:$c))], itin>;

class ArithF2<bits<6> op, bits<11> flags, string instr_asm,
              InstrItinClass itin> :
              TF<op, flags, (outs FGR32:$dst), (ins FGR32:$b),
                 !strconcat(instr_asm, "   $dst, $b"),
                 [], itin>;

class ArithIF<bits<6> op, bits<11> flags, string instr_asm,
              InstrItinClass itin> :
              TF<op, flags, (outs FGR32:$dst), (ins CPURegs:$b),
                 !strconcat(instr_asm, "   $dst, $b"),
                 [], itin>;

class ArithFI<bits<6> op, bits<11> flags, string instr_asm,
              InstrItinClass itin> :
              TF<op, flags, (outs CPURegs:$dst), (ins FGR32:$b),
                 !strconcat(instr_asm, "   $dst, $b"),
                 [], itin>;

class LogicF<bits<6> op, string instr_asm> :
             TAI<op, (outs FGR32:$dst), (ins FGR32:$b, FGR32:$c),
                 !strconcat(instr_asm, "   $dst, $b, $c"),
                 [],
                 IIAlu>;

class LogicFI<bits<6> op, string instr_asm> :
             TAI<op, (outs FGR32:$dst), (ins FGR32:$b, fimm:$c),
                 !strconcat(instr_asm, "   $dst, $b, $c"),
                 [],
                 IIAlu>;

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FPU Arithmetic Instructions
//===----------------------------------------------------------------------===//
let Predicates=[HasFPU] in {
  def FOR    :  LogicF<0x28, "or     ">;
  def FORI   : LogicFI<0x28, "ori    ">;
  def FADD   :  ArithF<0x16, 0x000, "fadd   ", fadd, IIAlu>;
  def FRSUB  : ArithFR<0x16, 0x080, "frsub  ", fsub, IIAlu>;
  def FMUL   :  ArithF<0x16, 0x100, "fmul   ", fmul, IIAlu>;
  def FDIV   :  ArithF<0x16, 0x180, "fdiv   ", fdiv, IIAlu>;

  def LWF    :   LoadFM<0x32, "lw     ", load>;
  def LWFI   :  LoadFMI<0x32, "lwi    ", load>;

  def SWF    :  StoreFM<0x32, "sw     ", store>;
  def SWFI   : StoreFMI<0x32, "swi    ", store>;
}

let Predicates=[HasFPU,HasSqrt] in {
  def FLT    : ArithIF<0x16, 0x280, "flt    ", IIAlu>;
  def FINT   : ArithFI<0x16, 0x300, "fint   ", IIAlu>;
  def FSQRT  : ArithF2<0x16, 0x300, "fsqrt  ", IIAlu>;
}

let isAsCheapAsAMove = 1 in {
  def FCMP_UN : CmpFN<0x16, 0x200, "fcmp.un", IIAlu>;
  def FCMP_LT : CmpFN<0x16, 0x210, "fcmp.lt", IIAlu>;
  def FCMP_EQ : CmpFN<0x16, 0x220, "fcmp.eq", IIAlu>;
  def FCMP_LE : CmpFN<0x16, 0x230, "fcmp.le", IIAlu>;
  def FCMP_GT : CmpFN<0x16, 0x240, "fcmp.gt", IIAlu>;
  def FCMP_NE : CmpFN<0x16, 0x250, "fcmp.ne", IIAlu>;
  def FCMP_GE : CmpFN<0x16, 0x260, "fcmp.ge", IIAlu>;
}


let usesCustomInserter = 1 in {
  def Select_FCC : MBlazePseudo<(outs FGR32:$dst),
    (ins FGR32:$T, FGR32:$F, CPURegs:$CMP, i32imm:$CC),
    "; SELECT_FCC PSEUDO!",
    []>;
}

// Floating point conversions
let Predicates=[HasFPU] in {
  def : Pat<(sint_to_fp CPURegs:$V), (FLT CPURegs:$V)>;
  def : Pat<(fp_to_sint FGR32:$V), (FINT FGR32:$V)>;
  def : Pat<(fsqrt FGR32:$V), (FSQRT FGR32:$V)>;
}

// SET_CC operations
let Predicates=[HasFPU] in {
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETEQ),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_EQ FGR32:$L, FGR32:$R), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETNE),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_EQ FGR32:$L, FGR32:$R), 1)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETOEQ),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_EQ FGR32:$L, FGR32:$R), 2)>;
 def : Pat<(setcc FGR32:$L, FGR32:$R, SETONE),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (XOR (FCMP_UN FGR32:$L, FGR32:$R),
                            (FCMP_EQ FGR32:$L, FGR32:$R)), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETONE),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (OR (FCMP_UN FGR32:$L, FGR32:$R),
                           (FCMP_EQ FGR32:$L, FGR32:$R)), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETGT),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_GT FGR32:$L, FGR32:$R), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETLT),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_LT FGR32:$L, FGR32:$R), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETGE),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_GE FGR32:$L, FGR32:$R), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETLE),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_LE FGR32:$L, FGR32:$R), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETOGT),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_GT FGR32:$L, FGR32:$R), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETOLT),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_LT FGR32:$L, FGR32:$R), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETOGE),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_GE FGR32:$L, FGR32:$R), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETOLE),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_LE FGR32:$L, FGR32:$R), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETUEQ),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (OR (FCMP_UN FGR32:$L, FGR32:$R),
                           (FCMP_EQ FGR32:$L, FGR32:$R)), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETUNE),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_NE FGR32:$L, FGR32:$R), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETUGT),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (OR (FCMP_UN FGR32:$L, FGR32:$R),
                           (FCMP_GT FGR32:$L, FGR32:$R)), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETULT),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (OR (FCMP_UN FGR32:$L, FGR32:$R),
                           (FCMP_LT FGR32:$L, FGR32:$R)), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETUGE),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (OR (FCMP_UN FGR32:$L, FGR32:$R),
                           (FCMP_GE FGR32:$L, FGR32:$R)), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETULE),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (OR (FCMP_UN FGR32:$L, FGR32:$R),
                           (FCMP_LE FGR32:$L, FGR32:$R)), 2)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETO),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_UN FGR32:$L, FGR32:$R), 1)>;
  def : Pat<(setcc FGR32:$L, FGR32:$R, SETUO),
            (Select_CC (ADDI R0, 1), (ADDI R0, 0),
                       (FCMP_UN FGR32:$L, FGR32:$R), 2)>;
}

// SELECT operations
def : Pat<(select CPURegs:$C, FGR32:$T, FGR32:$F),
          (Select_FCC FGR32:$T, FGR32:$F, CPURegs:$C, 2)>;

//===----------------------------------------------------------------------===//
// Patterns for Floating Point Instructions
//===----------------------------------------------------------------------===//
def : Pat<(f32 fpimm:$imm), (FORI F0, fpimm:$imm)>;
