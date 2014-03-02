//===-- SparcInstrAliases.td - Instruction Aliases for Sparc Target -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains instruction aliases for Sparc.
//===----------------------------------------------------------------------===//

// Instruction aliases for conditional moves.

// mov<cond> <ccreg> rs2, rd
multiclass intcond_mov_alias<string cond, int condVal, string ccreg,
                          Instruction movrr, Instruction movri,
                          Instruction fmovs, Instruction fmovd> {

  // mov<cond> (%icc|%xcc), rs2, rd
  def : InstAlias<!strconcat(!strconcat(!strconcat("mov", cond), ccreg),
                             ", $rs2, $rd"),
                  (movrr IntRegs:$rd, IntRegs:$rs2, condVal)>;

  // mov<cond> (%icc|%xcc), simm11, rd
  def : InstAlias<!strconcat(!strconcat(!strconcat("mov", cond), ccreg),
                             ", $simm11, $rd"),
                  (movri IntRegs:$rd, i32imm:$simm11, condVal)>;

  // fmovs<cond> (%icc|%xcc), $rs2, $rd
  def : InstAlias<!strconcat(!strconcat(!strconcat("fmovs", cond), ccreg),
                             ", $rs2, $rd"),
                  (fmovs FPRegs:$rd, FPRegs:$rs2, condVal)>;

  // fmovd<cond> (%icc|%xcc), $rs2, $rd
  def : InstAlias<!strconcat(!strconcat(!strconcat("fmovd", cond), ccreg),
                             ", $rs2, $rd"),
                  (fmovd DFPRegs:$rd, DFPRegs:$rs2, condVal)>;
}

// mov<cond> <ccreg> rs2, rd
multiclass fpcond_mov_alias<string cond, int condVal,
                           Instruction movrr, Instruction movri,
                           Instruction fmovs, Instruction fmovd> {

  // mov<cond> %fcc[0-3], rs2, rd
  def : InstAlias<!strconcat(!strconcat("mov", cond), " $cc, $rs2, $rd"),
                  (movrr IntRegs:$rd, FCCRegs:$cc, IntRegs:$rs2, condVal)>;

  // mov<cond> %fcc[0-3], simm11, rd
  def : InstAlias<!strconcat(!strconcat("mov", cond), " $cc, $simm11, $rd"),
                  (movri IntRegs:$rd, FCCRegs:$cc, i32imm:$simm11, condVal)>;

  // fmovs<cond> %fcc[0-3], $rs2, $rd
  def : InstAlias<!strconcat(!strconcat("fmovs", cond), " $cc, $rs2, $rd"),
                  (fmovs FPRegs:$rd, FCCRegs:$cc, FPRegs:$rs2, condVal)>;

  // fmovd<cond> %fcc[0-3], $rs2, $rd
  def : InstAlias<!strconcat(!strconcat("fmovd", cond), " $cc, $rs2, $rd"),
                  (fmovd DFPRegs:$rd, FCCRegs:$cc, DFPRegs:$rs2, condVal)>;
}

// Instruction aliases for integer conditional branches and moves.
multiclass int_cond_alias<string cond, int condVal> {

  // b<cond> $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), " $imm"),
                  (BCOND brtarget:$imm, condVal)>;

  // b<cond>,a $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",a $imm"),
                  (BCONDA brtarget:$imm, condVal)>;

  // b<cond> %icc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), " %icc, $imm"),
                  (BPICC brtarget:$imm, condVal)>, Requires<[HasV9]>;

  // b<cond>,pt %icc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",pt %icc, $imm"),
                  (BPICC brtarget:$imm, condVal)>, Requires<[HasV9]>;

  // b<cond>,a %icc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",a %icc, $imm"),
                  (BPICCA brtarget:$imm, condVal)>, Requires<[HasV9]>;

  // b<cond>,a,pt %icc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",a,pt %icc, $imm"),
                  (BPICCA brtarget:$imm, condVal)>, Requires<[HasV9]>;

  // b<cond>,pn %icc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",pn %icc, $imm"),
                  (BPICCNT brtarget:$imm, condVal)>, Requires<[HasV9]>;

  // b<cond>,a,pn %icc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",a,pn %icc, $imm"),
                  (BPICCANT brtarget:$imm, condVal)>, Requires<[HasV9]>;

  // b<cond> %xcc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), " %xcc, $imm"),
                  (BPXCC brtarget:$imm, condVal)>, Requires<[Is64Bit]>;

  // b<cond>,pt %xcc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",pt %xcc, $imm"),
                  (BPXCC brtarget:$imm, condVal)>, Requires<[Is64Bit]>;

  // b<cond>,a %xcc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",a %xcc, $imm"),
                  (BPXCCA brtarget:$imm, condVal)>, Requires<[Is64Bit]>;

  // b<cond>,a,pt %xcc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",a,pt %xcc, $imm"),
                  (BPXCCA brtarget:$imm, condVal)>, Requires<[Is64Bit]>;

  // b<cond>,pn %xcc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",pn %xcc, $imm"),
                  (BPXCCNT brtarget:$imm, condVal)>, Requires<[Is64Bit]>;

  // b<cond>,a,pn %xcc, $imm
  def : InstAlias<!strconcat(!strconcat("b", cond), ",a,pn %xcc, $imm"),
                  (BPXCCANT brtarget:$imm, condVal)>, Requires<[Is64Bit]>;


  defm : intcond_mov_alias<cond, condVal, " %icc",
                            MOVICCrr, MOVICCri,
                            FMOVS_ICC, FMOVD_ICC>, Requires<[HasV9]>;

  defm : intcond_mov_alias<cond, condVal, " %xcc",
                            MOVXCCrr, MOVXCCri,
                            FMOVS_XCC, FMOVD_XCC>, Requires<[Is64Bit]>;

  // fmovq<cond> (%icc|%xcc), $rs2, $rd
  def : InstAlias<!strconcat(!strconcat("fmovq", cond), " %icc, $rs2, $rd"),
                  (FMOVQ_ICC QFPRegs:$rd, QFPRegs:$rs2, condVal)>,
                  Requires<[HasV9, HasHardQuad]>;
  def : InstAlias<!strconcat(!strconcat("fmovq", cond), " %xcc, $rs2, $rd"),
                  (FMOVQ_XCC QFPRegs:$rd, QFPRegs:$rs2, condVal)>,
                  Requires<[Is64Bit, HasHardQuad]>;

}


// Instruction aliases for floating point conditional branches and moves.
multiclass fp_cond_alias<string cond, int condVal> {

  // fb<cond> $imm
  def : InstAlias<!strconcat(!strconcat("fb", cond), " $imm"),
                  (FBCOND brtarget:$imm, condVal), 0>;

  // fb<cond>,a $imm
  def : InstAlias<!strconcat(!strconcat("fb", cond), ",a $imm"),
                  (FBCONDA brtarget:$imm, condVal), 0>;

  // fb<cond> %fcc0, $imm
  def : InstAlias<!strconcat(!strconcat("fb", cond), " $cc, $imm"),
                  (BPFCC brtarget:$imm, condVal, FCCRegs:$cc)>,
                  Requires<[HasV9]>;

  // fb<cond>,pt %fcc0, $imm
  def : InstAlias<!strconcat(!strconcat("fb", cond), ",pt $cc, $imm"),
                  (BPFCC brtarget:$imm, condVal, FCCRegs:$cc)>,
                  Requires<[HasV9]>;

  // fb<cond>,a %fcc0, $imm
  def : InstAlias<!strconcat(!strconcat("fb", cond), ",a $cc, $imm"),
                  (BPFCCA brtarget:$imm, condVal, FCCRegs:$cc)>,
                  Requires<[HasV9]>;

  // fb<cond>,a,pt %fcc0, $imm
  def : InstAlias<!strconcat(!strconcat("fb", cond), ",a,pt $cc, $imm"),
                  (BPFCCA brtarget:$imm, condVal, FCCRegs:$cc)>,
                   Requires<[HasV9]>;

  // fb<cond>,pn %fcc0, $imm
  def : InstAlias<!strconcat(!strconcat("fb", cond), ",pn $cc, $imm"),
                  (BPFCCNT brtarget:$imm, condVal, FCCRegs:$cc)>,
                   Requires<[HasV9]>;

  // fb<cond>,a,pn %fcc0, $imm
  def : InstAlias<!strconcat(!strconcat("fb", cond), ",a,pn $cc, $imm"),
                  (BPFCCANT brtarget:$imm, condVal, FCCRegs:$cc)>, Requires<[HasV9]>;

  defm : fpcond_mov_alias<cond, condVal,
                          V9MOVFCCrr, V9MOVFCCri,
                          V9FMOVS_FCC, V9FMOVD_FCC>, Requires<[HasV9]>;

  // fmovq<cond> %fcc0, $rs2, $rd
  def : InstAlias<!strconcat(!strconcat("fmovq", cond), " $cc, $rs2, $rd"),
                  (V9FMOVQ_FCC QFPRegs:$rd, FCCRegs:$cc, QFPRegs:$rs2,
                                                          condVal)>,
                  Requires<[HasV9, HasHardQuad]>;
}

defm : int_cond_alias<"a",    0b1000>;
defm : int_cond_alias<"n",    0b0000>;
defm : int_cond_alias<"ne",   0b1001>;
defm : int_cond_alias<"e",    0b0001>;
defm : int_cond_alias<"g",    0b1010>;
defm : int_cond_alias<"le",   0b0010>;
defm : int_cond_alias<"ge",   0b1011>;
defm : int_cond_alias<"l",    0b0011>;
defm : int_cond_alias<"gu",   0b1100>;
defm : int_cond_alias<"leu",  0b0100>;
defm : int_cond_alias<"cc",   0b1101>;
defm : int_cond_alias<"cs",   0b0101>;
defm : int_cond_alias<"pos",  0b1110>;
defm : int_cond_alias<"neg",  0b0110>;
defm : int_cond_alias<"vc",   0b1111>;
defm : int_cond_alias<"vs",   0b0111>;

defm : fp_cond_alias<"a",     0b0000>;
defm : fp_cond_alias<"n",     0b1000>;
defm : fp_cond_alias<"u",     0b0111>;
defm : fp_cond_alias<"g",     0b0110>;
defm : fp_cond_alias<"ug",    0b0101>;
defm : fp_cond_alias<"l",     0b0100>;
defm : fp_cond_alias<"ul",    0b0011>;
defm : fp_cond_alias<"lg",    0b0010>;
defm : fp_cond_alias<"ne",    0b0001>;
defm : fp_cond_alias<"e",     0b1001>;
defm : fp_cond_alias<"ue",    0b1010>;
defm : fp_cond_alias<"ge",    0b1011>;
defm : fp_cond_alias<"uge",   0b1100>;
defm : fp_cond_alias<"le",    0b1101>;
defm : fp_cond_alias<"ule",   0b1110>;
defm : fp_cond_alias<"o",     0b1111>;

// Instruction aliases for JMPL.

// jmp addr -> jmpl addr, %g0
def : InstAlias<"jmp $addr", (JMPLrr G0, MEMrr:$addr)>;
def : InstAlias<"jmp $addr", (JMPLri G0, MEMri:$addr)>;

// call addr -> jmpl addr, %o7
def : InstAlias<"call $addr", (JMPLrr O7, MEMrr:$addr)>;
def : InstAlias<"call $addr", (JMPLri O7, MEMri:$addr)>;

// retl -> RETL 8
def : InstAlias<"retl", (RETL 8)>;

// ret -> RET 8
def : InstAlias<"ret", (RET 8)>;

// mov reg, rd -> or %g0, reg, rd
def : InstAlias<"mov $rs2, $rd", (ORrr IntRegs:$rd, G0, IntRegs:$rs2)>;

// mov simm13, rd -> or %g0, simm13, rd
def : InstAlias<"mov $simm13, $rd", (ORri IntRegs:$rd, G0, i32imm:$simm13)>;

// restore -> restore %g0, %g0, %g0
def : InstAlias<"restore", (RESTORErr G0, G0, G0)>;

def : MnemonicAlias<"addc", "addx">, Requires<[HasV9]>;
def : MnemonicAlias<"addccc", "addxcc">, Requires<[HasV9]>;

def : MnemonicAlias<"subc", "subx">, Requires<[HasV9]>;
def : MnemonicAlias<"subccc", "subxcc">, Requires<[HasV9]>;


def : InstAlias<"fcmps $rs1, $rs2", (V9FCMPS FCC0, FPRegs:$rs1, FPRegs:$rs2)>;
def : InstAlias<"fcmpd $rs1, $rs2", (V9FCMPD FCC0, DFPRegs:$rs1, DFPRegs:$rs2)>;
def : InstAlias<"fcmpq $rs1, $rs2", (V9FCMPQ FCC0, QFPRegs:$rs1, QFPRegs:$rs2)>,
                Requires<[HasHardQuad]>;

def : InstAlias<"fcmpes $rs1, $rs2", (V9FCMPES FCC0, FPRegs:$rs1, FPRegs:$rs2)>;
def : InstAlias<"fcmped $rs1, $rs2", (V9FCMPED FCC0, DFPRegs:$rs1,
                                                     DFPRegs:$rs2)>;
def : InstAlias<"fcmpeq $rs1, $rs2", (V9FCMPEQ FCC0, QFPRegs:$rs1,
                                                     QFPRegs:$rs2)>,
                Requires<[HasHardQuad]>;
