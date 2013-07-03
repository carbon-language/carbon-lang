//===-- SystemZInstrInfo.td - General SystemZ instructions ----*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Stack allocation
//===----------------------------------------------------------------------===//

def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i64imm:$amt),
                              [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                              [(callseq_end timm:$amt1, timm:$amt2)]>;

let neverHasSideEffects = 1 in {
  // Takes as input the value of the stack pointer after a dynamic allocation
  // has been made.  Sets the output to the address of the dynamically-
  // allocated area itself, skipping the outgoing arguments.
  //
  // This expands to an LA or LAY instruction.  We restrict the offset
  // to the range of LA and keep the LAY range in reserve for when
  // the size of the outgoing arguments is added.
  def ADJDYNALLOC : Pseudo<(outs GR64:$dst), (ins dynalloc12only:$src),
                           [(set GR64:$dst, dynalloc12only:$src)]>;
}

//===----------------------------------------------------------------------===//
// Control flow instructions
//===----------------------------------------------------------------------===//

// A return instruction.  R1 is the condition-code mask (all 1s)
// and R2 is the target address, which is always stored in %r14.
let isReturn = 1, isTerminator = 1, isBarrier = 1, hasCtrlDep = 1,
    R1 = 15, R2 = 14, isCodeGenOnly = 1 in {
  def RET : InstRR<0x07, (outs), (ins), "br\t%r14", [(z_retflag)]>;
}

// Unconditional branches.  R1 is the condition-code mask (all 1s).
let isBranch = 1, isTerminator = 1, isBarrier = 1, R1 = 15 in {
  let isIndirectBranch = 1 in
    def BR : InstRR<0x07, (outs), (ins ADDR64:$R2),
                    "br\t$R2", [(brind ADDR64:$R2)]>;

  // An assembler extended mnemonic for BRC.
  def J : InstRI<0xA74, (outs), (ins brtarget16:$I2), "j\t$I2",
                 [(br bb:$I2)]>;

  // An assembler extended mnemonic for BRCL.  (The extension is "G"
  // rather than "L" because "JL" is "Jump if Less".)
  def JG : InstRIL<0xC04, (outs), (ins brtarget32:$I2), "jg\t$I2", []>;
}

// Conditional branches.  It's easier for LLVM to handle these branches
// in their raw BRC/BRCL form, with the 4-bit condition-code mask being
// the first operand.  It seems friendlier to use mnemonic forms like
// JE and JLH when writing out the assembly though.
//
// Using a custom inserter for BRC gives us a chance to convert the BRC
// and a preceding compare into a single compare-and-branch instruction.
// The inserter makes no change in cases where a separate branch really
// is needed.
multiclass CondBranches<Operand ccmask, string short, string long> {
  let isBranch = 1, isTerminator = 1, Uses = [CC] in {
    def "" : InstRI<0xA74, (outs), (ins ccmask:$R1, brtarget16:$I2), short, []>;
    def L  : InstRIL<0xC04, (outs), (ins ccmask:$R1, brtarget32:$I2), long, []>;
  }
}
let isCodeGenOnly = 1, usesCustomInserter = 1 in
  defm BRC : CondBranches<cond4, "j$R1\t$I2", "jg$R1\t$I2">;
defm AsmBRC : CondBranches<uimm8zx4, "brc\t$R1, $I2", "brcl\t$R1, $I2">;

def : Pat<(z_br_ccmask cond4:$cond, bb:$dst), (BRC cond4:$cond, bb:$dst)>;

// Fused compare-and-branch instructions.  As for normal branches,
// we handle these instructions internally in their raw CRJ-like form,
// but use assembly macros like CRJE when writing them out.
//
// These instructions do not use or clobber the condition codes.
// We nevertheless pretend that they clobber CC, so that we can lower
// them to separate comparisons and BRCLs if the branch ends up being
// out of range.
multiclass CompareBranches<Operand ccmask, string pos1, string pos2> {
  let isBranch = 1, isTerminator = 1, Defs = [CC] in {
    def RJ  : InstRIEb<0xEC76, (outs), (ins GR32:$R1, GR32:$R2, ccmask:$M3,
                                            brtarget16:$RI4),
                       "crj"##pos1##"\t$R1, $R2, "##pos2##"$RI4", []>;
    def GRJ : InstRIEb<0xEC64, (outs), (ins GR64:$R1, GR64:$R2, ccmask:$M3,
                                            brtarget16:$RI4),
                       "cgrj"##pos1##"\t$R1, $R2, "##pos2##"$RI4", []>;
    def IJ  : InstRIEc<0xEC7E, (outs), (ins GR32:$R1, imm32sx8:$I2, ccmask:$M3,
                                            brtarget16:$RI4),
                       "cij"##pos1##"\t$R1, $I2, "##pos2##"$RI4", []>;
    def GIJ : InstRIEc<0xEC7C, (outs), (ins GR64:$R1, imm64sx8:$I2, ccmask:$M3,
                                            brtarget16:$RI4),
                       "cgij"##pos1##"\t$R1, $I2, "##pos2##"$RI4", []>;
  }
}
let isCodeGenOnly = 1 in
  defm C : CompareBranches<cond4, "$M3", "">;
defm AsmC : CompareBranches<uimm8zx4, "", "$M3, ">;

// Define AsmParser mnemonics for each general condition-code mask
// (integer or floating-point)
multiclass CondExtendedMnemonic<bits<4> ccmask, string name> {
  let R1 = ccmask in {
    def "" : InstRI<0xA74, (outs), (ins brtarget16:$I2),
                    "j"##name##"\t$I2", []>;
    def L  : InstRIL<0xC04, (outs), (ins brtarget32:$I2),
                     "jg"##name##"\t$I2", []>;
  }
}
defm AsmJO   : CondExtendedMnemonic<1,  "o">;
defm AsmJH   : CondExtendedMnemonic<2,  "h">;
defm AsmJNLE : CondExtendedMnemonic<3,  "nle">;
defm AsmJL   : CondExtendedMnemonic<4,  "l">;
defm AsmJNHE : CondExtendedMnemonic<5,  "nhe">;
defm AsmJLH  : CondExtendedMnemonic<6,  "lh">;
defm AsmJNE  : CondExtendedMnemonic<7,  "ne">;
defm AsmJE   : CondExtendedMnemonic<8,  "e">;
defm AsmJNLH : CondExtendedMnemonic<9,  "nlh">;
defm AsmJHE  : CondExtendedMnemonic<10, "he">;
defm AsmJNL  : CondExtendedMnemonic<11, "nl">;
defm AsmJLE  : CondExtendedMnemonic<12, "le">;
defm AsmJNH  : CondExtendedMnemonic<13, "nh">;
defm AsmJNO  : CondExtendedMnemonic<14, "no">;

// Define AsmParser mnemonics for each integer condition-code mask.
// This is like the list above, except that condition 3 is not possible
// and that the low bit of the mask is therefore always 0.  This means
// that each condition has two names.  Conditions "o" and "no" are not used.
//
// We don't make one of the two names an alias of the other because
// we need the custom parsing routines to select the correct register class.
multiclass IntCondExtendedMnemonicA<bits<4> ccmask, string name> {
  let M3 = ccmask in {
    def CR  : InstRIEb<0xEC76, (outs), (ins GR32:$R1, GR32:$R2,
                                            brtarget16:$RI4),
                       "crj"##name##"\t$R1, $R2, $RI4", []>;
    def CGR : InstRIEb<0xEC64, (outs), (ins GR64:$R1, GR64:$R2,
                                            brtarget16:$RI4),
                       "cgrj"##name##"\t$R1, $R2, $RI4", []>;
    def CI  : InstRIEc<0xEC7E, (outs), (ins GR32:$R1, imm32sx8:$I2,
                                            brtarget16:$RI4),
                       "cij"##name##"\t$R1, $I2, $RI4", []>;
    def CGI : InstRIEc<0xEC7C, (outs), (ins GR64:$R1, imm64sx8:$I2,
                                            brtarget16:$RI4),
                       "cgij"##name##"\t$R1, $I2, $RI4", []>;
  }
}
multiclass IntCondExtendedMnemonic<bits<4> ccmask, string name1, string name2>
  : IntCondExtendedMnemonicA<ccmask, name1> {
  let isAsmParserOnly = 1 in
    defm Alt : IntCondExtendedMnemonicA<ccmask, name2>;
}
defm AsmJH   : IntCondExtendedMnemonic<2,  "h",  "nle">;
defm AsmJL   : IntCondExtendedMnemonic<4,  "l",  "nhe">;
defm AsmJLH  : IntCondExtendedMnemonic<6,  "lh", "ne">;
defm AsmJE   : IntCondExtendedMnemonic<8,  "e",  "nlh">;
defm AsmJHE  : IntCondExtendedMnemonic<10, "he", "nl">;
defm AsmJLE  : IntCondExtendedMnemonic<12, "le", "nh">;

//===----------------------------------------------------------------------===//
// Select instructions
//===----------------------------------------------------------------------===//

def Select32 : SelectWrapper<GR32>;
def Select64 : SelectWrapper<GR64>;

defm CondStore8_32  : CondStores<GR32, nonvolatile_truncstorei8,
                                 nonvolatile_anyextloadi8, bdxaddr20only>;
defm CondStore16_32 : CondStores<GR32, nonvolatile_truncstorei16,
                                 nonvolatile_anyextloadi16, bdxaddr20only>;
defm CondStore32_32 : CondStores<GR32, nonvolatile_store,
                                 nonvolatile_load, bdxaddr20only>;

defm CondStore8  : CondStores<GR64, nonvolatile_truncstorei8,
                              nonvolatile_anyextloadi8, bdxaddr20only>;
defm CondStore16 : CondStores<GR64, nonvolatile_truncstorei16,
                              nonvolatile_anyextloadi16, bdxaddr20only>;
defm CondStore32 : CondStores<GR64, nonvolatile_truncstorei32,
                              nonvolatile_anyextloadi32, bdxaddr20only>;
defm CondStore64 : CondStores<GR64, nonvolatile_store,
                              nonvolatile_load, bdxaddr20only>;

//===----------------------------------------------------------------------===//
// Call instructions
//===----------------------------------------------------------------------===//

// The definitions here are for the call-clobbered registers.
let isCall = 1, Defs = [R0D, R1D, R2D, R3D, R4D, R5D, R14D,
                        F0D, F1D, F2D, F3D, F4D, F5D, F6D, F7D],
    R1 = 14, isCodeGenOnly = 1 in {
  def BRAS  : InstRI<0xA75, (outs), (ins pcrel16call:$I2, variable_ops),
                     "bras\t%r14, $I2", []>;
  def BRASL : InstRIL<0xC05, (outs), (ins pcrel32call:$I2, variable_ops),
                      "brasl\t%r14, $I2", [(z_call pcrel32call:$I2)]>;
  def BASR  : InstRR<0x0D, (outs), (ins ADDR64:$R2, variable_ops),
                     "basr\t%r14, $R2", [(z_call ADDR64:$R2)]>;
}

// Define the general form of the call instructions for the asm parser.
// These instructions don't hard-code %r14 as the return address register.
def AsmBRAS  : InstRI<0xA75, (outs), (ins GR64:$R1, brtarget16:$I2),
                      "bras\t$R1, $I2", []>;
def AsmBRASL : InstRIL<0xC05, (outs), (ins GR64:$R1, brtarget32:$I2),
                       "brasl\t$R1, $I2", []>;
def AsmBASR  : InstRR<0x0D, (outs), (ins GR64:$R1, ADDR64:$R2),
                      "basr\t$R1, $R2", []>;

//===----------------------------------------------------------------------===//
// Move instructions
//===----------------------------------------------------------------------===//

// Register moves.
let neverHasSideEffects = 1 in {
  def LR  : UnaryRR <"lr",  0x18,   null_frag, GR32, GR32>;
  def LGR : UnaryRRE<"lgr", 0xB904, null_frag, GR64, GR64>;
}

// Immediate moves.
let neverHasSideEffects = 1, isAsCheapAsAMove = 1, isMoveImm = 1,
    isReMaterializable = 1 in {
  // 16-bit sign-extended immediates.
  def LHI  : UnaryRI<"lhi",  0xA78, bitconvert, GR32, imm32sx16>;
  def LGHI : UnaryRI<"lghi", 0xA79, bitconvert, GR64, imm64sx16>;

  // Other 16-bit immediates.
  def LLILL : UnaryRI<"llill", 0xA5F, bitconvert, GR64, imm64ll16>;
  def LLILH : UnaryRI<"llilh", 0xA5E, bitconvert, GR64, imm64lh16>;
  def LLIHL : UnaryRI<"llihl", 0xA5D, bitconvert, GR64, imm64hl16>;
  def LLIHH : UnaryRI<"llihh", 0xA5C, bitconvert, GR64, imm64hh16>;

  // 32-bit immediates.
  def LGFI  : UnaryRIL<"lgfi",  0xC01, bitconvert, GR64, imm64sx32>;
  def LLILF : UnaryRIL<"llilf", 0xC0F, bitconvert, GR64, imm64lf32>;
  def LLIHF : UnaryRIL<"llihf", 0xC0E, bitconvert, GR64, imm64hf32>;
}

// Register loads.
let canFoldAsLoad = 1, SimpleBDXLoad = 1 in {
  defm L : UnaryRXPair<"l", 0x58, 0xE358, load, GR32>;
  def LG : UnaryRXY<"lg", 0xE304, load, GR64>;

  // These instructions are split after register allocation, so we don't
  // want a custom inserter.
  let Has20BitOffset = 1, HasIndex = 1, Is128Bit = 1 in {
    def L128 : Pseudo<(outs GR128:$dst), (ins bdxaddr20only128:$src),
                      [(set GR128:$dst, (load bdxaddr20only128:$src))]>;
  }
}
let canFoldAsLoad = 1 in {
  def LRL  : UnaryRILPC<"lrl",  0xC4D, aligned_load, GR32>;
  def LGRL : UnaryRILPC<"lgrl", 0xC48, aligned_load, GR64>;
}

// Register stores.
let SimpleBDXStore = 1 in {
  let isCodeGenOnly = 1 in
    defm ST32 : StoreRXPair<"st", 0x50, 0xE350, store, GR32>;
  def STG : StoreRXY<"stg", 0xE324, store, GR64>;

  // These instructions are split after register allocation, so we don't
  // want a custom inserter.
  let Has20BitOffset = 1, HasIndex = 1, Is128Bit = 1 in {
    def ST128 : Pseudo<(outs), (ins GR128:$src, bdxaddr20only128:$dst),
                       [(store GR128:$src, bdxaddr20only128:$dst)]>;
  }
}
let isCodeGenOnly = 1 in
  def STRL32 : StoreRILPC<"strl", 0xC4F, aligned_store, GR32>;
def STGRL : StoreRILPC<"stgrl", 0xC4B, aligned_store, GR64>;

// 8-bit immediate stores to 8-bit fields.
defm MVI : StoreSIPair<"mvi", 0x92, 0xEB52, truncstorei8, imm32zx8trunc>;

// 16-bit immediate stores to 16-, 32- or 64-bit fields.
def MVHHI : StoreSIL<"mvhhi", 0xE544, truncstorei16, imm32sx16trunc>;
def MVHI  : StoreSIL<"mvhi",  0xE54C, store,         imm32sx16>;
def MVGHI : StoreSIL<"mvghi", 0xE548, store,         imm64sx16>;

// Memory-to-memory moves.
let mayLoad = 1, mayStore = 1 in
  def MVC : InstSS<0xD2, (outs), (ins bdladdr12onlylen8:$BDL1,
                                      bdaddr12only:$BD2),
                   "mvc\t$BDL1, $BD2", []>;

//===----------------------------------------------------------------------===//
// Sign extensions
//===----------------------------------------------------------------------===//

// 32-bit extensions from registers.
let neverHasSideEffects = 1 in {
  def LBR : UnaryRRE<"lbr", 0xB926, sext8,  GR32, GR32>;
  def LHR : UnaryRRE<"lhr", 0xB927, sext16, GR32, GR32>;
}

// 64-bit extensions from registers.
let neverHasSideEffects = 1 in {
  def LGBR : UnaryRRE<"lgbr", 0xB906, sext8,  GR64, GR64>;
  def LGHR : UnaryRRE<"lghr", 0xB907, sext16, GR64, GR64>;
  def LGFR : UnaryRRE<"lgfr", 0xB914, sext32, GR64, GR32>;
}

// Match 32-to-64-bit sign extensions in which the source is already
// in a 64-bit register.
def : Pat<(sext_inreg GR64:$src, i32),
          (LGFR (EXTRACT_SUBREG GR64:$src, subreg_32bit))>;

// 32-bit extensions from memory.
def  LB   : UnaryRXY<"lb", 0xE376, sextloadi8, GR32>;
defm LH   : UnaryRXPair<"lh", 0x48, 0xE378, sextloadi16, GR32>;
def  LHRL : UnaryRILPC<"lhrl", 0xC45, aligned_sextloadi16, GR32>;

// 64-bit extensions from memory.
def LGB   : UnaryRXY<"lgb", 0xE377, sextloadi8,  GR64>;
def LGH   : UnaryRXY<"lgh", 0xE315, sextloadi16, GR64>;
def LGF   : UnaryRXY<"lgf", 0xE314, sextloadi32, GR64>;
def LGHRL : UnaryRILPC<"lghrl", 0xC44, aligned_sextloadi16, GR64>;
def LGFRL : UnaryRILPC<"lgfrl", 0xC4C, aligned_sextloadi32, GR64>;

// If the sign of a load-extend operation doesn't matter, use the signed ones.
// There's not really much to choose between the sign and zero extensions,
// but LH is more compact than LLH for small offsets.
def : Pat<(i32 (extloadi8  bdxaddr20only:$src)), (LB  bdxaddr20only:$src)>;
def : Pat<(i32 (extloadi16 bdxaddr12pair:$src)), (LH  bdxaddr12pair:$src)>;
def : Pat<(i32 (extloadi16 bdxaddr20pair:$src)), (LHY bdxaddr20pair:$src)>;

def : Pat<(i64 (extloadi8  bdxaddr20only:$src)), (LGB bdxaddr20only:$src)>;
def : Pat<(i64 (extloadi16 bdxaddr20only:$src)), (LGH bdxaddr20only:$src)>;
def : Pat<(i64 (extloadi32 bdxaddr20only:$src)), (LGF bdxaddr20only:$src)>;

//===----------------------------------------------------------------------===//
// Zero extensions
//===----------------------------------------------------------------------===//

// 32-bit extensions from registers.
let neverHasSideEffects = 1 in {
  def LLCR : UnaryRRE<"llcr", 0xB994, zext8,  GR32, GR32>;
  def LLHR : UnaryRRE<"llhr", 0xB995, zext16, GR32, GR32>;
}

// 64-bit extensions from registers.
let neverHasSideEffects = 1 in {
  def LLGCR : UnaryRRE<"llgcr", 0xB984, zext8,  GR64, GR64>;
  def LLGHR : UnaryRRE<"llghr", 0xB985, zext16, GR64, GR64>;
  def LLGFR : UnaryRRE<"llgfr", 0xB916, zext32, GR64, GR32>;
}

// Match 32-to-64-bit zero extensions in which the source is already
// in a 64-bit register.
def : Pat<(and GR64:$src, 0xffffffff),
          (LLGFR (EXTRACT_SUBREG GR64:$src, subreg_32bit))>;

// 32-bit extensions from memory.
def LLC   : UnaryRXY<"llc", 0xE394, zextloadi8,  GR32>;
def LLH   : UnaryRXY<"llh", 0xE395, zextloadi16, GR32>;
def LLHRL : UnaryRILPC<"llhrl", 0xC42, aligned_zextloadi16, GR32>;

// 64-bit extensions from memory.
def LLGC   : UnaryRXY<"llgc", 0xE390, zextloadi8,  GR64>;
def LLGH   : UnaryRXY<"llgh", 0xE391, zextloadi16, GR64>;
def LLGF   : UnaryRXY<"llgf", 0xE316, zextloadi32, GR64>;
def LLGHRL : UnaryRILPC<"llghrl", 0xC46, aligned_zextloadi16, GR64>;
def LLGFRL : UnaryRILPC<"llgfrl", 0xC4E, aligned_zextloadi32, GR64>;

//===----------------------------------------------------------------------===//
// Truncations
//===----------------------------------------------------------------------===//

// Truncations of 64-bit registers to 32-bit registers.
def : Pat<(i32 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, subreg_32bit)>;

// Truncations of 32-bit registers to memory.
let isCodeGenOnly = 1 in {
  defm STC32   : StoreRXPair<"stc", 0x42, 0xE372, truncstorei8,  GR32>;
  defm STH32   : StoreRXPair<"sth", 0x40, 0xE370, truncstorei16, GR32>;
  def  STHRL32 : StoreRILPC<"sthrl", 0xC47, aligned_truncstorei16, GR32>;
}

// Truncations of 64-bit registers to memory.
defm STC   : StoreRXPair<"stc", 0x42, 0xE372, truncstorei8,  GR64>;
defm STH   : StoreRXPair<"sth", 0x40, 0xE370, truncstorei16, GR64>;
def  STHRL : StoreRILPC<"sthrl", 0xC47, aligned_truncstorei16, GR64>;
defm ST    : StoreRXPair<"st", 0x50, 0xE350, truncstorei32, GR64>;
def  STRL  : StoreRILPC<"strl", 0xC4F, aligned_truncstorei32, GR64>;

//===----------------------------------------------------------------------===//
// Multi-register moves
//===----------------------------------------------------------------------===//

// Multi-register loads.
def LMG : LoadMultipleRSY<"lmg", 0xEB04, GR64>;

// Multi-register stores.
def STMG : StoreMultipleRSY<"stmg", 0xEB24, GR64>;

//===----------------------------------------------------------------------===//
// Byte swaps
//===----------------------------------------------------------------------===//

// Byte-swapping register moves.
let neverHasSideEffects = 1 in {
  def LRVR  : UnaryRRE<"lrvr",  0xB91F, bswap, GR32, GR32>;
  def LRVGR : UnaryRRE<"lrvgr", 0xB90F, bswap, GR64, GR64>;
}

// Byte-swapping loads.  Unlike normal loads, these instructions are
// allowed to access storage more than once.
def LRV  : UnaryRXY<"lrv",  0xE31E, loadu<bswap, nonvolatile_load>, GR32>;
def LRVG : UnaryRXY<"lrvg", 0xE30F, loadu<bswap, nonvolatile_load>, GR64>;

// Likewise byte-swapping stores.
def STRV  : StoreRXY<"strv",  0xE33E, storeu<bswap, nonvolatile_store>, GR32>;
def STRVG : StoreRXY<"strvg", 0xE32F, storeu<bswap, nonvolatile_store>, GR64>;

//===----------------------------------------------------------------------===//
// Load address instructions
//===----------------------------------------------------------------------===//

// Load BDX-style addresses.
let neverHasSideEffects = 1, isAsCheapAsAMove = 1, isReMaterializable = 1,
    DispKey = "la" in {
  let DispSize = "12" in
    def LA : InstRX<0x41, (outs GR64:$R1), (ins laaddr12pair:$XBD2),
                    "la\t$R1, $XBD2",
                    [(set GR64:$R1, laaddr12pair:$XBD2)]>;
  let DispSize = "20" in
    def LAY : InstRXY<0xE371, (outs GR64:$R1), (ins laaddr20pair:$XBD2),
                      "lay\t$R1, $XBD2",
                      [(set GR64:$R1, laaddr20pair:$XBD2)]>;
}

// Load a PC-relative address.  There's no version of this instruction
// with a 16-bit offset, so there's no relaxation.
let neverHasSideEffects = 1, isAsCheapAsAMove = 1, isMoveImm = 1,
    isReMaterializable = 1 in {
  def LARL : InstRIL<0xC00, (outs GR64:$R1), (ins pcrel32:$I2),
                     "larl\t$R1, $I2",
                     [(set GR64:$R1, pcrel32:$I2)]>;
}

//===----------------------------------------------------------------------===//
// Negation
//===----------------------------------------------------------------------===//

let Defs = [CC] in {
  def LCR   : UnaryRR <"lcr",   0x13,   ineg,      GR32, GR32>;
  def LCGR  : UnaryRRE<"lcgr",  0xB903, ineg,      GR64, GR64>;
  def LCGFR : UnaryRRE<"lcgfr", 0xB913, null_frag, GR64, GR32>;
}
defm : SXU<ineg, LCGFR>;

//===----------------------------------------------------------------------===//
// Insertion
//===----------------------------------------------------------------------===//

let isCodeGenOnly = 1 in
  defm IC32 : BinaryRXPair<"ic", 0x43, 0xE373, inserti8, GR32, zextloadi8>;
defm IC : BinaryRXPair<"ic", 0x43, 0xE373, inserti8, GR64, zextloadi8>;

defm : InsertMem<"inserti8", IC32,  GR32, zextloadi8, bdxaddr12pair>;
defm : InsertMem<"inserti8", IC32Y, GR32, zextloadi8, bdxaddr20pair>;

defm : InsertMem<"inserti8", IC,  GR64, zextloadi8, bdxaddr12pair>;
defm : InsertMem<"inserti8", ICY, GR64, zextloadi8, bdxaddr20pair>;

// Insertions of a 16-bit immediate, leaving other bits unaffected.
// We don't have or_as_insert equivalents of these operations because
// OI is available instead.
let isCodeGenOnly = 1 in {
  def IILL32 : BinaryRI<"iill", 0xA53, insertll, GR32, imm32ll16>;
  def IILH32 : BinaryRI<"iilh", 0xA52, insertlh, GR32, imm32lh16>;
}
def IILL : BinaryRI<"iill", 0xA53, insertll, GR64, imm64ll16>;
def IILH : BinaryRI<"iilh", 0xA52, insertlh, GR64, imm64lh16>;
def IIHL : BinaryRI<"iihl", 0xA51, inserthl, GR64, imm64hl16>;
def IIHH : BinaryRI<"iihh", 0xA50, inserthh, GR64, imm64hh16>;

// ...likewise for 32-bit immediates.  For GR32s this is a general
// full-width move.  (We use IILF rather than something like LLILF
// for 32-bit moves because IILF leaves the upper 32 bits of the
// GR64 unchanged.)
let isCodeGenOnly = 1, isAsCheapAsAMove = 1, isMoveImm = 1,
    isReMaterializable = 1 in {
  def IILF32 : UnaryRIL<"iilf", 0xC09, bitconvert, GR32, uimm32>;
}
def IILF : BinaryRIL<"iilf", 0xC09, insertlf, GR64, imm64lf32>;
def IIHF : BinaryRIL<"iihf", 0xC08, inserthf, GR64, imm64hf32>;

// An alternative model of inserthf, with the first operand being
// a zero-extended value.
def : Pat<(or (zext32 GR32:$src), imm64hf32:$imm),
          (IIHF (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR32:$src, subreg_32bit),
                imm64hf32:$imm)>;

//===----------------------------------------------------------------------===//
// Addition
//===----------------------------------------------------------------------===//

// Plain addition.
let Defs = [CC] in {
  // Addition of a register.
  let isCommutable = 1 in {
    def AR  : BinaryRR <"ar",  0x1A,   add, GR32, GR32>;
    def AGR : BinaryRRE<"agr", 0xB908, add, GR64, GR64>;
  }
  def AGFR : BinaryRRE<"agfr", 0xB918, null_frag, GR64, GR32>;

  // Addition of signed 16-bit immediates.
  def AHI  : BinaryRI<"ahi",  0xA7A, add, GR32, imm32sx16>;
  def AGHI : BinaryRI<"aghi", 0xA7B, add, GR64, imm64sx16>;

  // Addition of signed 32-bit immediates.
  def AFI  : BinaryRIL<"afi",  0xC29, add, GR32, simm32>;
  def AGFI : BinaryRIL<"agfi", 0xC28, add, GR64, imm64sx32>;

  // Addition of memory.
  defm AH  : BinaryRXPair<"ah", 0x4A, 0xE37A, add, GR32, sextloadi16>;
  defm A   : BinaryRXPair<"a",  0x5A, 0xE35A, add, GR32, load>;
  def  AGF : BinaryRXY<"agf", 0xE318, add, GR64, sextloadi32>;
  def  AG  : BinaryRXY<"ag",  0xE308, add, GR64, load>;

  // Addition to memory.
  def ASI  : BinarySIY<"asi",  0xEB6A, add, imm32sx8>;
  def AGSI : BinarySIY<"agsi", 0xEB7A, add, imm64sx8>;
}
defm : SXB<add, GR64, AGFR>;

// Addition producing a carry.
let Defs = [CC] in {
  // Addition of a register.
  let isCommutable = 1 in {
    def ALR  : BinaryRR <"alr",  0x1E,   addc, GR32, GR32>;
    def ALGR : BinaryRRE<"algr", 0xB90A, addc, GR64, GR64>;
  }
  def ALGFR : BinaryRRE<"algfr", 0xB91A, null_frag, GR64, GR32>;

  // Addition of unsigned 32-bit immediates.
  def ALFI  : BinaryRIL<"alfi",  0xC2B, addc, GR32, uimm32>;
  def ALGFI : BinaryRIL<"algfi", 0xC2A, addc, GR64, imm64zx32>;

  // Addition of memory.
  defm AL   : BinaryRXPair<"al", 0x5E, 0xE35E, addc, GR32, load>;
  def  ALGF : BinaryRXY<"algf", 0xE31A, addc, GR64, zextloadi32>;
  def  ALG  : BinaryRXY<"alg",  0xE30A, addc, GR64, load>;
}
defm : ZXB<addc, GR64, ALGFR>;

// Addition producing and using a carry.
let Defs = [CC], Uses = [CC] in {
  // Addition of a register.
  def ALCR  : BinaryRRE<"alcr",  0xB998, adde, GR32, GR32>;
  def ALCGR : BinaryRRE<"alcgr", 0xB988, adde, GR64, GR64>;

  // Addition of memory.
  def ALC  : BinaryRXY<"alc",  0xE398, adde, GR32, load>;
  def ALCG : BinaryRXY<"alcg", 0xE388, adde, GR64, load>;
}

//===----------------------------------------------------------------------===//
// Subtraction
//===----------------------------------------------------------------------===//

// Plain substraction.  Although immediate forms exist, we use the
// add-immediate instruction instead.
let Defs = [CC] in {
  // Subtraction of a register.
  def SR   : BinaryRR <"sr",   0x1B,   sub,       GR32, GR32>;
  def SGFR : BinaryRRE<"sgfr", 0xB919, null_frag, GR64, GR32>;
  def SGR  : BinaryRRE<"sgr",  0xB909, sub,       GR64, GR64>;

  // Subtraction of memory.
  defm SH  : BinaryRXPair<"sh", 0x4B, 0xE37B, sub, GR32, sextloadi16>;
  defm S   : BinaryRXPair<"s", 0x5B, 0xE35B, sub, GR32, load>;
  def  SGF : BinaryRXY<"sgf", 0xE319, sub, GR64, sextloadi32>;
  def  SG  : BinaryRXY<"sg",  0xE309, sub, GR64, load>;
}
defm : SXB<sub, GR64, SGFR>;

// Subtraction producing a carry.
let Defs = [CC] in {
  // Subtraction of a register.
  def SLR   : BinaryRR <"slr",   0x1F,   subc,      GR32, GR32>;
  def SLGFR : BinaryRRE<"slgfr", 0xB91B, null_frag, GR64, GR32>;
  def SLGR  : BinaryRRE<"slgr",  0xB90B, subc,      GR64, GR64>;

  // Subtraction of unsigned 32-bit immediates.  These don't match
  // subc because we prefer addc for constants.
  def SLFI  : BinaryRIL<"slfi",  0xC25, null_frag, GR32, uimm32>;
  def SLGFI : BinaryRIL<"slgfi", 0xC24, null_frag, GR64, imm64zx32>;

  // Subtraction of memory.
  defm SL   : BinaryRXPair<"sl", 0x5F, 0xE35F, subc, GR32, load>;
  def  SLGF : BinaryRXY<"slgf", 0xE31B, subc, GR64, zextloadi32>;
  def  SLG  : BinaryRXY<"slg",  0xE30B, subc, GR64, load>;
}
defm : ZXB<subc, GR64, SLGFR>;

// Subtraction producing and using a carry.
let Defs = [CC], Uses = [CC] in {
  // Subtraction of a register.
  def SLBR  : BinaryRRE<"slbr",  0xB999, sube, GR32, GR32>;
  def SLGBR : BinaryRRE<"slbgr", 0xB989, sube, GR64, GR64>;

  // Subtraction of memory.
  def SLB  : BinaryRXY<"slb",  0xE399, sube, GR32, load>;
  def SLBG : BinaryRXY<"slbg", 0xE389, sube, GR64, load>;
}

//===----------------------------------------------------------------------===//
// AND
//===----------------------------------------------------------------------===//

let Defs = [CC] in {
  // ANDs of a register.
  let isCommutable = 1 in {
    def NR  : BinaryRR <"nr",  0x14,   and, GR32, GR32>;
    def NGR : BinaryRRE<"ngr", 0xB980, and, GR64, GR64>;
  }

  // ANDs of a 16-bit immediate, leaving other bits unaffected.
  let isCodeGenOnly = 1 in {
    def NILL32 : BinaryRI<"nill", 0xA57, and, GR32, imm32ll16c>;
    def NILH32 : BinaryRI<"nilh", 0xA56, and, GR32, imm32lh16c>;
  }
  def NILL : BinaryRI<"nill", 0xA57, and, GR64, imm64ll16c>;
  def NILH : BinaryRI<"nilh", 0xA56, and, GR64, imm64lh16c>;
  def NIHL : BinaryRI<"nihl", 0xA55, and, GR64, imm64hl16c>;
  def NIHH : BinaryRI<"nihh", 0xA54, and, GR64, imm64hh16c>;

  // ANDs of a 32-bit immediate, leaving other bits unaffected.
  let isCodeGenOnly = 1 in
    def NILF32 : BinaryRIL<"nilf", 0xC0B, and, GR32, uimm32>;
  def NILF : BinaryRIL<"nilf", 0xC0B, and, GR64, imm64lf32c>;
  def NIHF : BinaryRIL<"nihf", 0xC0A, and, GR64, imm64hf32c>;

  // ANDs of memory.
  defm N  : BinaryRXPair<"n", 0x54, 0xE354, and, GR32, load>;
  def  NG : BinaryRXY<"ng", 0xE380, and, GR64, load>;

  // AND to memory
  defm NI : BinarySIPair<"ni", 0x94, 0xEB54, null_frag, uimm8>;
}
defm : RMWIByte<and, bdaddr12pair, NI>;
defm : RMWIByte<and, bdaddr20pair, NIY>;

//===----------------------------------------------------------------------===//
// OR
//===----------------------------------------------------------------------===//

let Defs = [CC] in {
  // ORs of a register.
  let isCommutable = 1 in {
    def OR  : BinaryRR <"or",  0x16,   or, GR32, GR32>;
    def OGR : BinaryRRE<"ogr", 0xB981, or, GR64, GR64>;
  }

  // ORs of a 16-bit immediate, leaving other bits unaffected.
  let isCodeGenOnly = 1 in {
    def OILL32 : BinaryRI<"oill", 0xA5B, or, GR32, imm32ll16>;
    def OILH32 : BinaryRI<"oilh", 0xA5A, or, GR32, imm32lh16>;
  }
  def OILL : BinaryRI<"oill", 0xA5B, or, GR64, imm64ll16>;
  def OILH : BinaryRI<"oilh", 0xA5A, or, GR64, imm64lh16>;
  def OIHL : BinaryRI<"oihl", 0xA59, or, GR64, imm64hl16>;
  def OIHH : BinaryRI<"oihh", 0xA58, or, GR64, imm64hh16>;

  // ORs of a 32-bit immediate, leaving other bits unaffected.
  let isCodeGenOnly = 1 in
    def OILF32 : BinaryRIL<"oilf", 0xC0D, or, GR32, uimm32>;
  def OILF : BinaryRIL<"oilf", 0xC0D, or, GR64, imm64lf32>;
  def OIHF : BinaryRIL<"oihf", 0xC0C, or, GR64, imm64hf32>;

  // ORs of memory.
  defm O  : BinaryRXPair<"o", 0x56, 0xE356, or, GR32, load>;
  def  OG : BinaryRXY<"og", 0xE381, or, GR64, load>;

  // OR to memory
  defm OI : BinarySIPair<"oi", 0x96, 0xEB56, null_frag, uimm8>;
}
defm : RMWIByte<or, bdaddr12pair, OI>;
defm : RMWIByte<or, bdaddr20pair, OIY>;

//===----------------------------------------------------------------------===//
// XOR
//===----------------------------------------------------------------------===//

let Defs = [CC] in {
  // XORs of a register.
  let isCommutable = 1 in {
    def XR  : BinaryRR <"xr",  0x17,   xor, GR32, GR32>;
    def XGR : BinaryRRE<"xgr", 0xB982, xor, GR64, GR64>;
  }

  // XORs of a 32-bit immediate, leaving other bits unaffected.
  let isCodeGenOnly = 1 in
    def XILF32 : BinaryRIL<"xilf", 0xC07, xor, GR32, uimm32>;
  def XILF : BinaryRIL<"xilf", 0xC07, xor, GR64, imm64lf32>;
  def XIHF : BinaryRIL<"xihf", 0xC06, xor, GR64, imm64hf32>;

  // XORs of memory.
  defm X  : BinaryRXPair<"x",0x57, 0xE357, xor, GR32, load>;
  def  XG : BinaryRXY<"xg", 0xE382, xor, GR64, load>;

  // XOR to memory
  defm XI : BinarySIPair<"xi", 0x97, 0xEB57, null_frag, uimm8>;
}
defm : RMWIByte<xor, bdaddr12pair, XI>;
defm : RMWIByte<xor, bdaddr20pair, XIY>;

//===----------------------------------------------------------------------===//
// Multiplication
//===----------------------------------------------------------------------===//

// Multiplication of a register.
let isCommutable = 1 in {
  def MSR  : BinaryRRE<"msr",  0xB252, mul, GR32, GR32>;
  def MSGR : BinaryRRE<"msgr", 0xB90C, mul, GR64, GR64>;
}
def MSGFR : BinaryRRE<"msgfr", 0xB91C, null_frag, GR64, GR32>;
defm : SXB<mul, GR64, MSGFR>;

// Multiplication of a signed 16-bit immediate.
def MHI  : BinaryRI<"mhi",  0xA7C, mul, GR32, imm32sx16>;
def MGHI : BinaryRI<"mghi", 0xA7D, mul, GR64, imm64sx16>;

// Multiplication of a signed 32-bit immediate.
def MSFI  : BinaryRIL<"msfi",  0xC21, mul, GR32, simm32>;
def MSGFI : BinaryRIL<"msgfi", 0xC20, mul, GR64, imm64sx32>;

// Multiplication of memory.
defm MH   : BinaryRXPair<"mh", 0x4C, 0xE37C, mul, GR32, sextloadi16>;
defm MS   : BinaryRXPair<"ms", 0x71, 0xE351, mul, GR32, load>;
def  MSGF : BinaryRXY<"msgf", 0xE31C, mul, GR64, sextloadi32>;
def  MSG  : BinaryRXY<"msg",  0xE30C, mul, GR64, load>;

// Multiplication of a register, producing two results.
def MLGR : BinaryRRE<"mlgr", 0xB986, z_umul_lohi64, GR128, GR64>;

// Multiplication of memory, producing two results.
def MLG : BinaryRXY<"mlg", 0xE386, z_umul_lohi64, GR128, load>;

//===----------------------------------------------------------------------===//
// Division and remainder
//===----------------------------------------------------------------------===//

// Division and remainder, from registers.
def DSGFR : BinaryRRE<"dsgfr", 0xB91D, z_sdivrem32, GR128, GR32>;
def DSGR  : BinaryRRE<"dsgr",  0xB90D, z_sdivrem64, GR128, GR64>;
def DLR   : BinaryRRE<"dlr",   0xB997, z_udivrem32, GR128, GR32>;
def DLGR  : BinaryRRE<"dlgr",  0xB987, z_udivrem64, GR128, GR64>;

// Division and remainder, from memory.
def DSGF : BinaryRXY<"dsgf", 0xE31D, z_sdivrem32, GR128, load>;
def DSG  : BinaryRXY<"dsg",  0xE30D, z_sdivrem64, GR128, load>;
def DL   : BinaryRXY<"dl",   0xE397, z_udivrem32, GR128, load>;
def DLG  : BinaryRXY<"dlg",  0xE387, z_udivrem64, GR128, load>;

//===----------------------------------------------------------------------===//
// Shifts
//===----------------------------------------------------------------------===//

// Shift left.
let neverHasSideEffects = 1 in {
  def SLL  : ShiftRS <"sll",  0x89,   shl, GR32, shift12only>;
  def SLLG : ShiftRSY<"sllg", 0xEB0D, shl, GR64, shift20only>;
}

// Logical shift right.
let neverHasSideEffects = 1 in {
  def SRL  : ShiftRS <"srl",  0x88,   srl, GR32, shift12only>;
  def SRLG : ShiftRSY<"srlg", 0xEB0C, srl, GR64, shift20only>;
}

// Arithmetic shift right.
let Defs = [CC] in {
  def SRA  : ShiftRS <"sra",  0x8A,   sra, GR32, shift12only>;
  def SRAG : ShiftRSY<"srag", 0xEB0A, sra, GR64, shift20only>;
}

// Rotate left.
let neverHasSideEffects = 1 in {
  def RLL  : ShiftRSY<"rll",  0xEB1D, rotl, GR32, shift20only>;
  def RLLG : ShiftRSY<"rllg", 0xEB1C, rotl, GR64, shift20only>;
}

// Rotate second operand left and inserted selected bits into first operand.
// These can act like 32-bit operands provided that the constant start and
// end bits (operands 2 and 3) are in the range [32, 64)
let Defs = [CC] in {
  let isCodeGenOnly = 1 in
    def RISBG32 : RotateSelectRIEf<"risbg",  0xEC55, GR32, GR32>;
  def RISBG : RotateSelectRIEf<"risbg",  0xEC55, GR64, GR64>;
}

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

// Signed comparisons.
let Defs = [CC] in {
  // Comparison with a register.
  def CR   : CompareRR <"cr",   0x19,   z_cmp,     GR32, GR32>;
  def CGFR : CompareRRE<"cgfr", 0xB930, null_frag, GR64, GR32>;
  def CGR  : CompareRRE<"cgr",  0xB920, z_cmp,     GR64, GR64>;

  // Comparison with a signed 16-bit immediate.
  def CHI  : CompareRI<"chi",  0xA7E, z_cmp, GR32, imm32sx16>;
  def CGHI : CompareRI<"cghi", 0xA7F, z_cmp, GR64, imm64sx16>;

  // Comparison with a signed 32-bit immediate.
  def CFI  : CompareRIL<"cfi",  0xC2D, z_cmp, GR32, simm32>;
  def CGFI : CompareRIL<"cgfi", 0xC2C, z_cmp, GR64, imm64sx32>;

  // Comparison with memory.
  defm CH    : CompareRXPair<"ch", 0x49, 0xE379, z_cmp, GR32, sextloadi16>;
  defm C     : CompareRXPair<"c",  0x59, 0xE359, z_cmp, GR32, load>;
  def  CGH   : CompareRXY<"cgh", 0xE334, z_cmp, GR64, sextloadi16>;
  def  CGF   : CompareRXY<"cgf", 0xE330, z_cmp, GR64, sextloadi32>;
  def  CG    : CompareRXY<"cg",  0xE320, z_cmp, GR64, load>;
  def  CHRL  : CompareRILPC<"chrl",  0xC65, z_cmp, GR32, aligned_sextloadi16>;
  def  CRL   : CompareRILPC<"crl",   0xC6D, z_cmp, GR32, aligned_load>;
  def  CGHRL : CompareRILPC<"cghrl", 0xC64, z_cmp, GR64, aligned_sextloadi16>;
  def  CGFRL : CompareRILPC<"cgfrl", 0xC6C, z_cmp, GR64, aligned_sextloadi32>;
  def  CGRL  : CompareRILPC<"cgrl",  0xC68, z_cmp, GR64, aligned_load>;

  // Comparison between memory and a signed 16-bit immediate.
  def CHHSI : CompareSIL<"chhsi", 0xE554, z_cmp, sextloadi16, imm32sx16>;
  def CHSI  : CompareSIL<"chsi",  0xE55C, z_cmp, load,        imm32sx16>;
  def CGHSI : CompareSIL<"cghsi", 0xE558, z_cmp, load,        imm64sx16>;
}
defm : SXB<z_cmp, GR64, CGFR>;

// Unsigned comparisons.
let Defs = [CC] in {
  // Comparison with a register.
  def CLR   : CompareRR <"clr",   0x15,   z_ucmp,    GR32, GR32>;
  def CLGFR : CompareRRE<"clgfr", 0xB931, null_frag, GR64, GR32>;
  def CLGR  : CompareRRE<"clgr",  0xB921, z_ucmp,    GR64, GR64>;

  // Comparison with a signed 32-bit immediate.
  def CLFI  : CompareRIL<"clfi",  0xC2F, z_ucmp, GR32, uimm32>;
  def CLGFI : CompareRIL<"clgfi", 0xC2E, z_ucmp, GR64, imm64zx32>;

  // Comparison with memory.
  defm CL     : CompareRXPair<"cl", 0x55, 0xE355, z_ucmp, GR32, load>;
  def  CLGF   : CompareRXY<"clgf", 0xE331, z_ucmp, GR64, zextloadi32>;
  def  CLG    : CompareRXY<"clg",  0xE321, z_ucmp, GR64, load>;
  def  CLHRL  : CompareRILPC<"clhrl",  0xC67, z_ucmp, GR32,
                             aligned_zextloadi16>;
  def  CLRL   : CompareRILPC<"clrl",   0xC6F, z_ucmp, GR32,
                             aligned_load>;
  def  CLGHRL : CompareRILPC<"clghrl", 0xC66, z_ucmp, GR64,
                             aligned_zextloadi16>;
  def  CLGFRL : CompareRILPC<"clgfrl", 0xC6E, z_ucmp, GR64,
                             aligned_zextloadi32>;
  def  CLGRL  : CompareRILPC<"clgrl",  0xC6A, z_ucmp, GR64,
                             aligned_load>;

  // Comparison between memory and an unsigned 8-bit immediate.
  defm CLI : CompareSIPair<"cli", 0x95, 0xEB55, z_ucmp, zextloadi8, imm32zx8>;

  // Comparison between memory and an unsigned 16-bit immediate.
  def CLHHSI : CompareSIL<"clhhsi", 0xE555, z_ucmp, zextloadi16, imm32zx16>;
  def CLFHSI : CompareSIL<"clfhsi", 0xE55D, z_ucmp, load,        imm32zx16>;
  def CLGHSI : CompareSIL<"clghsi", 0xE559, z_ucmp, load,        imm64zx16>;
}
defm : ZXB<z_ucmp, GR64, CLGFR>;

//===----------------------------------------------------------------------===//
// Atomic operations
//===----------------------------------------------------------------------===//

def ATOMIC_SWAPW        : AtomicLoadWBinaryReg<z_atomic_swapw>;
def ATOMIC_SWAP_32      : AtomicLoadBinaryReg32<atomic_swap_32>;
def ATOMIC_SWAP_64      : AtomicLoadBinaryReg64<atomic_swap_64>;

def ATOMIC_LOADW_AR     : AtomicLoadWBinaryReg<z_atomic_loadw_add>;
def ATOMIC_LOADW_AFI    : AtomicLoadWBinaryImm<z_atomic_loadw_add, simm32>;
def ATOMIC_LOAD_AR      : AtomicLoadBinaryReg32<atomic_load_add_32>;
def ATOMIC_LOAD_AHI     : AtomicLoadBinaryImm32<atomic_load_add_32, imm32sx16>;
def ATOMIC_LOAD_AFI     : AtomicLoadBinaryImm32<atomic_load_add_32, simm32>;
def ATOMIC_LOAD_AGR     : AtomicLoadBinaryReg64<atomic_load_add_64>;
def ATOMIC_LOAD_AGHI    : AtomicLoadBinaryImm64<atomic_load_add_64, imm64sx16>;
def ATOMIC_LOAD_AGFI    : AtomicLoadBinaryImm64<atomic_load_add_64, imm64sx32>;

def ATOMIC_LOADW_SR     : AtomicLoadWBinaryReg<z_atomic_loadw_sub>;
def ATOMIC_LOAD_SR      : AtomicLoadBinaryReg32<atomic_load_sub_32>;
def ATOMIC_LOAD_SGR     : AtomicLoadBinaryReg64<atomic_load_sub_64>;

def ATOMIC_LOADW_NR     : AtomicLoadWBinaryReg<z_atomic_loadw_and>;
def ATOMIC_LOADW_NILH   : AtomicLoadWBinaryImm<z_atomic_loadw_and, imm32lh16c>;
def ATOMIC_LOAD_NR      : AtomicLoadBinaryReg32<atomic_load_and_32>;
def ATOMIC_LOAD_NILL32  : AtomicLoadBinaryImm32<atomic_load_and_32, imm32ll16c>;
def ATOMIC_LOAD_NILH32  : AtomicLoadBinaryImm32<atomic_load_and_32, imm32lh16c>;
def ATOMIC_LOAD_NILF32  : AtomicLoadBinaryImm32<atomic_load_and_32, uimm32>;
def ATOMIC_LOAD_NGR     : AtomicLoadBinaryReg64<atomic_load_and_64>;
def ATOMIC_LOAD_NILL    : AtomicLoadBinaryImm64<atomic_load_and_64, imm64ll16c>;
def ATOMIC_LOAD_NILH    : AtomicLoadBinaryImm64<atomic_load_and_64, imm64lh16c>;
def ATOMIC_LOAD_NIHL    : AtomicLoadBinaryImm64<atomic_load_and_64, imm64hl16c>;
def ATOMIC_LOAD_NIHH    : AtomicLoadBinaryImm64<atomic_load_and_64, imm64hh16c>;
def ATOMIC_LOAD_NILF    : AtomicLoadBinaryImm64<atomic_load_and_64, imm64lf32c>;
def ATOMIC_LOAD_NIHF    : AtomicLoadBinaryImm64<atomic_load_and_64, imm64hf32c>;

def ATOMIC_LOADW_OR     : AtomicLoadWBinaryReg<z_atomic_loadw_or>;
def ATOMIC_LOADW_OILH   : AtomicLoadWBinaryImm<z_atomic_loadw_or, imm32lh16>;
def ATOMIC_LOAD_OR      : AtomicLoadBinaryReg32<atomic_load_or_32>;
def ATOMIC_LOAD_OILL32  : AtomicLoadBinaryImm32<atomic_load_or_32, imm32ll16>;
def ATOMIC_LOAD_OILH32  : AtomicLoadBinaryImm32<atomic_load_or_32, imm32lh16>;
def ATOMIC_LOAD_OILF32  : AtomicLoadBinaryImm32<atomic_load_or_32, uimm32>;
def ATOMIC_LOAD_OGR     : AtomicLoadBinaryReg64<atomic_load_or_64>;
def ATOMIC_LOAD_OILL    : AtomicLoadBinaryImm64<atomic_load_or_64, imm64ll16>;
def ATOMIC_LOAD_OILH    : AtomicLoadBinaryImm64<atomic_load_or_64, imm64lh16>;
def ATOMIC_LOAD_OIHL    : AtomicLoadBinaryImm64<atomic_load_or_64, imm64hl16>;
def ATOMIC_LOAD_OIHH    : AtomicLoadBinaryImm64<atomic_load_or_64, imm64hh16>;
def ATOMIC_LOAD_OILF    : AtomicLoadBinaryImm64<atomic_load_or_64, imm64lf32>;
def ATOMIC_LOAD_OIHF    : AtomicLoadBinaryImm64<atomic_load_or_64, imm64hf32>;

def ATOMIC_LOADW_XR     : AtomicLoadWBinaryReg<z_atomic_loadw_xor>;
def ATOMIC_LOADW_XILF   : AtomicLoadWBinaryImm<z_atomic_loadw_xor, uimm32>;
def ATOMIC_LOAD_XR      : AtomicLoadBinaryReg32<atomic_load_xor_32>;
def ATOMIC_LOAD_XILF32  : AtomicLoadBinaryImm32<atomic_load_xor_32, uimm32>;
def ATOMIC_LOAD_XGR     : AtomicLoadBinaryReg64<atomic_load_xor_64>;
def ATOMIC_LOAD_XILF    : AtomicLoadBinaryImm64<atomic_load_xor_64, imm64lf32>;
def ATOMIC_LOAD_XIHF    : AtomicLoadBinaryImm64<atomic_load_xor_64, imm64hf32>;

def ATOMIC_LOADW_NRi    : AtomicLoadWBinaryReg<z_atomic_loadw_nand>;
def ATOMIC_LOADW_NILHi  : AtomicLoadWBinaryImm<z_atomic_loadw_nand,
                                               imm32lh16c>;
def ATOMIC_LOAD_NRi     : AtomicLoadBinaryReg32<atomic_load_nand_32>;
def ATOMIC_LOAD_NILL32i : AtomicLoadBinaryImm32<atomic_load_nand_32,
                                                imm32ll16c>;
def ATOMIC_LOAD_NILH32i : AtomicLoadBinaryImm32<atomic_load_nand_32,
                                                imm32lh16c>;
def ATOMIC_LOAD_NILF32i : AtomicLoadBinaryImm32<atomic_load_nand_32, uimm32>;
def ATOMIC_LOAD_NGRi    : AtomicLoadBinaryReg64<atomic_load_nand_64>;
def ATOMIC_LOAD_NILLi   : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64ll16c>;
def ATOMIC_LOAD_NILHi   : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64lh16c>;
def ATOMIC_LOAD_NIHLi   : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64hl16c>;
def ATOMIC_LOAD_NIHHi   : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64hh16c>;
def ATOMIC_LOAD_NILFi   : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64lf32c>;
def ATOMIC_LOAD_NIHFi   : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64hf32c>;

def ATOMIC_LOADW_MIN    : AtomicLoadWBinaryReg<z_atomic_loadw_min>;
def ATOMIC_LOAD_MIN_32  : AtomicLoadBinaryReg32<atomic_load_min_32>;
def ATOMIC_LOAD_MIN_64  : AtomicLoadBinaryReg64<atomic_load_min_64>;

def ATOMIC_LOADW_MAX    : AtomicLoadWBinaryReg<z_atomic_loadw_max>;
def ATOMIC_LOAD_MAX_32  : AtomicLoadBinaryReg32<atomic_load_max_32>;
def ATOMIC_LOAD_MAX_64  : AtomicLoadBinaryReg64<atomic_load_max_64>;

def ATOMIC_LOADW_UMIN   : AtomicLoadWBinaryReg<z_atomic_loadw_umin>;
def ATOMIC_LOAD_UMIN_32 : AtomicLoadBinaryReg32<atomic_load_umin_32>;
def ATOMIC_LOAD_UMIN_64 : AtomicLoadBinaryReg64<atomic_load_umin_64>;

def ATOMIC_LOADW_UMAX   : AtomicLoadWBinaryReg<z_atomic_loadw_umax>;
def ATOMIC_LOAD_UMAX_32 : AtomicLoadBinaryReg32<atomic_load_umax_32>;
def ATOMIC_LOAD_UMAX_64 : AtomicLoadBinaryReg64<atomic_load_umax_64>;

def ATOMIC_CMP_SWAPW
  : Pseudo<(outs GR32:$dst), (ins bdaddr20only:$addr, GR32:$cmp, GR32:$swap,
                                  ADDR32:$bitshift, ADDR32:$negbitshift,
                                  uimm32:$bitsize),
           [(set GR32:$dst,
                 (z_atomic_cmp_swapw bdaddr20only:$addr, GR32:$cmp, GR32:$swap,
                                     ADDR32:$bitshift, ADDR32:$negbitshift,
                                     uimm32:$bitsize))]> {
  let Defs = [CC];
  let mayLoad = 1;
  let mayStore = 1;
  let usesCustomInserter = 1;
}

let Defs = [CC] in {
  defm CS  : CmpSwapRSPair<"cs", 0xBA, 0xEB14, atomic_cmp_swap_32, GR32>;
  def  CSG : CmpSwapRSY<"csg", 0xEB30, atomic_cmp_swap_64, GR64>;
}

//===----------------------------------------------------------------------===//
// Miscellaneous Instructions.
//===----------------------------------------------------------------------===//

// Read a 32-bit access register into a GR32.  As with all GR32 operations,
// the upper 32 bits of the enclosing GR64 remain unchanged, which is useful
// when a 64-bit address is stored in a pair of access registers.
def EAR : InstRRE<0xB24F, (outs GR32:$R1), (ins access_reg:$R2),
                  "ear\t$R1, $R2",
                  [(set GR32:$R1, (z_extract_access access_reg:$R2))]>;

// Find leftmost one, AKA count leading zeros.  The instruction actually
// returns a pair of GR64s, the first giving the number of leading zeros
// and the second giving a copy of the source with the leftmost one bit
// cleared.  We only use the first result here.
let Defs = [CC] in {
  def FLOGR : UnaryRRE<"flogr", 0xB983, null_frag, GR128, GR64>;
}
def : Pat<(ctlz GR64:$src),
          (EXTRACT_SUBREG (FLOGR GR64:$src), subreg_high)>;

// Use subregs to populate the "don't care" bits in a 32-bit to 64-bit anyext.
def : Pat<(i64 (anyext GR32:$src)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR32:$src, subreg_32bit)>;

// There are no 32-bit equivalents of LLILL and LLILH, so use a full
// 64-bit move followed by a subreg.  This preserves the invariant that
// all GR32 operations only modify the low 32 bits.
def : Pat<(i32 imm32ll16:$src),
          (EXTRACT_SUBREG (LLILL (LL16 imm:$src)), subreg_32bit)>;
def : Pat<(i32 imm32lh16:$src),
          (EXTRACT_SUBREG (LLILH (LH16 imm:$src)), subreg_32bit)>;

// Extend GR32s and GR64s to GR128s.
let usesCustomInserter = 1 in {
  def AEXT128_64 : Pseudo<(outs GR128:$dst), (ins GR64:$src), []>;
  def ZEXT128_32 : Pseudo<(outs GR128:$dst), (ins GR32:$src), []>;
  def ZEXT128_64 : Pseudo<(outs GR128:$dst), (ins GR64:$src), []>;
}

//===----------------------------------------------------------------------===//
// Peepholes.
//===----------------------------------------------------------------------===//

// Use AL* for GR64 additions of unsigned 32-bit values.
defm : ZXB<add, GR64, ALGFR>;
def  : Pat<(add GR64:$src1, imm64zx32:$src2),
           (ALGFI GR64:$src1, imm64zx32:$src2)>;
def  : Pat<(add GR64:$src1, (zextloadi32 bdxaddr20only:$addr)),
           (ALGF GR64:$src1, bdxaddr20only:$addr)>;

// Use SL* for GR64 subtractions of unsigned 32-bit values.
defm : ZXB<sub, GR64, SLGFR>;
def  : Pat<(add GR64:$src1, imm64zx32n:$src2),
           (SLGFI GR64:$src1, imm64zx32n:$src2)>;
def  : Pat<(sub GR64:$src1, (zextloadi32 bdxaddr20only:$addr)),
           (SLGF GR64:$src1, bdxaddr20only:$addr)>;
