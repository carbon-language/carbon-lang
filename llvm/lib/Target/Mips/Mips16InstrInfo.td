//===- Mips16InstrInfo.td - Target Description for Mips16  -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips16 instructions.
//
//===----------------------------------------------------------------------===//

def uimm5      : Operand<i8> {
  let DecoderMethod= "DecodeSimm16";
}

//
// RRR-type instruction format
//

class FRRR16_ins<bits<2> _f, string asmstr,  InstrItinClass itin> :
  FRRR16<_f, (outs CPU16Regs:$rz), (ins CPU16Regs:$rx, CPU16Regs:$ry),
         !strconcat(asmstr, "\t$rz, $rx, $ry"), [], itin>;

//
// I8_MOV32R instruction format (used only by MOV32R instruction)
//
class FI8_MOV32R16_ins<string asmstr, InstrItinClass itin>:
  FI8_MOV32R16<(outs CPURegs:$r32), (ins CPU16Regs:$rz),
               !strconcat(asmstr,  "\t$r32, $rz"), [], itin>;

//
// EXT-RI instruction format
//

class FEXT_RI16_ins_base<bits<5> _op, string asmstr, string asmstr2,
                         InstrItinClass itin>:
  FEXT_RI16<_op, (outs CPU16Regs:$rx), (ins simm16:$imm),
                  !strconcat(asmstr, asmstr2), [], itin>;

class FEXT_RI16_ins<bits<5> _op, string asmstr,
                    InstrItinClass itin>:
  FEXT_RI16_ins_base<_op, asmstr, "\t$rx, $imm", itin>;

class FEXT_RI16_PC_ins<bits<5> _op, string asmstr, InstrItinClass itin>:
  FEXT_RI16_ins_base<_op, asmstr, "\t$rx, $$pc, $imm", itin>;

//
// RR-type instruction format
//
let rx=0 in
class FRR16_JALRC_RA_only_ins<bits<1> nd_, bits<1> l_,
                              string asmstr, InstrItinClass itin>:
  FRR16_JALRC<nd_, l_, 1, (outs), (ins), !strconcat(asmstr, "\t $$ra"),
              [], itin> ;

//
// EXT-RRI instruction format
//

class FEXT_RRI16_mem_ins<bits<5> op, string asmstr, Operand MemOpnd,
                         InstrItinClass itin>:
  FEXT_RRI16<op, (outs CPU16Regs:$ry), (ins  MemOpnd:$addr),
             !strconcat(asmstr, "\t$ry, $addr"), [], itin>;

//
// EXT-SHIFT instruction format
//
class FEXT_SHIFT16_ins<bits<2> _f, string asmstr, InstrItinClass itin>:
  FEXT_SHIFT16<_f, (outs CPU16Regs:$rx), (ins CPU16Regs:$ry, uimm5:$sa),
               !strconcat(asmstr, "\t$rx, $ry, $sa"), [], itin>;

//
// Address operand
def mem16 : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops CPU16Regs, simm16);
  let EncoderMethod = "getMemEncoding";
}

//
// Format: ADDIU rx, pc, immediate MIPS16e
// Purpose: Add Immediate Unsigned Word (3-Operand, PC-Relative, Extended)
// To add a constant to the program counter.
//
class AddiuRxPcImmX16_base : FEXT_RI16_PC_ins<0b00001, "addiu", IIAlu>;
def AddiuRxPcImmX16   : AddiuRxPcImmX16_base;
//
// Format: ADDU rz, rx, ry MIPS16e
// Purpose: Add Unsigned Word (3-Operand)
// To add 32-bit integers.
//

class AdduRxRyRz16_base: FRRR16_ins<01, "addu", IIAlu>;
def AdduRxRyRz16: AdduRxRyRz16_base;

//
// Format: JR ra MIPS16e
// Purpose: Jump Register Through Register ra
// To execute a branch to the instruction address in the return
// address register.
//

def JrRa16: FRR16_JALRC_RA_only_ins<0, 0, "jr", IIAlu>;

//
// Format: LI rx, immediate MIPS16e
// Purpose: Load Immediate (Extended)
// To load a constant into a GPR.
//
def LiRxImmX16: FEXT_RI16_ins<0b01101, "li", IIAlu>;

//
// Format: LW ry, offset(rx) MIPS16e
// Purpose: Load Word (Extended)
// To load a word from memory as a signed value.
//
class LwRxRyOffMemX16_base: FEXT_RRI16_mem_ins<0b10011, "lw", mem16, IIAlu>;
def LwRxRyOffMemX16: LwRxRyOffMemX16_base;

//
// Format: MOVE r32, rz MIPS16e
// Purpose: Move
// To move the contents of a GPR to a GPR.
//
def Mov32R16: FI8_MOV32R16_ins<"move", IIAlu>;
//
// Format: RESTORE {ra,}{s0/s1/s0-1,}{framesize}
// (All args are optional) MIPS16e
// Purpose: Restore Registers and Deallocate Stack Frame
// To deallocate a stack frame before exit from a subroutine,
// restoring return address and static registers, and adjusting
// stack
//

// fixed form for restoring RA and the frame
// for direct object emitter, encoding needs to be adjusted for the
// frame size
//
let ra=1, s=0,s0=0,s1=0 in
def RestoreRaF16:
  FI8_SVRS16<0b1, (outs), (ins uimm16:$frame_size),
             "restore \t$$ra, $frame_size", [], IILoad >;

//
// Format: SAVE {ra,}{s0/s1/s0-1,}{framesize} (All arguments are optional)
// MIPS16e
// Purpose: Save Registers and Set Up Stack Frame
// To set up a stack frame on entry to a subroutine,
// saving return address and static registers, and adjusting stack
//
let ra=1, s=1,s0=0,s1=0 in
def SaveRaF16:
  FI8_SVRS16<0b1, (outs), (ins uimm16:$frame_size),
             "save \t$$ra, $frame_size", [], IILoad >;

//
// Format: SLL rx, ry, sa MIPS16e
// Purpose: Shift Word Left Logical (Extended)
// To execute a left-shift of a word by a fixed number of bits—0 to 31 bits.
//
def SllX16: FEXT_SHIFT16_ins<0b00, "sll", IIAlu>;

//
// Format: SW ry, offset(rx) MIPS16e
// Purpose: Store Word (Extended)
// To store a word to memory.
//
class SwRxRyOffMemX16_base: FEXT_RRI16_mem_ins<0b11011, "sw", mem16, IIAlu>;
def SwRxRyOffMemX16: SwRxRyOffMemX16_base;

class Mips16Pat<dag pattern, dag result> : Pat<pattern, result> {
  let Predicates = [InMips16Mode];
}

class ArithLogicR16Defs<SDNode OpNode, bit isComm = 0> {
  dag OutOperandList = (outs CPU16Regs:$rz);
  dag InOperandList = (ins CPU16Regs:$rx, CPU16Regs:$ry);
  list<dag> Pattern = [(set CPU16Regs:$rz,
                       (OpNode CPU16Regs:$rx, CPU16Regs:$ry))];
}

multiclass ArithLogicR16_base {
  def _add: AdduRxRyRz16_base, ArithLogicR16Defs<add, 1>;
}

defm ArithLogicR16_patt : ArithLogicR16_base;

class LoadM16Defs<PatFrag OpNode, Operand _MemOpnd, bit Pseudo=0> {
  bit isPseudo = Pseudo;
  Operand MemOpnd = _MemOpnd;
  dag OutOperandList = (outs CPU16Regs:$ry);
  dag InOperandList = (ins MemOpnd:$addr);
  list<dag> Pattern = [(set CPU16Regs:$ry, (OpNode addr:$addr))];
}

multiclass LoadM16_base {
  def _LwRxRyOffMemX16: LwRxRyOffMemX16_base, LoadM16Defs<load_a, mem16>;
}

defm LoadM16: LoadM16_base;

class StoreM16Defs<PatFrag OpNode, Operand _MemOpnd, bit Pseudo=0> {
  bit isPseudo = Pseudo;
  Operand MemOpnd = _MemOpnd;
  dag OutOperandList = (outs );
  dag InOperandList = (ins CPU16Regs:$ry, MemOpnd:$addr);
  list<dag> Pattern = [(OpNode CPU16Regs:$ry, addr:$addr)];
}

multiclass StoreM16_base {
  def _SwRxRyOffMemX16: SwRxRyOffMemX16_base, StoreM16Defs<store_a, mem16>;
}

defm StoreM16: StoreM16_base;

// Jump and Link (Call)
let isCall=1, hasDelaySlot=1 in
def JumpLinkReg16:
  FRR16_JALRC<0, 0, 0, (outs), (ins CPU16Regs:$rs),
              "jalr \t$rs", [(MipsJmpLink CPU16Regs:$rs)], IIBranch>;

// Mips16 pseudos
let isReturn=1, isTerminator=1, hasDelaySlot=1, isBarrier=1, hasCtrlDep=1,
  hasExtraSrcRegAllocReq = 1 in
def RetRA16 : MipsPseudo16<(outs), (ins), "", [(MipsRet)]>;

// As stack alignment is always done with addiu, we need a 16-bit immediate
// This is basically deprecated code but needs to be there for things
// to work.
let Defs = [SP], Uses = [SP] in {
def ADJCALLSTACKDOWN16 : MipsPseudo16<(outs), (ins uimm16:$amt),
                                      ";",
                                      [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP16   : MipsPseudo16<(outs), (ins uimm16:$amt1, uimm16:$amt2),
                                      ";",
                                      [(callseq_end timm:$amt1, timm:$amt2)]>;
}

// Small immediates
def : Mips16Pat<(i32 immZExt16:$in), (LiRxImmX16 immZExt16:$in)>;
def : Mips16Pat<(MipsLo tglobaladdr:$in), (LiRxImmX16 tglobaladdr:$in)>;
