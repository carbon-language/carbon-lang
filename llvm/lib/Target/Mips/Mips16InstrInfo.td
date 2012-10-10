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
//

//
// Address operand
def mem16 : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops CPU16Regs, simm16);
  let EncoderMethod = "getMemEncoding";
}

//
// Assembler formats in alphabetical order.
// Natural and pseudos are mixed together.
//
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

class FEXT_2RI16_ins<bits<5> _op, string asmstr,
                     InstrItinClass itin>:
  FEXT_RI16<_op, (outs CPU16Regs:$rx), (ins CPU16Regs:$rx_, simm16:$imm),
            !strconcat(asmstr, "\t$rx, $imm"), [], itin> {
  let Constraints = "$rx_ = $rx";
}

// this has an explicit sp argument that we ignore to work around a problem
// in the compiler
class FEXT_RI16_SP_explicit_ins<bits<5> _op, string asmstr,
                                InstrItinClass itin>:
  FEXT_RI16<_op, (outs CPU16Regs:$rx), (ins CPUSPReg:$ry, simm16:$imm),
            !strconcat(asmstr, "\t$rx, $imm ( $ry ); "), [], itin>;

//
// EXT-RRI instruction format
//

class FEXT_RRI16_mem_ins<bits<5> op, string asmstr, Operand MemOpnd,
                         InstrItinClass itin>:
  FEXT_RRI16<op, (outs CPU16Regs:$ry), (ins  MemOpnd:$addr),
             !strconcat(asmstr, "\t$ry, $addr"), [], itin>;

class FEXT_RRI16_mem2_ins<bits<5> op, string asmstr, Operand MemOpnd,
                          InstrItinClass itin>:
  FEXT_RRI16<op, (outs ), (ins  CPU16Regs:$ry, MemOpnd:$addr),
             !strconcat(asmstr, "\t$ry, $addr"), [], itin>;

//
// EXT-SHIFT instruction format
//
class FEXT_SHIFT16_ins<bits<2> _f, string asmstr, InstrItinClass itin>:
  FEXT_SHIFT16<_f, (outs CPU16Regs:$rx), (ins CPU16Regs:$ry, shamt:$sa),
               !strconcat(asmstr, "\t$rx, $ry, $sa"), [], itin>;


//
// I8_MOVR32 instruction format (used only by the MOVR32 instructio
//
class FI8_MOVR3216_ins<string asmstr, InstrItinClass itin>:
       FI8_MOVR3216<(outs CPU16Regs:$rz), (ins CPURegs:$r32),
       !strconcat(asmstr,  "\t$rz, $r32"), [], itin>;

//
// I8_MOV32R instruction format (used only by MOV32R instruction)
//

class FI8_MOV32R16_ins<string asmstr, InstrItinClass itin>:
  FI8_MOV32R16<(outs CPURegs:$r32), (ins CPU16Regs:$rz),
               !strconcat(asmstr,  "\t$r32, $rz"), [], itin>;

//
// This are pseudo formats for multiply
// This first one can be changed to non pseudo now.
//
// MULT
//
class FMULT16_ins<string asmstr, InstrItinClass itin> :
  MipsPseudo16<(outs), (ins CPU16Regs:$rx, CPU16Regs:$ry),
               !strconcat(asmstr, "\t$rx, $ry"), []>;

//
// MULT-LO
//
class FMULT16_LO_ins<string asmstr, InstrItinClass itin> :
  MipsPseudo16<(outs CPU16Regs:$rz), (ins CPU16Regs:$rx, CPU16Regs:$ry),
               !strconcat(asmstr, "\t$rx, $ry\n\tmflo\t$rz"), []> {
  let isCodeGenOnly=1;
}

//
// RR-type instruction format
//

class FRR16_ins<bits<5> f, string asmstr, InstrItinClass itin> :
  FRR16<f, (outs CPU16Regs:$rx), (ins CPU16Regs:$ry),
        !strconcat(asmstr, "\t$rx, $ry"), [], itin> {
}
class FRR16_M_ins<bits<5> f, string asmstr,
                  InstrItinClass itin> :
  FRR16<f, (outs CPU16Regs:$rx), (ins),
        !strconcat(asmstr, "\t$rx"), [], itin>;

class FRxRxRy16_ins<bits<5> f, string asmstr,
                    InstrItinClass itin> :
  FRR16<f, (outs CPU16Regs:$rz), (ins CPU16Regs:$rx, CPU16Regs:$ry),
            !strconcat(asmstr, "\t$rz, $ry"),
            [], itin> {
  let Constraints = "$rx = $rz";
}

let rx=0 in
class FRR16_JALRC_RA_only_ins<bits<1> nd_, bits<1> l_,
                              string asmstr, InstrItinClass itin>:
  FRR16_JALRC<nd_, l_, 1, (outs), (ins), !strconcat(asmstr, "\t $$ra"),
              [], itin> ;

//
// RRR-type instruction format
//

class FRRR16_ins<bits<2> _f, string asmstr,  InstrItinClass itin> :
  FRRR16<_f, (outs CPU16Regs:$rz), (ins CPU16Regs:$rx, CPU16Regs:$ry),
         !strconcat(asmstr, "\t$rz, $rx, $ry"), [], itin>;

//
// Some general instruction class info
//
//

class ArithLogic16Defs<bit isCom=0> {
  bits<5> shamt = 0;
  bit isCommutable = isCom;
  bit isReMaterializable = 1;
  bit neverHasSideEffects = 1;
}

class MayLoad {
  bit mayLoad = 1;
}

class MayStore {
  bit mayStore = 1;
}
//

// Format: ADDIU rx, immediate MIPS16e
// Purpose: Add Immediate Unsigned Word (2-Operand, Extended)
// To add a constant to a 32-bit integer.
//
def AddiuRxImmX16: FEXT_RI16_ins<0b01001, "addiu", IIAlu>;

def AddiuRxRxImmX16: FEXT_2RI16_ins<0b01001, "addiu", IIAlu>,
  ArithLogic16Defs<0>;

//

// Format: ADDIU rx, pc, immediate MIPS16e
// Purpose: Add Immediate Unsigned Word (3-Operand, PC-Relative, Extended)
// To add a constant to the program counter.
//
def AddiuRxPcImmX16: FEXT_RI16_PC_ins<0b00001, "addiu", IIAlu>;
//
// Format: ADDU rz, rx, ry MIPS16e
// Purpose: Add Unsigned Word (3-Operand)
// To add 32-bit integers.
//

def AdduRxRyRz16: FRRR16_ins<01, "addu", IIAlu>, ArithLogic16Defs<1>;

//
// Format: AND rx, ry MIPS16e
// Purpose: AND
// To do a bitwise logical AND.

def AndRxRxRy16: FRxRxRy16_ins<0b01100, "and", IIAlu>, ArithLogic16Defs<1>;

//
// Format: JR ra MIPS16e
// Purpose: Jump Register Through Register ra
// To execute a branch to the instruction address in the return
// address register.
//

def JrRa16: FRR16_JALRC_RA_only_ins<0, 0, "jr", IIAlu>;

//
// Format: LB ry, offset(rx) MIPS16e
// Purpose: Load Byte (Extended)
// To load a byte from memory as a signed value.
//
def LbRxRyOffMemX16: FEXT_RRI16_mem_ins<0b10011, "lb", mem16, IILoad>, MayLoad;

//
// Format: LBU ry, offset(rx) MIPS16e
// Purpose: Load Byte Unsigned (Extended)
// To load a byte from memory as a unsigned value.
//
def LbuRxRyOffMemX16:
  FEXT_RRI16_mem_ins<0b10100, "lbu", mem16, IILoad>, MayLoad;

//
// Format: LH ry, offset(rx) MIPS16e
// Purpose: Load Halfword signed (Extended)
// To load a halfword from memory as a signed value.
//
def LhRxRyOffMemX16: FEXT_RRI16_mem_ins<0b10100, "lh", mem16, IILoad>, MayLoad;

//
// Format: LHU ry, offset(rx) MIPS16e
// Purpose: Load Halfword unsigned (Extended)
// To load a halfword from memory as an unsigned value.
//
def LhuRxRyOffMemX16:
  FEXT_RRI16_mem_ins<0b10100, "lhu", mem16, IILoad>, MayLoad;

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
def LwRxRyOffMemX16: FEXT_RRI16_mem_ins<0b10011, "lw", mem16, IILoad>, MayLoad;

// Format: LW rx, offset(sp) MIPS16e
// Purpose: Load Word (SP-Relative, Extended)
// To load an SP-relative word from memory as a signed value.
//
def LwRxSpImmX16: FEXT_RI16_SP_explicit_ins<0b10110, "lw", IILoad>, MayLoad;

//
// Format: MOVE r32, rz MIPS16e
// Purpose: Move
// To move the contents of a GPR to a GPR.
//
def Move32R16: FI8_MOV32R16_ins<"move", IIAlu>;

//
// Format: MOVE ry, r32 MIPS16e
//Purpose: Move
// To move the contents of a GPR to a GPR.
//
def MoveR3216: FI8_MOVR3216_ins<"move", IIAlu>;

//
// Format: MFHI rx MIPS16e
// Purpose: Move From HI Register
// To copy the special purpose HI register to a GPR.
//
def Mfhi16: FRR16_M_ins<0b10000, "mfhi", IIAlu> {
  let Uses = [HI];
  let neverHasSideEffects = 1;
}

//
// Format: MFLO rx MIPS16e
// Purpose: Move From LO Register
// To copy the special purpose LO register to a GPR.
//
def Mflo16: FRR16_M_ins<0b10010, "mflo", IIAlu> {
  let Uses = [LO];
  let neverHasSideEffects = 1;
}

//
// Pseudo Instruction for mult
//
def MultRxRy16:  FMULT16_ins<"mult",  IIAlu> {
  let isCommutable = 1;
  let neverHasSideEffects = 1;
  let Defs = [HI, LO];
}

def MultuRxRy16: FMULT16_ins<"multu", IIAlu> {
  let isCommutable = 1;
  let neverHasSideEffects = 1;
  let Defs = [HI, LO];
}

//
// Format: MULT rx, ry MIPS16e
// Purpose: Multiply Word
// To multiply 32-bit signed integers.
//
def MultRxRyRz16: FMULT16_LO_ins<"mult", IIAlu> {
  let isCommutable = 1;
  let neverHasSideEffects = 1;
  let Defs = [HI, LO];
}

//
// Format: MULTU rx, ry MIPS16e
// Purpose: Multiply Unsigned Word
// To multiply 32-bit unsigned integers.
//
def MultuRxRyRz16: FMULT16_LO_ins<"multu", IIAlu> {
  let isCommutable = 1;
  let neverHasSideEffects = 1;
  let Defs = [HI, LO];
}

//
// Format: NEG rx, ry MIPS16e
// Purpose: Negate
// To negate an integer value.
//
def NegRxRy16: FRR16_ins<0b11101, "neg", IIAlu>;

//
// Format: NOT rx, ry MIPS16e
// Purpose: Not
// To complement an integer value
//
def NotRxRy16: FRR16_ins<0b01111, "not", IIAlu>;

//
// Format: OR rx, ry MIPS16e
// Purpose: Or
// To do a bitwise logical OR.
//
def OrRxRxRy16: FRxRxRy16_ins<0b01101, "or", IIAlu>, ArithLogic16Defs<1>;

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
let ra=1, s=0,s0=1,s1=1 in
def RestoreRaF16:
  FI8_SVRS16<0b1, (outs), (ins uimm16:$frame_size),
             "restore \t$$ra,  $$s0, $$s1, $frame_size", [], IILoad >, MayLoad {
  let isCodeGenOnly = 1;
}

//
// Format: SAVE {ra,}{s0/s1/s0-1,}{framesize} (All arguments are optional)
// MIPS16e
// Purpose: Save Registers and Set Up Stack Frame
// To set up a stack frame on entry to a subroutine,
// saving return address and static registers, and adjusting stack
//
let ra=1, s=1,s0=1,s1=1 in
def SaveRaF16:
  FI8_SVRS16<0b1, (outs), (ins uimm16:$frame_size),
             "save \t$$ra, $$s0, $$s1, $frame_size", [], IIStore >, MayStore {
  let isCodeGenOnly = 1;
}
//
// Format: SB ry, offset(rx) MIPS16e
// Purpose: Store Byte (Extended)
// To store a byte to memory.
//
def SbRxRyOffMemX16:
  FEXT_RRI16_mem2_ins<0b11000, "sb", mem16, IIStore>, MayStore;

//
// Format: SH ry, offset(rx) MIPS16e
// Purpose: Store Halfword (Extended)
// To store a halfword to memory.
//
def ShRxRyOffMemX16:
  FEXT_RRI16_mem2_ins<0b11001, "sh", mem16, IIStore>, MayStore;

//
// Format: SLL rx, ry, sa MIPS16e
// Purpose: Shift Word Left Logical (Extended)
// To execute a left-shift of a word by a fixed number of bits—0 to 31 bits.
//
def SllX16: FEXT_SHIFT16_ins<0b00, "sll", IIAlu>;

//
// Format: SLLV ry, rx MIPS16e
// Purpose: Shift Word Left Logical Variable
// To execute a left-shift of a word by a variable number of bits.
//
def SllvRxRy16 : FRxRxRy16_ins<0b00100, "sllv", IIAlu>;


//
// Format: SRAV ry, rx MIPS16e
// Purpose: Shift Word Right Arithmetic Variable
// To execute an arithmetic right-shift of a word by a variable
// number of bits.
//
def SravRxRy16: FRxRxRy16_ins<0b00111, "srav", IIAlu>;


//
// Format: SRA rx, ry, sa MIPS16e
// Purpose: Shift Word Right Arithmetic (Extended)
// To execute an arithmetic right-shift of a word by a fixed
// number of bits—1 to 8 bits.
//
def SraX16: FEXT_SHIFT16_ins<0b11, "sra", IIAlu>;


//
// Format: SRLV ry, rx MIPS16e
// Purpose: Shift Word Right Logical Variable
// To execute a logical right-shift of a word by a variable
// number of bits.
//
def SrlvRxRy16: FRxRxRy16_ins<0b00110, "srlv", IIAlu>;


//
// Format: SRL rx, ry, sa MIPS16e
// Purpose: Shift Word Right Logical (Extended)
// To execute a logical right-shift of a word by a fixed
// number of bits—1 to 31 bits.
//
def SrlX16: FEXT_SHIFT16_ins<0b10, "srl", IIAlu>;

//
// Format: SUBU rz, rx, ry MIPS16e
// Purpose: Subtract Unsigned Word
// To subtract 32-bit integers
//
def SubuRxRyRz16: FRRR16_ins<0b11, "subu", IIAlu>, ArithLogic16Defs<0>;

//
// Format: SW ry, offset(rx) MIPS16e
// Purpose: Store Word (Extended)
// To store a word to memory.
//
def SwRxRyOffMemX16:
  FEXT_RRI16_mem2_ins<0b11011, "sw", mem16, IIStore>, MayStore;

//
// Format: SW rx, offset(sp) MIPS16e
// Purpose: Store Word rx (SP-Relative)
// To store an SP-relative word to memory.
//
def SwRxSpImmX16: FEXT_RI16_SP_explicit_ins<0b11010, "sw", IIStore>, MayStore;

//
//
// Format: XOR rx, ry MIPS16e
// Purpose: Xor
// To do a bitwise logical XOR.
//
def XorRxRxRy16: FRxRxRy16_ins<0b01110, "xor", IIAlu>, ArithLogic16Defs<1>;

class Mips16Pat<dag pattern, dag result> : Pat<pattern, result> {
  let Predicates = [InMips16Mode];
}

// Unary Arith/Logic
//
class ArithLogicU_pat<PatFrag OpNode, Instruction I> :
  Mips16Pat<(OpNode CPU16Regs:$r),
            (I CPU16Regs:$r)>;

def: ArithLogicU_pat<not, NotRxRy16>;
def: ArithLogicU_pat<ineg, NegRxRy16>;

class ArithLogic16_pat<SDNode OpNode, Instruction I> :
  Mips16Pat<(OpNode CPU16Regs:$l, CPU16Regs:$r),
            (I CPU16Regs:$l, CPU16Regs:$r)>;

def: ArithLogic16_pat<add, AdduRxRyRz16>;
def: ArithLogic16_pat<and, AndRxRxRy16>;
def: ArithLogic16_pat<mul, MultRxRyRz16>;
def: ArithLogic16_pat<or, OrRxRxRy16>;
def: ArithLogic16_pat<sub, SubuRxRyRz16>;
def: ArithLogic16_pat<xor, XorRxRxRy16>;

// Arithmetic and logical instructions with 2 register operands.

class ArithLogicI16_pat<SDNode OpNode, PatFrag imm_type, Instruction I> :
  Mips16Pat<(OpNode CPU16Regs:$in, imm_type:$imm),
            (I CPU16Regs:$in, imm_type:$imm)>;

def: ArithLogicI16_pat<add, immSExt16, AddiuRxRxImmX16>;
def: ArithLogicI16_pat<shl, immZExt5, SllX16>;
def: ArithLogicI16_pat<srl, immZExt5, SrlX16>;
def: ArithLogicI16_pat<sra, immZExt5, SraX16>;

class shift_rotate_reg16_pat<SDNode OpNode, Instruction I> :
  Mips16Pat<(OpNode CPU16Regs:$r, CPU16Regs:$ra),
            (I CPU16Regs:$r, CPU16Regs:$ra)>;

def: shift_rotate_reg16_pat<shl, SllvRxRy16>;
def: shift_rotate_reg16_pat<sra, SravRxRy16>;
def: shift_rotate_reg16_pat<srl, SrlvRxRy16>;

class LoadM16_pat<PatFrag OpNode, Instruction I> :
  Mips16Pat<(OpNode addr:$addr), (I addr:$addr)>;

def: LoadM16_pat<sextloadi8, LbRxRyOffMemX16>;
def: LoadM16_pat<zextloadi8, LbuRxRyOffMemX16>;
def: LoadM16_pat<sextloadi16, LhRxRyOffMemX16>;
def: LoadM16_pat<zextloadi16, LhuRxRyOffMemX16>;
def: LoadM16_pat<load, LwRxRyOffMemX16>;

class StoreM16_pat<PatFrag OpNode, Instruction I> :
  Mips16Pat<(OpNode CPU16Regs:$r, addr:$addr), (I CPU16Regs:$r, addr:$addr)>;

def: StoreM16_pat<truncstorei8, SbRxRyOffMemX16>;
def: StoreM16_pat<truncstorei16, ShRxRyOffMemX16>;
def: StoreM16_pat<store, SwRxRyOffMemX16>;


// Jump and Link (Call)
let isCall=1, hasDelaySlot=1 in
def JumpLinkReg16:
  FRR16_JALRC<0, 0, 0, (outs), (ins CPU16Regs:$rs),
              "jalr \t$rs", [(MipsJmpLink CPU16Regs:$rs)], IIBranch>;

// Mips16 pseudos
let isReturn=1, isTerminator=1, hasDelaySlot=1, isBarrier=1, hasCtrlDep=1,
  hasExtraSrcRegAllocReq = 1 in
def RetRA16 : MipsPseudo16<(outs), (ins), "", [(MipsRet)]>;

// Small immediates
def: Mips16Pat<(i32 immZExt16:$in), (LiRxImmX16 immZExt16:$in)>;

def: Mips16Pat<(add CPU16Regs:$hi, (MipsLo tglobaladdr:$lo)),
               (AddiuRxRxImmX16 CPU16Regs:$hi, tglobaladdr:$lo)>;
