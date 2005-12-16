//===- SparcV8Instrs.td - Target Description for SparcV8 Target -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the SparcV8 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

class InstV8 : Instruction {          // SparcV8 instruction baseline
  field bits<32> Inst;

  let Namespace = "V8";

  bits<2> op;
  let Inst{31-30} = op;               // Top two bits are the 'op' field

  // Bit attributes specific to SparcV8 instructions
  bit isPasi       = 0; // Does this instruction affect an alternate addr space?
  bit isPrivileged = 0; // Is this a privileged instruction?
}

include "SparcV8InstrFormats.td"

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// Pseudo instructions.
class PseudoInstV8<string nm, dag ops> : InstV8  {
  let Name = nm;
  dag OperandList = ops;
}
def PHI : PseudoInstV8<"PHI", (ops variable_ops)>;
def ADJCALLSTACKDOWN : PseudoInstV8<"ADJCALLSTACKDOWN", (ops variable_ops)>;
def ADJCALLSTACKUP : PseudoInstV8<"ADJCALLSTACKUP", (ops variable_ops)>;
def IMPLICIT_USE : PseudoInstV8<"IMPLICIT_USE", (ops variable_ops)>;
def IMPLICIT_DEF : PseudoInstV8<"IMPLICIT_DEF", (ops variable_ops)>;
def FpMOVD : PseudoInstV8<"FpMOVD", (ops)>; // pseudo 64-bit double move

// Section A.3 - Synthetic Instructions, p. 85
// special cases of JMPL:
let isReturn = 1, isTerminator = 1, hasDelaySlot = 1 in {
  let rd = I7.Num, rs1 = G0.Num, simm13 = 8 in
    def RET : F3_2<2, 0b111000,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "ret">;
  let rd = O7.Num, rs1 = G0.Num, simm13 = 8 in
    def RETL: F3_2<2, 0b111000,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "retl">;
}
// CMP is a special case of SUBCC where destination is ignored, by setting it to
// %g0 (hardwired zero).
// FIXME: should keep track of the fact that it defs the integer condition codes
let rd = 0 in
  def CMPri: F3_2<2, 0b010100,
                  (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "cmp">;

// Section B.1 - Load Integer Instructions, p. 90
def LDSB: F3_2<3, 0b001001,
               (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "ldsb">;
def LDSH: F3_2<3, 0b001010,
               (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "ldsh">;
def LDUB: F3_2<3, 0b000001,
               (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "ldub">;
def LDUH: F3_2<3, 0b000010,
               (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "lduh">;
def LD  : F3_2<3, 0b000000,
               (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "ld">;
def LDD : F3_2<3, 0b000011,
               (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "ldd">;

// Section B.2 - Load Floating-point Instructions, p. 92
def LDFrr  : F3_1<3, 0b100000,
                  (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "ld">;
def LDFri  : F3_2<3, 0b100000,
                  (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "ld">;
def LDDFrr : F3_1<3, 0b100011,
                  (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "ldd">;
def LDDFri : F3_2<3, 0b100011,
                  (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "ldd">;
def LDFSRrr: F3_1<3, 0b100001,
                  (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "ld">;
def LDFSRri: F3_2<3, 0b100001,
                  (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "ld">;

// Section B.4 - Store Integer Instructions, p. 95
def STB : F3_2<3, 0b000101,
               (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "stb">;
def STH : F3_2<3, 0b000110,
               (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "sth">;
def ST  : F3_2<3, 0b000100,
               (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "st">;
def STD : F3_2<3, 0b000111,
               (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "std">;

// Section B.5 - Store Floating-point Instructions, p. 97
def STFrr   : F3_1<3, 0b100100,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "st">;
def STFri   : F3_2<3, 0b100100,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "st">;
def STDFrr  : F3_1<3, 0b100111,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "std">;
def STDFri  : F3_2<3, 0b100111,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "std">;
def STFSRrr : F3_1<3, 0b100101,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "st">;
def STFSRri : F3_2<3, 0b100101,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "st">;
def STDFQrr : F3_1<3, 0b100110,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "std">;
def STDFQri : F3_2<3, 0b100110,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "std">;

// Section B.9 - SETHI Instruction, p. 104
def SETHIi: F2_1<0b100, "sethi">;

// Section B.10 - NOP Instruction, p. 105
// (It's a special case of SETHI)
let rd = 0, imm22 = 0 in
  def NOP : F2_1<0b100, "nop">;

// Section B.11 - Logical Instructions, p. 106
def ANDrr   : F3_1<2, 0b000001,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "and">;
def ANDri   : F3_2<2, 0b000001,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "and">;
def ANDCCrr : F3_1<2, 0b010001,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "andcc">;
def ANDCCri : F3_2<2, 0b010001,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "andcc">;
def ANDNrr  : F3_1<2, 0b000101,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "andn">;
def ANDNri  : F3_2<2, 0b000101,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "andn">;
def ANDNCCrr: F3_1<2, 0b010101,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "andncc">;
def ANDNCCri: F3_2<2, 0b010101,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "andncc">;
def ORrr    : F3_1<2, 0b000010,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "or">;
def ORri    : F3_2<2, 0b000010,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "or">;
def ORCCrr  : F3_1<2, 0b010010,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "orcc">;
def ORCCri  : F3_2<2, 0b010010,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "orcc">;
def ORNrr   : F3_1<2, 0b000110,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "orn">;
def ORNri   : F3_2<2, 0b000110,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "orn">;
def ORNCCrr : F3_1<2, 0b010110,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "orncc">;
def ORNCCri : F3_2<2, 0b010110,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "orncc">;
def XORrr   : F3_1<2, 0b000011,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "xor">;
def XORri   : F3_2<2, 0b000011,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "xor">;
def XORCCrr : F3_1<2, 0b010011,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "xorcc">;
def XORCCri : F3_2<2, 0b010011,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "xorcc">;
def XNORrr  : F3_1<2, 0b000111,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "xnor">;
def XNORri  : F3_2<2, 0b000111,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "xnor">;
def XNORCCrr: F3_1<2, 0b010111,
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "xnorcc">;
def XNORCCri: F3_2<2, 0b010111,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "xnorcc">;

// Section B.12 - Shift Instructions, p. 107
def SLLrr : F3_1<2, 0b100101,
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "sll">;
def SLLri : F3_2<2, 0b100101,
                 (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "sll">;
def SRLrr : F3_1<2, 0b100110, 
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "srl">;
def SRLri : F3_2<2, 0b100110,
                 (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "srl">;
def SRArr : F3_1<2, 0b100111, 
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "sra">;
def SRAri : F3_2<2, 0b100111,
                 (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "sra">;

// Section B.13 - Add Instructions, p. 108
def ADDrr   : F3_1<2, 0b000000, 
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "add">;
def ADDri   : F3_2<2, 0b000000,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "add">;
def ADDCCrr : F3_1<2, 0b010000, 
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "addcc">;
def ADDCCri : F3_2<2, 0b010000,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "addcc">;
def ADDXrr  : F3_1<2, 0b001000, 
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "addx">;
def ADDXri  : F3_2<2, 0b001000,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "addx">;
def ADDXCCrr: F3_1<2, 0b011000, 
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "addxcc">;
def ADDXCCri: F3_2<2, 0b011000,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "addxcc">;

// Section B.15 - Subtract Instructions, p. 110
def SUBrr   : F3_1<2, 0b000100, 
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "sub">;
def SUBri   : F3_2<2, 0b000100,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "sub">;
def SUBCCrr : F3_1<2, 0b010100, 
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "subcc">;
def SUBCCri : F3_2<2, 0b010100,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "subcc">;
def SUBXrr  : F3_1<2, 0b001100, 
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "subx">;
def SUBXri  : F3_2<2, 0b001100,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "subx">;
def SUBXCCrr: F3_1<2, 0b011100, 
                 (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "subxcc">;
def SUBXCCri: F3_2<2, 0b011100,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "subxcc">;

// Section B.18 - Multiply Instructions, p. 113
def UMULrr  : F3_1<2, 0b001010, 
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "umul">;
def UMULri  : F3_2<2, 0b001010,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "umul">;
def SMULrr  : F3_1<2, 0b001011, 
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "smul">;
def SMULri  : F3_2<2, 0b001011,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "smul">;
def UMULCCrr: F3_1<2, 0b011010, 
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "umulcc">;
def UMULCCri: F3_2<2, 0b011010,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "umulcc">;
def SMULCCrr: F3_1<2, 0b011011, 
                   (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "smulcc">;
def SMULCCri: F3_2<2, 0b011011,
                   (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "smulcc">;

// Section B.19 - Divide Instructions, p. 115
def UDIVrr   : F3_1<2, 0b001110, 
                    (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "udiv">;
def UDIVri   : F3_2<2, 0b001110,
                    (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "udiv">;
def SDIVrr   : F3_1<2, 0b001111,
                    (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "sdiv">;
def SDIVri   : F3_2<2, 0b001111,
                    (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "sdiv">;
def UDIVCCrr : F3_1<2, 0b011110,
                    (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "udivcc">;
def UDIVCCri : F3_2<2, 0b011110,
                    (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "udivcc">;
def SDIVCCrr : F3_1<2, 0b011111,
                    (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "sdivcc">;
def SDIVCCri : F3_2<2, 0b011111,
                    (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "sdivcc">;

// Section B.20 - SAVE and RESTORE, p. 117
def SAVErr    : F3_1<2, 0b111100,
                     (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "save">;
def SAVEri    : F3_2<2, 0b111100,
                     (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "save">;
def RESTORErr : F3_1<2, 0b111101,
                     (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "restore">;
def RESTOREri : F3_2<2, 0b111101,
                     (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "restore">;

// Section B.21 - Branch on Integer Condition Codes Instructions, p. 119

// conditional branch class:
class BranchV8<bits<4> cc, string nm> : F2_2<cc, 0b010, nm> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
}

let isBarrier = 1 in
  def BA   : BranchV8<0b1000, "ba">;
def BN   : BranchV8<0b0000, "bn">;
def BNE  : BranchV8<0b1001, "bne">;
def BE   : BranchV8<0b0001, "be">;
def BG   : BranchV8<0b1010, "bg">;
def BLE  : BranchV8<0b0010, "ble">;
def BGE  : BranchV8<0b1011, "bge">;
def BL   : BranchV8<0b0011, "bl">;
def BGU  : BranchV8<0b1100, "bgu">;
def BLEU : BranchV8<0b0100, "bleu">;
def BCC  : BranchV8<0b1101, "bcc">;
def BCS  : BranchV8<0b0101, "bcs">;

// Section B.22 - Branch on Floating-point Condition Codes Instructions, p. 121

// floating-point conditional branch class:
class FPBranchV8<bits<4> cc, string nm> : F2_2<cc, 0b110, nm> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
}

def FBA  : FPBranchV8<0b1000, "fba">;
def FBN  : FPBranchV8<0b0000, "fbn">;
def FBU  : FPBranchV8<0b0111, "fbu">;
def FBG  : FPBranchV8<0b0110, "fbg">;
def FBUG : FPBranchV8<0b0101, "fbug">;
def FBL  : FPBranchV8<0b0100, "fbl">;
def FBUL : FPBranchV8<0b0011, "fbul">;
def FBLG : FPBranchV8<0b0010, "fblg">;
def FBNE : FPBranchV8<0b0001, "fbne">;
def FBE  : FPBranchV8<0b1001, "fbe">;
def FBUE : FPBranchV8<0b1010, "fbue">;
def FBGE : FPBranchV8<0b1011, "fbge">;
def FBUGE: FPBranchV8<0b1100, "fbuge">;
def FBLE : FPBranchV8<0b1101, "fble">;
def FBULE: FPBranchV8<0b1110, "fbule">;
def FBO  : FPBranchV8<0b1111, "fbo">;



// Section B.24 - Call and Link Instruction, p. 125
// This is the only Format 1 instruction
let Uses = [O0, O1, O2, O3, O4, O5], hasDelaySlot = 1, isCall = 1 in { 
  // pc-relative call:
  let Defs = [O0, O1, O2, O3, O4, O5, O7, G1, G2, G3, G4, G5, G6, G7,
    D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15] in
  def CALL : InstV8 {
    bits<30> disp;
    let op = 1;
    let Inst{29-0} = disp;
    let Name = "call";
  }

  // indirect call (O7 is an EXPLICIT def in indirect calls, so it cannot also
  // be an implicit def):
  let Defs = [O0, O1, O2, O3, O4, O5, G1, G2, G3, G4, G5, G6, G7,
    D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15] in
  def JMPLrr : F3_1<2, 0b111000, (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "jmpl">;
                // jmpl [rs1+rs2], rd
}

// Section B.29 - Write State Register Instructions
def WRrr : F3_1<2, 0b110000,
                (ops IntRegs:$a, IntRegs:$b, IntRegs:$c), "wr">;
def WRri : F3_2<2, 0b110000,
                (ops IntRegs:$a, IntRegs:$b, i32imm:$c), "wr">;                    // wr rs1, imm, rd

// Convert Integer to Floating-point Instructions, p. 141
def FITOS : F3_3<2, 0b110100, 0b011000100, "fitos">;
def FITOD : F3_3<2, 0b110100, 0b011001000, "fitod">;

// Convert Floating-point to Integer Instructions, p. 142
def FSTOI : F3_3<2, 0b110100, 0b011010001, "fstoi">;
def FDTOI : F3_3<2, 0b110100, 0b011010010, "fdtoi">;

// Convert between Floating-point Formats Instructions, p. 143
def FSTOD : F3_3<2, 0b110100, 0b011001001, "fstod">;
def FDTOS : F3_3<2, 0b110100, 0b011000110, "fdtos">;

// Floating-point Move Instructions, p. 144
def FMOVS : F3_3<2, 0b110100, 0b000000001, "fmovs">;
def FNEGS : F3_3<2, 0b110100, 0b000000101, "fnegs">;
def FABSS : F3_3<2, 0b110100, 0b000001001, "fabss">;

// Floating-point Add and Subtract Instructions, p. 146
def FADDS  : F3_3<2, 0b110100, 0b001000001, "fadds">;
def FADDD  : F3_3<2, 0b110100, 0b001000010, "faddd">;
def FSUBS  : F3_3<2, 0b110100, 0b001000101, "fsubs">;
def FSUBD  : F3_3<2, 0b110100, 0b001000110, "fsubd">;

// Floating-point Multiply and Divide Instructions, p. 147
def FMULS  : F3_3<2, 0b110100, 0b001001001, "fmuls">;
def FMULD  : F3_3<2, 0b110100, 0b001001010, "fmuld">;
def FSMULD : F3_3<2, 0b110100, 0b001101001, "fsmuld">;
def FDIVS  : F3_3<2, 0b110100, 0b001001101, "fdivs">;
def FDIVD  : F3_3<2, 0b110100, 0b001001110, "fdivd">;

// Floating-point Compare Instructions, p. 148
// Note: the 2nd template arg is different for these guys.
// Note 2: the result of a FCMP is not available until the 2nd cycle
// after the instr is retired, but there is no interlock. This behavior
// is modelled as a delay slot.
let hasDelaySlot = 1 in {
  def FCMPS  : F3_3<2, 0b110101, 0b001010001, "fcmps">;
  def FCMPD  : F3_3<2, 0b110101, 0b001010010, "fcmpd">;
  def FCMPES : F3_3<2, 0b110101, 0b001010101, "fcmpes">;
  def FCMPED : F3_3<2, 0b110101, 0b001010110, "fcmped">;
}

