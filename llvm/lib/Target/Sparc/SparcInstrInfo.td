//===- SparcInstrInfo.td - Target Description for Sparc Target ------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the Sparc instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

include "SparcInstrFormats.td"

//===----------------------------------------------------------------------===//
// Feature predicates.
//===----------------------------------------------------------------------===//

// HasV9 - This predicate is true when the target processor supports V9
// instructions.  Note that the machine may be running in 32-bit mode.
def HasV9   : Predicate<"Subtarget.isV9()">;

// HasNoV9 - This predicate is true when the target doesn't have V9
// instructions.  Use of this is just a hack for the isel not having proper
// costs for V8 instructions that are more expensive than their V9 ones.
def HasNoV9 : Predicate<"!Subtarget.isV9()">;

// HasVIS - This is true when the target processor has VIS extensions.
def HasVIS : Predicate<"Subtarget.isVIS()">;

// UseDeprecatedInsts - This predicate is true when the target processor is a
// V8, or when it is V9 but the V8 deprecated instructions are efficient enough
// to use when appropriate.  In either of these cases, the instruction selector
// will pick deprecated instructions.
def UseDeprecatedInsts : Predicate<"Subtarget.useDeprecatedV8Instructions()">;

//===----------------------------------------------------------------------===//
// Instruction Pattern Stuff
//===----------------------------------------------------------------------===//

def simm11  : PatLeaf<(imm), [{
  // simm11 predicate - True if the imm fits in a 11-bit sign extended field.
  return (((int)N->getValue() << (32-11)) >> (32-11)) == (int)N->getValue();
}]>;

def simm13  : PatLeaf<(imm), [{
  // simm13 predicate - True if the imm fits in a 13-bit sign extended field.
  return (((int)N->getValue() << (32-13)) >> (32-13)) == (int)N->getValue();
}]>;

def LO10 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((unsigned)N->getValue() & 1023, MVT::i32);
}]>;

def HI22 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return CurDAG->getTargetConstant((unsigned)N->getValue() >> 10, MVT::i32);
}]>;

def SETHIimm : PatLeaf<(imm), [{
  return (((unsigned)N->getValue() >> 10) << 10) == (unsigned)N->getValue();
}], HI22>;

// Addressing modes.
def ADDRrr : ComplexPattern<i32, 2, "SelectADDRrr", []>;
def ADDRri : ComplexPattern<i32, 2, "SelectADDRri", [frameindex]>;

// Address operands
def MEMrr : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let NumMIOperands = 2;
  let MIOperandInfo = (ops IntRegs, IntRegs);
}
def MEMri : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let NumMIOperands = 2;
  let MIOperandInfo = (ops IntRegs, i32imm);
}

// Branch targets have OtherVT type.
def brtarget : Operand<OtherVT>;
def calltarget : Operand<i32>;

// Operand for printing out a condition code.
let PrintMethod = "printCCOperand" in
  def CCOp : Operand<i32>;

def SDTSPcmpfcc : 
SDTypeProfile<0, 2, [SDTCisFP<0>, SDTCisSameAs<0, 1>]>;
def SDTSPbrcc : 
SDTypeProfile<0, 2, [SDTCisVT<0, OtherVT>, SDTCisVT<1, i32>]>;
def SDTSPselectcc :
SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, SDTCisVT<3, i32>]>;
def SDTSPFTOI :
SDTypeProfile<1, 1, [SDTCisVT<0, f32>, SDTCisFP<1>]>;
def SDTSPITOF :
SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVT<1, f32>]>;

def SPcmpicc : SDNode<"SPISD::CMPICC", SDTIntBinOp, [SDNPOutFlag]>;
def SPcmpfcc : SDNode<"SPISD::CMPFCC", SDTSPcmpfcc, [SDNPOutFlag]>;
def SPbricc : SDNode<"SPISD::BRICC", SDTSPbrcc, [SDNPHasChain, SDNPInFlag]>;
def SPbrfcc : SDNode<"SPISD::BRFCC", SDTSPbrcc, [SDNPHasChain, SDNPInFlag]>;

def SPhi    : SDNode<"SPISD::Hi", SDTIntUnaryOp>;
def SPlo    : SDNode<"SPISD::Lo", SDTIntUnaryOp>;

def SPftoi  : SDNode<"SPISD::FTOI", SDTSPFTOI>;
def SPitof  : SDNode<"SPISD::ITOF", SDTSPITOF>;

def SPselecticc : SDNode<"SPISD::SELECT_ICC", SDTSPselectcc, [SDNPInFlag]>;
def SPselectfcc : SDNode<"SPISD::SELECT_FCC", SDTSPselectcc, [SDNPInFlag]>;

// These are target-independent nodes, but have target-specific formats.
def SDT_SPCallSeq : SDTypeProfile<0, 1, [ SDTCisVT<0, i32> ]>;
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_SPCallSeq,
                           [SDNPHasChain, SDNPOutFlag]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_SPCallSeq,
                           [SDNPHasChain, SDNPOutFlag]>;

def SDT_SPCall    : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def call          : SDNode<"SPISD::CALL", SDT_SPCall,
	                   [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def SDT_SPRetFlag : SDTypeProfile<0, 0, []>;
def retflag       : SDNode<"SPISD::RET_FLAG", SDT_SPRetFlag,
	                   [SDNPHasChain, SDNPOptInFlag]>;

//===----------------------------------------------------------------------===//
// SPARC Flag Conditions
//===----------------------------------------------------------------------===//

// Note that these values must be kept in sync with the CCOp::CondCode enum
// values.
class ICC_VAL<int N> : PatLeaf<(i32 N)>;
def ICC_NE  : ICC_VAL< 9>;  // Not Equal
def ICC_E   : ICC_VAL< 1>;  // Equal
def ICC_G   : ICC_VAL<10>;  // Greater
def ICC_LE  : ICC_VAL< 2>;  // Less or Equal
def ICC_GE  : ICC_VAL<11>;  // Greater or Equal
def ICC_L   : ICC_VAL< 3>;  // Less
def ICC_GU  : ICC_VAL<12>;  // Greater Unsigned
def ICC_LEU : ICC_VAL< 4>;  // Less or Equal Unsigned
def ICC_CC  : ICC_VAL<13>;  // Carry Clear/Great or Equal Unsigned
def ICC_CS  : ICC_VAL< 5>;  // Carry Set/Less Unsigned
def ICC_POS : ICC_VAL<14>;  // Positive
def ICC_NEG : ICC_VAL< 6>;  // Negative
def ICC_VC  : ICC_VAL<15>;  // Overflow Clear
def ICC_VS  : ICC_VAL< 7>;  // Overflow Set

class FCC_VAL<int N> : PatLeaf<(i32 N)>;
def FCC_U   : FCC_VAL<23>;  // Unordered
def FCC_G   : FCC_VAL<22>;  // Greater
def FCC_UG  : FCC_VAL<21>;  // Unordered or Greater
def FCC_L   : FCC_VAL<20>;  // Less
def FCC_UL  : FCC_VAL<19>;  // Unordered or Less
def FCC_LG  : FCC_VAL<18>;  // Less or Greater
def FCC_NE  : FCC_VAL<17>;  // Not Equal
def FCC_E   : FCC_VAL<25>;  // Equal
def FCC_UE  : FCC_VAL<24>;  // Unordered or Equal
def FCC_GE  : FCC_VAL<25>;  // Greater or Equal
def FCC_UGE : FCC_VAL<26>;  // Unordered or Greater or Equal
def FCC_LE  : FCC_VAL<27>;  // Less or Equal
def FCC_ULE : FCC_VAL<28>;  // Unordered or Less or Equal
def FCC_O   : FCC_VAL<29>;  // Ordered


//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// Pseudo instructions.
class Pseudo<dag ops, string asmstr, list<dag> pattern>
   : InstSP<ops, asmstr, pattern>;

def ADJCALLSTACKDOWN : Pseudo<(ops i32imm:$amt),
                               "!ADJCALLSTACKDOWN $amt",
                               [(callseq_start imm:$amt)]>;
def ADJCALLSTACKUP : Pseudo<(ops i32imm:$amt),
                            "!ADJCALLSTACKUP $amt",
                            [(callseq_end imm:$amt)]>;
def IMPLICIT_DEF_Int : Pseudo<(ops IntRegs:$dst),
                              "!IMPLICIT_DEF $dst",
                              [(set IntRegs:$dst, (undef))]>;
def IMPLICIT_DEF_FP  : Pseudo<(ops FPRegs:$dst), "!IMPLICIT_DEF $dst",
                              [(set FPRegs:$dst, (undef))]>;
def IMPLICIT_DEF_DFP : Pseudo<(ops DFPRegs:$dst), "!IMPLICIT_DEF $dst",
                              [(set DFPRegs:$dst, (undef))]>;
                              
// FpMOVD/FpNEGD/FpABSD - These are lowered to single-precision ops by the 
// fpmover pass.
let Predicates = [HasNoV9] in {  // Only emit these in V8 mode.
  def FpMOVD : Pseudo<(ops DFPRegs:$dst, DFPRegs:$src),
                      "!FpMOVD $src, $dst", []>;
  def FpNEGD : Pseudo<(ops DFPRegs:$dst, DFPRegs:$src),
                      "!FpNEGD $src, $dst",
                      [(set DFPRegs:$dst, (fneg DFPRegs:$src))]>;
  def FpABSD : Pseudo<(ops DFPRegs:$dst, DFPRegs:$src),
                      "!FpABSD $src, $dst",
                      [(set DFPRegs:$dst, (fabs DFPRegs:$src))]>;
}

// SELECT_CC_* - Used to implement the SELECT_CC DAG operation.  Expanded by the
// scheduler into a branch sequence.  This has to handle all permutations of
// selection between i32/f32/f64 on ICC and FCC.
let usesCustomDAGSchedInserter = 1 in {   // Expanded by the scheduler.
  def SELECT_CC_Int_ICC
   : Pseudo<(ops IntRegs:$dst, IntRegs:$T, IntRegs:$F, i32imm:$Cond),
            "; SELECT_CC_Int_ICC PSEUDO!",
            [(set IntRegs:$dst, (SPselecticc IntRegs:$T, IntRegs:$F,
                                             imm:$Cond))]>;
  def SELECT_CC_Int_FCC
   : Pseudo<(ops IntRegs:$dst, IntRegs:$T, IntRegs:$F, i32imm:$Cond),
            "; SELECT_CC_Int_FCC PSEUDO!",
            [(set IntRegs:$dst, (SPselectfcc IntRegs:$T, IntRegs:$F,
                                             imm:$Cond))]>;
  def SELECT_CC_FP_ICC
   : Pseudo<(ops FPRegs:$dst, FPRegs:$T, FPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_FP_ICC PSEUDO!",
            [(set FPRegs:$dst, (SPselecticc FPRegs:$T, FPRegs:$F,
                                            imm:$Cond))]>;
  def SELECT_CC_FP_FCC
   : Pseudo<(ops FPRegs:$dst, FPRegs:$T, FPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_FP_FCC PSEUDO!",
            [(set FPRegs:$dst, (SPselectfcc FPRegs:$T, FPRegs:$F,
                                            imm:$Cond))]>;
  def SELECT_CC_DFP_ICC
   : Pseudo<(ops DFPRegs:$dst, DFPRegs:$T, DFPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_DFP_ICC PSEUDO!",
            [(set DFPRegs:$dst, (SPselecticc DFPRegs:$T, DFPRegs:$F,
                                             imm:$Cond))]>;
  def SELECT_CC_DFP_FCC
   : Pseudo<(ops DFPRegs:$dst, DFPRegs:$T, DFPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_DFP_FCC PSEUDO!",
            [(set DFPRegs:$dst, (SPselectfcc DFPRegs:$T, DFPRegs:$F,
                                             imm:$Cond))]>;
}


// Section A.3 - Synthetic Instructions, p. 85
// special cases of JMPL:
let isReturn = 1, isTerminator = 1, hasDelaySlot = 1, noResults = 1 in {
  let rd = O7.Num, rs1 = G0.Num, simm13 = 8 in
    def RETL: F3_2<2, 0b111000, (ops), "retl", [(retflag)]>;
}

// Section B.1 - Load Integer Instructions, p. 90
def LDSBrr : F3_1<3, 0b001001,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "ldsb [$addr], $dst",
                  [(set IntRegs:$dst, (sextload ADDRrr:$addr, i8))]>;
def LDSBri : F3_2<3, 0b001001,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "ldsb [$addr], $dst",
                  [(set IntRegs:$dst, (sextload ADDRri:$addr, i8))]>;
def LDSHrr : F3_1<3, 0b001010,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "ldsh [$addr], $dst",
                  [(set IntRegs:$dst, (sextload ADDRrr:$addr, i16))]>;
def LDSHri : F3_2<3, 0b001010,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "ldsh [$addr], $dst",
                  [(set IntRegs:$dst, (sextload ADDRri:$addr, i16))]>;
def LDUBrr : F3_1<3, 0b000001,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "ldub [$addr], $dst",
                  [(set IntRegs:$dst, (zextload ADDRrr:$addr, i8))]>;
def LDUBri : F3_2<3, 0b000001,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "ldub [$addr], $dst",
                  [(set IntRegs:$dst, (zextload ADDRri:$addr, i8))]>;
def LDUHrr : F3_1<3, 0b000010,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "lduh [$addr], $dst",
                  [(set IntRegs:$dst, (zextload ADDRrr:$addr, i16))]>;
def LDUHri : F3_2<3, 0b000010,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "lduh [$addr], $dst",
                  [(set IntRegs:$dst, (zextload ADDRri:$addr, i16))]>;
def LDrr   : F3_1<3, 0b000000,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "ld [$addr], $dst",
                  [(set IntRegs:$dst, (load ADDRrr:$addr))]>;
def LDri   : F3_2<3, 0b000000,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "ld [$addr], $dst",
                  [(set IntRegs:$dst, (load ADDRri:$addr))]>;

// Section B.2 - Load Floating-point Instructions, p. 92
def LDFrr  : F3_1<3, 0b100000,
                  (ops FPRegs:$dst, MEMrr:$addr),
                  "ld [$addr], $dst",
                  [(set FPRegs:$dst, (load ADDRrr:$addr))]>;
def LDFri  : F3_2<3, 0b100000,
                  (ops FPRegs:$dst, MEMri:$addr),
                  "ld [$addr], $dst",
                  [(set FPRegs:$dst, (load ADDRri:$addr))]>;
def LDDFrr : F3_1<3, 0b100011,
                  (ops DFPRegs:$dst, MEMrr:$addr),
                  "ldd [$addr], $dst",
                  [(set DFPRegs:$dst, (load ADDRrr:$addr))]>;
def LDDFri : F3_2<3, 0b100011,
                  (ops DFPRegs:$dst, MEMri:$addr),
                  "ldd [$addr], $dst",
                  [(set DFPRegs:$dst, (load ADDRri:$addr))]>;

// Section B.4 - Store Integer Instructions, p. 95
def STBrr : F3_1<3, 0b000101,
                 (ops MEMrr:$addr, IntRegs:$src),
                 "stb $src, [$addr]",
                 [(truncstore IntRegs:$src, ADDRrr:$addr, i8)]>;
def STBri : F3_2<3, 0b000101,
                 (ops MEMri:$addr, IntRegs:$src),
                 "stb $src, [$addr]",
                 [(truncstore IntRegs:$src, ADDRri:$addr, i8)]>;
def STHrr : F3_1<3, 0b000110,
                 (ops MEMrr:$addr, IntRegs:$src),
                 "sth $src, [$addr]",
                 [(truncstore IntRegs:$src, ADDRrr:$addr, i16)]>;
def STHri : F3_2<3, 0b000110,
                 (ops MEMri:$addr, IntRegs:$src),
                 "sth $src, [$addr]",
                 [(truncstore IntRegs:$src, ADDRri:$addr, i16)]>;
def STrr  : F3_1<3, 0b000100,
                 (ops MEMrr:$addr, IntRegs:$src),
                 "st $src, [$addr]",
                 [(store IntRegs:$src, ADDRrr:$addr)]>;
def STri  : F3_2<3, 0b000100,
                 (ops MEMri:$addr, IntRegs:$src),
                 "st $src, [$addr]",
                 [(store IntRegs:$src, ADDRri:$addr)]>;

// Section B.5 - Store Floating-point Instructions, p. 97
def STFrr   : F3_1<3, 0b100100,
                   (ops MEMrr:$addr, FPRegs:$src),
                   "st $src, [$addr]",
                   [(store FPRegs:$src, ADDRrr:$addr)]>;
def STFri   : F3_2<3, 0b100100,
                   (ops MEMri:$addr, FPRegs:$src),
                   "st $src, [$addr]",
                   [(store FPRegs:$src, ADDRri:$addr)]>;
def STDFrr  : F3_1<3, 0b100111,
                   (ops MEMrr:$addr, DFPRegs:$src),
                   "std  $src, [$addr]",
                   [(store DFPRegs:$src, ADDRrr:$addr)]>;
def STDFri  : F3_2<3, 0b100111,
                   (ops MEMri:$addr, DFPRegs:$src),
                   "std $src, [$addr]",
                   [(store DFPRegs:$src, ADDRri:$addr)]>;

// Section B.9 - SETHI Instruction, p. 104
def SETHIi: F2_1<0b100,
                 (ops IntRegs:$dst, i32imm:$src),
                 "sethi $src, $dst",
                 [(set IntRegs:$dst, SETHIimm:$src)]>;

// Section B.10 - NOP Instruction, p. 105
// (It's a special case of SETHI)
let rd = 0, imm22 = 0 in
  def NOP : F2_1<0b100, (ops), "nop", []>;

// Section B.11 - Logical Instructions, p. 106
def ANDrr   : F3_1<2, 0b000001,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "and $b, $c, $dst",
                   [(set IntRegs:$dst, (and IntRegs:$b, IntRegs:$c))]>;
def ANDri   : F3_2<2, 0b000001,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "and $b, $c, $dst",
                   [(set IntRegs:$dst, (and IntRegs:$b, simm13:$c))]>;
def ANDNrr  : F3_1<2, 0b000101,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "andn $b, $c, $dst",
                   [(set IntRegs:$dst, (and IntRegs:$b, (not IntRegs:$c)))]>;
def ANDNri  : F3_2<2, 0b000101,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "andn $b, $c, $dst", []>;
def ORrr    : F3_1<2, 0b000010,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "or $b, $c, $dst",
                   [(set IntRegs:$dst, (or IntRegs:$b, IntRegs:$c))]>;
def ORri    : F3_2<2, 0b000010,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "or $b, $c, $dst",
                   [(set IntRegs:$dst, (or IntRegs:$b, simm13:$c))]>;
def ORNrr   : F3_1<2, 0b000110,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "orn $b, $c, $dst",
                   [(set IntRegs:$dst, (or IntRegs:$b, (not IntRegs:$c)))]>;
def ORNri   : F3_2<2, 0b000110,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "orn $b, $c, $dst", []>;
def XORrr   : F3_1<2, 0b000011,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "xor $b, $c, $dst",
                   [(set IntRegs:$dst, (xor IntRegs:$b, IntRegs:$c))]>;
def XORri   : F3_2<2, 0b000011,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "xor $b, $c, $dst",
                   [(set IntRegs:$dst, (xor IntRegs:$b, simm13:$c))]>;
def XNORrr  : F3_1<2, 0b000111,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "xnor $b, $c, $dst",
                   [(set IntRegs:$dst, (not (xor IntRegs:$b, IntRegs:$c)))]>;
def XNORri  : F3_2<2, 0b000111,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "xnor $b, $c, $dst", []>;

// Section B.12 - Shift Instructions, p. 107
def SLLrr : F3_1<2, 0b100101,
                 (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                 "sll $b, $c, $dst",
                 [(set IntRegs:$dst, (shl IntRegs:$b, IntRegs:$c))]>;
def SLLri : F3_2<2, 0b100101,
                 (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                 "sll $b, $c, $dst",
                 [(set IntRegs:$dst, (shl IntRegs:$b, simm13:$c))]>;
def SRLrr : F3_1<2, 0b100110, 
                 (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                  "srl $b, $c, $dst",
                  [(set IntRegs:$dst, (srl IntRegs:$b, IntRegs:$c))]>;
def SRLri : F3_2<2, 0b100110,
                 (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                 "srl $b, $c, $dst", 
                 [(set IntRegs:$dst, (srl IntRegs:$b, simm13:$c))]>;
def SRArr : F3_1<2, 0b100111, 
                 (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                  "sra $b, $c, $dst",
                  [(set IntRegs:$dst, (sra IntRegs:$b, IntRegs:$c))]>;
def SRAri : F3_2<2, 0b100111,
                 (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                 "sra $b, $c, $dst",
                 [(set IntRegs:$dst, (sra IntRegs:$b, simm13:$c))]>;

// Section B.13 - Add Instructions, p. 108
def ADDrr   : F3_1<2, 0b000000, 
                  (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                  "add $b, $c, $dst",
                   [(set IntRegs:$dst, (add IntRegs:$b, IntRegs:$c))]>;
def ADDri   : F3_2<2, 0b000000,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "add $b, $c, $dst",
                   [(set IntRegs:$dst, (add IntRegs:$b, simm13:$c))]>;

// "LEA" forms of add (patterns to make tblgen happy)
def LEA_ADDri   : F3_2<2, 0b000000,
                   (ops IntRegs:$dst, MEMri:$addr),
                   "add ${addr:arith}, $dst",
                   [(set IntRegs:$dst, ADDRri:$addr)]>;
                   
def ADDCCrr : F3_1<2, 0b010000, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "addcc $b, $c, $dst",
                   [(set IntRegs:$dst, (addc IntRegs:$b, IntRegs:$c))]>;
def ADDCCri : F3_2<2, 0b010000,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "addcc $b, $c, $dst", 
                   [(set IntRegs:$dst, (addc IntRegs:$b, simm13:$c))]>;
def ADDXrr  : F3_1<2, 0b001000, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "addx $b, $c, $dst",
                   [(set IntRegs:$dst, (adde IntRegs:$b, IntRegs:$c))]>;
def ADDXri  : F3_2<2, 0b001000,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "addx $b, $c, $dst",
                   [(set IntRegs:$dst, (adde IntRegs:$b, simm13:$c))]>;

// Section B.15 - Subtract Instructions, p. 110
def SUBrr   : F3_1<2, 0b000100, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "sub $b, $c, $dst",
                   [(set IntRegs:$dst, (sub IntRegs:$b, IntRegs:$c))]>;
def SUBri   : F3_2<2, 0b000100,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "sub $b, $c, $dst",
                   [(set IntRegs:$dst, (sub IntRegs:$b, simm13:$c))]>;
def SUBXrr  : F3_1<2, 0b001100, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "subx $b, $c, $dst",
                   [(set IntRegs:$dst, (sube IntRegs:$b, IntRegs:$c))]>;
def SUBXri  : F3_2<2, 0b001100,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "subx $b, $c, $dst",
                   [(set IntRegs:$dst, (sube IntRegs:$b, simm13:$c))]>;
def SUBCCrr : F3_1<2, 0b010100, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "subcc $b, $c, $dst",
                   [(set IntRegs:$dst, (SPcmpicc IntRegs:$b, IntRegs:$c))]>;
def SUBCCri : F3_2<2, 0b010100,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "subcc $b, $c, $dst",
                   [(set IntRegs:$dst, (SPcmpicc IntRegs:$b, simm13:$c))]>;
def SUBXCCrr: F3_1<2, 0b011100, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "subxcc $b, $c, $dst", []>;

// Section B.18 - Multiply Instructions, p. 113
def UMULrr  : F3_1<2, 0b001010, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "umul $b, $c, $dst", []>;
def UMULri  : F3_2<2, 0b001010,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "umul $b, $c, $dst", []>;
                   
def SMULrr  : F3_1<2, 0b001011, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "smul $b, $c, $dst",
                   [(set IntRegs:$dst, (mul IntRegs:$b, IntRegs:$c))]>;
def SMULri  : F3_2<2, 0b001011,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "smul $b, $c, $dst",
                   [(set IntRegs:$dst, (mul IntRegs:$b, simm13:$c))]>;

/*
//===-------------------------
// Sparc Example
defm intinst{OPC1, OPC2}<bits Opc, string asmstr, SDNode code> {
  def OPC1 : F3_1<2, Opc, asmstr, (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                  [(set IntRegs:$dst, (code IntRegs:$b, IntRegs:$c))]>;
  def OPC2 : F3_2<2, Opc, asmstr, (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                  [(set IntRegs:$dst, (code IntRegs:$b, simm13:$c))]>;
}
defm intinst_np{OPC1, OPC2}<bits Opc, string asmstr> {
  def OPC1 : F3_1<2, Opc, asmstr, (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                  []>;
  def OPC2 : F3_2<2, Opc, asmstr, (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                  []>;
}

def { ADDXrr,  ADDXri} : intinstnp<0b001000,  "addx $b, $c, $dst">;
def {  SUBrr,   SUBri} : intinst  <0b000100,   "sub $b, $c, $dst",  sub>;
def intinstnp{ SUBXrr,  SUBXri}<0b001100,  "subx $b, $c, $dst">;
def intinst  {SUBCCrr, SUBCCri}<0b010100, "subcc $b, $c, $dst",  SPcmpicc>;
def intinst  { SMULrr,  SMULri}<0b001011,  "smul $b, $c, $dst",  mul>;

//===-------------------------
// X86 Example
defm cmov32<id OPC1, id OPC2, int opc, string asmstr, PatLeaf cond> {
  def OPC1 : I<opc, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
               asmstr+" {$src2, $dst|$dst, $src2}",
               [(set R32:$dst, (X86cmov R32:$src1, R32:$src2, cond))]>, TB;
  def OPC2 : I<opc, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
               asmstr+" {$src2, $dst|$dst, $src2}",
               [(set R32:$dst, (X86cmov R32:$src1,
                                        (loadi32 addr:$src2), cond))]>, TB;
}

def cmov<CMOVL32rr, CMOVL32rm, 0x4C, "cmovl", X86_COND_L>;
def cmov<CMOVB32rr, CMOVB32rm, 0x4C, "cmovb", X86_COND_B>;

//===-------------------------
// PPC Example

def fpunop<id OPC1, id OPC2, id FORM, int op1, int op2, int op3, string asmstr, 
           SDNode code> {
  def OPC1 : FORM<op1, op3, (ops F4RC:$frD, F4RC:$frB),
                  asmstr+" $frD, $frB", FPGeneral,
                  [(set F4RC:$frD, (code F4RC:$frB))]>;
  def OPC2 : FORM<op2, op3, (ops F8RC:$frD, F8RC:$frB),
                  asmstr+" $frD, $frB", FPGeneral,
                  [(set F8RC:$frD, (code F8RC:$frB))]>;
}

def fpunop< FABSS,  FABSD, XForm_26, 63, 63, 264,  "fabs",  fabs>;
def fpunop<FNABSS, FNABSD, XForm_26, 63, 63, 136, "fnabs", fnabs>;
def fpunop< FNEGS,  FNEGD, XForm_26, 63, 63,  40,  "fneg",  fneg>;
*/

// Section B.19 - Divide Instructions, p. 115
def UDIVrr   : F3_1<2, 0b001110, 
                    (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                    "udiv $b, $c, $dst", []>;
def UDIVri   : F3_2<2, 0b001110,
                    (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                    "udiv $b, $c, $dst", []>;
def SDIVrr   : F3_1<2, 0b001111,
                    (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                    "sdiv $b, $c, $dst", []>;
def SDIVri   : F3_2<2, 0b001111,
                    (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                    "sdiv $b, $c, $dst", []>;

// Section B.20 - SAVE and RESTORE, p. 117
def SAVErr    : F3_1<2, 0b111100,
                     (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                     "save $b, $c, $dst", []>;
def SAVEri    : F3_2<2, 0b111100,
                     (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                     "save $b, $c, $dst", []>;
def RESTORErr : F3_1<2, 0b111101,
                     (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                     "restore $b, $c, $dst", []>;
def RESTOREri : F3_2<2, 0b111101,
                     (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                     "restore $b, $c, $dst", []>;

// Section B.21 - Branch on Integer Condition Codes Instructions, p. 119

// conditional branch class:
class BranchSP<bits<4> cc, dag ops, string asmstr, list<dag> pattern>
 : F2_2<cc, 0b010, ops, asmstr, pattern> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let noResults = 1;
}

let isBarrier = 1 in
  def BA   : BranchSP<0b1000, (ops brtarget:$dst),
                      "ba $dst",
                      [(br bb:$dst)]>;
                      
// FIXME: the encoding for the JIT should look at the condition field.
def BCOND : BranchSP<0, (ops brtarget:$dst, CCOp:$cc),
                     "b$cc $dst",
                     [(SPbricc bb:$dst, imm:$cc)]>;


// Section B.22 - Branch on Floating-point Condition Codes Instructions, p. 121

// floating-point conditional branch class:
class FPBranchSP<bits<4> cc, dag ops, string asmstr, list<dag> pattern>
 : F2_2<cc, 0b110, ops, asmstr, pattern> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let noResults = 1;
}

// FIXME: the encoding for the JIT should look at the condition field.
def FBCOND  : FPBranchSP<0, (ops brtarget:$dst, CCOp:$cc),
                      "fb$cc $dst",
                      [(SPbrfcc bb:$dst, imm:$cc)]>;


// Section B.24 - Call and Link Instruction, p. 125
// This is the only Format 1 instruction
let Uses = [O0, O1, O2, O3, O4, O5],
    hasDelaySlot = 1, isCall = 1, noResults = 1,
    Defs = [O0, O1, O2, O3, O4, O5, O7, G1, G2, G3, G4, G5, G6, G7,
    D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15] in { 
  def CALL : InstSP<(ops calltarget:$dst),
                    "call $dst", []> {
    bits<30> disp;
    let op = 1;
    let Inst{29-0} = disp;
  }
  
  // indirect calls
  def JMPLrr : F3_1<2, 0b111000,
                    (ops MEMrr:$ptr),
                    "call $ptr",
                    [(call  ADDRrr:$ptr)]>;
  def JMPLri : F3_2<2, 0b111000,
                    (ops MEMri:$ptr),
                    "call $ptr",
                    [(call  ADDRri:$ptr)]>;
}

// Section B.28 - Read State Register Instructions
def RDY : F3_1<2, 0b101000,
               (ops IntRegs:$dst),
               "rd %y, $dst", []>;

// Section B.29 - Write State Register Instructions
def WRYrr : F3_1<2, 0b110000,
                 (ops IntRegs:$b, IntRegs:$c),
                 "wr $b, $c, %y", []>;
def WRYri : F3_2<2, 0b110000,
                 (ops IntRegs:$b, i32imm:$c),
                 "wr $b, $c, %y", []>;

// Convert Integer to Floating-point Instructions, p. 141
def FITOS : F3_3<2, 0b110100, 0b011000100,
                 (ops FPRegs:$dst, FPRegs:$src),
                 "fitos $src, $dst",
                 [(set FPRegs:$dst, (SPitof FPRegs:$src))]>;
def FITOD : F3_3<2, 0b110100, 0b011001000, 
                 (ops DFPRegs:$dst, FPRegs:$src),
                 "fitod $src, $dst",
                 [(set DFPRegs:$dst, (SPitof FPRegs:$src))]>;

// Convert Floating-point to Integer Instructions, p. 142
def FSTOI : F3_3<2, 0b110100, 0b011010001,
                 (ops FPRegs:$dst, FPRegs:$src),
                 "fstoi $src, $dst",
                 [(set FPRegs:$dst, (SPftoi FPRegs:$src))]>;
def FDTOI : F3_3<2, 0b110100, 0b011010010,
                 (ops FPRegs:$dst, DFPRegs:$src),
                 "fdtoi $src, $dst",
                 [(set FPRegs:$dst, (SPftoi DFPRegs:$src))]>;

// Convert between Floating-point Formats Instructions, p. 143
def FSTOD : F3_3<2, 0b110100, 0b011001001, 
                 (ops DFPRegs:$dst, FPRegs:$src),
                 "fstod $src, $dst",
                 [(set DFPRegs:$dst, (fextend FPRegs:$src))]>;
def FDTOS : F3_3<2, 0b110100, 0b011000110,
                 (ops FPRegs:$dst, DFPRegs:$src),
                 "fdtos $src, $dst",
                 [(set FPRegs:$dst, (fround DFPRegs:$src))]>;

// Floating-point Move Instructions, p. 144
def FMOVS : F3_3<2, 0b110100, 0b000000001,
                 (ops FPRegs:$dst, FPRegs:$src),
                 "fmovs $src, $dst", []>;
def FNEGS : F3_3<2, 0b110100, 0b000000101, 
                 (ops FPRegs:$dst, FPRegs:$src),
                 "fnegs $src, $dst",
                 [(set FPRegs:$dst, (fneg FPRegs:$src))]>;
def FABSS : F3_3<2, 0b110100, 0b000001001, 
                 (ops FPRegs:$dst, FPRegs:$src),
                 "fabss $src, $dst",
                 [(set FPRegs:$dst, (fabs FPRegs:$src))]>;


// Floating-point Square Root Instructions, p.145
def FSQRTS : F3_3<2, 0b110100, 0b000101001, 
                  (ops FPRegs:$dst, FPRegs:$src),
                  "fsqrts $src, $dst",
                  [(set FPRegs:$dst, (fsqrt FPRegs:$src))]>;
def FSQRTD : F3_3<2, 0b110100, 0b000101010, 
                  (ops DFPRegs:$dst, DFPRegs:$src),
                  "fsqrtd $src, $dst",
                  [(set DFPRegs:$dst, (fsqrt DFPRegs:$src))]>;



// Floating-point Add and Subtract Instructions, p. 146
def FADDS  : F3_3<2, 0b110100, 0b001000001,
                  (ops FPRegs:$dst, FPRegs:$src1, FPRegs:$src2),
                  "fadds $src1, $src2, $dst",
                  [(set FPRegs:$dst, (fadd FPRegs:$src1, FPRegs:$src2))]>;
def FADDD  : F3_3<2, 0b110100, 0b001000010,
                  (ops DFPRegs:$dst, DFPRegs:$src1, DFPRegs:$src2),
                  "faddd $src1, $src2, $dst",
                  [(set DFPRegs:$dst, (fadd DFPRegs:$src1, DFPRegs:$src2))]>;
def FSUBS  : F3_3<2, 0b110100, 0b001000101,
                  (ops FPRegs:$dst, FPRegs:$src1, FPRegs:$src2),
                  "fsubs $src1, $src2, $dst",
                  [(set FPRegs:$dst, (fsub FPRegs:$src1, FPRegs:$src2))]>;
def FSUBD  : F3_3<2, 0b110100, 0b001000110,
                  (ops DFPRegs:$dst, DFPRegs:$src1, DFPRegs:$src2),
                  "fsubd $src1, $src2, $dst",
                  [(set DFPRegs:$dst, (fsub DFPRegs:$src1, DFPRegs:$src2))]>;

// Floating-point Multiply and Divide Instructions, p. 147
def FMULS  : F3_3<2, 0b110100, 0b001001001,
                  (ops FPRegs:$dst, FPRegs:$src1, FPRegs:$src2),
                  "fmuls $src1, $src2, $dst",
                  [(set FPRegs:$dst, (fmul FPRegs:$src1, FPRegs:$src2))]>;
def FMULD  : F3_3<2, 0b110100, 0b001001010,
                  (ops DFPRegs:$dst, DFPRegs:$src1, DFPRegs:$src2),
                  "fmuld $src1, $src2, $dst",
                  [(set DFPRegs:$dst, (fmul DFPRegs:$src1, DFPRegs:$src2))]>;
def FSMULD : F3_3<2, 0b110100, 0b001101001,
                  (ops DFPRegs:$dst, FPRegs:$src1, FPRegs:$src2),
                  "fsmuld $src1, $src2, $dst",
                  [(set DFPRegs:$dst, (fmul (fextend FPRegs:$src1),
                                            (fextend FPRegs:$src2)))]>;
def FDIVS  : F3_3<2, 0b110100, 0b001001101,
                 (ops FPRegs:$dst, FPRegs:$src1, FPRegs:$src2),
                 "fdivs $src1, $src2, $dst",
                 [(set FPRegs:$dst, (fdiv FPRegs:$src1, FPRegs:$src2))]>;
def FDIVD  : F3_3<2, 0b110100, 0b001001110,
                 (ops DFPRegs:$dst, DFPRegs:$src1, DFPRegs:$src2),
                 "fdivd $src1, $src2, $dst",
                 [(set DFPRegs:$dst, (fdiv DFPRegs:$src1, DFPRegs:$src2))]>;

// Floating-point Compare Instructions, p. 148
// Note: the 2nd template arg is different for these guys.
// Note 2: the result of a FCMP is not available until the 2nd cycle
// after the instr is retired, but there is no interlock. This behavior
// is modelled with a forced noop after the instruction.
def FCMPS  : F3_3<2, 0b110101, 0b001010001,
                  (ops FPRegs:$src1, FPRegs:$src2),
                  "fcmps $src1, $src2\n\tnop",
                  [(SPcmpfcc FPRegs:$src1, FPRegs:$src2)]>;
def FCMPD  : F3_3<2, 0b110101, 0b001010010,
                  (ops DFPRegs:$src1, DFPRegs:$src2),
                  "fcmpd $src1, $src2\n\tnop",
                  [(SPcmpfcc DFPRegs:$src1, DFPRegs:$src2)]>;


//===----------------------------------------------------------------------===//
// V9 Instructions
//===----------------------------------------------------------------------===//

// V9 Conditional Moves.
let Predicates = [HasV9], isTwoAddress = 1 in {
  // Move Integer Register on Condition (MOVcc) p. 194 of the V9 manual.
  // FIXME: Add instruction encodings for the JIT some day.
  def MOVICCrr
    : Pseudo<(ops IntRegs:$dst, IntRegs:$T, IntRegs:$F, CCOp:$cc),
             "mov$cc %icc, $F, $dst",
             [(set IntRegs:$dst,
                         (SPselecticc IntRegs:$F, IntRegs:$T, imm:$cc))]>;
  def MOVICCri
    : Pseudo<(ops IntRegs:$dst, IntRegs:$T, i32imm:$F, CCOp:$cc),
             "mov$cc %icc, $F, $dst",
             [(set IntRegs:$dst,
                          (SPselecticc simm11:$F, IntRegs:$T, imm:$cc))]>;

  def MOVFCCrr
    : Pseudo<(ops IntRegs:$dst, IntRegs:$T, IntRegs:$F, CCOp:$cc),
             "mov$cc %fcc0, $F, $dst",
             [(set IntRegs:$dst,
                         (SPselectfcc IntRegs:$F, IntRegs:$T, imm:$cc))]>;
  def MOVFCCri
    : Pseudo<(ops IntRegs:$dst, IntRegs:$T, i32imm:$F, CCOp:$cc),
             "mov$cc %fcc0, $F, $dst",
             [(set IntRegs:$dst,
                          (SPselectfcc simm11:$F, IntRegs:$T, imm:$cc))]>;

  def FMOVS_ICC
    : Pseudo<(ops FPRegs:$dst, FPRegs:$T, FPRegs:$F, CCOp:$cc),
             "fmovs$cc %icc, $F, $dst",
             [(set FPRegs:$dst,
                         (SPselecticc FPRegs:$F, FPRegs:$T, imm:$cc))]>;
  def FMOVD_ICC
    : Pseudo<(ops DFPRegs:$dst, DFPRegs:$T, DFPRegs:$F, CCOp:$cc),
             "fmovd$cc %icc, $F, $dst",
             [(set DFPRegs:$dst,
                         (SPselecticc DFPRegs:$F, DFPRegs:$T, imm:$cc))]>;
  def FMOVS_FCC
    : Pseudo<(ops FPRegs:$dst, FPRegs:$T, FPRegs:$F, CCOp:$cc),
             "fmovs$cc %fcc0, $F, $dst",
             [(set FPRegs:$dst,
                         (SPselectfcc FPRegs:$F, FPRegs:$T, imm:$cc))]>;
  def FMOVD_FCC
    : Pseudo<(ops DFPRegs:$dst, DFPRegs:$T, DFPRegs:$F, CCOp:$cc),
             "fmovd$cc %fcc0, $F, $dst",
             [(set DFPRegs:$dst,
                         (SPselectfcc DFPRegs:$F, DFPRegs:$T, imm:$cc))]>;

}

// Floating-Point Move Instructions, p. 164 of the V9 manual.
let Predicates = [HasV9] in {
  def FMOVD : F3_3<2, 0b110100, 0b000000010,
                   (ops DFPRegs:$dst, DFPRegs:$src),
                   "fmovd $src, $dst", []>;
  def FNEGD : F3_3<2, 0b110100, 0b000000110, 
                   (ops DFPRegs:$dst, DFPRegs:$src),
                   "fnegd $src, $dst",
                   [(set DFPRegs:$dst, (fneg DFPRegs:$src))]>;
  def FABSD : F3_3<2, 0b110100, 0b000001010, 
                   (ops DFPRegs:$dst, DFPRegs:$src),
                   "fabsd $src, $dst",
                   [(set DFPRegs:$dst, (fabs DFPRegs:$src))]>;
}

// POPCrr - This does a ctpop of a 64-bit register.  As such, we have to clear
// the top 32-bits before using it.  To do this clearing, we use a SLLri X,0.
def POPCrr : F3_1<2, 0b101110, 
                  (ops IntRegs:$dst, IntRegs:$src),
                  "popc $src, $dst", []>, Requires<[HasV9]>;
def : Pat<(ctpop IntRegs:$src),
          (POPCrr (SLLri IntRegs:$src, 0))>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// Small immediates.
def : Pat<(i32 simm13:$val),
          (ORri G0, imm:$val)>;
// Arbitrary immediates.
def : Pat<(i32 imm:$val),
          (ORri (SETHIi (HI22 imm:$val)), (LO10 imm:$val))>;

// subc
def : Pat<(subc IntRegs:$b, IntRegs:$c),
          (SUBCCrr IntRegs:$b, IntRegs:$c)>;
def : Pat<(subc IntRegs:$b, simm13:$val),
          (SUBCCri IntRegs:$b, imm:$val)>;

// Global addresses, constant pool entries
def : Pat<(SPhi tglobaladdr:$in), (SETHIi tglobaladdr:$in)>;
def : Pat<(SPlo tglobaladdr:$in), (ORri G0, tglobaladdr:$in)>;
def : Pat<(SPhi tconstpool:$in), (SETHIi tconstpool:$in)>;
def : Pat<(SPlo tconstpool:$in), (ORri G0, tconstpool:$in)>;

// Add reg, lo.  This is used when taking the addr of a global/constpool entry.
def : Pat<(add IntRegs:$r, (SPlo tglobaladdr:$in)),
          (ADDri IntRegs:$r, tglobaladdr:$in)>;
def : Pat<(add IntRegs:$r, (SPlo tconstpool:$in)),
          (ADDri IntRegs:$r, tconstpool:$in)>;

// Calls: 
def : Pat<(call tglobaladdr:$dst),
          (CALL tglobaladdr:$dst)>;
def : Pat<(call texternalsym:$dst),
          (CALL texternalsym:$dst)>;

def : Pat<(ret), (RETL)>;

// Map integer extload's to zextloads.
def : Pat<(i32 (extload ADDRrr:$src, i1)), (LDUBrr ADDRrr:$src)>;
def : Pat<(i32 (extload ADDRri:$src, i1)), (LDUBri ADDRri:$src)>;
def : Pat<(i32 (extload ADDRrr:$src, i8)), (LDUBrr ADDRrr:$src)>;
def : Pat<(i32 (extload ADDRri:$src, i8)), (LDUBri ADDRri:$src)>;
def : Pat<(i32 (extload ADDRrr:$src, i16)), (LDUHrr ADDRrr:$src)>;
def : Pat<(i32 (extload ADDRri:$src, i16)), (LDUHri ADDRri:$src)>;

// zextload bool -> zextload byte
def : Pat<(i32 (zextload ADDRrr:$src, i1)), (LDUBrr ADDRrr:$src)>;
def : Pat<(i32 (zextload ADDRri:$src, i1)), (LDUBri ADDRri:$src)>;

// truncstore bool -> truncstore byte.
def : Pat<(truncstore IntRegs:$src, ADDRrr:$addr, i1), 
          (STBrr ADDRrr:$addr, IntRegs:$src)>;
def : Pat<(truncstore IntRegs:$src, ADDRri:$addr, i1), 
          (STBri ADDRri:$addr, IntRegs:$src)>;
