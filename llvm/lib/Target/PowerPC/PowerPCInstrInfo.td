//===- PowerPCInstrInfo.td - The PowerPC Instruction Set -----*- tablegen -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the subset of the 32-bit PowerPC instruction set, as used
// by the PowerPC instruction selector.
//
//===----------------------------------------------------------------------===//

include "PowerPCInstrFormats.td"

//===----------------------------------------------------------------------===//
// Selection DAG Type Constraint definitions.
//
// Note that the semantics of these constraints are hard coded into tblgen.  To
// modify or add constraints, you have to hack tblgen.
//

class SDTypeConstraint<int opnum> {
  int OperandNum = opnum;
}

// SDTCisVT - The specified operand has exactly this VT.
class SDTCisVT <int OpNum, ValueType vt> : SDTypeConstraint<OpNum> {
  ValueType VT = vt;
}

// SDTCisInt - The specified operand is has integer type.
class SDTCisInt<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisFP - The specified operand is has floating point type.
class SDTCisFP <int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisSameAs - The two specified operands have identical types.
class SDTCisSameAs<int OpNum, int OtherOp> : SDTypeConstraint<OpNum> {
  int OtherOperandNum = OtherOp;
}

// SDTCisVTSmallerThanOp - The specified operand is a VT SDNode, and its type is
// smaller than the 'Other' operand.
class SDTCisVTSmallerThanOp<int OpNum, int OtherOp> : SDTypeConstraint<OpNum> {
  int OtherOperandNum = OtherOp;
}

//===----------------------------------------------------------------------===//
// Selection DAG Type Profile definitions.
//
// These use the constraints defined above to describe the type requirements of
// the various nodes.  These are not hard coded into tblgen, allowing targets to
// add their own if needed.
//

// SDTypeProfile - This profile describes the type requirements of a Selection
// DAG node.
class SDTypeProfile<int numresults, int numoperands,
                    list<SDTypeConstraint> constraints> {
  int NumResults = numresults;
  int NumOperands = numoperands;
  list<SDTypeConstraint> Constraints = constraints;
}

// Builtin profiles.
def SDTImm    : SDTypeProfile<1, 0, [SDTCisInt<0>]>;      // for 'imm'.
def SDTVT     : SDTypeProfile<1, 0, [SDTCisVT<0, OtherVT>]>; // for 'vt'
def SDTIntBinOp : SDTypeProfile<1, 2, [   // add, and, or, xor, udiv, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisInt<0>
]>;
def SDTFPBinOp : SDTypeProfile<1, 2, [      // fadd, fmul, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisFP<0>
]>;
def SDTIntUnaryOp : SDTypeProfile<1, 1, [   // ctlz
  SDTCisSameAs<0, 1>, SDTCisInt<0>
]>;
def SDTExtInreg : SDTypeProfile<1, 2, [   // sext_inreg
  SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisVT<2, OtherVT>,
  SDTCisVTSmallerThanOp<2, 1>
]>;

//===----------------------------------------------------------------------===//
// Selection DAG Node Properties.
//
// Note: These are hard coded into tblgen.
//
class SDNodeProperty;
def SDNPCommutative : SDNodeProperty;   // X op Y == Y op X
def SDNPAssociative : SDNodeProperty;   // (X op Y) op Z == X op (Y op Z)

//===----------------------------------------------------------------------===//
// Selection DAG Node definitions.
//
class SDNode<string opcode, SDTypeProfile typeprof,
             list<SDNodeProperty> props = [], string sdclass = "SDNode"> {
  string Opcode  = opcode;
  string SDClass = sdclass;
  list<SDNodeProperty> Properties = props;
  SDTypeProfile TypeProfile = typeprof;
}

def set;
def node;

def imm        : SDNode<"ISD::Constant"  , SDTImm     , [], "ConstantSDNode">;
def vt         : SDNode<"ISD::VALUETYPE" , SDTVT      , [], "VTSDNode">;
def add        : SDNode<"ISD::ADD"       , SDTIntBinOp   ,
                        [SDNPCommutative, SDNPAssociative]>;
def sub        : SDNode<"ISD::SUB"       , SDTIntBinOp>;
def mul        : SDNode<"ISD::MUL"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def mulhs      : SDNode<"ISD::MULHS"     , SDTIntBinOp, [SDNPCommutative]>;
def mulhu      : SDNode<"ISD::MULHU"     , SDTIntBinOp, [SDNPCommutative]>;
def sdiv       : SDNode<"ISD::SDIV"      , SDTIntBinOp>;
def udiv       : SDNode<"ISD::UDIV"      , SDTIntBinOp>;
def srem       : SDNode<"ISD::SREM"      , SDTIntBinOp>;
def urem       : SDNode<"ISD::UREM"      , SDTIntBinOp>;
def and        : SDNode<"ISD::AND"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def or         : SDNode<"ISD::OR"        , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def xor        : SDNode<"ISD::XOR"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def fadd       : SDNode<"ISD::FADD"       , SDTFPBinOp, [SDNPCommutative]>;
def fsub       : SDNode<"ISD::FSUB"       , SDTFPBinOp>;
def fmul       : SDNode<"ISD::FMUL"       , SDTFPBinOp, [SDNPCommutative]>;
def fdiv       : SDNode<"ISD::FDIV"       , SDTFPBinOp>;
def frem       : SDNode<"ISD::FREM"       , SDTFPBinOp>;

def sext_inreg : SDNode<"ISD::SIGN_EXTEND_INREG", SDTExtInreg>;
def ctlz       : SDNode<"ISD::CTLZ"      , SDTIntUnaryOp>;

//===----------------------------------------------------------------------===//
// Selection DAG Node Transformation Functions.
//
// This mechanism allows targets to manipulate nodes in the output DAG once a
// match has been formed.  This is typically used to manipulate immediate
// values.
//
class SDNodeXForm<SDNode opc, code xformFunction> {
  SDNode Opcode = opc;
  code XFormFunction = xformFunction;
}

def NOOP_SDNodeXForm : SDNodeXForm<imm, [{}]>;


//===----------------------------------------------------------------------===//
// Selection DAG Pattern Fragments.
//
// Pattern fragments are reusable chunks of dags that match specific things.
// They can take arguments and have C++ predicates that control whether they
// match.  They are intended to make the patterns for common instructions more
// compact and readable.
//

/// PatFrag - Represents a pattern fragment.  This can match something on the
/// DAG, frame a single node to multiply nested other fragments.
///
class PatFrag<dag ops, dag frag, code pred = [{}],
              SDNodeXForm xform = NOOP_SDNodeXForm> {
  dag Operands = ops;
  dag Fragment = frag;
  code Predicate = pred;
  SDNodeXForm OperandTransform = xform;
}

// PatLeaf's are pattern fragments that have no operands.  This is just a helper
// to define immediates and other common things concisely.
class PatLeaf<dag frag, code pred = [{}], SDNodeXForm xform = NOOP_SDNodeXForm>
 : PatFrag<(ops), frag, pred, xform>;

// Leaf fragments.

def immAllOnes : PatLeaf<(imm), [{ return N->isAllOnesValue(); }]>;
def immZero    : PatLeaf<(imm), [{ return N->isNullValue();    }]>;

def vtInt      : PatLeaf<(vt),  [{ return MVT::isInteger(N->getVT()); }]>;
def vtFP       : PatLeaf<(vt),  [{ return MVT::isFloatingPoint(N->getVT()); }]>;

// Other helper fragments.

def not  : PatFrag<(ops node:$in), (xor node:$in, immAllOnes)>;
def ineg : PatFrag<(ops node:$in), (sub immZero, node:$in)>;

//===----------------------------------------------------------------------===//
// Selection DAG Pattern Support.
//
// Patterns are what are actually matched against the target-flavored
// instruction selection DAG.  Instructions defined by the target implicitly
// define patterns in most cases, but patterns can also be explicitly added when
// an operation is defined by a sequence of instructions (e.g. loading a large
// immediate value on RISC targets that do not support immediates as large as
// their GPRs).
//

class Pattern<dag patternToMatch, list<dag> resultInstrs> {
  dag       PatternToMatch = patternToMatch;
  list<dag> ResultInstrs   = resultInstrs;
}

// Pat - A simple (but common) form of a pattern, which produces a simple result
// not needing a full list.
class Pat<dag pattern, dag result> : Pattern<pattern, [result]>;

//===----------------------------------------------------------------------===//
// PowerPC specific transformation functions and pattern fragments.
//
def LO16 : SDNodeXForm<imm, [{
  // Transformation function: get the low 16 bits.
  return getI32Imm((unsigned short)N->getValue());
}]>;

def HI16 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return getI32Imm((unsigned)N->getValue() >> 16);
}]>;

def HA16 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  signed int Val = N->getValue();
  return getI32Imm((Val - (signed short)Val) >> 16);
}]>;


def immSExt16  : PatLeaf<(imm), [{
  // immSExt16 predicate - True if the immediate fits in a 16-bit sign extended
  // field.  Used by instructions like 'addi'.
  return (int)N->getValue() == (short)N->getValue();
}]>;
def immZExt16  : PatLeaf<(imm), [{
  // immZExt16 predicate - True if the immediate fits in a 16-bit zero extended
  // field.  Used by instructions like 'ori'.
  return (unsigned)N->getValue() == (unsigned short)N->getValue();
}], LO16>;

def imm16Shifted : PatLeaf<(imm), [{
  // imm16Shifted predicate - True if only bits in the top 16-bits of the
  // immediate are set.  Used by instructions like 'addis'.
  return ((unsigned)N->getValue() & 0xFFFF0000U) == (unsigned)N->getValue();
}], HI16>;

/*
// Example of a legalize expander: Only for PPC64.
def : Expander<(set i64:$dst, (fp_to_sint f64:$src)),
               [(set f64:$tmp , (FCTIDZ f64:$src)),
                (set i32:$tmpFI, (CreateNewFrameIndex 8, 8)),
                (store f64:$tmp, i32:$tmpFI),
                (set i64:$dst, (load i32:$tmpFI))],
                Subtarget_PPC64>;
*/

//===----------------------------------------------------------------------===//
// PowerPC Flag Definitions.

class isPPC64 { bit PPC64 = 1; }
class isVMX   { bit VMX = 1; }
class isDOT   {
  list<Register> Defs = [CR0];
  bit RC  = 1;
}



//===----------------------------------------------------------------------===//
// PowerPC Operand Definitions.

def u5imm   : Operand<i32> {
  let PrintMethod = "printU5ImmOperand";
}
def u6imm   : Operand<i32> {
  let PrintMethod = "printU6ImmOperand";
}
def s16imm  : Operand<i32> {
  let PrintMethod = "printS16ImmOperand";
}
def u16imm  : Operand<i32> {
  let PrintMethod = "printU16ImmOperand";
}
def target : Operand<i32> {
  let PrintMethod = "printBranchOperand";
}
def piclabel: Operand<i32> {
  let PrintMethod = "printPICLabel";
}
def symbolHi: Operand<i32> {
  let PrintMethod = "printSymbolHi";
}
def symbolLo: Operand<i32> {
  let PrintMethod = "printSymbolLo";
}
def crbitm: Operand<i8> {
  let PrintMethod = "printcrbitm";
}



//===----------------------------------------------------------------------===//
// PowerPC Instruction Definitions.

// Pseudo-instructions:
def PHI : Pseudo<(ops variable_ops), "; PHI">;

let isLoad = 1 in {
def ADJCALLSTACKDOWN : Pseudo<(ops u16imm:$amt), "; ADJCALLSTACKDOWN">;
def ADJCALLSTACKUP : Pseudo<(ops u16imm:$amt), "; ADJCALLSTACKUP">;
}
def IMPLICIT_DEF_GPR : Pseudo<(ops GPRC:$rD), "; $rD = IMPLICIT_DEF_GPRC">;
def IMPLICIT_DEF_FP  : Pseudo<(ops FPRC:$rD), "; %rD = IMPLICIT_DEF_FP">;

// SELECT_CC_* - Used to implement the SELECT_CC DAG operation.  Expanded by the
// scheduler into a branch sequence.
let usesCustomDAGSchedInserter = 1 in {  // Expanded by the scheduler.
  def SELECT_CC_Int : Pseudo<(ops GPRC:$dst, CRRC:$cond, GPRC:$T, GPRC:$F,
                              i32imm:$BROPC), "; SELECT_CC PSEUDO!">;
  def SELECT_CC_FP  : Pseudo<(ops FPRC:$dst, CRRC:$cond, FPRC:$T, FPRC:$F,
                              i32imm:$BROPC), "; SELECT_CC PSEUDO!">;
}


let isTerminator = 1 in {
  let isReturn = 1 in
    def BLR : XLForm_2_ext<19, 16, 20, 0, 0, (ops), "blr">;
  def BCTR : XLForm_2_ext<19, 528, 20, 0, 0, (ops), "bctr">;
}

let Defs = [LR] in
  def MovePCtoLR : Pseudo<(ops piclabel:$label), "bl $label">;

let isBranch = 1, isTerminator = 1 in {
  def COND_BRANCH : Pseudo<(ops CRRC:$crS, u16imm:$opc,
                                target:$true, target:$false),
                           "; COND_BRANCH">;
  def B   : IForm<18, 0, 0, (ops target:$func), "b $func">;
//def BA  : IForm<18, 1, 0, (ops target:$func), "ba $func">;
  def BL  : IForm<18, 0, 1, (ops target:$func), "bl $func">;
//def BLA : IForm<18, 1, 1, (ops target:$func), "bla $func">;

  // FIXME: 4*CR# needs to be added to the BI field!
  // This will only work for CR0 as it stands now
  def BLT : BForm<16, 0, 0, 12, 0, (ops CRRC:$crS, target:$block),
                  "blt $crS, $block">;
  def BLE : BForm<16, 0, 0, 4,  1, (ops CRRC:$crS, target:$block),
                  "ble $crS, $block">;
  def BEQ : BForm<16, 0, 0, 12, 2, (ops CRRC:$crS, target:$block),
                  "beq $crS, $block">;
  def BGE : BForm<16, 0, 0, 4,  0, (ops CRRC:$crS, target:$block),
                  "bge $crS, $block">;
  def BGT : BForm<16, 0, 0, 12, 1, (ops CRRC:$crS, target:$block),
                  "bgt $crS, $block">;
  def BNE : BForm<16, 0, 0, 4,  2, (ops CRRC:$crS, target:$block),
                  "bne $crS, $block">;
}

let isCall = 1, 
  // All calls clobber the non-callee saved registers...
  Defs = [R0,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,
          F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,
          LR,CTR,
          CR0,CR1,CR5,CR6,CR7] in {
  // Convenient aliases for call instructions
  def CALLpcrel : IForm<18, 0, 1, (ops target:$func, variable_ops), "bl $func">;
  def CALLindirect : XLForm_2_ext<19, 528, 20, 0, 1,
                                  (ops variable_ops), "bctrl">;
}

// D-Form instructions.  Most instructions that perform an operation on a
// register and an immediate are of this type.
//
let isLoad = 1 in {
def LBZ : DForm_1<34, (ops GPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lbz $rD, $disp($rA)">;
def LHA : DForm_1<42, (ops GPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lha $rD, $disp($rA)">;
def LHZ : DForm_1<40, (ops GPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lhz $rD, $disp($rA)">;
def LMW : DForm_1<46, (ops GPRC:$rD, s16imm:$disp, GPRC:$rA),
                  "lmw $rD, $disp($rA)">;
def LWZ : DForm_1<32, (ops GPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lwz $rD, $disp($rA)">;
def LWZU : DForm_1<35, (ops GPRC:$rD, s16imm:$disp, GPRC:$rA),
                   "lwzu $rD, $disp($rA)">;
}
def ADDI   : DForm_2<14, (ops GPRC:$rD, GPRC:$rA, s16imm:$imm),
                     "addi $rD, $rA, $imm",
                     [(set GPRC:$rD, (add GPRC:$rA, immSExt16:$imm))]>;
def ADDIC  : DForm_2<12, (ops GPRC:$rD, GPRC:$rA, s16imm:$imm),
                     "addic $rD, $rA, $imm",
                     []>;
def ADDICo : DForm_2<13, (ops GPRC:$rD, GPRC:$rA, s16imm:$imm),
                     "addic. $rD, $rA, $imm",
                     []>;
def ADDIS  : DForm_2<15, (ops GPRC:$rD, GPRC:$rA, symbolHi:$imm),
                     "addis $rD, $rA, $imm",
                     [(set GPRC:$rD, (add GPRC:$rA, imm16Shifted:$imm))]>;
def LA     : DForm_2<14, (ops GPRC:$rD, GPRC:$rA, symbolLo:$sym),
                     "la $rD, $sym($rA)",
                     []>;
def MULLI  : DForm_2< 7, (ops GPRC:$rD, GPRC:$rA, s16imm:$imm),
                     "mulli $rD, $rA, $imm",
                     [(set GPRC:$rD, (mul GPRC:$rA, immSExt16:$imm))]>;
def SUBFIC : DForm_2< 8, (ops GPRC:$rD, GPRC:$rA, s16imm:$imm),
                     "subfic $rD, $rA, $imm",
                     [(set GPRC:$rD, (sub immSExt16:$imm, GPRC:$rA))]>;
def LI  : DForm_2_r0<14, (ops GPRC:$rD, s16imm:$imm),
                     "li $rD, $imm",
                     [(set GPRC:$rD, immSExt16:$imm)]>;
def LIS : DForm_2_r0<15, (ops GPRC:$rD, symbolHi:$imm),
                     "lis $rD, $imm",
                     [(set GPRC:$rD, imm16Shifted:$imm)]>;
let isStore = 1 in {
def STMW : DForm_3<47, (ops GPRC:$rS, s16imm:$disp, GPRC:$rA),
                   "stmw $rS, $disp($rA)">;
def STB  : DForm_3<38, (ops GPRC:$rS, symbolLo:$disp, GPRC:$rA),
                   "stb $rS, $disp($rA)">;
def STH  : DForm_3<44, (ops GPRC:$rS, symbolLo:$disp, GPRC:$rA),
                   "sth $rS, $disp($rA)">;
def STW  : DForm_3<36, (ops GPRC:$rS, symbolLo:$disp, GPRC:$rA),
                   "stw $rS, $disp($rA)">;
def STWU : DForm_3<37, (ops GPRC:$rS, s16imm:$disp, GPRC:$rA),
                   "stwu $rS, $disp($rA)">;
}
def ANDIo : DForm_4<28, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "andi. $dst, $src1, $src2",
                    []>, isDOT;
def ANDISo : DForm_4<29, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "andis. $dst, $src1, $src2",
                    []>, isDOT;
def ORI   : DForm_4<24, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "ori $dst, $src1, $src2",
                    [(set GPRC:$dst, (or GPRC:$src1, immZExt16:$src2))]>;
def ORIS  : DForm_4<25, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "oris $dst, $src1, $src2",
                    [(set GPRC:$dst, (or GPRC:$src1, imm16Shifted:$src2))]>;
def XORI  : DForm_4<26, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "xori $dst, $src1, $src2",
                    [(set GPRC:$dst, (xor GPRC:$src1, immZExt16:$src2))]>;
def XORIS : DForm_4<27, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "xoris $dst, $src1, $src2",
                    [(set GPRC:$dst, (xor GPRC:$src1, imm16Shifted:$src2))]>;
def NOP   : DForm_4_zero<24, (ops), "nop">;
def CMPI  : DForm_5<11, (ops CRRC:$crD, i1imm:$L, GPRC:$rA, s16imm:$imm),
                    "cmpi $crD, $L, $rA, $imm">;
def CMPWI : DForm_5_ext<11, (ops CRRC:$crD, GPRC:$rA, s16imm:$imm),
                        "cmpwi $crD, $rA, $imm">;
def CMPDI : DForm_5_ext<11, (ops CRRC:$crD, GPRC:$rA, s16imm:$imm),
                        "cmpdi $crD, $rA, $imm">, isPPC64;
def CMPLI  : DForm_6<10, (ops CRRC:$dst, i1imm:$size, GPRC:$src1, u16imm:$src2),
                     "cmpli $dst, $size, $src1, $src2">;
def CMPLWI : DForm_6_ext<10, (ops CRRC:$dst, GPRC:$src1, u16imm:$src2),
                         "cmplwi $dst, $src1, $src2">;
def CMPLDI : DForm_6_ext<10, (ops CRRC:$dst, GPRC:$src1, u16imm:$src2),
                         "cmpldi $dst, $src1, $src2">, isPPC64;
let isLoad = 1 in {
def LFS : DForm_8<48, (ops FPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lfs $rD, $disp($rA)">;
def LFD : DForm_8<50, (ops FPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lfd $rD, $disp($rA)">;
}
let isStore = 1 in {
def STFS : DForm_9<52, (ops FPRC:$rS, symbolLo:$disp, GPRC:$rA),
                   "stfs $rS, $disp($rA)">;
def STFD : DForm_9<54, (ops FPRC:$rS, symbolLo:$disp, GPRC:$rA),
                   "stfd $rS, $disp($rA)">;
}

// DS-Form instructions.  Load/Store instructions available in PPC-64
//
let isLoad = 1 in {
def LWA  : DSForm_1<58, 2, (ops GPRC:$rT, s16imm:$DS, GPRC:$rA),
                    "lwa $rT, $DS($rA)">, isPPC64;
def LD   : DSForm_2<58, 0, (ops GPRC:$rT, s16imm:$DS, GPRC:$rA),
                    "ld $rT, $DS($rA)">, isPPC64;
}
let isStore = 1 in {
def STD  : DSForm_2<62, 0, (ops GPRC:$rT, s16imm:$DS, GPRC:$rA),
                    "std $rT, $DS($rA)">, isPPC64;
def STDU : DSForm_2<62, 1, (ops GPRC:$rT, s16imm:$DS, GPRC:$rA),
                    "stdu $rT, $DS($rA)">, isPPC64;
}

// X-Form instructions.  Most instructions that perform an operation on a
// register and another register are of this type.
//
let isLoad = 1 in {
def LBZX : XForm_1<31,  87, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lbzx $dst, $base, $index">;
def LHAX : XForm_1<31, 343, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lhax $dst, $base, $index">;
def LHZX : XForm_1<31, 279, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lhzx $dst, $base, $index">;
def LWAX : XForm_1<31, 341, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lwax $dst, $base, $index">, isPPC64;
def LWZX : XForm_1<31,  23, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lwzx $dst, $base, $index">;
def LDX  : XForm_1<31,  21, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "ldx $dst, $base, $index">, isPPC64;
}
def NAND : XForm_6<31, 476, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "nand $rA, $rS, $rB",
                   [(set GPRC:$rA, (not (and GPRC:$rS, GPRC:$rB)))]>;
def AND  : XForm_6<31,  28, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "and $rA, $rS, $rB",
                   [(set GPRC:$rA, (and GPRC:$rS, GPRC:$rB))]>;
def ANDo : XForm_6<31,  28, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "and. $rA, $rS, $rB",
                   []>, isDOT;
def ANDC : XForm_6<31,  60, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "andc $rA, $rS, $rB",
                   [(set GPRC:$rA, (and GPRC:$rS, (not GPRC:$rB)))]>;
def OR   : XForm_6<31, 444, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "or $rA, $rS, $rB",
                   [(set GPRC:$rA, (or GPRC:$rS, GPRC:$rB))]>;
def NOR  : XForm_6<31, 124, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "nor $rA, $rS, $rB",
                   [(set GPRC:$rA, (not (or GPRC:$rS, GPRC:$rB)))]>;
def ORo  : XForm_6<31, 444, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "or. $rA, $rS, $rB",
                   []>, isDOT;
def ORC  : XForm_6<31, 412, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "orc $rA, $rS, $rB",
                   [(set GPRC:$rA, (or GPRC:$rS, (not GPRC:$rB)))]>;
def EQV  : XForm_6<31, 284, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "eqv $rA, $rS, $rB",
                   [(set GPRC:$rA, (not (xor GPRC:$rS, GPRC:$rB)))]>;
def XOR  : XForm_6<31, 316, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "xor $rA, $rS, $rB",
                   [(set GPRC:$rA, (xor GPRC:$rS, GPRC:$rB))]>;                   
def SLD  : XForm_6<31,  27, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "sld $rA, $rS, $rB",
                   []>, isPPC64;
def SLW  : XForm_6<31,  24, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "slw $rA, $rS, $rB",
                   []>;
def SRD  : XForm_6<31, 539, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "srd $rA, $rS, $rB",
                   []>, isPPC64;
def SRW  : XForm_6<31, 536, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "srw $rA, $rS, $rB",
                   []>;
def SRAD : XForm_6<31, 794, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "srad $rA, $rS, $rB",
                   []>, isPPC64;
def SRAW : XForm_6<31, 792, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "sraw $rA, $rS, $rB",
                   []>;
let isStore = 1 in {
def STBX  : XForm_8<31, 215, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stbx $rS, $rA, $rB">;
def STHX  : XForm_8<31, 407, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "sthx $rS, $rA, $rB">;
def STWX  : XForm_8<31, 151, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stwx $rS, $rA, $rB">;
def STWUX : XForm_8<31, 183, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stwux $rS, $rA, $rB">;
def STDX  : XForm_8<31, 149, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stdx $rS, $rA, $rB">, isPPC64;
def STDUX : XForm_8<31, 181, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stdux $rS, $rA, $rB">, isPPC64;
}
def SRAWI : XForm_10<31, 824, (ops GPRC:$rA, GPRC:$rS, u5imm:$SH), 
                     "srawi $rA, $rS, $SH">;
def CNTLZW : XForm_11<31,  26, (ops GPRC:$rA, GPRC:$rS),
                      "cntlzw $rA, $rS",
                      [(set GPRC:$rA, (ctlz GPRC:$rS))]>;
def EXTSB  : XForm_11<31, 954, (ops GPRC:$rA, GPRC:$rS),
                      "extsb $rA, $rS",
                      [(set GPRC:$rA, (sext_inreg GPRC:$rS, i8))]>;
def EXTSH  : XForm_11<31, 922, (ops GPRC:$rA, GPRC:$rS),
                      "extsh $rA, $rS",
                      [(set GPRC:$rA, (sext_inreg GPRC:$rS, i16))]>;
def EXTSW  : XForm_11<31, 986, (ops GPRC:$rA, GPRC:$rS),
                      "extsw $rA, $rS",
                      []>, isPPC64;
def CMP    : XForm_16<31, 0, (ops CRRC:$crD, i1imm:$long, GPRC:$rA, GPRC:$rB),
                      "cmp $crD, $long, $rA, $rB">;
def CMPL   : XForm_16<31, 32, (ops CRRC:$crD, i1imm:$long, GPRC:$rA, GPRC:$rB),
                      "cmpl $crD, $long, $rA, $rB">;
def CMPW   : XForm_16_ext<31, 0, (ops CRRC:$crD, GPRC:$rA, GPRC:$rB),
                          "cmpw $crD, $rA, $rB">;
def CMPD   : XForm_16_ext<31, 0, (ops CRRC:$crD, GPRC:$rA, GPRC:$rB),
                          "cmpd $crD, $rA, $rB">, isPPC64;
def CMPLW  : XForm_16_ext<31, 32, (ops CRRC:$crD, GPRC:$rA, GPRC:$rB),
                          "cmplw $crD, $rA, $rB">;
def CMPLD  : XForm_16_ext<31, 32, (ops CRRC:$crD, GPRC:$rA, GPRC:$rB),
                          "cmpld $crD, $rA, $rB">, isPPC64;
def FCMPO  : XForm_17<63, 32, (ops CRRC:$crD, FPRC:$fA, FPRC:$fB),
                      "fcmpo $crD, $fA, $fB">;
def FCMPU  : XForm_17<63, 0, (ops CRRC:$crD, FPRC:$fA, FPRC:$fB),
                      "fcmpu $crD, $fA, $fB">;
let isLoad = 1 in {
def LFSX   : XForm_25<31, 535, (ops FPRC:$dst, GPRC:$base, GPRC:$index),
                      "lfsx $dst, $base, $index">;
def LFDX   : XForm_25<31, 599, (ops FPRC:$dst, GPRC:$base, GPRC:$index),
                      "lfdx $dst, $base, $index">;
}
def FCFID  : XForm_26<63, 846, (ops FPRC:$frD, FPRC:$frB),
                      "fcfid $frD, $frB">, isPPC64;
def FCTIDZ : XForm_26<63, 815, (ops FPRC:$frD, FPRC:$frB),
                      "fctidz $frD, $frB">, isPPC64;
def FCTIWZ : XForm_26<63, 15, (ops FPRC:$frD, FPRC:$frB),
                      "fctiwz $frD, $frB">;
def FABS   : XForm_26<63, 264, (ops FPRC:$frD, FPRC:$frB),
                      "fabs $frD, $frB">;
def FMR    : XForm_26<63, 72, (ops FPRC:$frD, FPRC:$frB),
                      "fmr $frD, $frB">;
def FNABS  : XForm_26<63, 136, (ops FPRC:$frD, FPRC:$frB),
                      "fnabs $frD, $frB">;
def FNEG   : XForm_26<63, 40, (ops FPRC:$frD, FPRC:$frB),
                      "fneg $frD, $frB">;
def FRSP   : XForm_26<63, 12, (ops FPRC:$frD, FPRC:$frB),
                      "frsp $frD, $frB">;
def FSQRT  : XForm_26<63, 22, (ops FPRC:$frD, FPRC:$frB),
                      "fsqrt $frD, $frB">;
def FSQRTS : XForm_26<59, 22, (ops FPRC:$frD, FPRC:$frB),
                      "fsqrts $frD, $frB">;
                      
let isStore = 1 in {
def STFSX : XForm_28<31, 663, (ops FPRC:$frS, GPRC:$rA, GPRC:$rB),
                     "stfsx $frS, $rA, $rB">;
def STFDX : XForm_28<31, 727, (ops FPRC:$frS, GPRC:$rA, GPRC:$rB),
                     "stfdx $frS, $rA, $rB">;
}

// XL-Form instructions.  condition register logical ops.
//
def MCRF   : XLForm_3<19, 0, (ops CRRC:$BF, CRRC:$BFA),
                      "mcrf $BF, $BFA">;

// XFX-Form instructions.  Instructions that deal with SPRs
//
// Note that although LR should be listed as `8' and CTR as `9' in the SPR
// field, the manual lists the groups of bits as [5-9] = 0, [0-4] = 8 or 9
// which means the SPR value needs to be multiplied by a factor of 32.
def MFCTR : XFXForm_1_ext<31, 339, 288, (ops GPRC:$rT), "mfctr $rT">;
def MFLR  : XFXForm_1_ext<31, 339, 256, (ops GPRC:$rT), "mflr $rT">;
def MFCR  : XFXForm_3<31, 19, (ops GPRC:$rT), "mfcr $rT">;
def MTCRF : XFXForm_5<31, 144, (ops crbitm:$FXM, GPRC:$rS),
                      "mtcrf $FXM, $rS">;
def MFOCRF : XFXForm_5a<31, 19, (ops GPRC:$rT, crbitm:$FXM),
                        "mfcr $rT, $FXM">;
def MTCTR : XFXForm_7_ext<31, 467, 288, (ops GPRC:$rS), "mtctr $rS">;
def MTLR  : XFXForm_7_ext<31, 467, 256, (ops GPRC:$rS), "mtlr $rS">;

// XS-Form instructions.  Just 'sradi'
//
def SRADI  : XSForm_1<31, 413, (ops GPRC:$rA, GPRC:$rS, u6imm:$SH),
                      "sradi $rA, $rS, $SH">, isPPC64;

// XO-Form instructions.  Arithmetic instructions that can set overflow bit
//
def ADD   : XOForm_1<31, 266, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "add $rT, $rA, $rB",
                     [(set GPRC:$rT, (add GPRC:$rA, GPRC:$rB))]>;
def ADDC  : XOForm_1<31, 10, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "addc $rT, $rA, $rB",
                     []>;
def ADDE  : XOForm_1<31, 138, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "adde $rT, $rA, $rB",
                     []>;
def DIVD  : XOForm_1<31, 489, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "divd $rT, $rA, $rB",
                     []>, isPPC64;
def DIVDU : XOForm_1<31, 457, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "divdu $rT, $rA, $rB",
                     []>, isPPC64;
def DIVW  : XOForm_1<31, 491, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "divw $rT, $rA, $rB",
                     [(set GPRC:$rT, (sdiv GPRC:$rA, GPRC:$rB))]>;
def DIVWU : XOForm_1<31, 459, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "divwu $rT, $rA, $rB",
                     [(set GPRC:$rT, (udiv GPRC:$rA, GPRC:$rB))]>;
def MULHW : XOForm_1<31, 75, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "mulhw $rT, $rA, $rB",
                     [(set GPRC:$rT, (mulhs GPRC:$rA, GPRC:$rB))]>;
def MULHWU : XOForm_1<31, 11, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "mulhwu $rT, $rA, $rB",
                     [(set GPRC:$rT, (mulhu GPRC:$rA, GPRC:$rB))]>;
def MULLD : XOForm_1<31, 233, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "mulld $rT, $rA, $rB",
                     []>, isPPC64;
def MULLW : XOForm_1<31, 235, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "mullw $rT, $rA, $rB",
                     [(set GPRC:$rT, (mul GPRC:$rA, GPRC:$rB))]>;
def SUBF  : XOForm_1<31, 40, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "subf $rT, $rA, $rB",
                     [(set GPRC:$rT, (sub GPRC:$rB, GPRC:$rA))]>;
def SUBFC : XOForm_1<31, 8, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "subfc $rT, $rA, $rB",
                     []>;
def SUBFE : XOForm_1<31, 136, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "subfe $rT, $rA, $rB",
                     []>;
def ADDME  : XOForm_3<31, 234, 0, (ops GPRC:$rT, GPRC:$rA),
                      "addme $rT, $rA",
                      []>;
def ADDZE  : XOForm_3<31, 202, 0, (ops GPRC:$rT, GPRC:$rA),
                      "addze $rT, $rA",
                      []>;
def NEG    : XOForm_3<31, 104, 0, (ops GPRC:$rT, GPRC:$rA),
                      "neg $rT, $rA",
                      [(set GPRC:$rT, (ineg GPRC:$rA))]>;
def SUBFZE : XOForm_3<31, 200, 0, (ops GPRC:$rT, GPRC:$rA),
                      "subfze $rT, $rA",
                      []>;

// A-Form instructions.  Most of the instructions executed in the FPU are of
// this type.
//
def FMADD : AForm_1<63, 29, 
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fmadd $FRT, $FRA, $FRC, $FRB">;
def FMADDS : AForm_1<59, 29,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fmadds $FRT, $FRA, $FRC, $FRB">;
def FMSUB : AForm_1<63, 28,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fmsub $FRT, $FRA, $FRC, $FRB">;
def FMSUBS : AForm_1<59, 28,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fmsubs $FRT, $FRA, $FRC, $FRB">;
def FNMADD : AForm_1<63, 31,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fnmadd $FRT, $FRA, $FRC, $FRB">;
def FNMADDS : AForm_1<59, 31,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fnmadds $FRT, $FRA, $FRC, $FRB">;
def FNMSUB : AForm_1<63, 30,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fnmsub $FRT, $FRA, $FRC, $FRB">;
def FNMSUBS : AForm_1<59, 30,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fnmsubs $FRT, $FRA, $FRC, $FRB">;
def FSEL  : AForm_1<63, 23,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fsel $FRT, $FRA, $FRC, $FRB">;
def FADD  : AForm_2<63, 21,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fadd $FRT, $FRA, $FRB">;
def FADDS : AForm_2<59, 21,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fadds $FRT, $FRA, $FRB">;
def FDIV  : AForm_2<63, 18,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fdiv $FRT, $FRA, $FRB">;
def FDIVS : AForm_2<59, 18,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fdivs $FRT, $FRA, $FRB">;
def FMUL  : AForm_3<63, 25,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fmul $FRT, $FRA, $FRB">;
def FMULS : AForm_3<59, 25,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fmuls $FRT, $FRA, $FRB">;
def FSUB  : AForm_2<63, 20,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fsub $FRT, $FRA, $FRB">;
def FSUBS : AForm_2<59, 20,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fsubs $FRT, $FRA, $FRB">;

// M-Form instructions.  rotate and mask instructions.
//
let isTwoAddress = 1, isCommutable = 1 in {
// RLWIMI can be commuted if the rotate amount is zero.
def RLWIMI : MForm_2<20,
                     (ops GPRC:$rA, GPRC:$rSi, GPRC:$rS, u5imm:$SH, u5imm:$MB, 
                      u5imm:$ME), "rlwimi $rA, $rS, $SH, $MB, $ME">;
}
def RLWINM : MForm_2<21,
                     (ops GPRC:$rA, GPRC:$rS, u5imm:$SH, u5imm:$MB, u5imm:$ME),
                     "rlwinm $rA, $rS, $SH, $MB, $ME">;
def RLWINMo : MForm_2<21,
                     (ops GPRC:$rA, GPRC:$rS, u5imm:$SH, u5imm:$MB, u5imm:$ME),
                     "rlwinm. $rA, $rS, $SH, $MB, $ME">, isDOT;
def RLWNM  : MForm_2<23,
                     (ops GPRC:$rA, GPRC:$rS, GPRC:$rB, u5imm:$MB, u5imm:$ME),
                     "rlwnm $rA, $rS, $rB, $MB, $ME">;

// MD-Form instructions.  64 bit rotate instructions.
//
def RLDICL : MDForm_1<30, 0,
                      (ops GPRC:$rA, GPRC:$rS, u6imm:$SH, u6imm:$MB),
                      "rldicl $rA, $rS, $SH, $MB">, isPPC64;
def RLDICR : MDForm_1<30, 1,
                      (ops GPRC:$rA, GPRC:$rS, u6imm:$SH, u6imm:$ME),
                      "rldicr $rA, $rS, $SH, $ME">, isPPC64;

//===----------------------------------------------------------------------===//
// PowerPC Instruction Patterns
//

// Arbitrary immediate support.  Implement in terms of LIS/ORI.
def : Pat<(i32 imm:$imm),
          (ORI (LIS (HI16 imm:$imm)), (LO16 imm:$imm))>;


// Implement the 'not' operation with the NOR instruction.
def NOT : Pat<(not GPRC:$in),
              (NOR GPRC:$in, GPRC:$in)>;

// EQV patterns
def EQV1 : Pat<(xor (not GPRC:$in1), GPRC:$in2),
               (EQV GPRC:$in1, GPRC:$in2)>;
// FIXME: This should be autogenerated from the above due to xor commutativity.
def EQV2 : Pat<(xor GPRC:$in1, (not GPRC:$in2)),
               (EQV GPRC:$in1, GPRC:$in2)>;

// ADD an arbitrary immediate.
def : Pat<(add GPRC:$in, imm:$imm),
          (ADDIS (ADDI GPRC:$in, (LO16 imm:$imm)), (HA16 imm:$imm))>;
// OR an arbitrary immediate.
def : Pat<(or GPRC:$in, imm:$imm),
          (ORIS (ORI GPRC:$in, (LO16 imm:$imm)), (HI16 imm:$imm))>;
// XOR an arbitrary immediate.
def : Pat<(xor GPRC:$in, imm:$imm),
          (XORIS (XORI GPRC:$in, (LO16 imm:$imm)), (HI16 imm:$imm))>;



// Same as above, but using a temporary. FIXME: implement temporaries :)
/*
def : Pattern<(xor GPRC:$in, imm:$imm),
              [(set GPRC:$tmp, (XORI GPRC:$in, (LO16 imm:$imm))),
               (XORIS GPRC:$tmp, (HI16 imm:$imm))]>;
*/


//===----------------------------------------------------------------------===//
// PowerPCInstrInfo Definition
//
def PowerPCInstrInfo : InstrInfo {
  let PHIInst  = PHI;

  let TSFlagsFields = [ "VMX", "PPC64" ];
  let TSFlagsShifts = [ 0, 1 ];

  let isLittleEndianEncoding = 1;
}

