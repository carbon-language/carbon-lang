//===- TargetSelectionDAG.td - Common code for DAG isels ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent interfaces used by SelectionDAG
// instruction selection generators.
//
//===----------------------------------------------------------------------===//

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
class SDTCisVT<int OpNum, ValueType vt> : SDTypeConstraint<OpNum> {
  ValueType VT = vt;
}

class SDTCisPtrTy<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisInt - The specified operand is has integer type.
class SDTCisInt<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisFP - The specified operand is has floating point type.
class SDTCisFP<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisSameAs - The two specified operands have identical types.
class SDTCisSameAs<int OpNum, int OtherOp> : SDTypeConstraint<OpNum> {
  int OtherOperandNum = OtherOp;
}

// SDTCisVTSmallerThanOp - The specified operand is a VT SDNode, and its type is
// smaller than the 'Other' operand.
class SDTCisVTSmallerThanOp<int OpNum, int OtherOp> : SDTypeConstraint<OpNum> {
  int OtherOperandNum = OtherOp;
}

class SDTCisOpSmallerThanOp<int SmallOp, int BigOp> : SDTypeConstraint<SmallOp>{
  int BigOperandNum = BigOp;
}

/// SDTCisIntVectorOfSameSize - This indicates that ThisOp and OtherOp are
/// packed vector types, and that ThisOp is the result of 
/// MVT::getIntVectorWithNumElements with the number of elements that ThisOp
/// has.
class SDTCisIntVectorOfSameSize<int ThisOp, int OtherOp>
  : SDTypeConstraint<ThisOp> {
  int OtherOpNum = OtherOp;
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
def SDTIntLeaf: SDTypeProfile<1, 0, [SDTCisInt<0>]>;      // for 'imm'.
def SDTFPLeaf : SDTypeProfile<1, 0, [SDTCisFP<0>]>;       // for 'fpimm'.
def SDTPtrLeaf: SDTypeProfile<1, 0, [SDTCisPtrTy<0>]>;      // for '&g'.
def SDTOther  : SDTypeProfile<1, 0, [SDTCisVT<0, OtherVT>]>; // for 'vt'.
def SDTUNDEF  : SDTypeProfile<1, 0, []>; // for 'undef'.
def SDTUnaryOp  : SDTypeProfile<1, 1, []>; // bitconvert

def SDTIntBinOp : SDTypeProfile<1, 2, [   // add, and, or, xor, udiv, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisInt<0>
]>;
def SDTIntShiftOp : SDTypeProfile<1, 2, [   // shl, sra, srl
  SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisInt<2>
]>;
def SDTFPBinOp : SDTypeProfile<1, 2, [      // fadd, fmul, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisFP<0>
]>;
def SDTFPSignOp : SDTypeProfile<1, 2, [      // fcopysign.
  SDTCisSameAs<0, 1>, SDTCisFP<0>, SDTCisFP<2>
]>;
def SDTFPTernaryOp : SDTypeProfile<1, 3, [      // fmadd, fnmsub, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>, SDTCisFP<0>
]>;
def SDTIntUnaryOp : SDTypeProfile<1, 1, [   // ctlz
  SDTCisSameAs<0, 1>, SDTCisInt<0>
]>;
def SDTIntExtendOp : SDTypeProfile<1, 1, [  // sext, zext, anyext
  SDTCisInt<0>, SDTCisInt<1>, SDTCisOpSmallerThanOp<1, 0>
]>;
def SDTIntTruncOp  : SDTypeProfile<1, 1, [  // trunc
  SDTCisInt<0>, SDTCisInt<1>, SDTCisOpSmallerThanOp<0, 1>
]>;
def SDTFPUnaryOp  : SDTypeProfile<1, 1, [   // fneg, fsqrt, etc
  SDTCisSameAs<0, 1>, SDTCisFP<0>
]>;
def SDTFPRoundOp  : SDTypeProfile<1, 1, [   // fround
  SDTCisFP<0>, SDTCisFP<1>, SDTCisOpSmallerThanOp<0, 1>
]>;
def SDTFPExtendOp  : SDTypeProfile<1, 1, [   // fextend
  SDTCisFP<0>, SDTCisFP<1>, SDTCisOpSmallerThanOp<1, 0>
]>;
def SDTIntToFPOp : SDTypeProfile<1, 1, [   // [su]int_to_fp 
  SDTCisFP<0>, SDTCisInt<1>
]>;
def SDTFPToIntOp : SDTypeProfile<1, 1, [   // fp_to_[su]int 
  SDTCisInt<0>, SDTCisFP<1>
]>;
def SDTExtInreg : SDTypeProfile<1, 2, [   // sext_inreg
  SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisVT<2, OtherVT>,
  SDTCisVTSmallerThanOp<2, 1>
]>;

def SDTSetCC : SDTypeProfile<1, 3, [ // setcc
  SDTCisInt<0>, SDTCisSameAs<1, 2>, SDTCisVT<3, OtherVT>
]>;

def SDTSelect : SDTypeProfile<1, 3, [ // select 
  SDTCisInt<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<2, 3>
]>;

def SDTSelectCC : SDTypeProfile<1, 5, [ // select_cc
  SDTCisSameAs<1, 2>, SDTCisSameAs<3, 4>, SDTCisSameAs<0, 3>,
  SDTCisVT<5, OtherVT>
]>;

def SDTBr : SDTypeProfile<0, 1, [ // br
  SDTCisVT<0, OtherVT>
]>;

def SDTBrcond : SDTypeProfile<0, 2, [ // brcond
  SDTCisInt<0>, SDTCisVT<1, OtherVT>
]>;

def SDTRet : SDTypeProfile<0, 0, []>; // ret

def SDTLoad : SDTypeProfile<1, 1, [ // load
  SDTCisPtrTy<1>  
]>;

def SDTStore : SDTypeProfile<0, 2, [ // store
  SDTCisPtrTy<1>  
]>;

def SDTExtLoad : SDTypeProfile<1, 3, [  // extload
  SDTCisPtrTy<1>, SDTCisVT<2, OtherVT>, SDTCisVT<3, OtherVT>
]>;
def SDTIntExtLoad : SDTypeProfile<1, 3, [  // sextload, zextload
  SDTCisInt<0>, SDTCisPtrTy<1>, SDTCisVT<2, OtherVT>, SDTCisVT<3, OtherVT>
]>;
def SDTTruncStore : SDTypeProfile<0, 4, [  // truncstore
  SDTCisPtrTy<1>, SDTCisVT<2, OtherVT>, SDTCisVT<3, OtherVT>
]>;

def SDTVecShuffle : SDTypeProfile<1, 3, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, SDTCisIntVectorOfSameSize<3, 0>
]>;

//===----------------------------------------------------------------------===//
// Selection DAG Node Properties.
//
// Note: These are hard coded into tblgen.
//
class SDNodeProperty;
def SDNPCommutative : SDNodeProperty;   // X op Y == Y op X
def SDNPAssociative : SDNodeProperty;   // (X op Y) op Z == X op (Y op Z)
def SDNPHasChain    : SDNodeProperty;   // R/W chain operand and result
def SDNPOutFlag     : SDNodeProperty;   // Write a flag result
def SDNPInFlag      : SDNodeProperty;   // Read a flag operand
def SDNPOptInFlag   : SDNodeProperty;   // Optionally read a flag operand

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
def srcvalue;

def imm        : SDNode<"ISD::Constant"  , SDTIntLeaf , [], "ConstantSDNode">;
def fpimm      : SDNode<"ISD::TargetConstantFP",
                         SDTFPLeaf, [], "ConstantFPSDNode">;
def vt         : SDNode<"ISD::VALUETYPE" , SDTOther   , [], "VTSDNode">;
def bb         : SDNode<"ISD::BasicBlock", SDTOther   , [], "BasicBlockSDNode">;
def cond       : SDNode<"ISD::CONDCODE"  , SDTOther   , [], "CondCodeSDNode">;
def undef      : SDNode<"ISD::UNDEF"     , SDTUNDEF   , []>;
def globaladdr : SDNode<"ISD::GlobalAddress",         SDTPtrLeaf, [],
                        "GlobalAddressSDNode">;
def tglobaladdr : SDNode<"ISD::TargetGlobalAddress",  SDTPtrLeaf, [],
                         "GlobalAddressSDNode">;
def constpool   : SDNode<"ISD::ConstantPool",         SDTPtrLeaf, [],
                         "ConstantPoolSDNode">;
def tconstpool  : SDNode<"ISD::TargetConstantPool",   SDTPtrLeaf, [],
                         "ConstantPoolSDNode">;
def frameindex  : SDNode<"ISD::FrameIndex",           SDTPtrLeaf, [],
                         "FrameIndexSDNode">;
def tframeindex : SDNode<"ISD::TargetFrameIndex",     SDTPtrLeaf, [],
                         "FrameIndexSDNode">;
def externalsym : SDNode<"ISD::ExternalSymbol",       SDTPtrLeaf, [],
                         "ExternalSymbolSDNode">;
def texternalsym: SDNode<"ISD::TargetExternalSymbol", SDTPtrLeaf, [],
                         "ExternalSymbolSDNode">;

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
def srl        : SDNode<"ISD::SRL"       , SDTIntShiftOp>;
def sra        : SDNode<"ISD::SRA"       , SDTIntShiftOp>;
def shl        : SDNode<"ISD::SHL"       , SDTIntShiftOp>;
def rotl       : SDNode<"ISD::ROTL"      , SDTIntShiftOp>;
def rotr       : SDNode<"ISD::ROTR"      , SDTIntShiftOp>;
def and        : SDNode<"ISD::AND"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def or         : SDNode<"ISD::OR"        , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def xor        : SDNode<"ISD::XOR"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def addc       : SDNode<"ISD::ADDC"      , SDTIntBinOp,
                        [SDNPCommutative, SDNPOutFlag]>;
def adde       : SDNode<"ISD::ADDE"      , SDTIntBinOp,
                        [SDNPCommutative, SDNPOutFlag, SDNPInFlag]>;
def subc       : SDNode<"ISD::SUBC"      , SDTIntBinOp,
                        [SDNPOutFlag]>;
def sube       : SDNode<"ISD::SUBE"      , SDTIntBinOp,
                        [SDNPOutFlag, SDNPInFlag]>;
                        
def sext_inreg : SDNode<"ISD::SIGN_EXTEND_INREG", SDTExtInreg>;
def bswap      : SDNode<"ISD::BSWAP"      , SDTIntUnaryOp>;
def ctlz       : SDNode<"ISD::CTLZ"       , SDTIntUnaryOp>;
def cttz       : SDNode<"ISD::CTTZ"       , SDTIntUnaryOp>;
def ctpop      : SDNode<"ISD::CTPOP"      , SDTIntUnaryOp>;
def sext       : SDNode<"ISD::SIGN_EXTEND", SDTIntExtendOp>;
def zext       : SDNode<"ISD::ZERO_EXTEND", SDTIntExtendOp>;
def anyext     : SDNode<"ISD::ANY_EXTEND" , SDTIntExtendOp>;
def trunc      : SDNode<"ISD::TRUNCATE"   , SDTIntTruncOp>;
def bitconvert : SDNode<"ISD::BIT_CONVERT", SDTUnaryOp>;
                        
def fadd       : SDNode<"ISD::FADD"       , SDTFPBinOp, [SDNPCommutative]>;
def fsub       : SDNode<"ISD::FSUB"       , SDTFPBinOp>;
def fmul       : SDNode<"ISD::FMUL"       , SDTFPBinOp, [SDNPCommutative]>;
def fdiv       : SDNode<"ISD::FDIV"       , SDTFPBinOp>;
def frem       : SDNode<"ISD::FREM"       , SDTFPBinOp>;
def fabs       : SDNode<"ISD::FABS"       , SDTFPUnaryOp>;
def fneg       : SDNode<"ISD::FNEG"       , SDTFPUnaryOp>;
def fsqrt      : SDNode<"ISD::FSQRT"      , SDTFPUnaryOp>;
def fsin       : SDNode<"ISD::FSIN"       , SDTFPUnaryOp>;
def fcos       : SDNode<"ISD::FCOS"       , SDTFPUnaryOp>;

def fround     : SDNode<"ISD::FP_ROUND"   , SDTFPRoundOp>;
def fextend    : SDNode<"ISD::FP_EXTEND"  , SDTFPExtendOp>;
def fcopysign  : SDNode<"ISD::FCOPYSIGN"  , SDTFPSignOp>;

def sint_to_fp : SDNode<"ISD::SINT_TO_FP" , SDTIntToFPOp>;
def uint_to_fp : SDNode<"ISD::UINT_TO_FP" , SDTIntToFPOp>;
def fp_to_sint : SDNode<"ISD::FP_TO_SINT" , SDTFPToIntOp>;
def fp_to_uint : SDNode<"ISD::FP_TO_UINT" , SDTFPToIntOp>;

def setcc      : SDNode<"ISD::SETCC"      , SDTSetCC>;
def select     : SDNode<"ISD::SELECT"     , SDTSelect>;
def selectcc   : SDNode<"ISD::SELECT_CC"  , SDTSelectCC>;

def brcond     : SDNode<"ISD::BRCOND"     , SDTBrcond, [SDNPHasChain]>;
def br         : SDNode<"ISD::BR"         , SDTBr,     [SDNPHasChain]>;
def ret        : SDNode<"ISD::RET"        , SDTRet,    [SDNPHasChain]>;

def load       : SDNode<"ISD::LOAD"       , SDTLoad,  [SDNPHasChain]>;
def store      : SDNode<"ISD::STORE"      , SDTStore, [SDNPHasChain]>;

// Do not use sextld and zextld directly. Use sextload and zextload (see
// below) which pass in a dummy srcvalue node which tblgen will skip over.
def sextld     : SDNode<"ISD::SEXTLOAD"   , SDTIntExtLoad, [SDNPHasChain]>;
def zextld     : SDNode<"ISD::ZEXTLOAD"   , SDTIntExtLoad, [SDNPHasChain]>;
def extld      : SDNode<"ISD::EXTLOAD"    , SDTExtLoad,    [SDNPHasChain]>;
def truncst    : SDNode<"ISD::TRUNCSTORE" , SDTTruncStore, [SDNPHasChain]>;

def vector_shuffle : SDNode<"ISD::VECTOR_SHUFFLE", SDTVecShuffle, []>;
def build_vector : SDNode<"ISD::BUILD_VECTOR", SDTypeProfile<1, 0, []>, []>;
def scalar_to_vector : SDNode<"ISD::SCALAR_TO_VECTOR", SDTypeProfile<1, 1, []>,
                              []>;

// Nodes for intrinsics, you should use the intrinsic itself and let tblgen use
// these internally.  Don't reference these directly.
def intrinsic_void : SDNode<"ISD::INTRINSIC_VOID", 
                            SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>,
                            [SDNPHasChain]>;
def intrinsic_w_chain : SDNode<"ISD::INTRINSIC_W_CHAIN", 
                               SDTypeProfile<1, -1, [SDTCisPtrTy<1>]>,
                               [SDNPHasChain]>;
def intrinsic_wo_chain : SDNode<"ISD::INTRINSIC_WO_CHAIN", 
                                SDTypeProfile<1, -1, [SDTCisPtrTy<1>]>, []>;


//===----------------------------------------------------------------------===//
// Selection DAG Condition Codes

class CondCode; // ISD::CondCode enums
def SETOEQ : CondCode; def SETOGT : CondCode;
def SETOGE : CondCode; def SETOLT : CondCode; def SETOLE : CondCode;
def SETONE : CondCode; def SETO   : CondCode; def SETUO  : CondCode;
def SETUEQ : CondCode; def SETUGT : CondCode; def SETUGE : CondCode;
def SETULT : CondCode; def SETULE : CondCode; def SETUNE : CondCode;

def SETEQ : CondCode; def SETGT : CondCode; def SETGE : CondCode;
def SETLT : CondCode; def SETLE : CondCode; def SETNE : CondCode;


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

def vtInt      : PatLeaf<(vt),  [{ return MVT::isInteger(N->getVT()); }]>;
def vtFP       : PatLeaf<(vt),  [{ return MVT::isFloatingPoint(N->getVT()); }]>;

def immAllOnes : PatLeaf<(imm), [{ return N->isAllOnesValue(); }]>;
def immAllOnesV: PatLeaf<(build_vector), [{
  return ISD::isBuildVectorAllOnes(N);
}]>;
def immAllZerosV: PatLeaf<(build_vector), [{
  return ISD::isBuildVectorAllZeros(N);
}]>;

// Other helper fragments.
def not  : PatFrag<(ops node:$in), (xor node:$in, immAllOnes)>;
def vnot : PatFrag<(ops node:$in), (xor node:$in, immAllOnesV)>;
def ineg : PatFrag<(ops node:$in), (sub 0, node:$in)>;

// extending load & truncstore fragments.
def sextload      : PatFrag<(ops node:$ptr, node:$vt),
                            (sextld node:$ptr, srcvalue:$dummy, node:$vt)>;
def zextload      : PatFrag<(ops node:$ptr, node:$vt),
                            (zextld node:$ptr, srcvalue:$dummy, node:$vt)>;
def extload       : PatFrag<(ops node:$ptr, node:$vt),
                            (extld node:$ptr, srcvalue:$dummy, node:$vt)>;
def truncstore    : PatFrag<(ops node:$val, node:$ptr, node:$vt),
                            (truncst node:$val, node:$ptr, srcvalue:$dummy, 
                            node:$vt)>;

// setcc convenience fragments.
def setoeq : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOEQ)>;
def setogt : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOGT)>;
def setoge : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOGE)>;
def setolt : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOLT)>;
def setole : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOLE)>;
def setone : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETONE)>;
def seto   : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETO)>;
def setuo  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUO)>;
def setueq : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUEQ)>;
def setugt : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUGT)>;
def setuge : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUGE)>;
def setult : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETULT)>;
def setule : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETULE)>;
def setune : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUNE)>;
def seteq  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETEQ)>;
def setgt  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETGT)>;
def setge  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETGE)>;
def setlt  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETLT)>;
def setle  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETLE)>;
def setne  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETNE)>;

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
  dag             PatternToMatch = patternToMatch;
  list<dag>       ResultInstrs   = resultInstrs;
  list<Predicate> Predicates     = [];  // See class Instruction in Target.td.
}

// Pat - A simple (but common) form of a pattern, which produces a simple result
// not needing a full list.
class Pat<dag pattern, dag result> : Pattern<pattern, [result]>;

//===----------------------------------------------------------------------===//
// Complex pattern definitions.
//
// Complex patterns, e.g. X86 addressing mode, requires pattern matching code
// in C++. NumOperands is the number of operands returned by the select function;
// SelectFunc is the name of the function used to pattern match the max. pattern;
// RootNodes are the list of possible root nodes of the sub-dags to match.
// e.g. X86 addressing mode - def addr : ComplexPattern<4, "SelectAddr", [add]>;
//
class ComplexPattern<ValueType ty, int numops, string fn, list<SDNode> roots = []> {
  ValueType Ty = ty;
  int NumOperands = numops;
  string SelectFunc = fn;
  list<SDNode> RootNodes = roots;
}

//===----------------------------------------------------------------------===//
// Dwarf support.
//
def SDT_dwarf_loc : SDTypeProfile<0, 3,
                      [SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>]>;
def dwarf_loc : SDNode<"ISD::DEBUG_LOC", SDT_dwarf_loc,[SDNPHasChain]>;

def SDT_dwarf_label : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def dwarf_label : SDNode<"ISD::DEBUG_LABEL", SDT_dwarf_label,[SDNPHasChain]>;



