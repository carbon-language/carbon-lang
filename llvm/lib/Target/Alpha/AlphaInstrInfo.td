//===- AlphaInstrInfo.td - The Alpha Instruction Set -------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

include "AlphaInstrFormats.td"

//********************
//Custom DAG Nodes
//********************

def SDTFPUnaryOpUnC  : SDTypeProfile<1, 1, [
  SDTCisFP<1>, SDTCisFP<0>
]>;
def Alpha_itoft   : SDNode<"AlphaISD::ITOFT_",    SDTIntToFPOp, []>;
def Alpha_ftoit   : SDNode<"AlphaISD::FTOIT_",    SDTFPToIntOp, []>;
def Alpha_cvtqt   : SDNode<"AlphaISD::CVTQT_",    SDTFPUnaryOpUnC, []>;
def Alpha_cvtqs   : SDNode<"AlphaISD::CVTQS_",    SDTFPUnaryOpUnC, []>;
def Alpha_cvttq   : SDNode<"AlphaISD::CVTTQ_"  ,  SDTFPUnaryOp, []>;
def Alpha_gprello : SDNode<"AlphaISD::GPRelLo",   SDTIntBinOp, []>;
def Alpha_gprelhi : SDNode<"AlphaISD::GPRelHi",   SDTIntBinOp, []>;
def Alpha_rellit  : SDNode<"AlphaISD::RelLit",    SDTIntBinOp, []>;

def retflag       : SDNode<"AlphaISD::RET_FLAG", SDTRet,
	                   [SDNPHasChain, SDNPOptInFlag]>;

// These are target-independent nodes, but have target-specific formats.
def SDT_AlphaCallSeq : SDTypeProfile<0, 1, [ SDTCisVT<0, i64> ]>;
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_AlphaCallSeq,
    		           [SDNPHasChain, SDNPOutFlag]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_AlphaCallSeq,
                           [SDNPHasChain, SDNPOutFlag]>;

//********************
//Paterns for matching
//********************
def invX : SDNodeXForm<imm, [{ //invert
  return getI64Imm(~N->getValue());
}]>;
def negX : SDNodeXForm<imm, [{ //negate
  return getI64Imm(~N->getValue() + 1);
}]>;
def SExt32 : SDNodeXForm<imm, [{ //signed extend int to long
  return getI64Imm(((int64_t)N->getValue() << 32) >> 32);
}]>;
def SExt16 : SDNodeXForm<imm, [{ //signed extend int to long
  return getI64Imm(((int64_t)N->getValue() << 48) >> 48);
}]>;
def LL16 : SDNodeXForm<imm, [{ //lda part of constant
  return getI64Imm(get_lda16(N->getValue()));
}]>;
def LH16 : SDNodeXForm<imm, [{ //ldah part of constant (or more if too big)
  return getI64Imm(get_ldah16(N->getValue()));
}]>;
def iZAPX : SDNodeXForm<imm, [{ // get imm to ZAPi
  return getI64Imm(get_zapImm((uint64_t)N->getValue()));
}]>;
def nearP2X : SDNodeXForm<imm, [{
  return getI64Imm(Log2_64(getNearPower2((uint64_t)N->getValue())));
}]>;
def nearP2RemX : SDNodeXForm<imm, [{
  uint64_t x = abs(N->getValue() - getNearPower2((uint64_t)N->getValue()));
  return getI64Imm(Log2_64(x));
}]>;

def immUExt8  : PatLeaf<(imm), [{ //imm fits in 8 bit zero extended field
  return (uint64_t)N->getValue() == (uint8_t)N->getValue();
}]>;
def immUExt8inv  : PatLeaf<(imm), [{ //inverted imm fits in 8 bit zero extended field
  return (uint64_t)~N->getValue() == (uint8_t)~N->getValue();
}], invX>;
def immUExt8neg  : PatLeaf<(imm), [{ //negated imm fits in 8 bit zero extended field
  return ((uint64_t)~N->getValue() + 1) == (uint8_t)((uint64_t)~N->getValue() + 1);
}], negX>;
def immSExt16  : PatLeaf<(imm), [{ //imm fits in 16 bit sign extended field
  return ((int64_t)N->getValue() << 48) >> 48 == (int64_t)N->getValue();
}]>;
def immSExt16int  : PatLeaf<(imm), [{ //(int)imm fits in a 16 bit sign extended field
  return ((int64_t)N->getValue() << 48) >> 48 == ((int64_t)N->getValue() << 32) >> 32;
}], SExt16>;
def immZAP  : PatLeaf<(imm), [{ //imm is good for zapi
  uint64_t build = get_zapImm((uint64_t)N->getValue());
  return build != 0;
}], iZAPX>;
def immFPZ  : PatLeaf<(fpimm), [{ //the only fpconstant nodes are +/- 0.0
  return true;
}]>;
def immRem1 : PatLeaf<(imm), [{
  return N->getValue() - getNearPower2((uint64_t)N->getValue()) == 1;
}]>;
def immRem3 : PatLeaf<(imm), [{
  return N->getValue() - getNearPower2((uint64_t)N->getValue()) == 3;
}]>;
def immRem4 : PatLeaf<(imm), [{
  return N->getValue() - getNearPower2((uint64_t)N->getValue()) == 4;
}]>;
def immRem5 : PatLeaf<(imm), [{
  return N->getValue() - getNearPower2((uint64_t)N->getValue()) == 5;
}]>;
def immRem1n : PatLeaf<(imm), [{
  return getNearPower2((uint64_t)N->getValue()) - N->getValue() == 1;
}]>;
def immRem3n : PatLeaf<(imm), [{
  return getNearPower2((uint64_t)N->getValue()) - N->getValue() == 3;
}]>;
def immRem4n : PatLeaf<(imm), [{
  return getNearPower2((uint64_t)N->getValue()) - N->getValue() == 4;
}]>;
def immRem5n : PatLeaf<(imm), [{
  return getNearPower2((uint64_t)N->getValue()) - N->getValue() == 5;
}]>;
def immRemP2n : PatLeaf<(imm), [{
  return isPowerOf2_64(getNearPower2((uint64_t)N->getValue()) - N->getValue());
}]>;
def immRemP2 : PatLeaf<(imm), [{
  return isPowerOf2_64(N->getValue() - getNearPower2((uint64_t)N->getValue()));
}]>;
def immUExt8ME : PatLeaf<(imm), [{ //use this imm for mulqi
  int64_t d =  abs((int64_t)N->getValue() - (int64_t)getNearPower2((uint64_t)N->getValue()));
  if (isPowerOf2_64(d)) return false;
  switch (d) {
    case 1: case 3: case 5: return false; 
    default: return (uint64_t)N->getValue() == (uint8_t)N->getValue();
  };
}]>;

def intop : PatFrag<(ops node:$op), (sext_inreg node:$op, i32)>;
def add4  : PatFrag<(ops node:$op1, node:$op2),
                    (add (shl node:$op1, 2), node:$op2)>;
def sub4  : PatFrag<(ops node:$op1, node:$op2),
                    (sub (shl node:$op1, 2), node:$op2)>;
def add8  : PatFrag<(ops node:$op1, node:$op2),
                    (add (shl node:$op1, 3), node:$op2)>;
def sub8  : PatFrag<(ops node:$op1, node:$op2),
                    (sub (shl node:$op1, 3), node:$op2)>;


//Pseudo ops for selection

def IDEF_I : PseudoInstAlpha<(ops GPRC:$RA), "#idef $RA",
             [(set GPRC:$RA, (undef))], s_pseudo>;
def IDEF_F32 : PseudoInstAlpha<(ops F4RC:$RA), "#idef $RA",
             [(set F4RC:$RA, (undef))], s_pseudo>;
def IDEF_F64 : PseudoInstAlpha<(ops F8RC:$RA), "#idef $RA",
             [(set F8RC:$RA, (undef))], s_pseudo>;

def WTF : PseudoInstAlpha<(ops variable_ops), "#wtf", [], s_pseudo>;

let isLoad = 1, hasCtrlDep = 1 in {
def ADJUSTSTACKUP : PseudoInstAlpha<(ops s64imm:$amt), "; ADJUP $amt", 
                [(callseq_start imm:$amt)], s_pseudo>;
def ADJUSTSTACKDOWN : PseudoInstAlpha<(ops s64imm:$amt), "; ADJDOWN $amt", 
                [(callseq_end imm:$amt)], s_pseudo>;
}
def ALTENT : PseudoInstAlpha<(ops s64imm:$TARGET), "$$$TARGET..ng:\n", [], s_pseudo>;
def PCLABEL : PseudoInstAlpha<(ops s64imm:$num), "PCMARKER_$num:\n",[], s_pseudo>;
def MEMLABEL : PseudoInstAlpha<(ops s64imm:$i, s64imm:$j, s64imm:$k, s64imm:$m),
         "LSMARKER$$$i$$$j$$$k$$$m:", [], s_pseudo>;


//***********************
//Real instructions
//***********************

//Operation Form:

//conditional moves, int

def CMOVLBC  : OForm4<  0x11, 0x16, "cmovlbc $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (xor GPRC:$RCOND, 1), GPRC:$RTRUE, GPRC:$RFALSE))], s_cmov>;
def CMOVLBS  : OForm4<  0x11, 0x14, "cmovlbs $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (and GPRC:$RCOND, 1), GPRC:$RTRUE, GPRC:$RFALSE))], s_cmov>;
def CMOVEQ   : OForm4<  0x11, 0x24, "cmoveq $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (seteq GPRC:$RCOND, 0), GPRC:$RTRUE, GPRC:$RFALSE))], s_cmov>;
def CMOVGE   : OForm4<  0x11, 0x46, "cmovge $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (setge GPRC:$RCOND, 0), GPRC:$RTRUE, GPRC:$RFALSE))], s_cmov>;
def CMOVGT   : OForm4<  0x11, 0x66, "cmovgt $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (setgt GPRC:$RCOND, 0), GPRC:$RTRUE, GPRC:$RFALSE))], s_cmov>;
def CMOVLE   : OForm4<  0x11, 0x64, "cmovle $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (setle GPRC:$RCOND, 0), GPRC:$RTRUE, GPRC:$RFALSE))], s_cmov>;
def CMOVLT   : OForm4<  0x11, 0x44, "cmovlt $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (setlt GPRC:$RCOND, 0), GPRC:$RTRUE, GPRC:$RFALSE))], s_cmov>;
def CMOVNE   : OForm4<  0x11, 0x26, "cmovne $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (setne GPRC:$RCOND, 0), GPRC:$RTRUE, GPRC:$RFALSE))], s_cmov>;

def CMOVEQi  : OForm4L< 0x11, 0x24, "cmoveq $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (setne GPRC:$RCOND, 0), GPRC:$RFALSE, immUExt8:$RTRUE))], s_cmov>;
def CMOVGEi  : OForm4L< 0x11, 0x46, "cmovge $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (setlt GPRC:$RCOND, 0), GPRC:$RFALSE, immUExt8:$RTRUE))], s_cmov>;
def CMOVGTi  : OForm4L< 0x11, 0x66, "cmovgt $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (setle GPRC:$RCOND, 0), GPRC:$RFALSE, immUExt8:$RTRUE))], s_cmov>;
def CMOVLEi  : OForm4L< 0x11, 0x64, "cmovle $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (setgt GPRC:$RCOND, 0), GPRC:$RFALSE, immUExt8:$RTRUE))], s_cmov>;
def CMOVLTi  : OForm4L< 0x11, 0x44, "cmovlt $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (setge GPRC:$RCOND, 0), GPRC:$RFALSE, immUExt8:$RTRUE))], s_cmov>;
def CMOVNEi  : OForm4L< 0x11, 0x26, "cmovne $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (seteq GPRC:$RCOND, 0), GPRC:$RFALSE, immUExt8:$RTRUE))], s_cmov>;
def CMOVLBCi : OForm4L< 0x11, 0x16, "cmovlbc $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (and GPRC:$RCOND, 1), GPRC:$RFALSE, immUExt8:$RTRUE))], s_cmov>;
def CMOVLBSi : OForm4L< 0x11, 0x14, "cmovlbs $RCOND,$RTRUE,$RDEST",
                [(set GPRC:$RDEST, (select (xor GPRC:$RCOND, 1), GPRC:$RFALSE, immUExt8:$RTRUE))], s_cmov>;


//General pattern for cmov
def : Pat<(select GPRC:$which, GPRC:$src1, GPRC:$src2),
      (CMOVNE GPRC:$src2, GPRC:$src1, GPRC:$which)>;
def : Pat<(select GPRC:$which, GPRC:$src1, immUExt8:$src2),
      (CMOVEQi GPRC:$src1, immUExt8:$src2, GPRC:$which)>;


def ADDL     : OForm< 0x10, 0x00, "addl $RA,$RB,$RC",
                      [(set GPRC:$RC, (intop (add GPRC:$RA, GPRC:$RB)))], s_iadd>;
def ADDLi    : OFormL<0x10, 0x00, "addl $RA,$L,$RC",
                      [(set GPRC:$RC, (intop (add GPRC:$RA, immUExt8:$L)))], s_iadd>;
def ADDQ     : OForm< 0x10, 0x20, "addq $RA,$RB,$RC",
                      [(set GPRC:$RC, (add GPRC:$RA, GPRC:$RB))], s_iadd>;
def ADDQi    : OFormL<0x10, 0x20, "addq $RA,$L,$RC",
                      [(set GPRC:$RC, (add GPRC:$RA, immUExt8:$L))], s_iadd>;
def AND      : OForm< 0x11, 0x00, "and $RA,$RB,$RC",
                      [(set GPRC:$RC, (and GPRC:$RA, GPRC:$RB))], s_ilog>;
def ANDi     : OFormL<0x11, 0x00, "and $RA,$L,$RC",
                      [(set GPRC:$RC, (and GPRC:$RA, immUExt8:$L))], s_ilog>;
def BIC      : OForm< 0x11, 0x08, "bic $RA,$RB,$RC",
                      [(set GPRC:$RC, (and GPRC:$RA, (not GPRC:$RB)))], s_ilog>;
def BICi     : OFormL<0x11, 0x08, "bic $RA,$L,$RC", 
                      [(set GPRC:$RC, (and GPRC:$RA, immUExt8inv:$L))], s_ilog>;
def BIS      : OForm< 0x11, 0x20, "bis $RA,$RB,$RC",
                      [(set GPRC:$RC, (or GPRC:$RA, GPRC:$RB))], s_ilog>;
def BISi     : OFormL<0x11, 0x20, "bis $RA,$L,$RC",
                      [(set GPRC:$RC, (or GPRC:$RA, immUExt8:$L))], s_ilog>;
def CTLZ     : OForm2<0x1C, 0x32, "CTLZ $RB,$RC", 
                      [(set GPRC:$RC, (ctlz GPRC:$RB))], s_imisc>;
def CTPOP    : OForm2<0x1C, 0x30, "CTPOP $RB,$RC", 
                      [(set GPRC:$RC, (ctpop GPRC:$RB))], s_imisc>;
def CTTZ     : OForm2<0x1C, 0x33, "CTTZ $RB,$RC", 
                      [(set GPRC:$RC, (cttz GPRC:$RB))], s_imisc>;
def EQV      : OForm< 0x11, 0x48, "eqv $RA,$RB,$RC",
                      [(set GPRC:$RC, (xor GPRC:$RA, (not GPRC:$RB)))], s_ilog>;
def EQVi     : OFormL<0x11, 0x48, "eqv $RA,$L,$RC", 
                      [(set GPRC:$RC, (xor GPRC:$RA, immUExt8inv:$L))], s_ilog>;
def EXTBL    : OForm< 0x12, 0x06, "EXTBL $RA,$RB,$RC", 
                      [(set GPRC:$RC, (and (srl GPRC:$RA, (shl GPRC:$RB, 3)), 255))], s_ishf>;
def EXTWL    : OForm< 0x12, 0x16, "EXTWL $RA,$RB,$RC", 
                      [(set GPRC:$RC, (and (srl GPRC:$RA, (shl GPRC:$RB, 3)), 65535))], s_ishf>;
def EXTLL    : OForm< 0x12, 0x26, "EXTLL $RA,$RB,$RC", 
                      [(set GPRC:$RC, (and (srl GPRC:$RA, (shl GPRC:$RB, 3)), 4294967295))], s_ishf>;

//def EXTBLi   : OFormL<0x12, 0x06, "EXTBL $RA,$L,$RC", []>; //Extract byte low
//def EXTLH    : OForm< 0x12, 0x6A, "EXTLH $RA,$RB,$RC", []>; //Extract longword high
//def EXTLHi   : OFormL<0x12, 0x6A, "EXTLH $RA,$L,$RC", []>; //Extract longword high
//def EXTLLi   : OFormL<0x12, 0x26, "EXTLL $RA,$L,$RC", []>; //Extract longword low
//def EXTQH    : OForm< 0x12, 0x7A, "EXTQH $RA,$RB,$RC", []>; //Extract quadword high
//def EXTQHi   : OFormL<0x12, 0x7A, "EXTQH $RA,$L,$RC", []>; //Extract quadword high
//def EXTQ     : OForm< 0x12, 0x36, "EXTQ $RA,$RB,$RC", []>; //Extract quadword low
//def EXTQi    : OFormL<0x12, 0x36, "EXTQ $RA,$L,$RC", []>; //Extract quadword low
//def EXTWH    : OForm< 0x12, 0x5A, "EXTWH $RA,$RB,$RC", []>; //Extract word high
//def EXTWHi   : OFormL<0x12, 0x5A, "EXTWH $RA,$L,$RC", []>; //Extract word high
//def EXTWLi   : OFormL<0x12, 0x16, "EXTWL $RA,$L,$RC", []>; //Extract word low

//def IMPLVER  : OForm< 0x11, 0x6C, "IMPLVER $RA,$RB,$RC", []>; //Implementation version
//def IMPLVERi : OFormL<0x11, 0x6C, "IMPLVER $RA,$L,$RC", []>; //Implementation version
//def INSBL    : OForm< 0x12, 0x0B, "INSBL $RA,$RB,$RC", []>; //Insert byte low
//def INSBLi   : OFormL<0x12, 0x0B, "INSBL $RA,$L,$RC", []>; //Insert byte low
//def INSLH    : OForm< 0x12, 0x67, "INSLH $RA,$RB,$RC", []>; //Insert longword high
//def INSLHi   : OFormL<0x12, 0x67, "INSLH $RA,$L,$RC", []>; //Insert longword high
//def INSLL    : OForm< 0x12, 0x2B, "INSLL $RA,$RB,$RC", []>; //Insert longword low
//def INSLLi   : OFormL<0x12, 0x2B, "INSLL $RA,$L,$RC", []>; //Insert longword low
//def INSQH    : OForm< 0x12, 0x77, "INSQH $RA,$RB,$RC", []>; //Insert quadword high
//def INSQHi   : OFormL<0x12, 0x77, "INSQH $RA,$L,$RC", []>; //Insert quadword high
//def INSQL    : OForm< 0x12, 0x3B, "INSQL $RA,$RB,$RC", []>; //Insert quadword low
//def INSQLi   : OFormL<0x12, 0x3B, "INSQL $RA,$L,$RC", []>; //Insert quadword low
//def INSWH    : OForm< 0x12, 0x57, "INSWH $RA,$RB,$RC", []>; //Insert word high
//def INSWHi   : OFormL<0x12, 0x57, "INSWH $RA,$L,$RC", []>; //Insert word high
//def INSWL    : OForm< 0x12, 0x1B, "INSWL $RA,$RB,$RC", []>; //Insert word low
//def INSWLi   : OFormL<0x12, 0x1B, "INSWL $RA,$L,$RC", []>; //Insert word low
//def MSKBL    : OForm< 0x12, 0x02, "MSKBL $RA,$RB,$RC", []>; //Mask byte low
//def MSKBLi   : OFormL<0x12, 0x02, "MSKBL $RA,$L,$RC", []>; //Mask byte low
//def MSKLH    : OForm< 0x12, 0x62, "MSKLH $RA,$RB,$RC", []>; //Mask longword high
//def MSKLHi   : OFormL<0x12, 0x62, "MSKLH $RA,$L,$RC", []>; //Mask longword high
//def MSKLL    : OForm< 0x12, 0x22, "MSKLL $RA,$RB,$RC", []>; //Mask longword low
//def MSKLLi   : OFormL<0x12, 0x22, "MSKLL $RA,$L,$RC", []>; //Mask longword low
//def MSKQH    : OForm< 0x12, 0x72, "MSKQH $RA,$RB,$RC", []>; //Mask quadword high
//def MSKQHi   : OFormL<0x12, 0x72, "MSKQH $RA,$L,$RC", []>; //Mask quadword high
//def MSKQL    : OForm< 0x12, 0x32, "MSKQL $RA,$RB,$RC", []>; //Mask quadword low
//def MSKQLi   : OFormL<0x12, 0x32, "MSKQL $RA,$L,$RC", []>; //Mask quadword low
//def MSKWH    : OForm< 0x12, 0x52, "MSKWH $RA,$RB,$RC", []>; //Mask word high
//def MSKWHi   : OFormL<0x12, 0x52, "MSKWH $RA,$L,$RC", []>; //Mask word high
//def MSKWL    : OForm< 0x12, 0x12, "MSKWL $RA,$RB,$RC", []>; //Mask word low
//def MSKWLi   : OFormL<0x12, 0x12, "MSKWL $RA,$L,$RC", []>; //Mask word low

def MULL     : OForm< 0x13, 0x00, "mull $RA,$RB,$RC",
                      [(set GPRC:$RC, (intop (mul GPRC:$RA, GPRC:$RB)))], s_imul>;
def MULLi    : OFormL<0x13, 0x00, "mull $RA,$L,$RC",
                      [(set GPRC:$RC, (intop (mul GPRC:$RA, immUExt8:$L)))], s_imul>;
def MULQ     : OForm< 0x13, 0x20, "mulq $RA,$RB,$RC",
                      [(set GPRC:$RC, (mul GPRC:$RA, GPRC:$RB))], s_imul>;
def MULQi    : OFormL<0x13, 0x20, "mulq $RA,$L,$RC",
                      [(set GPRC:$RC, (mul GPRC:$RA, immUExt8ME:$L))], s_imul>;
def ORNOT    : OForm< 0x11, 0x28, "ornot $RA,$RB,$RC",
                      [(set GPRC:$RC, (or GPRC:$RA, (not GPRC:$RB)))], s_ilog>;
def ORNOTi   : OFormL<0x11, 0x28, "ornot $RA,$L,$RC", 
                      [(set GPRC:$RC, (or GPRC:$RA, immUExt8inv:$L))], s_ilog>;
def S4ADDL   : OForm< 0x10, 0x02, "s4addl $RA,$RB,$RC", 
                      [(set GPRC:$RC, (intop (add4 GPRC:$RA, GPRC:$RB)))], s_iadd>;
def S4ADDLi  : OFormL<0x10, 0x02, "s4addl $RA,$L,$RC", 
                      [(set GPRC:$RC, (intop (add4 GPRC:$RA, immUExt8:$L)))], s_iadd>;
def S4ADDQ   : OForm< 0x10, 0x22, "s4addq $RA,$RB,$RC", 
                      [(set GPRC:$RC, (add4 GPRC:$RA, GPRC:$RB))], s_iadd>;
def S4ADDQi  : OFormL<0x10, 0x22, "s4addq $RA,$L,$RC", 
                      [(set GPRC:$RC, (add4 GPRC:$RA, immUExt8:$L))], s_iadd>;
def S4SUBL   : OForm< 0x10, 0x0B, "s4subl $RA,$RB,$RC",
                      [(set GPRC:$RC, (intop (sub4 GPRC:$RA, GPRC:$RB)))], s_iadd>;
def S4SUBLi  : OFormL<0x10, 0x0B, "s4subl $RA,$L,$RC",
                      [(set GPRC:$RC, (intop (sub4 GPRC:$RA, immUExt8:$L)))], s_iadd>;
def S4SUBQ   : OForm< 0x10, 0x2B, "s4subq $RA,$RB,$RC", 
                      [(set GPRC:$RC, (sub4 GPRC:$RA, GPRC:$RB))], s_iadd>;
def S4SUBQi  : OFormL<0x10, 0x2B, "s4subq $RA,$L,$RC", 
                      [(set GPRC:$RC, (sub4 GPRC:$RA, immUExt8:$L))], s_iadd>;
def S8ADDL   : OForm< 0x10, 0x12, "s8addl $RA,$RB,$RC", 
                      [(set GPRC:$RC, (intop (add8 GPRC:$RA, GPRC:$RB)))], s_iadd>;
def S8ADDLi  : OFormL<0x10, 0x12, "s8addl $RA,$L,$RC", 
                      [(set GPRC:$RC, (intop (add8 GPRC:$RA, immUExt8:$L)))], s_iadd>;
def S8ADDQ   : OForm< 0x10, 0x32, "s8addq $RA,$RB,$RC", 
                      [(set GPRC:$RC, (add8 GPRC:$RA, GPRC:$RB))], s_iadd>;
def S8ADDQi  : OFormL<0x10, 0x32, "s8addq $RA,$L,$RC", 
                      [(set GPRC:$RC, (add8 GPRC:$RA, immUExt8:$L))], s_iadd>;
def S8SUBL   : OForm< 0x10, 0x1B, "s8subl $RA,$RB,$RC", 
                      [(set GPRC:$RC, (intop (sub8 GPRC:$RA, GPRC:$RB)))], s_iadd>;
def S8SUBLi  : OFormL<0x10, 0x1B, "s8subl $RA,$L,$RC", 
                      [(set GPRC:$RC, (intop (add8 GPRC:$RA, immUExt8neg:$L)))], s_iadd>;
def S8SUBQ   : OForm< 0x10, 0x3B, "s8subq $RA,$RB,$RC", 
                      [(set GPRC:$RC, (sub8 GPRC:$RA, GPRC:$RB))], s_iadd>;
def S8SUBQi  : OFormL<0x10, 0x3B, "s8subq $RA,$L,$RC", 
                      [(set GPRC:$RC, (add8 GPRC:$RA, immUExt8neg:$L))], s_iadd>;
def SEXTB    : OForm2<0x1C, 0x00, "sextb $RB,$RC", 
                      [(set GPRC:$RC, (sext_inreg GPRC:$RB, i8))], s_ishf>;
def SEXTW    : OForm2<0x1C, 0x01, "sextw $RB,$RC", 
                      [(set GPRC:$RC, (sext_inreg GPRC:$RB, i16))], s_ishf>;
def SL       : OForm< 0x12, 0x39, "sll $RA,$RB,$RC",
                      [(set GPRC:$RC, (shl GPRC:$RA, GPRC:$RB))], s_ishf>;
def SLi      : OFormL<0x12, 0x39, "sll $RA,$L,$RC",
                      [(set GPRC:$RC, (shl GPRC:$RA, immUExt8:$L))], s_ishf>;
def SRA      : OForm< 0x12, 0x3C, "sra $RA,$RB,$RC",
                      [(set GPRC:$RC, (sra GPRC:$RA, GPRC:$RB))], s_ishf>;
def SRAi     : OFormL<0x12, 0x3C, "sra $RA,$L,$RC",
                      [(set GPRC:$RC, (sra GPRC:$RA, immUExt8:$L))], s_ishf>;
def SRL      : OForm< 0x12, 0x34, "srl $RA,$RB,$RC",
                      [(set GPRC:$RC, (srl GPRC:$RA, GPRC:$RB))], s_ishf>;
def SRLi     : OFormL<0x12, 0x34, "srl $RA,$L,$RC",
                      [(set GPRC:$RC, (srl GPRC:$RA, immUExt8:$L))], s_ishf>;
def SUBL     : OForm< 0x10, 0x09, "subl $RA,$RB,$RC",
                      [(set GPRC:$RC, (intop (sub GPRC:$RA, GPRC:$RB)))], s_iadd>;
def SUBLi    : OFormL<0x10, 0x09, "subl $RA,$L,$RC",
                      [(set GPRC:$RC, (intop (add GPRC:$RA, immUExt8neg:$L)))], s_iadd>;
def SUBQ     : OForm< 0x10, 0x29, "subq $RA,$RB,$RC",
                      [(set GPRC:$RC, (sub GPRC:$RA, GPRC:$RB))], s_iadd>;
def SUBQi    : OFormL<0x10, 0x29, "subq $RA,$L,$RC",
                      [(set GPRC:$RC, (add GPRC:$RA, immUExt8neg:$L))], s_iadd>;
def UMULH    : OForm< 0x13, 0x30, "umulh $RA,$RB,$RC",
                      [(set GPRC:$RC, (mulhu GPRC:$RA, GPRC:$RB))], s_imul>;                     
def UMULHi   : OFormL<0x13, 0x30, "umulh $RA,$L,$RC", 
                      [(set GPRC:$RC, (mulhu GPRC:$RA, immUExt8:$L))], s_imul>;
def XOR      : OForm< 0x11, 0x40, "xor $RA,$RB,$RC",
                      [(set GPRC:$RC, (xor GPRC:$RA, GPRC:$RB))], s_ilog>;
def XORi     : OFormL<0x11, 0x40, "xor $RA,$L,$RC",
                      [(set GPRC:$RC, (xor GPRC:$RA, immUExt8:$L))], s_ilog>;
//FIXME: what to do about zap? the cases it catches are very complex
def ZAP      : OForm< 0x12, 0x30, "zap $RA,$RB,$RC", [], s_ishf>; //Zero bytes
//ZAPi is useless give ZAPNOTi
def ZAPi     : OFormL<0x12, 0x30, "zap $RA,$L,$RC", [], s_ishf>; //Zero bytes
//FIXME: what to do about zapnot? see ZAP :)
def ZAPNOT   : OForm< 0x12, 0x31, "zapnot $RA,$RB,$RC", [], s_ishf>; //Zero bytes not
def ZAPNOTi  : OFormL<0x12, 0x31, "zapnot $RA,$L,$RC", 
                      [(set GPRC:$RC, (and GPRC:$RA, immZAP:$L))], s_ishf>; 

//Comparison, int
//So this is a waste of what this instruction can do, but it still saves something
def CMPBGE  : OForm< 0x10, 0x0F, "cmpbge $RA,$RB,$RC", 
                     [(set GPRC:$RC, (setuge (and GPRC:$RA, 255), (and GPRC:$RB, 255)))], s_ilog>;
def CMPBGEi : OFormL<0x10, 0x0F, "cmpbge $RA,$L,$RC",
                     [(set GPRC:$RC, (setuge (and GPRC:$RA, 255), immUExt8:$L))], s_ilog>;
def CMPEQ   : OForm< 0x10, 0x2D, "cmpeq $RA,$RB,$RC", 
                     [(set GPRC:$RC, (seteq GPRC:$RA, GPRC:$RB))], s_iadd>;
def CMPEQi  : OFormL<0x10, 0x2D, "cmpeq $RA,$L,$RC", 
                     [(set GPRC:$RC, (seteq GPRC:$RA, immUExt8:$L))], s_iadd>;
def CMPLE   : OForm< 0x10, 0x6D, "cmple $RA,$RB,$RC", 
                     [(set GPRC:$RC, (setle GPRC:$RA, GPRC:$RB))], s_iadd>;
def CMPLEi  : OFormL<0x10, 0x6D, "cmple $RA,$L,$RC",
                     [(set GPRC:$RC, (setle GPRC:$RA, immUExt8:$L))], s_iadd>;
def CMPLT   : OForm< 0x10, 0x4D, "cmplt $RA,$RB,$RC",
                     [(set GPRC:$RC, (setlt GPRC:$RA, GPRC:$RB))], s_iadd>;
def CMPLTi  : OFormL<0x10, 0x4D, "cmplt $RA,$L,$RC",
                     [(set GPRC:$RC, (setlt GPRC:$RA, immUExt8:$L))], s_iadd>;
def CMPULE  : OForm< 0x10, 0x3D, "cmpule $RA,$RB,$RC",
                     [(set GPRC:$RC, (setule GPRC:$RA, GPRC:$RB))], s_iadd>;
def CMPULEi : OFormL<0x10, 0x3D, "cmpule $RA,$L,$RC",
                     [(set GPRC:$RC, (setule GPRC:$RA, immUExt8:$L))], s_iadd>;
def CMPULT  : OForm< 0x10, 0x1D, "cmpult $RA,$RB,$RC",
                     [(set GPRC:$RC, (setult GPRC:$RA, GPRC:$RB))], s_iadd>;
def CMPULTi : OFormL<0x10, 0x1D, "cmpult $RA,$L,$RC", 
                      [(set GPRC:$RC, (setult GPRC:$RA, immUExt8:$L))], s_iadd>;

//Patterns for unsupported int comparisons
def : Pat<(setueq GPRC:$X, GPRC:$Y), (CMPEQ GPRC:$X, GPRC:$Y)>;
def : Pat<(setueq GPRC:$X, immUExt8:$Y), (CMPEQi GPRC:$X, immUExt8:$Y)>;

def : Pat<(setugt GPRC:$X, GPRC:$Y), (CMPULT GPRC:$Y, GPRC:$X)>;
def : Pat<(setugt immUExt8:$X, GPRC:$Y), (CMPULTi GPRC:$Y, immUExt8:$X)>;

def : Pat<(setuge GPRC:$X, GPRC:$Y), (CMPULE GPRC:$Y, GPRC:$X)>;
def : Pat<(setuge immUExt8:$X, GPRC:$Y), (CMPULEi GPRC:$Y, immUExt8:$X)>;

def : Pat<(setgt GPRC:$X, GPRC:$Y), (CMPLT GPRC:$Y, GPRC:$X)>;
def : Pat<(setgt immUExt8:$X, GPRC:$Y), (CMPLTi GPRC:$Y, immUExt8:$X)>;

def : Pat<(setge GPRC:$X, GPRC:$Y), (CMPLE GPRC:$Y, GPRC:$X)>;
def : Pat<(setge immUExt8:$X, GPRC:$Y), (CMPLEi GPRC:$Y, immUExt8:$X)>;

def : Pat<(setne GPRC:$X, GPRC:$Y), (CMPEQi (CMPEQ GPRC:$X, GPRC:$Y), 0)>;
def : Pat<(setne GPRC:$X, immUExt8:$Y), (CMPEQi (CMPEQi GPRC:$X, immUExt8:$Y), 0)>;

def : Pat<(setune GPRC:$X, GPRC:$Y), (CMPEQi (CMPEQ GPRC:$X, GPRC:$Y), 0)>;
def : Pat<(setune GPRC:$X, immUExt8:$Y), (CMPEQi (CMPEQ GPRC:$X, immUExt8:$Y), 0)>;


let isReturn = 1, isTerminator = 1, noResults = 1, Ra = 31, Rb = 26, disp = 1, Uses = [R26] in {
  def RETDAG : MbrForm< 0x1A, 0x02, (ops), "ret $$31,($$26),1", s_jsr>; //Return from subroutine
  def RETDAGp : MbrpForm< 0x1A, 0x02, (ops), "ret $$31,($$26),1", [(retflag)], s_jsr>; //Return from subroutine
}

def JMP : MbrForm< 0x1A, 0x00, (ops GPRC:$RD, GPRC:$RS, GPRC:$DISP), "jmp $RD,($RS),$DISP", s_jsr>; //Jump
let isCall = 1, noResults = 1, Ra = 26,
    Defs = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R16, R17, R18, R19,
            R20, R21, R22, R23, R24, R25, R26, R27, R28, R29,
            F0, F1,
            F10, F11, F12, F13, F14, F15, F16, F17, F18, F19,
            F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30], Uses = [R29] in {
    def BSR : BFormD<0x34, "bsr $$26,$$$DISP..ng", [], s_jsr>; //Branch to subroutine
}
let isCall = 1, noResults = 1, Ra = 26, Rb = 27, disp = 0,
    Defs = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R16, R17, R18, R19,
            R20, R21, R22, R23, R24, R25, R26, R27, R28, R29,
            F0, F1,
            F10, F11, F12, F13, F14, F15, F16, F17, F18, F19,
            F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30], Uses = [R27, R29] in {
    def JSR : MbrForm< 0x1A, 0x01, (ops ), "jsr $$26,($$27),0", s_jsr>; //Jump to subroutine
}

let isCall = 1, noResults = 1, Ra = 23, Rb = 27, disp = 0,
    Defs = [R23, R24, R25, R27, R28], Uses = [R24, R25, R27] in
  def JSRs : MbrForm< 0x1A, 0x01, (ops ), "jsr $$23,($$27),0", s_jsr>; //Jump to div or rem


def JSR_COROUTINE : MbrForm< 0x1A, 0x03, (ops GPRC:$RD, GPRC:$RS, s14imm:$DISP), "jsr_coroutine $RD,($RS),$DISP", s_jsr>; //Jump to subroutine return

let OperandList = (ops GPRC:$RA, s64imm:$DISP, GPRC:$RB) in {
def LDQ   : MForm<0x29, 0, 1, "ldq $RA,$DISP($RB)",
                 [(set GPRC:$RA, (load (add GPRC:$RB, immSExt16:$DISP)))], s_ild>;
def LDQr  : MForm<0x29, 0, 1, "ldq $RA,$DISP($RB)\t\t!gprellow",
                 [(set GPRC:$RA, (load (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB)))], s_ild>;
def LDL   : MForm<0x28, 0, 1, "ldl $RA,$DISP($RB)",
                 [(set GPRC:$RA, (sextload (add GPRC:$RB, immSExt16:$DISP), i32))], s_ild>;
def LDLr  : MForm<0x28, 0, 1, "ldl $RA,$DISP($RB)\t\t!gprellow",
                 [(set GPRC:$RA, (sextload (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB), i32))], s_ild>;
def LDBU  : MForm<0x0A, 0, 1, "ldbu $RA,$DISP($RB)",
                 [(set GPRC:$RA, (zextload (add GPRC:$RB, immSExt16:$DISP), i8))], s_ild>;
def LDBUr : MForm<0x0A, 0, 1, "ldbu $RA,$DISP($RB)\t\t!gprellow",
                 [(set GPRC:$RA, (zextload (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB), i8))], s_ild>;
def LDWU  : MForm<0x0C, 0, 1, "ldwu $RA,$DISP($RB)",
                 [(set GPRC:$RA, (zextload (add GPRC:$RB, immSExt16:$DISP), i16))], s_ild>;
def LDWUr : MForm<0x0C, 0, 1, "ldwu $RA,$DISP($RB)\t\t!gprellow",
                 [(set GPRC:$RA, (zextload (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB), i16))], s_ild>;
def STB   : MForm<0x0E, 1, 0, "stb $RA,$DISP($RB)",
		 [(truncstore GPRC:$RA, (add GPRC:$RB, immSExt16:$DISP), i8)], s_ist>;
def STBr  : MForm<0x0E, 1, 0, "stb $RA,$DISP($RB)\t\t!gprellow",
		 [(truncstore GPRC:$RA, (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB), i8)], s_ist>;
def STW   : MForm<0x0D, 1, 0, "stw $RA,$DISP($RB)",
		 [(truncstore GPRC:$RA, (add GPRC:$RB, immSExt16:$DISP), i16)], s_ist>;
def STWr  : MForm<0x0D, 1, 0, "stw $RA,$DISP($RB)\t\t!gprellow",
		 [(truncstore GPRC:$RA, (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB), i16)], s_ist>;
def STL   : MForm<0x2C, 1, 0, "stl $RA,$DISP($RB)",
		 [(truncstore GPRC:$RA, (add GPRC:$RB, immSExt16:$DISP), i32)], s_ist>;
def STLr  : MForm<0x2C, 1, 0, "stl $RA,$DISP($RB)\t\t!gprellow",
		 [(truncstore GPRC:$RA, (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB), i32)], s_ist>;
def STQ   : MForm<0x2D, 1, 0, "stq $RA,$DISP($RB)",
		 [(store GPRC:$RA, (add GPRC:$RB, immSExt16:$DISP))], s_ist>;
def STQr  : MForm<0x2D, 1, 0, "stq $RA,$DISP($RB)\t\t!gprellow",
		 [(store GPRC:$RA, (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB))], s_ist>;

//Load address
def LDA   : MForm<0x08, 0, 0, "lda $RA,$DISP($RB)",
                 [(set GPRC:$RA, (add GPRC:$RB, immSExt16:$DISP))], s_lda>;
def LDAr  : MForm<0x08, 0, 0, "lda $RA,$DISP($RB)\t\t!gprellow",
                 [(set GPRC:$RA, (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB))], s_lda>;  //Load address
def LDAH  : MForm<0x09, 0, 0, "ldah $RA,$DISP($RB)",
                 [], s_lda>;  //Load address high
def LDAHr : MForm<0x09, 0, 0, "ldah $RA,$DISP($RB)\t\t!gprelhigh",
                 [(set GPRC:$RA, (Alpha_gprelhi tglobaladdr:$DISP, GPRC:$RB))], s_lda>;  //Load address high
}

let OperandList = (ops F4RC:$RA, s64imm:$DISP, GPRC:$RB) in {
def STS  : MForm<0x26, 1, 0, "sts $RA,$DISP($RB)",
		[(store F4RC:$RA, (add GPRC:$RB, immSExt16:$DISP))], s_fst>;
def STSr : MForm<0x26, 1, 0, "sts $RA,$DISP($RB)\t\t!gprellow",
		[(store F4RC:$RA, (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB))], s_fst>;
def LDS  : MForm<0x22, 0, 1, "lds $RA,$DISP($RB)",
		[(set F4RC:$RA, (load (add GPRC:$RB, immSExt16:$DISP)))], s_fld>;
def LDSr : MForm<0x22, 0, 1, "lds $RA,$DISP($RB)\t\t!gprellow",
		[(set F4RC:$RA, (load (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB)))], s_fld>;
}
let OperandList = (ops F8RC:$RA, s64imm:$DISP, GPRC:$RB) in {
def STT  : MForm<0x27, 1, 0, "stt $RA,$DISP($RB)",
		[(store F8RC:$RA, (add GPRC:$RB, immSExt16:$DISP))], s_fst>;
def STTr : MForm<0x27, 1, 0, "stt $RA,$DISP($RB)\t\t!gprellow",
		[(store F8RC:$RA, (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB))], s_fst>;
def LDT  : MForm<0x23, 0, 1, "ldt $RA,$DISP($RB)",
		[(set F8RC:$RA, (load (add GPRC:$RB, immSExt16:$DISP)))], s_fld>;
def LDTr : MForm<0x23, 0, 1, "ldt $RA,$DISP($RB)\t\t!gprellow",
		[(set F8RC:$RA, (load (Alpha_gprello tglobaladdr:$DISP, GPRC:$RB)))], s_fld>;
}


//constpool rels
def : Pat<(i64 (load (Alpha_gprello tconstpool:$DISP, GPRC:$RB))),
          (LDQr tconstpool:$DISP, GPRC:$RB)>;
def : Pat<(i64 (sextload (Alpha_gprello tconstpool:$DISP, GPRC:$RB), i32)),
          (LDLr tconstpool:$DISP, GPRC:$RB)>;
def : Pat<(i64 (zextload (Alpha_gprello tconstpool:$DISP, GPRC:$RB), i8)),
          (LDBUr tconstpool:$DISP, GPRC:$RB)>;
def : Pat<(i64 (zextload (Alpha_gprello tconstpool:$DISP, GPRC:$RB), i16)),
          (LDWUr tconstpool:$DISP, GPRC:$RB)>;
def : Pat<(i64 (Alpha_gprello tconstpool:$DISP, GPRC:$RB)),
          (LDAr tconstpool:$DISP, GPRC:$RB)>;
def : Pat<(i64 (Alpha_gprelhi tconstpool:$DISP, GPRC:$RB)),
          (LDAHr tconstpool:$DISP, GPRC:$RB)>;
def : Pat<(f32 (load (Alpha_gprello tconstpool:$DISP, GPRC:$RB))),
          (LDSr tconstpool:$DISP, GPRC:$RB)>;
def : Pat<(f64 (load (Alpha_gprello tconstpool:$DISP, GPRC:$RB))),
          (LDTr tconstpool:$DISP, GPRC:$RB)>;


//misc ext patterns
def : Pat<(i64 (extload (add GPRC:$RB, immSExt16:$DISP), i8)),
          (LDBU   immSExt16:$DISP, GPRC:$RB)>;
def : Pat<(i64 (extload (add GPRC:$RB, immSExt16:$DISP), i16)),
          (LDWU  immSExt16:$DISP, GPRC:$RB)>;
def : Pat<(i64 (extload (add GPRC:$RB, immSExt16:$DISP), i32)),
          (LDL   immSExt16:$DISP, GPRC:$RB)>;

//0 disp patterns
def : Pat<(i64 (load GPRC:$addr)),
          (LDQ  0, GPRC:$addr)>;
def : Pat<(f64 (load GPRC:$addr)),
          (LDT  0, GPRC:$addr)>;
def : Pat<(f32 (load GPRC:$addr)),
          (LDS  0, GPRC:$addr)>;
def : Pat<(i64 (sextload GPRC:$addr, i32)),
          (LDL  0, GPRC:$addr)>;
def : Pat<(i64 (zextload GPRC:$addr, i16)),
          (LDWU 0, GPRC:$addr)>;
def : Pat<(i64 (zextload GPRC:$addr, i8)),
          (LDBU 0, GPRC:$addr)>;
def : Pat<(i64 (extload GPRC:$addr, i8)),
          (LDBU 0, GPRC:$addr)>;
def : Pat<(i64 (extload GPRC:$addr, i16)),
          (LDWU 0, GPRC:$addr)>;
def : Pat<(i64 (extload GPRC:$addr, i32)),
          (LDL  0, GPRC:$addr)>;

def : Pat<(store GPRC:$DATA, GPRC:$addr),
          (STQ  GPRC:$DATA, 0, GPRC:$addr)>;
def : Pat<(store F8RC:$DATA, GPRC:$addr),
          (STT  F8RC:$DATA, 0, GPRC:$addr)>;
def : Pat<(store F4RC:$DATA, GPRC:$addr),
          (STS  F4RC:$DATA, 0, GPRC:$addr)>;
def : Pat<(truncstore GPRC:$DATA, GPRC:$addr, i32),
          (STL  GPRC:$DATA, 0, GPRC:$addr)>;
def : Pat<(truncstore GPRC:$DATA, GPRC:$addr, i16),
          (STW GPRC:$DATA, 0, GPRC:$addr)>;
def : Pat<(truncstore GPRC:$DATA, GPRC:$addr, i8),
          (STB GPRC:$DATA, 0, GPRC:$addr)>;


//load address, rellocated gpdist form
let OperandList = (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB, s16imm:$NUM) in {
def LDAg  : MForm<0x08, 0, 1, "lda $RA,0($RB)\t\t!gpdisp!$NUM", [], s_lda>;  //Load address
def LDAHg : MForm<0x09, 0, 1, "ldah $RA,0($RB)\t\t!gpdisp!$NUM", [], s_lda>;  //Load address
}

//Load quad, rellocated literal form
let OperandList = (ops GPRC:$RA, s64imm:$DISP, GPRC:$RB) in 
def LDQl : MForm<0x29, 0, 1, "ldq $RA,$DISP($RB)\t\t!literal",
                 [(set GPRC:$RA, (Alpha_rellit tglobaladdr:$DISP, GPRC:$RB))], s_ild>;
def : Pat<(Alpha_rellit texternalsym:$ext, GPRC:$RB),
          (LDQl texternalsym:$ext, GPRC:$RB)>;


def RPCC : MfcForm<0x18, 0xC000, "rpcc $RA", s_rpcc>; //Read process cycle counter

//Basic Floating point ops

//Floats

let OperandList = (ops F4RC:$RC, F4RC:$RB), Fa = 31 in 
def SQRTS : FPForm<0x14, 0x58B, "sqrts/su $RB,$RC",
                   [(set F4RC:$RC, (fsqrt F4RC:$RB))], s_fsqrts>;

let OperandList = (ops F4RC:$RC, F4RC:$RA, F4RC:$RB) in {
def ADDS  : FPForm<0x16, 0x580, "adds/su $RA,$RB,$RC",
                   [(set F4RC:$RC, (fadd F4RC:$RA, F4RC:$RB))], s_fadd>;
def SUBS  : FPForm<0x16, 0x581, "subs/su $RA,$RB,$RC",
                   [(set F4RC:$RC, (fsub F4RC:$RA, F4RC:$RB))], s_fadd>;
def DIVS  : FPForm<0x16, 0x583, "divs/su $RA,$RB,$RC",
                   [(set F4RC:$RC, (fdiv F4RC:$RA, F4RC:$RB))], s_fdivs>;
def MULS  : FPForm<0x16, 0x582, "muls/su $RA,$RB,$RC",
                   [(set F4RC:$RC, (fmul F4RC:$RA, F4RC:$RB))], s_fmul>;

def CPYSS  : FPForm<0x17, 0x020, "cpys $RA,$RB,$RC",
                   [(set F4RC:$RC, (fcopysign F4RC:$RB, F4RC:$RA))], s_fadd>;
def CPYSES : FPForm<0x17, 0x022, "cpyse $RA,$RB,$RC",[], s_fadd>; //Copy sign and exponent
def CPYSNS : FPForm<0x17, 0x021, "cpysn $RA,$RB,$RC",
                   [(set F4RC:$RC, (fneg (fcopysign F4RC:$RB, F4RC:$RA)))], s_fadd>;
}

//Doubles

let OperandList = (ops F8RC:$RC, F8RC:$RB), Fa = 31 in 
def SQRTT : FPForm<0x14, 0x5AB, "sqrtt/su $RB,$RC",
                   [(set F8RC:$RC, (fsqrt F8RC:$RB))], s_fsqrtt>;

let OperandList = (ops F8RC:$RC, F8RC:$RA, F8RC:$RB) in {
def ADDT  : FPForm<0x16, 0x5A0, "addt/su $RA,$RB,$RC",
                   [(set F8RC:$RC, (fadd F8RC:$RA, F8RC:$RB))], s_fadd>;
def SUBT  : FPForm<0x16, 0x5A1, "subt/su $RA,$RB,$RC",
                   [(set F8RC:$RC, (fsub F8RC:$RA, F8RC:$RB))], s_fadd>;
def DIVT  : FPForm<0x16, 0x5A3, "divt/su $RA,$RB,$RC",
                   [(set F8RC:$RC, (fdiv F8RC:$RA, F8RC:$RB))], s_fdivt>;
def MULT  : FPForm<0x16, 0x5A2, "mult/su $RA,$RB,$RC",
                   [(set F8RC:$RC, (fmul F8RC:$RA, F8RC:$RB))], s_fmul>;

def CPYST  : FPForm<0x17, 0x020, "cpys $RA,$RB,$RC",
                   [(set F8RC:$RC, (fcopysign F8RC:$RB, F8RC:$RA))], s_fadd>;
def CPYSET : FPForm<0x17, 0x022, "cpyse $RA,$RB,$RC",[], s_fadd>; //Copy sign and exponent
def CPYSNT : FPForm<0x17, 0x021, "cpysn $RA,$RB,$RC",
                   [(set F8RC:$RC, (fneg (fcopysign F8RC:$RB, F8RC:$RA)))], s_fadd>;

def CMPTEQ : FPForm<0x16, 0x5A5, "cmpteq/su $RA,$RB,$RC", [], s_fadd>;
//                    [(set F8RC:$RC, (seteq F8RC:$RA, F8RC:$RB))]>;
def CMPTLE : FPForm<0x16, 0x5A7, "cmptle/su $RA,$RB,$RC", [], s_fadd>;
//                    [(set F8RC:$RC, (setle F8RC:$RA, F8RC:$RB))]>;
def CMPTLT : FPForm<0x16, 0x5A6, "cmptlt/su $RA,$RB,$RC", [], s_fadd>;
//                    [(set F8RC:$RC, (setlt F8RC:$RA, F8RC:$RB))]>;
def CMPTUN : FPForm<0x16, 0x5A4, "cmptun/su $RA,$RB,$RC", [], s_fadd>;
//                    [(set F8RC:$RC, (setuo F8RC:$RA, F8RC:$RB))]>;
}

//More CPYS forms:
let OperandList = (ops F8RC:$RC, F4RC:$RA, F8RC:$RB) in {
def CPYSTs  : FPForm<0x17, 0x020, "cpys $RA,$RB,$RC",
                   [(set F8RC:$RC, (fcopysign F8RC:$RB, F4RC:$RA))], s_fadd>;
def CPYSNTs : FPForm<0x17, 0x021, "cpysn $RA,$RB,$RC",
                   [(set F8RC:$RC, (fneg (fcopysign F8RC:$RB, F4RC:$RA)))], s_fadd>;
}
let OperandList = (ops F4RC:$RC, F8RC:$RA, F4RC:$RB) in {
def CPYSSt  : FPForm<0x17, 0x020, "cpys $RA,$RB,$RC",
                   [(set F4RC:$RC, (fcopysign F4RC:$RB, F8RC:$RA))], s_fadd>;
def CPYSESt : FPForm<0x17, 0x022, "cpyse $RA,$RB,$RC",[], s_fadd>; //Copy sign and exponent
def CPYSNSt : FPForm<0x17, 0x021, "cpysn $RA,$RB,$RC",
                   [(set F4RC:$RC, (fneg (fcopysign F4RC:$RB, F8RC:$RA)))], s_fadd>;
}

//conditional moves, floats
let OperandList = (ops F4RC:$RDEST, F4RC:$RFALSE, F4RC:$RTRUE, F8RC:$RCOND),
    isTwoAddress = 1 in {
def FCMOVEQS : FPForm<0x17, 0x02A, "fcmoveq $RCOND,$RTRUE,$RDEST",[], s_fcmov>; //FCMOVE if = zero
def FCMOVGES : FPForm<0x17, 0x02D, "fcmovge $RCOND,$RTRUE,$RDEST",[], s_fcmov>; //FCMOVE if >= zero
def FCMOVGTS : FPForm<0x17, 0x02F, "fcmovgt $RCOND,$RTRUE,$RDEST",[], s_fcmov>; //FCMOVE if > zero
def FCMOVLES : FPForm<0x17, 0x02E, "fcmovle $RCOND,$RTRUE,$RDEST",[], s_fcmov>; //FCMOVE if <= zero
def FCMOVLTS : FPForm<0x17, 0x02C, "fcmovlt $RCOND,$RTRUE,$RDEST",[], s_fcmov>; // FCMOVE if < zero
def FCMOVNES : FPForm<0x17, 0x02B, "fcmovne $RCOND,$RTRUE,$RDEST",[], s_fcmov>; //FCMOVE if != zero
}
//conditional moves, doubles
let OperandList = (ops F8RC:$RDEST, F8RC:$RFALSE, F8RC:$RTRUE, F8RC:$RCOND),
    isTwoAddress = 1 in {
def FCMOVEQT : FPForm<0x17, 0x02A, "fcmoveq $RCOND,$RTRUE,$RDEST", [], s_fcmov>;
def FCMOVGET : FPForm<0x17, 0x02D, "fcmovge $RCOND,$RTRUE,$RDEST", [], s_fcmov>;
def FCMOVGTT : FPForm<0x17, 0x02F, "fcmovgt $RCOND,$RTRUE,$RDEST", [], s_fcmov>;
def FCMOVLET : FPForm<0x17, 0x02E, "fcmovle $RCOND,$RTRUE,$RDEST", [], s_fcmov>;
def FCMOVLTT : FPForm<0x17, 0x02C, "fcmovlt $RCOND,$RTRUE,$RDEST", [], s_fcmov>;
def FCMOVNET : FPForm<0x17, 0x02B, "fcmovne $RCOND,$RTRUE,$RDEST", [], s_fcmov>;
}

//misc FP selects
//Select double
def : Pat<(select (seteq F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setoeq F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setueq F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;

def : Pat<(select (setne F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVEQT F8RC:$sf, F8RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setone F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVEQT F8RC:$sf, F8RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setune F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVEQT F8RC:$sf, F8RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;

def : Pat<(select (setgt F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLT F8RC:$RB, F8RC:$RA))>;
def : Pat<(select (setogt F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLT F8RC:$RB, F8RC:$RA))>;
def : Pat<(select (setugt F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLT F8RC:$RB, F8RC:$RA))>;

def : Pat<(select (setge F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLE F8RC:$RB, F8RC:$RA))>;
def : Pat<(select (setoge F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLE F8RC:$RB, F8RC:$RA))>;
def : Pat<(select (setuge F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLE F8RC:$RB, F8RC:$RA))>;

def : Pat<(select (setlt F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLT F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setolt F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLT F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setult F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLT F8RC:$RA, F8RC:$RB))>;

def : Pat<(select (setle F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLE F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setole F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLE F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setule F8RC:$RA, F8RC:$RB), F8RC:$st, F8RC:$sf),
      (FCMOVNET F8RC:$sf, F8RC:$st, (CMPTLE F8RC:$RA, F8RC:$RB))>;

//Select single
def : Pat<(select (seteq F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setoeq F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setueq F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;

def : Pat<(select (setne F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVEQS F4RC:$sf, F4RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setone F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVEQS F4RC:$sf, F4RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setune F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVEQS F4RC:$sf, F4RC:$st, (CMPTEQ F8RC:$RA, F8RC:$RB))>;

def : Pat<(select (setgt F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLT F8RC:$RB, F8RC:$RA))>;
def : Pat<(select (setogt F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLT F8RC:$RB, F8RC:$RA))>;
def : Pat<(select (setugt F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLT F8RC:$RB, F8RC:$RA))>;

def : Pat<(select (setge F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLE F8RC:$RB, F8RC:$RA))>;
def : Pat<(select (setoge F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLE F8RC:$RB, F8RC:$RA))>;
def : Pat<(select (setuge F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLE F8RC:$RB, F8RC:$RA))>;

def : Pat<(select (setlt F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLT F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setolt F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLT F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setult F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLT F8RC:$RA, F8RC:$RB))>;

def : Pat<(select (setle F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLE F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setole F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLE F8RC:$RA, F8RC:$RB))>;
def : Pat<(select (setule F8RC:$RA, F8RC:$RB), F4RC:$st, F4RC:$sf),
      (FCMOVNES F4RC:$sf, F4RC:$st, (CMPTLE F8RC:$RA, F8RC:$RB))>;



let OperandList = (ops GPRC:$RC, F4RC:$RA), Fb = 31 in 
def FTOIS : FPForm<0x1C, 0x078, "ftois $RA,$RC",[], s_ftoi>; //Floating to integer move, S_floating
let OperandList = (ops GPRC:$RC, F8RC:$RA), Fb = 31 in 
def FTOIT : FPForm<0x1C, 0x070, "ftoit $RA,$RC",
        [(set GPRC:$RC, (Alpha_ftoit F8RC:$RA))], s_ftoi>; //Floating to integer move
let OperandList = (ops F4RC:$RC, GPRC:$RA), Fb = 31 in 
def ITOFS : FPForm<0x14, 0x004, "itofs $RA,$RC",[], s_itof>; //Integer to floating move, S_floating
let OperandList = (ops F8RC:$RC, GPRC:$RA), Fb = 31 in 
def ITOFT : FPForm<0x14, 0x024, "itoft $RA,$RC",
        [(set F8RC:$RC, (Alpha_itoft GPRC:$RA))], s_itof>; //Integer to floating move


let OperandList = (ops F4RC:$RC, F8RC:$RB), Fa = 31 in 
def CVTQS : FPForm<0x16, 0x7BC, "cvtqs/sui $RB,$RC",
        [(set F4RC:$RC, (Alpha_cvtqs F8RC:$RB))], s_fadd>;
let OperandList = (ops F8RC:$RC, F8RC:$RB), Fa = 31 in 
def CVTQT : FPForm<0x16, 0x7BE, "cvtqt/sui $RB,$RC",
        [(set F8RC:$RC, (Alpha_cvtqt F8RC:$RB))], s_fadd>;
let OperandList = (ops F8RC:$RC, F8RC:$RB), Fa = 31 in 
def CVTTQ : FPForm<0x16, 0x52F, "cvttq/svc $RB,$RC",
        [(set F8RC:$RC, (Alpha_cvttq F8RC:$RB))], s_fadd>;
let OperandList = (ops F8RC:$RC, F4RC:$RB), Fa = 31 in 
def CVTST : FPForm<0x16, 0x6AC, "cvtst/s $RB,$RC",
                   [(set F8RC:$RC, (fextend F4RC:$RB))], s_fadd>;
let OperandList = (ops F4RC:$RC, F8RC:$RB), Fa = 31 in 
def CVTTS : FPForm<0x16, 0x7AC, "cvtts/sui $RB,$RC",
                   [(set F4RC:$RC, (fround F8RC:$RB))], s_fadd>;


/////////////////////////////////////////////////////////
//Branching
/////////////////////////////////////////////////////////
let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, noResults = 1 in {
let Ra = 31 in
def BR : BFormD<0x30, "br $$31,$DISP", [(br bb:$DISP)], s_ubr>;

//Branches, int
def BEQ  : BForm<0x39, "beq $RA,$DISP", 
                 [(brcond (seteq GPRC:$RA, 0), bb:$DISP)], s_icbr>;
def BGE  : BForm<0x3E, "bge $RA,$DISP", 
                 [(brcond (setge GPRC:$RA, 0), bb:$DISP)], s_icbr>;
def BGT  : BForm<0x3F, "bgt $RA,$DISP",
                 [(brcond (setgt GPRC:$RA, 0), bb:$DISP)], s_icbr>;
def BLBC : BForm<0x38, "blbc $RA,$DISP", [], s_icbr>; //TODO: Low bit clear
def BLBS : BForm<0x3C, "blbs $RA,$DISP",
                 [(brcond (and GPRC:$RA, 1), bb:$DISP)], s_icbr>;
def BLE  : BForm<0x3B, "ble $RA,$DISP",
                 [(brcond (setle GPRC:$RA, 0), bb:$DISP)], s_icbr>;
def BLT  : BForm<0x3A, "blt $RA,$DISP",
                 [(brcond (setlt GPRC:$RA, 0), bb:$DISP)], s_icbr>;
def BNE  : BForm<0x3D, "bne $RA,$DISP",
                 [(brcond (setne GPRC:$RA, 0), bb:$DISP)], s_icbr>;

//Branches, float
def FBEQ : FBForm<0x31, "fbeq $RA,$DISP", 
                  [(brcond (seteq F8RC:$RA, immFPZ), bb:$DISP)], s_fbr>;
def FBGE : FBForm<0x36, "fbge $RA,$DISP",
                  [(brcond (setge F8RC:$RA, immFPZ), bb:$DISP)], s_fbr>;
def FBGT : FBForm<0x37, "fbgt $RA,$DISP",
                  [(brcond (setgt F8RC:$RA, immFPZ), bb:$DISP)], s_fbr>;
def FBLE : FBForm<0x33, "fble $RA,$DISP",
                  [(brcond (setle F8RC:$RA, immFPZ), bb:$DISP)], s_fbr>;
def FBLT : FBForm<0x32, "fblt $RA,$DISP",
                  [(brcond (setlt F8RC:$RA, immFPZ), bb:$DISP)], s_fbr>;
def FBNE : FBForm<0x35, "fbne $RA,$DISP",
                  [(brcond (setne F8RC:$RA, immFPZ), bb:$DISP)], s_fbr>;
}

def : Pat<(brcond GPRC:$RA, bb:$DISP), (BNE GPRC:$RA, bb:$DISP)>;
def : Pat<(brcond (setne GPRC:$RA, GPRC:$RB), bb:$DISP),
          (BEQ (CMPEQ GPRC:$RA, GPRC:$RB), bb:$DISP)>;
def : Pat<(brcond (setne GPRC:$RA, immUExt8:$L), bb:$DISP),
          (BEQ (CMPEQi GPRC:$RA, immUExt8:$L), bb:$DISP)>;

def : Pat<(brcond (seteq F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTEQ F8RC:$RA, F8RC:$RB), bb:$DISP)>;
def : Pat<(brcond (setoeq F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTEQ F8RC:$RA, F8RC:$RB), bb:$DISP)>;
def : Pat<(brcond (setueq F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTEQ F8RC:$RA, F8RC:$RB), bb:$DISP)>;

def : Pat<(brcond (setlt F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLT F8RC:$RA, F8RC:$RB), bb:$DISP)>;
def : Pat<(brcond (setolt F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLT F8RC:$RA, F8RC:$RB), bb:$DISP)>;
def : Pat<(brcond (setult F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLT F8RC:$RA, F8RC:$RB), bb:$DISP)>;

def : Pat<(brcond (setle F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLE F8RC:$RA, F8RC:$RB), bb:$DISP)>;
def : Pat<(brcond (setole F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLE F8RC:$RA, F8RC:$RB), bb:$DISP)>;
def : Pat<(brcond (setule F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLE F8RC:$RA, F8RC:$RB), bb:$DISP)>;

def : Pat<(brcond (setgt F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLT F8RC:$RB, F8RC:$RA), bb:$DISP)>;
def : Pat<(brcond (setogt F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLT F8RC:$RB, F8RC:$RA), bb:$DISP)>;
def : Pat<(brcond (setugt F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLT F8RC:$RB, F8RC:$RA), bb:$DISP)>;

def : Pat<(brcond (setge F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLE F8RC:$RB, F8RC:$RA), bb:$DISP)>;
def : Pat<(brcond (setoge F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLE F8RC:$RB, F8RC:$RA), bb:$DISP)>;
def : Pat<(brcond (setuge F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBNE  (CMPTLE F8RC:$RB, F8RC:$RA), bb:$DISP)>;

def : Pat<(brcond (setne F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBEQ  (CMPTEQ F8RC:$RA, F8RC:$RB), bb:$DISP)>;
def : Pat<(brcond (setone F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBEQ  (CMPTEQ F8RC:$RA, F8RC:$RB), bb:$DISP)>;
def : Pat<(brcond (setune F8RC:$RA, F8RC:$RB), bb:$DISP),
          (FBEQ  (CMPTEQ F8RC:$RA, F8RC:$RB), bb:$DISP)>;


def : Pat<(brcond (setoeq F8RC:$RA, immFPZ), bb:$DISP),
          (FBEQ F8RC:$RA,bb:$DISP)>;
def : Pat<(brcond (setueq F8RC:$RA, immFPZ), bb:$DISP),
          (FBEQ F8RC:$RA,bb:$DISP)>;

def : Pat<(brcond (setoge F8RC:$RA, immFPZ), bb:$DISP),
          (FBGE F8RC:$RA,bb:$DISP)>;
def : Pat<(brcond (setuge F8RC:$RA, immFPZ), bb:$DISP),
          (FBGE F8RC:$RA,bb:$DISP)>;

def : Pat<(brcond (setogt F8RC:$RA, immFPZ), bb:$DISP),
          (FBGT F8RC:$RA,bb:$DISP)>;
def : Pat<(brcond (setugt F8RC:$RA, immFPZ), bb:$DISP),
          (FBGT F8RC:$RA,bb:$DISP)>;

def : Pat<(brcond (setole F8RC:$RA, immFPZ), bb:$DISP),
          (FBLE F8RC:$RA,bb:$DISP)>;
def : Pat<(brcond (setule F8RC:$RA, immFPZ), bb:$DISP),
          (FBLE F8RC:$RA,bb:$DISP)>;

def : Pat<(brcond (setolt F8RC:$RA, immFPZ), bb:$DISP),
          (FBLT F8RC:$RA,bb:$DISP)>;
def : Pat<(brcond (setult F8RC:$RA, immFPZ), bb:$DISP),
          (FBLT F8RC:$RA,bb:$DISP)>;

def : Pat<(brcond (setone F8RC:$RA, immFPZ), bb:$DISP),
          (FBNE F8RC:$RA,bb:$DISP)>;
def : Pat<(brcond (setune F8RC:$RA, immFPZ), bb:$DISP),
          (FBNE F8RC:$RA,bb:$DISP)>;

//End Branches

//S_floating : IEEE Single
//T_floating : IEEE Double

//Unused instructions
//Mnemonic Format Opcode Description
//CALL_PAL Pcd 00 Trap to PALcode
//ECB Mfc 18.E800 Evict cache block
//EXCB Mfc 18.0400 Exception barrier
//FETCH Mfc 18.8000 Prefetch data
//FETCH_M Mfc 18.A000 Prefetch data, modify intent
//LDL_L Mem 2A Load sign-extended longword locked
//LDQ_L Mem 2B Load quadword locked
//LDQ_U Mem 0B Load unaligned quadword
//MB Mfc 18.4000 Memory barrier
//STL_C Mem 2E Store longword conditional
//STQ_C Mem 2F Store quadword conditional
//STQ_U Mem 0F Store unaligned quadword
//TRAPB Mfc 18.0000 Trap barrier
//WH64 Mfc 18.F800 Write hint  64 bytes
//WMB Mfc 18.4400 Write memory barrier
//MF_FPCR F-P 17.025 Move from FPCR
//MT_FPCR F-P 17.024 Move to FPCR
//There are in the Multimedia extentions, so let's not use them yet
//def MAXSB8  : OForm<0x1C, 0x3E, "MAXSB8 $RA,$RB,$RC">; //Vector signed byte maximum
//def MAXSW4 : OForm< 0x1C, 0x3F, "MAXSW4 $RA,$RB,$RC">; //Vector signed word maximum
//def MAXUB8  : OForm<0x1C, 0x3C, "MAXUB8 $RA,$RB,$RC">; //Vector unsigned byte maximum
//def MAXUW4 : OForm< 0x1C, 0x3D, "MAXUW4 $RA,$RB,$RC">; //Vector unsigned word maximum
//def MINSB8 : OForm< 0x1C, 0x38, "MINSB8 $RA,$RB,$RC">; //Vector signed byte minimum
//def MINSW4 : OForm< 0x1C, 0x39, "MINSW4 $RA,$RB,$RC">; //Vector signed word minimum
//def MINUB8 : OForm< 0x1C, 0x3A, "MINUB8 $RA,$RB,$RC">; //Vector unsigned byte minimum
//def MINUW4 : OForm< 0x1C, 0x3B, "MINUW4 $RA,$RB,$RC">; //Vector unsigned word minimum
//def PERR : OForm< 0x1C, 0x31, "PERR $RA,$RB,$RC">; //Pixel error
//def PKLB : OForm< 0x1C, 0x37, "PKLB $RA,$RB,$RC">; //Pack longwords to bytes
//def PKWB  : OForm<0x1C, 0x36, "PKWB $RA,$RB,$RC">; //Pack words to bytes
//def UNPKBL : OForm< 0x1C, 0x35, "UNPKBL $RA,$RB,$RC">; //Unpack bytes to longwords
//def UNPKBW : OForm< 0x1C, 0x34, "UNPKBW $RA,$RB,$RC">; //Unpack bytes to words
//CVTLQ F-P 17.010 Convert longword to quadword
//CVTQL F-P 17.030 Convert quadword to longword
//def AMASK    : OForm< 0x11, 0x61, "AMASK $RA,$RB,$RC", []>; //Architecture mask
//def AMASKi   : OFormL<0x11, 0x61, "AMASK $RA,$L,$RC", []>; //Architecture mask


//Constant handling

def immConst2Part  : PatLeaf<(imm), [{
  //true if imm fits in a LDAH LDA pair
  int64_t val = (int64_t)N->getValue();
  return (val <= IMM_FULLHIGH  && val >= IMM_FULLLOW);
}]>;
def immConst2PartInt  : PatLeaf<(imm), [{
  //true if imm fits in a LDAH LDA pair with zeroext
  uint64_t uval = N->getValue();
  int32_t val32 = (int32_t)uval;
  return ((uval >> 32) == 0 && //empty upper bits
          val32 <= IMM_FULLHIGH);
//          val32 >= IMM_FULLLOW  + IMM_LOW  * IMM_MULT); //Always True
}], SExt32>;

def : Pat<(i64 immConst2Part:$imm),
          (LDA (LL16 immConst2Part:$imm), (LDAH (LH16 immConst2Part:$imm), R31))>;

def : Pat<(i64 immSExt16:$imm),
          (LDA immSExt16:$imm, R31)>;

def : Pat<(i64 immSExt16int:$imm),
          (ZAPNOTi (LDA (SExt16 immSExt16int:$imm), R31), 15)>;
def : Pat<(i64 immConst2PartInt:$imm),
          (ZAPNOTi (LDA (LL16 (SExt32 immConst2PartInt:$imm)), 
                        (LDAH (LH16 (SExt32 immConst2PartInt:$imm)), R31)), 15)>;


//TODO: I want to just define these like this!
//def : Pat<(i64 0),
//          (R31)>;
//def : Pat<(f64 0.0),
//          (F31)>;
//def : Pat<(f64 -0.0),
//          (CPYSNT F31, F31)>;
//def : Pat<(f32 0.0),
//          (F31)>;
//def : Pat<(f32 -0.0),
//          (CPYSNS F31, F31)>;

//Misc Patterns:

def : Pat<(sext_inreg GPRC:$RB, i32),
          (ADDLi GPRC:$RB, 0)>;

def : Pat<(fabs F8RC:$RB),
          (CPYST F31, F8RC:$RB)>;
def : Pat<(fabs F4RC:$RB),
          (CPYSS F31, F4RC:$RB)>;
def : Pat<(fneg F8RC:$RB),
          (CPYSNT F8RC:$RB, F8RC:$RB)>;
def : Pat<(fneg F4RC:$RB),
          (CPYSNS F4RC:$RB, F4RC:$RB)>;

def : Pat<(fcopysign F4RC:$A, (fneg F4RC:$B)),
          (CPYSNS F4RC:$B, F4RC:$A)>;
def : Pat<(fcopysign F8RC:$A, (fneg F8RC:$B)),
          (CPYSNT F8RC:$B, F8RC:$A)>;
def : Pat<(fcopysign F4RC:$A, (fneg F8RC:$B)),
          (CPYSNSt F8RC:$B, F4RC:$A)>;
def : Pat<(fcopysign F8RC:$A, (fneg F4RC:$B)),
          (CPYSNTs F4RC:$B, F8RC:$A)>;

//Yes, signed multiply high is ugly
def : Pat<(mulhs GPRC:$RA, GPRC:$RB),
          (SUBQ (UMULH GPRC:$RA, GPRC:$RB), (ADDQ (CMOVGE GPRC:$RB, R31, GPRC:$RA), 
                                                 (CMOVGE GPRC:$RA, R31, GPRC:$RB)))>;

//Stupid crazy arithmetic stuff:
def : Pat<(mul GPRC:$RA, 5), (S4ADDQ GPRC:$RA, GPRC:$RA)>;
def : Pat<(mul GPRC:$RA, 3), (S4SUBQ GPRC:$RA, GPRC:$RA)>;

def : Pat<(mul GPRC:$RA, immRem1:$imm), 
          (ADDQ (SL GPRC:$RA, (nearP2X immRem1:$imm)), GPRC:$RA)>;
def : Pat<(mul GPRC:$RA, immRem3:$imm),
          (ADDQ (SL GPRC:$RA, (nearP2X immRem3:$imm)), (S4SUBQ GPRC:$RA, GPRC:$RA))>;
def : Pat<(mul GPRC:$RA, immRem5:$imm),
          (ADDQ (SL GPRC:$RA, (nearP2X immRem5:$imm)), (S4ADDQ GPRC:$RA, GPRC:$RA))>;
def : Pat<(mul GPRC:$RA, immRem4:$imm),
          (S4ADDQ GPRC:$RA, (SL GPRC:$RA, (nearP2X immRem4:$imm)))>;
def : Pat<(mul GPRC:$RA, immRemP2:$imm),
          (ADDQ (SL GPRC:$RA, (nearP2X immRemP2:$imm)), (SLi GPRC:$RA, (nearP2RemX immRemP2:$imm)))>;

def : Pat<(mul GPRC:$RA, immRem1n:$imm), 
          (SUBQ (SL GPRC:$RA, (nearP2X immRem1n:$imm)), GPRC:$RA)>;
def : Pat<(mul GPRC:$RA, immRem3n:$imm),
          (SUBQ (SL GPRC:$RA, (nearP2X immRem3n:$imm)), (S4SUBQ GPRC:$RA, GPRC:$RA))>;
def : Pat<(mul GPRC:$RA, immRem5n:$imm),
          (SUBQ (SL GPRC:$RA, (nearP2X immRem5n:$imm)), (S4ADDQ GPRC:$RA, GPRC:$RA))>;
def : Pat<(mul GPRC:$RA, immRemP2n:$imm),
          (SUBQ (SL GPRC:$RA, (nearP2X immRemP2n:$imm)), (SLi GPRC:$RA, (nearP2RemX immRemP2n:$imm)))>;
