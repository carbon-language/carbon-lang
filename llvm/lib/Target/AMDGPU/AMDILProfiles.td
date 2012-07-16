//===- AMDILProfiles.td - AMD IL Profiles ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
// These are used for custom selection dag type profiles

//===----------------------------------------------------------------------===//
// Custom Selection DAG Type Profiles
//===----------------------------------------------------------------------===//
// SDTCisDP - The specified operand has double type
// Tablegen needs to be hacked to get this constraint to work
//class SDTCisDP<int OpNum> : SDTypeConstraint<OpNum>;

//===----------------------------------------------------------------------===//
// Generic Profile Types
//===----------------------------------------------------------------------===//

def SDTIL_GenUnaryOp : SDTypeProfile<1, 1, [
    SDTCisSameAs<0, 1>
    ]>;
def SDTIL_GenBinaryOp : SDTypeProfile<1, 2, [
    SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>
    ]>;
def SDTIL_GenTernaryOp : SDTypeProfile<1, 3, [
    SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, SDTCisSameAs<2, 3>
    ]>;
def SDTIL_GenCMovLog : SDTypeProfile<1, 3, [
    SDTCisSameAs<0, 2>, SDTCisSameAs<2, 3>, SDTCisInt<1>
    ]>;
def SDTIL_GenVecBuild : SDTypeProfile<1, 1, [
    SDTCisEltOfVec<1, 0>
    ]>;

def SDTIL_GenVecExtract : SDTypeProfile<1, 2, [
    SDTCisEltOfVec<0, 1>, SDTCisVT<2, i32>
    ]>;

def SDTIL_GenVecInsert : SDTypeProfile<1, 4, [
    SDTCisEltOfVec<2, 1>, SDTCisSameAs<0, 1>,
    SDTCisVT<3, i32>, SDTCisVT<4, i32>
    ]>;

def SDTIL_GenVecShuffle : SDTypeProfile <1, 2, [
    SDTCisSameAs<0, 1>, SDTCisVT<2, i32>
    ]>;

def SDTIL_GenVecConcat : SDTypeProfile <1, 2, [
    SDTCisSameAs<1, 2>
    ]>;
//===----------------------------------------------------------------------===//
// Conversion Profile Types
//===----------------------------------------------------------------------===//
def SDTIL_DPToFPOp : SDTypeProfile<1, 1, [
    SDTCisFP<0>, SDTCisFP<1>, SDTCisOpSmallerThanOp<0, 1>
    ]>; // d2f

def SDTIL_AnyToInt : SDTypeProfile<1, 1, [
    SDTCisInt<0>
    ]>;
def SDTIL_IntToAny : SDTypeProfile<1, 1, [
    SDTCisInt<1>
    ]>;
def SDTIL_GenBitConv : SDTypeProfile<1, 1, []>;
//===----------------------------------------------------------------------===//
// Scalar Profile Types
//===----------------------------------------------------------------------===//

// Add instruction pattern to handle offsets of memory operationns
def SDTIL_AddAddrri: SDTypeProfile<1, 2, [
    SDTCisInt<0>, SDTCisPtrTy<1>, SDTCisSameAs<0, 2>
    ]>;
def SDTIL_AddAddrir : SDTypeProfile<1, 2, [
    SDTCisInt<0>, SDTCisPtrTy<2>, SDTCisSameAs<0, 1>
    ]>;

def SDTIL_LCreate : SDTypeProfile<1, 2, [
    SDTCisVT<0, i64>, SDTCisVT<1, i32>, SDTCisSameAs<1, 2>
    ]>;
def SDTIL_LCreate2 : SDTypeProfile<1, 2, [
    SDTCisVT<0, v2i64>, SDTCisVT<1, v2i32>, SDTCisSameAs<1, 2>
    ]>;
def SDTIL_LComp : SDTypeProfile<1, 1, [
    SDTCisVT<0, i32>, SDTCisVT<1, i64>
    ]>;
def SDTIL_LComp2 : SDTypeProfile<1, 1, [
    SDTCisVT<0, v2i32>, SDTCisVT<1, v2i64>
    ]>;
def SDTIL_DCreate : SDTypeProfile<1, 2, [
    SDTCisVT<0, f64>, SDTCisVT<1, i32>, SDTCisSameAs<1, 2>
    ]>;
def SDTIL_DComp : SDTypeProfile<1, 1, [
    SDTCisVT<0, i32>, SDTCisVT<1, f64>
    ]>;
def SDTIL_DCreate2 : SDTypeProfile<1, 2, [
    SDTCisVT<0, v2f64>, SDTCisVT<1, v2i32>, SDTCisSameAs<1, 2>
    ]>;
def SDTIL_DComp2 : SDTypeProfile<1, 1, [
    SDTCisVT<0, v2i32>, SDTCisVT<1, v2f64>
    ]>;
//===----------------------------------------------------------------------===//
// Flow Control Profile Types
//===----------------------------------------------------------------------===//
// Profile for Normal Call
def SDTIL_Call : SDTypeProfile<0, 1, [
    SDTCisVT<0, i32>
    ]>;
// Branch instruction where second and third are basic blocks
def SDTIL_BRCond : SDTypeProfile<0, 2, [
    SDTCisVT<0, OtherVT>
    ]>;
// Comparison instruction
def SDTIL_Cmp  : SDTypeProfile<1, 3, [
    SDTCisSameAs<0, 2>, SDTCisSameAs<2,3>, SDTCisVT<1, i32>
    ]>;


//===----------------------------------------------------------------------===//
// Call Sequence Profiles
//===----------------------------------------------------------------------===//
def SDTIL_CallSeqStart  : SDCallSeqStart< [
    SDTCisVT<0, i32>
    ]>;
def SDTIL_CallSeqEnd    : SDCallSeqEnd< [
    SDTCisVT<0, i32>, SDTCisVT<1, i32>
    ]>;

//===----------------------------------------------------------------------===//
// Image Operation Profiles
//===----------------------------------------------------------------------===//
def SDTIL_ImageRead  : SDTypeProfile<1, 3, 
    [SDTCisVT<0, v4i32>, SDTCisPtrTy<1>, SDTCisVT<2, i32>, SDTCisVT<3, v4f32>]>;
def SDTIL_ImageWrite : SDTypeProfile<0, 3,
    [SDTCisPtrTy<0>, SDTCisVT<1, v2i32>, SDTCisVT<2, v4i32>]>;
def SDTIL_ImageWrite3D : SDTypeProfile<0, 3,
    [SDTCisPtrTy<0>, SDTCisVT<1, v4i32>, SDTCisVT<2, v4i32>]>;
def SDTIL_ImageInfo  : SDTypeProfile<1, 1,
    [SDTCisVT<0, v4i32>, SDTCisPtrTy<1>]>;
//===----------------------------------------------------------------------===//
// Atomic Operation Profiles
//===----------------------------------------------------------------------===//
def SDTIL_UniAtomNoRet : SDTypeProfile<0, 2, [
    SDTCisPtrTy<0>, SDTCisVT<1, i32>
    ]>;
def SDTIL_BinAtomNoRet : SDTypeProfile<0, 3, [
    SDTCisPtrTy<0>, SDTCisVT<1, i32>, SDTCisVT<2, i32>
    ]>;
def SDTIL_TriAtomNoRet : SDTypeProfile<0, 4, [
    SDTCisPtrTy<0>, SDTCisVT<1, i32>, SDTCisVT<2, i32>, SDTCisVT<3, i32>
    ]>;
def SDTIL_UniAtom : SDTypeProfile<1, 2, [
    SDTCisVT<0, i32>, SDTCisPtrTy<1>, SDTCisVT<2, i32>
    ]>;
def SDTIL_BinAtom : SDTypeProfile<1, 3, [
    SDTCisVT<0, i32>, SDTCisPtrTy<1>, SDTCisVT<2, i32>, SDTCisVT<3, i32>
    ]>;
def SDTIL_TriAtom : SDTypeProfile<1, 4, [
    SDTCisVT<0, i32>, SDTCisPtrTy<1>, SDTCisVT<2, i32>,
    SDTCisVT<3, i32>, SDTCisVT<4, i32>
    ]>;

def SDTIL_BinAtomFloat : SDTypeProfile<1, 3, [
    SDTCisVT<0, i32>, SDTCisPtrTy<1>, SDTCisVT<2, f32>, SDTCisVT<3, f32>
    ]>;
def SDTIL_BinAtomNoRetFloat : SDTypeProfile<0, 3, [
    SDTCisPtrTy<0>, SDTCisVT<1, f32>, SDTCisVT<2, f32>
    ]>;

def SDTIL_Append : SDTypeProfile<1, 1, [
    SDTCisVT<0, i32>, SDTCisPtrTy<1>
    ]>;
