//===-- SystemZOperators.td - SystemZ-specific operators ------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type profiles
//===----------------------------------------------------------------------===//
def SDT_CallSeqStart        : SDCallSeqStart<[SDTCisVT<0, i64>]>;
def SDT_CallSeqEnd          : SDCallSeqEnd<[SDTCisVT<0, i64>,
                                            SDTCisVT<1, i64>]>;
def SDT_ZCall               : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
def SDT_ZCmp                : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;
def SDT_ZICmp               : SDTypeProfile<0, 3,
                                            [SDTCisSameAs<0, 1>,
                                             SDTCisVT<2, i32>]>;
def SDT_ZBRCCMask           : SDTypeProfile<0, 3,
                                            [SDTCisVT<0, i32>,
                                             SDTCisVT<1, i32>,
                                             SDTCisVT<2, OtherVT>]>;
def SDT_ZSelectCCMask       : SDTypeProfile<1, 4,
                                            [SDTCisSameAs<0, 1>,
                                             SDTCisSameAs<1, 2>,
                                             SDTCisVT<3, i32>,
                                             SDTCisVT<4, i32>]>;
def SDT_ZWrapPtr            : SDTypeProfile<1, 1,
                                            [SDTCisSameAs<0, 1>,
                                             SDTCisPtrTy<0>]>;
def SDT_ZWrapOffset         : SDTypeProfile<1, 2,
                                            [SDTCisSameAs<0, 1>,
                                             SDTCisSameAs<0, 2>,
                                             SDTCisPtrTy<0>]>;
def SDT_ZAdjDynAlloc        : SDTypeProfile<1, 0, [SDTCisVT<0, i64>]>;
def SDT_ZExtractAccess      : SDTypeProfile<1, 1,
                                            [SDTCisVT<0, i32>,
                                             SDTCisVT<1, i32>]>;
def SDT_ZGR128Binary32      : SDTypeProfile<1, 2,
                                            [SDTCisVT<0, untyped>,
                                             SDTCisVT<1, untyped>,
                                             SDTCisVT<2, i32>]>;
def SDT_ZGR128Binary64      : SDTypeProfile<1, 2,
                                            [SDTCisVT<0, untyped>,
                                             SDTCisVT<1, untyped>,
                                             SDTCisVT<2, i64>]>;
def SDT_ZAtomicLoadBinaryW  : SDTypeProfile<1, 5,
                                            [SDTCisVT<0, i32>,
                                             SDTCisPtrTy<1>,
                                             SDTCisVT<2, i32>,
                                             SDTCisVT<3, i32>,
                                             SDTCisVT<4, i32>,
                                             SDTCisVT<5, i32>]>;
def SDT_ZAtomicCmpSwapW     : SDTypeProfile<1, 6,
                                            [SDTCisVT<0, i32>,
                                             SDTCisPtrTy<1>,
                                             SDTCisVT<2, i32>,
                                             SDTCisVT<3, i32>,
                                             SDTCisVT<4, i32>,
                                             SDTCisVT<5, i32>,
                                             SDTCisVT<6, i32>]>;
def SDT_ZMemMemLength       : SDTypeProfile<0, 3,
                                            [SDTCisPtrTy<0>,
                                             SDTCisPtrTy<1>,
                                             SDTCisVT<2, i64>]>;
def SDT_ZMemMemLoop         : SDTypeProfile<0, 4,
                                            [SDTCisPtrTy<0>,
                                             SDTCisPtrTy<1>,
                                             SDTCisVT<2, i64>,
                                             SDTCisVT<3, i64>]>;
def SDT_ZString             : SDTypeProfile<1, 3,
                                            [SDTCisPtrTy<0>,
                                             SDTCisPtrTy<1>,
                                             SDTCisPtrTy<2>,
                                             SDTCisVT<3, i32>]>;
def SDT_ZI32Intrinsic       : SDTypeProfile<1, 0, [SDTCisVT<0, i32>]>;
def SDT_ZPrefetch           : SDTypeProfile<0, 2,
                                            [SDTCisVT<0, i32>,
                                             SDTCisPtrTy<1>]>;

//===----------------------------------------------------------------------===//
// Node definitions
//===----------------------------------------------------------------------===//

// These are target-independent nodes, but have target-specific formats.
def callseq_start       : SDNode<"ISD::CALLSEQ_START", SDT_CallSeqStart,
                                 [SDNPHasChain, SDNPSideEffect, SDNPOutGlue]>;
def callseq_end         : SDNode<"ISD::CALLSEQ_END",   SDT_CallSeqEnd,
                                 [SDNPHasChain, SDNPSideEffect, SDNPOptInGlue,
                                  SDNPOutGlue]>;
def global_offset_table : SDNode<"ISD::GLOBAL_OFFSET_TABLE", SDTPtrLeaf>;

// Nodes for SystemZISD::*.  See SystemZISelLowering.h for more details.
def z_retflag           : SDNode<"SystemZISD::RET_FLAG", SDTNone,
                                 [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def z_call              : SDNode<"SystemZISD::CALL", SDT_ZCall,
                                 [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue,
                                  SDNPVariadic]>;
def z_sibcall           : SDNode<"SystemZISD::SIBCALL", SDT_ZCall,
                                 [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue,
                                  SDNPVariadic]>;
def z_tls_gdcall        : SDNode<"SystemZISD::TLS_GDCALL", SDT_ZCall,
                                 [SDNPHasChain, SDNPInGlue, SDNPOutGlue,
                                  SDNPVariadic]>;
def z_tls_ldcall        : SDNode<"SystemZISD::TLS_LDCALL", SDT_ZCall,
                                 [SDNPHasChain, SDNPInGlue, SDNPOutGlue,
                                  SDNPVariadic]>;
def z_pcrel_wrapper     : SDNode<"SystemZISD::PCREL_WRAPPER", SDT_ZWrapPtr, []>;
def z_pcrel_offset      : SDNode<"SystemZISD::PCREL_OFFSET",
                                 SDT_ZWrapOffset, []>;
def z_iabs              : SDNode<"SystemZISD::IABS", SDTIntUnaryOp, []>;
def z_icmp              : SDNode<"SystemZISD::ICMP", SDT_ZICmp, [SDNPOutGlue]>;
def z_fcmp              : SDNode<"SystemZISD::FCMP", SDT_ZCmp, [SDNPOutGlue]>;
def z_tm                : SDNode<"SystemZISD::TM", SDT_ZICmp, [SDNPOutGlue]>;
def z_br_ccmask         : SDNode<"SystemZISD::BR_CCMASK", SDT_ZBRCCMask,
                                 [SDNPHasChain, SDNPInGlue]>;
def z_select_ccmask     : SDNode<"SystemZISD::SELECT_CCMASK", SDT_ZSelectCCMask,
    		                 [SDNPInGlue]>;
def z_adjdynalloc       : SDNode<"SystemZISD::ADJDYNALLOC", SDT_ZAdjDynAlloc>;
def z_extract_access    : SDNode<"SystemZISD::EXTRACT_ACCESS",
                                 SDT_ZExtractAccess>;
def z_umul_lohi64       : SDNode<"SystemZISD::UMUL_LOHI64", SDT_ZGR128Binary64>;
def z_sdivrem32         : SDNode<"SystemZISD::SDIVREM32", SDT_ZGR128Binary32>;
def z_sdivrem64         : SDNode<"SystemZISD::SDIVREM64", SDT_ZGR128Binary64>;
def z_udivrem32         : SDNode<"SystemZISD::UDIVREM32", SDT_ZGR128Binary32>;
def z_udivrem64         : SDNode<"SystemZISD::UDIVREM64", SDT_ZGR128Binary64>;

def z_serialize         : SDNode<"SystemZISD::SERIALIZE", SDTNone,
                                 [SDNPHasChain, SDNPMayStore]>;

class AtomicWOp<string name, SDTypeProfile profile = SDT_ZAtomicLoadBinaryW>
  : SDNode<"SystemZISD::"##name, profile,
           [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;

def z_atomic_swapw      : AtomicWOp<"ATOMIC_SWAPW">;
def z_atomic_loadw_add  : AtomicWOp<"ATOMIC_LOADW_ADD">;
def z_atomic_loadw_sub  : AtomicWOp<"ATOMIC_LOADW_SUB">;
def z_atomic_loadw_and  : AtomicWOp<"ATOMIC_LOADW_AND">;
def z_atomic_loadw_or   : AtomicWOp<"ATOMIC_LOADW_OR">;
def z_atomic_loadw_xor  : AtomicWOp<"ATOMIC_LOADW_XOR">;
def z_atomic_loadw_nand : AtomicWOp<"ATOMIC_LOADW_NAND">;
def z_atomic_loadw_min  : AtomicWOp<"ATOMIC_LOADW_MIN">;
def z_atomic_loadw_max  : AtomicWOp<"ATOMIC_LOADW_MAX">;
def z_atomic_loadw_umin : AtomicWOp<"ATOMIC_LOADW_UMIN">;
def z_atomic_loadw_umax : AtomicWOp<"ATOMIC_LOADW_UMAX">;
def z_atomic_cmp_swapw  : AtomicWOp<"ATOMIC_CMP_SWAPW", SDT_ZAtomicCmpSwapW>;

def z_mvc               : SDNode<"SystemZISD::MVC", SDT_ZMemMemLength,
                                 [SDNPHasChain, SDNPMayStore, SDNPMayLoad]>;
def z_mvc_loop          : SDNode<"SystemZISD::MVC_LOOP", SDT_ZMemMemLoop,
                                 [SDNPHasChain, SDNPMayStore, SDNPMayLoad]>;
def z_nc                : SDNode<"SystemZISD::NC", SDT_ZMemMemLength,
                                  [SDNPHasChain, SDNPMayStore, SDNPMayLoad]>;
def z_nc_loop           : SDNode<"SystemZISD::NC_LOOP", SDT_ZMemMemLoop,
                                  [SDNPHasChain, SDNPMayStore, SDNPMayLoad]>;
def z_oc                : SDNode<"SystemZISD::OC", SDT_ZMemMemLength,
                                  [SDNPHasChain, SDNPMayStore, SDNPMayLoad]>;
def z_oc_loop           : SDNode<"SystemZISD::OC_LOOP", SDT_ZMemMemLoop,
                                  [SDNPHasChain, SDNPMayStore, SDNPMayLoad]>;
def z_xc                : SDNode<"SystemZISD::XC", SDT_ZMemMemLength,
                                  [SDNPHasChain, SDNPMayStore, SDNPMayLoad]>;
def z_xc_loop           : SDNode<"SystemZISD::XC_LOOP", SDT_ZMemMemLoop,
                                  [SDNPHasChain, SDNPMayStore, SDNPMayLoad]>;
def z_clc               : SDNode<"SystemZISD::CLC", SDT_ZMemMemLength,
                                 [SDNPHasChain, SDNPOutGlue, SDNPMayLoad]>;
def z_clc_loop          : SDNode<"SystemZISD::CLC_LOOP", SDT_ZMemMemLoop,
                                 [SDNPHasChain, SDNPOutGlue, SDNPMayLoad]>;
def z_strcmp            : SDNode<"SystemZISD::STRCMP", SDT_ZString,
                                 [SDNPHasChain, SDNPOutGlue, SDNPMayLoad]>;
def z_stpcpy            : SDNode<"SystemZISD::STPCPY", SDT_ZString,
                                 [SDNPHasChain, SDNPMayStore, SDNPMayLoad]>;
def z_search_string     : SDNode<"SystemZISD::SEARCH_STRING", SDT_ZString,
                                 [SDNPHasChain, SDNPOutGlue, SDNPMayLoad]>;
def z_ipm               : SDNode<"SystemZISD::IPM", SDT_ZI32Intrinsic,
                                 [SDNPInGlue]>;
def z_prefetch          : SDNode<"SystemZISD::PREFETCH", SDT_ZPrefetch,
                                 [SDNPHasChain, SDNPMayLoad, SDNPMayStore,
                                  SDNPMemOperand]>;

//===----------------------------------------------------------------------===//
// Pattern fragments
//===----------------------------------------------------------------------===//

// Signed and unsigned comparisons.
def z_scmp : PatFrag<(ops node:$a, node:$b), (z_icmp node:$a, node:$b, imm), [{
  unsigned Type = cast<ConstantSDNode>(N->getOperand(2))->getZExtValue();
  return Type != SystemZICMP::UnsignedOnly;
}]>;
def z_ucmp : PatFrag<(ops node:$a, node:$b), (z_icmp node:$a, node:$b, imm), [{
  unsigned Type = cast<ConstantSDNode>(N->getOperand(2))->getZExtValue();
  return Type != SystemZICMP::SignedOnly;
}]>;

// Register- and memory-based TEST UNDER MASK.
def z_tm_reg : PatFrag<(ops node:$a, node:$b), (z_tm node:$a, node:$b, imm)>;
def z_tm_mem : PatFrag<(ops node:$a, node:$b), (z_tm node:$a, node:$b, 0)>;

// Register sign-extend operations.  Sub-32-bit values are represented as i32s.
def sext8  : PatFrag<(ops node:$src), (sext_inreg node:$src, i8)>;
def sext16 : PatFrag<(ops node:$src), (sext_inreg node:$src, i16)>;
def sext32 : PatFrag<(ops node:$src), (sext (i32 node:$src))>;

// Register zero-extend operations.  Sub-32-bit values are represented as i32s.
def zext8  : PatFrag<(ops node:$src), (and node:$src, 0xff)>;
def zext16 : PatFrag<(ops node:$src), (and node:$src, 0xffff)>;
def zext32 : PatFrag<(ops node:$src), (zext (i32 node:$src))>;

// Typed floating-point loads.
def loadf32 : PatFrag<(ops node:$src), (f32 (load node:$src))>;
def loadf64 : PatFrag<(ops node:$src), (f64 (load node:$src))>;

// Extending loads in which the extension type can be signed.
def asextload : PatFrag<(ops node:$ptr), (unindexedload node:$ptr), [{
  unsigned Type = cast<LoadSDNode>(N)->getExtensionType();
  return Type == ISD::EXTLOAD || Type == ISD::SEXTLOAD;
}]>;
def asextloadi8 : PatFrag<(ops node:$ptr), (asextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def asextloadi16 : PatFrag<(ops node:$ptr), (asextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def asextloadi32 : PatFrag<(ops node:$ptr), (asextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;

// Extending loads in which the extension type can be unsigned.
def azextload : PatFrag<(ops node:$ptr), (unindexedload node:$ptr), [{
  unsigned Type = cast<LoadSDNode>(N)->getExtensionType();
  return Type == ISD::EXTLOAD || Type == ISD::ZEXTLOAD;
}]>;
def azextloadi8 : PatFrag<(ops node:$ptr), (azextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def azextloadi16 : PatFrag<(ops node:$ptr), (azextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def azextloadi32 : PatFrag<(ops node:$ptr), (azextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;

// Extending loads in which the extension type doesn't matter.
def anyextload : PatFrag<(ops node:$ptr), (unindexedload node:$ptr), [{
  return cast<LoadSDNode>(N)->getExtensionType() != ISD::NON_EXTLOAD;
}]>;
def anyextloadi8 : PatFrag<(ops node:$ptr), (anyextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def anyextloadi16 : PatFrag<(ops node:$ptr), (anyextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def anyextloadi32 : PatFrag<(ops node:$ptr), (anyextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;

// Aligned loads.
class AlignedLoad<SDPatternOperator load>
  : PatFrag<(ops node:$addr), (load node:$addr), [{
  auto *Load = cast<LoadSDNode>(N);
  return Load->getAlignment() >= Load->getMemoryVT().getStoreSize();
}]>;
def aligned_load         : AlignedLoad<load>;
def aligned_asextloadi16 : AlignedLoad<asextloadi16>;
def aligned_asextloadi32 : AlignedLoad<asextloadi32>;
def aligned_azextloadi16 : AlignedLoad<azextloadi16>;
def aligned_azextloadi32 : AlignedLoad<azextloadi32>;

// Aligned stores.
class AlignedStore<SDPatternOperator store>
  : PatFrag<(ops node:$src, node:$addr), (store node:$src, node:$addr), [{
  auto *Store = cast<StoreSDNode>(N);
  return Store->getAlignment() >= Store->getMemoryVT().getStoreSize();
}]>;
def aligned_store         : AlignedStore<store>;
def aligned_truncstorei16 : AlignedStore<truncstorei16>;
def aligned_truncstorei32 : AlignedStore<truncstorei32>;

// Non-volatile loads.  Used for instructions that might access the storage
// location multiple times.
class NonvolatileLoad<SDPatternOperator load>
  : PatFrag<(ops node:$addr), (load node:$addr), [{
  auto *Load = cast<LoadSDNode>(N);
  return !Load->isVolatile();
}]>;
def nonvolatile_load          : NonvolatileLoad<load>;
def nonvolatile_anyextloadi8  : NonvolatileLoad<anyextloadi8>;
def nonvolatile_anyextloadi16 : NonvolatileLoad<anyextloadi16>;
def nonvolatile_anyextloadi32 : NonvolatileLoad<anyextloadi32>;

// Non-volatile stores.
class NonvolatileStore<SDPatternOperator store>
  : PatFrag<(ops node:$src, node:$addr), (store node:$src, node:$addr), [{
  auto *Store = cast<StoreSDNode>(N);
  return !Store->isVolatile();
}]>;
def nonvolatile_store         : NonvolatileStore<store>;
def nonvolatile_truncstorei8  : NonvolatileStore<truncstorei8>;
def nonvolatile_truncstorei16 : NonvolatileStore<truncstorei16>;
def nonvolatile_truncstorei32 : NonvolatileStore<truncstorei32>;

// A store of a load that can be implemented using MVC.
def mvc_store : PatFrag<(ops node:$value, node:$addr),
                        (unindexedstore node:$value, node:$addr),
                        [{ return storeLoadCanUseMVC(N); }]>;

// Binary read-modify-write operations on memory in which the other
// operand is also memory and for which block operations like NC can
// be used.  There are two patterns for each operator, depending on
// which operand contains the "other" load.
multiclass block_op<SDPatternOperator operator> {
  def "1" : PatFrag<(ops node:$value, node:$addr),
                    (unindexedstore (operator node:$value,
                                              (unindexedload node:$addr)),
                                    node:$addr),
                    [{ return storeLoadCanUseBlockBinary(N, 0); }]>;
  def "2" : PatFrag<(ops node:$value, node:$addr),
                    (unindexedstore (operator (unindexedload node:$addr),
                                              node:$value),
                                    node:$addr),
                    [{ return storeLoadCanUseBlockBinary(N, 1); }]>;
}
defm block_and : block_op<and>;
defm block_or  : block_op<or>;
defm block_xor : block_op<xor>;

// Insertions.
def inserti8 : PatFrag<(ops node:$src1, node:$src2),
                       (or (and node:$src1, -256), node:$src2)>;
def insertll : PatFrag<(ops node:$src1, node:$src2),
                       (or (and node:$src1, 0xffffffffffff0000), node:$src2)>;
def insertlh : PatFrag<(ops node:$src1, node:$src2),
                       (or (and node:$src1, 0xffffffff0000ffff), node:$src2)>;
def inserthl : PatFrag<(ops node:$src1, node:$src2),
                       (or (and node:$src1, 0xffff0000ffffffff), node:$src2)>;
def inserthh : PatFrag<(ops node:$src1, node:$src2),
                       (or (and node:$src1, 0x0000ffffffffffff), node:$src2)>;
def insertlf : PatFrag<(ops node:$src1, node:$src2),
                       (or (and node:$src1, 0xffffffff00000000), node:$src2)>;
def inserthf : PatFrag<(ops node:$src1, node:$src2),
                       (or (and node:$src1, 0x00000000ffffffff), node:$src2)>;

// ORs that can be treated as insertions.
def or_as_inserti8 : PatFrag<(ops node:$src1, node:$src2),
                             (or node:$src1, node:$src2), [{
  unsigned BitWidth = N->getValueType(0).getScalarType().getSizeInBits();
  return CurDAG->MaskedValueIsZero(N->getOperand(0),
                                   APInt::getLowBitsSet(BitWidth, 8));
}]>;

// ORs that can be treated as reversed insertions.
def or_as_revinserti8 : PatFrag<(ops node:$src1, node:$src2),
                                (or node:$src1, node:$src2), [{
  unsigned BitWidth = N->getValueType(0).getScalarType().getSizeInBits();
  return CurDAG->MaskedValueIsZero(N->getOperand(1),
                                   APInt::getLowBitsSet(BitWidth, 8));
}]>;

// Negative integer absolute.
def z_inegabs : PatFrag<(ops node:$src), (ineg (z_iabs node:$src))>;

// Integer absolute, matching the canonical form generated by DAGCombiner.
def z_iabs32 : PatFrag<(ops node:$src),
                       (xor (add node:$src, (sra node:$src, (i32 31))),
                            (sra node:$src, (i32 31)))>;
def z_iabs64 : PatFrag<(ops node:$src),
                       (xor (add node:$src, (sra node:$src, (i32 63))),
                            (sra node:$src, (i32 63)))>;
def z_inegabs32 : PatFrag<(ops node:$src), (ineg (z_iabs32 node:$src))>;
def z_inegabs64 : PatFrag<(ops node:$src), (ineg (z_iabs64 node:$src))>;

// Fused multiply-add and multiply-subtract, but with the order of the
// operands matching SystemZ's MA and MS instructions.
def z_fma : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                    (fma node:$src2, node:$src3, node:$src1)>;
def z_fms : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                    (fma node:$src2, node:$src3, (fneg node:$src1))>;

// Floating-point negative absolute.
def fnabs : PatFrag<(ops node:$ptr), (fneg (fabs node:$ptr))>;

// Create a unary operator that loads from memory and then performs
// the given operation on it.
class loadu<SDPatternOperator operator, SDPatternOperator load = load>
  : PatFrag<(ops node:$addr), (operator (load node:$addr))>;

// Create a store operator that performs the given unary operation
// on the value before storing it.
class storeu<SDPatternOperator operator, SDPatternOperator store = store>
  : PatFrag<(ops node:$value, node:$addr),
            (store (operator node:$value), node:$addr)>;
