//===-- TargetLowering.cpp - Implement the TargetLowering class -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the TargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtarget.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

/// InitLibcallNames - Set default libcall names.
///
static void InitLibcallNames(const char **Names) {
  Names[RTLIB::SHL_I32] = "__ashlsi3";
  Names[RTLIB::SHL_I64] = "__ashldi3";
  Names[RTLIB::SHL_I128] = "__ashlti3";
  Names[RTLIB::SRL_I32] = "__lshrsi3";
  Names[RTLIB::SRL_I64] = "__lshrdi3";
  Names[RTLIB::SRL_I128] = "__lshrti3";
  Names[RTLIB::SRA_I32] = "__ashrsi3";
  Names[RTLIB::SRA_I64] = "__ashrdi3";
  Names[RTLIB::SRA_I128] = "__ashrti3";
  Names[RTLIB::MUL_I32] = "__mulsi3";
  Names[RTLIB::MUL_I64] = "__muldi3";
  Names[RTLIB::MUL_I128] = "__multi3";
  Names[RTLIB::SDIV_I32] = "__divsi3";
  Names[RTLIB::SDIV_I64] = "__divdi3";
  Names[RTLIB::SDIV_I128] = "__divti3";
  Names[RTLIB::UDIV_I32] = "__udivsi3";
  Names[RTLIB::UDIV_I64] = "__udivdi3";
  Names[RTLIB::UDIV_I128] = "__udivti3";
  Names[RTLIB::SREM_I32] = "__modsi3";
  Names[RTLIB::SREM_I64] = "__moddi3";
  Names[RTLIB::SREM_I128] = "__modti3";
  Names[RTLIB::UREM_I32] = "__umodsi3";
  Names[RTLIB::UREM_I64] = "__umoddi3";
  Names[RTLIB::UREM_I128] = "__umodti3";
  Names[RTLIB::NEG_I32] = "__negsi2";
  Names[RTLIB::NEG_I64] = "__negdi2";
  Names[RTLIB::ADD_F32] = "__addsf3";
  Names[RTLIB::ADD_F64] = "__adddf3";
  Names[RTLIB::ADD_F80] = "__addxf3";
  Names[RTLIB::ADD_PPCF128] = "__gcc_qadd";
  Names[RTLIB::SUB_F32] = "__subsf3";
  Names[RTLIB::SUB_F64] = "__subdf3";
  Names[RTLIB::SUB_F80] = "__subxf3";
  Names[RTLIB::SUB_PPCF128] = "__gcc_qsub";
  Names[RTLIB::MUL_F32] = "__mulsf3";
  Names[RTLIB::MUL_F64] = "__muldf3";
  Names[RTLIB::MUL_F80] = "__mulxf3";
  Names[RTLIB::MUL_PPCF128] = "__gcc_qmul";
  Names[RTLIB::DIV_F32] = "__divsf3";
  Names[RTLIB::DIV_F64] = "__divdf3";
  Names[RTLIB::DIV_F80] = "__divxf3";
  Names[RTLIB::DIV_PPCF128] = "__gcc_qdiv";
  Names[RTLIB::REM_F32] = "fmodf";
  Names[RTLIB::REM_F64] = "fmod";
  Names[RTLIB::REM_F80] = "fmodl";
  Names[RTLIB::REM_PPCF128] = "fmodl";
  Names[RTLIB::POWI_F32] = "__powisf2";
  Names[RTLIB::POWI_F64] = "__powidf2";
  Names[RTLIB::POWI_F80] = "__powixf2";
  Names[RTLIB::POWI_PPCF128] = "__powitf2";
  Names[RTLIB::SQRT_F32] = "sqrtf";
  Names[RTLIB::SQRT_F64] = "sqrt";
  Names[RTLIB::SQRT_F80] = "sqrtl";
  Names[RTLIB::SQRT_PPCF128] = "sqrtl";
  Names[RTLIB::SIN_F32] = "sinf";
  Names[RTLIB::SIN_F64] = "sin";
  Names[RTLIB::SIN_F80] = "sinl";
  Names[RTLIB::SIN_PPCF128] = "sinl";
  Names[RTLIB::COS_F32] = "cosf";
  Names[RTLIB::COS_F64] = "cos";
  Names[RTLIB::COS_F80] = "cosl";
  Names[RTLIB::COS_PPCF128] = "cosl";
  Names[RTLIB::POW_F32] = "powf";
  Names[RTLIB::POW_F64] = "pow";
  Names[RTLIB::POW_F80] = "powl";
  Names[RTLIB::POW_PPCF128] = "powl";
  Names[RTLIB::FPEXT_F32_F64] = "__extendsfdf2";
  Names[RTLIB::FPROUND_F64_F32] = "__truncdfsf2";
  Names[RTLIB::FPTOSINT_F32_I32] = "__fixsfsi";
  Names[RTLIB::FPTOSINT_F32_I64] = "__fixsfdi";
  Names[RTLIB::FPTOSINT_F32_I128] = "__fixsfti";
  Names[RTLIB::FPTOSINT_F64_I32] = "__fixdfsi";
  Names[RTLIB::FPTOSINT_F64_I64] = "__fixdfdi";
  Names[RTLIB::FPTOSINT_F64_I128] = "__fixdfti";
  Names[RTLIB::FPTOSINT_F80_I32] = "__fixxfsi";
  Names[RTLIB::FPTOSINT_F80_I64] = "__fixxfdi";
  Names[RTLIB::FPTOSINT_F80_I128] = "__fixxfti";
  Names[RTLIB::FPTOSINT_PPCF128_I32] = "__fixtfsi";
  Names[RTLIB::FPTOSINT_PPCF128_I64] = "__fixtfdi";
  Names[RTLIB::FPTOSINT_PPCF128_I128] = "__fixtfti";
  Names[RTLIB::FPTOUINT_F32_I32] = "__fixunssfsi";
  Names[RTLIB::FPTOUINT_F32_I64] = "__fixunssfdi";
  Names[RTLIB::FPTOUINT_F32_I128] = "__fixunssfti";
  Names[RTLIB::FPTOUINT_F64_I32] = "__fixunsdfsi";
  Names[RTLIB::FPTOUINT_F64_I64] = "__fixunsdfdi";
  Names[RTLIB::FPTOUINT_F64_I128] = "__fixunsdfti";
  Names[RTLIB::FPTOUINT_F80_I32] = "__fixunsxfsi";
  Names[RTLIB::FPTOUINT_F80_I64] = "__fixunsxfdi";
  Names[RTLIB::FPTOUINT_F80_I128] = "__fixunsxfti";
  Names[RTLIB::FPTOUINT_PPCF128_I32] = "__fixunstfsi";
  Names[RTLIB::FPTOUINT_PPCF128_I64] = "__fixunstfdi";
  Names[RTLIB::FPTOUINT_PPCF128_I128] = "__fixunstfti";
  Names[RTLIB::SINTTOFP_I32_F32] = "__floatsisf";
  Names[RTLIB::SINTTOFP_I32_F64] = "__floatsidf";
  Names[RTLIB::SINTTOFP_I64_F32] = "__floatdisf";
  Names[RTLIB::SINTTOFP_I64_F64] = "__floatdidf";
  Names[RTLIB::SINTTOFP_I64_F80] = "__floatdixf";
  Names[RTLIB::SINTTOFP_I64_PPCF128] = "__floatditf";
  Names[RTLIB::SINTTOFP_I128_F32] = "__floattisf";
  Names[RTLIB::SINTTOFP_I128_F64] = "__floattidf";
  Names[RTLIB::SINTTOFP_I128_F80] = "__floattixf";
  Names[RTLIB::SINTTOFP_I128_PPCF128] = "__floattitf";
  Names[RTLIB::UINTTOFP_I32_F32] = "__floatunsisf";
  Names[RTLIB::UINTTOFP_I32_F64] = "__floatunsidf";
  Names[RTLIB::UINTTOFP_I64_F32] = "__floatundisf";
  Names[RTLIB::UINTTOFP_I64_F64] = "__floatundidf";
  Names[RTLIB::OEQ_F32] = "__eqsf2";
  Names[RTLIB::OEQ_F64] = "__eqdf2";
  Names[RTLIB::UNE_F32] = "__nesf2";
  Names[RTLIB::UNE_F64] = "__nedf2";
  Names[RTLIB::OGE_F32] = "__gesf2";
  Names[RTLIB::OGE_F64] = "__gedf2";
  Names[RTLIB::OLT_F32] = "__ltsf2";
  Names[RTLIB::OLT_F64] = "__ltdf2";
  Names[RTLIB::OLE_F32] = "__lesf2";
  Names[RTLIB::OLE_F64] = "__ledf2";
  Names[RTLIB::OGT_F32] = "__gtsf2";
  Names[RTLIB::OGT_F64] = "__gtdf2";
  Names[RTLIB::UO_F32] = "__unordsf2";
  Names[RTLIB::UO_F64] = "__unorddf2";
  Names[RTLIB::O_F32] = "__unordsf2";
  Names[RTLIB::O_F64] = "__unorddf2";
}

/// InitCmpLibcallCCs - Set default comparison libcall CC.
///
static void InitCmpLibcallCCs(ISD::CondCode *CCs) {
  memset(CCs, ISD::SETCC_INVALID, sizeof(ISD::CondCode)*RTLIB::UNKNOWN_LIBCALL);
  CCs[RTLIB::OEQ_F32] = ISD::SETEQ;
  CCs[RTLIB::OEQ_F64] = ISD::SETEQ;
  CCs[RTLIB::UNE_F32] = ISD::SETNE;
  CCs[RTLIB::UNE_F64] = ISD::SETNE;
  CCs[RTLIB::OGE_F32] = ISD::SETGE;
  CCs[RTLIB::OGE_F64] = ISD::SETGE;
  CCs[RTLIB::OLT_F32] = ISD::SETLT;
  CCs[RTLIB::OLT_F64] = ISD::SETLT;
  CCs[RTLIB::OLE_F32] = ISD::SETLE;
  CCs[RTLIB::OLE_F64] = ISD::SETLE;
  CCs[RTLIB::OGT_F32] = ISD::SETGT;
  CCs[RTLIB::OGT_F64] = ISD::SETGT;
  CCs[RTLIB::UO_F32] = ISD::SETNE;
  CCs[RTLIB::UO_F64] = ISD::SETNE;
  CCs[RTLIB::O_F32] = ISD::SETEQ;
  CCs[RTLIB::O_F64] = ISD::SETEQ;
}

TargetLowering::TargetLowering(TargetMachine &tm)
  : TM(tm), TD(TM.getTargetData()) {
  assert(ISD::BUILTIN_OP_END <= OpActionsCapacity &&
         "Fixed size array in TargetLowering is not large enough!");
  // All operations default to being supported.
  memset(OpActions, 0, sizeof(OpActions));
  memset(LoadXActions, 0, sizeof(LoadXActions));
  memset(TruncStoreActions, 0, sizeof(TruncStoreActions));
  memset(IndexedModeActions, 0, sizeof(IndexedModeActions));
  memset(ConvertActions, 0, sizeof(ConvertActions));

  // Set default actions for various operations.
  for (unsigned VT = 0; VT != (unsigned)MVT::LAST_VALUETYPE; ++VT) {
    // Default all indexed load / store to expand.
    for (unsigned IM = (unsigned)ISD::PRE_INC;
         IM != (unsigned)ISD::LAST_INDEXED_MODE; ++IM) {
      setIndexedLoadAction(IM, (MVT::SimpleValueType)VT, Expand);
      setIndexedStoreAction(IM, (MVT::SimpleValueType)VT, Expand);
    }
    
    // These operations default to expand.
    setOperationAction(ISD::FGETSIGN, (MVT::SimpleValueType)VT, Expand);
  }

  // Most targets ignore the @llvm.prefetch intrinsic.
  setOperationAction(ISD::PREFETCH, MVT::Other, Expand);
  
  // ConstantFP nodes default to expand.  Targets can either change this to 
  // Legal, in which case all fp constants are legal, or use addLegalFPImmediate
  // to optimize expansions for certain constants.
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f80, Expand);

  // Default ISD::TRAP to expand (which turns it into abort).
  setOperationAction(ISD::TRAP, MVT::Other, Expand);
    
  IsLittleEndian = TD->isLittleEndian();
  UsesGlobalOffsetTable = false;
  ShiftAmountTy = PointerTy = getValueType(TD->getIntPtrType());
  ShiftAmtHandling = Undefined;
  memset(RegClassForVT, 0,MVT::LAST_VALUETYPE*sizeof(TargetRegisterClass*));
  memset(TargetDAGCombineArray, 0, array_lengthof(TargetDAGCombineArray));
  maxStoresPerMemset = maxStoresPerMemcpy = maxStoresPerMemmove = 8;
  allowUnalignedMemoryAccesses = false;
  UseUnderscoreSetJmp = false;
  UseUnderscoreLongJmp = false;
  SelectIsExpensive = false;
  IntDivIsCheap = false;
  Pow2DivIsCheap = false;
  StackPointerRegisterToSaveRestore = 0;
  ExceptionPointerRegister = 0;
  ExceptionSelectorRegister = 0;
  SetCCResultContents = UndefinedSetCCResult;
  SchedPreferenceInfo = SchedulingForLatency;
  JumpBufSize = 0;
  JumpBufAlignment = 0;
  IfCvtBlockSizeLimit = 2;
  IfCvtDupBlockSizeLimit = 0;
  PrefLoopAlignment = 0;

  InitLibcallNames(LibcallRoutineNames);
  InitCmpLibcallCCs(CmpLibcallCCs);

  // Tell Legalize whether the assembler supports DEBUG_LOC.
  if (!TM.getTargetAsmInfo()->hasDotLocAndDotFile())
    setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
}

TargetLowering::~TargetLowering() {}

/// computeRegisterProperties - Once all of the register classes are added,
/// this allows us to compute derived properties we expose.
void TargetLowering::computeRegisterProperties() {
  assert(MVT::LAST_VALUETYPE <= 32 &&
         "Too many value types for ValueTypeActions to hold!");

  // Everything defaults to needing one register.
  for (unsigned i = 0; i != MVT::LAST_VALUETYPE; ++i) {
    NumRegistersForVT[i] = 1;
    RegisterTypeForVT[i] = TransformToType[i] = (MVT::SimpleValueType)i;
  }
  // ...except isVoid, which doesn't need any registers.
  NumRegistersForVT[MVT::isVoid] = 0;

  // Find the largest integer register class.
  unsigned LargestIntReg = MVT::LAST_INTEGER_VALUETYPE;
  for (; RegClassForVT[LargestIntReg] == 0; --LargestIntReg)
    assert(LargestIntReg != MVT::i1 && "No integer registers defined!");

  // Every integer value type larger than this largest register takes twice as
  // many registers to represent as the previous ValueType.
  for (unsigned ExpandedReg = LargestIntReg + 1; ; ++ExpandedReg) {
    MVT EVT = (MVT::SimpleValueType)ExpandedReg;
    if (!EVT.isInteger())
      break;
    NumRegistersForVT[ExpandedReg] = 2*NumRegistersForVT[ExpandedReg-1];
    RegisterTypeForVT[ExpandedReg] = (MVT::SimpleValueType)LargestIntReg;
    TransformToType[ExpandedReg] = (MVT::SimpleValueType)(ExpandedReg - 1);
    ValueTypeActions.setTypeAction(EVT, Expand);
  }

  // Inspect all of the ValueType's smaller than the largest integer
  // register to see which ones need promotion.
  unsigned LegalIntReg = LargestIntReg;
  for (unsigned IntReg = LargestIntReg - 1;
       IntReg >= (unsigned)MVT::i1; --IntReg) {
    MVT IVT = (MVT::SimpleValueType)IntReg;
    if (isTypeLegal(IVT)) {
      LegalIntReg = IntReg;
    } else {
      RegisterTypeForVT[IntReg] = TransformToType[IntReg] =
        (MVT::SimpleValueType)LegalIntReg;
      ValueTypeActions.setTypeAction(IVT, Promote);
    }
  }

  // ppcf128 type is really two f64's.
  if (!isTypeLegal(MVT::ppcf128)) {
    NumRegistersForVT[MVT::ppcf128] = 2*NumRegistersForVT[MVT::f64];
    RegisterTypeForVT[MVT::ppcf128] = MVT::f64;
    TransformToType[MVT::ppcf128] = MVT::f64;
    ValueTypeActions.setTypeAction(MVT::ppcf128, Expand);
  }    

  // Decide how to handle f64. If the target does not have native f64 support,
  // expand it to i64 and we will be generating soft float library calls.
  if (!isTypeLegal(MVT::f64)) {
    NumRegistersForVT[MVT::f64] = NumRegistersForVT[MVT::i64];
    RegisterTypeForVT[MVT::f64] = RegisterTypeForVT[MVT::i64];
    TransformToType[MVT::f64] = MVT::i64;
    ValueTypeActions.setTypeAction(MVT::f64, Expand);
  }

  // Decide how to handle f32. If the target does not have native support for
  // f32, promote it to f64 if it is legal. Otherwise, expand it to i32.
  if (!isTypeLegal(MVT::f32)) {
    if (isTypeLegal(MVT::f64)) {
      NumRegistersForVT[MVT::f32] = NumRegistersForVT[MVT::f64];
      RegisterTypeForVT[MVT::f32] = RegisterTypeForVT[MVT::f64];
      TransformToType[MVT::f32] = MVT::f64;
      ValueTypeActions.setTypeAction(MVT::f32, Promote);
    } else {
      NumRegistersForVT[MVT::f32] = NumRegistersForVT[MVT::i32];
      RegisterTypeForVT[MVT::f32] = RegisterTypeForVT[MVT::i32];
      TransformToType[MVT::f32] = MVT::i32;
      ValueTypeActions.setTypeAction(MVT::f32, Expand);
    }
  }
  
  // Loop over all of the vector value types to see which need transformations.
  for (unsigned i = MVT::FIRST_VECTOR_VALUETYPE;
       i <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++i) {
    MVT VT = (MVT::SimpleValueType)i;
    if (!isTypeLegal(VT)) {
      MVT IntermediateVT, RegisterVT;
      unsigned NumIntermediates;
      NumRegistersForVT[i] =
        getVectorTypeBreakdown(VT,
                               IntermediateVT, NumIntermediates,
                               RegisterVT);
      RegisterTypeForVT[i] = RegisterVT;
      TransformToType[i] = MVT::Other; // this isn't actually used
      ValueTypeActions.setTypeAction(VT, Expand);
    }
  }
}

const char *TargetLowering::getTargetNodeName(unsigned Opcode) const {
  return NULL;
}


MVT TargetLowering::getSetCCResultType(const SDOperand &) const {
  return getValueType(TD->getIntPtrType());
}


/// getVectorTypeBreakdown - Vector types are broken down into some number of
/// legal first class types.  For example, MVT::v8f32 maps to 2 MVT::v4f32
/// with Altivec or SSE1, or 8 promoted MVT::f64 values with the X86 FP stack.
/// Similarly, MVT::v2i64 turns into 4 MVT::i32 values with both PPC and X86.
///
/// This method returns the number of registers needed, and the VT for each
/// register.  It also returns the VT and quantity of the intermediate values
/// before they are promoted/expanded.
///
unsigned TargetLowering::getVectorTypeBreakdown(MVT VT,
                                                MVT &IntermediateVT,
                                                unsigned &NumIntermediates,
                                      MVT &RegisterVT) const {
  // Figure out the right, legal destination reg to copy into.
  unsigned NumElts = VT.getVectorNumElements();
  MVT EltTy = VT.getVectorElementType();
  
  unsigned NumVectorRegs = 1;
  
  // FIXME: We don't support non-power-of-2-sized vectors for now.  Ideally we 
  // could break down into LHS/RHS like LegalizeDAG does.
  if (!isPowerOf2_32(NumElts)) {
    NumVectorRegs = NumElts;
    NumElts = 1;
  }
  
  // Divide the input until we get to a supported size.  This will always
  // end with a scalar if the target doesn't support vectors.
  while (NumElts > 1 && !isTypeLegal(MVT::getVectorVT(EltTy, NumElts))) {
    NumElts >>= 1;
    NumVectorRegs <<= 1;
  }

  NumIntermediates = NumVectorRegs;
  
  MVT NewVT = MVT::getVectorVT(EltTy, NumElts);
  if (!isTypeLegal(NewVT))
    NewVT = EltTy;
  IntermediateVT = NewVT;

  MVT DestVT = getTypeToTransformTo(NewVT);
  RegisterVT = DestVT;
  if (DestVT.bitsLT(NewVT)) {
    // Value is expanded, e.g. i64 -> i16.
    return NumVectorRegs*(NewVT.getSizeInBits()/DestVT.getSizeInBits());
  } else {
    // Otherwise, promotion or legal types use the same number of registers as
    // the vector decimated to the appropriate level.
    return NumVectorRegs;
  }
  
  return 1;
}

/// getByValTypeAlignment - Return the desired alignment for ByVal aggregate
/// function arguments in the caller parameter area.  This is the actual
/// alignment, not its logarithm.
unsigned TargetLowering::getByValTypeAlignment(const Type *Ty) const {
  return TD->getCallFrameTypeAlignment(Ty);
}

SDOperand TargetLowering::getPICJumpTableRelocBase(SDOperand Table,
                                                   SelectionDAG &DAG) const {
  if (usesGlobalOffsetTable())
    return DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, getPointerTy());
  return Table;
}

//===----------------------------------------------------------------------===//
//  Optimization Methods
//===----------------------------------------------------------------------===//

/// ShrinkDemandedConstant - Check to see if the specified operand of the 
/// specified instruction is a constant integer.  If so, check to see if there
/// are any bits set in the constant that are not demanded.  If so, shrink the
/// constant and return true.
bool TargetLowering::TargetLoweringOpt::ShrinkDemandedConstant(SDOperand Op, 
                                                        const APInt &Demanded) {
  // FIXME: ISD::SELECT, ISD::SELECT_CC
  switch(Op.getOpcode()) {
  default: break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      if (C->getAPIntValue().intersects(~Demanded)) {
        MVT VT = Op.getValueType();
        SDOperand New = DAG.getNode(Op.getOpcode(), VT, Op.getOperand(0),
                                    DAG.getConstant(Demanded &
                                                      C->getAPIntValue(), 
                                                    VT));
        return CombineTo(Op, New);
      }
    break;
  }
  return false;
}

/// SimplifyDemandedBits - Look at Op.  At this point, we know that only the
/// DemandedMask bits of the result of Op are ever used downstream.  If we can
/// use this information to simplify Op, create a new simplified DAG node and
/// return true, returning the original and new nodes in Old and New. Otherwise,
/// analyze the expression and return a mask of KnownOne and KnownZero bits for
/// the expression (used to simplify the caller).  The KnownZero/One bits may
/// only be accurate for those bits in the DemandedMask.
bool TargetLowering::SimplifyDemandedBits(SDOperand Op,
                                          const APInt &DemandedMask,
                                          APInt &KnownZero,
                                          APInt &KnownOne,
                                          TargetLoweringOpt &TLO,
                                          unsigned Depth) const {
  unsigned BitWidth = DemandedMask.getBitWidth();
  assert(Op.getValueSizeInBits() == BitWidth &&
         "Mask size mismatches value type size!");
  APInt NewMask = DemandedMask;

  // Don't know anything.
  KnownZero = KnownOne = APInt(BitWidth, 0);

  // Other users may use these bits.
  if (!Op.Val->hasOneUse()) { 
    if (Depth != 0) {
      // If not at the root, Just compute the KnownZero/KnownOne bits to 
      // simplify things downstream.
      TLO.DAG.ComputeMaskedBits(Op, DemandedMask, KnownZero, KnownOne, Depth);
      return false;
    }
    // If this is the root being simplified, allow it to have multiple uses,
    // just set the NewMask to all bits.
    NewMask = APInt::getAllOnesValue(BitWidth);
  } else if (DemandedMask == 0) {   
    // Not demanding any bits from Op.
    if (Op.getOpcode() != ISD::UNDEF)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::UNDEF, Op.getValueType()));
    return false;
  } else if (Depth == 6) {        // Limit search depth.
    return false;
  }

  APInt KnownZero2, KnownOne2, KnownZeroOut, KnownOneOut;
  switch (Op.getOpcode()) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    KnownOne = cast<ConstantSDNode>(Op)->getAPIntValue() & NewMask;
    KnownZero = ~KnownOne & NewMask;
    return false;   // Don't fall through, will infinitely loop.
  case ISD::AND:
    // If the RHS is a constant, check to see if the LHS would be zero without
    // using the bits from the RHS.  Below, we use knowledge about the RHS to
    // simplify the LHS, here we're using information from the LHS to simplify
    // the RHS.
    if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      APInt LHSZero, LHSOne;
      TLO.DAG.ComputeMaskedBits(Op.getOperand(0), NewMask,
                                LHSZero, LHSOne, Depth+1);
      // If the LHS already has zeros where RHSC does, this and is dead.
      if ((LHSZero & NewMask) == (~RHSC->getAPIntValue() & NewMask))
        return TLO.CombineTo(Op, Op.getOperand(0));
      // If any of the set bits in the RHS are known zero on the LHS, shrink
      // the constant.
      if (TLO.ShrinkDemandedConstant(Op, ~LHSZero & NewMask))
        return true;
    }
    
    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(Op.getOperand(0), ~KnownZero & NewMask,
                             KnownZero2, KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
      
    // If all of the demanded bits are known one on one side, return the other.
    // These bits cannot contribute to the result of the 'and'.
    if ((NewMask & ~KnownZero2 & KnownOne) == (~KnownZero2 & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((NewMask & ~KnownZero & KnownOne2) == (~KnownZero & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If all of the demanded bits in the inputs are known zeros, return zero.
    if ((NewMask & (KnownZero|KnownZero2)) == NewMask)
      return TLO.CombineTo(Op, TLO.DAG.getConstant(0, Op.getValueType()));
    // If the RHS is a constant, see if we can simplify it.
    if (TLO.ShrinkDemandedConstant(Op, ~KnownZero2 & NewMask))
      return true;
      
    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    break;
  case ISD::OR:
    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(Op.getOperand(0), ~KnownOne & NewMask,
                             KnownZero2, KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'or'.
    if ((NewMask & ~KnownOne2 & KnownZero) == (~KnownOne2 & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((NewMask & ~KnownOne & KnownZero2) == (~KnownOne & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If all of the potentially set bits on one side are known to be set on
    // the other side, just use the 'other' side.
    if ((NewMask & ~KnownZero & KnownOne2) == (~KnownZero & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((NewMask & ~KnownZero2 & KnownOne) == (~KnownZero2 & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If the RHS is a constant, see if we can simplify it.
    if (TLO.ShrinkDemandedConstant(Op, NewMask))
      return true;
          
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    break;
  case ISD::XOR:
    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(Op.getOperand(0), NewMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'xor'.
    if ((KnownZero & NewMask) == NewMask)
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((KnownZero2 & NewMask) == NewMask)
      return TLO.CombineTo(Op, Op.getOperand(1));
      
    // If all of the unknown bits are known to be zero on one side or the other
    // (but not both) turn this into an *inclusive* or.
    //    e.g. (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
    if ((NewMask & ~KnownZero & ~KnownZero2) == 0)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::OR, Op.getValueType(),
                                               Op.getOperand(0),
                                               Op.getOperand(1)));
    
    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOneOut = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    
    // If all of the demanded bits on one side are known, and all of the set
    // bits on that side are also known to be set on the other side, turn this
    // into an AND, as we know the bits will be cleared.
    //    e.g. (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
    if ((NewMask & (KnownZero|KnownOne)) == NewMask) { // all known
      if ((KnownOne & KnownOne2) == KnownOne) {
        MVT VT = Op.getValueType();
        SDOperand ANDC = TLO.DAG.getConstant(~KnownOne & NewMask, VT);
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::AND, VT, Op.getOperand(0),
                                                 ANDC));
      }
    }
    
    // If the RHS is a constant, see if we can simplify it.
    // for XOR, we prefer to force bits to 1 if they will make a -1.
    // if we can't force bits, try to shrink constant
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      APInt Expanded = C->getAPIntValue() | (~NewMask);
      // if we can expand it to have all bits set, do it
      if (Expanded.isAllOnesValue()) {
        if (Expanded != C->getAPIntValue()) {
          MVT VT = Op.getValueType();
          SDOperand New = TLO.DAG.getNode(Op.getOpcode(), VT, Op.getOperand(0),
                                          TLO.DAG.getConstant(Expanded, VT));
          return TLO.CombineTo(Op, New);
        }
        // if it already has all the bits set, nothing to change
        // but don't shrink either!
      } else if (TLO.ShrinkDemandedConstant(Op, NewMask)) {
        return true;
      }
    }

    KnownZero = KnownZeroOut;
    KnownOne  = KnownOneOut;
    break;
  case ISD::SELECT:
    if (SimplifyDemandedBits(Op.getOperand(2), NewMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If the operands are constants, see if we can simplify them.
    if (TLO.ShrinkDemandedConstant(Op, NewMask))
      return true;
    
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SELECT_CC:
    if (SimplifyDemandedBits(Op.getOperand(3), NewMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(2), NewMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If the operands are constants, see if we can simplify them.
    if (TLO.ShrinkDemandedConstant(Op, NewMask))
      return true;
      
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SHL:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getValue();
      SDOperand InOp = Op.getOperand(0);

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      // If this is ((X >>u C1) << ShAmt), see if we can simplify this into a
      // single shift.  We can do this if the bottom bits (which are shifted
      // out) are never demanded.
      if (InOp.getOpcode() == ISD::SRL &&
          isa<ConstantSDNode>(InOp.getOperand(1))) {
        if (ShAmt && (NewMask & APInt::getLowBitsSet(BitWidth, ShAmt)) == 0) {
          unsigned C1 = cast<ConstantSDNode>(InOp.getOperand(1))->getValue();
          unsigned Opc = ISD::SHL;
          int Diff = ShAmt-C1;
          if (Diff < 0) {
            Diff = -Diff;
            Opc = ISD::SRL;
          }          
          
          SDOperand NewSA = 
            TLO.DAG.getConstant(Diff, Op.getOperand(1).getValueType());
          MVT VT = Op.getValueType();
          return TLO.CombineTo(Op, TLO.DAG.getNode(Opc, VT,
                                                   InOp.getOperand(0), NewSA));
        }
      }      
      
      if (SimplifyDemandedBits(Op.getOperand(0), NewMask.lshr(ShAmt),
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      KnownZero <<= SA->getValue();
      KnownOne  <<= SA->getValue();
      // low bits known zero.
      KnownZero |= APInt::getLowBitsSet(BitWidth, SA->getValue());
    }
    break;
  case ISD::SRL:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      MVT VT = Op.getValueType();
      unsigned ShAmt = SA->getValue();
      unsigned VTSize = VT.getSizeInBits();
      SDOperand InOp = Op.getOperand(0);
      
      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      // If this is ((X << C1) >>u ShAmt), see if we can simplify this into a
      // single shift.  We can do this if the top bits (which are shifted out)
      // are never demanded.
      if (InOp.getOpcode() == ISD::SHL &&
          isa<ConstantSDNode>(InOp.getOperand(1))) {
        if (ShAmt && (NewMask & APInt::getHighBitsSet(VTSize, ShAmt)) == 0) {
          unsigned C1 = cast<ConstantSDNode>(InOp.getOperand(1))->getValue();
          unsigned Opc = ISD::SRL;
          int Diff = ShAmt-C1;
          if (Diff < 0) {
            Diff = -Diff;
            Opc = ISD::SHL;
          }          
          
          SDOperand NewSA =
            TLO.DAG.getConstant(Diff, Op.getOperand(1).getValueType());
          return TLO.CombineTo(Op, TLO.DAG.getNode(Opc, VT,
                                                   InOp.getOperand(0), NewSA));
        }
      }      
      
      // Compute the new bits that are at the top now.
      if (SimplifyDemandedBits(InOp, (NewMask << ShAmt),
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);

      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt);
      KnownZero |= HighBits;  // High bits known zero.
    }
    break;
  case ISD::SRA:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      MVT VT = Op.getValueType();
      unsigned ShAmt = SA->getValue();
      
      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      APInt InDemandedMask = (NewMask << ShAmt);

      // If any of the demanded bits are produced by the sign extension, we also
      // demand the input sign bit.
      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt);
      if (HighBits.intersects(NewMask))
        InDemandedMask |= APInt::getSignBit(VT.getSizeInBits());
      
      if (SimplifyDemandedBits(Op.getOperand(0), InDemandedMask,
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);
      
      // Handle the sign bit, adjusted to where it is now in the mask.
      APInt SignBit = APInt::getSignBit(BitWidth).lshr(ShAmt);
      
      // If the input sign bit is known to be zero, or if none of the top bits
      // are demanded, turn this into an unsigned shift right.
      if (KnownZero.intersects(SignBit) || (HighBits & ~NewMask) == HighBits) {
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SRL, VT, Op.getOperand(0),
                                                 Op.getOperand(1)));
      } else if (KnownOne.intersects(SignBit)) { // New bits are known one.
        KnownOne |= HighBits;
      }
    }
    break;
  case ISD::SIGN_EXTEND_INREG: {
    MVT EVT = cast<VTSDNode>(Op.getOperand(1))->getVT();

    // Sign extension.  Compute the demanded bits in the result that are not 
    // present in the input.
    APInt NewBits = APInt::getHighBitsSet(BitWidth,
                                          BitWidth - EVT.getSizeInBits()) &
                    NewMask;
    
    // If none of the extended bits are demanded, eliminate the sextinreg.
    if (NewBits == 0)
      return TLO.CombineTo(Op, Op.getOperand(0));

    APInt InSignBit = APInt::getSignBit(EVT.getSizeInBits());
    InSignBit.zext(BitWidth);
    APInt InputDemandedBits = APInt::getLowBitsSet(BitWidth,
                                                   EVT.getSizeInBits()) &
                              NewMask;
    
    // Since the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    InputDemandedBits |= InSignBit;

    if (SimplifyDemandedBits(Op.getOperand(0), InputDemandedBits,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    
    // If the input sign bit is known zero, convert this into a zero extension.
    if (KnownZero.intersects(InSignBit))
      return TLO.CombineTo(Op, 
                           TLO.DAG.getZeroExtendInReg(Op.getOperand(0), EVT));
    
    if (KnownOne.intersects(InSignBit)) {    // Input sign bit known set
      KnownOne |= NewBits;
      KnownZero &= ~NewBits;
    } else {                       // Input sign bit unknown
      KnownZero &= ~NewBits;
      KnownOne &= ~NewBits;
    }
    break;
  }
  case ISD::ZERO_EXTEND: {
    unsigned OperandBitWidth = Op.getOperand(0).getValueSizeInBits();
    APInt InMask = NewMask;
    InMask.trunc(OperandBitWidth);
    
    // If none of the top bits are demanded, convert this into an any_extend.
    APInt NewBits =
      APInt::getHighBitsSet(BitWidth, BitWidth - OperandBitWidth) & NewMask;
    if (!NewBits.intersects(NewMask))
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ANY_EXTEND, 
                                               Op.getValueType(), 
                                               Op.getOperand(0)));
    
    if (SimplifyDemandedBits(Op.getOperand(0), InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    KnownZero.zext(BitWidth);
    KnownOne.zext(BitWidth);
    KnownZero |= NewBits;
    break;
  }
  case ISD::SIGN_EXTEND: {
    MVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getSizeInBits();
    APInt InMask    = APInt::getLowBitsSet(BitWidth, InBits);
    APInt InSignBit = APInt::getBitsSet(BitWidth, InBits - 1, InBits);
    APInt NewBits   = ~InMask & NewMask;
    
    // If none of the top bits are demanded, convert this into an any_extend.
    if (NewBits == 0)
      return TLO.CombineTo(Op,TLO.DAG.getNode(ISD::ANY_EXTEND,Op.getValueType(),
                                           Op.getOperand(0)));
    
    // Since some of the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    APInt InDemandedBits = InMask & NewMask;
    InDemandedBits |= InSignBit;
    InDemandedBits.trunc(InBits);
    
    if (SimplifyDemandedBits(Op.getOperand(0), InDemandedBits, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    KnownZero.zext(BitWidth);
    KnownOne.zext(BitWidth);
    
    // If the sign bit is known zero, convert this to a zero extend.
    if (KnownZero.intersects(InSignBit))
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ZERO_EXTEND, 
                                               Op.getValueType(), 
                                               Op.getOperand(0)));
    
    // If the sign bit is known one, the top bits match.
    if (KnownOne.intersects(InSignBit)) {
      KnownOne  |= NewBits;
      KnownZero &= ~NewBits;
    } else {   // Otherwise, top bits aren't known.
      KnownOne  &= ~NewBits;
      KnownZero &= ~NewBits;
    }
    break;
  }
  case ISD::ANY_EXTEND: {
    unsigned OperandBitWidth = Op.getOperand(0).getValueSizeInBits();
    APInt InMask = NewMask;
    InMask.trunc(OperandBitWidth);
    if (SimplifyDemandedBits(Op.getOperand(0), InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    KnownZero.zext(BitWidth);
    KnownOne.zext(BitWidth);
    break;
  }
  case ISD::TRUNCATE: {
    // Simplify the input, using demanded bit information, and compute the known
    // zero/one bits live out.
    APInt TruncMask = NewMask;
    TruncMask.zext(Op.getOperand(0).getValueSizeInBits());
    if (SimplifyDemandedBits(Op.getOperand(0), TruncMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    KnownZero.trunc(BitWidth);
    KnownOne.trunc(BitWidth);
    
    // If the input is only used by this truncate, see if we can shrink it based
    // on the known demanded bits.
    if (Op.getOperand(0).Val->hasOneUse()) {
      SDOperand In = Op.getOperand(0);
      unsigned InBitWidth = In.getValueSizeInBits();
      switch (In.getOpcode()) {
      default: break;
      case ISD::SRL:
        // Shrink SRL by a constant if none of the high bits shifted in are
        // demanded.
        if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(In.getOperand(1))){
          APInt HighBits = APInt::getHighBitsSet(InBitWidth,
                                                 InBitWidth - BitWidth);
          HighBits = HighBits.lshr(ShAmt->getValue());
          HighBits.trunc(BitWidth);
          
          if (ShAmt->getValue() < BitWidth && !(HighBits & NewMask)) {
            // None of the shifted in bits are needed.  Add a truncate of the
            // shift input, then shift it.
            SDOperand NewTrunc = TLO.DAG.getNode(ISD::TRUNCATE, 
                                                 Op.getValueType(), 
                                                 In.getOperand(0));
            return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SRL,Op.getValueType(),
                                                   NewTrunc, In.getOperand(1)));
          }
        }
        break;
      }
    }
    
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    break;
  }
  case ISD::AssertZext: {
    MVT VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    APInt InMask = APInt::getLowBitsSet(BitWidth,
                                        VT.getSizeInBits());
    if (SimplifyDemandedBits(Op.getOperand(0), InMask & NewMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    KnownZero |= ~InMask & NewMask;
    break;
  }
  case ISD::BIT_CONVERT:
#if 0
    // If this is an FP->Int bitcast and if the sign bit is the only thing that
    // is demanded, turn this into a FGETSIGN.
    if (NewMask == MVT::getIntegerVTSignBit(Op.getValueType()) &&
        MVT::isFloatingPoint(Op.getOperand(0).getValueType()) &&
        !MVT::isVector(Op.getOperand(0).getValueType())) {
      // Only do this xform if FGETSIGN is valid or if before legalize.
      if (!TLO.AfterLegalize ||
          isOperationLegal(ISD::FGETSIGN, Op.getValueType())) {
        // Make a FGETSIGN + SHL to move the sign bit into the appropriate
        // place.  We expect the SHL to be eliminated by other optimizations.
        SDOperand Sign = TLO.DAG.getNode(ISD::FGETSIGN, Op.getValueType(), 
                                         Op.getOperand(0));
        unsigned ShVal = Op.getValueType().getSizeInBits()-1;
        SDOperand ShAmt = TLO.DAG.getConstant(ShVal, getShiftAmountTy());
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SHL, Op.getValueType(),
                                                 Sign, ShAmt));
      }
    }
#endif
    break;
  default:
    // Just use ComputeMaskedBits to compute output bits.
    TLO.DAG.ComputeMaskedBits(Op, NewMask, KnownZero, KnownOne, Depth);
    break;
  }
  
  // If we know the value of all of the demanded bits, return this as a
  // constant.
  if ((NewMask & (KnownZero|KnownOne)) == NewMask)
    return TLO.CombineTo(Op, TLO.DAG.getConstant(KnownOne, Op.getValueType()));
  
  return false;
}

/// computeMaskedBitsForTargetNode - Determine which of the bits specified 
/// in Mask are known to be either zero or one and return them in the 
/// KnownZero/KnownOne bitsets.
void TargetLowering::computeMaskedBitsForTargetNode(const SDOperand Op, 
                                                    const APInt &Mask,
                                                    APInt &KnownZero, 
                                                    APInt &KnownOne,
                                                    const SelectionDAG &DAG,
                                                    unsigned Depth) const {
  assert((Op.getOpcode() >= ISD::BUILTIN_OP_END ||
          Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_VOID) &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");
  KnownZero = KnownOne = APInt(Mask.getBitWidth(), 0);
}

/// ComputeNumSignBitsForTargetNode - This method can be implemented by
/// targets that want to expose additional information about sign bits to the
/// DAG Combiner.
unsigned TargetLowering::ComputeNumSignBitsForTargetNode(SDOperand Op,
                                                         unsigned Depth) const {
  assert((Op.getOpcode() >= ISD::BUILTIN_OP_END ||
          Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_VOID) &&
         "Should use ComputeNumSignBits if you don't know whether Op"
         " is a target node!");
  return 1;
}


/// SimplifySetCC - Try to simplify a setcc built with the specified operands 
/// and cc. If it is unable to simplify it, return a null SDOperand.
SDOperand
TargetLowering::SimplifySetCC(MVT VT, SDOperand N0, SDOperand N1,
                              ISD::CondCode Cond, bool foldBooleans,
                              DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  // These setcc operations always fold.
  switch (Cond) {
  default: break;
  case ISD::SETFALSE:
  case ISD::SETFALSE2: return DAG.getConstant(0, VT);
  case ISD::SETTRUE:
  case ISD::SETTRUE2:  return DAG.getConstant(1, VT);
  }

  if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val)) {
    const APInt &C1 = N1C->getAPIntValue();
    if (isa<ConstantSDNode>(N0.Val)) {
      return DAG.FoldSetCC(VT, N0, N1, Cond);
    } else {
      // If the LHS is '(srl (ctlz x), 5)', the RHS is 0/1, and this is an
      // equality comparison, then we're just comparing whether X itself is
      // zero.
      if (N0.getOpcode() == ISD::SRL && (C1 == 0 || C1 == 1) &&
          N0.getOperand(0).getOpcode() == ISD::CTLZ &&
          N0.getOperand(1).getOpcode() == ISD::Constant) {
        unsigned ShAmt = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
        if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
            ShAmt == Log2_32(N0.getValueType().getSizeInBits())) {
          if ((C1 == 0) == (Cond == ISD::SETEQ)) {
            // (srl (ctlz x), 5) == 0  -> X != 0
            // (srl (ctlz x), 5) != 1  -> X != 0
            Cond = ISD::SETNE;
          } else {
            // (srl (ctlz x), 5) != 0  -> X == 0
            // (srl (ctlz x), 5) == 1  -> X == 0
            Cond = ISD::SETEQ;
          }
          SDOperand Zero = DAG.getConstant(0, N0.getValueType());
          return DAG.getSetCC(VT, N0.getOperand(0).getOperand(0),
                              Zero, Cond);
        }
      }
      
      // If the LHS is a ZERO_EXTEND, perform the comparison on the input.
      if (N0.getOpcode() == ISD::ZERO_EXTEND) {
        unsigned InSize = N0.getOperand(0).getValueType().getSizeInBits();

        // If the comparison constant has bits in the upper part, the
        // zero-extended value could never match.
        if (C1.intersects(APInt::getHighBitsSet(C1.getBitWidth(),
                                                C1.getBitWidth() - InSize))) {
          switch (Cond) {
          case ISD::SETUGT:
          case ISD::SETUGE:
          case ISD::SETEQ: return DAG.getConstant(0, VT);
          case ISD::SETULT:
          case ISD::SETULE:
          case ISD::SETNE: return DAG.getConstant(1, VT);
          case ISD::SETGT:
          case ISD::SETGE:
            // True if the sign bit of C1 is set.
            return DAG.getConstant(C1.isNegative(), VT);
          case ISD::SETLT:
          case ISD::SETLE:
            // True if the sign bit of C1 isn't set.
            return DAG.getConstant(C1.isNonNegative(), VT);
          default:
            break;
          }
        }

        // Otherwise, we can perform the comparison with the low bits.
        switch (Cond) {
        case ISD::SETEQ:
        case ISD::SETNE:
        case ISD::SETUGT:
        case ISD::SETUGE:
        case ISD::SETULT:
        case ISD::SETULE:
          return DAG.getSetCC(VT, N0.getOperand(0),
                          DAG.getConstant(APInt(C1).trunc(InSize),
                                          N0.getOperand(0).getValueType()),
                          Cond);
        default:
          break;   // todo, be more careful with signed comparisons
        }
      } else if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
                 (Cond == ISD::SETEQ || Cond == ISD::SETNE)) {
        MVT ExtSrcTy = cast<VTSDNode>(N0.getOperand(1))->getVT();
        unsigned ExtSrcTyBits = ExtSrcTy.getSizeInBits();
        MVT ExtDstTy = N0.getValueType();
        unsigned ExtDstTyBits = ExtDstTy.getSizeInBits();

        // If the extended part has any inconsistent bits, it cannot ever
        // compare equal.  In other words, they have to be all ones or all
        // zeros.
        APInt ExtBits =
          APInt::getHighBitsSet(ExtDstTyBits, ExtDstTyBits - ExtSrcTyBits);
        if ((C1 & ExtBits) != 0 && (C1 & ExtBits) != ExtBits)
          return DAG.getConstant(Cond == ISD::SETNE, VT);
        
        SDOperand ZextOp;
        MVT Op0Ty = N0.getOperand(0).getValueType();
        if (Op0Ty == ExtSrcTy) {
          ZextOp = N0.getOperand(0);
        } else {
          APInt Imm = APInt::getLowBitsSet(ExtDstTyBits, ExtSrcTyBits);
          ZextOp = DAG.getNode(ISD::AND, Op0Ty, N0.getOperand(0),
                               DAG.getConstant(Imm, Op0Ty));
        }
        if (!DCI.isCalledByLegalizer())
          DCI.AddToWorklist(ZextOp.Val);
        // Otherwise, make this a use of a zext.
        return DAG.getSetCC(VT, ZextOp, 
                            DAG.getConstant(C1 & APInt::getLowBitsSet(
                                                               ExtDstTyBits,
                                                               ExtSrcTyBits), 
                                            ExtDstTy),
                            Cond);
      } else if ((N1C->isNullValue() || N1C->getAPIntValue() == 1) &&
                 (Cond == ISD::SETEQ || Cond == ISD::SETNE)) {
        
        // SETCC (SETCC), [0|1], [EQ|NE]  -> SETCC
        if (N0.getOpcode() == ISD::SETCC) {
          bool TrueWhenTrue = (Cond == ISD::SETEQ) ^ (N1C->getValue() != 1);
          if (TrueWhenTrue)
            return N0;
          
          // Invert the condition.
          ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();
          CC = ISD::getSetCCInverse(CC, 
                                   N0.getOperand(0).getValueType().isInteger());
          return DAG.getSetCC(VT, N0.getOperand(0), N0.getOperand(1), CC);
        }
        
        if ((N0.getOpcode() == ISD::XOR ||
             (N0.getOpcode() == ISD::AND && 
              N0.getOperand(0).getOpcode() == ISD::XOR &&
              N0.getOperand(1) == N0.getOperand(0).getOperand(1))) &&
            isa<ConstantSDNode>(N0.getOperand(1)) &&
            cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue() == 1) {
          // If this is (X^1) == 0/1, swap the RHS and eliminate the xor.  We
          // can only do this if the top bits are known zero.
          unsigned BitWidth = N0.getValueSizeInBits();
          if (DAG.MaskedValueIsZero(N0,
                                    APInt::getHighBitsSet(BitWidth,
                                                          BitWidth-1))) {
            // Okay, get the un-inverted input value.
            SDOperand Val;
            if (N0.getOpcode() == ISD::XOR)
              Val = N0.getOperand(0);
            else {
              assert(N0.getOpcode() == ISD::AND && 
                     N0.getOperand(0).getOpcode() == ISD::XOR);
              // ((X^1)&1)^1 -> X & 1
              Val = DAG.getNode(ISD::AND, N0.getValueType(),
                                N0.getOperand(0).getOperand(0),
                                N0.getOperand(1));
            }
            return DAG.getSetCC(VT, Val, N1,
                                Cond == ISD::SETEQ ? ISD::SETNE : ISD::SETEQ);
          }
        }
      }
      
      APInt MinVal, MaxVal;
      unsigned OperandBitSize = N1C->getValueType(0).getSizeInBits();
      if (ISD::isSignedIntSetCC(Cond)) {
        MinVal = APInt::getSignedMinValue(OperandBitSize);
        MaxVal = APInt::getSignedMaxValue(OperandBitSize);
      } else {
        MinVal = APInt::getMinValue(OperandBitSize);
        MaxVal = APInt::getMaxValue(OperandBitSize);
      }

      // Canonicalize GE/LE comparisons to use GT/LT comparisons.
      if (Cond == ISD::SETGE || Cond == ISD::SETUGE) {
        if (C1 == MinVal) return DAG.getConstant(1, VT);   // X >= MIN --> true
        // X >= C0 --> X > (C0-1)
        return DAG.getSetCC(VT, N0, DAG.getConstant(C1-1, N1.getValueType()),
                        (Cond == ISD::SETGE) ? ISD::SETGT : ISD::SETUGT);
      }

      if (Cond == ISD::SETLE || Cond == ISD::SETULE) {
        if (C1 == MaxVal) return DAG.getConstant(1, VT);   // X <= MAX --> true
        // X <= C0 --> X < (C0+1)
        return DAG.getSetCC(VT, N0, DAG.getConstant(C1+1, N1.getValueType()),
                        (Cond == ISD::SETLE) ? ISD::SETLT : ISD::SETULT);
      }

      if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MinVal)
        return DAG.getConstant(0, VT);      // X < MIN --> false
      if ((Cond == ISD::SETGE || Cond == ISD::SETUGE) && C1 == MinVal)
        return DAG.getConstant(1, VT);      // X >= MIN --> true
      if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MaxVal)
        return DAG.getConstant(0, VT);      // X > MAX --> false
      if ((Cond == ISD::SETLE || Cond == ISD::SETULE) && C1 == MaxVal)
        return DAG.getConstant(1, VT);      // X <= MAX --> true

      // Canonicalize setgt X, Min --> setne X, Min
      if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MinVal)
        return DAG.getSetCC(VT, N0, N1, ISD::SETNE);
      // Canonicalize setlt X, Max --> setne X, Max
      if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MaxVal)
        return DAG.getSetCC(VT, N0, N1, ISD::SETNE);

      // If we have setult X, 1, turn it into seteq X, 0
      if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MinVal+1)
        return DAG.getSetCC(VT, N0, DAG.getConstant(MinVal, N0.getValueType()),
                        ISD::SETEQ);
      // If we have setugt X, Max-1, turn it into seteq X, Max
      else if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MaxVal-1)
        return DAG.getSetCC(VT, N0, DAG.getConstant(MaxVal, N0.getValueType()),
                        ISD::SETEQ);

      // If we have "setcc X, C0", check to see if we can shrink the immediate
      // by changing cc.

      // SETUGT X, SINTMAX  -> SETLT X, 0
      if (Cond == ISD::SETUGT && OperandBitSize != 1 &&
          C1 == (~0ULL >> (65-OperandBitSize)))
        return DAG.getSetCC(VT, N0, DAG.getConstant(0, N1.getValueType()),
                            ISD::SETLT);

      // FIXME: Implement the rest of these.

      // Fold bit comparisons when we can.
      if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
          VT == N0.getValueType() && N0.getOpcode() == ISD::AND)
        if (ConstantSDNode *AndRHS =
                    dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
          if (Cond == ISD::SETNE && C1 == 0) {// (X & 8) != 0  -->  (X & 8) >> 3
            // Perform the xform if the AND RHS is a single bit.
            if (isPowerOf2_64(AndRHS->getValue())) {
              return DAG.getNode(ISD::SRL, VT, N0,
                             DAG.getConstant(Log2_64(AndRHS->getValue()),
                                             getShiftAmountTy()));
            }
          } else if (Cond == ISD::SETEQ && C1 == AndRHS->getValue()) {
            // (X & 8) == 8  -->  (X & 8) >> 3
            // Perform the xform if C1 is a single bit.
            if (C1.isPowerOf2()) {
              return DAG.getNode(ISD::SRL, VT, N0,
                          DAG.getConstant(C1.logBase2(), getShiftAmountTy()));
            }
          }
        }
    }
  } else if (isa<ConstantSDNode>(N0.Val)) {
      // Ensure that the constant occurs on the RHS.
    return DAG.getSetCC(VT, N1, N0, ISD::getSetCCSwappedOperands(Cond));
  }

  if (isa<ConstantFPSDNode>(N0.Val)) {
    // Constant fold or commute setcc.
    SDOperand O = DAG.FoldSetCC(VT, N0, N1, Cond);    
    if (O.Val) return O;
  } else if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N1.Val)) {
    // If the RHS of an FP comparison is a constant, simplify it away in
    // some cases.
    if (CFP->getValueAPF().isNaN()) {
      // If an operand is known to be a nan, we can fold it.
      switch (ISD::getUnorderedFlavor(Cond)) {
      default: assert(0 && "Unknown flavor!");
      case 0:  // Known false.
        return DAG.getConstant(0, VT);
      case 1:  // Known true.
        return DAG.getConstant(1, VT);
      case 2:  // Undefined.
        return DAG.getNode(ISD::UNDEF, VT);
      }
    }
    
    // Otherwise, we know the RHS is not a NaN.  Simplify the node to drop the
    // constant if knowing that the operand is non-nan is enough.  We prefer to
    // have SETO(x,x) instead of SETO(x, 0.0) because this avoids having to
    // materialize 0.0.
    if (Cond == ISD::SETO || Cond == ISD::SETUO)
      return DAG.getSetCC(VT, N0, N0, Cond);
  }

  if (N0 == N1) {
    // We can always fold X == X for integer setcc's.
    if (N0.getValueType().isInteger())
      return DAG.getConstant(ISD::isTrueWhenEqual(Cond), VT);
    unsigned UOF = ISD::getUnorderedFlavor(Cond);
    if (UOF == 2)   // FP operators that are undefined on NaNs.
      return DAG.getConstant(ISD::isTrueWhenEqual(Cond), VT);
    if (UOF == unsigned(ISD::isTrueWhenEqual(Cond)))
      return DAG.getConstant(UOF, VT);
    // Otherwise, we can't fold it.  However, we can simplify it to SETUO/SETO
    // if it is not already.
    ISD::CondCode NewCond = UOF == 0 ? ISD::SETO : ISD::SETUO;
    if (NewCond != Cond)
      return DAG.getSetCC(VT, N0, N1, NewCond);
  }

  if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
      N0.getValueType().isInteger()) {
    if (N0.getOpcode() == ISD::ADD || N0.getOpcode() == ISD::SUB ||
        N0.getOpcode() == ISD::XOR) {
      // Simplify (X+Y) == (X+Z) -->  Y == Z
      if (N0.getOpcode() == N1.getOpcode()) {
        if (N0.getOperand(0) == N1.getOperand(0))
          return DAG.getSetCC(VT, N0.getOperand(1), N1.getOperand(1), Cond);
        if (N0.getOperand(1) == N1.getOperand(1))
          return DAG.getSetCC(VT, N0.getOperand(0), N1.getOperand(0), Cond);
        if (DAG.isCommutativeBinOp(N0.getOpcode())) {
          // If X op Y == Y op X, try other combinations.
          if (N0.getOperand(0) == N1.getOperand(1))
            return DAG.getSetCC(VT, N0.getOperand(1), N1.getOperand(0), Cond);
          if (N0.getOperand(1) == N1.getOperand(0))
            return DAG.getSetCC(VT, N0.getOperand(0), N1.getOperand(1), Cond);
        }
      }
      
      if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(N1)) {
        if (ConstantSDNode *LHSR = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
          // Turn (X+C1) == C2 --> X == C2-C1
          if (N0.getOpcode() == ISD::ADD && N0.Val->hasOneUse()) {
            return DAG.getSetCC(VT, N0.getOperand(0),
                              DAG.getConstant(RHSC->getValue()-LHSR->getValue(),
                                N0.getValueType()), Cond);
          }
          
          // Turn (X^C1) == C2 into X == C1^C2 iff X&~C1 = 0.
          if (N0.getOpcode() == ISD::XOR)
            // If we know that all of the inverted bits are zero, don't bother
            // performing the inversion.
            if (DAG.MaskedValueIsZero(N0.getOperand(0), ~LHSR->getAPIntValue()))
              return
                DAG.getSetCC(VT, N0.getOperand(0),
                             DAG.getConstant(LHSR->getAPIntValue() ^
                                               RHSC->getAPIntValue(),
                                             N0.getValueType()),
                             Cond);
        }
        
        // Turn (C1-X) == C2 --> X == C1-C2
        if (ConstantSDNode *SUBC = dyn_cast<ConstantSDNode>(N0.getOperand(0))) {
          if (N0.getOpcode() == ISD::SUB && N0.Val->hasOneUse()) {
            return
              DAG.getSetCC(VT, N0.getOperand(1),
                           DAG.getConstant(SUBC->getAPIntValue() -
                                             RHSC->getAPIntValue(),
                                           N0.getValueType()),
                           Cond);
          }
        }          
      }

      // Simplify (X+Z) == X -->  Z == 0
      if (N0.getOperand(0) == N1)
        return DAG.getSetCC(VT, N0.getOperand(1),
                        DAG.getConstant(0, N0.getValueType()), Cond);
      if (N0.getOperand(1) == N1) {
        if (DAG.isCommutativeBinOp(N0.getOpcode()))
          return DAG.getSetCC(VT, N0.getOperand(0),
                          DAG.getConstant(0, N0.getValueType()), Cond);
        else if (N0.Val->hasOneUse()) {
          assert(N0.getOpcode() == ISD::SUB && "Unexpected operation!");
          // (Z-X) == X  --> Z == X<<1
          SDOperand SH = DAG.getNode(ISD::SHL, N1.getValueType(),
                                     N1, 
                                     DAG.getConstant(1, getShiftAmountTy()));
          if (!DCI.isCalledByLegalizer())
            DCI.AddToWorklist(SH.Val);
          return DAG.getSetCC(VT, N0.getOperand(0), SH, Cond);
        }
      }
    }

    if (N1.getOpcode() == ISD::ADD || N1.getOpcode() == ISD::SUB ||
        N1.getOpcode() == ISD::XOR) {
      // Simplify  X == (X+Z) -->  Z == 0
      if (N1.getOperand(0) == N0) {
        return DAG.getSetCC(VT, N1.getOperand(1),
                        DAG.getConstant(0, N1.getValueType()), Cond);
      } else if (N1.getOperand(1) == N0) {
        if (DAG.isCommutativeBinOp(N1.getOpcode())) {
          return DAG.getSetCC(VT, N1.getOperand(0),
                          DAG.getConstant(0, N1.getValueType()), Cond);
        } else if (N1.Val->hasOneUse()) {
          assert(N1.getOpcode() == ISD::SUB && "Unexpected operation!");
          // X == (Z-X)  --> X<<1 == Z
          SDOperand SH = DAG.getNode(ISD::SHL, N1.getValueType(), N0, 
                                     DAG.getConstant(1, getShiftAmountTy()));
          if (!DCI.isCalledByLegalizer())
            DCI.AddToWorklist(SH.Val);
          return DAG.getSetCC(VT, SH, N1.getOperand(0), Cond);
        }
      }
    }
  }

  // Fold away ALL boolean setcc's.
  SDOperand Temp;
  if (N0.getValueType() == MVT::i1 && foldBooleans) {
    switch (Cond) {
    default: assert(0 && "Unknown integer setcc!");
    case ISD::SETEQ:  // X == Y  -> (X^Y)^1
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N0, N1);
      N0 = DAG.getNode(ISD::XOR, MVT::i1, Temp, DAG.getConstant(1, MVT::i1));
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.Val);
      break;
    case ISD::SETNE:  // X != Y   -->  (X^Y)
      N0 = DAG.getNode(ISD::XOR, MVT::i1, N0, N1);
      break;
    case ISD::SETGT:  // X >s Y   -->  X == 0 & Y == 1  -->  X^1 & Y
    case ISD::SETULT: // X <u Y   -->  X == 0 & Y == 1  -->  X^1 & Y
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N0, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::AND, MVT::i1, N1, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.Val);
      break;
    case ISD::SETLT:  // X <s Y   --> X == 1 & Y == 0  -->  Y^1 & X
    case ISD::SETUGT: // X >u Y   --> X == 1 & Y == 0  -->  Y^1 & X
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N1, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::AND, MVT::i1, N0, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.Val);
      break;
    case ISD::SETULE: // X <=u Y  --> X == 0 | Y == 1  -->  X^1 | Y
    case ISD::SETGE:  // X >=s Y  --> X == 0 | Y == 1  -->  X^1 | Y
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N0, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::OR, MVT::i1, N1, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.Val);
      break;
    case ISD::SETUGE: // X >=u Y  --> X == 1 | Y == 0  -->  Y^1 | X
    case ISD::SETLE:  // X <=s Y  --> X == 1 | Y == 0  -->  Y^1 | X
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N1, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::OR, MVT::i1, N0, Temp);
      break;
    }
    if (VT != MVT::i1) {
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(N0.Val);
      // FIXME: If running after legalize, we probably can't do this.
      N0 = DAG.getNode(ISD::ZERO_EXTEND, VT, N0);
    }
    return N0;
  }

  // Could not fold it.
  return SDOperand();
}

/// isGAPlusOffset - Returns true (and the GlobalValue and the offset) if the
/// node is a GlobalAddress + offset.
bool TargetLowering::isGAPlusOffset(SDNode *N, GlobalValue* &GA,
                                    int64_t &Offset) const {
  if (isa<GlobalAddressSDNode>(N)) {
    GlobalAddressSDNode *GASD = cast<GlobalAddressSDNode>(N);
    GA = GASD->getGlobal();
    Offset += GASD->getOffset();
    return true;
  }

  if (N->getOpcode() == ISD::ADD) {
    SDOperand N1 = N->getOperand(0);
    SDOperand N2 = N->getOperand(1);
    if (isGAPlusOffset(N1.Val, GA, Offset)) {
      ConstantSDNode *V = dyn_cast<ConstantSDNode>(N2);
      if (V) {
        Offset += V->getSignExtended();
        return true;
      }
    } else if (isGAPlusOffset(N2.Val, GA, Offset)) {
      ConstantSDNode *V = dyn_cast<ConstantSDNode>(N1);
      if (V) {
        Offset += V->getSignExtended();
        return true;
      }
    }
  }
  return false;
}


/// isConsecutiveLoad - Return true if LD (which must be a LoadSDNode) is
/// loading 'Bytes' bytes from a location that is 'Dist' units away from the
/// location that the 'Base' load is loading from.
bool TargetLowering::isConsecutiveLoad(SDNode *LD, SDNode *Base,
                                       unsigned Bytes, int Dist,
                                       const MachineFrameInfo *MFI) const {
  if (LD->getOperand(0).Val != Base->getOperand(0).Val)
    return false;
  MVT VT = LD->getValueType(0);
  if (VT.getSizeInBits() / 8 != Bytes)
    return false;

  SDOperand Loc = LD->getOperand(1);
  SDOperand BaseLoc = Base->getOperand(1);
  if (Loc.getOpcode() == ISD::FrameIndex) {
    if (BaseLoc.getOpcode() != ISD::FrameIndex)
      return false;
    int FI  = cast<FrameIndexSDNode>(Loc)->getIndex();
    int BFI = cast<FrameIndexSDNode>(BaseLoc)->getIndex();
    int FS  = MFI->getObjectSize(FI);
    int BFS = MFI->getObjectSize(BFI);
    if (FS != BFS || FS != (int)Bytes) return false;
    return MFI->getObjectOffset(FI) == (MFI->getObjectOffset(BFI) + Dist*Bytes);
  }

  GlobalValue *GV1 = NULL;
  GlobalValue *GV2 = NULL;
  int64_t Offset1 = 0;
  int64_t Offset2 = 0;
  bool isGA1 = isGAPlusOffset(Loc.Val, GV1, Offset1);
  bool isGA2 = isGAPlusOffset(BaseLoc.Val, GV2, Offset2);
  if (isGA1 && isGA2 && GV1 == GV2)
    return Offset1 == (Offset2 + Dist*Bytes);
  return false;
}


SDOperand TargetLowering::
PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const {
  // Default implementation: no optimization.
  return SDOperand();
}

//===----------------------------------------------------------------------===//
//  Inline Assembler Implementation Methods
//===----------------------------------------------------------------------===//


TargetLowering::ConstraintType
TargetLowering::getConstraintType(const std::string &Constraint) const {
  // FIXME: lots more standard ones to handle.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;
    case 'r': return C_RegisterClass;
    case 'm':    // memory
    case 'o':    // offsetable
    case 'V':    // not offsetable
      return C_Memory;
    case 'i':    // Simple Integer or Relocatable Constant
    case 'n':    // Simple Integer
    case 's':    // Relocatable Constant
    case 'X':    // Allow ANY value.
    case 'I':    // Target registers.
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
      return C_Other;
    }
  }
  
  if (Constraint.size() > 1 && Constraint[0] == '{' && 
      Constraint[Constraint.size()-1] == '}')
    return C_Register;
  return C_Unknown;
}

/// LowerXConstraint - try to replace an X constraint, which matches anything,
/// with another that has more specific requirements based on the type of the
/// corresponding operand.
const char *TargetLowering::LowerXConstraint(MVT ConstraintVT) const{
  if (ConstraintVT.isInteger())
    return "r";
  if (ConstraintVT.isFloatingPoint())
    return "f";      // works for many targets
  return 0;
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void TargetLowering::LowerAsmOperandForConstraint(SDOperand Op,
                                                  char ConstraintLetter,
                                                  std::vector<SDOperand> &Ops,
                                                  SelectionDAG &DAG) const {
  switch (ConstraintLetter) {
  default: break;
  case 'X':     // Allows any operand; labels (basic block) use this.
    if (Op.getOpcode() == ISD::BasicBlock) {
      Ops.push_back(Op);
      return;
    }
    // fall through
  case 'i':    // Simple Integer or Relocatable Constant
  case 'n':    // Simple Integer
  case 's': {  // Relocatable Constant
    // These operands are interested in values of the form (GV+C), where C may
    // be folded in as an offset of GV, or it may be explicitly added.  Also, it
    // is possible and fine if either GV or C are missing.
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
    GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Op);
    
    // If we have "(add GV, C)", pull out GV/C
    if (Op.getOpcode() == ISD::ADD) {
      C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
      GA = dyn_cast<GlobalAddressSDNode>(Op.getOperand(0));
      if (C == 0 || GA == 0) {
        C = dyn_cast<ConstantSDNode>(Op.getOperand(0));
        GA = dyn_cast<GlobalAddressSDNode>(Op.getOperand(1));
      }
      if (C == 0 || GA == 0)
        C = 0, GA = 0;
    }
    
    // If we find a valid operand, map to the TargetXXX version so that the
    // value itself doesn't get selected.
    if (GA) {   // Either &GV   or   &GV+C
      if (ConstraintLetter != 'n') {
        int64_t Offs = GA->getOffset();
        if (C) Offs += C->getValue();
        Ops.push_back(DAG.getTargetGlobalAddress(GA->getGlobal(),
                                                 Op.getValueType(), Offs));
        return;
      }
    }
    if (C) {   // just C, no GV.
      // Simple constants are not allowed for 's'.
      if (ConstraintLetter != 's') {
        Ops.push_back(DAG.getTargetConstant(C->getValue(), Op.getValueType()));
        return;
      }
    }
    break;
  }
  }
}

std::vector<unsigned> TargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT VT) const {
  return std::vector<unsigned>();
}


std::pair<unsigned, const TargetRegisterClass*> TargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint,
                             MVT VT) const {
  if (Constraint[0] != '{')
    return std::pair<unsigned, const TargetRegisterClass*>(0, 0);
  assert(*(Constraint.end()-1) == '}' && "Not a brace enclosed constraint?");

  // Remove the braces from around the name.
  std::string RegName(Constraint.begin()+1, Constraint.end()-1);

  // Figure out which register class contains this reg.
  const TargetRegisterInfo *RI = TM.getRegisterInfo();
  for (TargetRegisterInfo::regclass_iterator RCI = RI->regclass_begin(),
       E = RI->regclass_end(); RCI != E; ++RCI) {
    const TargetRegisterClass *RC = *RCI;
    
    // If none of the the value types for this register class are valid, we 
    // can't use it.  For example, 64-bit reg classes on 32-bit targets.
    bool isLegal = false;
    for (TargetRegisterClass::vt_iterator I = RC->vt_begin(), E = RC->vt_end();
         I != E; ++I) {
      if (isTypeLegal(*I)) {
        isLegal = true;
        break;
      }
    }
    
    if (!isLegal) continue;
    
    for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end(); 
         I != E; ++I) {
      if (StringsEqualNoCase(RegName, RI->get(*I).AsmName))
        return std::make_pair(*I, RC);
    }
  }
  
  return std::pair<unsigned, const TargetRegisterClass*>(0, 0);
}

//===----------------------------------------------------------------------===//
// Constraint Selection.

/// getConstraintGenerality - Return an integer indicating how general CT
/// is.
static unsigned getConstraintGenerality(TargetLowering::ConstraintType CT) {
  switch (CT) {
  default: assert(0 && "Unknown constraint type!");
  case TargetLowering::C_Other:
  case TargetLowering::C_Unknown:
    return 0;
  case TargetLowering::C_Register:
    return 1;
  case TargetLowering::C_RegisterClass:
    return 2;
  case TargetLowering::C_Memory:
    return 3;
  }
}

/// ChooseConstraint - If there are multiple different constraints that we
/// could pick for this operand (e.g. "imr") try to pick the 'best' one.
/// This is somewhat tricky: constraints fall into four classes:
///    Other         -> immediates and magic values
///    Register      -> one specific register
///    RegisterClass -> a group of regs
///    Memory        -> memory
/// Ideally, we would pick the most specific constraint possible: if we have
/// something that fits into a register, we would pick it.  The problem here
/// is that if we have something that could either be in a register or in
/// memory that use of the register could cause selection of *other*
/// operands to fail: they might only succeed if we pick memory.  Because of
/// this the heuristic we use is:
///
///  1) If there is an 'other' constraint, and if the operand is valid for
///     that constraint, use it.  This makes us take advantage of 'i'
///     constraints when available.
///  2) Otherwise, pick the most general constraint present.  This prefers
///     'm' over 'r', for example.
///
static void ChooseConstraint(TargetLowering::AsmOperandInfo &OpInfo,
                             const TargetLowering &TLI,
                             SDOperand Op, SelectionDAG *DAG) {
  assert(OpInfo.Codes.size() > 1 && "Doesn't have multiple constraint options");
  unsigned BestIdx = 0;
  TargetLowering::ConstraintType BestType = TargetLowering::C_Unknown;
  int BestGenerality = -1;
  
  // Loop over the options, keeping track of the most general one.
  for (unsigned i = 0, e = OpInfo.Codes.size(); i != e; ++i) {
    TargetLowering::ConstraintType CType =
      TLI.getConstraintType(OpInfo.Codes[i]);
    
    // If this is an 'other' constraint, see if the operand is valid for it.
    // For example, on X86 we might have an 'rI' constraint.  If the operand
    // is an integer in the range [0..31] we want to use I (saving a load
    // of a register), otherwise we must use 'r'.
    if (CType == TargetLowering::C_Other && Op.Val) {
      assert(OpInfo.Codes[i].size() == 1 &&
             "Unhandled multi-letter 'other' constraint");
      std::vector<SDOperand> ResultOps;
      TLI.LowerAsmOperandForConstraint(Op, OpInfo.Codes[i][0],
                                       ResultOps, *DAG);
      if (!ResultOps.empty()) {
        BestType = CType;
        BestIdx = i;
        break;
      }
    }
    
    // This constraint letter is more general than the previous one, use it.
    int Generality = getConstraintGenerality(CType);
    if (Generality > BestGenerality) {
      BestType = CType;
      BestIdx = i;
      BestGenerality = Generality;
    }
  }
  
  OpInfo.ConstraintCode = OpInfo.Codes[BestIdx];
  OpInfo.ConstraintType = BestType;
}

/// ComputeConstraintToUse - Determines the constraint code and constraint
/// type to use for the specific AsmOperandInfo, setting
/// OpInfo.ConstraintCode and OpInfo.ConstraintType.
void TargetLowering::ComputeConstraintToUse(AsmOperandInfo &OpInfo,
                                            SDOperand Op, 
                                            SelectionDAG *DAG) const {
  assert(!OpInfo.Codes.empty() && "Must have at least one constraint");
  
  // Single-letter constraints ('r') are very common.
  if (OpInfo.Codes.size() == 1) {
    OpInfo.ConstraintCode = OpInfo.Codes[0];
    OpInfo.ConstraintType = getConstraintType(OpInfo.ConstraintCode);
  } else {
    ChooseConstraint(OpInfo, *this, Op, DAG);
  }
  
  // 'X' matches anything.
  if (OpInfo.ConstraintCode == "X" && OpInfo.CallOperandVal) {
    // Labels and constants are handled elsewhere ('X' is the only thing
    // that matches labels).
    if (isa<BasicBlock>(OpInfo.CallOperandVal) ||
        isa<ConstantInt>(OpInfo.CallOperandVal))
      return;
    
    // Otherwise, try to resolve it to something we know about by looking at
    // the actual operand type.
    if (const char *Repl = LowerXConstraint(OpInfo.ConstraintVT)) {
      OpInfo.ConstraintCode = Repl;
      OpInfo.ConstraintType = getConstraintType(OpInfo.ConstraintCode);
    }
  }
}

//===----------------------------------------------------------------------===//
//  Loop Strength Reduction hooks
//===----------------------------------------------------------------------===//

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
bool TargetLowering::isLegalAddressingMode(const AddrMode &AM, 
                                           const Type *Ty) const {
  // The default implementation of this implements a conservative RISCy, r+r and
  // r+i addr mode.

  // Allows a sign-extended 16-bit immediate field.
  if (AM.BaseOffs <= -(1LL << 16) || AM.BaseOffs >= (1LL << 16)-1)
    return false;
  
  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;
  
  // Only support r+r, 
  switch (AM.Scale) {
  case 0:  // "r+i" or just "i", depending on HasBaseReg.
    break;
  case 1:
    if (AM.HasBaseReg && AM.BaseOffs)  // "r+r+i" is not allowed.
      return false;
    // Otherwise we have r+r or r+i.
    break;
  case 2:
    if (AM.HasBaseReg || AM.BaseOffs)  // 2*r+r  or  2*r+i is not allowed.
      return false;
    // Allow 2*r as r+r.
    break;
  }
  
  return true;
}

// Magic for divide replacement

struct ms {
  int64_t m;  // magic number
  int64_t s;  // shift amount
};

struct mu {
  uint64_t m; // magic number
  int64_t a;  // add indicator
  int64_t s;  // shift amount
};

/// magic - calculate the magic numbers required to codegen an integer sdiv as
/// a sequence of multiply and shifts.  Requires that the divisor not be 0, 1,
/// or -1.
static ms magic32(int32_t d) {
  int32_t p;
  uint32_t ad, anc, delta, q1, r1, q2, r2, t;
  const uint32_t two31 = 0x80000000U;
  struct ms mag;
  
  ad = abs(d);
  t = two31 + ((uint32_t)d >> 31);
  anc = t - 1 - t%ad;   // absolute value of nc
  p = 31;               // initialize p
  q1 = two31/anc;       // initialize q1 = 2p/abs(nc)
  r1 = two31 - q1*anc;  // initialize r1 = rem(2p,abs(nc))
  q2 = two31/ad;        // initialize q2 = 2p/abs(d)
  r2 = two31 - q2*ad;   // initialize r2 = rem(2p,abs(d))
  do {
    p = p + 1;
    q1 = 2*q1;        // update q1 = 2p/abs(nc)
    r1 = 2*r1;        // update r1 = rem(2p/abs(nc))
    if (r1 >= anc) {  // must be unsigned comparison
      q1 = q1 + 1;
      r1 = r1 - anc;
    }
    q2 = 2*q2;        // update q2 = 2p/abs(d)
    r2 = 2*r2;        // update r2 = rem(2p/abs(d))
    if (r2 >= ad) {   // must be unsigned comparison
      q2 = q2 + 1;
      r2 = r2 - ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));
  
  mag.m = (int32_t)(q2 + 1); // make sure to sign extend
  if (d < 0) mag.m = -mag.m; // resulting magic number
  mag.s = p - 32;            // resulting shift
  return mag;
}

/// magicu - calculate the magic numbers required to codegen an integer udiv as
/// a sequence of multiply, add and shifts.  Requires that the divisor not be 0.
static mu magicu32(uint32_t d) {
  int32_t p;
  uint32_t nc, delta, q1, r1, q2, r2;
  struct mu magu;
  magu.a = 0;               // initialize "add" indicator
  nc = - 1 - (-d)%d;
  p = 31;                   // initialize p
  q1 = 0x80000000/nc;       // initialize q1 = 2p/nc
  r1 = 0x80000000 - q1*nc;  // initialize r1 = rem(2p,nc)
  q2 = 0x7FFFFFFF/d;        // initialize q2 = (2p-1)/d
  r2 = 0x7FFFFFFF - q2*d;   // initialize r2 = rem((2p-1),d)
  do {
    p = p + 1;
    if (r1 >= nc - r1 ) {
      q1 = 2*q1 + 1;  // update q1
      r1 = 2*r1 - nc; // update r1
    }
    else {
      q1 = 2*q1; // update q1
      r1 = 2*r1; // update r1
    }
    if (r2 + 1 >= d - r2) {
      if (q2 >= 0x7FFFFFFF) magu.a = 1;
      q2 = 2*q2 + 1;     // update q2
      r2 = 2*r2 + 1 - d; // update r2
    }
    else {
      if (q2 >= 0x80000000) magu.a = 1;
      q2 = 2*q2;     // update q2
      r2 = 2*r2 + 1; // update r2
    }
    delta = d - 1 - r2;
  } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
  magu.m = q2 + 1; // resulting magic number
  magu.s = p - 32;  // resulting shift
  return magu;
}

/// magic - calculate the magic numbers required to codegen an integer sdiv as
/// a sequence of multiply and shifts.  Requires that the divisor not be 0, 1,
/// or -1.
static ms magic64(int64_t d) {
  int64_t p;
  uint64_t ad, anc, delta, q1, r1, q2, r2, t;
  const uint64_t two63 = 9223372036854775808ULL; // 2^63
  struct ms mag;
  
  ad = d >= 0 ? d : -d;
  t = two63 + ((uint64_t)d >> 63);
  anc = t - 1 - t%ad;   // absolute value of nc
  p = 63;               // initialize p
  q1 = two63/anc;       // initialize q1 = 2p/abs(nc)
  r1 = two63 - q1*anc;  // initialize r1 = rem(2p,abs(nc))
  q2 = two63/ad;        // initialize q2 = 2p/abs(d)
  r2 = two63 - q2*ad;   // initialize r2 = rem(2p,abs(d))
  do {
    p = p + 1;
    q1 = 2*q1;        // update q1 = 2p/abs(nc)
    r1 = 2*r1;        // update r1 = rem(2p/abs(nc))
    if (r1 >= anc) {  // must be unsigned comparison
      q1 = q1 + 1;
      r1 = r1 - anc;
    }
    q2 = 2*q2;        // update q2 = 2p/abs(d)
    r2 = 2*r2;        // update r2 = rem(2p/abs(d))
    if (r2 >= ad) {   // must be unsigned comparison
      q2 = q2 + 1;
      r2 = r2 - ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));
  
  mag.m = q2 + 1;
  if (d < 0) mag.m = -mag.m; // resulting magic number
  mag.s = p - 64;            // resulting shift
  return mag;
}

/// magicu - calculate the magic numbers required to codegen an integer udiv as
/// a sequence of multiply, add and shifts.  Requires that the divisor not be 0.
static mu magicu64(uint64_t d)
{
  int64_t p;
  uint64_t nc, delta, q1, r1, q2, r2;
  struct mu magu;
  magu.a = 0;               // initialize "add" indicator
  nc = - 1 - (-d)%d;
  p = 63;                   // initialize p
  q1 = 0x8000000000000000ull/nc;       // initialize q1 = 2p/nc
  r1 = 0x8000000000000000ull - q1*nc;  // initialize r1 = rem(2p,nc)
  q2 = 0x7FFFFFFFFFFFFFFFull/d;        // initialize q2 = (2p-1)/d
  r2 = 0x7FFFFFFFFFFFFFFFull - q2*d;   // initialize r2 = rem((2p-1),d)
  do {
    p = p + 1;
    if (r1 >= nc - r1 ) {
      q1 = 2*q1 + 1;  // update q1
      r1 = 2*r1 - nc; // update r1
    }
    else {
      q1 = 2*q1; // update q1
      r1 = 2*r1; // update r1
    }
    if (r2 + 1 >= d - r2) {
      if (q2 >= 0x7FFFFFFFFFFFFFFFull) magu.a = 1;
      q2 = 2*q2 + 1;     // update q2
      r2 = 2*r2 + 1 - d; // update r2
    }
    else {
      if (q2 >= 0x8000000000000000ull) magu.a = 1;
      q2 = 2*q2;     // update q2
      r2 = 2*r2 + 1; // update r2
    }
    delta = d - 1 - r2;
  } while (p < 128 && (q1 < delta || (q1 == delta && r1 == 0)));
  magu.m = q2 + 1; // resulting magic number
  magu.s = p - 64;  // resulting shift
  return magu;
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand TargetLowering::BuildSDIV(SDNode *N, SelectionDAG &DAG, 
                                    std::vector<SDNode*>* Created) const {
  MVT VT = N->getValueType(0);
  
  // Check to see if we can do this.
  if (!isTypeLegal(VT) || (VT != MVT::i32 && VT != MVT::i64))
    return SDOperand();       // BuildSDIV only operates on i32 or i64
  
  int64_t d = cast<ConstantSDNode>(N->getOperand(1))->getSignExtended();
  ms magics = (VT == MVT::i32) ? magic32(d) : magic64(d);
  
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q;
  if (isOperationLegal(ISD::MULHS, VT))
    Q = DAG.getNode(ISD::MULHS, VT, N->getOperand(0),
                    DAG.getConstant(magics.m, VT));
  else if (isOperationLegal(ISD::SMUL_LOHI, VT))
    Q = SDOperand(DAG.getNode(ISD::SMUL_LOHI, DAG.getVTList(VT, VT),
                              N->getOperand(0),
                              DAG.getConstant(magics.m, VT)).Val, 1);
  else
    return SDOperand();       // No mulhs or equvialent
  // If d > 0 and m < 0, add the numerator
  if (d > 0 && magics.m < 0) { 
    Q = DAG.getNode(ISD::ADD, VT, Q, N->getOperand(0));
    if (Created)
      Created->push_back(Q.Val);
  }
  // If d < 0 and m > 0, subtract the numerator.
  if (d < 0 && magics.m > 0) {
    Q = DAG.getNode(ISD::SUB, VT, Q, N->getOperand(0));
    if (Created)
      Created->push_back(Q.Val);
  }
  // Shift right algebraic if shift value is nonzero
  if (magics.s > 0) {
    Q = DAG.getNode(ISD::SRA, VT, Q, 
                    DAG.getConstant(magics.s, getShiftAmountTy()));
    if (Created)
      Created->push_back(Q.Val);
  }
  // Extract the sign bit and add it to the quotient
  SDOperand T =
    DAG.getNode(ISD::SRL, VT, Q, DAG.getConstant(VT.getSizeInBits()-1,
                                                 getShiftAmountTy()));
  if (Created)
    Created->push_back(T.Val);
  return DAG.getNode(ISD::ADD, VT, Q, T);
}

/// BuildUDIVSequence - Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand TargetLowering::BuildUDIV(SDNode *N, SelectionDAG &DAG,
                                    std::vector<SDNode*>* Created) const {
  MVT VT = N->getValueType(0);
  
  // Check to see if we can do this.
  if (!isTypeLegal(VT) || (VT != MVT::i32 && VT != MVT::i64))
    return SDOperand();       // BuildUDIV only operates on i32 or i64
  
  uint64_t d = cast<ConstantSDNode>(N->getOperand(1))->getValue();
  mu magics = (VT == MVT::i32) ? magicu32(d) : magicu64(d);
  
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q;
  if (isOperationLegal(ISD::MULHU, VT))
    Q = DAG.getNode(ISD::MULHU, VT, N->getOperand(0),
                    DAG.getConstant(magics.m, VT));
  else if (isOperationLegal(ISD::UMUL_LOHI, VT))
    Q = SDOperand(DAG.getNode(ISD::UMUL_LOHI, DAG.getVTList(VT, VT),
                              N->getOperand(0),
                              DAG.getConstant(magics.m, VT)).Val, 1);
  else
    return SDOperand();       // No mulhu or equvialent
  if (Created)
    Created->push_back(Q.Val);

  if (magics.a == 0) {
    return DAG.getNode(ISD::SRL, VT, Q, 
                       DAG.getConstant(magics.s, getShiftAmountTy()));
  } else {
    SDOperand NPQ = DAG.getNode(ISD::SUB, VT, N->getOperand(0), Q);
    if (Created)
      Created->push_back(NPQ.Val);
    NPQ = DAG.getNode(ISD::SRL, VT, NPQ, 
                      DAG.getConstant(1, getShiftAmountTy()));
    if (Created)
      Created->push_back(NPQ.Val);
    NPQ = DAG.getNode(ISD::ADD, VT, NPQ, Q);
    if (Created)
      Created->push_back(NPQ.Val);
    return DAG.getNode(ISD::SRL, VT, NPQ, 
                       DAG.getConstant(magics.s-1, getShiftAmountTy()));
  }
}
