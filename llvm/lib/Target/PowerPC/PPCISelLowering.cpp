//===-- PPCISelLowering.cpp - PPC DAG Lowering Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPCISelLowering class.
//
//===----------------------------------------------------------------------===//

#include "PPCISelLowering.h"
#include "PPCTargetMachine.h"
#include "PPCPerfectShuffle.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

PPCTargetLowering::PPCTargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
    
  // Fold away setcc operations if possible.
  setSetCCIsExpensive();
  setPow2DivIsCheap();
  
  // Use _setjmp/_longjmp instead of setjmp/longjmp.
  setUseUnderscoreSetJmpLongJmp(true);
    
  // Set up the register classes.
  addRegisterClass(MVT::i32, PPC::GPRCRegisterClass);
  addRegisterClass(MVT::f32, PPC::F4RCRegisterClass);
  addRegisterClass(MVT::f64, PPC::F8RCRegisterClass);
  
  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);

  // PowerPC has no intrinsics for these particular operations
  setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
  setOperationAction(ISD::MEMSET, MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY, MVT::Other, Expand);
  
  // PowerPC has an i16 but no i8 (or i1) SEXTLOAD
  setOperationAction(ISD::SEXTLOAD, MVT::i1, Expand);
  setOperationAction(ISD::SEXTLOAD, MVT::i8, Expand);
  
  // PowerPC has no SREM/UREM instructions
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  
  // We don't support sin/cos/sqrt/fmod
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FREM , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::FREM , MVT::f32, Expand);
  
  // If we're enabling GP optimizations, use hardware square root
  if (!TM.getSubtarget<PPCSubtarget>().hasFSQRT()) {
    setOperationAction(ISD::FSQRT, MVT::f64, Expand);
    setOperationAction(ISD::FSQRT, MVT::f32, Expand);
  }
  
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
  
  // PowerPC does not have BSWAP, CTPOP or CTTZ
  setOperationAction(ISD::BSWAP, MVT::i32  , Expand);
  setOperationAction(ISD::CTPOP, MVT::i32  , Expand);
  setOperationAction(ISD::CTTZ , MVT::i32  , Expand);
  
  // PowerPC does not have ROTR
  setOperationAction(ISD::ROTR, MVT::i32   , Expand);
  
  // PowerPC does not have Select
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  setOperationAction(ISD::SELECT, MVT::f64, Expand);
  
  // PowerPC wants to turn select_cc of FP into fsel when possible.
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);

  // PowerPC wants to optimize integer setcc a bit
  setOperationAction(ISD::SETCC, MVT::i32, Custom);
  
  // PowerPC does not have BRCOND which requires SetCC
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);
  
  // PowerPC turns FP_TO_SINT into FCTIWZ and some load/stores.
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);

  // PowerPC does not have [U|S]INT_TO_FP
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);

  setOperationAction(ISD::BIT_CONVERT, MVT::f32, Expand);
  setOperationAction(ISD::BIT_CONVERT, MVT::i32, Expand);

  // PowerPC does not have truncstore for i1.
  setOperationAction(ISD::TRUNCSTORE, MVT::i1, Promote);

  // We cannot sextinreg(i1).  Expand to shifts.
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  
  
  // Support label based line numbers.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  // FIXME - use subtarget debug flags
  if (!TM.getSubtarget<PPCSubtarget>().isDarwin())
    setOperationAction(ISD::DEBUG_LABEL, MVT::Other, Expand);
  
  // We want to legalize GlobalAddress and ConstantPool nodes into the 
  // appropriate instructions to materialize the address.
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i32, Custom);
  setOperationAction(ISD::JumpTable,     MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i64, Custom);
  setOperationAction(ISD::JumpTable,     MVT::i64, Custom);
  
  // RET must be custom lowered, to meet ABI requirements
  setOperationAction(ISD::RET               , MVT::Other, Custom);
  
  // VASTART needs to be custom lowered to use the VarArgsFrameIndex
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);
  
  // Use the default implementation.
  setOperationAction(ISD::VAARG             , MVT::Other, Expand);
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE         , MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE      , MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Expand);
  
  // We want to custom lower some of our intrinsics.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  
  if (TM.getSubtarget<PPCSubtarget>().has64BitSupport()) {
    // They also have instructions for converting between i64 and fp.
    setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
    setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
    
    // FIXME: disable this lowered code.  This generates 64-bit register values,
    // and we don't model the fact that the top part is clobbered by calls.  We
    // need to flag these together so that the value isn't live across a call.
    //setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
    
    // To take advantage of the above i64 FP_TO_SINT, promote i32 FP_TO_UINT
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Promote);
  } else {
    // PowerPC does not have FP_TO_UINT on 32-bit implementations.
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  }

  if (TM.getSubtarget<PPCSubtarget>().use64BitRegs()) {
    // 64 bit PowerPC implementations can support i64 types directly
    addRegisterClass(MVT::i64, PPC::G8RCRegisterClass);
    // BUILD_PAIR can't be handled natively, and should be expanded to shl/or
    setOperationAction(ISD::BUILD_PAIR, MVT::i64, Expand);
  } else {
    // 32 bit PowerPC wants to expand i64 shifts itself.
    setOperationAction(ISD::SHL, MVT::i64, Custom);
    setOperationAction(ISD::SRL, MVT::i64, Custom);
    setOperationAction(ISD::SRA, MVT::i64, Custom);
  }

  if (TM.getSubtarget<PPCSubtarget>().hasAltivec()) {
    // First set operation action for all vector types to expand. Then we
    // will selectively turn on ones that can be effectively codegen'd.
    for (unsigned VT = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
         VT != (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++VT) {
      // add/sub are legal for all supported vector VT's.
      setOperationAction(ISD::ADD , (MVT::ValueType)VT, Legal);
      setOperationAction(ISD::SUB , (MVT::ValueType)VT, Legal);
      
      // We promote all shuffles to v16i8.
      setOperationAction(ISD::VECTOR_SHUFFLE, (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::VECTOR_SHUFFLE, (MVT::ValueType)VT, MVT::v16i8);

      // We promote all non-typed operations to v4i32.
      setOperationAction(ISD::AND   , (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::AND   , (MVT::ValueType)VT, MVT::v4i32);
      setOperationAction(ISD::OR    , (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::OR    , (MVT::ValueType)VT, MVT::v4i32);
      setOperationAction(ISD::XOR   , (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::XOR   , (MVT::ValueType)VT, MVT::v4i32);
      setOperationAction(ISD::LOAD  , (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::LOAD  , (MVT::ValueType)VT, MVT::v4i32);
      setOperationAction(ISD::SELECT, (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::SELECT, (MVT::ValueType)VT, MVT::v4i32);
      setOperationAction(ISD::STORE, (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::STORE, (MVT::ValueType)VT, MVT::v4i32);
      
      // No other operations are legal.
      setOperationAction(ISD::MUL , (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::SDIV, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::SREM, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::UDIV, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::UREM, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::FDIV, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::INSERT_VECTOR_ELT, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::BUILD_VECTOR, (MVT::ValueType)VT, Expand);

      setOperationAction(ISD::SCALAR_TO_VECTOR, (MVT::ValueType)VT, Expand);
    }

    // We can custom expand all VECTOR_SHUFFLEs to VPERM, others we can handle
    // with merges, splats, etc.
    setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16i8, Custom);

    setOperationAction(ISD::AND   , MVT::v4i32, Legal);
    setOperationAction(ISD::OR    , MVT::v4i32, Legal);
    setOperationAction(ISD::XOR   , MVT::v4i32, Legal);
    setOperationAction(ISD::LOAD  , MVT::v4i32, Legal);
    setOperationAction(ISD::SELECT, MVT::v4i32, Expand);
    setOperationAction(ISD::STORE , MVT::v4i32, Legal);
    
    addRegisterClass(MVT::v4f32, PPC::VRRCRegisterClass);
    addRegisterClass(MVT::v4i32, PPC::VRRCRegisterClass);
    addRegisterClass(MVT::v8i16, PPC::VRRCRegisterClass);
    addRegisterClass(MVT::v16i8, PPC::VRRCRegisterClass);
    
    setOperationAction(ISD::MUL, MVT::v4f32, Legal);
    setOperationAction(ISD::MUL, MVT::v4i32, Custom);
    setOperationAction(ISD::MUL, MVT::v8i16, Custom);
    setOperationAction(ISD::MUL, MVT::v16i8, Custom);

    setOperationAction(ISD::SCALAR_TO_VECTOR, MVT::v4f32, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR, MVT::v4i32, Custom);
    
    setOperationAction(ISD::BUILD_VECTOR, MVT::v16i8, Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v8i16, Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v4i32, Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v4f32, Custom);
  }
  
  setSetCCResultType(MVT::i32);
  setShiftAmountType(MVT::i32);
  setSetCCResultContents(ZeroOrOneSetCCResult);
  setStackPointerRegisterToSaveRestore(PPC::R1);
  
  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::SINT_TO_FP);
  setTargetDAGCombine(ISD::STORE);
  setTargetDAGCombine(ISD::BR_CC);
  
  computeRegisterProperties();
}

const char *PPCTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case PPCISD::FSEL:          return "PPCISD::FSEL";
  case PPCISD::FCFID:         return "PPCISD::FCFID";
  case PPCISD::FCTIDZ:        return "PPCISD::FCTIDZ";
  case PPCISD::FCTIWZ:        return "PPCISD::FCTIWZ";
  case PPCISD::STFIWX:        return "PPCISD::STFIWX";
  case PPCISD::VMADDFP:       return "PPCISD::VMADDFP";
  case PPCISD::VNMSUBFP:      return "PPCISD::VNMSUBFP";
  case PPCISD::VPERM:         return "PPCISD::VPERM";
  case PPCISD::Hi:            return "PPCISD::Hi";
  case PPCISD::Lo:            return "PPCISD::Lo";
  case PPCISD::GlobalBaseReg: return "PPCISD::GlobalBaseReg";
  case PPCISD::SRL:           return "PPCISD::SRL";
  case PPCISD::SRA:           return "PPCISD::SRA";
  case PPCISD::SHL:           return "PPCISD::SHL";
  case PPCISD::EXTSW_32:      return "PPCISD::EXTSW_32";
  case PPCISD::STD_32:        return "PPCISD::STD_32";
  case PPCISD::CALL:          return "PPCISD::CALL";
  case PPCISD::MTCTR:         return "PPCISD::MTCTR";
  case PPCISD::BCTRL:         return "PPCISD::BCTRL";
  case PPCISD::RET_FLAG:      return "PPCISD::RET_FLAG";
  case PPCISD::MFCR:          return "PPCISD::MFCR";
  case PPCISD::VCMP:          return "PPCISD::VCMP";
  case PPCISD::VCMPo:         return "PPCISD::VCMPo";
  case PPCISD::COND_BRANCH:   return "PPCISD::COND_BRANCH";
  }
}

//===----------------------------------------------------------------------===//
// Node matching predicates, for use by the tblgen matching code.
//===----------------------------------------------------------------------===//

/// isFloatingPointZero - Return true if this is 0.0 or -0.0.
static bool isFloatingPointZero(SDOperand Op) {
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Op))
    return CFP->isExactlyValue(-0.0) || CFP->isExactlyValue(0.0);
  else if (Op.getOpcode() == ISD::EXTLOAD || Op.getOpcode() == ISD::LOAD) {
    // Maybe this has already been legalized into the constant pool?
    if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Op.getOperand(1)))
      if (ConstantFP *CFP = dyn_cast<ConstantFP>(CP->get()))
        return CFP->isExactlyValue(-0.0) || CFP->isExactlyValue(0.0);
  }
  return false;
}

/// isConstantOrUndef - Op is either an undef node or a ConstantSDNode.  Return
/// true if Op is undef or if it matches the specified value.
static bool isConstantOrUndef(SDOperand Op, unsigned Val) {
  return Op.getOpcode() == ISD::UNDEF || 
         cast<ConstantSDNode>(Op)->getValue() == Val;
}

/// isVPKUHUMShuffleMask - Return true if this is the shuffle mask for a
/// VPKUHUM instruction.
bool PPC::isVPKUHUMShuffleMask(SDNode *N, bool isUnary) {
  if (!isUnary) {
    for (unsigned i = 0; i != 16; ++i)
      if (!isConstantOrUndef(N->getOperand(i),  i*2+1))
        return false;
  } else {
    for (unsigned i = 0; i != 8; ++i)
      if (!isConstantOrUndef(N->getOperand(i),  i*2+1) ||
          !isConstantOrUndef(N->getOperand(i+8),  i*2+1))
        return false;
  }
  return true;
}

/// isVPKUWUMShuffleMask - Return true if this is the shuffle mask for a
/// VPKUWUM instruction.
bool PPC::isVPKUWUMShuffleMask(SDNode *N, bool isUnary) {
  if (!isUnary) {
    for (unsigned i = 0; i != 16; i += 2)
      if (!isConstantOrUndef(N->getOperand(i  ),  i*2+2) ||
          !isConstantOrUndef(N->getOperand(i+1),  i*2+3))
        return false;
  } else {
    for (unsigned i = 0; i != 8; i += 2)
      if (!isConstantOrUndef(N->getOperand(i  ),  i*2+2) ||
          !isConstantOrUndef(N->getOperand(i+1),  i*2+3) ||
          !isConstantOrUndef(N->getOperand(i+8),  i*2+2) ||
          !isConstantOrUndef(N->getOperand(i+9),  i*2+3))
        return false;
  }
  return true;
}

/// isVMerge - Common function, used to match vmrg* shuffles.
///
static bool isVMerge(SDNode *N, unsigned UnitSize, 
                     unsigned LHSStart, unsigned RHSStart) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR &&
         N->getNumOperands() == 16 && "PPC only supports shuffles by bytes!");
  assert((UnitSize == 1 || UnitSize == 2 || UnitSize == 4) &&
         "Unsupported merge size!");
  
  for (unsigned i = 0; i != 8/UnitSize; ++i)     // Step over units
    for (unsigned j = 0; j != UnitSize; ++j) {   // Step over bytes within unit
      if (!isConstantOrUndef(N->getOperand(i*UnitSize*2+j),
                             LHSStart+j+i*UnitSize) ||
          !isConstantOrUndef(N->getOperand(i*UnitSize*2+UnitSize+j),
                             RHSStart+j+i*UnitSize))
        return false;
    }
      return true;
}

/// isVMRGLShuffleMask - Return true if this is a shuffle mask suitable for
/// a VRGL* instruction with the specified unit size (1,2 or 4 bytes).
bool PPC::isVMRGLShuffleMask(SDNode *N, unsigned UnitSize, bool isUnary) {
  if (!isUnary)
    return isVMerge(N, UnitSize, 8, 24);
  return isVMerge(N, UnitSize, 8, 8);
}

/// isVMRGHShuffleMask - Return true if this is a shuffle mask suitable for
/// a VRGH* instruction with the specified unit size (1,2 or 4 bytes).
bool PPC::isVMRGHShuffleMask(SDNode *N, unsigned UnitSize, bool isUnary) {
  if (!isUnary)
    return isVMerge(N, UnitSize, 0, 16);
  return isVMerge(N, UnitSize, 0, 0);
}


/// isVSLDOIShuffleMask - If this is a vsldoi shuffle mask, return the shift
/// amount, otherwise return -1.
int PPC::isVSLDOIShuffleMask(SDNode *N, bool isUnary) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR &&
         N->getNumOperands() == 16 && "PPC only supports shuffles by bytes!");
  // Find the first non-undef value in the shuffle mask.
  unsigned i;
  for (i = 0; i != 16 && N->getOperand(i).getOpcode() == ISD::UNDEF; ++i)
    /*search*/;
  
  if (i == 16) return -1;  // all undef.
  
  // Otherwise, check to see if the rest of the elements are consequtively
  // numbered from this value.
  unsigned ShiftAmt = cast<ConstantSDNode>(N->getOperand(i))->getValue();
  if (ShiftAmt < i) return -1;
  ShiftAmt -= i;

  if (!isUnary) {
    // Check the rest of the elements to see if they are consequtive.
    for (++i; i != 16; ++i)
      if (!isConstantOrUndef(N->getOperand(i), ShiftAmt+i))
        return -1;
  } else {
    // Check the rest of the elements to see if they are consequtive.
    for (++i; i != 16; ++i)
      if (!isConstantOrUndef(N->getOperand(i), (ShiftAmt+i) & 15))
        return -1;
  }
  
  return ShiftAmt;
}

/// isSplatShuffleMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a splat of a single element that is suitable for input to
/// VSPLTB/VSPLTH/VSPLTW.
bool PPC::isSplatShuffleMask(SDNode *N, unsigned EltSize) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR &&
         N->getNumOperands() == 16 &&
         (EltSize == 1 || EltSize == 2 || EltSize == 4));
  
  // This is a splat operation if each element of the permute is the same, and
  // if the value doesn't reference the second vector.
  unsigned ElementBase = 0;
  SDOperand Elt = N->getOperand(0);
  if (ConstantSDNode *EltV = dyn_cast<ConstantSDNode>(Elt))
    ElementBase = EltV->getValue();
  else
    return false;   // FIXME: Handle UNDEF elements too!

  if (cast<ConstantSDNode>(Elt)->getValue() >= 16)
    return false;
  
  // Check that they are consequtive.
  for (unsigned i = 1; i != EltSize; ++i) {
    if (!isa<ConstantSDNode>(N->getOperand(i)) ||
        cast<ConstantSDNode>(N->getOperand(i))->getValue() != i+ElementBase)
      return false;
  }
  
  assert(isa<ConstantSDNode>(Elt) && "Invalid VECTOR_SHUFFLE mask!");
  for (unsigned i = EltSize, e = 16; i != e; i += EltSize) {
    if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(N->getOperand(i)) &&
           "Invalid VECTOR_SHUFFLE mask!");
    for (unsigned j = 0; j != EltSize; ++j)
      if (N->getOperand(i+j) != N->getOperand(j))
        return false;
  }

  return true;
}

/// getVSPLTImmediate - Return the appropriate VSPLT* immediate to splat the
/// specified isSplatShuffleMask VECTOR_SHUFFLE mask.
unsigned PPC::getVSPLTImmediate(SDNode *N, unsigned EltSize) {
  assert(isSplatShuffleMask(N, EltSize));
  return cast<ConstantSDNode>(N->getOperand(0))->getValue() / EltSize;
}

/// get_VSPLTI_elt - If this is a build_vector of constants which can be formed
/// by using a vspltis[bhw] instruction of the specified element size, return
/// the constant being splatted.  The ByteSize field indicates the number of
/// bytes of each element [124] -> [bhw].
SDOperand PPC::get_VSPLTI_elt(SDNode *N, unsigned ByteSize, SelectionDAG &DAG) {
  SDOperand OpVal(0, 0);

  // If ByteSize of the splat is bigger than the element size of the
  // build_vector, then we have a case where we are checking for a splat where
  // multiple elements of the buildvector are folded together into a single
  // logical element of the splat (e.g. "vsplish 1" to splat {0,1}*8).
  unsigned EltSize = 16/N->getNumOperands();
  if (EltSize < ByteSize) {
    unsigned Multiple = ByteSize/EltSize;   // Number of BV entries per spltval.
    SDOperand UniquedVals[4];
    assert(Multiple > 1 && Multiple <= 4 && "How can this happen?");
    
    // See if all of the elements in the buildvector agree across.
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
      if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
      // If the element isn't a constant, bail fully out.
      if (!isa<ConstantSDNode>(N->getOperand(i))) return SDOperand();

          
      if (UniquedVals[i&(Multiple-1)].Val == 0)
        UniquedVals[i&(Multiple-1)] = N->getOperand(i);
      else if (UniquedVals[i&(Multiple-1)] != N->getOperand(i))
        return SDOperand();  // no match.
    }
    
    // Okay, if we reached this point, UniquedVals[0..Multiple-1] contains
    // either constant or undef values that are identical for each chunk.  See
    // if these chunks can form into a larger vspltis*.
    
    // Check to see if all of the leading entries are either 0 or -1.  If
    // neither, then this won't fit into the immediate field.
    bool LeadingZero = true;
    bool LeadingOnes = true;
    for (unsigned i = 0; i != Multiple-1; ++i) {
      if (UniquedVals[i].Val == 0) continue;  // Must have been undefs.
      
      LeadingZero &= cast<ConstantSDNode>(UniquedVals[i])->isNullValue();
      LeadingOnes &= cast<ConstantSDNode>(UniquedVals[i])->isAllOnesValue();
    }
    // Finally, check the least significant entry.
    if (LeadingZero) {
      if (UniquedVals[Multiple-1].Val == 0)
        return DAG.getTargetConstant(0, MVT::i32);  // 0,0,0,undef
      int Val = cast<ConstantSDNode>(UniquedVals[Multiple-1])->getValue();
      if (Val < 16)
        return DAG.getTargetConstant(Val, MVT::i32);  // 0,0,0,4 -> vspltisw(4)
    }
    if (LeadingOnes) {
      if (UniquedVals[Multiple-1].Val == 0)
        return DAG.getTargetConstant(~0U, MVT::i32);  // -1,-1,-1,undef
      int Val =cast<ConstantSDNode>(UniquedVals[Multiple-1])->getSignExtended();
      if (Val >= -16)                            // -1,-1,-1,-2 -> vspltisw(-2)
        return DAG.getTargetConstant(Val, MVT::i32);
    }
    
    return SDOperand();
  }
  
  // Check to see if this buildvec has a single non-undef value in its elements.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    if (OpVal.Val == 0)
      OpVal = N->getOperand(i);
    else if (OpVal != N->getOperand(i))
      return SDOperand();
  }
  
  if (OpVal.Val == 0) return SDOperand();  // All UNDEF: use implicit def.
  
  unsigned ValSizeInBytes = 0;
  uint64_t Value = 0;
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(OpVal)) {
    Value = CN->getValue();
    ValSizeInBytes = MVT::getSizeInBits(CN->getValueType(0))/8;
  } else if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(OpVal)) {
    assert(CN->getValueType(0) == MVT::f32 && "Only one legal FP vector type!");
    Value = FloatToBits(CN->getValue());
    ValSizeInBytes = 4;
  }

  // If the splat value is larger than the element value, then we can never do
  // this splat.  The only case that we could fit the replicated bits into our
  // immediate field for would be zero, and we prefer to use vxor for it.
  if (ValSizeInBytes < ByteSize) return SDOperand();
  
  // If the element value is larger than the splat value, cut it in half and
  // check to see if the two halves are equal.  Continue doing this until we
  // get to ByteSize.  This allows us to handle 0x01010101 as 0x01.
  while (ValSizeInBytes > ByteSize) {
    ValSizeInBytes >>= 1;
    
    // If the top half equals the bottom half, we're still ok.
    if (((Value >> (ValSizeInBytes*8)) & ((1 << (8*ValSizeInBytes))-1)) !=
         (Value                        & ((1 << (8*ValSizeInBytes))-1)))
      return SDOperand();
  }

  // Properly sign extend the value.
  int ShAmt = (4-ByteSize)*8;
  int MaskVal = ((int)Value << ShAmt) >> ShAmt;
  
  // If this is zero, don't match, zero matches ISD::isBuildVectorAllZeros.
  if (MaskVal == 0) return SDOperand();

  // Finally, if this value fits in a 5 bit sext field, return it
  if (((MaskVal << (32-5)) >> (32-5)) == MaskVal)
    return DAG.getTargetConstant(MaskVal, MVT::i32);
  return SDOperand();
}

//===----------------------------------------------------------------------===//
//  LowerOperation implementation
//===----------------------------------------------------------------------===//

static SDOperand LowerConstantPool(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType PtrVT = Op.getValueType();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  Constant *C = CP->get();
  SDOperand CPI = DAG.getTargetConstantPool(C, PtrVT, CP->getAlignment());
  SDOperand Zero = DAG.getConstant(0, PtrVT);

  const TargetMachine &TM = DAG.getTarget();
  
  SDOperand Hi = DAG.getNode(PPCISD::Hi, PtrVT, CPI, Zero);
  SDOperand Lo = DAG.getNode(PPCISD::Lo, PtrVT, CPI, Zero);

  // If this is a non-darwin platform, we don't support non-static relo models
  // yet.
  if (TM.getRelocationModel() == Reloc::Static ||
      !TM.getSubtarget<PPCSubtarget>().isDarwin()) {
    // Generate non-pic code that has direct accesses to the constant pool.
    // The address of the global is just (hi(&g)+lo(&g)).
    return DAG.getNode(ISD::ADD, PtrVT, Hi, Lo);
  }
  
  if (TM.getRelocationModel() == Reloc::PIC) {
    // With PIC, the first instruction is actually "GR+hi(&G)".
    Hi = DAG.getNode(ISD::ADD, PtrVT,
                     DAG.getNode(PPCISD::GlobalBaseReg, PtrVT), Hi);
  }
  
  Lo = DAG.getNode(ISD::ADD, PtrVT, Hi, Lo);
  return Lo;
}

static SDOperand LowerJumpTable(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType PtrVT = Op.getValueType();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDOperand JTI = DAG.getTargetJumpTable(JT->getIndex(), PtrVT);
  SDOperand Zero = DAG.getConstant(0, PtrVT);
  
  const TargetMachine &TM = DAG.getTarget();

  SDOperand Hi = DAG.getNode(PPCISD::Hi, PtrVT, JTI, Zero);
  SDOperand Lo = DAG.getNode(PPCISD::Lo, PtrVT, JTI, Zero);

  // If this is a non-darwin platform, we don't support non-static relo models
  // yet.
  if (TM.getRelocationModel() == Reloc::Static ||
      !TM.getSubtarget<PPCSubtarget>().isDarwin()) {
    // Generate non-pic code that has direct accesses to the constant pool.
    // The address of the global is just (hi(&g)+lo(&g)).
    return DAG.getNode(ISD::ADD, PtrVT, Hi, Lo);
  }
  
  if (TM.getRelocationModel() == Reloc::PIC) {
    // With PIC, the first instruction is actually "GR+hi(&G)".
    Hi = DAG.getNode(ISD::ADD, PtrVT,
                     DAG.getNode(PPCISD::GlobalBaseReg, MVT::i32), Hi);
  }
  
  Lo = DAG.getNode(ISD::ADD, PtrVT, Hi, Lo);
  return Lo;
}

static SDOperand LowerGlobalAddress(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType PtrVT = Op.getValueType();
  GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op);
  GlobalValue *GV = GSDN->getGlobal();
  SDOperand GA = DAG.getTargetGlobalAddress(GV, PtrVT, GSDN->getOffset());
  SDOperand Zero = DAG.getConstant(0, PtrVT);
  
  const TargetMachine &TM = DAG.getTarget();

  SDOperand Hi = DAG.getNode(PPCISD::Hi, PtrVT, GA, Zero);
  SDOperand Lo = DAG.getNode(PPCISD::Lo, PtrVT, GA, Zero);

  // If this is a non-darwin platform, we don't support non-static relo models
  // yet.
  if (TM.getRelocationModel() == Reloc::Static ||
      !TM.getSubtarget<PPCSubtarget>().isDarwin()) {
    // Generate non-pic code that has direct accesses to globals.
    // The address of the global is just (hi(&g)+lo(&g)).
    return DAG.getNode(ISD::ADD, PtrVT, Hi, Lo);
  }
  
  if (TM.getRelocationModel() == Reloc::PIC) {
    // With PIC, the first instruction is actually "GR+hi(&G)".
    Hi = DAG.getNode(ISD::ADD, PtrVT,
                     DAG.getNode(PPCISD::GlobalBaseReg, PtrVT), Hi);
  }
  
  Lo = DAG.getNode(ISD::ADD, PtrVT, Hi, Lo);
  
  if (!GV->hasWeakLinkage() && !GV->hasLinkOnceLinkage() &&
      (!GV->isExternal() || GV->hasNotBeenReadFromBytecode()))
    return Lo;
  
  // If the global is weak or external, we have to go through the lazy
  // resolution stub.
  return DAG.getLoad(PtrVT, DAG.getEntryNode(), Lo, DAG.getSrcValue(0));
}

static SDOperand LowerSETCC(SDOperand Op, SelectionDAG &DAG) {
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  
  // If we're comparing for equality to zero, expose the fact that this is
  // implented as a ctlz/srl pair on ppc, so that the dag combiner can
  // fold the new nodes.
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
    if (C->isNullValue() && CC == ISD::SETEQ) {
      MVT::ValueType VT = Op.getOperand(0).getValueType();
      SDOperand Zext = Op.getOperand(0);
      if (VT < MVT::i32) {
        VT = MVT::i32;
        Zext = DAG.getNode(ISD::ZERO_EXTEND, VT, Op.getOperand(0));
      } 
      unsigned Log2b = Log2_32(MVT::getSizeInBits(VT));
      SDOperand Clz = DAG.getNode(ISD::CTLZ, VT, Zext);
      SDOperand Scc = DAG.getNode(ISD::SRL, VT, Clz,
                                  DAG.getConstant(Log2b, MVT::i32));
      return DAG.getNode(ISD::TRUNCATE, MVT::i32, Scc);
    }
    // Leave comparisons against 0 and -1 alone for now, since they're usually 
    // optimized.  FIXME: revisit this when we can custom lower all setcc
    // optimizations.
    if (C->isAllOnesValue() || C->isNullValue())
      return SDOperand();
  }
  
  // If we have an integer seteq/setne, turn it into a compare against zero
  // by subtracting the rhs from the lhs, which is faster than setting a
  // condition register, reading it back out, and masking the correct bit.
  MVT::ValueType LHSVT = Op.getOperand(0).getValueType();
  if (MVT::isInteger(LHSVT) && (CC == ISD::SETEQ || CC == ISD::SETNE)) {
    MVT::ValueType VT = Op.getValueType();
    SDOperand Sub = DAG.getNode(ISD::SUB, LHSVT, Op.getOperand(0), 
                                Op.getOperand(1));
    return DAG.getSetCC(VT, Sub, DAG.getConstant(0, LHSVT), CC);
  }
  return SDOperand();
}

static SDOperand LowerVASTART(SDOperand Op, SelectionDAG &DAG,
                              unsigned VarArgsFrameIndex) {
  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i32);
  return DAG.getNode(ISD::STORE, MVT::Other, Op.getOperand(0), FR, 
                     Op.getOperand(1), Op.getOperand(2));
}

static SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG,
                                       int &VarArgsFrameIndex) {
  // TODO: add description of PPC stack frame format, or at least some docs.
  //
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SSARegMap *RegMap = MF.getSSARegMap();
  std::vector<SDOperand> ArgValues;
  SDOperand Root = Op.getOperand(0);
  
  unsigned ArgOffset = 24;
  const unsigned Num_GPR_Regs = 8;
  const unsigned Num_FPR_Regs = 13;
  const unsigned Num_VR_Regs  = 12;
  unsigned GPR_idx = 0, FPR_idx = 0, VR_idx = 0;
  
  static const unsigned GPR_32[] = {           // 32-bit registers.
    PPC::R3, PPC::R4, PPC::R5, PPC::R6,
    PPC::R7, PPC::R8, PPC::R9, PPC::R10,
  };
  static const unsigned GPR_64[] = {           // 64-bit registers.
    PPC::X3, PPC::X4, PPC::X5, PPC::X6,
    PPC::X7, PPC::X8, PPC::X9, PPC::X10,
  };
  static const unsigned FPR[] = {
    PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
    PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, PPC::F13
  };
  static const unsigned VR[] = {
    PPC::V2, PPC::V3, PPC::V4, PPC::V5, PPC::V6, PPC::V7, PPC::V8,
    PPC::V9, PPC::V10, PPC::V11, PPC::V12, PPC::V13
  };

  MVT::ValueType PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  bool isPPC64 = PtrVT == MVT::i64;
  const unsigned *GPR = isPPC64 ? GPR_64 : GPR_32;
  
  // Add DAG nodes to load the arguments or copy them out of registers.  On
  // entry to a function on PPC, the arguments start at offset 24, although the
  // first ones are often in registers.
  for (unsigned ArgNo = 0, e = Op.Val->getNumValues()-1; ArgNo != e; ++ArgNo) {
    SDOperand ArgVal;
    bool needsLoad = false;
    MVT::ValueType ObjectVT = Op.getValue(ArgNo).getValueType();
    unsigned ObjSize = MVT::getSizeInBits(ObjectVT)/8;

    unsigned CurArgOffset = ArgOffset;
    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i32:
      // All int arguments reserve stack space.
      ArgOffset += isPPC64 ? 8 : 4;

      if (GPR_idx != Num_GPR_Regs) {
        unsigned VReg = RegMap->createVirtualRegister(&PPC::GPRCRegClass);
        MF.addLiveIn(GPR[GPR_idx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, MVT::i32);
        ++GPR_idx;
      } else {
        needsLoad = true;
      }
      break;
    case MVT::i64:  // PPC64
      // All int arguments reserve stack space.
      ArgOffset += 8;
      
      if (GPR_idx != Num_GPR_Regs) {
        unsigned VReg = RegMap->createVirtualRegister(&PPC::G8RCRegClass);
        MF.addLiveIn(GPR[GPR_idx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, MVT::i64);
        ++GPR_idx;
      } else {
        needsLoad = true;
      }
      break;
    case MVT::f32:
    case MVT::f64:
      // All FP arguments reserve stack space.
      ArgOffset += ObjSize;

      // Every 4 bytes of argument space consumes one of the GPRs available for
      // argument passing.
      if (GPR_idx != Num_GPR_Regs) {
        ++GPR_idx;
        if (ObjSize == 8 && GPR_idx != Num_GPR_Regs)
          ++GPR_idx;
      }
      if (FPR_idx != Num_FPR_Regs) {
        unsigned VReg;
        if (ObjectVT == MVT::f32)
          VReg = RegMap->createVirtualRegister(&PPC::F4RCRegClass);
        else
          VReg = RegMap->createVirtualRegister(&PPC::F8RCRegClass);
        MF.addLiveIn(FPR[FPR_idx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, ObjectVT);
        ++FPR_idx;
      } else {
        needsLoad = true;
      }
      break;
    case MVT::v4f32:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v16i8:
      // Note that vector arguments in registers don't reserve stack space.
      if (VR_idx != Num_VR_Regs) {
        unsigned VReg = RegMap->createVirtualRegister(&PPC::VRRCRegClass);
        MF.addLiveIn(VR[VR_idx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, ObjectVT);
        ++VR_idx;
      } else {
        // This should be simple, but requires getting 16-byte aligned stack
        // values.
        assert(0 && "Loading VR argument not implemented yet!");
        needsLoad = true;
      }
      break;
    }
    
    // We need to load the argument to a virtual register if we determined above
    // that we ran out of physical registers of the appropriate type
    if (needsLoad) {
      // If the argument is actually used, emit a load from the right stack
      // slot.
      if (!Op.Val->hasNUsesOfValue(0, ArgNo)) {
        int FI = MFI->CreateFixedObject(ObjSize, CurArgOffset);
        SDOperand FIN = DAG.getFrameIndex(FI, PtrVT);
        ArgVal = DAG.getLoad(ObjectVT, Root, FIN,
                             DAG.getSrcValue(NULL));
      } else {
        // Don't emit a dead load.
        ArgVal = DAG.getNode(ISD::UNDEF, ObjectVT);
      }
    }
    
    ArgValues.push_back(ArgVal);
  }
  
  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  if (isVarArg) {
    VarArgsFrameIndex = MFI->CreateFixedObject(MVT::getSizeInBits(PtrVT)/8,
                                               ArgOffset);
    SDOperand FIN = DAG.getFrameIndex(VarArgsFrameIndex, PtrVT);
    // If this function is vararg, store any remaining integer argument regs
    // to their spots on the stack so that they may be loaded by deferencing the
    // result of va_next.
    std::vector<SDOperand> MemOps;
    for (; GPR_idx != Num_GPR_Regs; ++GPR_idx) {
      unsigned VReg = RegMap->createVirtualRegister(&PPC::GPRCRegClass);
      MF.addLiveIn(GPR[GPR_idx], VReg);
      SDOperand Val = DAG.getCopyFromReg(Root, VReg, PtrVT);
      SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Val.getValue(1),
                                    Val, FIN, DAG.getSrcValue(NULL));
      MemOps.push_back(Store);
      // Increment the address by four for the next argument to store
      SDOperand PtrOff = DAG.getConstant(MVT::getSizeInBits(PtrVT)/8, PtrVT);
      FIN = DAG.getNode(ISD::ADD, PtrOff.getValueType(), FIN, PtrOff);
    }
    if (!MemOps.empty())
      Root = DAG.getNode(ISD::TokenFactor, MVT::Other, MemOps);
  }
  
  ArgValues.push_back(Root);
 
  // Return the new list of results.
  std::vector<MVT::ValueType> RetVT(Op.Val->value_begin(),
                                    Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVT, ArgValues);
}

/// isCallCompatibleAddress - Return the immediate to use if the specified
/// 32-bit value is representable in the immediate field of a BxA instruction.
static SDNode *isBLACompatibleAddress(SDOperand Op, SelectionDAG &DAG) {
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
  if (!C) return 0;
  
  int Addr = C->getValue();
  if ((Addr & 3) != 0 ||  // Low 2 bits are implicitly zero.
      (Addr << 6 >> 6) != Addr)
    return 0;  // Top 6 bits have to be sext of immediate.
  
  return DAG.getConstant((int)C->getValue() >> 2, MVT::i32).Val;
}


static SDOperand LowerCALL(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Chain = Op.getOperand(0);
  unsigned CallingConv= cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  bool isVarArg       = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  bool isTailCall     = cast<ConstantSDNode>(Op.getOperand(3))->getValue() != 0;
  SDOperand Callee    = Op.getOperand(4);
  unsigned NumOps     = (Op.getNumOperands() - 5) / 2;

  MVT::ValueType PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  bool isPPC64 = PtrVT == MVT::i64;
  unsigned PtrByteSize = isPPC64 ? 8 : 4;

  
  // args_to_use will accumulate outgoing args for the PPCISD::CALL case in
  // SelectExpr to use to put the arguments in the appropriate registers.
  std::vector<SDOperand> args_to_use;
  
  // Count how many bytes are to be pushed on the stack, including the linkage
  // area, and parameter passing area.  We start with 24/48 bytes, which is
  // prereserved space for [SP][CR][LR][3 x unused].
  unsigned NumBytes = 6*PtrByteSize;
  
  // Add up all the space actually used.
  for (unsigned i = 0; i != NumOps; ++i)
    NumBytes += MVT::getSizeInBits(Op.getOperand(5+2*i).getValueType())/8;

  // The prolog code of the callee may store up to 8 GPR argument registers to
  // the stack, allowing va_start to index over them in memory if its varargs.
  // Because we cannot tell if this is needed on the caller side, we have to
  // conservatively assume that it is needed.  As such, make sure we have at
  // least enough stack space for the caller to store the 8 GPRs.
  if (NumBytes < 6*PtrByteSize+8*PtrByteSize)
    NumBytes = 6*PtrByteSize+8*PtrByteSize;
  
  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  Chain = DAG.getCALLSEQ_START(Chain,
                               DAG.getConstant(NumBytes, PtrVT));
  
  // Set up a copy of the stack pointer for use loading and storing any
  // arguments that may not fit in the registers available for argument
  // passing.
  SDOperand StackPtr;
  if (isPPC64)
    StackPtr = DAG.getRegister(PPC::X1, MVT::i64);
  else
    StackPtr = DAG.getRegister(PPC::R1, MVT::i32);
  
  // Figure out which arguments are going to go in registers, and which in
  // memory.  Also, if this is a vararg function, floating point operations
  // must be stored to our stack, and loaded into integer regs as well, if
  // any integer regs are available for argument passing.
  unsigned ArgOffset = 6*PtrByteSize;
  unsigned GPR_idx = 0, FPR_idx = 0, VR_idx = 0;
  static const unsigned GPR_32[] = {           // 32-bit registers.
    PPC::R3, PPC::R4, PPC::R5, PPC::R6,
    PPC::R7, PPC::R8, PPC::R9, PPC::R10,
  };
  static const unsigned GPR_64[] = {           // 64-bit registers.
    PPC::X3, PPC::X4, PPC::X5, PPC::X6,
    PPC::X7, PPC::X8, PPC::X9, PPC::X10,
  };
  static const unsigned FPR[] = {
    PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
    PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, PPC::F13
  };
  static const unsigned VR[] = {
    PPC::V2, PPC::V3, PPC::V4, PPC::V5, PPC::V6, PPC::V7, PPC::V8,
    PPC::V9, PPC::V10, PPC::V11, PPC::V12, PPC::V13
  };
  const unsigned NumGPRs = sizeof(GPR_32)/sizeof(GPR_32[0]);
  const unsigned NumFPRs = sizeof(FPR)/sizeof(FPR[0]);
  const unsigned NumVRs  = sizeof( VR)/sizeof( VR[0]);
  
  const unsigned *GPR = isPPC64 ? GPR_64 : GPR_32;

  std::vector<std::pair<unsigned, SDOperand> > RegsToPass;
  std::vector<SDOperand> MemOpChains;
  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);
    
    // PtrOff will be used to store the current argument to the stack if a
    // register cannot be found for it.
    SDOperand PtrOff = DAG.getConstant(ArgOffset, StackPtr.getValueType());
    PtrOff = DAG.getNode(ISD::ADD, PtrVT, StackPtr, PtrOff);

    // On PPC64, promote integers to 64-bit values.
    if (isPPC64 && Arg.getValueType() == MVT::i32) {
      unsigned ExtOp = ISD::ZERO_EXTEND;
      if (cast<ConstantSDNode>(Op.getOperand(5+2*i+1))->getValue())
        ExtOp = ISD::SIGN_EXTEND;
      Arg = DAG.getNode(ExtOp, MVT::i64, Arg);
    }
    
    switch (Arg.getValueType()) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i32:
    case MVT::i64:
      if (GPR_idx != NumGPRs) {
        RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Arg));
      } else {
        MemOpChains.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                          Arg, PtrOff, DAG.getSrcValue(NULL)));
      }
      ArgOffset += PtrByteSize;
      break;
    case MVT::f32:
    case MVT::f64:
      if (FPR_idx != NumFPRs) {
        RegsToPass.push_back(std::make_pair(FPR[FPR_idx++], Arg));

        if (isVarArg) {
          SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                        Arg, PtrOff,
                                        DAG.getSrcValue(NULL));
          MemOpChains.push_back(Store);

          // Float varargs are always shadowed in available integer registers
          if (GPR_idx != NumGPRs) {
            SDOperand Load = DAG.getLoad(PtrVT, Store, PtrOff,
                                         DAG.getSrcValue(NULL));
            MemOpChains.push_back(Load.getValue(1));
            RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));
          }
          if (GPR_idx != NumGPRs && Arg.getValueType() == MVT::f64) {
            SDOperand ConstFour = DAG.getConstant(4, PtrOff.getValueType());
            PtrOff = DAG.getNode(ISD::ADD, PtrVT, PtrOff, ConstFour);
            SDOperand Load = DAG.getLoad(PtrVT, Store, PtrOff,
                                         DAG.getSrcValue(NULL));
            MemOpChains.push_back(Load.getValue(1));
            RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));
          }
        } else {
          // If we have any FPRs remaining, we may also have GPRs remaining.
          // Args passed in FPRs consume either 1 (f32) or 2 (f64) available
          // GPRs.
          if (GPR_idx != NumGPRs)
            ++GPR_idx;
          if (GPR_idx != NumGPRs && Arg.getValueType() == MVT::f64 && !isPPC64)
            ++GPR_idx;
        }
      } else {
        MemOpChains.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                          Arg, PtrOff, DAG.getSrcValue(NULL)));
      }
      if (isPPC64)
        ArgOffset += 8;
      else
        ArgOffset += Arg.getValueType() == MVT::f32 ? 4 : 8;
      break;
    case MVT::v4f32:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v16i8:
      assert(!isVarArg && "Don't support passing vectors to varargs yet!");
      assert(VR_idx != NumVRs &&
             "Don't support passing more than 12 vector args yet!");
      RegsToPass.push_back(std::make_pair(VR[VR_idx++], Arg));
      break;
    }
  }
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, MemOpChains);
  
  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDOperand InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, RegsToPass[i].second,
                             InFlag);
    InFlag = Chain.getValue(1);
  }
  
  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.

  std::vector<SDOperand> Ops;
  unsigned CallOpc = PPCISD::CALL;
  
  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), Callee.getValueType());
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), Callee.getValueType());
  else if (SDNode *Dest = isBLACompatibleAddress(Callee, DAG))
    // If this is an absolute destination address, use the munged value.
    Callee = SDOperand(Dest, 0);
  else {
    // Otherwise, this is an indirect call.  We have to use a MTCTR/BCTRL pair
    // to do the call, we can't use PPCISD::CALL.
    Ops.push_back(Chain);
    Ops.push_back(Callee);
    
    if (InFlag.Val)
      Ops.push_back(InFlag);
    Chain = DAG.getNode(PPCISD::MTCTR, NodeTys, Ops);
    InFlag = Chain.getValue(1);
    
    // Copy the callee address into R12 on darwin.
    Chain = DAG.getCopyToReg(Chain, PPC::R12, Callee, InFlag);
    InFlag = Chain.getValue(1);

    NodeTys.clear();
    NodeTys.push_back(MVT::Other);
    NodeTys.push_back(MVT::Flag);
    Ops.clear();
    Ops.push_back(Chain);
    CallOpc = PPCISD::BCTRL;
    Callee.Val = 0;
  }

  // If this is a direct call, pass the chain and the callee.
  if (Callee.Val) {
    Ops.push_back(Chain);
    Ops.push_back(Callee);
  }
  
  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first, 
                                  RegsToPass[i].second.getValueType()));
  
  if (InFlag.Val)
    Ops.push_back(InFlag);
  Chain = DAG.getNode(CallOpc, NodeTys, Ops);
  InFlag = Chain.getValue(1);

  std::vector<SDOperand> ResultVals;
  NodeTys.clear();
  
  // If the call has results, copy the values out of the ret val registers.
  switch (Op.Val->getValueType(0)) {
  default: assert(0 && "Unexpected ret value!");
  case MVT::Other: break;
  case MVT::i32:
    if (Op.Val->getValueType(1) == MVT::i32) {
      Chain = DAG.getCopyFromReg(Chain, PPC::R4, MVT::i32, InFlag).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      Chain = DAG.getCopyFromReg(Chain, PPC::R3, MVT::i32,
                                 Chain.getValue(2)).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      NodeTys.push_back(MVT::i32);
    } else {
      Chain = DAG.getCopyFromReg(Chain, PPC::R3, MVT::i32, InFlag).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
    }
    NodeTys.push_back(MVT::i32);
    break;
  case MVT::i64:
    Chain = DAG.getCopyFromReg(Chain, PPC::X3, MVT::i64, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i64);
    break;
  case MVT::f32:
  case MVT::f64:
    Chain = DAG.getCopyFromReg(Chain, PPC::F1, Op.Val->getValueType(0),
                               InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(Op.Val->getValueType(0));
    break;
  case MVT::v4f32:
  case MVT::v4i32:
  case MVT::v8i16:
  case MVT::v16i8:
    Chain = DAG.getCopyFromReg(Chain, PPC::V2, Op.Val->getValueType(0),
                                   InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(Op.Val->getValueType(0));
    break;
  }
  
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, PtrVT));
  NodeTys.push_back(MVT::Other);
  
  // If the function returns void, just return the chain.
  if (ResultVals.empty())
    return Chain;
  
  // Otherwise, merge everything together with a MERGE_VALUES node.
  ResultVals.push_back(Chain);
  SDOperand Res = DAG.getNode(ISD::MERGE_VALUES, NodeTys, ResultVals);
  return Res.getValue(Op.ResNo);
}

static SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Copy;
  switch(Op.getNumOperands()) {
  default:
    assert(0 && "Do not know how to return this many arguments!");
    abort();
  case 1: 
    return SDOperand(); // ret void is legal
  case 3: {
    MVT::ValueType ArgVT = Op.getOperand(1).getValueType();
    unsigned ArgReg;
    if (ArgVT == MVT::i32) {
      ArgReg = PPC::R3;
    } else if (ArgVT == MVT::i64) {
      ArgReg = PPC::X3;
    } else if (MVT::isFloatingPoint(ArgVT)) {
      ArgReg = PPC::F1;
    } else {
      assert(MVT::isVector(ArgVT));
      ArgReg = PPC::V2;
    }
    
    Copy = DAG.getCopyToReg(Op.getOperand(0), ArgReg, Op.getOperand(1),
                            SDOperand());
    
    // If we haven't noted the R3/F1 are live out, do so now.
    if (DAG.getMachineFunction().liveout_empty())
      DAG.getMachineFunction().addLiveOut(ArgReg);
    break;
  }
  case 5:
    Copy = DAG.getCopyToReg(Op.getOperand(0), PPC::R3, Op.getOperand(3), 
                            SDOperand());
    Copy = DAG.getCopyToReg(Copy, PPC::R4, Op.getOperand(1),Copy.getValue(1));
    // If we haven't noted the R3+R4 are live out, do so now.
    if (DAG.getMachineFunction().liveout_empty()) {
      DAG.getMachineFunction().addLiveOut(PPC::R3);
      DAG.getMachineFunction().addLiveOut(PPC::R4);
    }
    break;
  }
  return DAG.getNode(PPCISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
}

/// LowerSELECT_CC - Lower floating point select_cc's into fsel instruction when
/// possible.
static SDOperand LowerSELECT_CC(SDOperand Op, SelectionDAG &DAG) {
  // Not FP? Not a fsel.
  if (!MVT::isFloatingPoint(Op.getOperand(0).getValueType()) ||
      !MVT::isFloatingPoint(Op.getOperand(2).getValueType()))
    return SDOperand();
  
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  
  // Cannot handle SETEQ/SETNE.
  if (CC == ISD::SETEQ || CC == ISD::SETNE) return SDOperand();
  
  MVT::ValueType ResVT = Op.getValueType();
  MVT::ValueType CmpVT = Op.getOperand(0).getValueType();
  SDOperand LHS = Op.getOperand(0), RHS = Op.getOperand(1);
  SDOperand TV  = Op.getOperand(2), FV  = Op.getOperand(3);
  
  // If the RHS of the comparison is a 0.0, we don't need to do the
  // subtraction at all.
  if (isFloatingPointZero(RHS))
    switch (CC) {
    default: break;       // SETUO etc aren't handled by fsel.
    case ISD::SETULT:
    case ISD::SETOLT:
    case ISD::SETLT:
      std::swap(TV, FV);  // fsel is natively setge, swap operands for setlt
    case ISD::SETUGE:
    case ISD::SETOGE:
    case ISD::SETGE:
      if (LHS.getValueType() == MVT::f32)   // Comparison is always 64-bits
        LHS = DAG.getNode(ISD::FP_EXTEND, MVT::f64, LHS);
      return DAG.getNode(PPCISD::FSEL, ResVT, LHS, TV, FV);
    case ISD::SETUGT:
    case ISD::SETOGT:
    case ISD::SETGT:
      std::swap(TV, FV);  // fsel is natively setge, swap operands for setlt
    case ISD::SETULE:
    case ISD::SETOLE:
    case ISD::SETLE:
      if (LHS.getValueType() == MVT::f32)   // Comparison is always 64-bits
        LHS = DAG.getNode(ISD::FP_EXTEND, MVT::f64, LHS);
      return DAG.getNode(PPCISD::FSEL, ResVT,
                         DAG.getNode(ISD::FNEG, MVT::f64, LHS), TV, FV);
    }
      
      SDOperand Cmp;
  switch (CC) {
  default: break;       // SETUO etc aren't handled by fsel.
  case ISD::SETULT:
  case ISD::SETOLT:
  case ISD::SETLT:
    Cmp = DAG.getNode(ISD::FSUB, CmpVT, LHS, RHS);
    if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
      Cmp = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, ResVT, Cmp, FV, TV);
  case ISD::SETUGE:
  case ISD::SETOGE:
  case ISD::SETGE:
    Cmp = DAG.getNode(ISD::FSUB, CmpVT, LHS, RHS);
    if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
      Cmp = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, ResVT, Cmp, TV, FV);
  case ISD::SETUGT:
  case ISD::SETOGT:
  case ISD::SETGT:
    Cmp = DAG.getNode(ISD::FSUB, CmpVT, RHS, LHS);
    if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
      Cmp = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, ResVT, Cmp, FV, TV);
  case ISD::SETULE:
  case ISD::SETOLE:
  case ISD::SETLE:
    Cmp = DAG.getNode(ISD::FSUB, CmpVT, RHS, LHS);
    if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
      Cmp = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, ResVT, Cmp, TV, FV);
  }
  return SDOperand();
}

static SDOperand LowerFP_TO_SINT(SDOperand Op, SelectionDAG &DAG) {
  assert(MVT::isFloatingPoint(Op.getOperand(0).getValueType()));
  SDOperand Src = Op.getOperand(0);
  if (Src.getValueType() == MVT::f32)
    Src = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Src);
  
  SDOperand Tmp;
  switch (Op.getValueType()) {
  default: assert(0 && "Unhandled FP_TO_SINT type in custom expander!");
  case MVT::i32:
    Tmp = DAG.getNode(PPCISD::FCTIWZ, MVT::f64, Src);
    break;
  case MVT::i64:
    Tmp = DAG.getNode(PPCISD::FCTIDZ, MVT::f64, Src);
    break;
  }
  
  // Convert the FP value to an int value through memory.
  SDOperand Bits = DAG.getNode(ISD::BIT_CONVERT, MVT::i64, Tmp);
  if (Op.getValueType() == MVT::i32)
    Bits = DAG.getNode(ISD::TRUNCATE, MVT::i32, Bits);
  return Bits;
}

static SDOperand LowerSINT_TO_FP(SDOperand Op, SelectionDAG &DAG) {
  if (Op.getOperand(0).getValueType() == MVT::i64) {
    SDOperand Bits = DAG.getNode(ISD::BIT_CONVERT, MVT::f64, Op.getOperand(0));
    SDOperand FP = DAG.getNode(PPCISD::FCFID, MVT::f64, Bits);
    if (Op.getValueType() == MVT::f32)
      FP = DAG.getNode(ISD::FP_ROUND, MVT::f32, FP);
    return FP;
  }
  
  assert(Op.getOperand(0).getValueType() == MVT::i32 &&
         "Unhandled SINT_TO_FP type in custom expander!");
  // Since we only generate this in 64-bit mode, we can take advantage of
  // 64-bit registers.  In particular, sign extend the input value into the
  // 64-bit register with extsw, store the WHOLE 64-bit value into the stack
  // then lfd it and fcfid it.
  MachineFrameInfo *FrameInfo = DAG.getMachineFunction().getFrameInfo();
  int FrameIdx = FrameInfo->CreateStackObject(8, 8);
  SDOperand FIdx = DAG.getFrameIndex(FrameIdx, MVT::i32);
  
  SDOperand Ext64 = DAG.getNode(PPCISD::EXTSW_32, MVT::i32,
                                Op.getOperand(0));
  
  // STD the extended value into the stack slot.
  SDOperand Store = DAG.getNode(PPCISD::STD_32, MVT::Other,
                                DAG.getEntryNode(), Ext64, FIdx,
                                DAG.getSrcValue(NULL));
  // Load the value as a double.
  SDOperand Ld = DAG.getLoad(MVT::f64, Store, FIdx, DAG.getSrcValue(NULL));
  
  // FCFID it and return it.
  SDOperand FP = DAG.getNode(PPCISD::FCFID, MVT::f64, Ld);
  if (Op.getValueType() == MVT::f32)
    FP = DAG.getNode(ISD::FP_ROUND, MVT::f32, FP);
  return FP;
}

static SDOperand LowerSHL(SDOperand Op, SelectionDAG &DAG,
                          MVT::ValueType PtrVT) {
  assert(Op.getValueType() == MVT::i64 &&
         Op.getOperand(1).getValueType() == MVT::i32 && "Unexpected SHL!");
  // The generic code does a fine job expanding shift by a constant.
  if (isa<ConstantSDNode>(Op.getOperand(1))) return SDOperand();
  
  // Otherwise, expand into a bunch of logical ops.  Note that these ops
  // depend on the PPC behavior for oversized shift amounts.
  SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                             DAG.getConstant(0, PtrVT));
  SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                             DAG.getConstant(1, PtrVT));
  SDOperand Amt = Op.getOperand(1);
  
  SDOperand Tmp1 = DAG.getNode(ISD::SUB, MVT::i32,
                               DAG.getConstant(32, MVT::i32), Amt);
  SDOperand Tmp2 = DAG.getNode(PPCISD::SHL, MVT::i32, Hi, Amt);
  SDOperand Tmp3 = DAG.getNode(PPCISD::SRL, MVT::i32, Lo, Tmp1);
  SDOperand Tmp4 = DAG.getNode(ISD::OR , MVT::i32, Tmp2, Tmp3);
  SDOperand Tmp5 = DAG.getNode(ISD::ADD, MVT::i32, Amt,
                               DAG.getConstant(-32U, MVT::i32));
  SDOperand Tmp6 = DAG.getNode(PPCISD::SHL, MVT::i32, Lo, Tmp5);
  SDOperand OutHi = DAG.getNode(ISD::OR, MVT::i32, Tmp4, Tmp6);
  SDOperand OutLo = DAG.getNode(PPCISD::SHL, MVT::i32, Lo, Amt);
  return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, OutLo, OutHi);
}

static SDOperand LowerSRL(SDOperand Op, SelectionDAG &DAG,
                          MVT::ValueType PtrVT) {
  assert(Op.getValueType() == MVT::i64 &&
         Op.getOperand(1).getValueType() == MVT::i32 && "Unexpected SHL!");
  // The generic code does a fine job expanding shift by a constant.
  if (isa<ConstantSDNode>(Op.getOperand(1))) return SDOperand();
  
  // Otherwise, expand into a bunch of logical ops.  Note that these ops
  // depend on the PPC behavior for oversized shift amounts.
  SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                             DAG.getConstant(0, PtrVT));
  SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                             DAG.getConstant(1, PtrVT));
  SDOperand Amt = Op.getOperand(1);
  
  SDOperand Tmp1 = DAG.getNode(ISD::SUB, MVT::i32,
                               DAG.getConstant(32, MVT::i32), Amt);
  SDOperand Tmp2 = DAG.getNode(PPCISD::SRL, MVT::i32, Lo, Amt);
  SDOperand Tmp3 = DAG.getNode(PPCISD::SHL, MVT::i32, Hi, Tmp1);
  SDOperand Tmp4 = DAG.getNode(ISD::OR , MVT::i32, Tmp2, Tmp3);
  SDOperand Tmp5 = DAG.getNode(ISD::ADD, MVT::i32, Amt,
                               DAG.getConstant(-32U, MVT::i32));
  SDOperand Tmp6 = DAG.getNode(PPCISD::SRL, MVT::i32, Hi, Tmp5);
  SDOperand OutLo = DAG.getNode(ISD::OR, MVT::i32, Tmp4, Tmp6);
  SDOperand OutHi = DAG.getNode(PPCISD::SRL, MVT::i32, Hi, Amt);
  return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, OutLo, OutHi);
}

static SDOperand LowerSRA(SDOperand Op, SelectionDAG &DAG,
                          MVT::ValueType PtrVT) {
  assert(Op.getValueType() == MVT::i64 &&
         Op.getOperand(1).getValueType() == MVT::i32 && "Unexpected SRA!");
  // The generic code does a fine job expanding shift by a constant.
  if (isa<ConstantSDNode>(Op.getOperand(1))) return SDOperand();
  
  // Otherwise, expand into a bunch of logical ops, followed by a select_cc.
  SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                             DAG.getConstant(0, PtrVT));
  SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                             DAG.getConstant(1, PtrVT));
  SDOperand Amt = Op.getOperand(1);
  
  SDOperand Tmp1 = DAG.getNode(ISD::SUB, MVT::i32,
                               DAG.getConstant(32, MVT::i32), Amt);
  SDOperand Tmp2 = DAG.getNode(PPCISD::SRL, MVT::i32, Lo, Amt);
  SDOperand Tmp3 = DAG.getNode(PPCISD::SHL, MVT::i32, Hi, Tmp1);
  SDOperand Tmp4 = DAG.getNode(ISD::OR , MVT::i32, Tmp2, Tmp3);
  SDOperand Tmp5 = DAG.getNode(ISD::ADD, MVT::i32, Amt,
                               DAG.getConstant(-32U, MVT::i32));
  SDOperand Tmp6 = DAG.getNode(PPCISD::SRA, MVT::i32, Hi, Tmp5);
  SDOperand OutHi = DAG.getNode(PPCISD::SRA, MVT::i32, Hi, Amt);
  SDOperand OutLo = DAG.getSelectCC(Tmp5, DAG.getConstant(0, MVT::i32),
                                    Tmp4, Tmp6, ISD::SETLE);
  return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, OutLo, OutHi);
}

//===----------------------------------------------------------------------===//
// Vector related lowering.
//

// If this is a vector of constants or undefs, get the bits.  A bit in
// UndefBits is set if the corresponding element of the vector is an 
// ISD::UNDEF value.  For undefs, the corresponding VectorBits values are
// zero.   Return true if this is not an array of constants, false if it is.
//
static bool GetConstantBuildVectorBits(SDNode *BV, uint64_t VectorBits[2],
                                       uint64_t UndefBits[2]) {
  // Start with zero'd results.
  VectorBits[0] = VectorBits[1] = UndefBits[0] = UndefBits[1] = 0;
  
  unsigned EltBitSize = MVT::getSizeInBits(BV->getOperand(0).getValueType());
  for (unsigned i = 0, e = BV->getNumOperands(); i != e; ++i) {
    SDOperand OpVal = BV->getOperand(i);
    
    unsigned PartNo = i >= e/2;     // In the upper 128 bits?
    unsigned SlotNo = e/2 - (i & (e/2-1))-1;  // Which subpiece of the uint64_t.

    uint64_t EltBits = 0;
    if (OpVal.getOpcode() == ISD::UNDEF) {
      uint64_t EltUndefBits = ~0U >> (32-EltBitSize);
      UndefBits[PartNo] |= EltUndefBits << (SlotNo*EltBitSize);
      continue;
    } else if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(OpVal)) {
      EltBits = CN->getValue() & (~0U >> (32-EltBitSize));
    } else if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(OpVal)) {
      assert(CN->getValueType(0) == MVT::f32 &&
             "Only one legal FP vector type!");
      EltBits = FloatToBits(CN->getValue());
    } else {
      // Nonconstant element.
      return true;
    }
    
    VectorBits[PartNo] |= EltBits << (SlotNo*EltBitSize);
  }
  
  //printf("%llx %llx  %llx %llx\n", 
  //       VectorBits[0], VectorBits[1], UndefBits[0], UndefBits[1]);
  return false;
}

// If this is a splat (repetition) of a value across the whole vector, return
// the smallest size that splats it.  For example, "0x01010101010101..." is a
// splat of 0x01, 0x0101, and 0x01010101.  We return SplatBits = 0x01 and 
// SplatSize = 1 byte.
static bool isConstantSplat(const uint64_t Bits128[2], 
                            const uint64_t Undef128[2],
                            unsigned &SplatBits, unsigned &SplatUndef,
                            unsigned &SplatSize) {
  
  // Don't let undefs prevent splats from matching.  See if the top 64-bits are
  // the same as the lower 64-bits, ignoring undefs.
  if ((Bits128[0] & ~Undef128[1]) != (Bits128[1] & ~Undef128[0]))
    return false;  // Can't be a splat if two pieces don't match.
  
  uint64_t Bits64  = Bits128[0] | Bits128[1];
  uint64_t Undef64 = Undef128[0] & Undef128[1];
  
  // Check that the top 32-bits are the same as the lower 32-bits, ignoring
  // undefs.
  if ((Bits64 & (~Undef64 >> 32)) != ((Bits64 >> 32) & ~Undef64))
    return false;  // Can't be a splat if two pieces don't match.

  uint32_t Bits32  = uint32_t(Bits64) | uint32_t(Bits64 >> 32);
  uint32_t Undef32 = uint32_t(Undef64) & uint32_t(Undef64 >> 32);

  // If the top 16-bits are different than the lower 16-bits, ignoring
  // undefs, we have an i32 splat.
  if ((Bits32 & (~Undef32 >> 16)) != ((Bits32 >> 16) & ~Undef32)) {
    SplatBits = Bits32;
    SplatUndef = Undef32;
    SplatSize = 4;
    return true;
  }
  
  uint16_t Bits16  = uint16_t(Bits32)  | uint16_t(Bits32 >> 16);
  uint16_t Undef16 = uint16_t(Undef32) & uint16_t(Undef32 >> 16);

  // If the top 8-bits are different than the lower 8-bits, ignoring
  // undefs, we have an i16 splat.
  if ((Bits16 & (uint16_t(~Undef16) >> 8)) != ((Bits16 >> 8) & ~Undef16)) {
    SplatBits = Bits16;
    SplatUndef = Undef16;
    SplatSize = 2;
    return true;
  }
  
  // Otherwise, we have an 8-bit splat.
  SplatBits  = uint8_t(Bits16)  | uint8_t(Bits16 >> 8);
  SplatUndef = uint8_t(Undef16) & uint8_t(Undef16 >> 8);
  SplatSize = 1;
  return true;
}

/// BuildSplatI - Build a canonical splati of Val with an element size of
/// SplatSize.  Cast the result to VT.
static SDOperand BuildSplatI(int Val, unsigned SplatSize, MVT::ValueType VT,
                             SelectionDAG &DAG) {
  assert(Val >= -16 && Val <= 15 && "vsplti is out of range!");
  
  // Force vspltis[hw] -1 to vspltisb -1.
  if (Val == -1) SplatSize = 1;
  
  static const MVT::ValueType VTys[] = { // canonical VT to use for each size.
    MVT::v16i8, MVT::v8i16, MVT::Other, MVT::v4i32
  };
  MVT::ValueType CanonicalVT = VTys[SplatSize-1];
  
  // Build a canonical splat for this value.
  SDOperand Elt = DAG.getConstant(Val, MVT::getVectorBaseType(CanonicalVT));
  std::vector<SDOperand> Ops(MVT::getVectorNumElements(CanonicalVT), Elt);
  SDOperand Res = DAG.getNode(ISD::BUILD_VECTOR, CanonicalVT, Ops);
  return DAG.getNode(ISD::BIT_CONVERT, VT, Res);
}

/// BuildIntrinsicOp - Return a binary operator intrinsic node with the
/// specified intrinsic ID.
static SDOperand BuildIntrinsicOp(unsigned IID, SDOperand LHS, SDOperand RHS,
                                  SelectionDAG &DAG, 
                                  MVT::ValueType DestVT = MVT::Other) {
  if (DestVT == MVT::Other) DestVT = LHS.getValueType();
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DestVT,
                     DAG.getConstant(IID, MVT::i32), LHS, RHS);
}

/// BuildIntrinsicOp - Return a ternary operator intrinsic node with the
/// specified intrinsic ID.
static SDOperand BuildIntrinsicOp(unsigned IID, SDOperand Op0, SDOperand Op1,
                                  SDOperand Op2, SelectionDAG &DAG, 
                                  MVT::ValueType DestVT = MVT::Other) {
  if (DestVT == MVT::Other) DestVT = Op0.getValueType();
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DestVT,
                     DAG.getConstant(IID, MVT::i32), Op0, Op1, Op2);
}


/// BuildVSLDOI - Return a VECTOR_SHUFFLE that is a vsldoi of the specified
/// amount.  The result has the specified value type.
static SDOperand BuildVSLDOI(SDOperand LHS, SDOperand RHS, unsigned Amt,
                             MVT::ValueType VT, SelectionDAG &DAG) {
  // Force LHS/RHS to be the right type.
  LHS = DAG.getNode(ISD::BIT_CONVERT, MVT::v16i8, LHS);
  RHS = DAG.getNode(ISD::BIT_CONVERT, MVT::v16i8, RHS);
  
  std::vector<SDOperand> Ops;
  for (unsigned i = 0; i != 16; ++i)
    Ops.push_back(DAG.getConstant(i+Amt, MVT::i32));
  SDOperand T = DAG.getNode(ISD::VECTOR_SHUFFLE, MVT::v16i8, LHS, RHS,
                            DAG.getNode(ISD::BUILD_VECTOR, MVT::v16i8, Ops));
  return DAG.getNode(ISD::BIT_CONVERT, VT, T);
}

// If this is a case we can't handle, return null and let the default
// expansion code take care of it.  If we CAN select this case, and if it
// selects to a single instruction, return Op.  Otherwise, if we can codegen
// this case more efficiently than a constant pool load, lower it to the
// sequence of ops that should be used.
static SDOperand LowerBUILD_VECTOR(SDOperand Op, SelectionDAG &DAG) {
  // If this is a vector of constants or undefs, get the bits.  A bit in
  // UndefBits is set if the corresponding element of the vector is an 
  // ISD::UNDEF value.  For undefs, the corresponding VectorBits values are
  // zero. 
  uint64_t VectorBits[2];
  uint64_t UndefBits[2];
  if (GetConstantBuildVectorBits(Op.Val, VectorBits, UndefBits))
    return SDOperand();   // Not a constant vector.
  
  // If this is a splat (repetition) of a value across the whole vector, return
  // the smallest size that splats it.  For example, "0x01010101010101..." is a
  // splat of 0x01, 0x0101, and 0x01010101.  We return SplatBits = 0x01 and 
  // SplatSize = 1 byte.
  unsigned SplatBits, SplatUndef, SplatSize;
  if (isConstantSplat(VectorBits, UndefBits, SplatBits, SplatUndef, SplatSize)){
    bool HasAnyUndefs = (UndefBits[0] | UndefBits[1]) != 0;
    
    // First, handle single instruction cases.
    
    // All zeros?
    if (SplatBits == 0) {
      // Canonicalize all zero vectors to be v4i32.
      if (Op.getValueType() != MVT::v4i32 || HasAnyUndefs) {
        SDOperand Z = DAG.getConstant(0, MVT::i32);
        Z = DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32, Z, Z, Z, Z);
        Op = DAG.getNode(ISD::BIT_CONVERT, Op.getValueType(), Z);
      }
      return Op;
    }

    // If the sign extended value is in the range [-16,15], use VSPLTI[bhw].
    int32_t SextVal= int32_t(SplatBits << (32-8*SplatSize)) >> (32-8*SplatSize);
    if (SextVal >= -16 && SextVal <= 15)
      return BuildSplatI(SextVal, SplatSize, Op.getValueType(), DAG);
    
    
    // Two instruction sequences.
    
    // If this value is in the range [-32,30] and is even, use:
    //    tmp = VSPLTI[bhw], result = add tmp, tmp
    if (SextVal >= -32 && SextVal <= 30 && (SextVal & 1) == 0) {
      Op = BuildSplatI(SextVal >> 1, SplatSize, Op.getValueType(), DAG);
      return DAG.getNode(ISD::ADD, Op.getValueType(), Op, Op);
    }
    
    // If this is 0x8000_0000 x 4, turn into vspltisw + vslw.  If it is 
    // 0x7FFF_FFFF x 4, turn it into not(0x8000_0000).  This is important
    // for fneg/fabs.
    if (SplatSize == 4 && SplatBits == (0x7FFFFFFF&~SplatUndef)) {
      // Make -1 and vspltisw -1:
      SDOperand OnesV = BuildSplatI(-1, 4, MVT::v4i32, DAG);
      
      // Make the VSLW intrinsic, computing 0x8000_0000.
      SDOperand Res = BuildIntrinsicOp(Intrinsic::ppc_altivec_vslw, OnesV, 
                                       OnesV, DAG);
      
      // xor by OnesV to invert it.
      Res = DAG.getNode(ISD::XOR, MVT::v4i32, Res, OnesV);
      return DAG.getNode(ISD::BIT_CONVERT, Op.getValueType(), Res);
    }

    // Check to see if this is a wide variety of vsplti*, binop self cases.
    unsigned SplatBitSize = SplatSize*8;
    static const char SplatCsts[] = {
      -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7,
      -8, 8, -9, 9, -10, 10, -11, 11, -12, 12, -13, 13, 14, -14, 15, -15, -16
    };
    for (unsigned idx = 0; idx < sizeof(SplatCsts)/sizeof(SplatCsts[0]); ++idx){
      // Indirect through the SplatCsts array so that we favor 'vsplti -1' for
      // cases which are ambiguous (e.g. formation of 0x8000_0000).  'vsplti -1'
      int i = SplatCsts[idx];
      
      // Figure out what shift amount will be used by altivec if shifted by i in
      // this splat size.
      unsigned TypeShiftAmt = i & (SplatBitSize-1);
      
      // vsplti + shl self.
      if (SextVal == (i << (int)TypeShiftAmt)) {
        Op = BuildSplatI(i, SplatSize, Op.getValueType(), DAG);
        static const unsigned IIDs[] = { // Intrinsic to use for each size.
          Intrinsic::ppc_altivec_vslb, Intrinsic::ppc_altivec_vslh, 0,
          Intrinsic::ppc_altivec_vslw
        };
        return BuildIntrinsicOp(IIDs[SplatSize-1], Op, Op, DAG);
      }
      
      // vsplti + srl self.
      if (SextVal == (int)((unsigned)i >> TypeShiftAmt)) {
        Op = BuildSplatI(i, SplatSize, Op.getValueType(), DAG);
        static const unsigned IIDs[] = { // Intrinsic to use for each size.
          Intrinsic::ppc_altivec_vsrb, Intrinsic::ppc_altivec_vsrh, 0,
          Intrinsic::ppc_altivec_vsrw
        };
        return BuildIntrinsicOp(IIDs[SplatSize-1], Op, Op, DAG);
      }
      
      // vsplti + sra self.
      if (SextVal == (int)((unsigned)i >> TypeShiftAmt)) {
        Op = BuildSplatI(i, SplatSize, Op.getValueType(), DAG);
        static const unsigned IIDs[] = { // Intrinsic to use for each size.
          Intrinsic::ppc_altivec_vsrab, Intrinsic::ppc_altivec_vsrah, 0,
          Intrinsic::ppc_altivec_vsraw
        };
        return BuildIntrinsicOp(IIDs[SplatSize-1], Op, Op, DAG);
      }
      
      // vsplti + rol self.
      if (SextVal == (int)(((unsigned)i << TypeShiftAmt) |
                           ((unsigned)i >> (SplatBitSize-TypeShiftAmt)))) {
        Op = BuildSplatI(i, SplatSize, Op.getValueType(), DAG);
        static const unsigned IIDs[] = { // Intrinsic to use for each size.
          Intrinsic::ppc_altivec_vrlb, Intrinsic::ppc_altivec_vrlh, 0,
          Intrinsic::ppc_altivec_vrlw
        };
        return BuildIntrinsicOp(IIDs[SplatSize-1], Op, Op, DAG);
      }

      // t = vsplti c, result = vsldoi t, t, 1
      if (SextVal == ((i << 8) | (i >> (TypeShiftAmt-8)))) {
        SDOperand T = BuildSplatI(i, SplatSize, MVT::v16i8, DAG);
        return BuildVSLDOI(T, T, 1, Op.getValueType(), DAG);
      }
      // t = vsplti c, result = vsldoi t, t, 2
      if (SextVal == ((i << 16) | (i >> (TypeShiftAmt-16)))) {
        SDOperand T = BuildSplatI(i, SplatSize, MVT::v16i8, DAG);
        return BuildVSLDOI(T, T, 2, Op.getValueType(), DAG);
      }
      // t = vsplti c, result = vsldoi t, t, 3
      if (SextVal == ((i << 24) | (i >> (TypeShiftAmt-24)))) {
        SDOperand T = BuildSplatI(i, SplatSize, MVT::v16i8, DAG);
        return BuildVSLDOI(T, T, 3, Op.getValueType(), DAG);
      }
    }
    
    // Three instruction sequences.
    
    // Odd, in range [17,31]:  (vsplti C)-(vsplti -16).
    if (SextVal >= 0 && SextVal <= 31) {
      SDOperand LHS = BuildSplatI(SextVal-16, SplatSize, Op.getValueType(),DAG);
      SDOperand RHS = BuildSplatI(-16, SplatSize, Op.getValueType(), DAG);
      return DAG.getNode(ISD::SUB, Op.getValueType(), LHS, RHS);
    }
    // Odd, in range [-31,-17]:  (vsplti C)+(vsplti -16).
    if (SextVal >= -31 && SextVal <= 0) {
      SDOperand LHS = BuildSplatI(SextVal+16, SplatSize, Op.getValueType(),DAG);
      SDOperand RHS = BuildSplatI(-16, SplatSize, Op.getValueType(), DAG);
      return DAG.getNode(ISD::ADD, Op.getValueType(), LHS, RHS);
    }
  }
    
  return SDOperand();
}

/// GeneratePerfectShuffle - Given an entry in the perfect-shuffle table, emit
/// the specified operations to build the shuffle.
static SDOperand GeneratePerfectShuffle(unsigned PFEntry, SDOperand LHS,
                                        SDOperand RHS, SelectionDAG &DAG) {
  unsigned OpNum = (PFEntry >> 26) & 0x0F;
  unsigned LHSID  = (PFEntry >> 13) & ((1 << 13)-1);
  unsigned RHSID = (PFEntry >>  0) & ((1 << 13)-1);
  
  enum {
    OP_COPY = 0,  // Copy, used for things like <u,u,u,3> to say it is <0,1,2,3>
    OP_VMRGHW,
    OP_VMRGLW,
    OP_VSPLTISW0,
    OP_VSPLTISW1,
    OP_VSPLTISW2,
    OP_VSPLTISW3,
    OP_VSLDOI4,
    OP_VSLDOI8,
    OP_VSLDOI12
  };
  
  if (OpNum == OP_COPY) {
    if (LHSID == (1*9+2)*9+3) return LHS;
    assert(LHSID == ((4*9+5)*9+6)*9+7 && "Illegal OP_COPY!");
    return RHS;
  }
  
  SDOperand OpLHS, OpRHS;
  OpLHS = GeneratePerfectShuffle(PerfectShuffleTable[LHSID], LHS, RHS, DAG);
  OpRHS = GeneratePerfectShuffle(PerfectShuffleTable[RHSID], LHS, RHS, DAG);
  
  unsigned ShufIdxs[16];
  switch (OpNum) {
  default: assert(0 && "Unknown i32 permute!");
  case OP_VMRGHW:
    ShufIdxs[ 0] =  0; ShufIdxs[ 1] =  1; ShufIdxs[ 2] =  2; ShufIdxs[ 3] =  3;
    ShufIdxs[ 4] = 16; ShufIdxs[ 5] = 17; ShufIdxs[ 6] = 18; ShufIdxs[ 7] = 19;
    ShufIdxs[ 8] =  4; ShufIdxs[ 9] =  5; ShufIdxs[10] =  6; ShufIdxs[11] =  7;
    ShufIdxs[12] = 20; ShufIdxs[13] = 21; ShufIdxs[14] = 22; ShufIdxs[15] = 23;
    break;
  case OP_VMRGLW:
    ShufIdxs[ 0] =  8; ShufIdxs[ 1] =  9; ShufIdxs[ 2] = 10; ShufIdxs[ 3] = 11;
    ShufIdxs[ 4] = 24; ShufIdxs[ 5] = 25; ShufIdxs[ 6] = 26; ShufIdxs[ 7] = 27;
    ShufIdxs[ 8] = 12; ShufIdxs[ 9] = 13; ShufIdxs[10] = 14; ShufIdxs[11] = 15;
    ShufIdxs[12] = 28; ShufIdxs[13] = 29; ShufIdxs[14] = 30; ShufIdxs[15] = 31;
    break;
  case OP_VSPLTISW0:
    for (unsigned i = 0; i != 16; ++i)
      ShufIdxs[i] = (i&3)+0;
    break;
  case OP_VSPLTISW1:
    for (unsigned i = 0; i != 16; ++i)
      ShufIdxs[i] = (i&3)+4;
    break;
  case OP_VSPLTISW2:
    for (unsigned i = 0; i != 16; ++i)
      ShufIdxs[i] = (i&3)+8;
    break;
  case OP_VSPLTISW3:
    for (unsigned i = 0; i != 16; ++i)
      ShufIdxs[i] = (i&3)+12;
    break;
  case OP_VSLDOI4:
    return BuildVSLDOI(OpLHS, OpRHS, 4, OpLHS.getValueType(), DAG);
  case OP_VSLDOI8:
    return BuildVSLDOI(OpLHS, OpRHS, 8, OpLHS.getValueType(), DAG);
  case OP_VSLDOI12:
    return BuildVSLDOI(OpLHS, OpRHS, 12, OpLHS.getValueType(), DAG);
  }
  std::vector<SDOperand> Ops;
  for (unsigned i = 0; i != 16; ++i)
    Ops.push_back(DAG.getConstant(ShufIdxs[i], MVT::i32));
  
  return DAG.getNode(ISD::VECTOR_SHUFFLE, OpLHS.getValueType(), OpLHS, OpRHS,
                     DAG.getNode(ISD::BUILD_VECTOR, MVT::v16i8, Ops));
}

/// LowerVECTOR_SHUFFLE - Return the code we lower for VECTOR_SHUFFLE.  If this
/// is a shuffle we can handle in a single instruction, return it.  Otherwise,
/// return the code it can be lowered into.  Worst case, it can always be
/// lowered into a vperm.
static SDOperand LowerVECTOR_SHUFFLE(SDOperand Op, SelectionDAG &DAG) {
  SDOperand V1 = Op.getOperand(0);
  SDOperand V2 = Op.getOperand(1);
  SDOperand PermMask = Op.getOperand(2);
  
  // Cases that are handled by instructions that take permute immediates
  // (such as vsplt*) should be left as VECTOR_SHUFFLE nodes so they can be
  // selected by the instruction selector.
  if (V2.getOpcode() == ISD::UNDEF) {
    if (PPC::isSplatShuffleMask(PermMask.Val, 1) ||
        PPC::isSplatShuffleMask(PermMask.Val, 2) ||
        PPC::isSplatShuffleMask(PermMask.Val, 4) ||
        PPC::isVPKUWUMShuffleMask(PermMask.Val, true) ||
        PPC::isVPKUHUMShuffleMask(PermMask.Val, true) ||
        PPC::isVSLDOIShuffleMask(PermMask.Val, true) != -1 ||
        PPC::isVMRGLShuffleMask(PermMask.Val, 1, true) ||
        PPC::isVMRGLShuffleMask(PermMask.Val, 2, true) ||
        PPC::isVMRGLShuffleMask(PermMask.Val, 4, true) ||
        PPC::isVMRGHShuffleMask(PermMask.Val, 1, true) ||
        PPC::isVMRGHShuffleMask(PermMask.Val, 2, true) ||
        PPC::isVMRGHShuffleMask(PermMask.Val, 4, true)) {
      return Op;
    }
  }
  
  // Altivec has a variety of "shuffle immediates" that take two vector inputs
  // and produce a fixed permutation.  If any of these match, do not lower to
  // VPERM.
  if (PPC::isVPKUWUMShuffleMask(PermMask.Val, false) ||
      PPC::isVPKUHUMShuffleMask(PermMask.Val, false) ||
      PPC::isVSLDOIShuffleMask(PermMask.Val, false) != -1 ||
      PPC::isVMRGLShuffleMask(PermMask.Val, 1, false) ||
      PPC::isVMRGLShuffleMask(PermMask.Val, 2, false) ||
      PPC::isVMRGLShuffleMask(PermMask.Val, 4, false) ||
      PPC::isVMRGHShuffleMask(PermMask.Val, 1, false) ||
      PPC::isVMRGHShuffleMask(PermMask.Val, 2, false) ||
      PPC::isVMRGHShuffleMask(PermMask.Val, 4, false))
    return Op;
  
  // Check to see if this is a shuffle of 4-byte values.  If so, we can use our
  // perfect shuffle table to emit an optimal matching sequence.
  unsigned PFIndexes[4];
  bool isFourElementShuffle = true;
  for (unsigned i = 0; i != 4 && isFourElementShuffle; ++i) { // Element number
    unsigned EltNo = 8;   // Start out undef.
    for (unsigned j = 0; j != 4; ++j) {  // Intra-element byte.
      if (PermMask.getOperand(i*4+j).getOpcode() == ISD::UNDEF)
        continue;   // Undef, ignore it.
      
      unsigned ByteSource = 
        cast<ConstantSDNode>(PermMask.getOperand(i*4+j))->getValue();
      if ((ByteSource & 3) != j) {
        isFourElementShuffle = false;
        break;
      }
      
      if (EltNo == 8) {
        EltNo = ByteSource/4;
      } else if (EltNo != ByteSource/4) {
        isFourElementShuffle = false;
        break;
      }
    }
    PFIndexes[i] = EltNo;
  }
    
  // If this shuffle can be expressed as a shuffle of 4-byte elements, use the 
  // perfect shuffle vector to determine if it is cost effective to do this as
  // discrete instructions, or whether we should use a vperm.
  if (isFourElementShuffle) {
    // Compute the index in the perfect shuffle table.
    unsigned PFTableIndex = 
      PFIndexes[0]*9*9*9+PFIndexes[1]*9*9+PFIndexes[2]*9+PFIndexes[3];
    
    unsigned PFEntry = PerfectShuffleTable[PFTableIndex];
    unsigned Cost  = (PFEntry >> 30);
    
    // Determining when to avoid vperm is tricky.  Many things affect the cost
    // of vperm, particularly how many times the perm mask needs to be computed.
    // For example, if the perm mask can be hoisted out of a loop or is already
    // used (perhaps because there are multiple permutes with the same shuffle
    // mask?) the vperm has a cost of 1.  OTOH, hoisting the permute mask out of
    // the loop requires an extra register.
    //
    // As a compromise, we only emit discrete instructions if the shuffle can be
    // generated in 3 or fewer operations.  When we have loop information 
    // available, if this block is within a loop, we should avoid using vperm
    // for 3-operation perms and use a constant pool load instead.
    if (Cost < 3) 
      return GeneratePerfectShuffle(PFEntry, V1, V2, DAG);
  }
  
  // Lower this to a VPERM(V1, V2, V3) expression, where V3 is a constant
  // vector that will get spilled to the constant pool.
  if (V2.getOpcode() == ISD::UNDEF) V2 = V1;
  
  // The SHUFFLE_VECTOR mask is almost exactly what we want for vperm, except
  // that it is in input element units, not in bytes.  Convert now.
  MVT::ValueType EltVT = MVT::getVectorBaseType(V1.getValueType());
  unsigned BytesPerElement = MVT::getSizeInBits(EltVT)/8;
  
  std::vector<SDOperand> ResultMask;
  for (unsigned i = 0, e = PermMask.getNumOperands(); i != e; ++i) {
    unsigned SrcElt;
    if (PermMask.getOperand(i).getOpcode() == ISD::UNDEF)
      SrcElt = 0;
    else 
      SrcElt = cast<ConstantSDNode>(PermMask.getOperand(i))->getValue();
    
    for (unsigned j = 0; j != BytesPerElement; ++j)
      ResultMask.push_back(DAG.getConstant(SrcElt*BytesPerElement+j,
                                           MVT::i8));
  }
  
  SDOperand VPermMask = DAG.getNode(ISD::BUILD_VECTOR, MVT::v16i8, ResultMask);
  return DAG.getNode(PPCISD::VPERM, V1.getValueType(), V1, V2, VPermMask);
}

/// getAltivecCompareInfo - Given an intrinsic, return false if it is not an
/// altivec comparison.  If it is, return true and fill in Opc/isDot with
/// information about the intrinsic.
static bool getAltivecCompareInfo(SDOperand Intrin, int &CompareOpc,
                                  bool &isDot) {
  unsigned IntrinsicID = cast<ConstantSDNode>(Intrin.getOperand(0))->getValue();
  CompareOpc = -1;
  isDot = false;
  switch (IntrinsicID) {
  default: return false;
    // Comparison predicates.
  case Intrinsic::ppc_altivec_vcmpbfp_p:  CompareOpc = 966; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpeqfp_p: CompareOpc = 198; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpequb_p: CompareOpc =   6; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpequh_p: CompareOpc =  70; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpequw_p: CompareOpc = 134; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgefp_p: CompareOpc = 454; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtfp_p: CompareOpc = 710; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtsb_p: CompareOpc = 774; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtsh_p: CompareOpc = 838; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtsw_p: CompareOpc = 902; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtub_p: CompareOpc = 518; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtuh_p: CompareOpc = 582; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtuw_p: CompareOpc = 646; isDot = 1; break;
    
    // Normal Comparisons.
  case Intrinsic::ppc_altivec_vcmpbfp:    CompareOpc = 966; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpeqfp:   CompareOpc = 198; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpequb:   CompareOpc =   6; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpequh:   CompareOpc =  70; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpequw:   CompareOpc = 134; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgefp:   CompareOpc = 454; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtfp:   CompareOpc = 710; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtsb:   CompareOpc = 774; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtsh:   CompareOpc = 838; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtsw:   CompareOpc = 902; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtub:   CompareOpc = 518; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtuh:   CompareOpc = 582; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtuw:   CompareOpc = 646; isDot = 0; break;
  }
  return true;
}

/// LowerINTRINSIC_WO_CHAIN - If this is an intrinsic that we want to custom
/// lower, do it, otherwise return null.
static SDOperand LowerINTRINSIC_WO_CHAIN(SDOperand Op, SelectionDAG &DAG) {
  // If this is a lowered altivec predicate compare, CompareOpc is set to the
  // opcode number of the comparison.
  int CompareOpc;
  bool isDot;
  if (!getAltivecCompareInfo(Op, CompareOpc, isDot))
    return SDOperand();    // Don't custom lower most intrinsics.
  
  // If this is a non-dot comparison, make the VCMP node and we are done.
  if (!isDot) {
    SDOperand Tmp = DAG.getNode(PPCISD::VCMP, Op.getOperand(2).getValueType(),
                                Op.getOperand(1), Op.getOperand(2),
                                DAG.getConstant(CompareOpc, MVT::i32));
    return DAG.getNode(ISD::BIT_CONVERT, Op.getValueType(), Tmp);
  }
  
  // Create the PPCISD altivec 'dot' comparison node.
  std::vector<SDOperand> Ops;
  std::vector<MVT::ValueType> VTs;
  Ops.push_back(Op.getOperand(2));  // LHS
  Ops.push_back(Op.getOperand(3));  // RHS
  Ops.push_back(DAG.getConstant(CompareOpc, MVT::i32));
  VTs.push_back(Op.getOperand(2).getValueType());
  VTs.push_back(MVT::Flag);
  SDOperand CompNode = DAG.getNode(PPCISD::VCMPo, VTs, Ops);
  
  // Now that we have the comparison, emit a copy from the CR to a GPR.
  // This is flagged to the above dot comparison.
  SDOperand Flags = DAG.getNode(PPCISD::MFCR, MVT::i32,
                                DAG.getRegister(PPC::CR6, MVT::i32),
                                CompNode.getValue(1)); 
  
  // Unpack the result based on how the target uses it.
  unsigned BitNo;   // Bit # of CR6.
  bool InvertBit;   // Invert result?
  switch (cast<ConstantSDNode>(Op.getOperand(1))->getValue()) {
  default:  // Can't happen, don't crash on invalid number though.
  case 0:   // Return the value of the EQ bit of CR6.
    BitNo = 0; InvertBit = false;
    break;
  case 1:   // Return the inverted value of the EQ bit of CR6.
    BitNo = 0; InvertBit = true;
    break;
  case 2:   // Return the value of the LT bit of CR6.
    BitNo = 2; InvertBit = false;
    break;
  case 3:   // Return the inverted value of the LT bit of CR6.
    BitNo = 2; InvertBit = true;
    break;
  }
  
  // Shift the bit into the low position.
  Flags = DAG.getNode(ISD::SRL, MVT::i32, Flags,
                      DAG.getConstant(8-(3-BitNo), MVT::i32));
  // Isolate the bit.
  Flags = DAG.getNode(ISD::AND, MVT::i32, Flags,
                      DAG.getConstant(1, MVT::i32));
  
  // If we are supposed to, toggle the bit.
  if (InvertBit)
    Flags = DAG.getNode(ISD::XOR, MVT::i32, Flags,
                        DAG.getConstant(1, MVT::i32));
  return Flags;
}

static SDOperand LowerSCALAR_TO_VECTOR(SDOperand Op, SelectionDAG &DAG) {
  // Create a stack slot that is 16-byte aligned.
  MachineFrameInfo *FrameInfo = DAG.getMachineFunction().getFrameInfo();
  int FrameIdx = FrameInfo->CreateStackObject(16, 16);
  SDOperand FIdx = DAG.getFrameIndex(FrameIdx, MVT::i32);
  
  // Store the input value into Value#0 of the stack slot.
  SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, DAG.getEntryNode(),
                                Op.getOperand(0), FIdx,DAG.getSrcValue(NULL));
  // Load it out.
  return DAG.getLoad(Op.getValueType(), Store, FIdx, DAG.getSrcValue(NULL));
}

static SDOperand LowerMUL(SDOperand Op, SelectionDAG &DAG) {
  if (Op.getValueType() == MVT::v4i32) {
    SDOperand LHS = Op.getOperand(0), RHS = Op.getOperand(1);
    
    SDOperand Zero  = BuildSplatI(  0, 1, MVT::v4i32, DAG);
    SDOperand Neg16 = BuildSplatI(-16, 4, MVT::v4i32, DAG); // +16 as shift amt.
    
    SDOperand RHSSwap =   // = vrlw RHS, 16
      BuildIntrinsicOp(Intrinsic::ppc_altivec_vrlw, RHS, Neg16, DAG);
    
    // Shrinkify inputs to v8i16.
    LHS = DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16, LHS);
    RHS = DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16, RHS);
    RHSSwap = DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16, RHSSwap);
    
    // Low parts multiplied together, generating 32-bit results (we ignore the
    // top parts).
    SDOperand LoProd = BuildIntrinsicOp(Intrinsic::ppc_altivec_vmulouh,
                                        LHS, RHS, DAG, MVT::v4i32);
    
    SDOperand HiProd = BuildIntrinsicOp(Intrinsic::ppc_altivec_vmsumuhm,
                                        LHS, RHSSwap, Zero, DAG, MVT::v4i32);
    // Shift the high parts up 16 bits.
    HiProd = BuildIntrinsicOp(Intrinsic::ppc_altivec_vslw, HiProd, Neg16, DAG);
    return DAG.getNode(ISD::ADD, MVT::v4i32, LoProd, HiProd);
  } else if (Op.getValueType() == MVT::v8i16) {
    SDOperand LHS = Op.getOperand(0), RHS = Op.getOperand(1);
    
    SDOperand Zero = BuildSplatI(0, 1, MVT::v8i16, DAG);

    return BuildIntrinsicOp(Intrinsic::ppc_altivec_vmladduhm,
                            LHS, RHS, Zero, DAG);
  } else if (Op.getValueType() == MVT::v16i8) {
    SDOperand LHS = Op.getOperand(0), RHS = Op.getOperand(1);
    
    // Multiply the even 8-bit parts, producing 16-bit sums.
    SDOperand EvenParts = BuildIntrinsicOp(Intrinsic::ppc_altivec_vmuleub,
                                           LHS, RHS, DAG, MVT::v8i16);
    EvenParts = DAG.getNode(ISD::BIT_CONVERT, MVT::v16i8, EvenParts);
    
    // Multiply the odd 8-bit parts, producing 16-bit sums.
    SDOperand OddParts = BuildIntrinsicOp(Intrinsic::ppc_altivec_vmuloub,
                                          LHS, RHS, DAG, MVT::v8i16);
    OddParts = DAG.getNode(ISD::BIT_CONVERT, MVT::v16i8, OddParts);
    
    // Merge the results together.
    std::vector<SDOperand> Ops;
    for (unsigned i = 0; i != 8; ++i) {
      Ops.push_back(DAG.getConstant(2*i+1, MVT::i8));
      Ops.push_back(DAG.getConstant(2*i+1+16, MVT::i8));
    }
    
    return DAG.getNode(ISD::VECTOR_SHUFFLE, MVT::v16i8, EvenParts, OddParts,
                       DAG.getNode(ISD::BUILD_VECTOR, MVT::v16i8, Ops));
  } else {
    assert(0 && "Unknown mul to lower!");
    abort();
  }
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDOperand PPCTargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Wasn't expecting to be able to lower this!"); 
  case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
  case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
  case ISD::JumpTable:          return LowerJumpTable(Op, DAG);
  case ISD::SETCC:              return LowerSETCC(Op, DAG);
  case ISD::VASTART:            return LowerVASTART(Op, DAG, VarArgsFrameIndex);
  case ISD::FORMAL_ARGUMENTS:
      return LowerFORMAL_ARGUMENTS(Op, DAG, VarArgsFrameIndex);
  case ISD::CALL:               return LowerCALL(Op, DAG);
  case ISD::RET:                return LowerRET(Op, DAG);
    
  case ISD::SELECT_CC:          return LowerSELECT_CC(Op, DAG);
  case ISD::FP_TO_SINT:         return LowerFP_TO_SINT(Op, DAG);
  case ISD::SINT_TO_FP:         return LowerSINT_TO_FP(Op, DAG);

  // Lower 64-bit shifts.
  case ISD::SHL:                return LowerSHL(Op, DAG, getPointerTy());
  case ISD::SRL:                return LowerSRL(Op, DAG, getPointerTy());
  case ISD::SRA:                return LowerSRA(Op, DAG, getPointerTy());

  // Vector-related lowering.
  case ISD::BUILD_VECTOR:       return LowerBUILD_VECTOR(Op, DAG);
  case ISD::VECTOR_SHUFFLE:     return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::SCALAR_TO_VECTOR:   return LowerSCALAR_TO_VECTOR(Op, DAG);
  case ISD::MUL:                return LowerMUL(Op, DAG);
  }
  return SDOperand();
}

//===----------------------------------------------------------------------===//
//  Other Lowering Code
//===----------------------------------------------------------------------===//

MachineBasicBlock *
PPCTargetLowering::InsertAtEndOfBasicBlock(MachineInstr *MI,
                                           MachineBasicBlock *BB) {
  assert((MI->getOpcode() == PPC::SELECT_CC_I4 ||
          MI->getOpcode() == PPC::SELECT_CC_I8 ||
          MI->getOpcode() == PPC::SELECT_CC_F4 ||
          MI->getOpcode() == PPC::SELECT_CC_F8 ||
          MI->getOpcode() == PPC::SELECT_CC_VRRC) &&
         "Unexpected instr type to insert");
  
  // To "insert" a SELECT_CC instruction, we actually have to insert the diamond
  // control-flow pattern.  The incoming instruction knows the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  ilist<MachineBasicBlock>::iterator It = BB;
  ++It;
  
  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   cmpTY ccX, r1, r2
  //   bCC copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *thisMBB = BB;
  MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
  BuildMI(BB, MI->getOperand(4).getImmedValue(), 2)
    .addReg(MI->getOperand(1).getReg()).addMBB(sinkMBB);
  MachineFunction *F = BB->getParent();
  F->getBasicBlockList().insert(It, copy0MBB);
  F->getBasicBlockList().insert(It, sinkMBB);
  // Update machine-CFG edges by first adding all successors of the current
  // block to the new block which will contain the Phi node for the select.
  for(MachineBasicBlock::succ_iterator i = BB->succ_begin(), 
      e = BB->succ_end(); i != e; ++i)
    sinkMBB->addSuccessor(*i);
  // Next, remove all successors of the current block, and add the true
  // and fallthrough blocks as its successors.
  while(!BB->succ_empty())
    BB->removeSuccessor(BB->succ_begin());
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);
  
  //  copy0MBB:
  //   %FalseValue = ...
  //   # fallthrough to sinkMBB
  BB = copy0MBB;
  
  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);
  
  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
  //  ...
  BB = sinkMBB;
  BuildMI(BB, PPC::PHI, 4, MI->getOperand(0).getReg())
    .addReg(MI->getOperand(3).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

  delete MI;   // The pseudo instruction is gone now.
  return BB;
}

//===----------------------------------------------------------------------===//
// Target Optimization Hooks
//===----------------------------------------------------------------------===//

SDOperand PPCTargetLowering::PerformDAGCombine(SDNode *N, 
                                               DAGCombinerInfo &DCI) const {
  TargetMachine &TM = getTargetMachine();
  SelectionDAG &DAG = DCI.DAG;
  switch (N->getOpcode()) {
  default: break;
  case ISD::SINT_TO_FP:
    if (TM.getSubtarget<PPCSubtarget>().has64BitSupport()) {
      if (N->getOperand(0).getOpcode() == ISD::FP_TO_SINT) {
        // Turn (sint_to_fp (fp_to_sint X)) -> fctidz/fcfid without load/stores.
        // We allow the src/dst to be either f32/f64, but the intermediate
        // type must be i64.
        if (N->getOperand(0).getValueType() == MVT::i64) {
          SDOperand Val = N->getOperand(0).getOperand(0);
          if (Val.getValueType() == MVT::f32) {
            Val = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Val);
            DCI.AddToWorklist(Val.Val);
          }
            
          Val = DAG.getNode(PPCISD::FCTIDZ, MVT::f64, Val);
          DCI.AddToWorklist(Val.Val);
          Val = DAG.getNode(PPCISD::FCFID, MVT::f64, Val);
          DCI.AddToWorklist(Val.Val);
          if (N->getValueType(0) == MVT::f32) {
            Val = DAG.getNode(ISD::FP_ROUND, MVT::f32, Val);
            DCI.AddToWorklist(Val.Val);
          }
          return Val;
        } else if (N->getOperand(0).getValueType() == MVT::i32) {
          // If the intermediate type is i32, we can avoid the load/store here
          // too.
        }
      }
    }
    break;
  case ISD::STORE:
    // Turn STORE (FP_TO_SINT F) -> STFIWX(FCTIWZ(F)).
    if (TM.getSubtarget<PPCSubtarget>().hasSTFIWX() &&
        N->getOperand(1).getOpcode() == ISD::FP_TO_SINT &&
        N->getOperand(1).getValueType() == MVT::i32) {
      SDOperand Val = N->getOperand(1).getOperand(0);
      if (Val.getValueType() == MVT::f32) {
        Val = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Val);
        DCI.AddToWorklist(Val.Val);
      }
      Val = DAG.getNode(PPCISD::FCTIWZ, MVT::f64, Val);
      DCI.AddToWorklist(Val.Val);

      Val = DAG.getNode(PPCISD::STFIWX, MVT::Other, N->getOperand(0), Val,
                        N->getOperand(2), N->getOperand(3));
      DCI.AddToWorklist(Val.Val);
      return Val;
    }
    break;
  case PPCISD::VCMP: {
    // If a VCMPo node already exists with exactly the same operands as this
    // node, use its result instead of this node (VCMPo computes both a CR6 and
    // a normal output).
    //
    if (!N->getOperand(0).hasOneUse() &&
        !N->getOperand(1).hasOneUse() &&
        !N->getOperand(2).hasOneUse()) {
      
      // Scan all of the users of the LHS, looking for VCMPo's that match.
      SDNode *VCMPoNode = 0;
      
      SDNode *LHSN = N->getOperand(0).Val;
      for (SDNode::use_iterator UI = LHSN->use_begin(), E = LHSN->use_end();
           UI != E; ++UI)
        if ((*UI)->getOpcode() == PPCISD::VCMPo &&
            (*UI)->getOperand(1) == N->getOperand(1) &&
            (*UI)->getOperand(2) == N->getOperand(2) &&
            (*UI)->getOperand(0) == N->getOperand(0)) {
          VCMPoNode = *UI;
          break;
        }
      
      // If there is no VCMPo node, or if the flag value has a single use, don't
      // transform this.
      if (!VCMPoNode || VCMPoNode->hasNUsesOfValue(0, 1))
        break;
        
      // Look at the (necessarily single) use of the flag value.  If it has a 
      // chain, this transformation is more complex.  Note that multiple things
      // could use the value result, which we should ignore.
      SDNode *FlagUser = 0;
      for (SDNode::use_iterator UI = VCMPoNode->use_begin(); 
           FlagUser == 0; ++UI) {
        assert(UI != VCMPoNode->use_end() && "Didn't find user!");
        SDNode *User = *UI;
        for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i) {
          if (User->getOperand(i) == SDOperand(VCMPoNode, 1)) {
            FlagUser = User;
            break;
          }
        }
      }
      
      // If the user is a MFCR instruction, we know this is safe.  Otherwise we
      // give up for right now.
      if (FlagUser->getOpcode() == PPCISD::MFCR)
        return SDOperand(VCMPoNode, 0);
    }
    break;
  }
  case ISD::BR_CC: {
    // If this is a branch on an altivec predicate comparison, lower this so
    // that we don't have to do a MFCR: instead, branch directly on CR6.  This
    // lowering is done pre-legalize, because the legalizer lowers the predicate
    // compare down to code that is difficult to reassemble.
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(1))->get();
    SDOperand LHS = N->getOperand(2), RHS = N->getOperand(3);
    int CompareOpc;
    bool isDot;
    
    if (LHS.getOpcode() == ISD::INTRINSIC_WO_CHAIN &&
        isa<ConstantSDNode>(RHS) && (CC == ISD::SETEQ || CC == ISD::SETNE) &&
        getAltivecCompareInfo(LHS, CompareOpc, isDot)) {
      assert(isDot && "Can't compare against a vector result!");
      
      // If this is a comparison against something other than 0/1, then we know
      // that the condition is never/always true.
      unsigned Val = cast<ConstantSDNode>(RHS)->getValue();
      if (Val != 0 && Val != 1) {
        if (CC == ISD::SETEQ)      // Cond never true, remove branch.
          return N->getOperand(0);
        // Always !=, turn it into an unconditional branch.
        return DAG.getNode(ISD::BR, MVT::Other, 
                           N->getOperand(0), N->getOperand(4));
      }
    
      bool BranchOnWhenPredTrue = (CC == ISD::SETEQ) ^ (Val == 0);
      
      // Create the PPCISD altivec 'dot' comparison node.
      std::vector<SDOperand> Ops;
      std::vector<MVT::ValueType> VTs;
      Ops.push_back(LHS.getOperand(2));  // LHS of compare
      Ops.push_back(LHS.getOperand(3));  // RHS of compare
      Ops.push_back(DAG.getConstant(CompareOpc, MVT::i32));
      VTs.push_back(LHS.getOperand(2).getValueType());
      VTs.push_back(MVT::Flag);
      SDOperand CompNode = DAG.getNode(PPCISD::VCMPo, VTs, Ops);
      
      // Unpack the result based on how the target uses it.
      unsigned CompOpc;
      switch (cast<ConstantSDNode>(LHS.getOperand(1))->getValue()) {
      default:  // Can't happen, don't crash on invalid number though.
      case 0:   // Branch on the value of the EQ bit of CR6.
        CompOpc = BranchOnWhenPredTrue ? PPC::BEQ : PPC::BNE;
        break;
      case 1:   // Branch on the inverted value of the EQ bit of CR6.
        CompOpc = BranchOnWhenPredTrue ? PPC::BNE : PPC::BEQ;
        break;
      case 2:   // Branch on the value of the LT bit of CR6.
        CompOpc = BranchOnWhenPredTrue ? PPC::BLT : PPC::BGE;
        break;
      case 3:   // Branch on the inverted value of the LT bit of CR6.
        CompOpc = BranchOnWhenPredTrue ? PPC::BGE : PPC::BLT;
        break;
      }

      return DAG.getNode(PPCISD::COND_BRANCH, MVT::Other, N->getOperand(0),
                         DAG.getRegister(PPC::CR6, MVT::i32),
                         DAG.getConstant(CompOpc, MVT::i32),
                         N->getOperand(4), CompNode.getValue(1));
    }
    break;
  }
  }
  
  return SDOperand();
}

//===----------------------------------------------------------------------===//
// Inline Assembly Support
//===----------------------------------------------------------------------===//

void PPCTargetLowering::computeMaskedBitsForTargetNode(const SDOperand Op,
                                                       uint64_t Mask,
                                                       uint64_t &KnownZero, 
                                                       uint64_t &KnownOne,
                                                       unsigned Depth) const {
  KnownZero = 0;
  KnownOne = 0;
  switch (Op.getOpcode()) {
  default: break;
  case ISD::INTRINSIC_WO_CHAIN: {
    switch (cast<ConstantSDNode>(Op.getOperand(0))->getValue()) {
    default: break;
    case Intrinsic::ppc_altivec_vcmpbfp_p:
    case Intrinsic::ppc_altivec_vcmpeqfp_p:
    case Intrinsic::ppc_altivec_vcmpequb_p:
    case Intrinsic::ppc_altivec_vcmpequh_p:
    case Intrinsic::ppc_altivec_vcmpequw_p:
    case Intrinsic::ppc_altivec_vcmpgefp_p:
    case Intrinsic::ppc_altivec_vcmpgtfp_p:
    case Intrinsic::ppc_altivec_vcmpgtsb_p:
    case Intrinsic::ppc_altivec_vcmpgtsh_p:
    case Intrinsic::ppc_altivec_vcmpgtsw_p:
    case Intrinsic::ppc_altivec_vcmpgtub_p:
    case Intrinsic::ppc_altivec_vcmpgtuh_p:
    case Intrinsic::ppc_altivec_vcmpgtuw_p:
      KnownZero = ~1U;  // All bits but the low one are known to be zero.
      break;
    }        
  }
  }
}


/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
PPCTargetLowering::ConstraintType 
PPCTargetLowering::getConstraintType(char ConstraintLetter) const {
  switch (ConstraintLetter) {
  default: break;
  case 'b':
  case 'r':
  case 'f':
  case 'v':
  case 'y':
    return C_RegisterClass;
  }  
  return TargetLowering::getConstraintType(ConstraintLetter);
}


std::vector<unsigned> PPCTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {      // GCC RS6000 Constraint Letters
    default: break;  // Unknown constriant letter
    case 'b': 
      return make_vector<unsigned>(/*no R0*/ PPC::R1 , PPC::R2 , PPC::R3 ,
                                   PPC::R4 , PPC::R5 , PPC::R6 , PPC::R7 ,
                                   PPC::R8 , PPC::R9 , PPC::R10, PPC::R11, 
                                   PPC::R12, PPC::R13, PPC::R14, PPC::R15, 
                                   PPC::R16, PPC::R17, PPC::R18, PPC::R19, 
                                   PPC::R20, PPC::R21, PPC::R22, PPC::R23, 
                                   PPC::R24, PPC::R25, PPC::R26, PPC::R27, 
                                   PPC::R28, PPC::R29, PPC::R30, PPC::R31, 
                                   0);
    case 'r': 
      return make_vector<unsigned>(PPC::R0 , PPC::R1 , PPC::R2 , PPC::R3 ,
                                   PPC::R4 , PPC::R5 , PPC::R6 , PPC::R7 ,
                                   PPC::R8 , PPC::R9 , PPC::R10, PPC::R11, 
                                   PPC::R12, PPC::R13, PPC::R14, PPC::R15, 
                                   PPC::R16, PPC::R17, PPC::R18, PPC::R19, 
                                   PPC::R20, PPC::R21, PPC::R22, PPC::R23, 
                                   PPC::R24, PPC::R25, PPC::R26, PPC::R27, 
                                   PPC::R28, PPC::R29, PPC::R30, PPC::R31, 
                                   0);
    case 'f': 
      return make_vector<unsigned>(PPC::F0 , PPC::F1 , PPC::F2 , PPC::F3 ,
                                   PPC::F4 , PPC::F5 , PPC::F6 , PPC::F7 ,
                                   PPC::F8 , PPC::F9 , PPC::F10, PPC::F11, 
                                   PPC::F12, PPC::F13, PPC::F14, PPC::F15, 
                                   PPC::F16, PPC::F17, PPC::F18, PPC::F19, 
                                   PPC::F20, PPC::F21, PPC::F22, PPC::F23, 
                                   PPC::F24, PPC::F25, PPC::F26, PPC::F27, 
                                   PPC::F28, PPC::F29, PPC::F30, PPC::F31, 
                                   0);
    case 'v': 
      return make_vector<unsigned>(PPC::V0 , PPC::V1 , PPC::V2 , PPC::V3 ,
                                   PPC::V4 , PPC::V5 , PPC::V6 , PPC::V7 ,
                                   PPC::V8 , PPC::V9 , PPC::V10, PPC::V11, 
                                   PPC::V12, PPC::V13, PPC::V14, PPC::V15, 
                                   PPC::V16, PPC::V17, PPC::V18, PPC::V19, 
                                   PPC::V20, PPC::V21, PPC::V22, PPC::V23, 
                                   PPC::V24, PPC::V25, PPC::V26, PPC::V27, 
                                   PPC::V28, PPC::V29, PPC::V30, PPC::V31, 
                                   0);
    case 'y': 
      return make_vector<unsigned>(PPC::CR0, PPC::CR1, PPC::CR2, PPC::CR3,
                                   PPC::CR4, PPC::CR5, PPC::CR6, PPC::CR7,
                                   0);
    }
  }
  
  return std::vector<unsigned>();
}

// isOperandValidForConstraint
bool PPCTargetLowering::
isOperandValidForConstraint(SDOperand Op, char Letter) {
  switch (Letter) {
  default: break;
  case 'I':
  case 'J':
  case 'K':
  case 'L':
  case 'M':
  case 'N':
  case 'O':
  case 'P': {
    if (!isa<ConstantSDNode>(Op)) return false;  // Must be an immediate.
    unsigned Value = cast<ConstantSDNode>(Op)->getValue();
    switch (Letter) {
    default: assert(0 && "Unknown constraint letter!");
    case 'I':  // "I" is a signed 16-bit constant.
      return (short)Value == (int)Value;
    case 'J':  // "J" is a constant with only the high-order 16 bits nonzero.
    case 'L':  // "L" is a signed 16-bit constant shifted left 16 bits.
      return (short)Value == 0;
    case 'K':  // "K" is a constant with only the low-order 16 bits nonzero.
      return (Value >> 16) == 0;
    case 'M':  // "M" is a constant that is greater than 31.
      return Value > 31;
    case 'N':  // "N" is a positive constant that is an exact power of two.
      return (int)Value > 0 && isPowerOf2_32(Value);
    case 'O':  // "O" is the constant zero. 
      return Value == 0;
    case 'P':  // "P" is a constant whose negation is a signed 16-bit constant.
      return (short)-Value == (int)-Value;
    }
    break;
  }
  }
  
  // Handle standard constraint letters.
  return TargetLowering::isOperandValidForConstraint(Op, Letter);
}

/// isLegalAddressImmediate - Return true if the integer value can be used
/// as the offset of the target addressing mode.
bool PPCTargetLowering::isLegalAddressImmediate(int64_t V) const {
  // PPC allows a sign-extended 16-bit immediate field.
  return (V > -(1 << 16) && V < (1 << 16)-1);
}
