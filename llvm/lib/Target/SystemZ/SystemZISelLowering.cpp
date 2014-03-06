//===-- SystemZISelLowering.cpp - SystemZ DAG lowering implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemZTargetLowering class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "systemz-lower"

#include "SystemZISelLowering.h"
#include "SystemZCallingConv.h"
#include "SystemZConstantPoolValue.h"
#include "SystemZMachineFunctionInfo.h"
#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include <cctype>

using namespace llvm;

namespace {
// Represents a sequence for extracting a 0/1 value from an IPM result:
// (((X ^ XORValue) + AddValue) >> Bit)
struct IPMConversion {
  IPMConversion(unsigned xorValue, int64_t addValue, unsigned bit)
    : XORValue(xorValue), AddValue(addValue), Bit(bit) {}

  int64_t XORValue;
  int64_t AddValue;
  unsigned Bit;
};

// Represents information about a comparison.
struct Comparison {
  Comparison(SDValue Op0In, SDValue Op1In)
    : Op0(Op0In), Op1(Op1In), Opcode(0), ICmpType(0), CCValid(0), CCMask(0) {}

  // The operands to the comparison.
  SDValue Op0, Op1;

  // The opcode that should be used to compare Op0 and Op1.
  unsigned Opcode;

  // A SystemZICMP value.  Only used for integer comparisons.
  unsigned ICmpType;

  // The mask of CC values that Opcode can produce.
  unsigned CCValid;

  // The mask of CC values for which the original condition is true.
  unsigned CCMask;
};
} // end anonymous namespace

// Classify VT as either 32 or 64 bit.
static bool is32Bit(EVT VT) {
  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::i32:
    return true;
  case MVT::i64:
    return false;
  default:
    llvm_unreachable("Unsupported type");
  }
}

// Return a version of MachineOperand that can be safely used before the
// final use.
static MachineOperand earlyUseOperand(MachineOperand Op) {
  if (Op.isReg())
    Op.setIsKill(false);
  return Op;
}

SystemZTargetLowering::SystemZTargetLowering(SystemZTargetMachine &tm)
  : TargetLowering(tm, new TargetLoweringObjectFileELF()),
    Subtarget(*tm.getSubtargetImpl()), TM(tm) {
  MVT PtrVT = getPointerTy();

  // Set up the register classes.
  if (Subtarget.hasHighWord())
    addRegisterClass(MVT::i32, &SystemZ::GRX32BitRegClass);
  else
    addRegisterClass(MVT::i32, &SystemZ::GR32BitRegClass);
  addRegisterClass(MVT::i64,  &SystemZ::GR64BitRegClass);
  addRegisterClass(MVT::f32,  &SystemZ::FP32BitRegClass);
  addRegisterClass(MVT::f64,  &SystemZ::FP64BitRegClass);
  addRegisterClass(MVT::f128, &SystemZ::FP128BitRegClass);

  // Compute derived properties from the register classes
  computeRegisterProperties();

  // Set up special registers.
  setExceptionPointerRegister(SystemZ::R6D);
  setExceptionSelectorRegister(SystemZ::R7D);
  setStackPointerRegisterToSaveRestore(SystemZ::R15D);

  // TODO: It may be better to default to latency-oriented scheduling, however
  // LLVM's current latency-oriented scheduler can't handle physreg definitions
  // such as SystemZ has with CC, so set this to the register-pressure
  // scheduler, because it can.
  setSchedulingPreference(Sched::RegPressure);

  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent); // FIXME: Is this correct?

  // Instructions are strings of 2-byte aligned 2-byte values.
  setMinFunctionAlignment(2);

  // Handle operations that are handled in a similar way for all types.
  for (unsigned I = MVT::FIRST_INTEGER_VALUETYPE;
       I <= MVT::LAST_FP_VALUETYPE;
       ++I) {
    MVT VT = MVT::SimpleValueType(I);
    if (isTypeLegal(VT)) {
      // Lower SET_CC into an IPM-based sequence.
      setOperationAction(ISD::SETCC, VT, Custom);

      // Expand SELECT(C, A, B) into SELECT_CC(X, 0, A, B, NE).
      setOperationAction(ISD::SELECT, VT, Expand);

      // Lower SELECT_CC and BR_CC into separate comparisons and branches.
      setOperationAction(ISD::SELECT_CC, VT, Custom);
      setOperationAction(ISD::BR_CC,     VT, Custom);
    }
  }

  // Expand jump table branches as address arithmetic followed by an
  // indirect jump.
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);

  // Expand BRCOND into a BR_CC (see above).
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);

  // Handle integer types.
  for (unsigned I = MVT::FIRST_INTEGER_VALUETYPE;
       I <= MVT::LAST_INTEGER_VALUETYPE;
       ++I) {
    MVT VT = MVT::SimpleValueType(I);
    if (isTypeLegal(VT)) {
      // Expand individual DIV and REMs into DIVREMs.
      setOperationAction(ISD::SDIV, VT, Expand);
      setOperationAction(ISD::UDIV, VT, Expand);
      setOperationAction(ISD::SREM, VT, Expand);
      setOperationAction(ISD::UREM, VT, Expand);
      setOperationAction(ISD::SDIVREM, VT, Custom);
      setOperationAction(ISD::UDIVREM, VT, Custom);

      // Lower ATOMIC_LOAD and ATOMIC_STORE into normal volatile loads and
      // stores, putting a serialization instruction after the stores.
      setOperationAction(ISD::ATOMIC_LOAD,  VT, Custom);
      setOperationAction(ISD::ATOMIC_STORE, VT, Custom);

      // Lower ATOMIC_LOAD_SUB into ATOMIC_LOAD_ADD if LAA and LAAG are
      // available, or if the operand is constant.
      setOperationAction(ISD::ATOMIC_LOAD_SUB, VT, Custom);

      // No special instructions for these.
      setOperationAction(ISD::CTPOP,           VT, Expand);
      setOperationAction(ISD::CTTZ,            VT, Expand);
      setOperationAction(ISD::CTTZ_ZERO_UNDEF, VT, Expand);
      setOperationAction(ISD::CTLZ_ZERO_UNDEF, VT, Expand);
      setOperationAction(ISD::ROTR,            VT, Expand);

      // Use *MUL_LOHI where possible instead of MULH*.
      setOperationAction(ISD::MULHS, VT, Expand);
      setOperationAction(ISD::MULHU, VT, Expand);
      setOperationAction(ISD::SMUL_LOHI, VT, Custom);
      setOperationAction(ISD::UMUL_LOHI, VT, Custom);

      // We have instructions for signed but not unsigned FP conversion.
      setOperationAction(ISD::FP_TO_UINT, VT, Expand);
    }
  }

  // Type legalization will convert 8- and 16-bit atomic operations into
  // forms that operate on i32s (but still keeping the original memory VT).
  // Lower them into full i32 operations.
  setOperationAction(ISD::ATOMIC_SWAP,      MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_ADD,  MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_SUB,  MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_AND,  MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_OR,   MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_XOR,  MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_NAND, MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_MIN,  MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_MAX,  MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_UMIN, MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_UMAX, MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_CMP_SWAP,  MVT::i32, Custom);

  // We have instructions for signed but not unsigned FP conversion.
  // Handle unsigned 32-bit types as signed 64-bit types.
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Promote);
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Expand);

  // We have native support for a 64-bit CTLZ, via FLOGR.
  setOperationAction(ISD::CTLZ, MVT::i32, Promote);
  setOperationAction(ISD::CTLZ, MVT::i64, Legal);

  // Give LowerOperation the chance to replace 64-bit ORs with subregs.
  setOperationAction(ISD::OR, MVT::i64, Custom);

  // Give LowerOperation the chance to optimize SIGN_EXTEND sequences.
  setOperationAction(ISD::SIGN_EXTEND, MVT::i64, Custom);

  // FIXME: Can we support these natively?
  setOperationAction(ISD::SRL_PARTS, MVT::i64, Expand);
  setOperationAction(ISD::SHL_PARTS, MVT::i64, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i64, Expand);

  // We have native instructions for i8, i16 and i32 extensions, but not i1.
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::EXTLOAD,  MVT::i1, Promote);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  // Handle the various types of symbolic address.
  setOperationAction(ISD::ConstantPool,     PtrVT, Custom);
  setOperationAction(ISD::GlobalAddress,    PtrVT, Custom);
  setOperationAction(ISD::GlobalTLSAddress, PtrVT, Custom);
  setOperationAction(ISD::BlockAddress,     PtrVT, Custom);
  setOperationAction(ISD::JumpTable,        PtrVT, Custom);

  // We need to handle dynamic allocations specially because of the
  // 160-byte area at the bottom of the stack.
  setOperationAction(ISD::DYNAMIC_STACKALLOC, PtrVT, Custom);

  // Use custom expanders so that we can force the function to use
  // a frame pointer.
  setOperationAction(ISD::STACKSAVE,    MVT::Other, Custom);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Custom);

  // Handle prefetches with PFD or PFDRL.
  setOperationAction(ISD::PREFETCH, MVT::Other, Custom);

  // Handle floating-point types.
  for (unsigned I = MVT::FIRST_FP_VALUETYPE;
       I <= MVT::LAST_FP_VALUETYPE;
       ++I) {
    MVT VT = MVT::SimpleValueType(I);
    if (isTypeLegal(VT)) {
      // We can use FI for FRINT.
      setOperationAction(ISD::FRINT, VT, Legal);

      // We can use the extended form of FI for other rounding operations.
      if (Subtarget.hasFPExtension()) {
        setOperationAction(ISD::FNEARBYINT, VT, Legal);
        setOperationAction(ISD::FFLOOR, VT, Legal);
        setOperationAction(ISD::FCEIL, VT, Legal);
        setOperationAction(ISD::FTRUNC, VT, Legal);
        setOperationAction(ISD::FROUND, VT, Legal);
      }

      // No special instructions for these.
      setOperationAction(ISD::FSIN, VT, Expand);
      setOperationAction(ISD::FCOS, VT, Expand);
      setOperationAction(ISD::FREM, VT, Expand);
    }
  }

  // We have fused multiply-addition for f32 and f64 but not f128.
  setOperationAction(ISD::FMA, MVT::f32,  Legal);
  setOperationAction(ISD::FMA, MVT::f64,  Legal);
  setOperationAction(ISD::FMA, MVT::f128, Expand);

  // Needed so that we don't try to implement f128 constant loads using
  // a load-and-extend of a f80 constant (in cases where the constant
  // would fit in an f80).
  setLoadExtAction(ISD::EXTLOAD, MVT::f80, Expand);

  // Floating-point truncation and stores need to be done separately.
  setTruncStoreAction(MVT::f64,  MVT::f32, Expand);
  setTruncStoreAction(MVT::f128, MVT::f32, Expand);
  setTruncStoreAction(MVT::f128, MVT::f64, Expand);

  // We have 64-bit FPR<->GPR moves, but need special handling for
  // 32-bit forms.
  setOperationAction(ISD::BITCAST, MVT::i32, Custom);
  setOperationAction(ISD::BITCAST, MVT::f32, Custom);

  // VASTART and VACOPY need to deal with the SystemZ-specific varargs
  // structure, but VAEND is a no-op.
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VACOPY,  MVT::Other, Custom);
  setOperationAction(ISD::VAEND,   MVT::Other, Expand);

  // We want to use MVC in preference to even a single load/store pair.
  MaxStoresPerMemcpy = 0;
  MaxStoresPerMemcpyOptSize = 0;

  // The main memset sequence is a byte store followed by an MVC.
  // Two STC or MV..I stores win over that, but the kind of fused stores
  // generated by target-independent code don't when the byte value is
  // variable.  E.g.  "STC <reg>;MHI <reg>,257;STH <reg>" is not better
  // than "STC;MVC".  Handle the choice in target-specific code instead.
  MaxStoresPerMemset = 0;
  MaxStoresPerMemsetOptSize = 0;
}

EVT SystemZTargetLowering::getSetCCResultType(LLVMContext &, EVT VT) const {
  if (!VT.isVector())
    return MVT::i32;
  return VT.changeVectorElementTypeToInteger();
}

bool SystemZTargetLowering::isFMAFasterThanFMulAndFAdd(EVT VT) const {
  VT = VT.getScalarType();

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
  case MVT::f64:
    return true;
  case MVT::f128:
    return false;
  default:
    break;
  }

  return false;
}

bool SystemZTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  // We can load zero using LZ?R and negative zero using LZ?R;LC?BR.
  return Imm.isZero() || Imm.isNegZero();
}

bool SystemZTargetLowering::allowsUnalignedMemoryAccesses(EVT VT,
                                                          unsigned,
                                                          bool *Fast) const {
  // Unaligned accesses should never be slower than the expanded version.
  // We check specifically for aligned accesses in the few cases where
  // they are required.
  if (Fast)
    *Fast = true;
  return true;
}
  
bool SystemZTargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                                  Type *Ty) const {
  // Punt on globals for now, although they can be used in limited
  // RELATIVE LONG cases.
  if (AM.BaseGV)
    return false;

  // Require a 20-bit signed offset.
  if (!isInt<20>(AM.BaseOffs))
    return false;

  // Indexing is OK but no scale factor can be applied.
  return AM.Scale == 0 || AM.Scale == 1;
}

bool SystemZTargetLowering::isTruncateFree(Type *FromType, Type *ToType) const {
  if (!FromType->isIntegerTy() || !ToType->isIntegerTy())
    return false;
  unsigned FromBits = FromType->getPrimitiveSizeInBits();
  unsigned ToBits = ToType->getPrimitiveSizeInBits();
  return FromBits > ToBits;
}

bool SystemZTargetLowering::isTruncateFree(EVT FromVT, EVT ToVT) const {
  if (!FromVT.isInteger() || !ToVT.isInteger())
    return false;
  unsigned FromBits = FromVT.getSizeInBits();
  unsigned ToBits = ToVT.getSizeInBits();
  return FromBits > ToBits;
}

//===----------------------------------------------------------------------===//
// Inline asm support
//===----------------------------------------------------------------------===//

TargetLowering::ConstraintType
SystemZTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'a': // Address register
    case 'd': // Data register (equivalent to 'r')
    case 'f': // Floating-point register
    case 'h': // High-part register
    case 'r': // General-purpose register
      return C_RegisterClass;

    case 'Q': // Memory with base and unsigned 12-bit displacement
    case 'R': // Likewise, plus an index
    case 'S': // Memory with base and signed 20-bit displacement
    case 'T': // Likewise, plus an index
    case 'm': // Equivalent to 'T'.
      return C_Memory;

    case 'I': // Unsigned 8-bit constant
    case 'J': // Unsigned 12-bit constant
    case 'K': // Signed 16-bit constant
    case 'L': // Signed 20-bit displacement (on all targets we support)
    case 'M': // 0x7fffffff
      return C_Other;

    default:
      break;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

TargetLowering::ConstraintWeight SystemZTargetLowering::
getSingleConstraintMatchWeight(AsmOperandInfo &info,
                               const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;
  // If we don't have a value, we can't do a match,
  // but allow it at the lowest weight.
  if (CallOperandVal == NULL)
    return CW_Default;
  Type *type = CallOperandVal->getType();
  // Look at the constraint type.
  switch (*constraint) {
  default:
    weight = TargetLowering::getSingleConstraintMatchWeight(info, constraint);
    break;

  case 'a': // Address register
  case 'd': // Data register (equivalent to 'r')
  case 'h': // High-part register
  case 'r': // General-purpose register
    if (CallOperandVal->getType()->isIntegerTy())
      weight = CW_Register;
    break;

  case 'f': // Floating-point register
    if (type->isFloatingPointTy())
      weight = CW_Register;
    break;

  case 'I': // Unsigned 8-bit constant
    if (auto *C = dyn_cast<ConstantInt>(CallOperandVal))
      if (isUInt<8>(C->getZExtValue()))
        weight = CW_Constant;
    break;

  case 'J': // Unsigned 12-bit constant
    if (auto *C = dyn_cast<ConstantInt>(CallOperandVal))
      if (isUInt<12>(C->getZExtValue()))
        weight = CW_Constant;
    break;

  case 'K': // Signed 16-bit constant
    if (auto *C = dyn_cast<ConstantInt>(CallOperandVal))
      if (isInt<16>(C->getSExtValue()))
        weight = CW_Constant;
    break;

  case 'L': // Signed 20-bit displacement (on all targets we support)
    if (auto *C = dyn_cast<ConstantInt>(CallOperandVal))
      if (isInt<20>(C->getSExtValue()))
        weight = CW_Constant;
    break;

  case 'M': // 0x7fffffff
    if (auto *C = dyn_cast<ConstantInt>(CallOperandVal))
      if (C->getZExtValue() == 0x7fffffff)
        weight = CW_Constant;
    break;
  }
  return weight;
}

// Parse a "{tNNN}" register constraint for which the register type "t"
// has already been verified.  MC is the class associated with "t" and
// Map maps 0-based register numbers to LLVM register numbers.
static std::pair<unsigned, const TargetRegisterClass *>
parseRegisterNumber(const std::string &Constraint,
                    const TargetRegisterClass *RC, const unsigned *Map) {
  assert(*(Constraint.end()-1) == '}' && "Missing '}'");
  if (isdigit(Constraint[2])) {
    std::string Suffix(Constraint.data() + 2, Constraint.size() - 2);
    unsigned Index = atoi(Suffix.c_str());
    if (Index < 16 && Map[Index])
      return std::make_pair(Map[Index], RC);
  }
  return std::make_pair(0u, static_cast<TargetRegisterClass*>(0));
}

std::pair<unsigned, const TargetRegisterClass *> SystemZTargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint, MVT VT) const {
  if (Constraint.size() == 1) {
    // GCC Constraint Letters
    switch (Constraint[0]) {
    default: break;
    case 'd': // Data register (equivalent to 'r')
    case 'r': // General-purpose register
      if (VT == MVT::i64)
        return std::make_pair(0U, &SystemZ::GR64BitRegClass);
      else if (VT == MVT::i128)
        return std::make_pair(0U, &SystemZ::GR128BitRegClass);
      return std::make_pair(0U, &SystemZ::GR32BitRegClass);

    case 'a': // Address register
      if (VT == MVT::i64)
        return std::make_pair(0U, &SystemZ::ADDR64BitRegClass);
      else if (VT == MVT::i128)
        return std::make_pair(0U, &SystemZ::ADDR128BitRegClass);
      return std::make_pair(0U, &SystemZ::ADDR32BitRegClass);

    case 'h': // High-part register (an LLVM extension)
      return std::make_pair(0U, &SystemZ::GRH32BitRegClass);

    case 'f': // Floating-point register
      if (VT == MVT::f64)
        return std::make_pair(0U, &SystemZ::FP64BitRegClass);
      else if (VT == MVT::f128)
        return std::make_pair(0U, &SystemZ::FP128BitRegClass);
      return std::make_pair(0U, &SystemZ::FP32BitRegClass);
    }
  }
  if (Constraint[0] == '{') {
    // We need to override the default register parsing for GPRs and FPRs
    // because the interpretation depends on VT.  The internal names of
    // the registers are also different from the external names
    // (F0D and F0S instead of F0, etc.).
    if (Constraint[1] == 'r') {
      if (VT == MVT::i32)
        return parseRegisterNumber(Constraint, &SystemZ::GR32BitRegClass,
                                   SystemZMC::GR32Regs);
      if (VT == MVT::i128)
        return parseRegisterNumber(Constraint, &SystemZ::GR128BitRegClass,
                                   SystemZMC::GR128Regs);
      return parseRegisterNumber(Constraint, &SystemZ::GR64BitRegClass,
                                 SystemZMC::GR64Regs);
    }
    if (Constraint[1] == 'f') {
      if (VT == MVT::f32)
        return parseRegisterNumber(Constraint, &SystemZ::FP32BitRegClass,
                                   SystemZMC::FP32Regs);
      if (VT == MVT::f128)
        return parseRegisterNumber(Constraint, &SystemZ::FP128BitRegClass,
                                   SystemZMC::FP128Regs);
      return parseRegisterNumber(Constraint, &SystemZ::FP64BitRegClass,
                                 SystemZMC::FP64Regs);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

void SystemZTargetLowering::
LowerAsmOperandForConstraint(SDValue Op, std::string &Constraint,
                             std::vector<SDValue> &Ops,
                             SelectionDAG &DAG) const {
  // Only support length 1 constraints for now.
  if (Constraint.length() == 1) {
    switch (Constraint[0]) {
    case 'I': // Unsigned 8-bit constant
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (isUInt<8>(C->getZExtValue()))
          Ops.push_back(DAG.getTargetConstant(C->getZExtValue(),
                                              Op.getValueType()));
      return;

    case 'J': // Unsigned 12-bit constant
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (isUInt<12>(C->getZExtValue()))
          Ops.push_back(DAG.getTargetConstant(C->getZExtValue(),
                                              Op.getValueType()));
      return;

    case 'K': // Signed 16-bit constant
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (isInt<16>(C->getSExtValue()))
          Ops.push_back(DAG.getTargetConstant(C->getSExtValue(),
                                              Op.getValueType()));
      return;

    case 'L': // Signed 20-bit displacement (on all targets we support)
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (isInt<20>(C->getSExtValue()))
          Ops.push_back(DAG.getTargetConstant(C->getSExtValue(),
                                              Op.getValueType()));
      return;

    case 'M': // 0x7fffffff
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (C->getZExtValue() == 0x7fffffff)
          Ops.push_back(DAG.getTargetConstant(C->getZExtValue(),
                                              Op.getValueType()));
      return;
    }
  }
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

//===----------------------------------------------------------------------===//
// Calling conventions
//===----------------------------------------------------------------------===//

#include "SystemZGenCallingConv.inc"

bool SystemZTargetLowering::allowTruncateForTailCall(Type *FromType,
                                                     Type *ToType) const {
  return isTruncateFree(FromType, ToType);
}

bool SystemZTargetLowering::mayBeEmittedAsTailCall(CallInst *CI) const {
  if (!CI->isTailCall())
    return false;
  return true;
}

// Value is a value that has been passed to us in the location described by VA
// (and so has type VA.getLocVT()).  Convert Value to VA.getValVT(), chaining
// any loads onto Chain.
static SDValue convertLocVTToValVT(SelectionDAG &DAG, SDLoc DL,
                                   CCValAssign &VA, SDValue Chain,
                                   SDValue Value) {
  // If the argument has been promoted from a smaller type, insert an
  // assertion to capture this.
  if (VA.getLocInfo() == CCValAssign::SExt)
    Value = DAG.getNode(ISD::AssertSext, DL, VA.getLocVT(), Value,
                        DAG.getValueType(VA.getValVT()));
  else if (VA.getLocInfo() == CCValAssign::ZExt)
    Value = DAG.getNode(ISD::AssertZext, DL, VA.getLocVT(), Value,
                        DAG.getValueType(VA.getValVT()));

  if (VA.isExtInLoc())
    Value = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Value);
  else if (VA.getLocInfo() == CCValAssign::Indirect)
    Value = DAG.getLoad(VA.getValVT(), DL, Chain, Value,
                        MachinePointerInfo(), false, false, false, 0);
  else
    assert(VA.getLocInfo() == CCValAssign::Full && "Unsupported getLocInfo");
  return Value;
}

// Value is a value of type VA.getValVT() that we need to copy into
// the location described by VA.  Return a copy of Value converted to
// VA.getValVT().  The caller is responsible for handling indirect values.
static SDValue convertValVTToLocVT(SelectionDAG &DAG, SDLoc DL,
                                   CCValAssign &VA, SDValue Value) {
  switch (VA.getLocInfo()) {
  case CCValAssign::SExt:
    return DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Value);
  case CCValAssign::ZExt:
    return DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Value);
  case CCValAssign::AExt:
    return DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Value);
  case CCValAssign::Full:
    return Value;
  default:
    llvm_unreachable("Unhandled getLocInfo()");
  }
}

SDValue SystemZTargetLowering::
LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                     const SmallVectorImpl<ISD::InputArg> &Ins,
                     SDLoc DL, SelectionDAG &DAG,
                     SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SystemZMachineFunctionInfo *FuncInfo =
    MF.getInfo<SystemZMachineFunctionInfo>();
  auto *TFL = static_cast<const SystemZFrameLowering *>(TM.getFrameLowering());

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, TM, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_SystemZ);

  unsigned NumFixedGPRs = 0;
  unsigned NumFixedFPRs = 0;
  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    SDValue ArgValue;
    CCValAssign &VA = ArgLocs[I];
    EVT LocVT = VA.getLocVT();
    if (VA.isRegLoc()) {
      // Arguments passed in registers
      const TargetRegisterClass *RC;
      switch (LocVT.getSimpleVT().SimpleTy) {
      default:
        // Integers smaller than i64 should be promoted to i64.
        llvm_unreachable("Unexpected argument type");
      case MVT::i32:
        NumFixedGPRs += 1;
        RC = &SystemZ::GR32BitRegClass;
        break;
      case MVT::i64:
        NumFixedGPRs += 1;
        RC = &SystemZ::GR64BitRegClass;
        break;
      case MVT::f32:
        NumFixedFPRs += 1;
        RC = &SystemZ::FP32BitRegClass;
        break;
      case MVT::f64:
        NumFixedFPRs += 1;
        RC = &SystemZ::FP64BitRegClass;
        break;
      }

      unsigned VReg = MRI.createVirtualRegister(RC);
      MRI.addLiveIn(VA.getLocReg(), VReg);
      ArgValue = DAG.getCopyFromReg(Chain, DL, VReg, LocVT);
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");

      // Create the frame index object for this incoming parameter.
      int FI = MFI->CreateFixedObject(LocVT.getSizeInBits() / 8,
                                      VA.getLocMemOffset(), true);

      // Create the SelectionDAG nodes corresponding to a load
      // from this parameter.  Unpromoted ints and floats are
      // passed as right-justified 8-byte values.
      EVT PtrVT = getPointerTy();
      SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
      if (VA.getLocVT() == MVT::i32 || VA.getLocVT() == MVT::f32)
        FIN = DAG.getNode(ISD::ADD, DL, PtrVT, FIN, DAG.getIntPtrConstant(4));
      ArgValue = DAG.getLoad(LocVT, DL, Chain, FIN,
                             MachinePointerInfo::getFixedStack(FI),
                             false, false, false, 0);
    }

    // Convert the value of the argument register into the value that's
    // being passed.
    InVals.push_back(convertLocVTToValVT(DAG, DL, VA, Chain, ArgValue));
  }

  if (IsVarArg) {
    // Save the number of non-varargs registers for later use by va_start, etc.
    FuncInfo->setVarArgsFirstGPR(NumFixedGPRs);
    FuncInfo->setVarArgsFirstFPR(NumFixedFPRs);

    // Likewise the address (in the form of a frame index) of where the
    // first stack vararg would be.  The 1-byte size here is arbitrary.
    int64_t StackSize = CCInfo.getNextStackOffset();
    FuncInfo->setVarArgsFrameIndex(MFI->CreateFixedObject(1, StackSize, true));

    // ...and a similar frame index for the caller-allocated save area
    // that will be used to store the incoming registers.
    int64_t RegSaveOffset = TFL->getOffsetOfLocalArea();
    unsigned RegSaveIndex = MFI->CreateFixedObject(1, RegSaveOffset, true);
    FuncInfo->setRegSaveFrameIndex(RegSaveIndex);

    // Store the FPR varargs in the reserved frame slots.  (We store the
    // GPRs as part of the prologue.)
    if (NumFixedFPRs < SystemZ::NumArgFPRs) {
      SDValue MemOps[SystemZ::NumArgFPRs];
      for (unsigned I = NumFixedFPRs; I < SystemZ::NumArgFPRs; ++I) {
        unsigned Offset = TFL->getRegSpillOffset(SystemZ::ArgFPRs[I]);
        int FI = MFI->CreateFixedObject(8, RegSaveOffset + Offset, true);
        SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
        unsigned VReg = MF.addLiveIn(SystemZ::ArgFPRs[I],
                                     &SystemZ::FP64BitRegClass);
        SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, VReg, MVT::f64);
        MemOps[I] = DAG.getStore(ArgValue.getValue(1), DL, ArgValue, FIN,
                                 MachinePointerInfo::getFixedStack(FI),
                                 false, false, 0);

      }
      // Join the stores, which are independent of one another.
      Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other,
                          &MemOps[NumFixedFPRs],
                          SystemZ::NumArgFPRs - NumFixedFPRs);
    }
  }

  return Chain;
}

static bool canUseSiblingCall(CCState ArgCCInfo,
                              SmallVectorImpl<CCValAssign> &ArgLocs) {
  // Punt if there are any indirect or stack arguments, or if the call
  // needs the call-saved argument register R6.
  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    CCValAssign &VA = ArgLocs[I];
    if (VA.getLocInfo() == CCValAssign::Indirect)
      return false;
    if (!VA.isRegLoc())
      return false;
    unsigned Reg = VA.getLocReg();
    if (Reg == SystemZ::R6H || Reg == SystemZ::R6L || Reg == SystemZ::R6D)
      return false;
  }
  return true;
}

SDValue
SystemZTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                 SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  MachineFunction &MF = DAG.getMachineFunction();
  EVT PtrVT = getPointerTy();

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState ArgCCInfo(CallConv, IsVarArg, MF, TM, ArgLocs, *DAG.getContext());
  ArgCCInfo.AnalyzeCallOperands(Outs, CC_SystemZ);

  // We don't support GuaranteedTailCallOpt, only automatically-detected
  // sibling calls.
  if (IsTailCall && !canUseSiblingCall(ArgCCInfo, ArgLocs))
    IsTailCall = false;

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = ArgCCInfo.getNextStackOffset();

  // Mark the start of the call.
  if (!IsTailCall)
    Chain = DAG.getCALLSEQ_START(Chain, DAG.getConstant(NumBytes, PtrVT, true),
                                 DL);

  // Copy argument values to their designated locations.
  SmallVector<std::pair<unsigned, SDValue>, 9> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    CCValAssign &VA = ArgLocs[I];
    SDValue ArgValue = OutVals[I];

    if (VA.getLocInfo() == CCValAssign::Indirect) {
      // Store the argument in a stack slot and pass its address.
      SDValue SpillSlot = DAG.CreateStackTemporary(VA.getValVT());
      int FI = cast<FrameIndexSDNode>(SpillSlot)->getIndex();
      MemOpChains.push_back(DAG.getStore(Chain, DL, ArgValue, SpillSlot,
                                         MachinePointerInfo::getFixedStack(FI),
                                         false, false, 0));
      ArgValue = SpillSlot;
    } else
      ArgValue = convertValVTToLocVT(DAG, DL, VA, ArgValue);

    if (VA.isRegLoc())
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    else {
      assert(VA.isMemLoc() && "Argument not register or memory");

      // Work out the address of the stack slot.  Unpromoted ints and
      // floats are passed as right-justified 8-byte values.
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, SystemZ::R15D, PtrVT);
      unsigned Offset = SystemZMC::CallFrameSize + VA.getLocMemOffset();
      if (VA.getLocVT() == MVT::i32 || VA.getLocVT() == MVT::f32)
        Offset += 4;
      SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                                    DAG.getIntPtrConstant(Offset));

      // Emit the store.
      MemOpChains.push_back(DAG.getStore(Chain, DL, ArgValue, Address,
                                         MachinePointerInfo(),
                                         false, false, 0));
    }
  }

  // Join the stores, which are independent of one another.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Accept direct calls by converting symbolic call addresses to the
  // associated Target* opcodes.  Force %r1 to be used for indirect
  // tail calls.
  SDValue Glue;
  if (auto *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), DL, PtrVT);
    Callee = DAG.getNode(SystemZISD::PCREL_WRAPPER, DL, PtrVT, Callee);
  } else if (auto *E = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), PtrVT);
    Callee = DAG.getNode(SystemZISD::PCREL_WRAPPER, DL, PtrVT, Callee);
  } else if (IsTailCall) {
    Chain = DAG.getCopyToReg(Chain, DL, SystemZ::R1D, Callee, Glue);
    Glue = Chain.getValue(1);
    Callee = DAG.getRegister(SystemZ::R1D, Callee.getValueType());
  }

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  for (unsigned I = 0, E = RegsToPass.size(); I != E; ++I) {
    Chain = DAG.getCopyToReg(Chain, DL, RegsToPass[I].first,
                             RegsToPass[I].second, Glue);
    Glue = Chain.getValue(1);
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (unsigned I = 0, E = RegsToPass.size(); I != E; ++I)
    Ops.push_back(DAG.getRegister(RegsToPass[I].first,
                                  RegsToPass[I].second.getValueType()));

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  // Emit the call.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  if (IsTailCall)
    return DAG.getNode(SystemZISD::SIBCALL, DL, NodeTys, &Ops[0], Ops.size());
  Chain = DAG.getNode(SystemZISD::CALL, DL, NodeTys, &Ops[0], Ops.size());
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(NumBytes, PtrVT, true),
                             DAG.getConstant(0, PtrVT, true),
                             Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, TM, RetLocs, *DAG.getContext());
  RetCCInfo.AnalyzeCallResult(Ins, RetCC_SystemZ);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];

    // Copy the value out, gluing the copy to the end of the call sequence.
    SDValue RetValue = DAG.getCopyFromReg(Chain, DL, VA.getLocReg(),
                                          VA.getLocVT(), Glue);
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    // Convert the value of the return register into the value that's
    // being returned.
    InVals.push_back(convertLocVTToValVT(DAG, DL, VA, Chain, RetValue));
  }

  return Chain;
}

SDValue
SystemZTargetLowering::LowerReturn(SDValue Chain,
                                   CallingConv::ID CallConv, bool IsVarArg,
                                   const SmallVectorImpl<ISD::OutputArg> &Outs,
                                   const SmallVectorImpl<SDValue> &OutVals,
                                   SDLoc DL, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();

  // Assign locations to each returned value.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, TM, RetLocs, *DAG.getContext());
  RetCCInfo.AnalyzeReturn(Outs, RetCC_SystemZ);

  // Quick exit for void returns
  if (RetLocs.empty())
    return DAG.getNode(SystemZISD::RET_FLAG, DL, MVT::Other, Chain);

  // Copy the result values into the output registers.
  SDValue Glue;
  SmallVector<SDValue, 4> RetOps;
  RetOps.push_back(Chain);
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];
    SDValue RetValue = OutVals[I];

    // Make the return register live on exit.
    assert(VA.isRegLoc() && "Can only return in registers!");

    // Promote the value as required.
    RetValue = convertValVTToLocVT(DAG, DL, VA, RetValue);

    // Chain and glue the copies together.
    unsigned Reg = VA.getLocReg();
    Chain = DAG.getCopyToReg(Chain, DL, Reg, RetValue, Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(Reg, VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(SystemZISD::RET_FLAG, DL, MVT::Other,
                     RetOps.data(), RetOps.size());
}

SDValue SystemZTargetLowering::
prepareVolatileOrAtomicLoad(SDValue Chain, SDLoc DL, SelectionDAG &DAG) const {
  return DAG.getNode(SystemZISD::SERIALIZE, DL, MVT::Other, Chain);
}

// CC is a comparison that will be implemented using an integer or
// floating-point comparison.  Return the condition code mask for
// a branch on true.  In the integer case, CCMASK_CMP_UO is set for
// unsigned comparisons and clear for signed ones.  In the floating-point
// case, CCMASK_CMP_UO has its normal mask meaning (unordered).
static unsigned CCMaskForCondCode(ISD::CondCode CC) {
#define CONV(X) \
  case ISD::SET##X: return SystemZ::CCMASK_CMP_##X; \
  case ISD::SETO##X: return SystemZ::CCMASK_CMP_##X; \
  case ISD::SETU##X: return SystemZ::CCMASK_CMP_UO | SystemZ::CCMASK_CMP_##X

  switch (CC) {
  default:
    llvm_unreachable("Invalid integer condition!");

  CONV(EQ);
  CONV(NE);
  CONV(GT);
  CONV(GE);
  CONV(LT);
  CONV(LE);

  case ISD::SETO:  return SystemZ::CCMASK_CMP_O;
  case ISD::SETUO: return SystemZ::CCMASK_CMP_UO;
  }
#undef CONV
}

// Return a sequence for getting a 1 from an IPM result when CC has a
// value in CCMask and a 0 when CC has a value in CCValid & ~CCMask.
// The handling of CC values outside CCValid doesn't matter.
static IPMConversion getIPMConversion(unsigned CCValid, unsigned CCMask) {
  // Deal with cases where the result can be taken directly from a bit
  // of the IPM result.
  if (CCMask == (CCValid & (SystemZ::CCMASK_1 | SystemZ::CCMASK_3)))
    return IPMConversion(0, 0, SystemZ::IPM_CC);
  if (CCMask == (CCValid & (SystemZ::CCMASK_2 | SystemZ::CCMASK_3)))
    return IPMConversion(0, 0, SystemZ::IPM_CC + 1);

  // Deal with cases where we can add a value to force the sign bit
  // to contain the right value.  Putting the bit in 31 means we can
  // use SRL rather than RISBG(L), and also makes it easier to get a
  // 0/-1 value, so it has priority over the other tests below.
  //
  // These sequences rely on the fact that the upper two bits of the
  // IPM result are zero.
  uint64_t TopBit = uint64_t(1) << 31;
  if (CCMask == (CCValid & SystemZ::CCMASK_0))
    return IPMConversion(0, -(1 << SystemZ::IPM_CC), 31);
  if (CCMask == (CCValid & (SystemZ::CCMASK_0 | SystemZ::CCMASK_1)))
    return IPMConversion(0, -(2 << SystemZ::IPM_CC), 31);
  if (CCMask == (CCValid & (SystemZ::CCMASK_0
                            | SystemZ::CCMASK_1
                            | SystemZ::CCMASK_2)))
    return IPMConversion(0, -(3 << SystemZ::IPM_CC), 31);
  if (CCMask == (CCValid & SystemZ::CCMASK_3))
    return IPMConversion(0, TopBit - (3 << SystemZ::IPM_CC), 31);
  if (CCMask == (CCValid & (SystemZ::CCMASK_1
                            | SystemZ::CCMASK_2
                            | SystemZ::CCMASK_3)))
    return IPMConversion(0, TopBit - (1 << SystemZ::IPM_CC), 31);

  // Next try inverting the value and testing a bit.  0/1 could be
  // handled this way too, but we dealt with that case above.
  if (CCMask == (CCValid & (SystemZ::CCMASK_0 | SystemZ::CCMASK_2)))
    return IPMConversion(-1, 0, SystemZ::IPM_CC);

  // Handle cases where adding a value forces a non-sign bit to contain
  // the right value.
  if (CCMask == (CCValid & (SystemZ::CCMASK_1 | SystemZ::CCMASK_2)))
    return IPMConversion(0, 1 << SystemZ::IPM_CC, SystemZ::IPM_CC + 1);
  if (CCMask == (CCValid & (SystemZ::CCMASK_0 | SystemZ::CCMASK_3)))
    return IPMConversion(0, -(1 << SystemZ::IPM_CC), SystemZ::IPM_CC + 1);

  // The remaining cases are 1, 2, 0/1/3 and 0/2/3.  All these are
  // can be done by inverting the low CC bit and applying one of the
  // sign-based extractions above.
  if (CCMask == (CCValid & SystemZ::CCMASK_1))
    return IPMConversion(1 << SystemZ::IPM_CC, -(1 << SystemZ::IPM_CC), 31);
  if (CCMask == (CCValid & SystemZ::CCMASK_2))
    return IPMConversion(1 << SystemZ::IPM_CC,
                         TopBit - (3 << SystemZ::IPM_CC), 31);
  if (CCMask == (CCValid & (SystemZ::CCMASK_0
                            | SystemZ::CCMASK_1
                            | SystemZ::CCMASK_3)))
    return IPMConversion(1 << SystemZ::IPM_CC, -(3 << SystemZ::IPM_CC), 31);
  if (CCMask == (CCValid & (SystemZ::CCMASK_0
                            | SystemZ::CCMASK_2
                            | SystemZ::CCMASK_3)))
    return IPMConversion(1 << SystemZ::IPM_CC,
                         TopBit - (1 << SystemZ::IPM_CC), 31);

  llvm_unreachable("Unexpected CC combination");
}

// If C can be converted to a comparison against zero, adjust the operands
// as necessary.
static void adjustZeroCmp(SelectionDAG &DAG, Comparison &C) {
  if (C.ICmpType == SystemZICMP::UnsignedOnly)
    return;

  auto *ConstOp1 = dyn_cast<ConstantSDNode>(C.Op1.getNode());
  if (!ConstOp1)
    return;

  int64_t Value = ConstOp1->getSExtValue();
  if ((Value == -1 && C.CCMask == SystemZ::CCMASK_CMP_GT) ||
      (Value == -1 && C.CCMask == SystemZ::CCMASK_CMP_LE) ||
      (Value == 1 && C.CCMask == SystemZ::CCMASK_CMP_LT) ||
      (Value == 1 && C.CCMask == SystemZ::CCMASK_CMP_GE)) {
    C.CCMask ^= SystemZ::CCMASK_CMP_EQ;
    C.Op1 = DAG.getConstant(0, C.Op1.getValueType());
  }
}

// If a comparison described by C is suitable for CLI(Y), CHHSI or CLHHSI,
// adjust the operands as necessary.
static void adjustSubwordCmp(SelectionDAG &DAG, Comparison &C) {
  // For us to make any changes, it must a comparison between a single-use
  // load and a constant.
  if (!C.Op0.hasOneUse() ||
      C.Op0.getOpcode() != ISD::LOAD ||
      C.Op1.getOpcode() != ISD::Constant)
    return;

  // We must have an 8- or 16-bit load.
  auto *Load = cast<LoadSDNode>(C.Op0);
  unsigned NumBits = Load->getMemoryVT().getStoreSizeInBits();
  if (NumBits != 8 && NumBits != 16)
    return;

  // The load must be an extending one and the constant must be within the
  // range of the unextended value.
  auto *ConstOp1 = cast<ConstantSDNode>(C.Op1);
  uint64_t Value = ConstOp1->getZExtValue();
  uint64_t Mask = (1 << NumBits) - 1;
  if (Load->getExtensionType() == ISD::SEXTLOAD) {
    // Make sure that ConstOp1 is in range of C.Op0.
    int64_t SignedValue = ConstOp1->getSExtValue();
    if (uint64_t(SignedValue) + (uint64_t(1) << (NumBits - 1)) > Mask)
      return;
    if (C.ICmpType != SystemZICMP::SignedOnly) {
      // Unsigned comparison between two sign-extended values is equivalent
      // to unsigned comparison between two zero-extended values.
      Value &= Mask;
    } else if (NumBits == 8) {
      // Try to treat the comparison as unsigned, so that we can use CLI.
      // Adjust CCMask and Value as necessary.
      if (Value == 0 && C.CCMask == SystemZ::CCMASK_CMP_LT)
        // Test whether the high bit of the byte is set.
        Value = 127, C.CCMask = SystemZ::CCMASK_CMP_GT;
      else if (Value == 0 && C.CCMask == SystemZ::CCMASK_CMP_GE)
        // Test whether the high bit of the byte is clear.
        Value = 128, C.CCMask = SystemZ::CCMASK_CMP_LT;
      else
        // No instruction exists for this combination.
        return;
      C.ICmpType = SystemZICMP::UnsignedOnly;
    }
  } else if (Load->getExtensionType() == ISD::ZEXTLOAD) {
    if (Value > Mask)
      return;
    assert(C.ICmpType == SystemZICMP::Any &&
           "Signedness shouldn't matter here.");
  } else
    return;

  // Make sure that the first operand is an i32 of the right extension type.
  ISD::LoadExtType ExtType = (C.ICmpType == SystemZICMP::SignedOnly ?
                              ISD::SEXTLOAD :
                              ISD::ZEXTLOAD);
  if (C.Op0.getValueType() != MVT::i32 ||
      Load->getExtensionType() != ExtType)
    C.Op0 = DAG.getExtLoad(ExtType, SDLoc(Load), MVT::i32,
                           Load->getChain(), Load->getBasePtr(),
                           Load->getPointerInfo(), Load->getMemoryVT(),
                           Load->isVolatile(), Load->isNonTemporal(),
                           Load->getAlignment());

  // Make sure that the second operand is an i32 with the right value.
  if (C.Op1.getValueType() != MVT::i32 ||
      Value != ConstOp1->getZExtValue())
    C.Op1 = DAG.getConstant(Value, MVT::i32);
}

// Return true if Op is either an unextended load, or a load suitable
// for integer register-memory comparisons of type ICmpType.
static bool isNaturalMemoryOperand(SDValue Op, unsigned ICmpType) {
  auto *Load = dyn_cast<LoadSDNode>(Op.getNode());
  if (Load) {
    // There are no instructions to compare a register with a memory byte.
    if (Load->getMemoryVT() == MVT::i8)
      return false;
    // Otherwise decide on extension type.
    switch (Load->getExtensionType()) {
    case ISD::NON_EXTLOAD:
      return true;
    case ISD::SEXTLOAD:
      return ICmpType != SystemZICMP::UnsignedOnly;
    case ISD::ZEXTLOAD:
      return ICmpType != SystemZICMP::SignedOnly;
    default:
      break;
    }
  }
  return false;
}

// Return true if it is better to swap the operands of C.
static bool shouldSwapCmpOperands(const Comparison &C) {
  // Leave f128 comparisons alone, since they have no memory forms.
  if (C.Op0.getValueType() == MVT::f128)
    return false;

  // Always keep a floating-point constant second, since comparisons with
  // zero can use LOAD TEST and comparisons with other constants make a
  // natural memory operand.
  if (isa<ConstantFPSDNode>(C.Op1))
    return false;

  // Never swap comparisons with zero since there are many ways to optimize
  // those later.
  auto *ConstOp1 = dyn_cast<ConstantSDNode>(C.Op1);
  if (ConstOp1 && ConstOp1->getZExtValue() == 0)
    return false;

  // Also keep natural memory operands second if the loaded value is
  // only used here.  Several comparisons have memory forms.
  if (isNaturalMemoryOperand(C.Op1, C.ICmpType) && C.Op1.hasOneUse())
    return false;

  // Look for cases where Cmp0 is a single-use load and Cmp1 isn't.
  // In that case we generally prefer the memory to be second.
  if (isNaturalMemoryOperand(C.Op0, C.ICmpType) && C.Op0.hasOneUse()) {
    // The only exceptions are when the second operand is a constant and
    // we can use things like CHHSI.
    if (!ConstOp1)
      return true;
    // The unsigned memory-immediate instructions can handle 16-bit
    // unsigned integers.
    if (C.ICmpType != SystemZICMP::SignedOnly &&
        isUInt<16>(ConstOp1->getZExtValue()))
      return false;
    // The signed memory-immediate instructions can handle 16-bit
    // signed integers.
    if (C.ICmpType != SystemZICMP::UnsignedOnly &&
        isInt<16>(ConstOp1->getSExtValue()))
      return false;
    return true;
  }

  // Try to promote the use of CGFR and CLGFR.
  unsigned Opcode0 = C.Op0.getOpcode();
  if (C.ICmpType != SystemZICMP::UnsignedOnly && Opcode0 == ISD::SIGN_EXTEND)
    return true;
  if (C.ICmpType != SystemZICMP::SignedOnly && Opcode0 == ISD::ZERO_EXTEND)
    return true;
  if (C.ICmpType != SystemZICMP::SignedOnly &&
      Opcode0 == ISD::AND &&
      C.Op0.getOperand(1).getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(C.Op0.getOperand(1))->getZExtValue() == 0xffffffff)
    return true;

  return false;
}

// Return a version of comparison CC mask CCMask in which the LT and GT
// actions are swapped.
static unsigned reverseCCMask(unsigned CCMask) {
  return ((CCMask & SystemZ::CCMASK_CMP_EQ) |
          (CCMask & SystemZ::CCMASK_CMP_GT ? SystemZ::CCMASK_CMP_LT : 0) |
          (CCMask & SystemZ::CCMASK_CMP_LT ? SystemZ::CCMASK_CMP_GT : 0) |
          (CCMask & SystemZ::CCMASK_CMP_UO));
}

// Check whether C tests for equality between X and Y and whether X - Y
// or Y - X is also computed.  In that case it's better to compare the
// result of the subtraction against zero.
static void adjustForSubtraction(SelectionDAG &DAG, Comparison &C) {
  if (C.CCMask == SystemZ::CCMASK_CMP_EQ ||
      C.CCMask == SystemZ::CCMASK_CMP_NE) {
    for (auto I = C.Op0->use_begin(), E = C.Op0->use_end(); I != E; ++I) {
      SDNode *N = *I;
      if (N->getOpcode() == ISD::SUB &&
          ((N->getOperand(0) == C.Op0 && N->getOperand(1) == C.Op1) ||
           (N->getOperand(0) == C.Op1 && N->getOperand(1) == C.Op0))) {
        C.Op0 = SDValue(N, 0);
        C.Op1 = DAG.getConstant(0, N->getValueType(0));
        return;
      }
    }
  }
}

// Check whether C compares a floating-point value with zero and if that
// floating-point value is also negated.  In this case we can use the
// negation to set CC, so avoiding separate LOAD AND TEST and
// LOAD (NEGATIVE/COMPLEMENT) instructions.
static void adjustForFNeg(Comparison &C) {
  auto *C1 = dyn_cast<ConstantFPSDNode>(C.Op1);
  if (C1 && C1->isZero()) {
    for (auto I = C.Op0->use_begin(), E = C.Op0->use_end(); I != E; ++I) {
      SDNode *N = *I;
      if (N->getOpcode() == ISD::FNEG) {
        C.Op0 = SDValue(N, 0);
        C.CCMask = reverseCCMask(C.CCMask);
        return;
      }
    }
  }
}

// Check whether C compares (shl X, 32) with 0 and whether X is
// also sign-extended.  In that case it is better to test the result
// of the sign extension using LTGFR.
//
// This case is important because InstCombine transforms a comparison
// with (sext (trunc X)) into a comparison with (shl X, 32).
static void adjustForLTGFR(Comparison &C) {
  // Check for a comparison between (shl X, 32) and 0.
  if (C.Op0.getOpcode() == ISD::SHL &&
      C.Op0.getValueType() == MVT::i64 &&
      C.Op1.getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(C.Op1)->getZExtValue() == 0) {
    auto *C1 = dyn_cast<ConstantSDNode>(C.Op0.getOperand(1));
    if (C1 && C1->getZExtValue() == 32) {
      SDValue ShlOp0 = C.Op0.getOperand(0);
      // See whether X has any SIGN_EXTEND_INREG uses.
      for (auto I = ShlOp0->use_begin(), E = ShlOp0->use_end(); I != E; ++I) {
        SDNode *N = *I;
        if (N->getOpcode() == ISD::SIGN_EXTEND_INREG &&
            cast<VTSDNode>(N->getOperand(1))->getVT() == MVT::i32) {
          C.Op0 = SDValue(N, 0);
          return;
        }
      }
    }
  }
}

// If C compares the truncation of an extending load, try to compare
// the untruncated value instead.  This exposes more opportunities to
// reuse CC.
static void adjustICmpTruncate(SelectionDAG &DAG, Comparison &C) {
  if (C.Op0.getOpcode() == ISD::TRUNCATE &&
      C.Op0.getOperand(0).getOpcode() == ISD::LOAD &&
      C.Op1.getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(C.Op1)->getZExtValue() == 0) {
    auto *L = cast<LoadSDNode>(C.Op0.getOperand(0));
    if (L->getMemoryVT().getStoreSizeInBits()
        <= C.Op0.getValueType().getSizeInBits()) {
      unsigned Type = L->getExtensionType();
      if ((Type == ISD::ZEXTLOAD && C.ICmpType != SystemZICMP::SignedOnly) ||
          (Type == ISD::SEXTLOAD && C.ICmpType != SystemZICMP::UnsignedOnly)) {
        C.Op0 = C.Op0.getOperand(0);
        C.Op1 = DAG.getConstant(0, C.Op0.getValueType());
      }
    }
  }
}

// Return true if shift operation N has an in-range constant shift value.
// Store it in ShiftVal if so.
static bool isSimpleShift(SDValue N, unsigned &ShiftVal) {
  auto *Shift = dyn_cast<ConstantSDNode>(N.getOperand(1));
  if (!Shift)
    return false;

  uint64_t Amount = Shift->getZExtValue();
  if (Amount >= N.getValueType().getSizeInBits())
    return false;

  ShiftVal = Amount;
  return true;
}

// Check whether an AND with Mask is suitable for a TEST UNDER MASK
// instruction and whether the CC value is descriptive enough to handle
// a comparison of type Opcode between the AND result and CmpVal.
// CCMask says which comparison result is being tested and BitSize is
// the number of bits in the operands.  If TEST UNDER MASK can be used,
// return the corresponding CC mask, otherwise return 0.
static unsigned getTestUnderMaskCond(unsigned BitSize, unsigned CCMask,
                                     uint64_t Mask, uint64_t CmpVal,
                                     unsigned ICmpType) {
  assert(Mask != 0 && "ANDs with zero should have been removed by now");

  // Check whether the mask is suitable for TMHH, TMHL, TMLH or TMLL.
  if (!SystemZ::isImmLL(Mask) && !SystemZ::isImmLH(Mask) &&
      !SystemZ::isImmHL(Mask) && !SystemZ::isImmHH(Mask))
    return 0;

  // Work out the masks for the lowest and highest bits.
  unsigned HighShift = 63 - countLeadingZeros(Mask);
  uint64_t High = uint64_t(1) << HighShift;
  uint64_t Low = uint64_t(1) << countTrailingZeros(Mask);

  // Signed ordered comparisons are effectively unsigned if the sign
  // bit is dropped.
  bool EffectivelyUnsigned = (ICmpType != SystemZICMP::SignedOnly);

  // Check for equality comparisons with 0, or the equivalent.
  if (CmpVal == 0) {
    if (CCMask == SystemZ::CCMASK_CMP_EQ)
      return SystemZ::CCMASK_TM_ALL_0;
    if (CCMask == SystemZ::CCMASK_CMP_NE)
      return SystemZ::CCMASK_TM_SOME_1;
  }
  if (EffectivelyUnsigned && CmpVal <= Low) {
    if (CCMask == SystemZ::CCMASK_CMP_LT)
      return SystemZ::CCMASK_TM_ALL_0;
    if (CCMask == SystemZ::CCMASK_CMP_GE)
      return SystemZ::CCMASK_TM_SOME_1;
  }
  if (EffectivelyUnsigned && CmpVal < Low) {
    if (CCMask == SystemZ::CCMASK_CMP_LE)
      return SystemZ::CCMASK_TM_ALL_0;
    if (CCMask == SystemZ::CCMASK_CMP_GT)
      return SystemZ::CCMASK_TM_SOME_1;
  }

  // Check for equality comparisons with the mask, or the equivalent.
  if (CmpVal == Mask) {
    if (CCMask == SystemZ::CCMASK_CMP_EQ)
      return SystemZ::CCMASK_TM_ALL_1;
    if (CCMask == SystemZ::CCMASK_CMP_NE)
      return SystemZ::CCMASK_TM_SOME_0;
  }
  if (EffectivelyUnsigned && CmpVal >= Mask - Low && CmpVal < Mask) {
    if (CCMask == SystemZ::CCMASK_CMP_GT)
      return SystemZ::CCMASK_TM_ALL_1;
    if (CCMask == SystemZ::CCMASK_CMP_LE)
      return SystemZ::CCMASK_TM_SOME_0;
  }
  if (EffectivelyUnsigned && CmpVal > Mask - Low && CmpVal <= Mask) {
    if (CCMask == SystemZ::CCMASK_CMP_GE)
      return SystemZ::CCMASK_TM_ALL_1;
    if (CCMask == SystemZ::CCMASK_CMP_LT)
      return SystemZ::CCMASK_TM_SOME_0;
  }

  // Check for ordered comparisons with the top bit.
  if (EffectivelyUnsigned && CmpVal >= Mask - High && CmpVal < High) {
    if (CCMask == SystemZ::CCMASK_CMP_LE)
      return SystemZ::CCMASK_TM_MSB_0;
    if (CCMask == SystemZ::CCMASK_CMP_GT)
      return SystemZ::CCMASK_TM_MSB_1;
  }
  if (EffectivelyUnsigned && CmpVal > Mask - High && CmpVal <= High) {
    if (CCMask == SystemZ::CCMASK_CMP_LT)
      return SystemZ::CCMASK_TM_MSB_0;
    if (CCMask == SystemZ::CCMASK_CMP_GE)
      return SystemZ::CCMASK_TM_MSB_1;
  }

  // If there are just two bits, we can do equality checks for Low and High
  // as well.
  if (Mask == Low + High) {
    if (CCMask == SystemZ::CCMASK_CMP_EQ && CmpVal == Low)
      return SystemZ::CCMASK_TM_MIXED_MSB_0;
    if (CCMask == SystemZ::CCMASK_CMP_NE && CmpVal == Low)
      return SystemZ::CCMASK_TM_MIXED_MSB_0 ^ SystemZ::CCMASK_ANY;
    if (CCMask == SystemZ::CCMASK_CMP_EQ && CmpVal == High)
      return SystemZ::CCMASK_TM_MIXED_MSB_1;
    if (CCMask == SystemZ::CCMASK_CMP_NE && CmpVal == High)
      return SystemZ::CCMASK_TM_MIXED_MSB_1 ^ SystemZ::CCMASK_ANY;
  }

  // Looks like we've exhausted our options.
  return 0;
}

// See whether C can be implemented as a TEST UNDER MASK instruction.
// Update the arguments with the TM version if so.
static void adjustForTestUnderMask(SelectionDAG &DAG, Comparison &C) {
  // Check that we have a comparison with a constant.
  auto *ConstOp1 = dyn_cast<ConstantSDNode>(C.Op1);
  if (!ConstOp1)
    return;
  uint64_t CmpVal = ConstOp1->getZExtValue();

  // Check whether the nonconstant input is an AND with a constant mask.
  Comparison NewC(C);
  uint64_t MaskVal;
  ConstantSDNode *Mask = 0;
  if (C.Op0.getOpcode() == ISD::AND) {
    NewC.Op0 = C.Op0.getOperand(0);
    NewC.Op1 = C.Op0.getOperand(1);
    Mask = dyn_cast<ConstantSDNode>(NewC.Op1);
    if (!Mask)
      return;
    MaskVal = Mask->getZExtValue();
  } else {
    // There is no instruction to compare with a 64-bit immediate
    // so use TMHH instead if possible.  We need an unsigned ordered
    // comparison with an i64 immediate.
    if (NewC.Op0.getValueType() != MVT::i64 ||
        NewC.CCMask == SystemZ::CCMASK_CMP_EQ ||
        NewC.CCMask == SystemZ::CCMASK_CMP_NE ||
        NewC.ICmpType == SystemZICMP::SignedOnly)
      return;
    // Convert LE and GT comparisons into LT and GE.
    if (NewC.CCMask == SystemZ::CCMASK_CMP_LE ||
        NewC.CCMask == SystemZ::CCMASK_CMP_GT) {
      if (CmpVal == uint64_t(-1))
        return;
      CmpVal += 1;
      NewC.CCMask ^= SystemZ::CCMASK_CMP_EQ;
    }
    // If the low N bits of Op1 are zero than the low N bits of Op0 can
    // be masked off without changing the result.
    MaskVal = -(CmpVal & -CmpVal);
    NewC.ICmpType = SystemZICMP::UnsignedOnly;
  }

  // Check whether the combination of mask, comparison value and comparison
  // type are suitable.
  unsigned BitSize = NewC.Op0.getValueType().getSizeInBits();
  unsigned NewCCMask, ShiftVal;
  if (NewC.ICmpType != SystemZICMP::SignedOnly &&
      NewC.Op0.getOpcode() == ISD::SHL &&
      isSimpleShift(NewC.Op0, ShiftVal) &&
      (NewCCMask = getTestUnderMaskCond(BitSize, NewC.CCMask,
                                        MaskVal >> ShiftVal,
                                        CmpVal >> ShiftVal,
                                        SystemZICMP::Any))) {
    NewC.Op0 = NewC.Op0.getOperand(0);
    MaskVal >>= ShiftVal;
  } else if (NewC.ICmpType != SystemZICMP::SignedOnly &&
             NewC.Op0.getOpcode() == ISD::SRL &&
             isSimpleShift(NewC.Op0, ShiftVal) &&
             (NewCCMask = getTestUnderMaskCond(BitSize, NewC.CCMask,
                                               MaskVal << ShiftVal,
                                               CmpVal << ShiftVal,
                                               SystemZICMP::UnsignedOnly))) {
    NewC.Op0 = NewC.Op0.getOperand(0);
    MaskVal <<= ShiftVal;
  } else {
    NewCCMask = getTestUnderMaskCond(BitSize, NewC.CCMask, MaskVal, CmpVal,
                                     NewC.ICmpType);
    if (!NewCCMask)
      return;
  }

  // Go ahead and make the change.
  C.Opcode = SystemZISD::TM;
  C.Op0 = NewC.Op0;
  if (Mask && Mask->getZExtValue() == MaskVal)
    C.Op1 = SDValue(Mask, 0);
  else
    C.Op1 = DAG.getConstant(MaskVal, C.Op0.getValueType());
  C.CCValid = SystemZ::CCMASK_TM;
  C.CCMask = NewCCMask;
}

// Decide how to implement a comparison of type Cond between CmpOp0 with CmpOp1.
static Comparison getCmp(SelectionDAG &DAG, SDValue CmpOp0, SDValue CmpOp1,
                         ISD::CondCode Cond) {
  Comparison C(CmpOp0, CmpOp1);
  C.CCMask = CCMaskForCondCode(Cond);
  if (C.Op0.getValueType().isFloatingPoint()) {
    C.CCValid = SystemZ::CCMASK_FCMP;
    C.Opcode = SystemZISD::FCMP;
    adjustForFNeg(C);
  } else {
    C.CCValid = SystemZ::CCMASK_ICMP;
    C.Opcode = SystemZISD::ICMP;
    // Choose the type of comparison.  Equality and inequality tests can
    // use either signed or unsigned comparisons.  The choice also doesn't
    // matter if both sign bits are known to be clear.  In those cases we
    // want to give the main isel code the freedom to choose whichever
    // form fits best.
    if (C.CCMask == SystemZ::CCMASK_CMP_EQ ||
        C.CCMask == SystemZ::CCMASK_CMP_NE ||
        (DAG.SignBitIsZero(C.Op0) && DAG.SignBitIsZero(C.Op1)))
      C.ICmpType = SystemZICMP::Any;
    else if (C.CCMask & SystemZ::CCMASK_CMP_UO)
      C.ICmpType = SystemZICMP::UnsignedOnly;
    else
      C.ICmpType = SystemZICMP::SignedOnly;
    C.CCMask &= ~SystemZ::CCMASK_CMP_UO;
    adjustZeroCmp(DAG, C);
    adjustSubwordCmp(DAG, C);
    adjustForSubtraction(DAG, C);
    adjustForLTGFR(C);
    adjustICmpTruncate(DAG, C);
  }

  if (shouldSwapCmpOperands(C)) {
    std::swap(C.Op0, C.Op1);
    C.CCMask = reverseCCMask(C.CCMask);
  }

  adjustForTestUnderMask(DAG, C);
  return C;
}

// Emit the comparison instruction described by C.
static SDValue emitCmp(SelectionDAG &DAG, SDLoc DL, Comparison &C) {
  if (C.Opcode == SystemZISD::ICMP)
    return DAG.getNode(SystemZISD::ICMP, DL, MVT::Glue, C.Op0, C.Op1,
                       DAG.getConstant(C.ICmpType, MVT::i32));
  if (C.Opcode == SystemZISD::TM) {
    bool RegisterOnly = (bool(C.CCMask & SystemZ::CCMASK_TM_MIXED_MSB_0) !=
                         bool(C.CCMask & SystemZ::CCMASK_TM_MIXED_MSB_1));
    return DAG.getNode(SystemZISD::TM, DL, MVT::Glue, C.Op0, C.Op1,
                       DAG.getConstant(RegisterOnly, MVT::i32));
  }
  return DAG.getNode(C.Opcode, DL, MVT::Glue, C.Op0, C.Op1);
}

// Implement a 32-bit *MUL_LOHI operation by extending both operands to
// 64 bits.  Extend is the extension type to use.  Store the high part
// in Hi and the low part in Lo.
static void lowerMUL_LOHI32(SelectionDAG &DAG, SDLoc DL,
                            unsigned Extend, SDValue Op0, SDValue Op1,
                            SDValue &Hi, SDValue &Lo) {
  Op0 = DAG.getNode(Extend, DL, MVT::i64, Op0);
  Op1 = DAG.getNode(Extend, DL, MVT::i64, Op1);
  SDValue Mul = DAG.getNode(ISD::MUL, DL, MVT::i64, Op0, Op1);
  Hi = DAG.getNode(ISD::SRL, DL, MVT::i64, Mul, DAG.getConstant(32, MVT::i64));
  Hi = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Hi);
  Lo = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Mul);
}

// Lower a binary operation that produces two VT results, one in each
// half of a GR128 pair.  Op0 and Op1 are the VT operands to the operation,
// Extend extends Op0 to a GR128, and Opcode performs the GR128 operation
// on the extended Op0 and (unextended) Op1.  Store the even register result
// in Even and the odd register result in Odd.
static void lowerGR128Binary(SelectionDAG &DAG, SDLoc DL, EVT VT,
                             unsigned Extend, unsigned Opcode,
                             SDValue Op0, SDValue Op1,
                             SDValue &Even, SDValue &Odd) {
  SDNode *In128 = DAG.getMachineNode(Extend, DL, MVT::Untyped, Op0);
  SDValue Result = DAG.getNode(Opcode, DL, MVT::Untyped,
                               SDValue(In128, 0), Op1);
  bool Is32Bit = is32Bit(VT);
  Even = DAG.getTargetExtractSubreg(SystemZ::even128(Is32Bit), DL, VT, Result);
  Odd = DAG.getTargetExtractSubreg(SystemZ::odd128(Is32Bit), DL, VT, Result);
}

// Return an i32 value that is 1 if the CC value produced by Glue is
// in the mask CCMask and 0 otherwise.  CC is known to have a value
// in CCValid, so other values can be ignored.
static SDValue emitSETCC(SelectionDAG &DAG, SDLoc DL, SDValue Glue,
                         unsigned CCValid, unsigned CCMask) {
  IPMConversion Conversion = getIPMConversion(CCValid, CCMask);
  SDValue Result = DAG.getNode(SystemZISD::IPM, DL, MVT::i32, Glue);

  if (Conversion.XORValue)
    Result = DAG.getNode(ISD::XOR, DL, MVT::i32, Result,
                         DAG.getConstant(Conversion.XORValue, MVT::i32));

  if (Conversion.AddValue)
    Result = DAG.getNode(ISD::ADD, DL, MVT::i32, Result,
                         DAG.getConstant(Conversion.AddValue, MVT::i32));

  // The SHR/AND sequence should get optimized to an RISBG.
  Result = DAG.getNode(ISD::SRL, DL, MVT::i32, Result,
                       DAG.getConstant(Conversion.Bit, MVT::i32));
  if (Conversion.Bit != 31)
    Result = DAG.getNode(ISD::AND, DL, MVT::i32, Result,
                         DAG.getConstant(1, MVT::i32));
  return Result;
}

SDValue SystemZTargetLowering::lowerSETCC(SDValue Op,
                                          SelectionDAG &DAG) const {
  SDValue CmpOp0   = Op.getOperand(0);
  SDValue CmpOp1   = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  SDLoc DL(Op);

  Comparison C(getCmp(DAG, CmpOp0, CmpOp1, CC));
  SDValue Glue = emitCmp(DAG, DL, C);
  return emitSETCC(DAG, DL, Glue, C.CCValid, C.CCMask);
}

SDValue SystemZTargetLowering::lowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain    = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue CmpOp0   = Op.getOperand(2);
  SDValue CmpOp1   = Op.getOperand(3);
  SDValue Dest     = Op.getOperand(4);
  SDLoc DL(Op);

  Comparison C(getCmp(DAG, CmpOp0, CmpOp1, CC));
  SDValue Glue = emitCmp(DAG, DL, C);
  return DAG.getNode(SystemZISD::BR_CCMASK, DL, Op.getValueType(),
                     Chain, DAG.getConstant(C.CCValid, MVT::i32),
                     DAG.getConstant(C.CCMask, MVT::i32), Dest, Glue);
}

// Return true if Pos is CmpOp and Neg is the negative of CmpOp,
// allowing Pos and Neg to be wider than CmpOp.
static bool isAbsolute(SDValue CmpOp, SDValue Pos, SDValue Neg) {
  return (Neg.getOpcode() == ISD::SUB &&
          Neg.getOperand(0).getOpcode() == ISD::Constant &&
          cast<ConstantSDNode>(Neg.getOperand(0))->getZExtValue() == 0 &&
          Neg.getOperand(1) == Pos &&
          (Pos == CmpOp ||
           (Pos.getOpcode() == ISD::SIGN_EXTEND &&
            Pos.getOperand(0) == CmpOp)));
}

// Return the absolute or negative absolute of Op; IsNegative decides which.
static SDValue getAbsolute(SelectionDAG &DAG, SDLoc DL, SDValue Op,
                           bool IsNegative) {
  Op = DAG.getNode(SystemZISD::IABS, DL, Op.getValueType(), Op);
  if (IsNegative)
    Op = DAG.getNode(ISD::SUB, DL, Op.getValueType(),
                     DAG.getConstant(0, Op.getValueType()), Op);
  return Op;
}

SDValue SystemZTargetLowering::lowerSELECT_CC(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDValue CmpOp0   = Op.getOperand(0);
  SDValue CmpOp1   = Op.getOperand(1);
  SDValue TrueOp   = Op.getOperand(2);
  SDValue FalseOp  = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDLoc DL(Op);

  Comparison C(getCmp(DAG, CmpOp0, CmpOp1, CC));

  // Check for absolute and negative-absolute selections, including those
  // where the comparison value is sign-extended (for LPGFR and LNGFR).
  // This check supplements the one in DAGCombiner.
  if (C.Opcode == SystemZISD::ICMP &&
      C.CCMask != SystemZ::CCMASK_CMP_EQ &&
      C.CCMask != SystemZ::CCMASK_CMP_NE &&
      C.Op1.getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(C.Op1)->getZExtValue() == 0) {
    if (isAbsolute(C.Op0, TrueOp, FalseOp))
      return getAbsolute(DAG, DL, TrueOp, C.CCMask & SystemZ::CCMASK_CMP_LT);
    if (isAbsolute(C.Op0, FalseOp, TrueOp))
      return getAbsolute(DAG, DL, FalseOp, C.CCMask & SystemZ::CCMASK_CMP_GT);
  }

  SDValue Glue = emitCmp(DAG, DL, C);

  // Special case for handling -1/0 results.  The shifts we use here
  // should get optimized with the IPM conversion sequence.
  auto *TrueC = dyn_cast<ConstantSDNode>(TrueOp);
  auto *FalseC = dyn_cast<ConstantSDNode>(FalseOp);
  if (TrueC && FalseC) {
    int64_t TrueVal = TrueC->getSExtValue();
    int64_t FalseVal = FalseC->getSExtValue();
    if ((TrueVal == -1 && FalseVal == 0) || (TrueVal == 0 && FalseVal == -1)) {
      // Invert the condition if we want -1 on false.
      if (TrueVal == 0)
        C.CCMask ^= C.CCValid;
      SDValue Result = emitSETCC(DAG, DL, Glue, C.CCValid, C.CCMask);
      EVT VT = Op.getValueType();
      // Extend the result to VT.  Upper bits are ignored.
      if (!is32Bit(VT))
        Result = DAG.getNode(ISD::ANY_EXTEND, DL, VT, Result);
      // Sign-extend from the low bit.
      SDValue ShAmt = DAG.getConstant(VT.getSizeInBits() - 1, MVT::i32);
      SDValue Shl = DAG.getNode(ISD::SHL, DL, VT, Result, ShAmt);
      return DAG.getNode(ISD::SRA, DL, VT, Shl, ShAmt);
    }
  }

  SmallVector<SDValue, 5> Ops;
  Ops.push_back(TrueOp);
  Ops.push_back(FalseOp);
  Ops.push_back(DAG.getConstant(C.CCValid, MVT::i32));
  Ops.push_back(DAG.getConstant(C.CCMask, MVT::i32));
  Ops.push_back(Glue);

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  return DAG.getNode(SystemZISD::SELECT_CCMASK, DL, VTs, &Ops[0], Ops.size());
}

SDValue SystemZTargetLowering::lowerGlobalAddress(GlobalAddressSDNode *Node,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Node);
  const GlobalValue *GV = Node->getGlobal();
  int64_t Offset = Node->getOffset();
  EVT PtrVT = getPointerTy();
  Reloc::Model RM = TM.getRelocationModel();
  CodeModel::Model CM = TM.getCodeModel();

  SDValue Result;
  if (Subtarget.isPC32DBLSymbol(GV, RM, CM)) {
    // Assign anchors at 1<<12 byte boundaries.
    uint64_t Anchor = Offset & ~uint64_t(0xfff);
    Result = DAG.getTargetGlobalAddress(GV, DL, PtrVT, Anchor);
    Result = DAG.getNode(SystemZISD::PCREL_WRAPPER, DL, PtrVT, Result);

    // The offset can be folded into the address if it is aligned to a halfword.
    Offset -= Anchor;
    if (Offset != 0 && (Offset & 1) == 0) {
      SDValue Full = DAG.getTargetGlobalAddress(GV, DL, PtrVT, Anchor + Offset);
      Result = DAG.getNode(SystemZISD::PCREL_OFFSET, DL, PtrVT, Full, Result);
      Offset = 0;
    }
  } else {
    Result = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, SystemZII::MO_GOT);
    Result = DAG.getNode(SystemZISD::PCREL_WRAPPER, DL, PtrVT, Result);
    Result = DAG.getLoad(PtrVT, DL, DAG.getEntryNode(), Result,
                         MachinePointerInfo::getGOT(), false, false, false, 0);
  }

  // If there was a non-zero offset that we didn't fold, create an explicit
  // addition for it.
  if (Offset != 0)
    Result = DAG.getNode(ISD::ADD, DL, PtrVT, Result,
                         DAG.getConstant(Offset, PtrVT));

  return Result;
}

SDValue SystemZTargetLowering::lowerGlobalTLSAddress(GlobalAddressSDNode *Node,
						     SelectionDAG &DAG) const {
  SDLoc DL(Node);
  const GlobalValue *GV = Node->getGlobal();
  EVT PtrVT = getPointerTy();
  TLSModel::Model model = TM.getTLSModel(GV);

  if (model != TLSModel::LocalExec)
    llvm_unreachable("only local-exec TLS mode supported");

  // The high part of the thread pointer is in access register 0.
  SDValue TPHi = DAG.getNode(SystemZISD::EXTRACT_ACCESS, DL, MVT::i32,
                             DAG.getConstant(0, MVT::i32));
  TPHi = DAG.getNode(ISD::ANY_EXTEND, DL, PtrVT, TPHi);

  // The low part of the thread pointer is in access register 1.
  SDValue TPLo = DAG.getNode(SystemZISD::EXTRACT_ACCESS, DL, MVT::i32,
                             DAG.getConstant(1, MVT::i32));
  TPLo = DAG.getNode(ISD::ZERO_EXTEND, DL, PtrVT, TPLo);

  // Merge them into a single 64-bit address.
  SDValue TPHiShifted = DAG.getNode(ISD::SHL, DL, PtrVT, TPHi,
				    DAG.getConstant(32, PtrVT));
  SDValue TP = DAG.getNode(ISD::OR, DL, PtrVT, TPHiShifted, TPLo);

  // Get the offset of GA from the thread pointer.
  SystemZConstantPoolValue *CPV =
    SystemZConstantPoolValue::Create(GV, SystemZCP::NTPOFF);

  // Force the offset into the constant pool and load it from there.
  SDValue CPAddr = DAG.getConstantPool(CPV, PtrVT, 8);
  SDValue Offset = DAG.getLoad(PtrVT, DL, DAG.getEntryNode(),
			       CPAddr, MachinePointerInfo::getConstantPool(),
			       false, false, false, 0);

  // Add the base and offset together.
  return DAG.getNode(ISD::ADD, DL, PtrVT, TP, Offset);
}

SDValue SystemZTargetLowering::lowerBlockAddress(BlockAddressSDNode *Node,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Node);
  const BlockAddress *BA = Node->getBlockAddress();
  int64_t Offset = Node->getOffset();
  EVT PtrVT = getPointerTy();

  SDValue Result = DAG.getTargetBlockAddress(BA, PtrVT, Offset);
  Result = DAG.getNode(SystemZISD::PCREL_WRAPPER, DL, PtrVT, Result);
  return Result;
}

SDValue SystemZTargetLowering::lowerJumpTable(JumpTableSDNode *JT,
                                              SelectionDAG &DAG) const {
  SDLoc DL(JT);
  EVT PtrVT = getPointerTy();
  SDValue Result = DAG.getTargetJumpTable(JT->getIndex(), PtrVT);

  // Use LARL to load the address of the table.
  return DAG.getNode(SystemZISD::PCREL_WRAPPER, DL, PtrVT, Result);
}

SDValue SystemZTargetLowering::lowerConstantPool(ConstantPoolSDNode *CP,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(CP);
  EVT PtrVT = getPointerTy();

  SDValue Result;
  if (CP->isMachineConstantPoolEntry())
    Result = DAG.getTargetConstantPool(CP->getMachineCPVal(), PtrVT,
				       CP->getAlignment());
  else
    Result = DAG.getTargetConstantPool(CP->getConstVal(), PtrVT,
				       CP->getAlignment(), CP->getOffset());

  // Use LARL to load the address of the constant pool entry.
  return DAG.getNode(SystemZISD::PCREL_WRAPPER, DL, PtrVT, Result);
}

SDValue SystemZTargetLowering::lowerBITCAST(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue In = Op.getOperand(0);
  EVT InVT = In.getValueType();
  EVT ResVT = Op.getValueType();

  if (InVT == MVT::i32 && ResVT == MVT::f32) {
    SDValue In64;
    if (Subtarget.hasHighWord()) {
      SDNode *U64 = DAG.getMachineNode(TargetOpcode::IMPLICIT_DEF, DL,
                                       MVT::i64);
      In64 = DAG.getTargetInsertSubreg(SystemZ::subreg_h32, DL,
                                       MVT::i64, SDValue(U64, 0), In);
    } else {
      In64 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, In);
      In64 = DAG.getNode(ISD::SHL, DL, MVT::i64, In64,
                         DAG.getConstant(32, MVT::i64));
    }
    SDValue Out64 = DAG.getNode(ISD::BITCAST, DL, MVT::f64, In64);
    return DAG.getTargetExtractSubreg(SystemZ::subreg_h32,
                                      DL, MVT::f32, Out64);
  }
  if (InVT == MVT::f32 && ResVT == MVT::i32) {
    SDNode *U64 = DAG.getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, MVT::f64);
    SDValue In64 = DAG.getTargetInsertSubreg(SystemZ::subreg_h32, DL,
                                             MVT::f64, SDValue(U64, 0), In);
    SDValue Out64 = DAG.getNode(ISD::BITCAST, DL, MVT::i64, In64);
    if (Subtarget.hasHighWord())
      return DAG.getTargetExtractSubreg(SystemZ::subreg_h32, DL,
                                        MVT::i32, Out64);
    SDValue Shift = DAG.getNode(ISD::SRL, DL, MVT::i64, Out64,
                                DAG.getConstant(32, MVT::i64));
    return DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Shift);
  }
  llvm_unreachable("Unexpected bitcast combination");
}

SDValue SystemZTargetLowering::lowerVASTART(SDValue Op,
                                            SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  SystemZMachineFunctionInfo *FuncInfo =
    MF.getInfo<SystemZMachineFunctionInfo>();
  EVT PtrVT = getPointerTy();

  SDValue Chain   = Op.getOperand(0);
  SDValue Addr    = Op.getOperand(1);
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  SDLoc DL(Op);

  // The initial values of each field.
  const unsigned NumFields = 4;
  SDValue Fields[NumFields] = {
    DAG.getConstant(FuncInfo->getVarArgsFirstGPR(), PtrVT),
    DAG.getConstant(FuncInfo->getVarArgsFirstFPR(), PtrVT),
    DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(), PtrVT),
    DAG.getFrameIndex(FuncInfo->getRegSaveFrameIndex(), PtrVT)
  };

  // Store each field into its respective slot.
  SDValue MemOps[NumFields];
  unsigned Offset = 0;
  for (unsigned I = 0; I < NumFields; ++I) {
    SDValue FieldAddr = Addr;
    if (Offset != 0)
      FieldAddr = DAG.getNode(ISD::ADD, DL, PtrVT, FieldAddr,
                              DAG.getIntPtrConstant(Offset));
    MemOps[I] = DAG.getStore(Chain, DL, Fields[I], FieldAddr,
                             MachinePointerInfo(SV, Offset),
                             false, false, 0);
    Offset += 8;
  }
  return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOps, NumFields);
}

SDValue SystemZTargetLowering::lowerVACOPY(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDValue Chain      = Op.getOperand(0);
  SDValue DstPtr     = Op.getOperand(1);
  SDValue SrcPtr     = Op.getOperand(2);
  const Value *DstSV = cast<SrcValueSDNode>(Op.getOperand(3))->getValue();
  const Value *SrcSV = cast<SrcValueSDNode>(Op.getOperand(4))->getValue();
  SDLoc DL(Op);

  return DAG.getMemcpy(Chain, DL, DstPtr, SrcPtr, DAG.getIntPtrConstant(32),
                       /*Align*/8, /*isVolatile*/false, /*AlwaysInline*/false,
                       MachinePointerInfo(DstSV), MachinePointerInfo(SrcSV));
}

SDValue SystemZTargetLowering::
lowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Size  = Op.getOperand(1);
  SDLoc DL(Op);

  unsigned SPReg = getStackPointerRegisterToSaveRestore();

  // Get a reference to the stack pointer.
  SDValue OldSP = DAG.getCopyFromReg(Chain, DL, SPReg, MVT::i64);

  // Get the new stack pointer value.
  SDValue NewSP = DAG.getNode(ISD::SUB, DL, MVT::i64, OldSP, Size);

  // Copy the new stack pointer back.
  Chain = DAG.getCopyToReg(Chain, DL, SPReg, NewSP);

  // The allocated data lives above the 160 bytes allocated for the standard
  // frame, plus any outgoing stack arguments.  We don't know how much that
  // amounts to yet, so emit a special ADJDYNALLOC placeholder.
  SDValue ArgAdjust = DAG.getNode(SystemZISD::ADJDYNALLOC, DL, MVT::i64);
  SDValue Result = DAG.getNode(ISD::ADD, DL, MVT::i64, NewSP, ArgAdjust);

  SDValue Ops[2] = { Result, Chain };
  return DAG.getMergeValues(Ops, 2, DL);
}

SDValue SystemZTargetLowering::lowerSMUL_LOHI(SDValue Op,
                                              SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  SDValue Ops[2];
  if (is32Bit(VT))
    // Just do a normal 64-bit multiplication and extract the results.
    // We define this so that it can be used for constant division.
    lowerMUL_LOHI32(DAG, DL, ISD::SIGN_EXTEND, Op.getOperand(0),
                    Op.getOperand(1), Ops[1], Ops[0]);
  else {
    // Do a full 128-bit multiplication based on UMUL_LOHI64:
    //
    //   (ll * rl) + ((lh * rl) << 64) + ((ll * rh) << 64)
    //
    // but using the fact that the upper halves are either all zeros
    // or all ones:
    //
    //   (ll * rl) - ((lh & rl) << 64) - ((ll & rh) << 64)
    //
    // and grouping the right terms together since they are quicker than the
    // multiplication:
    //
    //   (ll * rl) - (((lh & rl) + (ll & rh)) << 64)
    SDValue C63 = DAG.getConstant(63, MVT::i64);
    SDValue LL = Op.getOperand(0);
    SDValue RL = Op.getOperand(1);
    SDValue LH = DAG.getNode(ISD::SRA, DL, VT, LL, C63);
    SDValue RH = DAG.getNode(ISD::SRA, DL, VT, RL, C63);
    // UMUL_LOHI64 returns the low result in the odd register and the high
    // result in the even register.  SMUL_LOHI is defined to return the
    // low half first, so the results are in reverse order.
    lowerGR128Binary(DAG, DL, VT, SystemZ::AEXT128_64, SystemZISD::UMUL_LOHI64,
                     LL, RL, Ops[1], Ops[0]);
    SDValue NegLLTimesRH = DAG.getNode(ISD::AND, DL, VT, LL, RH);
    SDValue NegLHTimesRL = DAG.getNode(ISD::AND, DL, VT, LH, RL);
    SDValue NegSum = DAG.getNode(ISD::ADD, DL, VT, NegLLTimesRH, NegLHTimesRL);
    Ops[1] = DAG.getNode(ISD::SUB, DL, VT, Ops[1], NegSum);
  }
  return DAG.getMergeValues(Ops, 2, DL);
}

SDValue SystemZTargetLowering::lowerUMUL_LOHI(SDValue Op,
                                              SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  SDValue Ops[2];
  if (is32Bit(VT))
    // Just do a normal 64-bit multiplication and extract the results.
    // We define this so that it can be used for constant division.
    lowerMUL_LOHI32(DAG, DL, ISD::ZERO_EXTEND, Op.getOperand(0),
                    Op.getOperand(1), Ops[1], Ops[0]);
  else
    // UMUL_LOHI64 returns the low result in the odd register and the high
    // result in the even register.  UMUL_LOHI is defined to return the
    // low half first, so the results are in reverse order.
    lowerGR128Binary(DAG, DL, VT, SystemZ::AEXT128_64, SystemZISD::UMUL_LOHI64,
                     Op.getOperand(0), Op.getOperand(1), Ops[1], Ops[0]);
  return DAG.getMergeValues(Ops, 2, DL);
}

SDValue SystemZTargetLowering::lowerSDIVREM(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned Opcode;

  // We use DSGF for 32-bit division.
  if (is32Bit(VT)) {
    Op0 = DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, Op0);
    Opcode = SystemZISD::SDIVREM32;
  } else if (DAG.ComputeNumSignBits(Op1) > 32) {
    Op1 = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Op1);
    Opcode = SystemZISD::SDIVREM32;
  } else    
    Opcode = SystemZISD::SDIVREM64;

  // DSG(F) takes a 64-bit dividend, so the even register in the GR128
  // input is "don't care".  The instruction returns the remainder in
  // the even register and the quotient in the odd register.
  SDValue Ops[2];
  lowerGR128Binary(DAG, DL, VT, SystemZ::AEXT128_64, Opcode,
                   Op0, Op1, Ops[1], Ops[0]);
  return DAG.getMergeValues(Ops, 2, DL);
}

SDValue SystemZTargetLowering::lowerUDIVREM(SDValue Op,
                                            SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  // DL(G) uses a double-width dividend, so we need to clear the even
  // register in the GR128 input.  The instruction returns the remainder
  // in the even register and the quotient in the odd register.
  SDValue Ops[2];
  if (is32Bit(VT))
    lowerGR128Binary(DAG, DL, VT, SystemZ::ZEXT128_32, SystemZISD::UDIVREM32,
                     Op.getOperand(0), Op.getOperand(1), Ops[1], Ops[0]);
  else
    lowerGR128Binary(DAG, DL, VT, SystemZ::ZEXT128_64, SystemZISD::UDIVREM64,
                     Op.getOperand(0), Op.getOperand(1), Ops[1], Ops[0]);
  return DAG.getMergeValues(Ops, 2, DL);
}

SDValue SystemZTargetLowering::lowerOR(SDValue Op, SelectionDAG &DAG) const {
  assert(Op.getValueType() == MVT::i64 && "Should be 64-bit operation");

  // Get the known-zero masks for each operand.
  SDValue Ops[] = { Op.getOperand(0), Op.getOperand(1) };
  APInt KnownZero[2], KnownOne[2];
  DAG.ComputeMaskedBits(Ops[0], KnownZero[0], KnownOne[0]);
  DAG.ComputeMaskedBits(Ops[1], KnownZero[1], KnownOne[1]);

  // See if the upper 32 bits of one operand and the lower 32 bits of the
  // other are known zero.  They are the low and high operands respectively.
  uint64_t Masks[] = { KnownZero[0].getZExtValue(),
                       KnownZero[1].getZExtValue() };
  unsigned High, Low;
  if ((Masks[0] >> 32) == 0xffffffff && uint32_t(Masks[1]) == 0xffffffff)
    High = 1, Low = 0;
  else if ((Masks[1] >> 32) == 0xffffffff && uint32_t(Masks[0]) == 0xffffffff)
    High = 0, Low = 1;
  else
    return Op;

  SDValue LowOp = Ops[Low];
  SDValue HighOp = Ops[High];

  // If the high part is a constant, we're better off using IILH.
  if (HighOp.getOpcode() == ISD::Constant)
    return Op;

  // If the low part is a constant that is outside the range of LHI,
  // then we're better off using IILF.
  if (LowOp.getOpcode() == ISD::Constant) {
    int64_t Value = int32_t(cast<ConstantSDNode>(LowOp)->getZExtValue());
    if (!isInt<16>(Value))
      return Op;
  }

  // Check whether the high part is an AND that doesn't change the
  // high 32 bits and just masks out low bits.  We can skip it if so.
  if (HighOp.getOpcode() == ISD::AND &&
      HighOp.getOperand(1).getOpcode() == ISD::Constant) {
    SDValue HighOp0 = HighOp.getOperand(0);
    uint64_t Mask = cast<ConstantSDNode>(HighOp.getOperand(1))->getZExtValue();
    if (DAG.MaskedValueIsZero(HighOp0, APInt(64, ~(Mask | 0xffffffff))))
      HighOp = HighOp0;
  }

  // Take advantage of the fact that all GR32 operations only change the
  // low 32 bits by truncating Low to an i32 and inserting it directly
  // using a subreg.  The interesting cases are those where the truncation
  // can be folded.
  SDLoc DL(Op);
  SDValue Low32 = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, LowOp);
  return DAG.getTargetInsertSubreg(SystemZ::subreg_l32, DL,
                                   MVT::i64, HighOp, Low32);
}

SDValue SystemZTargetLowering::lowerSIGN_EXTEND(SDValue Op,
                                                SelectionDAG &DAG) const {
  // Convert (sext (ashr (shl X, C1), C2)) to
  // (ashr (shl (anyext X), C1'), C2')), since wider shifts are as
  // cheap as narrower ones.
  SDValue N0 = Op.getOperand(0);
  EVT VT = Op.getValueType();
  if (N0.hasOneUse() && N0.getOpcode() == ISD::SRA) {
    auto *SraAmt = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    SDValue Inner = N0.getOperand(0);
    if (SraAmt && Inner.hasOneUse() && Inner.getOpcode() == ISD::SHL) {
      auto *ShlAmt = dyn_cast<ConstantSDNode>(Inner.getOperand(1));
      if (ShlAmt) {
        unsigned Extra = (VT.getSizeInBits() -
                          N0.getValueType().getSizeInBits());
        unsigned NewShlAmt = ShlAmt->getZExtValue() + Extra;
        unsigned NewSraAmt = SraAmt->getZExtValue() + Extra;
        EVT ShiftVT = N0.getOperand(1).getValueType();
        SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, SDLoc(Inner), VT,
                                  Inner.getOperand(0));
        SDValue Shl = DAG.getNode(ISD::SHL, SDLoc(Inner), VT, Ext,
                                  DAG.getConstant(NewShlAmt, ShiftVT));
        return DAG.getNode(ISD::SRA, SDLoc(N0), VT, Shl,
                           DAG.getConstant(NewSraAmt, ShiftVT));
      }
    }
  }
  return SDValue();
}

// Op is an atomic load.  Lower it into a normal volatile load.
SDValue SystemZTargetLowering::lowerATOMIC_LOAD(SDValue Op,
                                                SelectionDAG &DAG) const {
  auto *Node = cast<AtomicSDNode>(Op.getNode());
  return DAG.getExtLoad(ISD::EXTLOAD, SDLoc(Op), Op.getValueType(),
                        Node->getChain(), Node->getBasePtr(),
                        Node->getMemoryVT(), Node->getMemOperand());
}

// Op is an atomic store.  Lower it into a normal volatile store followed
// by a serialization.
SDValue SystemZTargetLowering::lowerATOMIC_STORE(SDValue Op,
                                                 SelectionDAG &DAG) const {
  auto *Node = cast<AtomicSDNode>(Op.getNode());
  SDValue Chain = DAG.getTruncStore(Node->getChain(), SDLoc(Op), Node->getVal(),
                                    Node->getBasePtr(), Node->getMemoryVT(),
                                    Node->getMemOperand());
  return SDValue(DAG.getMachineNode(SystemZ::Serialize, SDLoc(Op), MVT::Other,
                                    Chain), 0);
}

// Op is an 8-, 16-bit or 32-bit ATOMIC_LOAD_* operation.  Lower the first
// two into the fullword ATOMIC_LOADW_* operation given by Opcode.
SDValue SystemZTargetLowering::lowerATOMIC_LOAD_OP(SDValue Op,
                                                   SelectionDAG &DAG,
                                                   unsigned Opcode) const {
  auto *Node = cast<AtomicSDNode>(Op.getNode());

  // 32-bit operations need no code outside the main loop.
  EVT NarrowVT = Node->getMemoryVT();
  EVT WideVT = MVT::i32;
  if (NarrowVT == WideVT)
    return Op;

  int64_t BitSize = NarrowVT.getSizeInBits();
  SDValue ChainIn = Node->getChain();
  SDValue Addr = Node->getBasePtr();
  SDValue Src2 = Node->getVal();
  MachineMemOperand *MMO = Node->getMemOperand();
  SDLoc DL(Node);
  EVT PtrVT = Addr.getValueType();

  // Convert atomic subtracts of constants into additions.
  if (Opcode == SystemZISD::ATOMIC_LOADW_SUB)
    if (auto *Const = dyn_cast<ConstantSDNode>(Src2)) {
      Opcode = SystemZISD::ATOMIC_LOADW_ADD;
      Src2 = DAG.getConstant(-Const->getSExtValue(), Src2.getValueType());
    }

  // Get the address of the containing word.
  SDValue AlignedAddr = DAG.getNode(ISD::AND, DL, PtrVT, Addr,
                                    DAG.getConstant(-4, PtrVT));

  // Get the number of bits that the word must be rotated left in order
  // to bring the field to the top bits of a GR32.
  SDValue BitShift = DAG.getNode(ISD::SHL, DL, PtrVT, Addr,
                                 DAG.getConstant(3, PtrVT));
  BitShift = DAG.getNode(ISD::TRUNCATE, DL, WideVT, BitShift);

  // Get the complementing shift amount, for rotating a field in the top
  // bits back to its proper position.
  SDValue NegBitShift = DAG.getNode(ISD::SUB, DL, WideVT,
                                    DAG.getConstant(0, WideVT), BitShift);

  // Extend the source operand to 32 bits and prepare it for the inner loop.
  // ATOMIC_SWAPW uses RISBG to rotate the field left, but all other
  // operations require the source to be shifted in advance.  (This shift
  // can be folded if the source is constant.)  For AND and NAND, the lower
  // bits must be set, while for other opcodes they should be left clear.
  if (Opcode != SystemZISD::ATOMIC_SWAPW)
    Src2 = DAG.getNode(ISD::SHL, DL, WideVT, Src2,
                       DAG.getConstant(32 - BitSize, WideVT));
  if (Opcode == SystemZISD::ATOMIC_LOADW_AND ||
      Opcode == SystemZISD::ATOMIC_LOADW_NAND)
    Src2 = DAG.getNode(ISD::OR, DL, WideVT, Src2,
                       DAG.getConstant(uint32_t(-1) >> BitSize, WideVT));

  // Construct the ATOMIC_LOADW_* node.
  SDVTList VTList = DAG.getVTList(WideVT, MVT::Other);
  SDValue Ops[] = { ChainIn, AlignedAddr, Src2, BitShift, NegBitShift,
                    DAG.getConstant(BitSize, WideVT) };
  SDValue AtomicOp = DAG.getMemIntrinsicNode(Opcode, DL, VTList, Ops,
                                             array_lengthof(Ops),
                                             NarrowVT, MMO);

  // Rotate the result of the final CS so that the field is in the lower
  // bits of a GR32, then truncate it.
  SDValue ResultShift = DAG.getNode(ISD::ADD, DL, WideVT, BitShift,
                                    DAG.getConstant(BitSize, WideVT));
  SDValue Result = DAG.getNode(ISD::ROTL, DL, WideVT, AtomicOp, ResultShift);

  SDValue RetOps[2] = { Result, AtomicOp.getValue(1) };
  return DAG.getMergeValues(RetOps, 2, DL);
}

// Op is an ATOMIC_LOAD_SUB operation.  Lower 8- and 16-bit operations
// into ATOMIC_LOADW_SUBs and decide whether to convert 32- and 64-bit
// operations into additions.
SDValue SystemZTargetLowering::lowerATOMIC_LOAD_SUB(SDValue Op,
                                                    SelectionDAG &DAG) const {
  auto *Node = cast<AtomicSDNode>(Op.getNode());
  EVT MemVT = Node->getMemoryVT();
  if (MemVT == MVT::i32 || MemVT == MVT::i64) {
    // A full-width operation.
    assert(Op.getValueType() == MemVT && "Mismatched VTs");
    SDValue Src2 = Node->getVal();
    SDValue NegSrc2;
    SDLoc DL(Src2);

    if (auto *Op2 = dyn_cast<ConstantSDNode>(Src2)) {
      // Use an addition if the operand is constant and either LAA(G) is
      // available or the negative value is in the range of A(G)FHI.
      int64_t Value = (-Op2->getAPIntValue()).getSExtValue();
      if (isInt<32>(Value) || TM.getSubtargetImpl()->hasInterlockedAccess1())
        NegSrc2 = DAG.getConstant(Value, MemVT);
    } else if (TM.getSubtargetImpl()->hasInterlockedAccess1())
      // Use LAA(G) if available.
      NegSrc2 = DAG.getNode(ISD::SUB, DL, MemVT, DAG.getConstant(0, MemVT),
                            Src2);

    if (NegSrc2.getNode())
      return DAG.getAtomic(ISD::ATOMIC_LOAD_ADD, DL, MemVT,
                           Node->getChain(), Node->getBasePtr(), NegSrc2,
                           Node->getMemOperand(), Node->getOrdering(),
                           Node->getSynchScope());

    // Use the node as-is.
    return Op;
  }

  return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_LOADW_SUB);
}

// Node is an 8- or 16-bit ATOMIC_CMP_SWAP operation.  Lower the first two
// into a fullword ATOMIC_CMP_SWAPW operation.
SDValue SystemZTargetLowering::lowerATOMIC_CMP_SWAP(SDValue Op,
                                                    SelectionDAG &DAG) const {
  auto *Node = cast<AtomicSDNode>(Op.getNode());

  // We have native support for 32-bit compare and swap.
  EVT NarrowVT = Node->getMemoryVT();
  EVT WideVT = MVT::i32;
  if (NarrowVT == WideVT)
    return Op;

  int64_t BitSize = NarrowVT.getSizeInBits();
  SDValue ChainIn = Node->getOperand(0);
  SDValue Addr = Node->getOperand(1);
  SDValue CmpVal = Node->getOperand(2);
  SDValue SwapVal = Node->getOperand(3);
  MachineMemOperand *MMO = Node->getMemOperand();
  SDLoc DL(Node);
  EVT PtrVT = Addr.getValueType();

  // Get the address of the containing word.
  SDValue AlignedAddr = DAG.getNode(ISD::AND, DL, PtrVT, Addr,
                                    DAG.getConstant(-4, PtrVT));

  // Get the number of bits that the word must be rotated left in order
  // to bring the field to the top bits of a GR32.
  SDValue BitShift = DAG.getNode(ISD::SHL, DL, PtrVT, Addr,
                                 DAG.getConstant(3, PtrVT));
  BitShift = DAG.getNode(ISD::TRUNCATE, DL, WideVT, BitShift);

  // Get the complementing shift amount, for rotating a field in the top
  // bits back to its proper position.
  SDValue NegBitShift = DAG.getNode(ISD::SUB, DL, WideVT,
                                    DAG.getConstant(0, WideVT), BitShift);

  // Construct the ATOMIC_CMP_SWAPW node.
  SDVTList VTList = DAG.getVTList(WideVT, MVT::Other);
  SDValue Ops[] = { ChainIn, AlignedAddr, CmpVal, SwapVal, BitShift,
                    NegBitShift, DAG.getConstant(BitSize, WideVT) };
  SDValue AtomicOp = DAG.getMemIntrinsicNode(SystemZISD::ATOMIC_CMP_SWAPW, DL,
                                             VTList, Ops, array_lengthof(Ops),
                                             NarrowVT, MMO);
  return AtomicOp;
}

SDValue SystemZTargetLowering::lowerSTACKSAVE(SDValue Op,
                                              SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MF.getInfo<SystemZMachineFunctionInfo>()->setManipulatesSP(true);
  return DAG.getCopyFromReg(Op.getOperand(0), SDLoc(Op),
                            SystemZ::R15D, Op.getValueType());
}

SDValue SystemZTargetLowering::lowerSTACKRESTORE(SDValue Op,
                                                 SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MF.getInfo<SystemZMachineFunctionInfo>()->setManipulatesSP(true);
  return DAG.getCopyToReg(Op.getOperand(0), SDLoc(Op),
                          SystemZ::R15D, Op.getOperand(1));
}

SDValue SystemZTargetLowering::lowerPREFETCH(SDValue Op,
                                             SelectionDAG &DAG) const {
  bool IsData = cast<ConstantSDNode>(Op.getOperand(4))->getZExtValue();
  if (!IsData)
    // Just preserve the chain.
    return Op.getOperand(0);

  bool IsWrite = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue();
  unsigned Code = IsWrite ? SystemZ::PFD_WRITE : SystemZ::PFD_READ;
  auto *Node = cast<MemIntrinsicSDNode>(Op.getNode());
  SDValue Ops[] = {
    Op.getOperand(0),
    DAG.getConstant(Code, MVT::i32),
    Op.getOperand(1)
  };
  return DAG.getMemIntrinsicNode(SystemZISD::PREFETCH, SDLoc(Op),
                                 Node->getVTList(), Ops, array_lengthof(Ops),
                                 Node->getMemoryVT(), Node->getMemOperand());
}

SDValue SystemZTargetLowering::LowerOperation(SDValue Op,
                                              SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::BR_CC:
    return lowerBR_CC(Op, DAG);
  case ISD::SELECT_CC:
    return lowerSELECT_CC(Op, DAG);
  case ISD::SETCC:
    return lowerSETCC(Op, DAG);
  case ISD::GlobalAddress:
    return lowerGlobalAddress(cast<GlobalAddressSDNode>(Op), DAG);
  case ISD::GlobalTLSAddress:
    return lowerGlobalTLSAddress(cast<GlobalAddressSDNode>(Op), DAG);
  case ISD::BlockAddress:
    return lowerBlockAddress(cast<BlockAddressSDNode>(Op), DAG);
  case ISD::JumpTable:
    return lowerJumpTable(cast<JumpTableSDNode>(Op), DAG);
  case ISD::ConstantPool:
    return lowerConstantPool(cast<ConstantPoolSDNode>(Op), DAG);
  case ISD::BITCAST:
    return lowerBITCAST(Op, DAG);
  case ISD::VASTART:
    return lowerVASTART(Op, DAG);
  case ISD::VACOPY:
    return lowerVACOPY(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC:
    return lowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::SMUL_LOHI:
    return lowerSMUL_LOHI(Op, DAG);
  case ISD::UMUL_LOHI:
    return lowerUMUL_LOHI(Op, DAG);
  case ISD::SDIVREM:
    return lowerSDIVREM(Op, DAG);
  case ISD::UDIVREM:
    return lowerUDIVREM(Op, DAG);
  case ISD::OR:
    return lowerOR(Op, DAG);
  case ISD::SIGN_EXTEND:
    return lowerSIGN_EXTEND(Op, DAG);
  case ISD::ATOMIC_SWAP:
    return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_SWAPW);
  case ISD::ATOMIC_STORE:
    return lowerATOMIC_STORE(Op, DAG);
  case ISD::ATOMIC_LOAD:
    return lowerATOMIC_LOAD(Op, DAG);
  case ISD::ATOMIC_LOAD_ADD:
    return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_LOADW_ADD);
  case ISD::ATOMIC_LOAD_SUB:
    return lowerATOMIC_LOAD_SUB(Op, DAG);
  case ISD::ATOMIC_LOAD_AND:
    return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_LOADW_AND);
  case ISD::ATOMIC_LOAD_OR:
    return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_LOADW_OR);
  case ISD::ATOMIC_LOAD_XOR:
    return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_LOADW_XOR);
  case ISD::ATOMIC_LOAD_NAND:
    return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_LOADW_NAND);
  case ISD::ATOMIC_LOAD_MIN:
    return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_LOADW_MIN);
  case ISD::ATOMIC_LOAD_MAX:
    return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_LOADW_MAX);
  case ISD::ATOMIC_LOAD_UMIN:
    return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_LOADW_UMIN);
  case ISD::ATOMIC_LOAD_UMAX:
    return lowerATOMIC_LOAD_OP(Op, DAG, SystemZISD::ATOMIC_LOADW_UMAX);
  case ISD::ATOMIC_CMP_SWAP:
    return lowerATOMIC_CMP_SWAP(Op, DAG);
  case ISD::STACKSAVE:
    return lowerSTACKSAVE(Op, DAG);
  case ISD::STACKRESTORE:
    return lowerSTACKRESTORE(Op, DAG);
  case ISD::PREFETCH:
    return lowerPREFETCH(Op, DAG);
  default:
    llvm_unreachable("Unexpected node to lower");
  }
}

const char *SystemZTargetLowering::getTargetNodeName(unsigned Opcode) const {
#define OPCODE(NAME) case SystemZISD::NAME: return "SystemZISD::" #NAME
  switch (Opcode) {
    OPCODE(RET_FLAG);
    OPCODE(CALL);
    OPCODE(SIBCALL);
    OPCODE(PCREL_WRAPPER);
    OPCODE(PCREL_OFFSET);
    OPCODE(IABS);
    OPCODE(ICMP);
    OPCODE(FCMP);
    OPCODE(TM);
    OPCODE(BR_CCMASK);
    OPCODE(SELECT_CCMASK);
    OPCODE(ADJDYNALLOC);
    OPCODE(EXTRACT_ACCESS);
    OPCODE(UMUL_LOHI64);
    OPCODE(SDIVREM64);
    OPCODE(UDIVREM32);
    OPCODE(UDIVREM64);
    OPCODE(MVC);
    OPCODE(MVC_LOOP);
    OPCODE(NC);
    OPCODE(NC_LOOP);
    OPCODE(OC);
    OPCODE(OC_LOOP);
    OPCODE(XC);
    OPCODE(XC_LOOP);
    OPCODE(CLC);
    OPCODE(CLC_LOOP);
    OPCODE(STRCMP);
    OPCODE(STPCPY);
    OPCODE(SEARCH_STRING);
    OPCODE(IPM);
    OPCODE(SERIALIZE);
    OPCODE(ATOMIC_SWAPW);
    OPCODE(ATOMIC_LOADW_ADD);
    OPCODE(ATOMIC_LOADW_SUB);
    OPCODE(ATOMIC_LOADW_AND);
    OPCODE(ATOMIC_LOADW_OR);
    OPCODE(ATOMIC_LOADW_XOR);
    OPCODE(ATOMIC_LOADW_NAND);
    OPCODE(ATOMIC_LOADW_MIN);
    OPCODE(ATOMIC_LOADW_MAX);
    OPCODE(ATOMIC_LOADW_UMIN);
    OPCODE(ATOMIC_LOADW_UMAX);
    OPCODE(ATOMIC_CMP_SWAPW);
    OPCODE(PREFETCH);
  }
  return NULL;
#undef OPCODE
}

//===----------------------------------------------------------------------===//
// Custom insertion
//===----------------------------------------------------------------------===//

// Create a new basic block after MBB.
static MachineBasicBlock *emitBlockAfter(MachineBasicBlock *MBB) {
  MachineFunction &MF = *MBB->getParent();
  MachineBasicBlock *NewMBB = MF.CreateMachineBasicBlock(MBB->getBasicBlock());
  MF.insert(std::next(MachineFunction::iterator(MBB)), NewMBB);
  return NewMBB;
}

// Split MBB after MI and return the new block (the one that contains
// instructions after MI).
static MachineBasicBlock *splitBlockAfter(MachineInstr *MI,
                                          MachineBasicBlock *MBB) {
  MachineBasicBlock *NewMBB = emitBlockAfter(MBB);
  NewMBB->splice(NewMBB->begin(), MBB,
                 std::next(MachineBasicBlock::iterator(MI)), MBB->end());
  NewMBB->transferSuccessorsAndUpdatePHIs(MBB);
  return NewMBB;
}

// Split MBB before MI and return the new block (the one that contains MI).
static MachineBasicBlock *splitBlockBefore(MachineInstr *MI,
                                           MachineBasicBlock *MBB) {
  MachineBasicBlock *NewMBB = emitBlockAfter(MBB);
  NewMBB->splice(NewMBB->begin(), MBB, MI, MBB->end());
  NewMBB->transferSuccessorsAndUpdatePHIs(MBB);
  return NewMBB;
}

// Force base value Base into a register before MI.  Return the register.
static unsigned forceReg(MachineInstr *MI, MachineOperand &Base,
                         const SystemZInstrInfo *TII) {
  if (Base.isReg())
    return Base.getReg();

  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction &MF = *MBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  unsigned Reg = MRI.createVirtualRegister(&SystemZ::ADDR64BitRegClass);
  BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(SystemZ::LA), Reg)
    .addOperand(Base).addImm(0).addReg(0);
  return Reg;
}

// Implement EmitInstrWithCustomInserter for pseudo Select* instruction MI.
MachineBasicBlock *
SystemZTargetLowering::emitSelect(MachineInstr *MI,
                                  MachineBasicBlock *MBB) const {
  const SystemZInstrInfo *TII = TM.getInstrInfo();

  unsigned DestReg  = MI->getOperand(0).getReg();
  unsigned TrueReg  = MI->getOperand(1).getReg();
  unsigned FalseReg = MI->getOperand(2).getReg();
  unsigned CCValid  = MI->getOperand(3).getImm();
  unsigned CCMask   = MI->getOperand(4).getImm();
  DebugLoc DL       = MI->getDebugLoc();

  MachineBasicBlock *StartMBB = MBB;
  MachineBasicBlock *JoinMBB  = splitBlockBefore(MI, MBB);
  MachineBasicBlock *FalseMBB = emitBlockAfter(StartMBB);

  //  StartMBB:
  //   BRC CCMask, JoinMBB
  //   # fallthrough to FalseMBB
  MBB = StartMBB;
  BuildMI(MBB, DL, TII->get(SystemZ::BRC))
    .addImm(CCValid).addImm(CCMask).addMBB(JoinMBB);
  MBB->addSuccessor(JoinMBB);
  MBB->addSuccessor(FalseMBB);

  //  FalseMBB:
  //   # fallthrough to JoinMBB
  MBB = FalseMBB;
  MBB->addSuccessor(JoinMBB);

  //  JoinMBB:
  //   %Result = phi [ %FalseReg, FalseMBB ], [ %TrueReg, StartMBB ]
  //  ...
  MBB = JoinMBB;
  BuildMI(*MBB, MI, DL, TII->get(SystemZ::PHI), DestReg)
    .addReg(TrueReg).addMBB(StartMBB)
    .addReg(FalseReg).addMBB(FalseMBB);

  MI->eraseFromParent();
  return JoinMBB;
}

// Implement EmitInstrWithCustomInserter for pseudo CondStore* instruction MI.
// StoreOpcode is the store to use and Invert says whether the store should
// happen when the condition is false rather than true.  If a STORE ON
// CONDITION is available, STOCOpcode is its opcode, otherwise it is 0.
MachineBasicBlock *
SystemZTargetLowering::emitCondStore(MachineInstr *MI,
                                     MachineBasicBlock *MBB,
                                     unsigned StoreOpcode, unsigned STOCOpcode,
                                     bool Invert) const {
  const SystemZInstrInfo *TII = TM.getInstrInfo();

  unsigned SrcReg     = MI->getOperand(0).getReg();
  MachineOperand Base = MI->getOperand(1);
  int64_t Disp        = MI->getOperand(2).getImm();
  unsigned IndexReg   = MI->getOperand(3).getReg();
  unsigned CCValid    = MI->getOperand(4).getImm();
  unsigned CCMask     = MI->getOperand(5).getImm();
  DebugLoc DL         = MI->getDebugLoc();

  StoreOpcode = TII->getOpcodeForOffset(StoreOpcode, Disp);

  // Use STOCOpcode if possible.  We could use different store patterns in
  // order to avoid matching the index register, but the performance trade-offs
  // might be more complicated in that case.
  if (STOCOpcode && !IndexReg && TM.getSubtargetImpl()->hasLoadStoreOnCond()) {
    if (Invert)
      CCMask ^= CCValid;
    BuildMI(*MBB, MI, DL, TII->get(STOCOpcode))
      .addReg(SrcReg).addOperand(Base).addImm(Disp)
      .addImm(CCValid).addImm(CCMask);
    MI->eraseFromParent();
    return MBB;
  }

  // Get the condition needed to branch around the store.
  if (!Invert)
    CCMask ^= CCValid;

  MachineBasicBlock *StartMBB = MBB;
  MachineBasicBlock *JoinMBB  = splitBlockBefore(MI, MBB);
  MachineBasicBlock *FalseMBB = emitBlockAfter(StartMBB);

  //  StartMBB:
  //   BRC CCMask, JoinMBB
  //   # fallthrough to FalseMBB
  MBB = StartMBB;
  BuildMI(MBB, DL, TII->get(SystemZ::BRC))
    .addImm(CCValid).addImm(CCMask).addMBB(JoinMBB);
  MBB->addSuccessor(JoinMBB);
  MBB->addSuccessor(FalseMBB);

  //  FalseMBB:
  //   store %SrcReg, %Disp(%Index,%Base)
  //   # fallthrough to JoinMBB
  MBB = FalseMBB;
  BuildMI(MBB, DL, TII->get(StoreOpcode))
    .addReg(SrcReg).addOperand(Base).addImm(Disp).addReg(IndexReg);
  MBB->addSuccessor(JoinMBB);

  MI->eraseFromParent();
  return JoinMBB;
}

// Implement EmitInstrWithCustomInserter for pseudo ATOMIC_LOAD{,W}_*
// or ATOMIC_SWAP{,W} instruction MI.  BinOpcode is the instruction that
// performs the binary operation elided by "*", or 0 for ATOMIC_SWAP{,W}.
// BitSize is the width of the field in bits, or 0 if this is a partword
// ATOMIC_LOADW_* or ATOMIC_SWAPW instruction, in which case the bitsize
// is one of the operands.  Invert says whether the field should be
// inverted after performing BinOpcode (e.g. for NAND).
MachineBasicBlock *
SystemZTargetLowering::emitAtomicLoadBinary(MachineInstr *MI,
                                            MachineBasicBlock *MBB,
                                            unsigned BinOpcode,
                                            unsigned BitSize,
                                            bool Invert) const {
  const SystemZInstrInfo *TII = TM.getInstrInfo();
  MachineFunction &MF = *MBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool IsSubWord = (BitSize < 32);

  // Extract the operands.  Base can be a register or a frame index.
  // Src2 can be a register or immediate.
  unsigned Dest        = MI->getOperand(0).getReg();
  MachineOperand Base  = earlyUseOperand(MI->getOperand(1));
  int64_t Disp         = MI->getOperand(2).getImm();
  MachineOperand Src2  = earlyUseOperand(MI->getOperand(3));
  unsigned BitShift    = (IsSubWord ? MI->getOperand(4).getReg() : 0);
  unsigned NegBitShift = (IsSubWord ? MI->getOperand(5).getReg() : 0);
  DebugLoc DL          = MI->getDebugLoc();
  if (IsSubWord)
    BitSize = MI->getOperand(6).getImm();

  // Subword operations use 32-bit registers.
  const TargetRegisterClass *RC = (BitSize <= 32 ?
                                   &SystemZ::GR32BitRegClass :
                                   &SystemZ::GR64BitRegClass);
  unsigned LOpcode  = BitSize <= 32 ? SystemZ::L  : SystemZ::LG;
  unsigned CSOpcode = BitSize <= 32 ? SystemZ::CS : SystemZ::CSG;

  // Get the right opcodes for the displacement.
  LOpcode  = TII->getOpcodeForOffset(LOpcode,  Disp);
  CSOpcode = TII->getOpcodeForOffset(CSOpcode, Disp);
  assert(LOpcode && CSOpcode && "Displacement out of range");

  // Create virtual registers for temporary results.
  unsigned OrigVal       = MRI.createVirtualRegister(RC);
  unsigned OldVal        = MRI.createVirtualRegister(RC);
  unsigned NewVal        = (BinOpcode || IsSubWord ?
                            MRI.createVirtualRegister(RC) : Src2.getReg());
  unsigned RotatedOldVal = (IsSubWord ? MRI.createVirtualRegister(RC) : OldVal);
  unsigned RotatedNewVal = (IsSubWord ? MRI.createVirtualRegister(RC) : NewVal);

  // Insert a basic block for the main loop.
  MachineBasicBlock *StartMBB = MBB;
  MachineBasicBlock *DoneMBB  = splitBlockBefore(MI, MBB);
  MachineBasicBlock *LoopMBB  = emitBlockAfter(StartMBB);

  //  StartMBB:
  //   ...
  //   %OrigVal = L Disp(%Base)
  //   # fall through to LoopMMB
  MBB = StartMBB;
  BuildMI(MBB, DL, TII->get(LOpcode), OrigVal)
    .addOperand(Base).addImm(Disp).addReg(0);
  MBB->addSuccessor(LoopMBB);

  //  LoopMBB:
  //   %OldVal        = phi [ %OrigVal, StartMBB ], [ %Dest, LoopMBB ]
  //   %RotatedOldVal = RLL %OldVal, 0(%BitShift)
  //   %RotatedNewVal = OP %RotatedOldVal, %Src2
  //   %NewVal        = RLL %RotatedNewVal, 0(%NegBitShift)
  //   %Dest          = CS %OldVal, %NewVal, Disp(%Base)
  //   JNE LoopMBB
  //   # fall through to DoneMMB
  MBB = LoopMBB;
  BuildMI(MBB, DL, TII->get(SystemZ::PHI), OldVal)
    .addReg(OrigVal).addMBB(StartMBB)
    .addReg(Dest).addMBB(LoopMBB);
  if (IsSubWord)
    BuildMI(MBB, DL, TII->get(SystemZ::RLL), RotatedOldVal)
      .addReg(OldVal).addReg(BitShift).addImm(0);
  if (Invert) {
    // Perform the operation normally and then invert every bit of the field.
    unsigned Tmp = MRI.createVirtualRegister(RC);
    BuildMI(MBB, DL, TII->get(BinOpcode), Tmp)
      .addReg(RotatedOldVal).addOperand(Src2);
    if (BitSize < 32)
      // XILF with the upper BitSize bits set.
      BuildMI(MBB, DL, TII->get(SystemZ::XILF), RotatedNewVal)
        .addReg(Tmp).addImm(uint32_t(~0 << (32 - BitSize)));
    else if (BitSize == 32)
      // XILF with every bit set.
      BuildMI(MBB, DL, TII->get(SystemZ::XILF), RotatedNewVal)
        .addReg(Tmp).addImm(~uint32_t(0));
    else {
      // Use LCGR and add -1 to the result, which is more compact than
      // an XILF, XILH pair.
      unsigned Tmp2 = MRI.createVirtualRegister(RC);
      BuildMI(MBB, DL, TII->get(SystemZ::LCGR), Tmp2).addReg(Tmp);
      BuildMI(MBB, DL, TII->get(SystemZ::AGHI), RotatedNewVal)
        .addReg(Tmp2).addImm(-1);
    }
  } else if (BinOpcode)
    // A simply binary operation.
    BuildMI(MBB, DL, TII->get(BinOpcode), RotatedNewVal)
      .addReg(RotatedOldVal).addOperand(Src2);
  else if (IsSubWord)
    // Use RISBG to rotate Src2 into position and use it to replace the
    // field in RotatedOldVal.
    BuildMI(MBB, DL, TII->get(SystemZ::RISBG32), RotatedNewVal)
      .addReg(RotatedOldVal).addReg(Src2.getReg())
      .addImm(32).addImm(31 + BitSize).addImm(32 - BitSize);
  if (IsSubWord)
    BuildMI(MBB, DL, TII->get(SystemZ::RLL), NewVal)
      .addReg(RotatedNewVal).addReg(NegBitShift).addImm(0);
  BuildMI(MBB, DL, TII->get(CSOpcode), Dest)
    .addReg(OldVal).addReg(NewVal).addOperand(Base).addImm(Disp);
  BuildMI(MBB, DL, TII->get(SystemZ::BRC))
    .addImm(SystemZ::CCMASK_CS).addImm(SystemZ::CCMASK_CS_NE).addMBB(LoopMBB);
  MBB->addSuccessor(LoopMBB);
  MBB->addSuccessor(DoneMBB);

  MI->eraseFromParent();
  return DoneMBB;
}

// Implement EmitInstrWithCustomInserter for pseudo
// ATOMIC_LOAD{,W}_{,U}{MIN,MAX} instruction MI.  CompareOpcode is the
// instruction that should be used to compare the current field with the
// minimum or maximum value.  KeepOldMask is the BRC condition-code mask
// for when the current field should be kept.  BitSize is the width of
// the field in bits, or 0 if this is a partword ATOMIC_LOADW_* instruction.
MachineBasicBlock *
SystemZTargetLowering::emitAtomicLoadMinMax(MachineInstr *MI,
                                            MachineBasicBlock *MBB,
                                            unsigned CompareOpcode,
                                            unsigned KeepOldMask,
                                            unsigned BitSize) const {
  const SystemZInstrInfo *TII = TM.getInstrInfo();
  MachineFunction &MF = *MBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool IsSubWord = (BitSize < 32);

  // Extract the operands.  Base can be a register or a frame index.
  unsigned Dest        = MI->getOperand(0).getReg();
  MachineOperand Base  = earlyUseOperand(MI->getOperand(1));
  int64_t  Disp        = MI->getOperand(2).getImm();
  unsigned Src2        = MI->getOperand(3).getReg();
  unsigned BitShift    = (IsSubWord ? MI->getOperand(4).getReg() : 0);
  unsigned NegBitShift = (IsSubWord ? MI->getOperand(5).getReg() : 0);
  DebugLoc DL          = MI->getDebugLoc();
  if (IsSubWord)
    BitSize = MI->getOperand(6).getImm();

  // Subword operations use 32-bit registers.
  const TargetRegisterClass *RC = (BitSize <= 32 ?
                                   &SystemZ::GR32BitRegClass :
                                   &SystemZ::GR64BitRegClass);
  unsigned LOpcode  = BitSize <= 32 ? SystemZ::L  : SystemZ::LG;
  unsigned CSOpcode = BitSize <= 32 ? SystemZ::CS : SystemZ::CSG;

  // Get the right opcodes for the displacement.
  LOpcode  = TII->getOpcodeForOffset(LOpcode,  Disp);
  CSOpcode = TII->getOpcodeForOffset(CSOpcode, Disp);
  assert(LOpcode && CSOpcode && "Displacement out of range");

  // Create virtual registers for temporary results.
  unsigned OrigVal       = MRI.createVirtualRegister(RC);
  unsigned OldVal        = MRI.createVirtualRegister(RC);
  unsigned NewVal        = MRI.createVirtualRegister(RC);
  unsigned RotatedOldVal = (IsSubWord ? MRI.createVirtualRegister(RC) : OldVal);
  unsigned RotatedAltVal = (IsSubWord ? MRI.createVirtualRegister(RC) : Src2);
  unsigned RotatedNewVal = (IsSubWord ? MRI.createVirtualRegister(RC) : NewVal);

  // Insert 3 basic blocks for the loop.
  MachineBasicBlock *StartMBB  = MBB;
  MachineBasicBlock *DoneMBB   = splitBlockBefore(MI, MBB);
  MachineBasicBlock *LoopMBB   = emitBlockAfter(StartMBB);
  MachineBasicBlock *UseAltMBB = emitBlockAfter(LoopMBB);
  MachineBasicBlock *UpdateMBB = emitBlockAfter(UseAltMBB);

  //  StartMBB:
  //   ...
  //   %OrigVal     = L Disp(%Base)
  //   # fall through to LoopMMB
  MBB = StartMBB;
  BuildMI(MBB, DL, TII->get(LOpcode), OrigVal)
    .addOperand(Base).addImm(Disp).addReg(0);
  MBB->addSuccessor(LoopMBB);

  //  LoopMBB:
  //   %OldVal        = phi [ %OrigVal, StartMBB ], [ %Dest, UpdateMBB ]
  //   %RotatedOldVal = RLL %OldVal, 0(%BitShift)
  //   CompareOpcode %RotatedOldVal, %Src2
  //   BRC KeepOldMask, UpdateMBB
  MBB = LoopMBB;
  BuildMI(MBB, DL, TII->get(SystemZ::PHI), OldVal)
    .addReg(OrigVal).addMBB(StartMBB)
    .addReg(Dest).addMBB(UpdateMBB);
  if (IsSubWord)
    BuildMI(MBB, DL, TII->get(SystemZ::RLL), RotatedOldVal)
      .addReg(OldVal).addReg(BitShift).addImm(0);
  BuildMI(MBB, DL, TII->get(CompareOpcode))
    .addReg(RotatedOldVal).addReg(Src2);
  BuildMI(MBB, DL, TII->get(SystemZ::BRC))
    .addImm(SystemZ::CCMASK_ICMP).addImm(KeepOldMask).addMBB(UpdateMBB);
  MBB->addSuccessor(UpdateMBB);
  MBB->addSuccessor(UseAltMBB);

  //  UseAltMBB:
  //   %RotatedAltVal = RISBG %RotatedOldVal, %Src2, 32, 31 + BitSize, 0
  //   # fall through to UpdateMMB
  MBB = UseAltMBB;
  if (IsSubWord)
    BuildMI(MBB, DL, TII->get(SystemZ::RISBG32), RotatedAltVal)
      .addReg(RotatedOldVal).addReg(Src2)
      .addImm(32).addImm(31 + BitSize).addImm(0);
  MBB->addSuccessor(UpdateMBB);

  //  UpdateMBB:
  //   %RotatedNewVal = PHI [ %RotatedOldVal, LoopMBB ],
  //                        [ %RotatedAltVal, UseAltMBB ]
  //   %NewVal        = RLL %RotatedNewVal, 0(%NegBitShift)
  //   %Dest          = CS %OldVal, %NewVal, Disp(%Base)
  //   JNE LoopMBB
  //   # fall through to DoneMMB
  MBB = UpdateMBB;
  BuildMI(MBB, DL, TII->get(SystemZ::PHI), RotatedNewVal)
    .addReg(RotatedOldVal).addMBB(LoopMBB)
    .addReg(RotatedAltVal).addMBB(UseAltMBB);
  if (IsSubWord)
    BuildMI(MBB, DL, TII->get(SystemZ::RLL), NewVal)
      .addReg(RotatedNewVal).addReg(NegBitShift).addImm(0);
  BuildMI(MBB, DL, TII->get(CSOpcode), Dest)
    .addReg(OldVal).addReg(NewVal).addOperand(Base).addImm(Disp);
  BuildMI(MBB, DL, TII->get(SystemZ::BRC))
    .addImm(SystemZ::CCMASK_CS).addImm(SystemZ::CCMASK_CS_NE).addMBB(LoopMBB);
  MBB->addSuccessor(LoopMBB);
  MBB->addSuccessor(DoneMBB);

  MI->eraseFromParent();
  return DoneMBB;
}

// Implement EmitInstrWithCustomInserter for pseudo ATOMIC_CMP_SWAPW
// instruction MI.
MachineBasicBlock *
SystemZTargetLowering::emitAtomicCmpSwapW(MachineInstr *MI,
                                          MachineBasicBlock *MBB) const {
  const SystemZInstrInfo *TII = TM.getInstrInfo();
  MachineFunction &MF = *MBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Extract the operands.  Base can be a register or a frame index.
  unsigned Dest        = MI->getOperand(0).getReg();
  MachineOperand Base  = earlyUseOperand(MI->getOperand(1));
  int64_t  Disp        = MI->getOperand(2).getImm();
  unsigned OrigCmpVal  = MI->getOperand(3).getReg();
  unsigned OrigSwapVal = MI->getOperand(4).getReg();
  unsigned BitShift    = MI->getOperand(5).getReg();
  unsigned NegBitShift = MI->getOperand(6).getReg();
  int64_t  BitSize     = MI->getOperand(7).getImm();
  DebugLoc DL          = MI->getDebugLoc();

  const TargetRegisterClass *RC = &SystemZ::GR32BitRegClass;

  // Get the right opcodes for the displacement.
  unsigned LOpcode  = TII->getOpcodeForOffset(SystemZ::L,  Disp);
  unsigned CSOpcode = TII->getOpcodeForOffset(SystemZ::CS, Disp);
  assert(LOpcode && CSOpcode && "Displacement out of range");

  // Create virtual registers for temporary results.
  unsigned OrigOldVal   = MRI.createVirtualRegister(RC);
  unsigned OldVal       = MRI.createVirtualRegister(RC);
  unsigned CmpVal       = MRI.createVirtualRegister(RC);
  unsigned SwapVal      = MRI.createVirtualRegister(RC);
  unsigned StoreVal     = MRI.createVirtualRegister(RC);
  unsigned RetryOldVal  = MRI.createVirtualRegister(RC);
  unsigned RetryCmpVal  = MRI.createVirtualRegister(RC);
  unsigned RetrySwapVal = MRI.createVirtualRegister(RC);

  // Insert 2 basic blocks for the loop.
  MachineBasicBlock *StartMBB = MBB;
  MachineBasicBlock *DoneMBB  = splitBlockBefore(MI, MBB);
  MachineBasicBlock *LoopMBB  = emitBlockAfter(StartMBB);
  MachineBasicBlock *SetMBB   = emitBlockAfter(LoopMBB);

  //  StartMBB:
  //   ...
  //   %OrigOldVal     = L Disp(%Base)
  //   # fall through to LoopMMB
  MBB = StartMBB;
  BuildMI(MBB, DL, TII->get(LOpcode), OrigOldVal)
    .addOperand(Base).addImm(Disp).addReg(0);
  MBB->addSuccessor(LoopMBB);

  //  LoopMBB:
  //   %OldVal        = phi [ %OrigOldVal, EntryBB ], [ %RetryOldVal, SetMBB ]
  //   %CmpVal        = phi [ %OrigCmpVal, EntryBB ], [ %RetryCmpVal, SetMBB ]
  //   %SwapVal       = phi [ %OrigSwapVal, EntryBB ], [ %RetrySwapVal, SetMBB ]
  //   %Dest          = RLL %OldVal, BitSize(%BitShift)
  //                      ^^ The low BitSize bits contain the field
  //                         of interest.
  //   %RetryCmpVal   = RISBG32 %CmpVal, %Dest, 32, 63-BitSize, 0
  //                      ^^ Replace the upper 32-BitSize bits of the
  //                         comparison value with those that we loaded,
  //                         so that we can use a full word comparison.
  //   CR %Dest, %RetryCmpVal
  //   JNE DoneMBB
  //   # Fall through to SetMBB
  MBB = LoopMBB;
  BuildMI(MBB, DL, TII->get(SystemZ::PHI), OldVal)
    .addReg(OrigOldVal).addMBB(StartMBB)
    .addReg(RetryOldVal).addMBB(SetMBB);
  BuildMI(MBB, DL, TII->get(SystemZ::PHI), CmpVal)
    .addReg(OrigCmpVal).addMBB(StartMBB)
    .addReg(RetryCmpVal).addMBB(SetMBB);
  BuildMI(MBB, DL, TII->get(SystemZ::PHI), SwapVal)
    .addReg(OrigSwapVal).addMBB(StartMBB)
    .addReg(RetrySwapVal).addMBB(SetMBB);
  BuildMI(MBB, DL, TII->get(SystemZ::RLL), Dest)
    .addReg(OldVal).addReg(BitShift).addImm(BitSize);
  BuildMI(MBB, DL, TII->get(SystemZ::RISBG32), RetryCmpVal)
    .addReg(CmpVal).addReg(Dest).addImm(32).addImm(63 - BitSize).addImm(0);
  BuildMI(MBB, DL, TII->get(SystemZ::CR))
    .addReg(Dest).addReg(RetryCmpVal);
  BuildMI(MBB, DL, TII->get(SystemZ::BRC))
    .addImm(SystemZ::CCMASK_ICMP)
    .addImm(SystemZ::CCMASK_CMP_NE).addMBB(DoneMBB);
  MBB->addSuccessor(DoneMBB);
  MBB->addSuccessor(SetMBB);

  //  SetMBB:
  //   %RetrySwapVal = RISBG32 %SwapVal, %Dest, 32, 63-BitSize, 0
  //                      ^^ Replace the upper 32-BitSize bits of the new
  //                         value with those that we loaded.
  //   %StoreVal    = RLL %RetrySwapVal, -BitSize(%NegBitShift)
  //                      ^^ Rotate the new field to its proper position.
  //   %RetryOldVal = CS %Dest, %StoreVal, Disp(%Base)
  //   JNE LoopMBB
  //   # fall through to ExitMMB
  MBB = SetMBB;
  BuildMI(MBB, DL, TII->get(SystemZ::RISBG32), RetrySwapVal)
    .addReg(SwapVal).addReg(Dest).addImm(32).addImm(63 - BitSize).addImm(0);
  BuildMI(MBB, DL, TII->get(SystemZ::RLL), StoreVal)
    .addReg(RetrySwapVal).addReg(NegBitShift).addImm(-BitSize);
  BuildMI(MBB, DL, TII->get(CSOpcode), RetryOldVal)
    .addReg(OldVal).addReg(StoreVal).addOperand(Base).addImm(Disp);
  BuildMI(MBB, DL, TII->get(SystemZ::BRC))
    .addImm(SystemZ::CCMASK_CS).addImm(SystemZ::CCMASK_CS_NE).addMBB(LoopMBB);
  MBB->addSuccessor(LoopMBB);
  MBB->addSuccessor(DoneMBB);

  MI->eraseFromParent();
  return DoneMBB;
}

// Emit an extension from a GR32 or GR64 to a GR128.  ClearEven is true
// if the high register of the GR128 value must be cleared or false if
// it's "don't care".  SubReg is subreg_l32 when extending a GR32
// and subreg_l64 when extending a GR64.
MachineBasicBlock *
SystemZTargetLowering::emitExt128(MachineInstr *MI,
                                  MachineBasicBlock *MBB,
                                  bool ClearEven, unsigned SubReg) const {
  const SystemZInstrInfo *TII = TM.getInstrInfo();
  MachineFunction &MF = *MBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DebugLoc DL = MI->getDebugLoc();

  unsigned Dest  = MI->getOperand(0).getReg();
  unsigned Src   = MI->getOperand(1).getReg();
  unsigned In128 = MRI.createVirtualRegister(&SystemZ::GR128BitRegClass);

  BuildMI(*MBB, MI, DL, TII->get(TargetOpcode::IMPLICIT_DEF), In128);
  if (ClearEven) {
    unsigned NewIn128 = MRI.createVirtualRegister(&SystemZ::GR128BitRegClass);
    unsigned Zero64   = MRI.createVirtualRegister(&SystemZ::GR64BitRegClass);

    BuildMI(*MBB, MI, DL, TII->get(SystemZ::LLILL), Zero64)
      .addImm(0);
    BuildMI(*MBB, MI, DL, TII->get(TargetOpcode::INSERT_SUBREG), NewIn128)
      .addReg(In128).addReg(Zero64).addImm(SystemZ::subreg_h64);
    In128 = NewIn128;
  }
  BuildMI(*MBB, MI, DL, TII->get(TargetOpcode::INSERT_SUBREG), Dest)
    .addReg(In128).addReg(Src).addImm(SubReg);

  MI->eraseFromParent();
  return MBB;
}

MachineBasicBlock *
SystemZTargetLowering::emitMemMemWrapper(MachineInstr *MI,
                                         MachineBasicBlock *MBB,
                                         unsigned Opcode) const {
  const SystemZInstrInfo *TII = TM.getInstrInfo();
  MachineFunction &MF = *MBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DebugLoc DL = MI->getDebugLoc();

  MachineOperand DestBase = earlyUseOperand(MI->getOperand(0));
  uint64_t       DestDisp = MI->getOperand(1).getImm();
  MachineOperand SrcBase  = earlyUseOperand(MI->getOperand(2));
  uint64_t       SrcDisp  = MI->getOperand(3).getImm();
  uint64_t       Length   = MI->getOperand(4).getImm();

  // When generating more than one CLC, all but the last will need to
  // branch to the end when a difference is found.
  MachineBasicBlock *EndMBB = (Length > 256 && Opcode == SystemZ::CLC ?
                               splitBlockAfter(MI, MBB) : 0);

  // Check for the loop form, in which operand 5 is the trip count.
  if (MI->getNumExplicitOperands() > 5) {
    bool HaveSingleBase = DestBase.isIdenticalTo(SrcBase);

    uint64_t StartCountReg = MI->getOperand(5).getReg();
    uint64_t StartSrcReg   = forceReg(MI, SrcBase, TII);
    uint64_t StartDestReg  = (HaveSingleBase ? StartSrcReg :
                              forceReg(MI, DestBase, TII));

    const TargetRegisterClass *RC = &SystemZ::ADDR64BitRegClass;
    uint64_t ThisSrcReg  = MRI.createVirtualRegister(RC);
    uint64_t ThisDestReg = (HaveSingleBase ? ThisSrcReg :
                            MRI.createVirtualRegister(RC));
    uint64_t NextSrcReg  = MRI.createVirtualRegister(RC);
    uint64_t NextDestReg = (HaveSingleBase ? NextSrcReg :
                            MRI.createVirtualRegister(RC));

    RC = &SystemZ::GR64BitRegClass;
    uint64_t ThisCountReg = MRI.createVirtualRegister(RC);
    uint64_t NextCountReg = MRI.createVirtualRegister(RC);

    MachineBasicBlock *StartMBB = MBB;
    MachineBasicBlock *DoneMBB = splitBlockBefore(MI, MBB);
    MachineBasicBlock *LoopMBB = emitBlockAfter(StartMBB);
    MachineBasicBlock *NextMBB = (EndMBB ? emitBlockAfter(LoopMBB) : LoopMBB);

    //  StartMBB:
    //   # fall through to LoopMMB
    MBB->addSuccessor(LoopMBB);

    //  LoopMBB:
    //   %ThisDestReg = phi [ %StartDestReg, StartMBB ],
    //                      [ %NextDestReg, NextMBB ]
    //   %ThisSrcReg = phi [ %StartSrcReg, StartMBB ],
    //                     [ %NextSrcReg, NextMBB ]
    //   %ThisCountReg = phi [ %StartCountReg, StartMBB ],
    //                       [ %NextCountReg, NextMBB ]
    //   ( PFD 2, 768+DestDisp(%ThisDestReg) )
    //   Opcode DestDisp(256,%ThisDestReg), SrcDisp(%ThisSrcReg)
    //   ( JLH EndMBB )
    //
    // The prefetch is used only for MVC.  The JLH is used only for CLC.
    MBB = LoopMBB;

    BuildMI(MBB, DL, TII->get(SystemZ::PHI), ThisDestReg)
      .addReg(StartDestReg).addMBB(StartMBB)
      .addReg(NextDestReg).addMBB(NextMBB);
    if (!HaveSingleBase)
      BuildMI(MBB, DL, TII->get(SystemZ::PHI), ThisSrcReg)
        .addReg(StartSrcReg).addMBB(StartMBB)
        .addReg(NextSrcReg).addMBB(NextMBB);
    BuildMI(MBB, DL, TII->get(SystemZ::PHI), ThisCountReg)
      .addReg(StartCountReg).addMBB(StartMBB)
      .addReg(NextCountReg).addMBB(NextMBB);
    if (Opcode == SystemZ::MVC)
      BuildMI(MBB, DL, TII->get(SystemZ::PFD))
        .addImm(SystemZ::PFD_WRITE)
        .addReg(ThisDestReg).addImm(DestDisp + 768).addReg(0);
    BuildMI(MBB, DL, TII->get(Opcode))
      .addReg(ThisDestReg).addImm(DestDisp).addImm(256)
      .addReg(ThisSrcReg).addImm(SrcDisp);
    if (EndMBB) {
      BuildMI(MBB, DL, TII->get(SystemZ::BRC))
        .addImm(SystemZ::CCMASK_ICMP).addImm(SystemZ::CCMASK_CMP_NE)
        .addMBB(EndMBB);
      MBB->addSuccessor(EndMBB);
      MBB->addSuccessor(NextMBB);
    }

    // NextMBB:
    //   %NextDestReg = LA 256(%ThisDestReg)
    //   %NextSrcReg = LA 256(%ThisSrcReg)
    //   %NextCountReg = AGHI %ThisCountReg, -1
    //   CGHI %NextCountReg, 0
    //   JLH LoopMBB
    //   # fall through to DoneMMB
    //
    // The AGHI, CGHI and JLH should be converted to BRCTG by later passes.
    MBB = NextMBB;

    BuildMI(MBB, DL, TII->get(SystemZ::LA), NextDestReg)
      .addReg(ThisDestReg).addImm(256).addReg(0);
    if (!HaveSingleBase)
      BuildMI(MBB, DL, TII->get(SystemZ::LA), NextSrcReg)
        .addReg(ThisSrcReg).addImm(256).addReg(0);
    BuildMI(MBB, DL, TII->get(SystemZ::AGHI), NextCountReg)
      .addReg(ThisCountReg).addImm(-1);
    BuildMI(MBB, DL, TII->get(SystemZ::CGHI))
      .addReg(NextCountReg).addImm(0);
    BuildMI(MBB, DL, TII->get(SystemZ::BRC))
      .addImm(SystemZ::CCMASK_ICMP).addImm(SystemZ::CCMASK_CMP_NE)
      .addMBB(LoopMBB);
    MBB->addSuccessor(LoopMBB);
    MBB->addSuccessor(DoneMBB);

    DestBase = MachineOperand::CreateReg(NextDestReg, false);
    SrcBase = MachineOperand::CreateReg(NextSrcReg, false);
    Length &= 255;
    MBB = DoneMBB;
  }
  // Handle any remaining bytes with straight-line code.
  while (Length > 0) {
    uint64_t ThisLength = std::min(Length, uint64_t(256));
    // The previous iteration might have created out-of-range displacements.
    // Apply them using LAY if so.
    if (!isUInt<12>(DestDisp)) {
      unsigned Reg = MRI.createVirtualRegister(&SystemZ::ADDR64BitRegClass);
      BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(SystemZ::LAY), Reg)
        .addOperand(DestBase).addImm(DestDisp).addReg(0);
      DestBase = MachineOperand::CreateReg(Reg, false);
      DestDisp = 0;
    }
    if (!isUInt<12>(SrcDisp)) {
      unsigned Reg = MRI.createVirtualRegister(&SystemZ::ADDR64BitRegClass);
      BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(SystemZ::LAY), Reg)
        .addOperand(SrcBase).addImm(SrcDisp).addReg(0);
      SrcBase = MachineOperand::CreateReg(Reg, false);
      SrcDisp = 0;
    }
    BuildMI(*MBB, MI, DL, TII->get(Opcode))
      .addOperand(DestBase).addImm(DestDisp).addImm(ThisLength)
      .addOperand(SrcBase).addImm(SrcDisp);
    DestDisp += ThisLength;
    SrcDisp += ThisLength;
    Length -= ThisLength;
    // If there's another CLC to go, branch to the end if a difference
    // was found.
    if (EndMBB && Length > 0) {
      MachineBasicBlock *NextMBB = splitBlockBefore(MI, MBB);
      BuildMI(MBB, DL, TII->get(SystemZ::BRC))
        .addImm(SystemZ::CCMASK_ICMP).addImm(SystemZ::CCMASK_CMP_NE)
        .addMBB(EndMBB);
      MBB->addSuccessor(EndMBB);
      MBB->addSuccessor(NextMBB);
      MBB = NextMBB;
    }
  }
  if (EndMBB) {
    MBB->addSuccessor(EndMBB);
    MBB = EndMBB;
    MBB->addLiveIn(SystemZ::CC);
  }

  MI->eraseFromParent();
  return MBB;
}

// Decompose string pseudo-instruction MI into a loop that continually performs
// Opcode until CC != 3.
MachineBasicBlock *
SystemZTargetLowering::emitStringWrapper(MachineInstr *MI,
                                         MachineBasicBlock *MBB,
                                         unsigned Opcode) const {
  const SystemZInstrInfo *TII = TM.getInstrInfo();
  MachineFunction &MF = *MBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DebugLoc DL = MI->getDebugLoc();

  uint64_t End1Reg   = MI->getOperand(0).getReg();
  uint64_t Start1Reg = MI->getOperand(1).getReg();
  uint64_t Start2Reg = MI->getOperand(2).getReg();
  uint64_t CharReg   = MI->getOperand(3).getReg();

  const TargetRegisterClass *RC = &SystemZ::GR64BitRegClass;
  uint64_t This1Reg = MRI.createVirtualRegister(RC);
  uint64_t This2Reg = MRI.createVirtualRegister(RC);
  uint64_t End2Reg  = MRI.createVirtualRegister(RC);

  MachineBasicBlock *StartMBB = MBB;
  MachineBasicBlock *DoneMBB = splitBlockBefore(MI, MBB);
  MachineBasicBlock *LoopMBB = emitBlockAfter(StartMBB);

  //  StartMBB:
  //   # fall through to LoopMMB
  MBB->addSuccessor(LoopMBB);

  //  LoopMBB:
  //   %This1Reg = phi [ %Start1Reg, StartMBB ], [ %End1Reg, LoopMBB ]
  //   %This2Reg = phi [ %Start2Reg, StartMBB ], [ %End2Reg, LoopMBB ]
  //   R0L = %CharReg
  //   %End1Reg, %End2Reg = CLST %This1Reg, %This2Reg -- uses R0L
  //   JO LoopMBB
  //   # fall through to DoneMMB
  //
  // The load of R0L can be hoisted by post-RA LICM.
  MBB = LoopMBB;

  BuildMI(MBB, DL, TII->get(SystemZ::PHI), This1Reg)
    .addReg(Start1Reg).addMBB(StartMBB)
    .addReg(End1Reg).addMBB(LoopMBB);
  BuildMI(MBB, DL, TII->get(SystemZ::PHI), This2Reg)
    .addReg(Start2Reg).addMBB(StartMBB)
    .addReg(End2Reg).addMBB(LoopMBB);
  BuildMI(MBB, DL, TII->get(TargetOpcode::COPY), SystemZ::R0L).addReg(CharReg);
  BuildMI(MBB, DL, TII->get(Opcode))
    .addReg(End1Reg, RegState::Define).addReg(End2Reg, RegState::Define)
    .addReg(This1Reg).addReg(This2Reg);
  BuildMI(MBB, DL, TII->get(SystemZ::BRC))
    .addImm(SystemZ::CCMASK_ANY).addImm(SystemZ::CCMASK_3).addMBB(LoopMBB);
  MBB->addSuccessor(LoopMBB);
  MBB->addSuccessor(DoneMBB);

  DoneMBB->addLiveIn(SystemZ::CC);

  MI->eraseFromParent();
  return DoneMBB;
}

MachineBasicBlock *SystemZTargetLowering::
EmitInstrWithCustomInserter(MachineInstr *MI, MachineBasicBlock *MBB) const {
  switch (MI->getOpcode()) {
  case SystemZ::Select32Mux:
  case SystemZ::Select32:
  case SystemZ::SelectF32:
  case SystemZ::Select64:
  case SystemZ::SelectF64:
  case SystemZ::SelectF128:
    return emitSelect(MI, MBB);

  case SystemZ::CondStore8Mux:
    return emitCondStore(MI, MBB, SystemZ::STCMux, 0, false);
  case SystemZ::CondStore8MuxInv:
    return emitCondStore(MI, MBB, SystemZ::STCMux, 0, true);
  case SystemZ::CondStore16Mux:
    return emitCondStore(MI, MBB, SystemZ::STHMux, 0, false);
  case SystemZ::CondStore16MuxInv:
    return emitCondStore(MI, MBB, SystemZ::STHMux, 0, true);
  case SystemZ::CondStore8:
    return emitCondStore(MI, MBB, SystemZ::STC, 0, false);
  case SystemZ::CondStore8Inv:
    return emitCondStore(MI, MBB, SystemZ::STC, 0, true);
  case SystemZ::CondStore16:
    return emitCondStore(MI, MBB, SystemZ::STH, 0, false);
  case SystemZ::CondStore16Inv:
    return emitCondStore(MI, MBB, SystemZ::STH, 0, true);
  case SystemZ::CondStore32:
    return emitCondStore(MI, MBB, SystemZ::ST, SystemZ::STOC, false);
  case SystemZ::CondStore32Inv:
    return emitCondStore(MI, MBB, SystemZ::ST, SystemZ::STOC, true);
  case SystemZ::CondStore64:
    return emitCondStore(MI, MBB, SystemZ::STG, SystemZ::STOCG, false);
  case SystemZ::CondStore64Inv:
    return emitCondStore(MI, MBB, SystemZ::STG, SystemZ::STOCG, true);
  case SystemZ::CondStoreF32:
    return emitCondStore(MI, MBB, SystemZ::STE, 0, false);
  case SystemZ::CondStoreF32Inv:
    return emitCondStore(MI, MBB, SystemZ::STE, 0, true);
  case SystemZ::CondStoreF64:
    return emitCondStore(MI, MBB, SystemZ::STD, 0, false);
  case SystemZ::CondStoreF64Inv:
    return emitCondStore(MI, MBB, SystemZ::STD, 0, true);

  case SystemZ::AEXT128_64:
    return emitExt128(MI, MBB, false, SystemZ::subreg_l64);
  case SystemZ::ZEXT128_32:
    return emitExt128(MI, MBB, true, SystemZ::subreg_l32);
  case SystemZ::ZEXT128_64:
    return emitExt128(MI, MBB, true, SystemZ::subreg_l64);

  case SystemZ::ATOMIC_SWAPW:
    return emitAtomicLoadBinary(MI, MBB, 0, 0);
  case SystemZ::ATOMIC_SWAP_32:
    return emitAtomicLoadBinary(MI, MBB, 0, 32);
  case SystemZ::ATOMIC_SWAP_64:
    return emitAtomicLoadBinary(MI, MBB, 0, 64);

  case SystemZ::ATOMIC_LOADW_AR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::AR, 0);
  case SystemZ::ATOMIC_LOADW_AFI:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::AFI, 0);
  case SystemZ::ATOMIC_LOAD_AR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::AR, 32);
  case SystemZ::ATOMIC_LOAD_AHI:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::AHI, 32);
  case SystemZ::ATOMIC_LOAD_AFI:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::AFI, 32);
  case SystemZ::ATOMIC_LOAD_AGR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::AGR, 64);
  case SystemZ::ATOMIC_LOAD_AGHI:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::AGHI, 64);
  case SystemZ::ATOMIC_LOAD_AGFI:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::AGFI, 64);

  case SystemZ::ATOMIC_LOADW_SR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::SR, 0);
  case SystemZ::ATOMIC_LOAD_SR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::SR, 32);
  case SystemZ::ATOMIC_LOAD_SGR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::SGR, 64);

  case SystemZ::ATOMIC_LOADW_NR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NR, 0);
  case SystemZ::ATOMIC_LOADW_NILH:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILH, 0);
  case SystemZ::ATOMIC_LOAD_NR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NR, 32);
  case SystemZ::ATOMIC_LOAD_NILL:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILL, 32);
  case SystemZ::ATOMIC_LOAD_NILH:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILH, 32);
  case SystemZ::ATOMIC_LOAD_NILF:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILF, 32);
  case SystemZ::ATOMIC_LOAD_NGR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NGR, 64);
  case SystemZ::ATOMIC_LOAD_NILL64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILL64, 64);
  case SystemZ::ATOMIC_LOAD_NILH64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILH64, 64);
  case SystemZ::ATOMIC_LOAD_NIHL64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NIHL64, 64);
  case SystemZ::ATOMIC_LOAD_NIHH64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NIHH64, 64);
  case SystemZ::ATOMIC_LOAD_NILF64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILF64, 64);
  case SystemZ::ATOMIC_LOAD_NIHF64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NIHF64, 64);

  case SystemZ::ATOMIC_LOADW_OR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OR, 0);
  case SystemZ::ATOMIC_LOADW_OILH:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OILH, 0);
  case SystemZ::ATOMIC_LOAD_OR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OR, 32);
  case SystemZ::ATOMIC_LOAD_OILL:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OILL, 32);
  case SystemZ::ATOMIC_LOAD_OILH:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OILH, 32);
  case SystemZ::ATOMIC_LOAD_OILF:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OILF, 32);
  case SystemZ::ATOMIC_LOAD_OGR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OGR, 64);
  case SystemZ::ATOMIC_LOAD_OILL64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OILL64, 64);
  case SystemZ::ATOMIC_LOAD_OILH64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OILH64, 64);
  case SystemZ::ATOMIC_LOAD_OIHL64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OIHL64, 64);
  case SystemZ::ATOMIC_LOAD_OIHH64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OIHH64, 64);
  case SystemZ::ATOMIC_LOAD_OILF64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OILF64, 64);
  case SystemZ::ATOMIC_LOAD_OIHF64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::OIHF64, 64);

  case SystemZ::ATOMIC_LOADW_XR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::XR, 0);
  case SystemZ::ATOMIC_LOADW_XILF:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::XILF, 0);
  case SystemZ::ATOMIC_LOAD_XR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::XR, 32);
  case SystemZ::ATOMIC_LOAD_XILF:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::XILF, 32);
  case SystemZ::ATOMIC_LOAD_XGR:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::XGR, 64);
  case SystemZ::ATOMIC_LOAD_XILF64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::XILF64, 64);
  case SystemZ::ATOMIC_LOAD_XIHF64:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::XIHF64, 64);

  case SystemZ::ATOMIC_LOADW_NRi:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NR, 0, true);
  case SystemZ::ATOMIC_LOADW_NILHi:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILH, 0, true);
  case SystemZ::ATOMIC_LOAD_NRi:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NR, 32, true);
  case SystemZ::ATOMIC_LOAD_NILLi:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILL, 32, true);
  case SystemZ::ATOMIC_LOAD_NILHi:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILH, 32, true);
  case SystemZ::ATOMIC_LOAD_NILFi:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILF, 32, true);
  case SystemZ::ATOMIC_LOAD_NGRi:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NGR, 64, true);
  case SystemZ::ATOMIC_LOAD_NILL64i:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILL64, 64, true);
  case SystemZ::ATOMIC_LOAD_NILH64i:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILH64, 64, true);
  case SystemZ::ATOMIC_LOAD_NIHL64i:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NIHL64, 64, true);
  case SystemZ::ATOMIC_LOAD_NIHH64i:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NIHH64, 64, true);
  case SystemZ::ATOMIC_LOAD_NILF64i:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NILF64, 64, true);
  case SystemZ::ATOMIC_LOAD_NIHF64i:
    return emitAtomicLoadBinary(MI, MBB, SystemZ::NIHF64, 64, true);

  case SystemZ::ATOMIC_LOADW_MIN:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CR,
                                SystemZ::CCMASK_CMP_LE, 0);
  case SystemZ::ATOMIC_LOAD_MIN_32:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CR,
                                SystemZ::CCMASK_CMP_LE, 32);
  case SystemZ::ATOMIC_LOAD_MIN_64:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CGR,
                                SystemZ::CCMASK_CMP_LE, 64);

  case SystemZ::ATOMIC_LOADW_MAX:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CR,
                                SystemZ::CCMASK_CMP_GE, 0);
  case SystemZ::ATOMIC_LOAD_MAX_32:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CR,
                                SystemZ::CCMASK_CMP_GE, 32);
  case SystemZ::ATOMIC_LOAD_MAX_64:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CGR,
                                SystemZ::CCMASK_CMP_GE, 64);

  case SystemZ::ATOMIC_LOADW_UMIN:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CLR,
                                SystemZ::CCMASK_CMP_LE, 0);
  case SystemZ::ATOMIC_LOAD_UMIN_32:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CLR,
                                SystemZ::CCMASK_CMP_LE, 32);
  case SystemZ::ATOMIC_LOAD_UMIN_64:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CLGR,
                                SystemZ::CCMASK_CMP_LE, 64);

  case SystemZ::ATOMIC_LOADW_UMAX:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CLR,
                                SystemZ::CCMASK_CMP_GE, 0);
  case SystemZ::ATOMIC_LOAD_UMAX_32:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CLR,
                                SystemZ::CCMASK_CMP_GE, 32);
  case SystemZ::ATOMIC_LOAD_UMAX_64:
    return emitAtomicLoadMinMax(MI, MBB, SystemZ::CLGR,
                                SystemZ::CCMASK_CMP_GE, 64);

  case SystemZ::ATOMIC_CMP_SWAPW:
    return emitAtomicCmpSwapW(MI, MBB);
  case SystemZ::MVCSequence:
  case SystemZ::MVCLoop:
    return emitMemMemWrapper(MI, MBB, SystemZ::MVC);
  case SystemZ::NCSequence:
  case SystemZ::NCLoop:
    return emitMemMemWrapper(MI, MBB, SystemZ::NC);
  case SystemZ::OCSequence:
  case SystemZ::OCLoop:
    return emitMemMemWrapper(MI, MBB, SystemZ::OC);
  case SystemZ::XCSequence:
  case SystemZ::XCLoop:
    return emitMemMemWrapper(MI, MBB, SystemZ::XC);
  case SystemZ::CLCSequence:
  case SystemZ::CLCLoop:
    return emitMemMemWrapper(MI, MBB, SystemZ::CLC);
  case SystemZ::CLSTLoop:
    return emitStringWrapper(MI, MBB, SystemZ::CLST);
  case SystemZ::MVSTLoop:
    return emitStringWrapper(MI, MBB, SystemZ::MVST);
  case SystemZ::SRSTLoop:
    return emitStringWrapper(MI, MBB, SystemZ::SRST);
  default:
    llvm_unreachable("Unexpected instr type to insert");
  }
}
