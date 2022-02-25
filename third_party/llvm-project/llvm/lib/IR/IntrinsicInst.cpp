//===-- IntrinsicInst.cpp - Intrinsic Instruction Wrappers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that make it really easy to deal with intrinsic
// functions.
//
// All intrinsic function calls are instances of the call instruction, so these
// are all subclasses of the CallInst class.  Note that none of these classes
// has state or virtual methods, which is an important part of this gross/neat
// hack working.
//
// In some cases, arguments to intrinsics need to be generic and are defined as
// type pointer to empty struct { }*.  To access the real item of interest the
// cast instruction needs to be stripped away.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IntrinsicInst.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Statepoint.h"

#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
/// DbgVariableIntrinsic - This is the common base class for debug info
/// intrinsics for variables.
///

iterator_range<DbgVariableIntrinsic::location_op_iterator>
DbgVariableIntrinsic::location_ops() const {
  auto *MD = getRawLocation();
  assert(MD && "First operand of DbgVariableIntrinsic should be non-null.");

  // If operand is ValueAsMetadata, return a range over just that operand.
  if (auto *VAM = dyn_cast<ValueAsMetadata>(MD)) {
    return {location_op_iterator(VAM), location_op_iterator(VAM + 1)};
  }
  // If operand is DIArgList, return a range over its args.
  if (auto *AL = dyn_cast<DIArgList>(MD))
    return {location_op_iterator(AL->args_begin()),
            location_op_iterator(AL->args_end())};
  // Operand must be an empty metadata tuple, so return empty iterator.
  return {location_op_iterator(static_cast<ValueAsMetadata *>(nullptr)),
          location_op_iterator(static_cast<ValueAsMetadata *>(nullptr))};
}

Value *DbgVariableIntrinsic::getVariableLocationOp(unsigned OpIdx) const {
  auto *MD = getRawLocation();
  assert(MD && "First operand of DbgVariableIntrinsic should be non-null.");
  if (auto *AL = dyn_cast<DIArgList>(MD))
    return AL->getArgs()[OpIdx]->getValue();
  if (isa<MDNode>(MD))
    return nullptr;
  assert(
      isa<ValueAsMetadata>(MD) &&
      "Attempted to get location operand from DbgVariableIntrinsic with none.");
  auto *V = cast<ValueAsMetadata>(MD);
  assert(OpIdx == 0 && "Operand Index must be 0 for a debug intrinsic with a "
                       "single location operand.");
  return V->getValue();
}

static ValueAsMetadata *getAsMetadata(Value *V) {
  return isa<MetadataAsValue>(V) ? dyn_cast<ValueAsMetadata>(
                                       cast<MetadataAsValue>(V)->getMetadata())
                                 : ValueAsMetadata::get(V);
}

void DbgVariableIntrinsic::replaceVariableLocationOp(Value *OldValue,
                                                     Value *NewValue) {
  assert(NewValue && "Values must be non-null");
  auto Locations = location_ops();
  auto OldIt = find(Locations, OldValue);
  assert(OldIt != Locations.end() && "OldValue must be a current location");
  if (!hasArgList()) {
    Value *NewOperand = isa<MetadataAsValue>(NewValue)
                            ? NewValue
                            : MetadataAsValue::get(
                                  getContext(), ValueAsMetadata::get(NewValue));
    return setArgOperand(0, NewOperand);
  }
  SmallVector<ValueAsMetadata *, 4> MDs;
  ValueAsMetadata *NewOperand = getAsMetadata(NewValue);
  for (auto *VMD : Locations)
    MDs.push_back(VMD == *OldIt ? NewOperand : getAsMetadata(VMD));
  setArgOperand(
      0, MetadataAsValue::get(getContext(), DIArgList::get(getContext(), MDs)));
}
void DbgVariableIntrinsic::replaceVariableLocationOp(unsigned OpIdx,
                                                     Value *NewValue) {
  assert(OpIdx < getNumVariableLocationOps() && "Invalid Operand Index");
  if (!hasArgList()) {
    Value *NewOperand = isa<MetadataAsValue>(NewValue)
                            ? NewValue
                            : MetadataAsValue::get(
                                  getContext(), ValueAsMetadata::get(NewValue));
    return setArgOperand(0, NewOperand);
  }
  SmallVector<ValueAsMetadata *, 4> MDs;
  ValueAsMetadata *NewOperand = getAsMetadata(NewValue);
  for (unsigned Idx = 0; Idx < getNumVariableLocationOps(); ++Idx)
    MDs.push_back(Idx == OpIdx ? NewOperand
                               : getAsMetadata(getVariableLocationOp(Idx)));
  setArgOperand(
      0, MetadataAsValue::get(getContext(), DIArgList::get(getContext(), MDs)));
}

void DbgVariableIntrinsic::addVariableLocationOps(ArrayRef<Value *> NewValues,
                                                  DIExpression *NewExpr) {
  assert(NewExpr->hasAllLocationOps(getNumVariableLocationOps() +
                                    NewValues.size()) &&
         "NewExpr for debug variable intrinsic does not reference every "
         "location operand.");
  assert(!is_contained(NewValues, nullptr) && "New values must be non-null");
  setArgOperand(2, MetadataAsValue::get(getContext(), NewExpr));
  SmallVector<ValueAsMetadata *, 4> MDs;
  for (auto *VMD : location_ops())
    MDs.push_back(getAsMetadata(VMD));
  for (auto *VMD : NewValues)
    MDs.push_back(getAsMetadata(VMD));
  setArgOperand(
      0, MetadataAsValue::get(getContext(), DIArgList::get(getContext(), MDs)));
}

Optional<uint64_t> DbgVariableIntrinsic::getFragmentSizeInBits() const {
  if (auto Fragment = getExpression()->getFragmentInfo())
    return Fragment->SizeInBits;
  return getVariable()->getSizeInBits();
}

int llvm::Intrinsic::lookupLLVMIntrinsicByName(ArrayRef<const char *> NameTable,
                                               StringRef Name) {
  assert(Name.startswith("llvm."));

  // Do successive binary searches of the dotted name components. For
  // "llvm.gc.experimental.statepoint.p1i8.p1i32", we will find the range of
  // intrinsics starting with "llvm.gc", then "llvm.gc.experimental", then
  // "llvm.gc.experimental.statepoint", and then we will stop as the range is
  // size 1. During the search, we can skip the prefix that we already know is
  // identical. By using strncmp we consider names with differing suffixes to
  // be part of the equal range.
  size_t CmpEnd = 4; // Skip the "llvm" component.
  const char *const *Low = NameTable.begin();
  const char *const *High = NameTable.end();
  const char *const *LastLow = Low;
  while (CmpEnd < Name.size() && High - Low > 0) {
    size_t CmpStart = CmpEnd;
    CmpEnd = Name.find('.', CmpStart + 1);
    CmpEnd = CmpEnd == StringRef::npos ? Name.size() : CmpEnd;
    auto Cmp = [CmpStart, CmpEnd](const char *LHS, const char *RHS) {
      return strncmp(LHS + CmpStart, RHS + CmpStart, CmpEnd - CmpStart) < 0;
    };
    LastLow = Low;
    std::tie(Low, High) = std::equal_range(Low, High, Name.data(), Cmp);
  }
  if (High - Low > 0)
    LastLow = Low;

  if (LastLow == NameTable.end())
    return -1;
  StringRef NameFound = *LastLow;
  if (Name == NameFound ||
      (Name.startswith(NameFound) && Name[NameFound.size()] == '.'))
    return LastLow - NameTable.begin();
  return -1;
}

Value *InstrProfIncrementInst::getStep() const {
  if (InstrProfIncrementInstStep::classof(this)) {
    return const_cast<Value *>(getArgOperand(4));
  }
  const Module *M = getModule();
  LLVMContext &Context = M->getContext();
  return ConstantInt::get(Type::getInt64Ty(Context), 1);
}

Optional<RoundingMode> ConstrainedFPIntrinsic::getRoundingMode() const {
  unsigned NumOperands = getNumArgOperands();
  Metadata *MD = nullptr;
  auto *MAV = dyn_cast<MetadataAsValue>(getArgOperand(NumOperands - 2));
  if (MAV)
    MD = MAV->getMetadata();
  if (!MD || !isa<MDString>(MD))
    return None;
  return StrToRoundingMode(cast<MDString>(MD)->getString());
}

Optional<fp::ExceptionBehavior>
ConstrainedFPIntrinsic::getExceptionBehavior() const {
  unsigned NumOperands = getNumArgOperands();
  Metadata *MD = nullptr;
  auto *MAV = dyn_cast<MetadataAsValue>(getArgOperand(NumOperands - 1));
  if (MAV)
    MD = MAV->getMetadata();
  if (!MD || !isa<MDString>(MD))
    return None;
  return StrToExceptionBehavior(cast<MDString>(MD)->getString());
}

bool ConstrainedFPIntrinsic::isDefaultFPEnvironment() const {
  Optional<fp::ExceptionBehavior> Except = getExceptionBehavior();
  if (Except) {
    if (Except.getValue() != fp::ebIgnore)
      return false;
  }

  Optional<RoundingMode> Rounding = getRoundingMode();
  if (Rounding) {
    if (Rounding.getValue() != RoundingMode::NearestTiesToEven)
      return false;
  }

  return true;
}

FCmpInst::Predicate ConstrainedFPCmpIntrinsic::getPredicate() const {
  Metadata *MD = cast<MetadataAsValue>(getArgOperand(2))->getMetadata();
  if (!MD || !isa<MDString>(MD))
    return FCmpInst::BAD_FCMP_PREDICATE;
  return StringSwitch<FCmpInst::Predicate>(cast<MDString>(MD)->getString())
      .Case("oeq", FCmpInst::FCMP_OEQ)
      .Case("ogt", FCmpInst::FCMP_OGT)
      .Case("oge", FCmpInst::FCMP_OGE)
      .Case("olt", FCmpInst::FCMP_OLT)
      .Case("ole", FCmpInst::FCMP_OLE)
      .Case("one", FCmpInst::FCMP_ONE)
      .Case("ord", FCmpInst::FCMP_ORD)
      .Case("uno", FCmpInst::FCMP_UNO)
      .Case("ueq", FCmpInst::FCMP_UEQ)
      .Case("ugt", FCmpInst::FCMP_UGT)
      .Case("uge", FCmpInst::FCMP_UGE)
      .Case("ult", FCmpInst::FCMP_ULT)
      .Case("ule", FCmpInst::FCMP_ULE)
      .Case("une", FCmpInst::FCMP_UNE)
      .Default(FCmpInst::BAD_FCMP_PREDICATE);
}

bool ConstrainedFPIntrinsic::isUnaryOp() const {
  switch (getIntrinsicID()) {
  default:
    return false;
#define INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC)                         \
  case Intrinsic::INTRINSIC:                                                   \
    return NARG == 1;
#include "llvm/IR/ConstrainedOps.def"
  }
}

bool ConstrainedFPIntrinsic::isTernaryOp() const {
  switch (getIntrinsicID()) {
  default:
    return false;
#define INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC)                         \
  case Intrinsic::INTRINSIC:                                                   \
    return NARG == 3;
#include "llvm/IR/ConstrainedOps.def"
  }
}

bool ConstrainedFPIntrinsic::classof(const IntrinsicInst *I) {
  switch (I->getIntrinsicID()) {
#define INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                        \
  case Intrinsic::INTRINSIC:
#include "llvm/IR/ConstrainedOps.def"
    return true;
  default:
    return false;
  }
}

ElementCount VPIntrinsic::getStaticVectorLength() const {
  auto GetVectorLengthOfType = [](const Type *T) -> ElementCount {
    const auto *VT = cast<VectorType>(T);
    auto ElemCount = VT->getElementCount();
    return ElemCount;
  };

  Value *VPMask = getMaskParam();
  assert(VPMask && "No mask param?");
  return GetVectorLengthOfType(VPMask->getType());
}

Value *VPIntrinsic::getMaskParam() const {
  if (auto MaskPos = getMaskParamPos(getIntrinsicID()))
    return getArgOperand(MaskPos.getValue());
  return nullptr;
}

void VPIntrinsic::setMaskParam(Value *NewMask) {
  auto MaskPos = getMaskParamPos(getIntrinsicID());
  setArgOperand(*MaskPos, NewMask);
}

Value *VPIntrinsic::getVectorLengthParam() const {
  if (auto EVLPos = getVectorLengthParamPos(getIntrinsicID()))
    return getArgOperand(EVLPos.getValue());
  return nullptr;
}

void VPIntrinsic::setVectorLengthParam(Value *NewEVL) {
  auto EVLPos = getVectorLengthParamPos(getIntrinsicID());
  setArgOperand(*EVLPos, NewEVL);
}

Optional<unsigned> VPIntrinsic::getMaskParamPos(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  default:
    return None;

#define BEGIN_REGISTER_VP_INTRINSIC(VPID, MASKPOS, VLENPOS)                    \
  case Intrinsic::VPID:                                                        \
    return MASKPOS;
#include "llvm/IR/VPIntrinsics.def"
  }
}

Optional<unsigned>
VPIntrinsic::getVectorLengthParamPos(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  default:
    return None;

#define BEGIN_REGISTER_VP_INTRINSIC(VPID, MASKPOS, VLENPOS)                    \
  case Intrinsic::VPID:                                                        \
    return VLENPOS;
#include "llvm/IR/VPIntrinsics.def"
  }
}

/// \return the alignment of the pointer used by this load/store/gather or
/// scatter.
MaybeAlign VPIntrinsic::getPointerAlignment() const {
  Optional<unsigned> PtrParamOpt = getMemoryPointerParamPos(getIntrinsicID());
  assert(PtrParamOpt.hasValue() && "no pointer argument!");
  return getParamAlign(PtrParamOpt.getValue());
}

/// \return The pointer operand of this load,store, gather or scatter.
Value *VPIntrinsic::getMemoryPointerParam() const {
  if (auto PtrParamOpt = getMemoryPointerParamPos(getIntrinsicID()))
    return getArgOperand(PtrParamOpt.getValue());
  return nullptr;
}

Optional<unsigned> VPIntrinsic::getMemoryPointerParamPos(Intrinsic::ID VPID) {
  switch (VPID) {
  default:
    return None;

#define HANDLE_VP_IS_MEMOP(VPID, POINTERPOS, DATAPOS)                          \
  case Intrinsic::VPID:                                                        \
    return POINTERPOS;
#include "llvm/IR/VPIntrinsics.def"
  }
}

/// \return The data (payload) operand of this store or scatter.
Value *VPIntrinsic::getMemoryDataParam() const {
  auto DataParamOpt = getMemoryDataParamPos(getIntrinsicID());
  if (!DataParamOpt.hasValue())
    return nullptr;
  return getArgOperand(DataParamOpt.getValue());
}

Optional<unsigned> VPIntrinsic::getMemoryDataParamPos(Intrinsic::ID VPID) {
  switch (VPID) {
  default:
    return None;

#define HANDLE_VP_IS_MEMOP(VPID, POINTERPOS, DATAPOS)                          \
  case Intrinsic::VPID:                                                        \
    return DATAPOS;
#include "llvm/IR/VPIntrinsics.def"
  }
}

bool VPIntrinsic::isVPIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  default:
    return false;

#define BEGIN_REGISTER_VP_INTRINSIC(VPID, MASKPOS, VLENPOS)                    \
  case Intrinsic::VPID:                                                        \
    break;
#include "llvm/IR/VPIntrinsics.def"
  }
  return true;
}

// Equivalent non-predicated opcode
Optional<unsigned> VPIntrinsic::getFunctionalOpcodeForVP(Intrinsic::ID ID) {
  Optional<unsigned> FunctionalOC;
  switch (ID) {
  default:
    break;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...) case Intrinsic::VPID:
#define HANDLE_VP_TO_OPC(OPC) FunctionalOC = Instruction::OPC;
#define END_REGISTER_VP_INTRINSIC(...) break;
#include "llvm/IR/VPIntrinsics.def"
  }

  return FunctionalOC;
}

Intrinsic::ID VPIntrinsic::getForOpcode(unsigned IROPC) {
  switch (IROPC) {
  default:
    return Intrinsic::not_intrinsic;

#define HANDLE_VP_TO_OPC(OPC) case Instruction::OPC:
#define END_REGISTER_VP_INTRINSIC(VPID) return Intrinsic::VPID;
#include "llvm/IR/VPIntrinsics.def"
  }
}

bool VPIntrinsic::canIgnoreVectorLengthParam() const {
  using namespace PatternMatch;

  ElementCount EC = getStaticVectorLength();

  // No vlen param - no lanes masked-off by it.
  auto *VLParam = getVectorLengthParam();
  if (!VLParam)
    return true;

  // Note that the VP intrinsic causes undefined behavior if the Explicit Vector
  // Length parameter is strictly greater-than the number of vector elements of
  // the operation. This function returns true when this is detected statically
  // in the IR.

  // Check whether "W == vscale * EC.getKnownMinValue()"
  if (EC.isScalable()) {
    // Undig the DL
    const auto *ParMod = this->getModule();
    if (!ParMod)
      return false;
    const auto &DL = ParMod->getDataLayout();

    // Compare vscale patterns
    uint64_t VScaleFactor;
    if (match(VLParam, m_c_Mul(m_ConstantInt(VScaleFactor), m_VScale(DL))))
      return VScaleFactor >= EC.getKnownMinValue();
    return (EC.getKnownMinValue() == 1) && match(VLParam, m_VScale(DL));
  }

  // standard SIMD operation
  const auto *VLConst = dyn_cast<ConstantInt>(VLParam);
  if (!VLConst)
    return false;

  uint64_t VLNum = VLConst->getZExtValue();
  if (VLNum >= EC.getKnownMinValue())
    return true;

  return false;
}

Function *VPIntrinsic::getDeclarationForParams(Module *M, Intrinsic::ID VPID,
                                               ArrayRef<Value *> Params) {
  assert(isVPIntrinsic(VPID) && "not a VP intrinsic");
  Function *VPFunc;
  switch (VPID) {
  default: {
    Type *OverloadTy = Params[0]->getType();
    if (VPReductionIntrinsic::isVPReduction(VPID))
      OverloadTy =
          Params[*VPReductionIntrinsic::getVectorParamPos(VPID)]->getType();

    VPFunc = Intrinsic::getDeclaration(M, VPID, OverloadTy);
    break;
  }
  case Intrinsic::vp_load:
    VPFunc = Intrinsic::getDeclaration(
        M, VPID,
        {Params[0]->getType()->getPointerElementType(), Params[0]->getType()});
    break;
  case Intrinsic::vp_gather:
    VPFunc = Intrinsic::getDeclaration(
        M, VPID,
        {VectorType::get(cast<VectorType>(Params[0]->getType())
                             ->getElementType()
                             ->getPointerElementType(),
                         cast<VectorType>(Params[0]->getType())),
         Params[0]->getType()});
    break;
  case Intrinsic::vp_store:
    VPFunc = Intrinsic::getDeclaration(
        M, VPID,
        {Params[1]->getType()->getPointerElementType(), Params[1]->getType()});
    break;
  case Intrinsic::vp_scatter:
    VPFunc = Intrinsic::getDeclaration(
        M, VPID, {Params[0]->getType(), Params[1]->getType()});
    break;
  }
  assert(VPFunc && "Could not declare VP intrinsic");
  return VPFunc;
}

bool VPReductionIntrinsic::isVPReduction(Intrinsic::ID ID) {
  switch (ID) {
  default:
    return false;
#define HANDLE_VP_REDUCTION(VPID, STARTPOS, VECTORPOS)                         \
  case Intrinsic::VPID:                                                        \
    break;
#include "llvm/IR/VPIntrinsics.def"
  }
  return true;
}

unsigned VPReductionIntrinsic::getVectorParamPos() const {
  return *VPReductionIntrinsic::getVectorParamPos(getIntrinsicID());
}

unsigned VPReductionIntrinsic::getStartParamPos() const {
  return *VPReductionIntrinsic::getStartParamPos(getIntrinsicID());
}

Optional<unsigned> VPReductionIntrinsic::getVectorParamPos(Intrinsic::ID ID) {
  switch (ID) {
#define HANDLE_VP_REDUCTION(VPID, STARTPOS, VECTORPOS)                         \
  case Intrinsic::VPID:                                                        \
    return VECTORPOS;
#include "llvm/IR/VPIntrinsics.def"
  default:
    return None;
  }
}

Optional<unsigned> VPReductionIntrinsic::getStartParamPos(Intrinsic::ID ID) {
  switch (ID) {
#define HANDLE_VP_REDUCTION(VPID, STARTPOS, VECTORPOS)                         \
  case Intrinsic::VPID:                                                        \
    return STARTPOS;
#include "llvm/IR/VPIntrinsics.def"
  default:
    return None;
  }
}

Instruction::BinaryOps BinaryOpIntrinsic::getBinaryOp() const {
  switch (getIntrinsicID()) {
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_sat:
  case Intrinsic::sadd_sat:
    return Instruction::Add;
  case Intrinsic::usub_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_sat:
  case Intrinsic::ssub_sat:
    return Instruction::Sub;
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow:
    return Instruction::Mul;
  default:
    llvm_unreachable("Invalid intrinsic");
  }
}

bool BinaryOpIntrinsic::isSigned() const {
  switch (getIntrinsicID()) {
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::sadd_sat:
  case Intrinsic::ssub_sat:
    return true;
  default:
    return false;
  }
}

unsigned BinaryOpIntrinsic::getNoWrapKind() const {
  if (isSigned())
    return OverflowingBinaryOperator::NoSignedWrap;
  else
    return OverflowingBinaryOperator::NoUnsignedWrap;
}

const GCStatepointInst *GCProjectionInst::getStatepoint() const {
  const Value *Token = getArgOperand(0);

  // This takes care both of relocates for call statepoints and relocates
  // on normal path of invoke statepoint.
  if (!isa<LandingPadInst>(Token))
    return cast<GCStatepointInst>(Token);

  // This relocate is on exceptional path of an invoke statepoint
  const BasicBlock *InvokeBB =
    cast<Instruction>(Token)->getParent()->getUniquePredecessor();

  assert(InvokeBB && "safepoints should have unique landingpads");
  assert(InvokeBB->getTerminator() &&
         "safepoint block should be well formed");

  return cast<GCStatepointInst>(InvokeBB->getTerminator());
}

Value *GCRelocateInst::getBasePtr() const {
  if (auto Opt = getStatepoint()->getOperandBundle(LLVMContext::OB_gc_live))
    return *(Opt->Inputs.begin() + getBasePtrIndex());
  return *(getStatepoint()->arg_begin() + getBasePtrIndex());
}

Value *GCRelocateInst::getDerivedPtr() const {
  if (auto Opt = getStatepoint()->getOperandBundle(LLVMContext::OB_gc_live))
    return *(Opt->Inputs.begin() + getDerivedPtrIndex());
  return *(getStatepoint()->arg_begin() + getDerivedPtrIndex());
}
