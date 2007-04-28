//===-- llvm/Target/TargetLowering.h - Target Lowering Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes how to lower LLVM code to machine code.  This has two
// main components:
//
//  1. Which ValueTypes are natively supported by the target.
//  2. Which operations are supported for supported ValueTypes.
//  3. Cost thresholds for alternative implementations of certain operations.
//
// In addition it has a few other components, like information about FP
// immediates.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETLOWERING_H
#define LLVM_TARGET_TARGETLOWERING_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include <map>
#include <vector>

namespace llvm {
  class Value;
  class Function;
  class TargetMachine;
  class TargetData;
  class TargetRegisterClass;
  class SDNode;
  class SDOperand;
  class SelectionDAG;
  class MachineBasicBlock;
  class MachineInstr;
  class VectorType;

//===----------------------------------------------------------------------===//
/// TargetLowering - This class defines information used to lower LLVM code to
/// legal SelectionDAG operators that the target instruction selector can accept
/// natively.
///
/// This class also defines callbacks that targets must implement to lower
/// target-specific constructs to SelectionDAG operators.
///
class TargetLowering {
public:
  /// LegalizeAction - This enum indicates whether operations are valid for a
  /// target, and if not, what action should be used to make them valid.
  enum LegalizeAction {
    Legal,      // The target natively supports this operation.
    Promote,    // This operation should be executed in a larger type.
    Expand,     // Try to expand this to other ops, otherwise use a libcall.
    Custom      // Use the LowerOperation hook to implement custom lowering.
  };

  enum OutOfRangeShiftAmount {
    Undefined,  // Oversized shift amounts are undefined (default).
    Mask,       // Shift amounts are auto masked (anded) to value size.
    Extend      // Oversized shift pulls in zeros or sign bits.
  };

  enum SetCCResultValue {
    UndefinedSetCCResult,          // SetCC returns a garbage/unknown extend.
    ZeroOrOneSetCCResult,          // SetCC returns a zero extended result.
    ZeroOrNegativeOneSetCCResult   // SetCC returns a sign extended result.
  };

  enum SchedPreference {
    SchedulingForLatency,          // Scheduling for shortest total latency.
    SchedulingForRegPressure       // Scheduling for lowest register pressure.
  };

  TargetLowering(TargetMachine &TM);
  virtual ~TargetLowering();

  TargetMachine &getTargetMachine() const { return TM; }
  const TargetData *getTargetData() const { return TD; }

  bool isLittleEndian() const { return IsLittleEndian; }
  MVT::ValueType getPointerTy() const { return PointerTy; }
  MVT::ValueType getShiftAmountTy() const { return ShiftAmountTy; }
  OutOfRangeShiftAmount getShiftAmountFlavor() const {return ShiftAmtHandling; }

  /// usesGlobalOffsetTable - Return true if this target uses a GOT for PIC
  /// codegen.
  bool usesGlobalOffsetTable() const { return UsesGlobalOffsetTable; }
  
  /// isSelectExpensive - Return true if the select operation is expensive for
  /// this target.
  bool isSelectExpensive() const { return SelectIsExpensive; }
  
  /// isIntDivCheap() - Return true if integer divide is usually cheaper than
  /// a sequence of several shifts, adds, and multiplies for this target.
  bool isIntDivCheap() const { return IntDivIsCheap; }

  /// isPow2DivCheap() - Return true if pow2 div is cheaper than a chain of
  /// srl/add/sra.
  bool isPow2DivCheap() const { return Pow2DivIsCheap; }
  
  /// getSetCCResultTy - Return the ValueType of the result of setcc operations.
  ///
  MVT::ValueType getSetCCResultTy() const { return SetCCResultTy; }

  /// getSetCCResultContents - For targets without boolean registers, this flag
  /// returns information about the contents of the high-bits in the setcc
  /// result register.
  SetCCResultValue getSetCCResultContents() const { return SetCCResultContents;}

  /// getSchedulingPreference - Return target scheduling preference.
  SchedPreference getSchedulingPreference() const {
    return SchedPreferenceInfo;
  }

  /// getRegClassFor - Return the register class that should be used for the
  /// specified value type.  This may only be called on legal types.
  TargetRegisterClass *getRegClassFor(MVT::ValueType VT) const {
    TargetRegisterClass *RC = RegClassForVT[VT];
    assert(RC && "This value type is not natively supported!");
    return RC;
  }
  
  /// isTypeLegal - Return true if the target has native support for the
  /// specified value type.  This means that it has a register that directly
  /// holds it without promotions or expansions.
  bool isTypeLegal(MVT::ValueType VT) const {
    return RegClassForVT[VT] != 0;
  }

  class ValueTypeActionImpl {
    /// ValueTypeActions - This is a bitvector that contains two bits for each
    /// value type, where the two bits correspond to the LegalizeAction enum.
    /// This can be queried with "getTypeAction(VT)".
    uint32_t ValueTypeActions[2];
  public:
    ValueTypeActionImpl() {
      ValueTypeActions[0] = ValueTypeActions[1] = 0;
    }
    ValueTypeActionImpl(const ValueTypeActionImpl &RHS) {
      ValueTypeActions[0] = RHS.ValueTypeActions[0];
      ValueTypeActions[1] = RHS.ValueTypeActions[1];
    }
    
    LegalizeAction getTypeAction(MVT::ValueType VT) const {
      return (LegalizeAction)((ValueTypeActions[VT>>4] >> ((2*VT) & 31)) & 3);
    }
    void setTypeAction(MVT::ValueType VT, LegalizeAction Action) {
      assert(unsigned(VT >> 4) < 
             sizeof(ValueTypeActions)/sizeof(ValueTypeActions[0]));
      ValueTypeActions[VT>>4] |= Action << ((VT*2) & 31);
    }
  };
  
  const ValueTypeActionImpl &getValueTypeActions() const {
    return ValueTypeActions;
  }
  
  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal (return 'Legal') or we need to promote it to a larger
  /// type (return 'Promote'), or we need to expand it into multiple registers
  /// of smaller integer type (return 'Expand').  'Custom' is not an option.
  LegalizeAction getTypeAction(MVT::ValueType VT) const {
    return ValueTypeActions.getTypeAction(VT);
  }

  /// getTypeToTransformTo - For types supported by the target, this is an
  /// identity function.  For types that must be promoted to larger types, this
  /// returns the larger type to promote to.  For integer types that are larger
  /// than the largest integer register, this contains one step in the expansion
  /// to get to the smaller register. For illegal floating point types, this
  /// returns the integer type to transform to.
  MVT::ValueType getTypeToTransformTo(MVT::ValueType VT) const {
    return TransformToType[VT];
  }
  
  /// getTypeToExpandTo - For types supported by the target, this is an
  /// identity function.  For types that must be expanded (i.e. integer types
  /// that are larger than the largest integer register or illegal floating
  /// point types), this returns the largest legal type it will be expanded to.
  MVT::ValueType getTypeToExpandTo(MVT::ValueType VT) const {
    while (true) {
      switch (getTypeAction(VT)) {
      case Legal:
        return VT;
      case Expand:
        VT = TransformToType[VT];
        break;
      default:
        assert(false && "Type is not legal nor is it to be expanded!");
        return VT;
      }
    }
    return VT;
  }

  /// getVectorTypeBreakdown - Vector types are broken down into some number of
  /// legal first class types.  For example, <8 x float> maps to 2 MVT::v4f32
  /// with Altivec or SSE1, or 8 promoted MVT::f64 values with the X86 FP stack.
  /// Similarly, <2 x long> turns into 4 MVT::i32 values with both PPC and X86.
  ///
  /// This method returns the number of registers needed, and the VT for each
  /// register.  It also returns the VT of the VectorType elements before they
  /// are promoted/expanded.
  ///
  unsigned getVectorTypeBreakdown(const VectorType *PTy, 
                                  MVT::ValueType &PTyElementVT,
                                  MVT::ValueType &PTyLegalElementVT) const;
  
  typedef std::vector<double>::const_iterator legal_fpimm_iterator;
  legal_fpimm_iterator legal_fpimm_begin() const {
    return LegalFPImmediates.begin();
  }
  legal_fpimm_iterator legal_fpimm_end() const {
    return LegalFPImmediates.end();
  }
  
  /// isShuffleMaskLegal - Targets can use this to indicate that they only
  /// support *some* VECTOR_SHUFFLE operations, those with specific masks.
  /// By default, if a target supports the VECTOR_SHUFFLE node, all mask values
  /// are assumed to be legal.
  virtual bool isShuffleMaskLegal(SDOperand Mask, MVT::ValueType VT) const {
    return true;
  }

  /// isVectorClearMaskLegal - Similar to isShuffleMaskLegal. This is
  /// used by Targets can use this to indicate if there is a suitable
  /// VECTOR_SHUFFLE that can be used to replace a VAND with a constant
  /// pool entry.
  virtual bool isVectorClearMaskLegal(std::vector<SDOperand> &BVOps,
                                      MVT::ValueType EVT,
                                      SelectionDAG &DAG) const {
    return false;
  }

  /// getOperationAction - Return how this operation should be treated: either
  /// it is legal, needs to be promoted to a larger size, needs to be
  /// expanded to some other code sequence, or the target has a custom expander
  /// for it.
  LegalizeAction getOperationAction(unsigned Op, MVT::ValueType VT) const {
    return (LegalizeAction)((OpActions[Op] >> (2*VT)) & 3);
  }
  
  /// isOperationLegal - Return true if the specified operation is legal on this
  /// target.
  bool isOperationLegal(unsigned Op, MVT::ValueType VT) const {
    return getOperationAction(Op, VT) == Legal ||
           getOperationAction(Op, VT) == Custom;
  }
  
  /// getLoadXAction - Return how this load with extension should be treated:
  /// either it is legal, needs to be promoted to a larger size, needs to be
  /// expanded to some other code sequence, or the target has a custom expander
  /// for it.
  LegalizeAction getLoadXAction(unsigned LType, MVT::ValueType VT) const {
    return (LegalizeAction)((LoadXActions[LType] >> (2*VT)) & 3);
  }
  
  /// isLoadXLegal - Return true if the specified load with extension is legal
  /// on this target.
  bool isLoadXLegal(unsigned LType, MVT::ValueType VT) const {
    return getLoadXAction(LType, VT) == Legal ||
           getLoadXAction(LType, VT) == Custom;
  }
  
  /// getStoreXAction - Return how this store with truncation should be treated:
  /// either it is legal, needs to be promoted to a larger size, needs to be
  /// expanded to some other code sequence, or the target has a custom expander
  /// for it.
  LegalizeAction getStoreXAction(MVT::ValueType VT) const {
    return (LegalizeAction)((StoreXActions >> (2*VT)) & 3);
  }
  
  /// isStoreXLegal - Return true if the specified store with truncation is
  /// legal on this target.
  bool isStoreXLegal(MVT::ValueType VT) const {
    return getStoreXAction(VT) == Legal || getStoreXAction(VT) == Custom;
  }

  /// getIndexedLoadAction - Return how the indexed load should be treated:
  /// either it is legal, needs to be promoted to a larger size, needs to be
  /// expanded to some other code sequence, or the target has a custom expander
  /// for it.
  LegalizeAction
  getIndexedLoadAction(unsigned IdxMode, MVT::ValueType VT) const {
    return (LegalizeAction)((IndexedModeActions[0][IdxMode] >> (2*VT)) & 3);
  }

  /// isIndexedLoadLegal - Return true if the specified indexed load is legal
  /// on this target.
  bool isIndexedLoadLegal(unsigned IdxMode, MVT::ValueType VT) const {
    return getIndexedLoadAction(IdxMode, VT) == Legal ||
           getIndexedLoadAction(IdxMode, VT) == Custom;
  }
  
  /// getIndexedStoreAction - Return how the indexed store should be treated:
  /// either it is legal, needs to be promoted to a larger size, needs to be
  /// expanded to some other code sequence, or the target has a custom expander
  /// for it.
  LegalizeAction
  getIndexedStoreAction(unsigned IdxMode, MVT::ValueType VT) const {
    return (LegalizeAction)((IndexedModeActions[1][IdxMode] >> (2*VT)) & 3);
  }  
  
  /// isIndexedStoreLegal - Return true if the specified indexed load is legal
  /// on this target.
  bool isIndexedStoreLegal(unsigned IdxMode, MVT::ValueType VT) const {
    return getIndexedStoreAction(IdxMode, VT) == Legal ||
           getIndexedStoreAction(IdxMode, VT) == Custom;
  }
  
  /// getTypeToPromoteTo - If the action for this operation is to promote, this
  /// method returns the ValueType to promote to.
  MVT::ValueType getTypeToPromoteTo(unsigned Op, MVT::ValueType VT) const {
    assert(getOperationAction(Op, VT) == Promote &&
           "This operation isn't promoted!");

    // See if this has an explicit type specified.
    std::map<std::pair<unsigned, MVT::ValueType>, 
             MVT::ValueType>::const_iterator PTTI =
      PromoteToType.find(std::make_pair(Op, VT));
    if (PTTI != PromoteToType.end()) return PTTI->second;
    
    assert((MVT::isInteger(VT) || MVT::isFloatingPoint(VT)) &&
           "Cannot autopromote this type, add it with AddPromotedToType.");
    
    MVT::ValueType NVT = VT;
    do {
      NVT = (MVT::ValueType)(NVT+1);
      assert(MVT::isInteger(NVT) == MVT::isInteger(VT) && NVT != MVT::isVoid &&
             "Didn't find type to promote to!");
    } while (!isTypeLegal(NVT) ||
              getOperationAction(Op, NVT) == Promote);
    return NVT;
  }

  /// getValueType - Return the MVT::ValueType corresponding to this LLVM type.
  /// This is fixed by the LLVM operations except for the pointer size.  If
  /// AllowUnknown is true, this will return MVT::Other for types with no MVT
  /// counterpart (e.g. structs), otherwise it will assert.
  MVT::ValueType getValueType(const Type *Ty, bool AllowUnknown = false) const {
    MVT::ValueType VT = MVT::getValueType(Ty, AllowUnknown);
    return VT == MVT::iPTR ? PointerTy : VT;
  }

  /// getNumElements - Return the number of registers that this ValueType will
  /// eventually require.  This is one for any types promoted to live in larger
  /// registers, but may be more than one for types (like i64) that are split
  /// into pieces.
  unsigned getNumElements(MVT::ValueType VT) const {
    return NumElementsForVT[VT];
  }
  
  /// hasTargetDAGCombine - If true, the target has custom DAG combine
  /// transformations that it can perform for the specified node.
  bool hasTargetDAGCombine(ISD::NodeType NT) const {
    return TargetDAGCombineArray[NT >> 3] & (1 << (NT&7));
  }

  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memset. The value is set by the target at the
  /// performance threshold for such a replacement.
  /// @brief Get maximum # of store operations permitted for llvm.memset
  unsigned getMaxStoresPerMemset() const { return maxStoresPerMemset; }

  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memcpy. The value is set by the target at the
  /// performance threshold for such a replacement.
  /// @brief Get maximum # of store operations permitted for llvm.memcpy
  unsigned getMaxStoresPerMemcpy() const { return maxStoresPerMemcpy; }

  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memmove. The value is set by the target at the
  /// performance threshold for such a replacement.
  /// @brief Get maximum # of store operations permitted for llvm.memmove
  unsigned getMaxStoresPerMemmove() const { return maxStoresPerMemmove; }

  /// This function returns true if the target allows unaligned memory accesses.
  /// This is used, for example, in situations where an array copy/move/set is 
  /// converted to a sequence of store operations. It's use helps to ensure that
  /// such replacements don't generate code that causes an alignment error 
  /// (trap) on the target machine. 
  /// @brief Determine if the target supports unaligned memory accesses.
  bool allowsUnalignedMemoryAccesses() const {
    return allowUnalignedMemoryAccesses;
  }
  
  /// usesUnderscoreSetJmp - Determine if we should use _setjmp or setjmp
  /// to implement llvm.setjmp.
  bool usesUnderscoreSetJmp() const {
    return UseUnderscoreSetJmp;
  }

  /// usesUnderscoreLongJmp - Determine if we should use _longjmp or longjmp
  /// to implement llvm.longjmp.
  bool usesUnderscoreLongJmp() const {
    return UseUnderscoreLongJmp;
  }

  /// getStackPointerRegisterToSaveRestore - If a physical register, this
  /// specifies the register that llvm.savestack/llvm.restorestack should save
  /// and restore.
  unsigned getStackPointerRegisterToSaveRestore() const {
    return StackPointerRegisterToSaveRestore;
  }

  /// getExceptionAddressRegister - If a physical register, this returns
  /// the register that receives the exception address on entry to a landing
  /// pad.
  unsigned getExceptionAddressRegister() const {
    return ExceptionPointerRegister;
  }

  /// getExceptionSelectorRegister - If a physical register, this returns
  /// the register that receives the exception typeid on entry to a landing
  /// pad.
  unsigned getExceptionSelectorRegister() const {
    return ExceptionSelectorRegister;
  }

  /// getJumpBufSize - returns the target's jmp_buf size in bytes (if never
  /// set, the default is 200)
  unsigned getJumpBufSize() const {
    return JumpBufSize;
  }

  /// getJumpBufAlignment - returns the target's jmp_buf alignment in bytes
  /// (if never set, the default is 0)
  unsigned getJumpBufAlignment() const {
    return JumpBufAlignment;
  }

  /// getPreIndexedAddressParts - returns true by value, base pointer and
  /// offset pointer and addressing mode by reference if the node's address
  /// can be legally represented as pre-indexed load / store address.
  virtual bool getPreIndexedAddressParts(SDNode *N, SDOperand &Base,
                                         SDOperand &Offset,
                                         ISD::MemIndexedMode &AM,
                                         SelectionDAG &DAG) {
    return false;
  }
  
  /// getPostIndexedAddressParts - returns true by value, base pointer and
  /// offset pointer and addressing mode by reference if this node can be
  /// combined with a load / store to form a post-indexed load / store.
  virtual bool getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                          SDOperand &Base, SDOperand &Offset,
                                          ISD::MemIndexedMode &AM,
                                          SelectionDAG &DAG) {
    return false;
  }
  
  //===--------------------------------------------------------------------===//
  // TargetLowering Optimization Methods
  //
  
  /// TargetLoweringOpt - A convenience struct that encapsulates a DAG, and two
  /// SDOperands for returning information from TargetLowering to its clients
  /// that want to combine 
  struct TargetLoweringOpt {
    SelectionDAG &DAG;
    SDOperand Old;
    SDOperand New;

    TargetLoweringOpt(SelectionDAG &InDAG) : DAG(InDAG) {}
    
    bool CombineTo(SDOperand O, SDOperand N) { 
      Old = O; 
      New = N; 
      return true;
    }
    
    /// ShrinkDemandedConstant - Check to see if the specified operand of the 
    /// specified instruction is a constant integer.  If so, check to see if there
    /// are any bits set in the constant that are not demanded.  If so, shrink the
    /// constant and return true.
    bool ShrinkDemandedConstant(SDOperand Op, uint64_t Demanded);
  };
                                                
  /// MaskedValueIsZero - Return true if 'Op & Mask' is known to be zero.  We
  /// use this predicate to simplify operations downstream.  Op and Mask are
  /// known to be the same type.
  bool MaskedValueIsZero(SDOperand Op, uint64_t Mask, unsigned Depth = 0)
    const;
  
  /// ComputeMaskedBits - Determine which of the bits specified in Mask are
  /// known to be either zero or one and return them in the KnownZero/KnownOne
  /// bitsets.  This code only analyzes bits in Mask, in order to short-circuit
  /// processing.  Targets can implement the computeMaskedBitsForTargetNode 
  /// method, to allow target nodes to be understood.
  void ComputeMaskedBits(SDOperand Op, uint64_t Mask, uint64_t &KnownZero,
                         uint64_t &KnownOne, unsigned Depth = 0) const;
    
  /// SimplifyDemandedBits - Look at Op.  At this point, we know that only the
  /// DemandedMask bits of the result of Op are ever used downstream.  If we can
  /// use this information to simplify Op, create a new simplified DAG node and
  /// return true, returning the original and new nodes in Old and New. 
  /// Otherwise, analyze the expression and return a mask of KnownOne and 
  /// KnownZero bits for the expression (used to simplify the caller).  
  /// The KnownZero/One bits may only be accurate for those bits in the 
  /// DemandedMask.
  bool SimplifyDemandedBits(SDOperand Op, uint64_t DemandedMask, 
                            uint64_t &KnownZero, uint64_t &KnownOne,
                            TargetLoweringOpt &TLO, unsigned Depth = 0) const;
  
  /// computeMaskedBitsForTargetNode - Determine which of the bits specified in
  /// Mask are known to be either zero or one and return them in the 
  /// KnownZero/KnownOne bitsets.
  virtual void computeMaskedBitsForTargetNode(const SDOperand Op,
                                              uint64_t Mask,
                                              uint64_t &KnownZero, 
                                              uint64_t &KnownOne,
                                              unsigned Depth = 0) const;

  /// ComputeNumSignBits - Return the number of times the sign bit of the
  /// register is replicated into the other bits.  We know that at least 1 bit
  /// is always equal to the sign bit (itself), but other cases can give us
  /// information.  For example, immediately after an "SRA X, 2", we know that
  /// the top 3 bits are all equal to each other, so we return 3.
  unsigned ComputeNumSignBits(SDOperand Op, unsigned Depth = 0) const;
  
  /// ComputeNumSignBitsForTargetNode - This method can be implemented by
  /// targets that want to expose additional information about sign bits to the
  /// DAG Combiner.
  virtual unsigned ComputeNumSignBitsForTargetNode(SDOperand Op,
                                                   unsigned Depth = 0) const;
  
  struct DAGCombinerInfo {
    void *DC;  // The DAG Combiner object.
    bool BeforeLegalize;
    bool CalledByLegalizer;
  public:
    SelectionDAG &DAG;
    
    DAGCombinerInfo(SelectionDAG &dag, bool bl, bool cl, void *dc)
      : DC(dc), BeforeLegalize(bl), CalledByLegalizer(cl), DAG(dag) {}
    
    bool isBeforeLegalize() const { return BeforeLegalize; }
    bool isCalledByLegalizer() const { return CalledByLegalizer; }
    
    void AddToWorklist(SDNode *N);
    SDOperand CombineTo(SDNode *N, const std::vector<SDOperand> &To);
    SDOperand CombineTo(SDNode *N, SDOperand Res);
    SDOperand CombineTo(SDNode *N, SDOperand Res0, SDOperand Res1);
  };

  /// SimplifySetCC - Try to simplify a setcc built with the specified operands 
  /// and cc. If it is unable to simplify it, return a null SDOperand.
  SDOperand SimplifySetCC(MVT::ValueType VT, SDOperand N0, SDOperand N1,
                          ISD::CondCode Cond, bool foldBooleans,
                          DAGCombinerInfo &DCI) const;

  /// PerformDAGCombine - This method will be invoked for all target nodes and
  /// for any target-independent nodes that the target has registered with
  /// invoke it for.
  ///
  /// The semantics are as follows:
  /// Return Value:
  ///   SDOperand.Val == 0   - No change was made
  ///   SDOperand.Val == N   - N was replaced, is dead, and is already handled.
  ///   otherwise            - N should be replaced by the returned Operand.
  ///
  /// In addition, methods provided by DAGCombinerInfo may be used to perform
  /// more complex transformations.
  ///
  virtual SDOperand PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  
  //===--------------------------------------------------------------------===//
  // TargetLowering Configuration Methods - These methods should be invoked by
  // the derived class constructor to configure this object for the target.
  //

protected:
  /// setUsesGlobalOffsetTable - Specify that this target does or doesn't use a
  /// GOT for PC-relative code.
  void setUsesGlobalOffsetTable(bool V) { UsesGlobalOffsetTable = V; }

  /// setShiftAmountType - Describe the type that should be used for shift
  /// amounts.  This type defaults to the pointer type.
  void setShiftAmountType(MVT::ValueType VT) { ShiftAmountTy = VT; }

  /// setSetCCResultType - Describe the type that shoudl be used as the result
  /// of a setcc operation.  This defaults to the pointer type.
  void setSetCCResultType(MVT::ValueType VT) { SetCCResultTy = VT; }

  /// setSetCCResultContents - Specify how the target extends the result of a
  /// setcc operation in a register.
  void setSetCCResultContents(SetCCResultValue Ty) { SetCCResultContents = Ty; }

  /// setSchedulingPreference - Specify the target scheduling preference.
  void setSchedulingPreference(SchedPreference Pref) {
    SchedPreferenceInfo = Pref;
  }

  /// setShiftAmountFlavor - Describe how the target handles out of range shift
  /// amounts.
  void setShiftAmountFlavor(OutOfRangeShiftAmount OORSA) {
    ShiftAmtHandling = OORSA;
  }

  /// setUseUnderscoreSetJmp - Indicate whether this target prefers to
  /// use _setjmp to implement llvm.setjmp or the non _ version.
  /// Defaults to false.
  void setUseUnderscoreSetJmp(bool Val) {
    UseUnderscoreSetJmp = Val;
  }

  /// setUseUnderscoreLongJmp - Indicate whether this target prefers to
  /// use _longjmp to implement llvm.longjmp or the non _ version.
  /// Defaults to false.
  void setUseUnderscoreLongJmp(bool Val) {
    UseUnderscoreLongJmp = Val;
  }

  /// setStackPointerRegisterToSaveRestore - If set to a physical register, this
  /// specifies the register that llvm.savestack/llvm.restorestack should save
  /// and restore.
  void setStackPointerRegisterToSaveRestore(unsigned R) {
    StackPointerRegisterToSaveRestore = R;
  }
  
  /// setExceptionPointerRegister - If set to a physical register, this sets
  /// the register that receives the exception address on entry to a landing
  /// pad.
  void setExceptionPointerRegister(unsigned R) {
    ExceptionPointerRegister = R;
  }

  /// setExceptionSelectorRegister - If set to a physical register, this sets
  /// the register that receives the exception typeid on entry to a landing
  /// pad.
  void setExceptionSelectorRegister(unsigned R) {
    ExceptionSelectorRegister = R;
  }

  /// SelectIsExpensive - Tells the code generator not to expand operations
  /// into sequences that use the select operations if possible.
  void setSelectIsExpensive() { SelectIsExpensive = true; }

  /// setIntDivIsCheap - Tells the code generator that integer divide is
  /// expensive, and if possible, should be replaced by an alternate sequence
  /// of instructions not containing an integer divide.
  void setIntDivIsCheap(bool isCheap = true) { IntDivIsCheap = isCheap; }
  
  /// setPow2DivIsCheap - Tells the code generator that it shouldn't generate
  /// srl/add/sra for a signed divide by power of two, and let the target handle
  /// it.
  void setPow2DivIsCheap(bool isCheap = true) { Pow2DivIsCheap = isCheap; }
  
  /// addRegisterClass - Add the specified register class as an available
  /// regclass for the specified value type.  This indicates the selector can
  /// handle values of that class natively.
  void addRegisterClass(MVT::ValueType VT, TargetRegisterClass *RC) {
    AvailableRegClasses.push_back(std::make_pair(VT, RC));
    RegClassForVT[VT] = RC;
  }

  /// computeRegisterProperties - Once all of the register classes are added,
  /// this allows us to compute derived properties we expose.
  void computeRegisterProperties();

  /// setOperationAction - Indicate that the specified operation does not work
  /// with the specified type and indicate what to do about it.
  void setOperationAction(unsigned Op, MVT::ValueType VT,
                          LegalizeAction Action) {
    assert(VT < 32 && Op < sizeof(OpActions)/sizeof(OpActions[0]) &&
           "Table isn't big enough!");
    OpActions[Op] &= ~(uint64_t(3UL) << VT*2);
    OpActions[Op] |= (uint64_t)Action << VT*2;
  }
  
  /// setLoadXAction - Indicate that the specified load with extension does not
  /// work with the with specified type and indicate what to do about it.
  void setLoadXAction(unsigned ExtType, MVT::ValueType VT,
                      LegalizeAction Action) {
    assert(VT < 32 && ExtType < sizeof(LoadXActions)/sizeof(LoadXActions[0]) &&
           "Table isn't big enough!");
    LoadXActions[ExtType] &= ~(uint64_t(3UL) << VT*2);
    LoadXActions[ExtType] |= (uint64_t)Action << VT*2;
  }
  
  /// setStoreXAction - Indicate that the specified store with truncation does
  /// not work with the with specified type and indicate what to do about it.
  void setStoreXAction(MVT::ValueType VT, LegalizeAction Action) {
    assert(VT < 32 && "Table isn't big enough!");
    StoreXActions &= ~(uint64_t(3UL) << VT*2);
    StoreXActions |= (uint64_t)Action << VT*2;
  }

  /// setIndexedLoadAction - Indicate that the specified indexed load does or
  /// does not work with the with specified type and indicate what to do abort
  /// it. NOTE: All indexed mode loads are initialized to Expand in
  /// TargetLowering.cpp
  void setIndexedLoadAction(unsigned IdxMode, MVT::ValueType VT,
                            LegalizeAction Action) {
    assert(VT < 32 && IdxMode <
           sizeof(IndexedModeActions[0]) / sizeof(IndexedModeActions[0][0]) &&
           "Table isn't big enough!");
    IndexedModeActions[0][IdxMode] &= ~(uint64_t(3UL) << VT*2);
    IndexedModeActions[0][IdxMode] |= (uint64_t)Action << VT*2;
  }
  
  /// setIndexedStoreAction - Indicate that the specified indexed store does or
  /// does not work with the with specified type and indicate what to do about
  /// it. NOTE: All indexed mode stores are initialized to Expand in
  /// TargetLowering.cpp
  void setIndexedStoreAction(unsigned IdxMode, MVT::ValueType VT,
                             LegalizeAction Action) {
    assert(VT < 32 && IdxMode <
           sizeof(IndexedModeActions[1]) / sizeof(IndexedModeActions[1][0]) &&
           "Table isn't big enough!");
    IndexedModeActions[1][IdxMode] &= ~(uint64_t(3UL) << VT*2);
    IndexedModeActions[1][IdxMode] |= (uint64_t)Action << VT*2;
  }
  
  /// AddPromotedToType - If Opc/OrigVT is specified as being promoted, the
  /// promotion code defaults to trying a larger integer/fp until it can find
  /// one that works.  If that default is insufficient, this method can be used
  /// by the target to override the default.
  void AddPromotedToType(unsigned Opc, MVT::ValueType OrigVT, 
                         MVT::ValueType DestVT) {
    PromoteToType[std::make_pair(Opc, OrigVT)] = DestVT;
  }

  /// addLegalFPImmediate - Indicate that this target can instruction select
  /// the specified FP immediate natively.
  void addLegalFPImmediate(double Imm) {
    LegalFPImmediates.push_back(Imm);
  }

  /// setTargetDAGCombine - Targets should invoke this method for each target
  /// independent node that they want to provide a custom DAG combiner for by
  /// implementing the PerformDAGCombine virtual method.
  void setTargetDAGCombine(ISD::NodeType NT) {
    TargetDAGCombineArray[NT >> 3] |= 1 << (NT&7);
  }
  
  /// setJumpBufSize - Set the target's required jmp_buf buffer size (in
  /// bytes); default is 200
  void setJumpBufSize(unsigned Size) {
    JumpBufSize = Size;
  }

  /// setJumpBufAlignment - Set the target's required jmp_buf buffer
  /// alignment (in bytes); default is 0
  void setJumpBufAlignment(unsigned Align) {
    JumpBufAlignment = Align;
  }
  
public:

  //===--------------------------------------------------------------------===//
  // Lowering methods - These methods must be implemented by targets so that
  // the SelectionDAGLowering code knows how to lower these.
  //

  /// LowerArguments - This hook must be implemented to indicate how we should
  /// lower the arguments for the specified function, into the specified DAG.
  virtual std::vector<SDOperand>
  LowerArguments(Function &F, SelectionDAG &DAG);

  /// LowerCallTo - This hook lowers an abstract call to a function into an
  /// actual call.  This returns a pair of operands.  The first element is the
  /// return value for the function (if RetTy is not VoidTy).  The second
  /// element is the outgoing token chain.
  struct ArgListEntry {
    SDOperand Node;
    const Type* Ty;
    bool isSExt;
    bool isZExt;
    bool isInReg;
    bool isSRet;

    ArgListEntry():isSExt(false), isZExt(false), isInReg(false), isSRet(false) { };
  };
  typedef std::vector<ArgListEntry> ArgListTy;
  virtual std::pair<SDOperand, SDOperand>
  LowerCallTo(SDOperand Chain, const Type *RetTy, bool RetTyIsSigned, 
              bool isVarArg, unsigned CallingConv, bool isTailCall, 
              SDOperand Callee, ArgListTy &Args, SelectionDAG &DAG);

  /// LowerOperation - This callback is invoked for operations that are 
  /// unsupported by the target, which are registered to use 'custom' lowering,
  /// and whose defined values are all legal.
  /// If the target has no operations that require custom lowering, it need not
  /// implement this.  The default implementation of this aborts.
  virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);

  /// CustomPromoteOperation - This callback is invoked for operations that are
  /// unsupported by the target, are registered to use 'custom' lowering, and
  /// whose type needs to be promoted.
  virtual SDOperand CustomPromoteOperation(SDOperand Op, SelectionDAG &DAG);
  
  /// getTargetNodeName() - This method returns the name of a target specific
  /// DAG node.
  virtual const char *getTargetNodeName(unsigned Opcode) const;

  //===--------------------------------------------------------------------===//
  // Inline Asm Support hooks
  //
  
  enum ConstraintType {
    C_Register,            // Constraint represents a single register.
    C_RegisterClass,       // Constraint represents one or more registers.
    C_Memory,              // Memory constraint.
    C_Other,               // Something else.
    C_Unknown              // Unsupported constraint.
  };
  
  /// getConstraintType - Given a constraint, return the type of constraint it
  /// is for this target.
  virtual ConstraintType getConstraintType(const std::string &Constraint) const;
  
  
  /// getRegClassForInlineAsmConstraint - Given a constraint letter (e.g. "r"),
  /// return a list of registers that can be used to satisfy the constraint.
  /// This should only be used for C_RegisterClass constraints.
  virtual std::vector<unsigned> 
  getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                    MVT::ValueType VT) const;

  /// getRegForInlineAsmConstraint - Given a physical register constraint (e.g.
  /// {edx}), return the register number and the register class for the
  /// register.
  ///
  /// Given a register class constraint, like 'r', if this corresponds directly
  /// to an LLVM register class, return a register of 0 and the register class
  /// pointer.
  ///
  /// This should only be used for C_Register constraints.  On error,
  /// this returns a register number of 0 and a null register class pointer..
  virtual std::pair<unsigned, const TargetRegisterClass*> 
    getRegForInlineAsmConstraint(const std::string &Constraint,
                                 MVT::ValueType VT) const;
  
  
  /// isOperandValidForConstraint - Return the specified operand (possibly
  /// modified) if the specified SDOperand is valid for the specified target
  /// constraint letter, otherwise return null.
  virtual SDOperand 
    isOperandValidForConstraint(SDOperand Op, char ConstraintLetter,
                                SelectionDAG &DAG);
  
  //===--------------------------------------------------------------------===//
  // Scheduler hooks
  //
  
  // InsertAtEndOfBasicBlock - This method should be implemented by targets that
  // mark instructions with the 'usesCustomDAGSchedInserter' flag.  These
  // instructions are special in various ways, which require special support to
  // insert.  The specified MachineInstr is created but not inserted into any
  // basic blocks, and the scheduler passes ownership of it to this method.
  virtual MachineBasicBlock *InsertAtEndOfBasicBlock(MachineInstr *MI,
                                                     MachineBasicBlock *MBB);

  //===--------------------------------------------------------------------===//
  // Addressing mode description hooks (used by LSR etc).
  //

  /// AddrMode - This represents an addressing mode of:
  ///    BaseGV + BaseOffs + BaseReg + Scale*ScaleReg
  /// If BaseGV is null,  there is no BaseGV.
  /// If BaseOffs is zero, there is no base offset.
  /// If HasBaseReg is false, there is no base register.
  /// If Scale is zero, there is no ScaleReg.  Scale of 1 indicates a reg with
  /// no scale.
  ///
  struct AddrMode {
    GlobalValue *BaseGV;
    int64_t      BaseOffs;
    bool         HasBaseReg;
    int64_t      Scale;
    AddrMode() : BaseGV(0), BaseOffs(0), HasBaseReg(false), Scale(0) {}
  };
  
  /// isLegalAddressingMode - Return true if the addressing mode represented by
  /// AM is legal for this target, for a load/store of the specified type.
  /// TODO: Handle pre/postinc as well.
  virtual bool isLegalAddressingMode(const AddrMode &AM, const Type *Ty) const;

  //===--------------------------------------------------------------------===//
  // Div utility functions
  //
  SDOperand BuildSDIV(SDNode *N, SelectionDAG &DAG, 
                      std::vector<SDNode*>* Created) const;
  SDOperand BuildUDIV(SDNode *N, SelectionDAG &DAG, 
                      std::vector<SDNode*>* Created) const;


  //===--------------------------------------------------------------------===//
  // Runtime Library hooks
  //

  /// setLibcallName - Rename the default libcall routine name for the specified
  /// libcall.
  void setLibcallName(RTLIB::Libcall Call, const char *Name) {
    LibcallRoutineNames[Call] = Name;
  }

  /// getLibcallName - Get the libcall routine name for the specified libcall.
  ///
  const char *getLibcallName(RTLIB::Libcall Call) const {
    return LibcallRoutineNames[Call];
  }

  /// setCmpLibcallCC - Override the default CondCode to be used to test the
  /// result of the comparison libcall against zero.
  void setCmpLibcallCC(RTLIB::Libcall Call, ISD::CondCode CC) {
    CmpLibcallCCs[Call] = CC;
  }

  /// getCmpLibcallCC - Get the CondCode that's to be used to test the result of
  /// the comparison libcall against zero.
  ISD::CondCode getCmpLibcallCC(RTLIB::Libcall Call) const {
    return CmpLibcallCCs[Call];
  }

private:
  TargetMachine &TM;
  const TargetData *TD;

  /// IsLittleEndian - True if this is a little endian target.
  ///
  bool IsLittleEndian;

  /// PointerTy - The type to use for pointers, usually i32 or i64.
  ///
  MVT::ValueType PointerTy;

  /// UsesGlobalOffsetTable - True if this target uses a GOT for PIC codegen.
  ///
  bool UsesGlobalOffsetTable;
  
  /// ShiftAmountTy - The type to use for shift amounts, usually i8 or whatever
  /// PointerTy is.
  MVT::ValueType ShiftAmountTy;

  OutOfRangeShiftAmount ShiftAmtHandling;

  /// SelectIsExpensive - Tells the code generator not to expand operations
  /// into sequences that use the select operations if possible.
  bool SelectIsExpensive;

  /// IntDivIsCheap - Tells the code generator not to expand integer divides by
  /// constants into a sequence of muls, adds, and shifts.  This is a hack until
  /// a real cost model is in place.  If we ever optimize for size, this will be
  /// set to true unconditionally.
  bool IntDivIsCheap;
  
  /// Pow2DivIsCheap - Tells the code generator that it shouldn't generate
  /// srl/add/sra for a signed divide by power of two, and let the target handle
  /// it.
  bool Pow2DivIsCheap;
  
  /// SetCCResultTy - The type that SetCC operations use.  This defaults to the
  /// PointerTy.
  MVT::ValueType SetCCResultTy;

  /// SetCCResultContents - Information about the contents of the high-bits in
  /// the result of a setcc comparison operation.
  SetCCResultValue SetCCResultContents;

  /// SchedPreferenceInfo - The target scheduling preference: shortest possible
  /// total cycles or lowest register usage.
  SchedPreference SchedPreferenceInfo;
  
  /// UseUnderscoreSetJmp - This target prefers to use _setjmp to implement
  /// llvm.setjmp.  Defaults to false.
  bool UseUnderscoreSetJmp;

  /// UseUnderscoreLongJmp - This target prefers to use _longjmp to implement
  /// llvm.longjmp.  Defaults to false.
  bool UseUnderscoreLongJmp;

  /// JumpBufSize - The size, in bytes, of the target's jmp_buf buffers
  unsigned JumpBufSize;
  
  /// JumpBufAlignment - The alignment, in bytes, of the target's jmp_buf
  /// buffers
  unsigned JumpBufAlignment;
  
  /// StackPointerRegisterToSaveRestore - If set to a physical register, this
  /// specifies the register that llvm.savestack/llvm.restorestack should save
  /// and restore.
  unsigned StackPointerRegisterToSaveRestore;

  /// ExceptionPointerRegister - If set to a physical register, this specifies
  /// the register that receives the exception address on entry to a landing
  /// pad.
  unsigned ExceptionPointerRegister;

  /// ExceptionSelectorRegister - If set to a physical register, this specifies
  /// the register that receives the exception typeid on entry to a landing
  /// pad.
  unsigned ExceptionSelectorRegister;

  /// RegClassForVT - This indicates the default register class to use for
  /// each ValueType the target supports natively.
  TargetRegisterClass *RegClassForVT[MVT::LAST_VALUETYPE];
  unsigned char NumElementsForVT[MVT::LAST_VALUETYPE];

  /// TransformToType - For any value types we are promoting or expanding, this
  /// contains the value type that we are changing to.  For Expanded types, this
  /// contains one step of the expand (e.g. i64 -> i32), even if there are
  /// multiple steps required (e.g. i64 -> i16).  For types natively supported
  /// by the system, this holds the same type (e.g. i32 -> i32).
  MVT::ValueType TransformToType[MVT::LAST_VALUETYPE];

  /// OpActions - For each operation and each value type, keep a LegalizeAction
  /// that indicates how instruction selection should deal with the operation.
  /// Most operations are Legal (aka, supported natively by the target), but
  /// operations that are not should be described.  Note that operations on
  /// non-legal value types are not described here.
  uint64_t OpActions[156];
  
  /// LoadXActions - For each load of load extension type and each value type,
  /// keep a LegalizeAction that indicates how instruction selection should deal
  /// with the load.
  uint64_t LoadXActions[ISD::LAST_LOADX_TYPE];
  
  /// StoreXActions - For each store with truncation of each value type, keep a
  /// LegalizeAction that indicates how instruction selection should deal with
  /// the store.
  uint64_t StoreXActions;

  /// IndexedModeActions - For each indexed mode and each value type, keep a
  /// pair of LegalizeAction that indicates how instruction selection should
  /// deal with the load / store.
  uint64_t IndexedModeActions[2][ISD::LAST_INDEXED_MODE];
  
  ValueTypeActionImpl ValueTypeActions;

  std::vector<double> LegalFPImmediates;

  std::vector<std::pair<MVT::ValueType,
                        TargetRegisterClass*> > AvailableRegClasses;

  /// TargetDAGCombineArray - Targets can specify ISD nodes that they would
  /// like PerformDAGCombine callbacks for by calling setTargetDAGCombine(),
  /// which sets a bit in this array.
  unsigned char TargetDAGCombineArray[156/(sizeof(unsigned char)*8)];
  
  /// PromoteToType - For operations that must be promoted to a specific type,
  /// this holds the destination type.  This map should be sparse, so don't hold
  /// it as an array.
  ///
  /// Targets add entries to this map with AddPromotedToType(..), clients access
  /// this with getTypeToPromoteTo(..).
  std::map<std::pair<unsigned, MVT::ValueType>, MVT::ValueType> PromoteToType;

  /// LibcallRoutineNames - Stores the name each libcall.
  ///
  const char *LibcallRoutineNames[RTLIB::UNKNOWN_LIBCALL];

  /// CmpLibcallCCs - The ISD::CondCode that should be used to test the result
  /// of each of the comparison libcall against zero.
  ISD::CondCode CmpLibcallCCs[RTLIB::UNKNOWN_LIBCALL];

protected:
  /// When lowering %llvm.memset this field specifies the maximum number of
  /// store operations that may be substituted for the call to memset. Targets
  /// must set this value based on the cost threshold for that target. Targets
  /// should assume that the memset will be done using as many of the largest
  /// store operations first, followed by smaller ones, if necessary, per
  /// alignment restrictions. For example, storing 9 bytes on a 32-bit machine
  /// with 16-bit alignment would result in four 2-byte stores and one 1-byte
  /// store.  This only applies to setting a constant array of a constant size.
  /// @brief Specify maximum number of store instructions per memset call.
  unsigned maxStoresPerMemset;

  /// When lowering %llvm.memcpy this field specifies the maximum number of
  /// store operations that may be substituted for a call to memcpy. Targets
  /// must set this value based on the cost threshold for that target. Targets
  /// should assume that the memcpy will be done using as many of the largest
  /// store operations first, followed by smaller ones, if necessary, per
  /// alignment restrictions. For example, storing 7 bytes on a 32-bit machine
  /// with 32-bit alignment would result in one 4-byte store, a one 2-byte store
  /// and one 1-byte store. This only applies to copying a constant array of
  /// constant size.
  /// @brief Specify maximum bytes of store instructions per memcpy call.
  unsigned maxStoresPerMemcpy;

  /// When lowering %llvm.memmove this field specifies the maximum number of
  /// store instructions that may be substituted for a call to memmove. Targets
  /// must set this value based on the cost threshold for that target. Targets
  /// should assume that the memmove will be done using as many of the largest
  /// store operations first, followed by smaller ones, if necessary, per
  /// alignment restrictions. For example, moving 9 bytes on a 32-bit machine
  /// with 8-bit alignment would result in nine 1-byte stores.  This only
  /// applies to copying a constant array of constant size.
  /// @brief Specify maximum bytes of store instructions per memmove call.
  unsigned maxStoresPerMemmove;

  /// This field specifies whether the target machine permits unaligned memory
  /// accesses.  This is used, for example, to determine the size of store 
  /// operations when copying small arrays and other similar tasks.
  /// @brief Indicate whether the target permits unaligned memory accesses.
  bool allowUnalignedMemoryAccesses;
};
} // end llvm namespace

#endif
