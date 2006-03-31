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

#include "llvm/Type.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/DataTypes.h"
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
  const TargetData &getTargetData() const { return TD; }

  bool isLittleEndian() const { return IsLittleEndian; }
  MVT::ValueType getPointerTy() const { return PointerTy; }
  MVT::ValueType getShiftAmountTy() const { return ShiftAmountTy; }
  OutOfRangeShiftAmount getShiftAmountFlavor() const {return ShiftAmtHandling; }

  /// isSetCCExpensive - Return true if the setcc operation is expensive for
  /// this target.
  bool isSetCCExpensive() const { return SetCCIsExpensive; }
  
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
  /// returns the larger type to promote to.  For types that are larger than the
  /// largest integer register, this contains one step in the expansion to get
  /// to the smaller register.
  MVT::ValueType getTypeToTransformTo(MVT::ValueType VT) const {
    return TransformToType[VT];
  }

  /// getPackedTypeBreakdown - Packed types are broken down into some number of
  /// legal scalar types.  For example, <8 x float> maps to 2 MVT::v2f32 values
  /// with Altivec or SSE1, or 8 promoted MVT::f64 values with the X86 FP stack.
  /// Similarly, <2 x long> turns into 4 MVT::i32 values with both PPC and X86.
  ///
  /// This method returns the number of registers needed, and the VT for each
  /// register.  It also returns the VT of the PackedType elements before they
  /// are promoted/expanded.
  ///
  unsigned getPackedTypeBreakdown(const PackedType *PTy, 
                                  MVT::ValueType &PTyElementVT,
                                  MVT::ValueType &PTyLegalElementVT) const;
  
  typedef std::vector<double>::const_iterator legal_fpimm_iterator;
  legal_fpimm_iterator legal_fpimm_begin() const {
    return LegalFPImmediates.begin();
  }
  legal_fpimm_iterator legal_fpimm_end() const {
    return LegalFPImmediates.end();
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
  
  
  /// isVectorShuffleLegal - Return true if a vector shuffle is legal with the
  /// specified mask and type.  Targets can specify exactly which masks they
  /// support and the code generator is tasked with not creating illegal masks.
  bool isShuffleLegal(MVT::ValueType VT, SDOperand Mask) const {
    return isOperationLegal(ISD::VECTOR_SHUFFLE, VT) && 
           isShuffleMaskLegal(Mask, VT);
  }

  /// getTypeToPromoteTo - If the action for this operation is to promote, this
  /// method returns the ValueType to promote to.
  MVT::ValueType getTypeToPromoteTo(unsigned Op, MVT::ValueType VT) const {
    assert(getOperationAction(Op, VT) == Promote &&
           "This operation isn't promoted!");
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
  /// This is fixed by the LLVM operations except for the pointer size.
  MVT::ValueType getValueType(const Type *Ty) const {
    switch (Ty->getTypeID()) {
    default: assert(0 && "Unknown type!");
    case Type::VoidTyID:    return MVT::isVoid;
    case Type::BoolTyID:    return MVT::i1;
    case Type::UByteTyID:
    case Type::SByteTyID:   return MVT::i8;
    case Type::ShortTyID:
    case Type::UShortTyID:  return MVT::i16;
    case Type::IntTyID:
    case Type::UIntTyID:    return MVT::i32;
    case Type::LongTyID:
    case Type::ULongTyID:   return MVT::i64;
    case Type::FloatTyID:   return MVT::f32;
    case Type::DoubleTyID:  return MVT::f64;
    case Type::PointerTyID: return PointerTy;
    case Type::PackedTyID:  return MVT::Vector;
    }
  }

  /// getNumElements - Return the number of registers that this ValueType will
  /// eventually require.  This is always one for all non-integer types, is
  /// one for any types promoted to live in larger registers, but may be more
  /// than one for types (like i64) that are split into pieces.
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
  
  /// usesUnderscoreSetJmpLongJmp - Determine if we should use _setjmp or setjmp
  /// to implement llvm.setjmp.
  bool usesUnderscoreSetJmpLongJmp() const {
    return UseUnderscoreSetJmpLongJmp;
  }
  
  /// getStackPointerRegisterToSaveRestore - If a physical register, this
  /// specifies the register that llvm.savestack/llvm.restorestack should save
  /// and restore.
  unsigned getStackPointerRegisterToSaveRestore() const {
    return StackPointerRegisterToSaveRestore;
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

  struct DAGCombinerInfo {
    void *DC;  // The DAG Combiner object.
    bool BeforeLegalize;
  public:
    SelectionDAG &DAG;
    
    DAGCombinerInfo(SelectionDAG &dag, bool bl, void *dc)
      : DC(dc), BeforeLegalize(bl), DAG(dag) {}
    
    bool isBeforeLegalize() const { return BeforeLegalize; }
    
    void AddToWorklist(SDNode *N);
    SDOperand CombineTo(SDNode *N, const std::vector<SDOperand> &To);
    SDOperand CombineTo(SDNode *N, SDOperand Res);
    SDOperand CombineTo(SDNode *N, SDOperand Res0, SDOperand Res1);
  };

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

  /// setUseUnderscoreSetJmpLongJmp - Indicate whether this target prefers to
  /// use _setjmp and _longjmp to or implement llvm.setjmp/llvm.longjmp or
  /// the non _ versions.  Defaults to false.
  void setUseUnderscoreSetJmpLongJmp(bool Val) {
    UseUnderscoreSetJmpLongJmp = Val;
  }
  
  /// setStackPointerRegisterToSaveRestore - If set to a physical register, this
  /// specifies the register that llvm.savestack/llvm.restorestack should save
  /// and restore.
  void setStackPointerRegisterToSaveRestore(unsigned R) {
    StackPointerRegisterToSaveRestore = R;
  }
  
  /// setSetCCIxExpensive - This is a short term hack for targets that codegen
  /// setcc as a conditional branch.  This encourages the code generator to fold
  /// setcc operations into other operations if possible.
  void setSetCCIsExpensive() { SetCCIsExpensive = true; }

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
    OpActions[Op] &= ~(3ULL << VT*2);
    OpActions[Op] |= (uint64_t)Action << VT*2;
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
  
  /// isShuffleMaskLegal - Targets can use this to indicate that they only
  /// support *some* VECTOR_SHUFFLE operations, those with specific masks.
  /// By default, if a target supports the VECTOR_SHUFFLE node, all mask values
  /// are assumed to be legal.
  virtual bool isShuffleMaskLegal(SDOperand Mask, MVT::ValueType VT) const {
    return true;
  }
  
public:

  //===--------------------------------------------------------------------===//
  // Lowering methods - These methods must be implemented by targets so that
  // the SelectionDAGLowering code knows how to lower these.
  //

  /// LowerArguments - This hook must be implemented to indicate how we should
  /// lower the arguments for the specified function, into the specified DAG.
  virtual std::vector<SDOperand>
  LowerArguments(Function &F, SelectionDAG &DAG) = 0;

  /// LowerCallTo - This hook lowers an abstract call to a function into an
  /// actual call.  This returns a pair of operands.  The first element is the
  /// return value for the function (if RetTy is not VoidTy).  The second
  /// element is the outgoing token chain.
  typedef std::vector<std::pair<SDOperand, const Type*> > ArgListTy;
  virtual std::pair<SDOperand, SDOperand>
  LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg,
              unsigned CallingConv, bool isTailCall, SDOperand Callee,
              ArgListTy &Args, SelectionDAG &DAG) = 0;

  /// LowerFrameReturnAddress - This hook lowers a call to llvm.returnaddress or
  /// llvm.frameaddress (depending on the value of the first argument).  The
  /// return values are the result pointer and the resultant token chain.  If
  /// not implemented, both of these intrinsics will return null.
  virtual std::pair<SDOperand, SDOperand>
  LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                          SelectionDAG &DAG);

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
  
  /// getConstraintType - Given a constraint letter, return the type of
  /// constraint it is for this target.
  virtual ConstraintType getConstraintType(char ConstraintLetter) const;
  
  
  /// getRegClassForInlineAsmConstraint - Given a constraint letter (e.g. "r"),
  /// return a list of registers that can be used to satisfy the constraint.
  /// This should only be used for C_RegisterClass constraints.
  virtual std::vector<unsigned> 
  getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                    MVT::ValueType VT) const;

  /// getRegForInlineAsmConstraint - Given a physical register constraint (e.g.
  /// {edx}), return the register number and the register class for the
  /// register.  This should only be used for C_Register constraints.  On error,
  /// this returns a register number of 0.
  virtual std::pair<unsigned, const TargetRegisterClass*> 
    getRegForInlineAsmConstraint(const std::string &Constraint,
                                 MVT::ValueType VT) const;
  
  
  /// isOperandValidForConstraint - Return true if the specified SDOperand is
  /// valid for the specified target constraint letter.
  virtual bool isOperandValidForConstraint(SDOperand Op, char ConstraintLetter);
  
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
  // Loop Strength Reduction hooks
  //
  
  /// isLegalAddressImmediate - Return true if the integer value or GlobalValue
  /// can be used as the offset of the target addressing mode.
  virtual bool isLegalAddressImmediate(int64_t V) const;
  virtual bool isLegalAddressImmediate(GlobalValue *GV) const;

  typedef std::vector<unsigned>::const_iterator legal_am_scale_iterator;
  legal_am_scale_iterator legal_am_scale_begin() const {
    return LegalAddressScales.begin();
  }
  legal_am_scale_iterator legal_am_scale_end() const {
    return LegalAddressScales.end();
  }

protected:
  /// addLegalAddressScale - Add a integer (> 1) value which can be used as
  /// scale in the target addressing mode. Note: the ordering matters so the
  /// least efficient ones should be entered first.
  void addLegalAddressScale(unsigned Scale) {
    LegalAddressScales.push_back(Scale);
  }

private:
  std::vector<unsigned> LegalAddressScales;
  
private:
  TargetMachine &TM;
  const TargetData &TD;

  /// IsLittleEndian - True if this is a little endian target.
  ///
  bool IsLittleEndian;

  /// PointerTy - The type to use for pointers, usually i32 or i64.
  ///
  MVT::ValueType PointerTy;

  /// ShiftAmountTy - The type to use for shift amounts, usually i8 or whatever
  /// PointerTy is.
  MVT::ValueType ShiftAmountTy;

  OutOfRangeShiftAmount ShiftAmtHandling;

  /// SetCCIsExpensive - This is a short term hack for targets that codegen
  /// setcc as a conditional branch.  This encourages the code generator to fold
  /// setcc operations into other operations if possible.
  bool SetCCIsExpensive;

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
  
  /// UseUnderscoreSetJmpLongJmp - This target prefers to use _setjmp and
  /// _longjmp to implement llvm.setjmp/llvm.longjmp.  Defaults to false.
  bool UseUnderscoreSetJmpLongJmp;
  
  /// StackPointerRegisterToSaveRestore - If set to a physical register, this
  /// specifies the register that llvm.savestack/llvm.restorestack should save
  /// and restore.
  unsigned StackPointerRegisterToSaveRestore;

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
  
  ValueTypeActionImpl ValueTypeActions;

  std::vector<double> LegalFPImmediates;

  std::vector<std::pair<MVT::ValueType,
                        TargetRegisterClass*> > AvailableRegClasses;

  /// TargetDAGCombineArray - Targets can specify ISD nodes that they would
  /// like PerformDAGCombine callbacks for by calling setTargetDAGCombine(),
  /// which sets a bit in this array.
  unsigned char TargetDAGCombineArray[156/(sizeof(unsigned char)*8)];
  
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
