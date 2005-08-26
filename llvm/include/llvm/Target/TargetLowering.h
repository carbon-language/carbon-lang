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
#include "llvm/CodeGen/ValueTypes.h"
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
    Custom,     // Use the LowerOperation hook to implement custom lowering.
  };

  enum OutOfRangeShiftAmount {
    Undefined,  // Oversized shift amounts are undefined (default).
    Mask,       // Shift amounts are auto masked (anded) to value size.
    Extend,     // Oversized shift pulls in zeros or sign bits.
  };

  enum SetCCResultValue {
    UndefinedSetCCResult,          // SetCC returns a garbage/unknown extend.
    ZeroOrOneSetCCResult,          // SetCC returns a zero extended result.
    ZeroOrNegativeOneSetCCResult,  // SetCC returns a sign extended result.
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

  /// getSetCCResultTy - Return the ValueType of the result of setcc operations.
  ///
  MVT::ValueType getSetCCResultTy() const { return SetCCResultTy; }

  /// getSetCCResultContents - For targets without boolean registers, this flag
  /// returns information about the contents of the high-bits in the setcc
  /// result register.
  SetCCResultValue getSetCCResultContents() const { return SetCCResultContents;}

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

  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal (return 'Legal') or we need to promote it to a larger
  /// type (return 'Promote'), or we need to expand it into multiple registers
  /// of smaller integer type (return 'Expand').  'Custom' is not an option.
  LegalizeAction getTypeAction(MVT::ValueType VT) const {
    return (LegalizeAction)((ValueTypeActions >> (2*VT)) & 3);
  }
  unsigned getValueTypeActions() const { return ValueTypeActions; }

  /// getTypeToTransformTo - For types supported by the target, this is an
  /// identity function.  For types that must be promoted to larger types, this
  /// returns the larger type to promote to.  For types that are larger than the
  /// largest integer register, this contains one step in the expansion to get
  /// to the smaller register.
  MVT::ValueType getTypeToTransformTo(MVT::ValueType VT) const {
    return TransformToType[VT];
  }

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
    return getOperationAction(Op, VT) == Legal;
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
    }
  }

  /// getNumElements - Return the number of registers that this ValueType will
  /// eventually require.  This is always one for all non-integer types, is
  /// one for any types promoted to live in larger registers, but may be more
  /// than one for types (like i64) that are split into pieces.
  unsigned getNumElements(MVT::ValueType VT) const {
    return NumElementsForVT[VT];
  }

  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memset. The value is set by the target at the
  /// performance threshold for such a replacement.
  /// @brief Get maximum # of store operations permitted for llvm.memset
  unsigned getMaxStoresPerMemSet() const { return maxStoresPerMemSet; }

  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memcpy. The value is set by the target at the
  /// performance threshold for such a replacement.
  /// @brief Get maximum # of store operations permitted for llvm.memcpy
  unsigned getMaxStoresPerMemCpy() const { return maxStoresPerMemCpy; }

  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memmove. The value is set by the target at the
  /// performance threshold for such a replacement.
  /// @brief Get maximum # of store operations permitted for llvm.memmove
  unsigned getMaxStoresPerMemMove() const { return maxStoresPerMemMove; }

  /// This function returns true if the target allows unaligned stores. This is
  /// used in situations where an array copy/move/set is converted to a sequence
  /// of store operations. It ensures that such replacements don't generate
  /// code that causes an alignment error (trap) on the target machine.
  /// @brief Determine if the target supports unaligned stores.
  bool allowsUnalignedStores() const { return allowUnalignedStores; }

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

  /// setShiftAmountFlavor - Describe how the target handles out of range shift
  /// amounts.
  void setShiftAmountFlavor(OutOfRangeShiftAmount OORSA) {
    ShiftAmtHandling = OORSA;
  }

  /// setSetCCIxExpensive - This is a short term hack for targets that codegen
  /// setcc as a conditional branch.  This encourages the code generator to fold
  /// setcc operations into other operations if possible.
  void setSetCCIsExpensive() { SetCCIsExpensive = true; }

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
    assert(VT < 16 && Op < sizeof(OpActions)/sizeof(OpActions[0]) &&
           "Table isn't big enough!");
    OpActions[Op] |= Action << VT*2;
  }

  /// addLegalFPImmediate - Indicate that this target can instruction select
  /// the specified FP immediate natively.
  void addLegalFPImmediate(double Imm) {
    LegalFPImmediates.push_back(Imm);
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

  /// LowerVAStart - This lowers the llvm.va_start intrinsic.  If not
  /// implemented, this method prints a message and aborts.  This method should
  /// return the modified chain value.  Note that VAListPtr* correspond to the
  /// llvm.va_start operand.
  virtual SDOperand LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                 Value *VAListV, SelectionDAG &DAG);

  /// LowerVAEnd - This lowers llvm.va_end and returns the resultant chain.  If
  /// not implemented, this defaults to a noop.
  virtual SDOperand LowerVAEnd(SDOperand Chain, SDOperand LP, Value *LV,
                               SelectionDAG &DAG);

  /// LowerVACopy - This lowers llvm.va_copy and returns the resultant chain.
  /// If not implemented, this defaults to loading a pointer from the input and
  /// storing it to the output.
  virtual SDOperand LowerVACopy(SDOperand Chain, SDOperand SrcP, Value *SrcV,
                                SDOperand DestP, Value *DestV,
                                SelectionDAG &DAG);

  /// LowerVAArg - This lowers the vaarg instruction.  If not implemented, this
  /// prints a message and aborts.
  virtual std::pair<SDOperand,SDOperand>
  LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
             const Type *ArgTy, SelectionDAG &DAG);

  /// LowerFrameReturnAddress - This hook lowers a call to llvm.returnaddress or
  /// llvm.frameaddress (depending on the value of the first argument).  The
  /// return values are the result pointer and the resultant token chain.  If
  /// not implemented, both of these intrinsics will return null.
  virtual std::pair<SDOperand, SDOperand>
  LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                          SelectionDAG &DAG);

  /// LowerOperation - For operations that are unsupported by the target, and
  /// which are registered to use 'custom' lowering.  This callback is invoked.
  /// If the target has no operations that require custom lowering, it need not
  /// implement this.  The default implementation of this aborts.
  virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);

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

  /// SetCCResultTy - The type that SetCC operations use.  This defaults to the
  /// PointerTy.
  MVT::ValueType SetCCResultTy;

  /// SetCCResultContents - Information about the contents of the high-bits in
  /// the result of a setcc comparison operation.
  SetCCResultValue SetCCResultContents;

  /// RegClassForVT - This indicates the default register class to use for
  /// each ValueType the target supports natively.
  TargetRegisterClass *RegClassForVT[MVT::LAST_VALUETYPE];
  unsigned char NumElementsForVT[MVT::LAST_VALUETYPE];

  /// ValueTypeActions - This is a bitvector that contains two bits for each
  /// value type, where the two bits correspond to the LegalizeAction enum.
  /// This can be queried with "getTypeAction(VT)".
  unsigned ValueTypeActions;

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
  unsigned OpActions[128];

  std::vector<double> LegalFPImmediates;

  std::vector<std::pair<MVT::ValueType,
                        TargetRegisterClass*> > AvailableRegClasses;

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
  unsigned maxStoresPerMemSet;

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
  unsigned maxStoresPerMemCpy;

  /// When lowering %llvm.memmove this field specifies the maximum number of
  /// store instructions that may be substituted for a call to memmove. Targets
  /// must set this value based on the cost threshold for that target. Targets
  /// should assume that the memmove will be done using as many of the largest
  /// store operations first, followed by smaller ones, if necessary, per
  /// alignment restrictions. For example, moving 9 bytes on a 32-bit machine
  /// with 8-bit alignment would result in nine 1-byte stores.  This only
  /// applies to copying a constant array of constant size.
  /// @brief Specify maximum bytes of store instructions per memmove call.
  unsigned maxStoresPerMemMove;

  /// This field specifies whether the target machine permits unaligned stores.
  /// This is used to determine the size of store operations for copying
  /// small arrays and other similar tasks.
  /// @brief Indicate whether the target machine permits unaligned stores.
  bool allowUnalignedStores;
};
} // end llvm namespace

#endif
