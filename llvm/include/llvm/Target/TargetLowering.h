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
  class Function;
  class TargetMachine;
  class TargetData;
  class TargetRegisterClass;
  class SDNode;
  class SDOperand;
  class SelectionDAG;

//===----------------------------------------------------------------------===//
/// TargetLowering - This class defines information used to lower LLVM code to
/// legal SelectionDAG operators that the target instruction selector can accept
/// natively.
///
/// This class also defines callbacks that targets must implement to lower
/// target-specific constructs to SelectionDAG operators.
///
class TargetLowering {
  TargetMachine &TM;
  const TargetData &TD;
  
  MVT::ValueType PointerTy;
  bool IsLittleEndian;
  
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

  unsigned short UnsupportedOps[128];
  
  std::vector<double> LegalFPImmediates;
  
  std::vector<std::pair<MVT::ValueType,
                        TargetRegisterClass*> > AvailableRegClasses;
public:
  TargetLowering(TargetMachine &TM);
  virtual ~TargetLowering() {}

  TargetMachine &getTargetMachine() const { return TM; }
  const TargetData &getTargetData() const { return TD; }
  
  bool isLittleEndian() const { return IsLittleEndian; }
  MVT::ValueType getPointerTy() const { return PointerTy; }
  
  TargetRegisterClass *getRegClassFor(MVT::ValueType VT) {
    TargetRegisterClass *RC = RegClassForVT[VT];
    assert(RC && "This value type is not natively supported!");
    return RC;
  }
  
  /// hasNativeSupportFor - Return true if the target has native support for the
  /// specified value type.  This means that it has a register that directly
  /// holds it without promotions or expansions.
  bool hasNativeSupportFor(MVT::ValueType VT) {
    return RegClassForVT[VT] != 0;
  }

  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal (return 0) or we need to promote it to a larger type
  /// (return 1), or we need to expand it into multiple registers of smaller
  /// integer type (return 2).
  unsigned getTypeAction(MVT::ValueType VT) const {
    return (ValueTypeActions >> (2*VT)) & 3;
  }
  unsigned getValueTypeActions() const { return ValueTypeActions; }

  /// getTypeToTransformTo - For types supported by the target, this is an
  /// identity function.  For types that must be promoted to larger types, this
  /// returns the larger type to promote to.  For types that are larger than the
  /// largest integer register, this contains one step in the expansion to get
  /// to the smaller register.
  MVT::ValueType getTypeToTransformTo(MVT::ValueType VT) {
    return TransformToType[VT];
  }
  
  typedef std::vector<double>::const_iterator legal_fpimm_iterator;
  legal_fpimm_iterator legal_fpimm_begin() const {
    return LegalFPImmediates.begin();
  }
  legal_fpimm_iterator legal_fpimm_end() const {
    return LegalFPImmediates.end();
  }
  
  bool isOperationSupported(unsigned Op, MVT::ValueType VT) {
    return (UnsupportedOps[Op] & (1 << VT)) == 0;
  }
  
  MVT::ValueType getValueType(const Type *Ty) {
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

  //===--------------------------------------------------------------------===//
  // TargetLowering Configuration Methods - These methods should be invoked by
  // the derived class constructor to configure this object for the target.
  //

protected:

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
  

  void setOperationUnsupported(unsigned Op, MVT::ValueType VT) {
    assert(VT < 16 && "Too many value types!");
    UnsupportedOps[Op] |= 1 << VT;
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
  LowerCallTo(SDOperand Chain, const Type *RetTy, SDOperand Callee,
              ArgListTy &Args, SelectionDAG &DAG) = 0;

  
  /// LowerVAStart - This lowers the llvm.va_start intrinsic.  If not
  /// implemented, this method prints a message and aborts.
  virtual std::pair<SDOperand, SDOperand>
  LowerVAStart(SDOperand Chain, SelectionDAG &DAG);

  /// LowerVAEnd - This lowers llvm.va_end and returns the resultant chain.  If
  /// not implemented, this defaults to a noop.
  virtual SDOperand LowerVAEnd(SDOperand Chain, SDOperand L, SelectionDAG &DAG);

  /// LowerVACopy - This lowers llvm.va_copy and returns the resultant
  /// value/chain pair.  If not implemented, this defaults to returning the
  /// input operand.
  virtual std::pair<SDOperand,SDOperand>
  LowerVACopy(SDOperand Chain, SDOperand L, SelectionDAG &DAG);

  /// LowerVAArgNext - This lowers the vaarg and vanext instructions (depending
  /// on whether the first argument is true).  If not implemented, this prints a
  /// message and aborts.
  virtual std::pair<SDOperand,SDOperand>
  LowerVAArgNext(bool isVANext, SDOperand Chain, SDOperand VAList,
                 const Type *ArgTy, SelectionDAG &DAG);

  /// LowerFrameReturnAddress - This hook lowers a call to llvm.returnaddress or
  /// llvm.frameaddress (depending on the value of the first argument).  The
  /// return values are the result pointer and the resultant token chain.  If
  /// not implemented, both of these intrinsics will return null.
  virtual std::pair<SDOperand, SDOperand>
  LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                          SelectionDAG &DAG);
};
} // end llvm namespace

#endif
