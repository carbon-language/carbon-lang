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
  
  /// hasNativeSupportFor 
  bool hasNativeSupportFor(MVT::ValueType VT) {
    return RegClassForVT[VT] != 0;
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
  /// actual call.
  typedef std::vector<std::pair<SDOperand, const Type*> > ArgListTy;
  virtual SDNode *LowerCallTo(const Type *RetTy, SDOperand Callee,
                              ArgListTy &Args, SelectionDAG &DAG) = 0;
};
} // end llvm namespace

#endif
