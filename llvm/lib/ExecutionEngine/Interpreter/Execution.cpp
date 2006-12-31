//===-- Execution.cpp - Implement code to simulate the program ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains the actual instruction interpreter.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "interpreter"
#include "Interpreter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include <cmath>
using namespace llvm;

STATISTIC(NumDynamicInsts, "Number of dynamic instructions executed");
static Interpreter *TheEE = 0;


//===----------------------------------------------------------------------===//
//                     Value Manipulation code
//===----------------------------------------------------------------------===//

static GenericValue executeAddInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty);
static GenericValue executeSubInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty);
static GenericValue executeMulInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty);
static GenericValue executeUDivInst(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty);
static GenericValue executeSDivInst(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty);
static GenericValue executeFDivInst(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty);
static GenericValue executeURemInst(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty);
static GenericValue executeSRemInst(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty);
static GenericValue executeFRemInst(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty);
static GenericValue executeAndInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty);
static GenericValue executeOrInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty);
static GenericValue executeXorInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty);
static GenericValue executeCmpInst(unsigned predicate, GenericValue Src1, 
                                   GenericValue Src2, const Type *Ty);
static GenericValue executeShlInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty);
static GenericValue executeLShrInst(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty);
static GenericValue executeAShrInst(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty);
static GenericValue executeSelectInst(GenericValue Src1, GenericValue Src2,
                                      GenericValue Src3);

GenericValue Interpreter::getConstantExprValue (ConstantExpr *CE,
                                                ExecutionContext &SF) {
  switch (CE->getOpcode()) {
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    return executeCastOperation(Instruction::CastOps(CE->getOpcode()), 
                                CE->getOperand(0), CE->getType(), SF);
  case Instruction::GetElementPtr:
    return executeGEPOperation(CE->getOperand(0), gep_type_begin(CE),
                               gep_type_end(CE), SF);
  case Instruction::Add:
    return executeAddInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Sub:
    return executeSubInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Mul:
    return executeMulInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::SDiv:
    return executeSDivInst(getOperandValue(CE->getOperand(0), SF),
                           getOperandValue(CE->getOperand(1), SF),
                           CE->getOperand(0)->getType());
  case Instruction::UDiv:
    return executeUDivInst(getOperandValue(CE->getOperand(0), SF),
                           getOperandValue(CE->getOperand(1), SF),
                           CE->getOperand(0)->getType());
  case Instruction::FDiv:
    return executeFDivInst(getOperandValue(CE->getOperand(0), SF),
                           getOperandValue(CE->getOperand(1), SF),
                           CE->getOperand(0)->getType());
  case Instruction::URem:
    return executeURemInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::SRem:
    return executeSRemInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::FRem:
    return executeFRemInst(getOperandValue(CE->getOperand(0), SF),
                           getOperandValue(CE->getOperand(1), SF),
                           CE->getOperand(0)->getType());
  case Instruction::And:
    return executeAndInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Or:
    return executeOrInst(getOperandValue(CE->getOperand(0), SF),
                         getOperandValue(CE->getOperand(1), SF),
                         CE->getOperand(0)->getType());
  case Instruction::Xor:
    return executeXorInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::FCmp:
  case Instruction::ICmp:
    return executeCmpInst(CE->getPredicate(),
                          getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Shl:
    return executeShlInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::LShr:
    return executeLShrInst(getOperandValue(CE->getOperand(0), SF),
                           getOperandValue(CE->getOperand(1), SF),
                           CE->getOperand(0)->getType());
  case Instruction::AShr:
    return executeAShrInst(getOperandValue(CE->getOperand(0), SF),
                           getOperandValue(CE->getOperand(1), SF),
                           CE->getOperand(0)->getType());
  case Instruction::Select:
    return executeSelectInst(getOperandValue(CE->getOperand(0), SF),
                             getOperandValue(CE->getOperand(1), SF),
                             getOperandValue(CE->getOperand(2), SF));
  default:
    cerr << "Unhandled ConstantExpr: " << *CE << "\n";
    abort();
    return GenericValue();
  }
}

GenericValue Interpreter::getOperandValue(Value *V, ExecutionContext &SF) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    return getConstantExprValue(CE, SF);
  } else if (Constant *CPV = dyn_cast<Constant>(V)) {
    return getConstantValue(CPV);
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    return PTOGV(getPointerToGlobal(GV));
  } else {
    return SF.Values[V];
  }
}

static void SetValue(Value *V, GenericValue Val, ExecutionContext &SF) {
  SF.Values[V] = Val;
}

void Interpreter::initializeExecutionEngine() {
  TheEE = this;
}

//===----------------------------------------------------------------------===//
//                    Binary Instruction Implementations
//===----------------------------------------------------------------------===//

#define IMPLEMENT_BINARY_OPERATOR(OP, TY) \
   case Type::TY##TyID: Dest.TY##Val = Src1.TY##Val OP Src2.TY##Val; break

static GenericValue executeAddInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(+, Int8);
    IMPLEMENT_BINARY_OPERATOR(+, Int16);
    IMPLEMENT_BINARY_OPERATOR(+, Int32);
    IMPLEMENT_BINARY_OPERATOR(+, Int64);
    IMPLEMENT_BINARY_OPERATOR(+, Float);
    IMPLEMENT_BINARY_OPERATOR(+, Double);
  default:
    cerr << "Unhandled type for Add instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeSubInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(-, Int8);
    IMPLEMENT_BINARY_OPERATOR(-, Int16);
    IMPLEMENT_BINARY_OPERATOR(-, Int32);
    IMPLEMENT_BINARY_OPERATOR(-, Int64);
    IMPLEMENT_BINARY_OPERATOR(-, Float);
    IMPLEMENT_BINARY_OPERATOR(-, Double);
  default:
    cerr << "Unhandled type for Sub instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeMulInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(*, Int8);
    IMPLEMENT_BINARY_OPERATOR(*, Int16);
    IMPLEMENT_BINARY_OPERATOR(*, Int32);
    IMPLEMENT_BINARY_OPERATOR(*, Int64);
    IMPLEMENT_BINARY_OPERATOR(*, Float);
    IMPLEMENT_BINARY_OPERATOR(*, Double);
  default:
    cerr << "Unhandled type for Mul instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

#define IMPLEMENT_SIGNLESS_BINOP(OP, TY, CAST) \
   case Type::TY##TyID: Dest.TY##Val = \
    ((CAST)Src1.TY##Val) OP ((CAST)Src2.TY##Val); break

static GenericValue executeUDivInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_SIGNLESS_BINOP(/, Int8,  uint8_t);
    IMPLEMENT_SIGNLESS_BINOP(/, Int16, uint16_t);
    IMPLEMENT_SIGNLESS_BINOP(/, Int32, uint32_t);
    IMPLEMENT_SIGNLESS_BINOP(/, Int64, uint64_t);
  default:
    cerr << "Unhandled type for UDiv instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeSDivInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_SIGNLESS_BINOP(/, Int8,  int8_t);
    IMPLEMENT_SIGNLESS_BINOP(/, Int16, int16_t);
    IMPLEMENT_SIGNLESS_BINOP(/, Int32, int32_t);
    IMPLEMENT_SIGNLESS_BINOP(/, Int64, int64_t);
  default:
    cerr << "Unhandled type for SDiv instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeFDivInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(/, Float);
    IMPLEMENT_BINARY_OPERATOR(/, Double);
  default:
    cerr << "Unhandled type for Div instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeURemInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_SIGNLESS_BINOP(%, Int8,  uint8_t);
    IMPLEMENT_SIGNLESS_BINOP(%, Int16, uint16_t);
    IMPLEMENT_SIGNLESS_BINOP(%, Int32, uint32_t);
    IMPLEMENT_SIGNLESS_BINOP(%, Int64, uint64_t );
  default:
    cerr << "Unhandled type for URem instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeSRemInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_SIGNLESS_BINOP(%, Int8,  int8_t);
    IMPLEMENT_SIGNLESS_BINOP(%, Int16, int16_t);
    IMPLEMENT_SIGNLESS_BINOP(%, Int32, int32_t);
    IMPLEMENT_SIGNLESS_BINOP(%, Int64, int64_t);
  default:
    cerr << "Unhandled type for Rem instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeFRemInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
  case Type::FloatTyID:
    Dest.FloatVal = fmod(Src1.FloatVal, Src2.FloatVal);
    break;
  case Type::DoubleTyID:
    Dest.DoubleVal = fmod(Src1.DoubleVal, Src2.DoubleVal);
    break;
  default:
    cerr << "Unhandled type for Rem instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeAndInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(&, Bool);
    IMPLEMENT_BINARY_OPERATOR(&, Int8);
    IMPLEMENT_BINARY_OPERATOR(&, Int16);
    IMPLEMENT_BINARY_OPERATOR(&, Int32);
    IMPLEMENT_BINARY_OPERATOR(&, Int64);
  default:
    cerr << "Unhandled type for And instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeOrInst(GenericValue Src1, GenericValue Src2,
                                  const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(|, Bool);
    IMPLEMENT_BINARY_OPERATOR(|, Int8);
    IMPLEMENT_BINARY_OPERATOR(|, Int16);
    IMPLEMENT_BINARY_OPERATOR(|, Int32);
    IMPLEMENT_BINARY_OPERATOR(|, Int64);
  default:
    cerr << "Unhandled type for Or instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeXorInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(^, Bool);
    IMPLEMENT_BINARY_OPERATOR(^, Int8);
    IMPLEMENT_BINARY_OPERATOR(^, Int16);
    IMPLEMENT_BINARY_OPERATOR(^, Int32);
    IMPLEMENT_BINARY_OPERATOR(^, Int64);
  default:
    cerr << "Unhandled type for Xor instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

#define IMPLEMENT_ICMP(OP, TY, CAST) \
   case Type::TY##TyID: Dest.BoolVal = \
     ((CAST)Src1.TY##Val) OP ((CAST)Src2.TY##Val); break

// Handle pointers specially because they must be compared with only as much
// width as the host has.  We _do not_ want to be comparing 64 bit values when
// running on a 32-bit target, otherwise the upper 32 bits might mess up
// comparisons if they contain garbage.
#define IMPLEMENT_POINTERCMP(OP) \
   case Type::PointerTyID: \
        Dest.BoolVal = (void*)(intptr_t)Src1.PointerVal OP \
                       (void*)(intptr_t)Src2.PointerVal; break

static GenericValue executeICMP_EQ(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_ICMP(==, Int8,  uint8_t);
    IMPLEMENT_ICMP(==, Int16, uint16_t);
    IMPLEMENT_ICMP(==, Int32, uint32_t);
    IMPLEMENT_ICMP(==, Int64, uint64_t);
    IMPLEMENT_POINTERCMP(==);
  default:
    cerr << "Unhandled type for ICMP_EQ predicate: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeICMP_NE(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_ICMP(!=, Int8,  uint8_t);
    IMPLEMENT_ICMP(!=, Int16, uint16_t);
    IMPLEMENT_ICMP(!=, Int32, uint32_t);
    IMPLEMENT_ICMP(!=, Int64, uint64_t);
    IMPLEMENT_POINTERCMP(!=);
  default:
    cerr << "Unhandled type for ICMP_NE predicate: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeICMP_ULT(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_ICMP(<, Int8,  uint8_t);
    IMPLEMENT_ICMP(<, Int16, uint16_t);
    IMPLEMENT_ICMP(<, Int32, uint32_t);
    IMPLEMENT_ICMP(<, Int64, uint64_t);
    IMPLEMENT_POINTERCMP(<);
  default:
    cerr << "Unhandled type for ICMP_ULT predicate: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeICMP_SLT(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_ICMP(<, Int8,  int8_t);
    IMPLEMENT_ICMP(<, Int16, int16_t);
    IMPLEMENT_ICMP(<, Int32, int32_t);
    IMPLEMENT_ICMP(<, Int64, int64_t);
    IMPLEMENT_POINTERCMP(<);
  default:
    cerr << "Unhandled type for ICMP_SLT predicate: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeICMP_UGT(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_ICMP(>, Int8,  uint8_t);
    IMPLEMENT_ICMP(>, Int16, uint16_t);
    IMPLEMENT_ICMP(>, Int32, uint32_t);
    IMPLEMENT_ICMP(>, Int64, uint64_t);
    IMPLEMENT_POINTERCMP(>);
  default:
    cerr << "Unhandled type for ICMP_UGT predicate: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeICMP_SGT(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_ICMP(>, Int8,  int8_t);
    IMPLEMENT_ICMP(>, Int16, int16_t);
    IMPLEMENT_ICMP(>, Int32, int32_t);
    IMPLEMENT_ICMP(>, Int64, int64_t);
    IMPLEMENT_POINTERCMP(>);
  default:
    cerr << "Unhandled type for ICMP_SGT predicate: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeICMP_ULE(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_ICMP(<=, Int8,  uint8_t);
    IMPLEMENT_ICMP(<=, Int16, uint16_t);
    IMPLEMENT_ICMP(<=, Int32, uint32_t);
    IMPLEMENT_ICMP(<=, Int64, uint64_t);
    IMPLEMENT_POINTERCMP(<=);
  default:
    cerr << "Unhandled type for ICMP_ULE predicate: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeICMP_SLE(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_ICMP(<=, Int8,  int8_t);
    IMPLEMENT_ICMP(<=, Int16, int16_t);
    IMPLEMENT_ICMP(<=, Int32, int32_t);
    IMPLEMENT_ICMP(<=, Int64, int64_t);
    IMPLEMENT_POINTERCMP(<=);
  default:
    cerr << "Unhandled type for ICMP_SLE predicate: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeICMP_UGE(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_ICMP(>=, Int8,  uint8_t);
    IMPLEMENT_ICMP(>=, Int16, uint16_t);
    IMPLEMENT_ICMP(>=, Int32, uint32_t);
    IMPLEMENT_ICMP(>=, Int64, uint64_t);
    IMPLEMENT_POINTERCMP(>=);
  default:
    cerr << "Unhandled type for ICMP_UGE predicate: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeICMP_SGE(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_ICMP(>=, Int8,  int8_t);
    IMPLEMENT_ICMP(>=, Int16, int16_t);
    IMPLEMENT_ICMP(>=, Int32, int32_t);
    IMPLEMENT_ICMP(>=, Int64, int64_t);
    IMPLEMENT_POINTERCMP(>=);
  default:
    cerr << "Unhandled type for ICMP_SGE predicate: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

void Interpreter::visitICmpInst(ICmpInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue R;   // Result
  
  switch (I.getPredicate()) {
  case ICmpInst::ICMP_EQ:  R = executeICMP_EQ(Src1, Src2, Ty);  break;
  case ICmpInst::ICMP_NE:  R = executeICMP_NE(Src1, Src2, Ty);  break;
  case ICmpInst::ICMP_ULT: R = executeICMP_ULT(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_SLT: R = executeICMP_SLT(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_UGT: R = executeICMP_UGT(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_SGT: R = executeICMP_SGT(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_ULE: R = executeICMP_ULE(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_SLE: R = executeICMP_SLE(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_UGE: R = executeICMP_UGE(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_SGE: R = executeICMP_SGE(Src1, Src2, Ty); break;
  default:
    cerr << "Don't know how to handle this ICmp predicate!\n-->" << I;
    abort();
  }
 
  SetValue(&I, R, SF);
}

#define IMPLEMENT_FCMP(OP, TY) \
   case Type::TY##TyID: Dest.BoolVal = Src1.TY##Val OP Src2.TY##Val; break

static GenericValue executeFCMP_EQ(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(==, Float);
    IMPLEMENT_FCMP(==, Double);
  default:
    cerr << "Unhandled type for SetEQ instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeFCMP_NE(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(!=, Float);
    IMPLEMENT_FCMP(!=, Double);

  default:
    cerr << "Unhandled type for SetNE instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeFCMP_LE(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(<=, Float);
    IMPLEMENT_FCMP(<=, Double);
  default:
    cerr << "Unhandled type for SetLE instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeFCMP_GE(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(>=, Float);
    IMPLEMENT_FCMP(>=, Double);
  default:
    cerr << "Unhandled type for SetGE instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeFCMP_LT(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(<, Float);
    IMPLEMENT_FCMP(<, Double);
  default:
    cerr << "Unhandled type for SetLT instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeFCMP_GT(GenericValue Src1, GenericValue Src2,
                                     const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(>, Float);
    IMPLEMENT_FCMP(>, Double);
  default:
    cerr << "Unhandled type for SetGT instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

void Interpreter::visitFCmpInst(FCmpInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue R;   // Result
  
  switch (I.getPredicate()) {
  case FCmpInst::FCMP_FALSE: R.BoolVal = false;
  case FCmpInst::FCMP_ORD:   R = executeFCMP_EQ(Src1, Src2, Ty); break; ///???
  case FCmpInst::FCMP_UNO:   R = executeFCMP_NE(Src1, Src2, Ty); break; ///???
  case FCmpInst::FCMP_OEQ:
  case FCmpInst::FCMP_UEQ:   R = executeFCMP_EQ(Src1, Src2, Ty);  break;
  case FCmpInst::FCMP_ONE:
  case FCmpInst::FCMP_UNE:   R = executeFCMP_NE(Src1, Src2, Ty);  break;
  case FCmpInst::FCMP_OLT:
  case FCmpInst::FCMP_ULT:   R = executeFCMP_LT(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OGT:
  case FCmpInst::FCMP_UGT:   R = executeFCMP_GT(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OLE:
  case FCmpInst::FCMP_ULE:   R = executeFCMP_LE(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OGE:
  case FCmpInst::FCMP_UGE:   R = executeFCMP_GE(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_TRUE:  R.BoolVal = true;
  default:
    cerr << "Don't know how to handle this FCmp predicate!\n-->" << I;
    abort();
  }
 
  SetValue(&I, R, SF);
}

static GenericValue executeCmpInst(unsigned predicate, GenericValue Src1, 
                                   GenericValue Src2, const Type *Ty) {
  GenericValue Result;
  switch (predicate) {
  case ICmpInst::ICMP_EQ:    return executeICMP_EQ(Src1, Src2, Ty);
  case ICmpInst::ICMP_NE:    return executeICMP_NE(Src1, Src2, Ty);
  case ICmpInst::ICMP_UGT:   return executeICMP_UGT(Src1, Src2, Ty);
  case ICmpInst::ICMP_SGT:   return executeICMP_SGT(Src1, Src2, Ty);
  case ICmpInst::ICMP_ULT:   return executeICMP_ULT(Src1, Src2, Ty);
  case ICmpInst::ICMP_SLT:   return executeICMP_SLT(Src1, Src2, Ty);
  case ICmpInst::ICMP_UGE:   return executeICMP_UGE(Src1, Src2, Ty);
  case ICmpInst::ICMP_SGE:   return executeICMP_SGE(Src1, Src2, Ty);
  case ICmpInst::ICMP_ULE:   return executeICMP_ULE(Src1, Src2, Ty);
  case ICmpInst::ICMP_SLE:   return executeICMP_SLE(Src1, Src2, Ty);
  case FCmpInst::FCMP_ORD:   return executeFCMP_EQ(Src1, Src2, Ty); break; 
  case FCmpInst::FCMP_UNO:   return executeFCMP_NE(Src1, Src2, Ty); break; 
  case FCmpInst::FCMP_OEQ:
  case FCmpInst::FCMP_UEQ:   return executeFCMP_EQ(Src1, Src2, Ty);  break;
  case FCmpInst::FCMP_ONE:
  case FCmpInst::FCMP_UNE:   return executeFCMP_NE(Src1, Src2, Ty);  break;
  case FCmpInst::FCMP_OLT:
  case FCmpInst::FCMP_ULT:   return executeFCMP_LT(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OGT:
  case FCmpInst::FCMP_UGT:   return executeFCMP_GT(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OLE:
  case FCmpInst::FCMP_ULE:   return executeFCMP_LE(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OGE:
  case FCmpInst::FCMP_UGE:   return executeFCMP_GE(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_FALSE: { 
    GenericValue Result;
    Result.BoolVal = false; 
    return Result;
  }
  case FCmpInst::FCMP_TRUE: {
    GenericValue Result;
    Result.BoolVal = true;
    return Result;
  }
  default:
    cerr << "Unhandled Cmp predicate\n";
    abort();
  }
}

void Interpreter::visitBinaryOperator(BinaryOperator &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue R;   // Result

  switch (I.getOpcode()) {
  case Instruction::Add:   R = executeAddInst  (Src1, Src2, Ty); break;
  case Instruction::Sub:   R = executeSubInst  (Src1, Src2, Ty); break;
  case Instruction::Mul:   R = executeMulInst  (Src1, Src2, Ty); break;
  case Instruction::UDiv:  R = executeUDivInst (Src1, Src2, Ty); break;
  case Instruction::SDiv:  R = executeSDivInst (Src1, Src2, Ty); break;
  case Instruction::FDiv:  R = executeFDivInst (Src1, Src2, Ty); break;
  case Instruction::URem:  R = executeURemInst (Src1, Src2, Ty); break;
  case Instruction::SRem:  R = executeSRemInst (Src1, Src2, Ty); break;
  case Instruction::FRem:  R = executeFRemInst (Src1, Src2, Ty); break;
  case Instruction::And:   R = executeAndInst  (Src1, Src2, Ty); break;
  case Instruction::Or:    R = executeOrInst   (Src1, Src2, Ty); break;
  case Instruction::Xor:   R = executeXorInst  (Src1, Src2, Ty); break;
  default:
    cerr << "Don't know how to handle this binary operator!\n-->" << I;
    abort();
  }

  SetValue(&I, R, SF);
}

static GenericValue executeSelectInst(GenericValue Src1, GenericValue Src2,
                                      GenericValue Src3) {
  return Src1.BoolVal ? Src2 : Src3;
}

void Interpreter::visitSelectInst(SelectInst &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue Src3 = getOperandValue(I.getOperand(2), SF);
  GenericValue R = executeSelectInst(Src1, Src2, Src3);
  SetValue(&I, R, SF);
}


//===----------------------------------------------------------------------===//
//                     Terminator Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::exitCalled(GenericValue GV) {
  // runAtExitHandlers() assumes there are no stack frames, but
  // if exit() was called, then it had a stack frame. Blow away
  // the stack before interpreting atexit handlers.
  ECStack.clear ();
  runAtExitHandlers ();
  exit (GV.Int32Val);
}

/// Pop the last stack frame off of ECStack and then copy the result
/// back into the result variable if we are not returning void. The
/// result variable may be the ExitValue, or the Value of the calling
/// CallInst if there was a previous stack frame. This method may
/// invalidate any ECStack iterators you have. This method also takes
/// care of switching to the normal destination BB, if we are returning
/// from an invoke.
///
void Interpreter::popStackAndReturnValueToCaller (const Type *RetTy,
                                                  GenericValue Result) {
  // Pop the current stack frame.
  ECStack.pop_back();

  if (ECStack.empty()) {  // Finished main.  Put result into exit code...
    if (RetTy && RetTy->isIntegral()) {          // Nonvoid return type?
      ExitValue = Result;   // Capture the exit value of the program
    } else {
      memset(&ExitValue, 0, sizeof(ExitValue));
    }
  } else {
    // If we have a previous stack frame, and we have a previous call,
    // fill in the return value...
    ExecutionContext &CallingSF = ECStack.back();
    if (Instruction *I = CallingSF.Caller.getInstruction()) {
      if (CallingSF.Caller.getType() != Type::VoidTy)      // Save result...
        SetValue(I, Result, CallingSF);
      if (InvokeInst *II = dyn_cast<InvokeInst> (I))
        SwitchToNewBasicBlock (II->getNormalDest (), CallingSF);
      CallingSF.Caller = CallSite();          // We returned from the call...
    }
  }
}

void Interpreter::visitReturnInst(ReturnInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *RetTy = Type::VoidTy;
  GenericValue Result;

  // Save away the return value... (if we are not 'ret void')
  if (I.getNumOperands()) {
    RetTy  = I.getReturnValue()->getType();
    Result = getOperandValue(I.getReturnValue(), SF);
  }

  popStackAndReturnValueToCaller(RetTy, Result);
}

void Interpreter::visitUnwindInst(UnwindInst &I) {
  // Unwind stack
  Instruction *Inst;
  do {
    ECStack.pop_back ();
    if (ECStack.empty ())
      abort ();
    Inst = ECStack.back ().Caller.getInstruction ();
  } while (!(Inst && isa<InvokeInst> (Inst)));

  // Return from invoke
  ExecutionContext &InvokingSF = ECStack.back ();
  InvokingSF.Caller = CallSite ();

  // Go to exceptional destination BB of invoke instruction
  SwitchToNewBasicBlock(cast<InvokeInst>(Inst)->getUnwindDest(), InvokingSF);
}

void Interpreter::visitUnreachableInst(UnreachableInst &I) {
  cerr << "ERROR: Program executed an 'unreachable' instruction!\n";
  abort();
}

void Interpreter::visitBranchInst(BranchInst &I) {
  ExecutionContext &SF = ECStack.back();
  BasicBlock *Dest;

  Dest = I.getSuccessor(0);          // Uncond branches have a fixed dest...
  if (!I.isUnconditional()) {
    Value *Cond = I.getCondition();
    if (getOperandValue(Cond, SF).BoolVal == 0) // If false cond...
      Dest = I.getSuccessor(1);
  }
  SwitchToNewBasicBlock(Dest, SF);
}

void Interpreter::visitSwitchInst(SwitchInst &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue CondVal = getOperandValue(I.getOperand(0), SF);
  const Type *ElTy = I.getOperand(0)->getType();

  // Check to see if any of the cases match...
  BasicBlock *Dest = 0;
  for (unsigned i = 2, e = I.getNumOperands(); i != e; i += 2)
    if (executeICMP_EQ(CondVal,
                       getOperandValue(I.getOperand(i), SF), ElTy).BoolVal) {
      Dest = cast<BasicBlock>(I.getOperand(i+1));
      break;
    }

  if (!Dest) Dest = I.getDefaultDest();   // No cases matched: use default
  SwitchToNewBasicBlock(Dest, SF);
}

// SwitchToNewBasicBlock - This method is used to jump to a new basic block.
// This function handles the actual updating of block and instruction iterators
// as well as execution of all of the PHI nodes in the destination block.
//
// This method does this because all of the PHI nodes must be executed
// atomically, reading their inputs before any of the results are updated.  Not
// doing this can cause problems if the PHI nodes depend on other PHI nodes for
// their inputs.  If the input PHI node is updated before it is read, incorrect
// results can happen.  Thus we use a two phase approach.
//
void Interpreter::SwitchToNewBasicBlock(BasicBlock *Dest, ExecutionContext &SF){
  BasicBlock *PrevBB = SF.CurBB;      // Remember where we came from...
  SF.CurBB   = Dest;                  // Update CurBB to branch destination
  SF.CurInst = SF.CurBB->begin();     // Update new instruction ptr...

  if (!isa<PHINode>(SF.CurInst)) return;  // Nothing fancy to do

  // Loop over all of the PHI nodes in the current block, reading their inputs.
  std::vector<GenericValue> ResultValues;

  for (; PHINode *PN = dyn_cast<PHINode>(SF.CurInst); ++SF.CurInst) {
    // Search for the value corresponding to this previous bb...
    int i = PN->getBasicBlockIndex(PrevBB);
    assert(i != -1 && "PHINode doesn't contain entry for predecessor??");
    Value *IncomingValue = PN->getIncomingValue(i);

    // Save the incoming value for this PHI node...
    ResultValues.push_back(getOperandValue(IncomingValue, SF));
  }

  // Now loop over all of the PHI nodes setting their values...
  SF.CurInst = SF.CurBB->begin();
  for (unsigned i = 0; isa<PHINode>(SF.CurInst); ++SF.CurInst, ++i) {
    PHINode *PN = cast<PHINode>(SF.CurInst);
    SetValue(PN, ResultValues[i], SF);
  }
}

//===----------------------------------------------------------------------===//
//                     Memory Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::visitAllocationInst(AllocationInst &I) {
  ExecutionContext &SF = ECStack.back();

  const Type *Ty = I.getType()->getElementType();  // Type to be allocated

  // Get the number of elements being allocated by the array...
  unsigned NumElements = getOperandValue(I.getOperand(0), SF).Int32Val;

  // Allocate enough memory to hold the type...
  void *Memory = malloc(NumElements * (size_t)TD.getTypeSize(Ty));

  GenericValue Result = PTOGV(Memory);
  assert(Result.PointerVal != 0 && "Null pointer returned by malloc!");
  SetValue(&I, Result, SF);

  if (I.getOpcode() == Instruction::Alloca)
    ECStack.back().Allocas.add(Memory);
}

void Interpreter::visitFreeInst(FreeInst &I) {
  ExecutionContext &SF = ECStack.back();
  assert(isa<PointerType>(I.getOperand(0)->getType()) && "Freeing nonptr?");
  GenericValue Value = getOperandValue(I.getOperand(0), SF);
  // TODO: Check to make sure memory is allocated
  free(GVTOP(Value));   // Free memory
}

// getElementOffset - The workhorse for getelementptr.
//
GenericValue Interpreter::executeGEPOperation(Value *Ptr, gep_type_iterator I,
                                              gep_type_iterator E,
                                              ExecutionContext &SF) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "Cannot getElementOffset of a nonpointer type!");

  PointerTy Total = 0;

  for (; I != E; ++I) {
    if (const StructType *STy = dyn_cast<StructType>(*I)) {
      const StructLayout *SLO = TD.getStructLayout(STy);

      const ConstantInt *CPU = cast<ConstantInt>(I.getOperand());
      unsigned Index = unsigned(CPU->getZExtValue());

      Total += (PointerTy)SLO->MemberOffsets[Index];
    } else {
      const SequentialType *ST = cast<SequentialType>(*I);
      // Get the index number for the array... which must be long type...
      GenericValue IdxGV = getOperandValue(I.getOperand(), SF);

      uint64_t Idx;
      switch (I.getOperand()->getType()->getTypeID()) {
      default: assert(0 && "Illegal getelementptr index for sequential type!");
      case Type::Int8TyID:  Idx = IdxGV.Int8Val; break;
      case Type::Int16TyID: Idx = IdxGV.Int16Val; break;
      case Type::Int32TyID: Idx = IdxGV.Int32Val; break;
      case Type::Int64TyID: Idx = IdxGV.Int64Val; break;
      }
      Total += PointerTy(TD.getTypeSize(ST->getElementType())*Idx);
    }
  }

  GenericValue Result;
  Result.PointerVal = getOperandValue(Ptr, SF).PointerVal + Total;
  return Result;
}

void Interpreter::visitGetElementPtrInst(GetElementPtrInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, TheEE->executeGEPOperation(I.getPointerOperand(),
                                   gep_type_begin(I), gep_type_end(I), SF), SF);
}

void Interpreter::visitLoadInst(LoadInst &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue SRC = getOperandValue(I.getPointerOperand(), SF);
  GenericValue *Ptr = (GenericValue*)GVTOP(SRC);
  GenericValue Result = LoadValueFromMemory(Ptr, I.getType());
  SetValue(&I, Result, SF);
}

void Interpreter::visitStoreInst(StoreInst &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue Val = getOperandValue(I.getOperand(0), SF);
  GenericValue SRC = getOperandValue(I.getPointerOperand(), SF);
  StoreValueToMemory(Val, (GenericValue *)GVTOP(SRC),
                     I.getOperand(0)->getType());
}

//===----------------------------------------------------------------------===//
//                 Miscellaneous Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::visitCallSite(CallSite CS) {
  ExecutionContext &SF = ECStack.back();

  // Check to see if this is an intrinsic function call...
  if (Function *F = CS.getCalledFunction())
   if (F->isExternal ())
    switch (F->getIntrinsicID()) {
    case Intrinsic::not_intrinsic:
      break;
    case Intrinsic::vastart: { // va_start
      GenericValue ArgIndex;
      ArgIndex.UIntPairVal.first = ECStack.size() - 1;
      ArgIndex.UIntPairVal.second = 0;
      SetValue(CS.getInstruction(), ArgIndex, SF);
      return;
    }
    case Intrinsic::vaend:    // va_end is a noop for the interpreter
      return;
    case Intrinsic::vacopy:   // va_copy: dest = src
      SetValue(CS.getInstruction(), getOperandValue(*CS.arg_begin(), SF), SF);
      return;
    default:
      // If it is an unknown intrinsic function, use the intrinsic lowering
      // class to transform it into hopefully tasty LLVM code.
      //
      Instruction *Prev = CS.getInstruction()->getPrev();
      BasicBlock *Parent = CS.getInstruction()->getParent();
      IL->LowerIntrinsicCall(cast<CallInst>(CS.getInstruction()));

      // Restore the CurInst pointer to the first instruction newly inserted, if
      // any.
      if (!Prev) {
        SF.CurInst = Parent->begin();
      } else {
        SF.CurInst = Prev;
        ++SF.CurInst;
      }
      return;
    }

  SF.Caller = CS;
  std::vector<GenericValue> ArgVals;
  const unsigned NumArgs = SF.Caller.arg_size();
  ArgVals.reserve(NumArgs);
  for (CallSite::arg_iterator i = SF.Caller.arg_begin(),
         e = SF.Caller.arg_end(); i != e; ++i) {
    Value *V = *i;
    ArgVals.push_back(getOperandValue(V, SF));
    // Promote all integral types whose size is < sizeof(int) into ints.  We do
    // this by zero or sign extending the value as appropriate according to the
    // source type.
    const Type *Ty = V->getType();
    if (Ty->isIntegral() && Ty->getPrimitiveSize() < 4) {
      if (Ty == Type::Int16Ty)
        ArgVals.back().Int32Val = ArgVals.back().Int16Val;
      else if (Ty == Type::Int8Ty)
        ArgVals.back().Int32Val = ArgVals.back().Int8Val;
      else if (Ty == Type::BoolTy)
        ArgVals.back().Int32Val = ArgVals.back().BoolVal;
      else
        assert(0 && "Unknown type!");
    }
  }

  // To handle indirect calls, we must get the pointer value from the argument
  // and treat it as a function pointer.
  GenericValue SRC = getOperandValue(SF.Caller.getCalledValue(), SF);
  callFunction((Function*)GVTOP(SRC), ArgVals);
}

#define IMPLEMENT_SHIFT(OP, TY) \
   case Type::TY##TyID: Dest.TY##Val = Src1.TY##Val OP Src2.Int8Val; break

#define IMPLEMENT_SIGNLESS_SHIFT(OP, TY, CAST) \
   case Type::TY##TyID: Dest.TY##Val = ((CAST)Src1.TY##Val) OP Src2.Int8Val; \
     break

static GenericValue executeShlInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_SHIFT(<<, Int8);
    IMPLEMENT_SHIFT(<<, Int16);
    IMPLEMENT_SHIFT(<<, Int32);
    IMPLEMENT_SHIFT(<<, Int64);
  default:
    cerr << "Unhandled type for Shl instruction: " << *Ty << "\n";
  }
  return Dest;
}

static GenericValue executeLShrInst(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_SIGNLESS_SHIFT(>>, Int8,  uint8_t);
    IMPLEMENT_SIGNLESS_SHIFT(>>, Int16, uint16_t);
    IMPLEMENT_SIGNLESS_SHIFT(>>, Int32, uint32_t);
    IMPLEMENT_SIGNLESS_SHIFT(>>, Int64, uint64_t);
  default:
    cerr << "Unhandled type for LShr instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeAShrInst(GenericValue Src1, GenericValue Src2,
                                    const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_SIGNLESS_SHIFT(>>, Int8,  int8_t);
    IMPLEMENT_SIGNLESS_SHIFT(>>, Int16, int16_t);
    IMPLEMENT_SIGNLESS_SHIFT(>>, Int32, int32_t);
    IMPLEMENT_SIGNLESS_SHIFT(>>, Int64, int64_t);
  default:
    cerr << "Unhandled type for AShr instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

void Interpreter::visitShl(ShiftInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue Dest;
  Dest = executeShlInst (Src1, Src2, Ty);
  SetValue(&I, Dest, SF);
}

void Interpreter::visitLShr(ShiftInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue Dest;
  Dest = executeLShrInst (Src1, Src2, Ty);
  SetValue(&I, Dest, SF);
}

void Interpreter::visitAShr(ShiftInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue Dest;
  Dest = executeAShrInst (Src1, Src2, Ty);
  SetValue(&I, Dest, SF);
}

#define IMPLEMENT_CAST_START \
  switch (DstTy->getTypeID()) {

#define IMPLEMENT_CAST(STY, DTY, CAST) \
     case Type::STY##TyID: Dest.DTY##Val = (CAST(Src.STY##Val)); break;

#define IMPLEMENT_CAST_CASE(DTY, CAST)          \
  case Type::DTY##TyID:                         \
    switch (SrcTy->getTypeID()) {               \
      IMPLEMENT_CAST(Bool,   DTY, CAST);        \
      IMPLEMENT_CAST(Int8,   DTY, CAST);        \
      IMPLEMENT_CAST(Int16,  DTY, CAST);        \
      IMPLEMENT_CAST(Int32,  DTY, CAST);        \
      IMPLEMENT_CAST(Int64,  DTY, CAST);        \
      IMPLEMENT_CAST(Pointer,DTY, CAST);        \
      IMPLEMENT_CAST(Float,  DTY, CAST);        \
      IMPLEMENT_CAST(Double, DTY, CAST);        \
    default:                                    \
      cerr << "Unhandled cast: "                \
        << *SrcTy << " to " << *DstTy << "\n";  \
      abort();                                  \
    }                                           \
    break

#define IMPLEMENT_CAST_END                      \
  default: cerr                                 \
      << "Unhandled dest type for cast instruction: "  \
      << *DstTy << "\n";                        \
    abort();                                    \
  }

GenericValue Interpreter::executeCastOperation(Instruction::CastOps opcode,
                                               Value *SrcVal, const Type *DstTy,
                                               ExecutionContext &SF) {
  const Type *SrcTy = SrcVal->getType();
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);

  if (opcode == Instruction::Trunc && DstTy->getTypeID() == Type::BoolTyID) {
    // For truncations to bool, we must clear the high order bits of the source
    switch (SrcTy->getTypeID()) {
      case Type::BoolTyID:  Src.BoolVal  &= 1; break;
      case Type::Int8TyID:  Src.Int8Val  &= 1; break;
      case Type::Int16TyID: Src.Int16Val &= 1; break;
      case Type::Int32TyID: Src.Int32Val &= 1; break;
      case Type::Int64TyID: Src.Int64Val &= 1; break;
      default:
        assert(0 && "Can't trunc a non-integer!");
        break;
    }
  } else if (opcode == Instruction::SExt && 
             SrcTy->getTypeID() == Type::BoolTyID) {
    // For sign extension from bool, we must extend the source bits.
    SrcTy = Type::Int64Ty;
    Src.Int64Val = 0 - Src.BoolVal;
  }

  switch (opcode) {
    case Instruction::Trunc:     // src integer, dest integral (can't be long)
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Bool , (bool));
      IMPLEMENT_CAST_CASE(Int8 , (uint8_t));
      IMPLEMENT_CAST_CASE(Int16, (uint16_t));
      IMPLEMENT_CAST_CASE(Int32, (uint32_t));
      IMPLEMENT_CAST_CASE(Int64, (uint64_t));
      IMPLEMENT_CAST_END
      break;
    case Instruction::ZExt:      // src integral (can't be long), dest integer
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Int8 , (uint8_t));
      IMPLEMENT_CAST_CASE(Int16, (uint16_t));
      IMPLEMENT_CAST_CASE(Int32, (uint32_t));
      IMPLEMENT_CAST_CASE(Int64, (uint64_t));
      IMPLEMENT_CAST_END
      break;
    case Instruction::SExt:      // src integral (can't be long), dest integer
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Int8 , (uint8_t)(int8_t));
      IMPLEMENT_CAST_CASE(Int16, (uint16_t)(int16_t));
      IMPLEMENT_CAST_CASE(Int32, (uint32_t)(int32_t));
      IMPLEMENT_CAST_CASE(Int64, (uint64_t)(int64_t));
      IMPLEMENT_CAST_END
      break;
    case Instruction::FPTrunc:   // src double, dest float
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Float  , (float));
      IMPLEMENT_CAST_END
      break;
    case Instruction::FPExt:     // src float, dest double
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Double , (double));
      IMPLEMENT_CAST_END
      break;
    case Instruction::UIToFP:    // src integral, dest floating
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Float  , (float)(uint64_t));
      IMPLEMENT_CAST_CASE(Double , (double)(uint64_t));
      IMPLEMENT_CAST_END
      break;
    case Instruction::SIToFP:    // src integeral, dest floating
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Float  , (float)(int64_t));
      IMPLEMENT_CAST_CASE(Double , (double)(int64_t));
      IMPLEMENT_CAST_END
      break;
    case Instruction::FPToUI:    // src floating, dest integral
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Bool , (bool));
      IMPLEMENT_CAST_CASE(Int8 , (uint8_t));
      IMPLEMENT_CAST_CASE(Int16, (uint16_t));
      IMPLEMENT_CAST_CASE(Int32, (uint32_t ));
      IMPLEMENT_CAST_CASE(Int64, (uint64_t));
      IMPLEMENT_CAST_END
      break;
    case Instruction::FPToSI:    // src floating, dest integral
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Bool , (bool));
      IMPLEMENT_CAST_CASE(Int8 , (uint8_t) (int8_t));
      IMPLEMENT_CAST_CASE(Int16, (uint16_t)(int16_t));
      IMPLEMENT_CAST_CASE(Int32, (uint32_t)(int32_t));
      IMPLEMENT_CAST_CASE(Int64, (uint64_t)(int64_t));
      IMPLEMENT_CAST_END
      break;
    case Instruction::PtrToInt:  // src pointer,  dest integral
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Bool , (bool));
      IMPLEMENT_CAST_CASE(Int8 , (uint8_t));
      IMPLEMENT_CAST_CASE(Int16, (uint16_t));
      IMPLEMENT_CAST_CASE(Int32, (uint32_t));
      IMPLEMENT_CAST_CASE(Int64, (uint64_t));
      IMPLEMENT_CAST_END
      break;
    case Instruction::IntToPtr:  // src integral, dest pointer
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Pointer, (PointerTy));
      IMPLEMENT_CAST_END
      break;
    case Instruction::BitCast:   // src any, dest any (same size)
      IMPLEMENT_CAST_START
      IMPLEMENT_CAST_CASE(Bool   , (bool));
      IMPLEMENT_CAST_CASE(Int8   , (uint8_t));
      IMPLEMENT_CAST_CASE(Int16  , (uint16_t));
      IMPLEMENT_CAST_CASE(Int32  , (uint32_t));
      IMPLEMENT_CAST_CASE(Int64  , (uint64_t));
      IMPLEMENT_CAST_CASE(Pointer, (PointerTy));
      IMPLEMENT_CAST_CASE(Float  , (float));
      IMPLEMENT_CAST_CASE(Double , (double));
      IMPLEMENT_CAST_END
      break;
    default:
      cerr << "Invalid cast opcode for cast instruction: " << opcode << "\n";
      abort();
  }
  return Dest;
}

void Interpreter::visitCastInst(CastInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeCastOperation(I.getOpcode(), I.getOperand(0), 
                                    I.getType(), SF), SF);
}

#define IMPLEMENT_VAARG(TY) \
   case Type::TY##TyID: Dest.TY##Val = Src.TY##Val; break

void Interpreter::visitVAArgInst(VAArgInst &I) {
  ExecutionContext &SF = ECStack.back();

  // Get the incoming valist parameter.  LLI treats the valist as a
  // (ec-stack-depth var-arg-index) pair.
  GenericValue VAList = getOperandValue(I.getOperand(0), SF);
  GenericValue Dest;
  GenericValue Src = ECStack[VAList.UIntPairVal.first]
   .VarArgs[VAList.UIntPairVal.second];
  const Type *Ty = I.getType();
  switch (Ty->getTypeID()) {
    IMPLEMENT_VAARG(Int8);
    IMPLEMENT_VAARG(Int16);
    IMPLEMENT_VAARG(Int32);
    IMPLEMENT_VAARG(Int64);
    IMPLEMENT_VAARG(Pointer);
    IMPLEMENT_VAARG(Float);
    IMPLEMENT_VAARG(Double);
    IMPLEMENT_VAARG(Bool);
  default:
    cerr << "Unhandled dest type for vaarg instruction: " << *Ty << "\n";
    abort();
  }

  // Set the Value of this Instruction.
  SetValue(&I, Dest, SF);

  // Move the pointer to the next vararg.
  ++VAList.UIntPairVal.second;
}

//===----------------------------------------------------------------------===//
//                        Dispatch and Execution Code
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// callFunction - Execute the specified function...
//
void Interpreter::callFunction(Function *F,
                               const std::vector<GenericValue> &ArgVals) {
  assert((ECStack.empty() || ECStack.back().Caller.getInstruction() == 0 ||
          ECStack.back().Caller.arg_size() == ArgVals.size()) &&
         "Incorrect number of arguments passed into function call!");
  // Make a new stack frame... and fill it in.
  ECStack.push_back(ExecutionContext());
  ExecutionContext &StackFrame = ECStack.back();
  StackFrame.CurFunction = F;

  // Special handling for external functions.
  if (F->isExternal()) {
    GenericValue Result = callExternalFunction (F, ArgVals);
    // Simulate a 'ret' instruction of the appropriate type.
    popStackAndReturnValueToCaller (F->getReturnType (), Result);
    return;
  }

  // Get pointers to first LLVM BB & Instruction in function.
  StackFrame.CurBB     = F->begin();
  StackFrame.CurInst   = StackFrame.CurBB->begin();

  // Run through the function arguments and initialize their values...
  assert((ArgVals.size() == F->arg_size() ||
         (ArgVals.size() > F->arg_size() && F->getFunctionType()->isVarArg()))&&
         "Invalid number of values passed to function invocation!");

  // Handle non-varargs arguments...
  unsigned i = 0;
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end(); AI != E; ++AI, ++i)
    SetValue(AI, ArgVals[i], StackFrame);

  // Handle varargs arguments...
  StackFrame.VarArgs.assign(ArgVals.begin()+i, ArgVals.end());
}

void Interpreter::run() {
  while (!ECStack.empty()) {
    // Interpret a single instruction & increment the "PC".
    ExecutionContext &SF = ECStack.back();  // Current stack frame
    Instruction &I = *SF.CurInst++;         // Increment before execute

    // Track the number of dynamic instructions executed.
    ++NumDynamicInsts;

    DOUT << "About to interpret: " << I;
    visit(I);   // Dispatch to one of the visit* methods...
  }
}
