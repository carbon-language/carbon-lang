//===-- ConstantFolding.cpp - Analyze constant folding possibilities ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions determines the possibility of performing constant
// folding.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include <cerrno>
#include <cmath>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Constant Folding internal helper functions
//===----------------------------------------------------------------------===//

/// IsConstantOffsetFromGlobal - If this constant is actually a constant offset
/// from a global, return the global and the constant.  Because of
/// constantexprs, this function is recursive.
static bool IsConstantOffsetFromGlobal(Constant *C, GlobalValue *&GV,
                                       int64_t &Offset, const TargetData &TD) {
  // Trivial case, constant is the global.
  if ((GV = dyn_cast<GlobalValue>(C))) {
    Offset = 0;
    return true;
  }
  
  // Otherwise, if this isn't a constant expr, bail out.
  ConstantExpr *CE = dyn_cast<ConstantExpr>(C);
  if (!CE) return false;
  
  // Look through ptr->int and ptr->ptr casts.
  if (CE->getOpcode() == Instruction::PtrToInt ||
      CE->getOpcode() == Instruction::BitCast)
    return IsConstantOffsetFromGlobal(CE->getOperand(0), GV, Offset, TD);
  
  // i32* getelementptr ([5 x i32]* @a, i32 0, i32 5)    
  if (CE->getOpcode() == Instruction::GetElementPtr) {
    // Cannot compute this if the element type of the pointer is missing size
    // info.
    if (!cast<PointerType>(CE->getOperand(0)->getType())->getElementType()->isSized())
      return false;
    
    // If the base isn't a global+constant, we aren't either.
    if (!IsConstantOffsetFromGlobal(CE->getOperand(0), GV, Offset, TD))
      return false;
    
    // Otherwise, add any offset that our operands provide.
    gep_type_iterator GTI = gep_type_begin(CE);
    for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i, ++GTI) {
      ConstantInt *CI = dyn_cast<ConstantInt>(CE->getOperand(i));
      if (!CI) return false;  // Index isn't a simple constant?
      if (CI->getZExtValue() == 0) continue;  // Not adding anything.
      
      if (const StructType *ST = dyn_cast<StructType>(*GTI)) {
        // N = N + Offset
        Offset += TD.getStructLayout(ST)->getElementOffset(CI->getZExtValue());
      } else {
        const SequentialType *SQT = cast<SequentialType>(*GTI);
        Offset += TD.getTypeSize(SQT->getElementType())*CI->getSExtValue();
      }
    }
    return true;
  }
  
  return false;
}


/// SymbolicallyEvaluateBinop - One of Op0/Op1 is a constant expression.
/// Attempt to symbolically evaluate the result of  a binary operator merging
/// these together.  If target data info is available, it is provided as TD, 
/// otherwise TD is null.
static Constant *SymbolicallyEvaluateBinop(unsigned Opc, Constant *Op0,
                                           Constant *Op1, const TargetData *TD){
  // SROA
  
  // Fold (and 0xffffffff00000000, (shl x, 32)) -> shl.
  // Fold (lshr (or X, Y), 32) -> (lshr [X/Y], 32) if one doesn't contribute
  // bits.
  
  
  // If the constant expr is something like &A[123] - &A[4].f, fold this into a
  // constant.  This happens frequently when iterating over a global array.
  if (Opc == Instruction::Sub && TD) {
    GlobalValue *GV1, *GV2;
    int64_t Offs1, Offs2;
    
    if (IsConstantOffsetFromGlobal(Op0, GV1, Offs1, *TD))
      if (IsConstantOffsetFromGlobal(Op1, GV2, Offs2, *TD) &&
          GV1 == GV2) {
        // (&GV+C1) - (&GV+C2) -> C1-C2, pointer arithmetic cannot overflow.
        return ConstantInt::get(Op0->getType(), Offs1-Offs2);
      }
  }
    
  // TODO: Fold icmp setne/seteq as well.
  return 0;
}

/// SymbolicallyEvaluateGEP - If we can symbolically evaluate the specified GEP
/// constant expression, do so.
static Constant *SymbolicallyEvaluateGEP(Constant** Ops, unsigned NumOps,
                                         const Type *ResultTy,
                                         const TargetData *TD) {
  Constant *Ptr = Ops[0];
  if (!cast<PointerType>(Ptr->getType())->getElementType()->isSized())
    return 0;
  
  if (TD && Ptr->isNullValue()) {
    // If this is a constant expr gep that is effectively computing an
    // "offsetof", fold it into 'cast int Size to T*' instead of 'gep 0, 0, 12'
    bool isFoldableGEP = true;
    for (unsigned i = 1; i != NumOps; ++i)
      if (!isa<ConstantInt>(Ops[i])) {
        isFoldableGEP = false;
        break;
      }
    if (isFoldableGEP) {
      uint64_t Offset = TD->getIndexedOffset(Ptr->getType(),
                                             (Value**)Ops+1, NumOps-1);
      Constant *C = ConstantInt::get(TD->getIntPtrType(), Offset);
      return ConstantExpr::getIntToPtr(C, ResultTy);
    }
  }
  
  return 0;
}


//===----------------------------------------------------------------------===//
// Constant Folding public APIs
//===----------------------------------------------------------------------===//


/// ConstantFoldInstruction - Attempt to constant fold the specified
/// instruction.  If successful, the constant result is returned, if not, null
/// is returned.  Note that this function can only fail when attempting to fold
/// instructions like loads and stores, which have no constant expression form.
///
Constant *llvm::ConstantFoldInstruction(Instruction *I, const TargetData *TD) {
  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    if (PN->getNumIncomingValues() == 0)
      return Constant::getNullValue(PN->getType());

    Constant *Result = dyn_cast<Constant>(PN->getIncomingValue(0));
    if (Result == 0) return 0;

    // Handle PHI nodes specially here...
    for (unsigned i = 1, e = PN->getNumIncomingValues(); i != e; ++i)
      if (PN->getIncomingValue(i) != Result && PN->getIncomingValue(i) != PN)
        return 0;   // Not all the same incoming constants...

    // If we reach here, all incoming values are the same constant.
    return Result;
  }

  // Scan the operand list, checking to see if they are all constants, if so,
  // hand off to ConstantFoldInstOperands.
  SmallVector<Constant*, 8> Ops;
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (Constant *Op = dyn_cast<Constant>(I->getOperand(i)))
      Ops.push_back(Op);
    else
      return 0;  // All operands not constant!

  return ConstantFoldInstOperands(I, &Ops[0], Ops.size(), TD);
}

/// ConstantFoldInstOperands - Attempt to constant fold an instruction with the
/// specified opcode and operands.  If successful, the constant result is
/// returned, if not, null is returned.  Note that this function can fail when
/// attempting to fold instructions like loads and stores, which have no
/// constant expression form.
///
Constant *llvm::ConstantFoldInstOperands(const Instruction* I, 
                                         Constant** Ops, unsigned NumOps,
                                         const TargetData *TD) {
  unsigned Opc = I->getOpcode();
  const Type *DestTy = I->getType();

  // Handle easy binops first.
  if (isa<BinaryOperator>(I)) {
    if (isa<ConstantExpr>(Ops[0]) || isa<ConstantExpr>(Ops[1]))
      if (Constant *C = SymbolicallyEvaluateBinop(I->getOpcode(), Ops[0],
                                                  Ops[1], TD))
        return C;
    
    return ConstantExpr::get(Opc, Ops[0], Ops[1]);
  }
  
  switch (Opc) {
  default: return 0;
  case Instruction::Call:
    if (Function *F = dyn_cast<Function>(Ops[0]))
      if (canConstantFoldCallTo(F))
        return ConstantFoldCall(F, Ops+1, NumOps-1);
    return 0;
  case Instruction::ICmp:
  case Instruction::FCmp:
    return ConstantExpr::getCompare(cast<CmpInst>(I)->getPredicate(), Ops[0], 
                                    Ops[1]);
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
    return ConstantExpr::getCast(Opc, Ops[0], DestTy);
  case Instruction::Select:
    return ConstantExpr::getSelect(Ops[0], Ops[1], Ops[2]);
  case Instruction::ExtractElement:
    return ConstantExpr::getExtractElement(Ops[0], Ops[1]);
  case Instruction::InsertElement:
    return ConstantExpr::getInsertElement(Ops[0], Ops[1], Ops[2]);
  case Instruction::ShuffleVector:
    return ConstantExpr::getShuffleVector(Ops[0], Ops[1], Ops[2]);
  case Instruction::GetElementPtr:
    if (Constant *C = SymbolicallyEvaluateGEP(Ops, NumOps, I->getType(), TD))
      return C;
    
    return ConstantExpr::getGetElementPtr(Ops[0], Ops+1, NumOps-1);
  }
}

/// ConstantFoldLoadThroughGEPConstantExpr - Given a constant and a
/// getelementptr constantexpr, return the constant value being addressed by the
/// constant expression, or null if something is funny and we can't decide.
Constant *llvm::ConstantFoldLoadThroughGEPConstantExpr(Constant *C, 
                                                       ConstantExpr *CE) {
  if (CE->getOperand(1) != Constant::getNullValue(CE->getOperand(1)->getType()))
    return 0;  // Do not allow stepping over the value!
  
  // Loop over all of the operands, tracking down which value we are
  // addressing...
  gep_type_iterator I = gep_type_begin(CE), E = gep_type_end(CE);
  for (++I; I != E; ++I)
    if (const StructType *STy = dyn_cast<StructType>(*I)) {
      ConstantInt *CU = cast<ConstantInt>(I.getOperand());
      assert(CU->getZExtValue() < STy->getNumElements() &&
             "Struct index out of range!");
      unsigned El = (unsigned)CU->getZExtValue();
      if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
        C = CS->getOperand(El);
      } else if (isa<ConstantAggregateZero>(C)) {
        C = Constant::getNullValue(STy->getElementType(El));
      } else if (isa<UndefValue>(C)) {
        C = UndefValue::get(STy->getElementType(El));
      } else {
        return 0;
      }
    } else if (ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand())) {
      if (const ArrayType *ATy = dyn_cast<ArrayType>(*I)) {
        if (CI->getZExtValue() >= ATy->getNumElements())
         return 0;
        if (ConstantArray *CA = dyn_cast<ConstantArray>(C))
          C = CA->getOperand(CI->getZExtValue());
        else if (isa<ConstantAggregateZero>(C))
          C = Constant::getNullValue(ATy->getElementType());
        else if (isa<UndefValue>(C))
          C = UndefValue::get(ATy->getElementType());
        else
          return 0;
      } else if (const VectorType *PTy = dyn_cast<VectorType>(*I)) {
        if (CI->getZExtValue() >= PTy->getNumElements())
          return 0;
        if (ConstantVector *CP = dyn_cast<ConstantVector>(C))
          C = CP->getOperand(CI->getZExtValue());
        else if (isa<ConstantAggregateZero>(C))
          C = Constant::getNullValue(PTy->getElementType());
        else if (isa<UndefValue>(C))
          C = UndefValue::get(PTy->getElementType());
        else
          return 0;
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  return C;
}


//===----------------------------------------------------------------------===//
//  Constant Folding for Calls
//

/// canConstantFoldCallTo - Return true if its even possible to fold a call to
/// the specified function.
bool
llvm::canConstantFoldCallTo(Function *F) {
  const std::string &Name = F->getName();

  switch (F->getIntrinsicID()) {
  case Intrinsic::sqrt_f32:
  case Intrinsic::sqrt_f64:
  case Intrinsic::powi_f32:
  case Intrinsic::powi_f64:
  case Intrinsic::bswap:
  case Intrinsic::ctpop:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
    return true;
  default: break;
  }

  switch (Name[0])
  {
    case 'a':
      return Name == "acos" || Name == "asin" || Name == "atan" ||
             Name == "atan2";
    case 'c':
      return Name == "ceil" || Name == "cos" || Name == "cosf" ||
             Name == "cosh";
    case 'e':
      return Name == "exp";
    case 'f':
      return Name == "fabs" || Name == "fmod" || Name == "floor";
    case 'l':
      return Name == "log" || Name == "log10";
    case 'p':
      return Name == "pow";
    case 's':
      return Name == "sin" || Name == "sinh" || 
             Name == "sqrt" || Name == "sqrtf";
    case 't':
      return Name == "tan" || Name == "tanh";
    default:
      return false;
  }
}

static Constant *ConstantFoldFP(double (*NativeFP)(double), double V, 
                                const Type *Ty) {
  errno = 0;
  V = NativeFP(V);
  if (errno == 0)
    return ConstantFP::get(Ty, V);
  errno = 0;
  return 0;
}

/// ConstantFoldCall - Attempt to constant fold a call to the specified function
/// with the specified arguments, returning null if unsuccessful.
Constant *
llvm::ConstantFoldCall(Function *F, Constant** Operands, unsigned NumOperands) {
  const std::string &Name = F->getName();
  const Type *Ty = F->getReturnType();

  if (NumOperands == 1) {
    if (ConstantFP *Op = dyn_cast<ConstantFP>(Operands[0])) {
      double V = Op->getValue();
      switch (Name[0])
      {
        case 'a':
          if (Name == "acos")
            return ConstantFoldFP(acos, V, Ty);
          else if (Name == "asin")
            return ConstantFoldFP(asin, V, Ty);
          else if (Name == "atan")
            return ConstantFP::get(Ty, atan(V));
          break;
        case 'c':
          if (Name == "ceil")
            return ConstantFoldFP(ceil, V, Ty);
          else if (Name == "cos")
            return ConstantFP::get(Ty, cos(V));
          else if (Name == "cosh")
            return ConstantFP::get(Ty, cosh(V));
          break;
        case 'e':
          if (Name == "exp")
            return ConstantFP::get(Ty, exp(V));
          break;
        case 'f':
          if (Name == "fabs")
            return ConstantFP::get(Ty, fabs(V));
          else if (Name == "floor")
            return ConstantFoldFP(floor, V, Ty);
          break;
        case 'l':
          if (Name == "log" && V > 0)
            return ConstantFP::get(Ty, log(V));
          else if (Name == "log10" && V > 0)
            return ConstantFoldFP(log10, V, Ty);
          else if (Name == "llvm.sqrt.f32" || Name == "llvm.sqrt.f64") {
            if (V >= -0.0)
              return ConstantFP::get(Ty, sqrt(V));
            else // Undefined
              return ConstantFP::get(Ty, 0.0);
          }
          break;
        case 's':
          if (Name == "sin")
            return ConstantFP::get(Ty, sin(V));
          else if (Name == "sinh")
            return ConstantFP::get(Ty, sinh(V));
          else if (Name == "sqrt" && V >= 0)
            return ConstantFP::get(Ty, sqrt(V));
          else if (Name == "sqrtf" && V >= 0)
            return ConstantFP::get(Ty, sqrt((float)V));
          break;
        case 't':
          if (Name == "tan")
            return ConstantFP::get(Ty, tan(V));
          else if (Name == "tanh")
            return ConstantFP::get(Ty, tanh(V));
          break;
        default:
          break;
      }
    } else if (ConstantInt *Op = dyn_cast<ConstantInt>(Operands[0])) {
      if (Name.size() > 11 && !memcmp(&Name[0], "llvm.bswap", 10)) {
        return ConstantInt::get(Op->getValue().byteSwap());
      } else if (Name.size() > 11 && !memcmp(&Name[0],"llvm.ctpop",10)) {
        uint64_t ctpop = Op->getValue().countPopulation();
        return ConstantInt::get(Type::Int32Ty, ctpop);
      } else if (Name.size() > 10 && !memcmp(&Name[0], "llvm.cttz", 9)) {
        uint64_t cttz = Op->getValue().countTrailingZeros();
        return ConstantInt::get(Type::Int32Ty, cttz);
      } else if (Name.size() > 10 && !memcmp(&Name[0], "llvm.ctlz", 9)) {
        uint64_t ctlz = Op->getValue().countLeadingZeros();
        return ConstantInt::get(Type::Int32Ty, ctlz);
      }
    }
  } else if (NumOperands == 2) {
    if (ConstantFP *Op1 = dyn_cast<ConstantFP>(Operands[0])) {
      double Op1V = Op1->getValue();
      if (ConstantFP *Op2 = dyn_cast<ConstantFP>(Operands[1])) {
        double Op2V = Op2->getValue();

        if (Name == "pow") {
          errno = 0;
          double V = pow(Op1V, Op2V);
          if (errno == 0)
            return ConstantFP::get(Ty, V);
        } else if (Name == "fmod") {
          errno = 0;
          double V = fmod(Op1V, Op2V);
          if (errno == 0)
            return ConstantFP::get(Ty, V);
        } else if (Name == "atan2") {
          return ConstantFP::get(Ty, atan2(Op1V,Op2V));
        }
      } else if (ConstantInt *Op2C = dyn_cast<ConstantInt>(Operands[1])) {
        if (Name == "llvm.powi.f32") {
          return ConstantFP::get(Ty, std::pow((float)Op1V,
                                              (int)Op2C->getZExtValue()));
        } else if (Name == "llvm.powi.f64") {
          return ConstantFP::get(Ty, std::pow((double)Op1V,
                                              (int)Op2C->getZExtValue()));
        }
      }
    }
  }
  return 0;
}

