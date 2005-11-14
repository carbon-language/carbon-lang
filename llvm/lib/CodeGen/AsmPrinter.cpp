//===-- AsmPrinter.cpp - Common AsmPrinter code ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AsmPrinter class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

bool AsmPrinter::doInitialization(Module &M) {
  Mang = new Mangler(M, GlobalPrefix);
  return false;
}

bool AsmPrinter::doFinalization(Module &M) {
  delete Mang; Mang = 0;
  return false;
}

void AsmPrinter::setupMachineFunction(MachineFunction &MF) {
  // What's my mangled name?
  CurrentFnName = Mang->getValueName(MF.getFunction());
}

// emitAlignment - Emit an alignment directive to the specified power of two.
void AsmPrinter::emitAlignment(unsigned NumBits, const GlobalValue *GV) const {
  if (GV && GV->getAlignment())
    NumBits = Log2_32(GV->getAlignment());
  if (NumBits == 0) return;   // No need to emit alignment.
  if (AlignmentIsInBytes) NumBits = 1 << NumBits;
  O << AlignDirective << NumBits << "\n";
}

/// emitZeros - Emit a block of zeros.
///
void AsmPrinter::emitZeros(uint64_t NumZeros) const {
  if (NumZeros) {
    if (ZeroDirective)
      O << ZeroDirective << NumZeros << "\n";
    else {
      for (; NumZeros; --NumZeros)
        O << Data8bitsDirective << "0\n";
    }
  }
}

// Print out the specified constant, without a storage class.  Only the
// constants valid in constant expressions can occur here.
void AsmPrinter::emitConstantValueOnly(const Constant *CV) {
  if (CV->isNullValue() || isa<UndefValue>(CV))
    O << "0";
  else if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
    assert(CB == ConstantBool::True);
    O << "1";
  } else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV))
    if (((CI->getValue() << 32) >> 32) == CI->getValue())
      O << CI->getValue();
    else
      O << (uint64_t)CI->getValue();
  else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV))
    O << CI->getValue();
  else if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV)) {
    // This is a constant address for a global variable or function. Use the
    // name of the variable or function as the address value, possibly
    // decorating it with GlobalVarAddrPrefix/Suffix or
    // FunctionAddrPrefix/Suffix (these all default to "" )
    if (isa<Function>(GV))
      O << FunctionAddrPrefix << Mang->getValueName(GV) << FunctionAddrSuffix;
    else
      O << GlobalVarAddrPrefix << Mang->getValueName(GV) << GlobalVarAddrSuffix;
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    const TargetData &TD = TM.getTargetData();
    switch(CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      // generate a symbolic expression for the byte address
      const Constant *ptrVal = CE->getOperand(0);
      std::vector<Value*> idxVec(CE->op_begin()+1, CE->op_end());
      if (int64_t Offset = TD.getIndexedOffset(ptrVal->getType(), idxVec)) {
        if (Offset)
          O << "(";
        emitConstantValueOnly(ptrVal);
        if (Offset > 0)
          O << ") + " << Offset;
        else if (Offset < 0)
          O << ") - " << -Offset;
      } else {
        emitConstantValueOnly(ptrVal);
      }
      break;
    }
    case Instruction::Cast: {
      // Support only non-converting or widening casts for now, that is, ones
      // that do not involve a change in value.  This assertion is really gross,
      // and may not even be a complete check.
      Constant *Op = CE->getOperand(0);
      const Type *OpTy = Op->getType(), *Ty = CE->getType();

      // Remember, kids, pointers can be losslessly converted back and forth
      // into 32-bit or wider integers, regardless of signedness. :-P
      assert(((isa<PointerType>(OpTy)
               && (Ty == Type::LongTy || Ty == Type::ULongTy
                   || Ty == Type::IntTy || Ty == Type::UIntTy))
              || (isa<PointerType>(Ty)
                  && (OpTy == Type::LongTy || OpTy == Type::ULongTy
                      || OpTy == Type::IntTy || OpTy == Type::UIntTy))
              || (((TD.getTypeSize(Ty) >= TD.getTypeSize(OpTy))
                   && OpTy->isLosslesslyConvertibleTo(Ty))))
             && "FIXME: Don't yet support this kind of constant cast expr");
      O << "(";
      emitConstantValueOnly(Op);
      O << ")";
      break;
    }
    case Instruction::Add:
      O << "(";
      emitConstantValueOnly(CE->getOperand(0));
      O << ") + (";
      emitConstantValueOnly(CE->getOperand(1));
      O << ")";
      break;
    default:
      assert(0 && "Unsupported operator!");
    }
  } else {
    assert(0 && "Unknown constant value!");
  }
}

/// toOctal - Convert the low order bits of X into an octal digit.
///
static inline char toOctal(int X) {
  return (X&7)+'0';
}

/// printAsCString - Print the specified array as a C compatible string, only if
/// the predicate isString is true.
///
static void printAsCString(std::ostream &O, const ConstantArray *CVA,
                           unsigned LastElt) {
  assert(CVA->isString() && "Array is not string compatible!");

  O << "\"";
  for (unsigned i = 0; i != LastElt; ++i) {
    unsigned char C =
        (unsigned char)cast<ConstantInt>(CVA->getOperand(i))->getRawValue();

    if (C == '"') {
      O << "\\\"";
    } else if (C == '\\') {
      O << "\\\\";
    } else if (isprint(C)) {
      O << C;
    } else {
      switch(C) {
      case '\b': O << "\\b"; break;
      case '\f': O << "\\f"; break;
      case '\n': O << "\\n"; break;
      case '\r': O << "\\r"; break;
      case '\t': O << "\\t"; break;
      default:
        O << '\\';
        O << toOctal(C >> 6);
        O << toOctal(C >> 3);
        O << toOctal(C >> 0);
        break;
      }
    }
  }
  O << "\"";
}

/// emitGlobalConstant - Print a general LLVM constant to the .s file.
///
void AsmPrinter::emitGlobalConstant(const Constant *CV) {
  const TargetData &TD = TM.getTargetData();

  if (CV->isNullValue() || isa<UndefValue>(CV)) {
    emitZeros(TD.getTypeSize(CV->getType()));
    return;
  } else if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    if (CVA->isString()) {
      unsigned NumElts = CVA->getNumOperands();
      if (AscizDirective && NumElts && 
          cast<ConstantInt>(CVA->getOperand(NumElts-1))->getRawValue() == 0) {
        O << AscizDirective;
        printAsCString(O, CVA, NumElts-1);
      } else {
        O << AsciiDirective;
        printAsCString(O, CVA, NumElts);
      }
      O << "\n";
    } else { // Not a string.  Print the values in successive locations
      for (unsigned i = 0, e = CVA->getNumOperands(); i != e; ++i)
        emitGlobalConstant(CVA->getOperand(i));
    }
    return;
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    // Print the fields in successive locations. Pad to align if needed!
    const StructLayout *cvsLayout = TD.getStructLayout(CVS->getType());
    uint64_t sizeSoFar = 0;
    for (unsigned i = 0, e = CVS->getNumOperands(); i != e; ++i) {
      const Constant* field = CVS->getOperand(i);

      // Check if padding is needed and insert one or more 0s.
      uint64_t fieldSize = TD.getTypeSize(field->getType());
      uint64_t padSize = ((i == e-1? cvsLayout->StructSize
                           : cvsLayout->MemberOffsets[i+1])
                          - cvsLayout->MemberOffsets[i]) - fieldSize;
      sizeSoFar += fieldSize + padSize;

      // Now print the actual field value
      emitGlobalConstant(field);

      // Insert the field padding unless it's zero bytes...
      emitZeros(padSize);
    }
    assert(sizeSoFar == cvsLayout->StructSize &&
           "Layout of constant struct may be incorrect!");
    return;
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    // FP Constants are printed as integer constants to avoid losing
    // precision...
    double Val = CFP->getValue();
    if (CFP->getType() == Type::DoubleTy) {
      if (Data64bitsDirective)
        O << Data64bitsDirective << DoubleToBits(Val) << "\t" << CommentString
          << " double value: " << Val << "\n";
      else if (TD.isBigEndian()) {
        O << Data32bitsDirective << unsigned(DoubleToBits(Val) >> 32)
          << "\t" << CommentString << " double most significant word "
          << Val << "\n";
        O << Data32bitsDirective << unsigned(DoubleToBits(Val))
          << "\t" << CommentString << " double least significant word "
          << Val << "\n";
      } else {
        O << Data32bitsDirective << unsigned(DoubleToBits(Val))
          << "\t" << CommentString << " double least significant word " << Val
          << "\n";
        O << Data32bitsDirective << unsigned(DoubleToBits(Val) >> 32)
          << "\t" << CommentString << " double most significant word " << Val
          << "\n";
      }
      return;
    } else {
      O << Data32bitsDirective << FloatToBits(Val) << "\t" << CommentString
        << " float " << Val << "\n";
      return;
    }
  } else if (CV->getType() == Type::ULongTy || CV->getType() == Type::LongTy) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
      uint64_t Val = CI->getRawValue();

      if (Data64bitsDirective)
        O << Data64bitsDirective << Val << "\n";
      else if (TD.isBigEndian()) {
        O << Data32bitsDirective << unsigned(Val >> 32)
          << "\t" << CommentString << " Double-word most significant word "
          << Val << "\n";
        O << Data32bitsDirective << unsigned(Val)
          << "\t" << CommentString << " Double-word least significant word "
          << Val << "\n";
      } else {
        O << Data32bitsDirective << unsigned(Val)
          << "\t" << CommentString << " Double-word least significant word "
          << Val << "\n";
        O << Data32bitsDirective << unsigned(Val >> 32)
          << "\t" << CommentString << " Double-word most significant word "
          << Val << "\n";
      }
      return;
    }
  }

  const Type *type = CV->getType();
  switch (type->getTypeID()) {
  case Type::BoolTyID:
  case Type::UByteTyID: case Type::SByteTyID:
    O << Data8bitsDirective;
    break;
  case Type::UShortTyID: case Type::ShortTyID:
    O << Data16bitsDirective;
    break;
  case Type::PointerTyID:
    if (TD.getPointerSize() == 8) {
      O << Data64bitsDirective;
      break;
    }
    //Fall through for pointer size == int size
  case Type::UIntTyID: case Type::IntTyID:
    O << Data32bitsDirective;
    break;
  case Type::ULongTyID: case Type::LongTyID:
    assert(Data64bitsDirective &&"Target cannot handle 64-bit constant exprs!");
    O << Data64bitsDirective;
    break;
  case Type::FloatTyID: case Type::DoubleTyID:
    assert (0 && "Should have already output floating point constant.");
  default:
    assert (0 && "Can't handle printing this type of thing");
    break;
  }
  emitConstantValueOnly(CV);
  O << "\n";
}
