//===-- ConstantWriter.cpp - Functions for writing constants --------------===//
//
// This file implements the routines for encoding constants to a bytecode 
// stream.
//
//===----------------------------------------------------------------------===//

#include "WriterInternals.h"
#include "llvm/Constants.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"

void BytecodeWriter::outputType(const Type *T) {
  output_vbr((unsigned)T->getPrimitiveID(), Out);
  
  // That's all there is to handling primitive types...
  if (T->isPrimitiveType())
    return;     // We might do this if we alias a prim type: %x = type int
  
  switch (T->getPrimitiveID()) {   // Handle derived types now.
  case Type::FunctionTyID: {
    const FunctionType *MT = cast<FunctionType>(T);
    int Slot = Table.getValSlot(MT->getReturnType());
    assert(Slot != -1 && "Type used but not available!!");
    output_vbr((unsigned)Slot, Out);

    // Output the number of arguments to method (+1 if varargs):
    output_vbr(MT->getParamTypes().size()+MT->isVarArg(), Out);

    // Output all of the arguments...
    FunctionType::ParamTypes::const_iterator I = MT->getParamTypes().begin();
    for (; I != MT->getParamTypes().end(); ++I) {
      Slot = Table.getValSlot(*I);
      assert(Slot != -1 && "Type used but not available!!");
      output_vbr((unsigned)Slot, Out);
    }

    // Terminate list with VoidTy if we are a varargs function...
    if (MT->isVarArg())
      output_vbr((unsigned)Type::VoidTy->getPrimitiveID(), Out);
    break;
  }

  case Type::ArrayTyID: {
    const ArrayType *AT = cast<ArrayType>(T);
    int Slot = Table.getValSlot(AT->getElementType());
    assert(Slot != -1 && "Type used but not available!!");
    output_vbr((unsigned)Slot, Out);
    //std::cerr << "Type slot = " << Slot << " Type = " << T->getName() << endl;

    output_vbr(AT->getNumElements(), Out);
    break;
  }

  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(T);

    // Output all of the element types...
    StructType::ElementTypes::const_iterator I = ST->getElementTypes().begin();
    for (; I != ST->getElementTypes().end(); ++I) {
      int Slot = Table.getValSlot(*I);
      assert(Slot != -1 && "Type used but not available!!");
      output_vbr((unsigned)Slot, Out);
    }

    // Terminate list with VoidTy
    output_vbr((unsigned)Type::VoidTy->getPrimitiveID(), Out);
    break;
  }

  case Type::PointerTyID: {
    const PointerType *PT = cast<PointerType>(T);
    int Slot = Table.getValSlot(PT->getElementType());
    assert(Slot != -1 && "Type used but not available!!");
    output_vbr((unsigned)Slot, Out);
    break;
  }

  case Type::OpaqueTyID: {
    // No need to emit anything, just the count of opaque types is enough.
    break;
  }

  //case Type::PackedTyID:
  default:
    std::cerr << __FILE__ << ":" << __LINE__ << ": Don't know how to serialize"
              << " Type '" << T->getDescription() << "'\n";
    break;
  }
}

bool BytecodeWriter::outputConstant(const Constant *CPV) {
  assert((CPV->getType()->isPrimitiveType() || !CPV->isNullValue()) &&
         "Shouldn't output null constants!");

  // We must check for a ConstantExpr before switching by type because
  // a ConstantExpr can be of any type, and has no explicit value.
  // 
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CPV)) {
    // FIXME: Encoding of constant exprs could be much more compact!
    assert(CE->getNumOperands() > 0 && "ConstantExpr with 0 operands");
    output_vbr(CE->getNumOperands(), Out);   // flags as an expr
    output_vbr(CE->getOpcode(), Out);        // flags as an expr
    
    for (User::const_op_iterator OI = CE->op_begin(); OI != CE->op_end(); ++OI){
      int Slot = Table.getValSlot(*OI);
      assert(Slot != -1 && "Unknown constant used in ConstantExpr!!");
      output_vbr((unsigned)Slot, Out);
      Slot = Table.getValSlot((*OI)->getType());
      output_vbr((unsigned)Slot, Out);
    }
    return false;
  } else {
    output_vbr((unsigned)0, Out);       // flag as not a ConstantExpr
  }
  
  switch (CPV->getType()->getPrimitiveID()) {
  case Type::BoolTyID:    // Boolean Types
    if (cast<ConstantBool>(CPV)->getValue())
      output_vbr(1U, Out);
    else
      output_vbr(0U, Out);
    break;

  case Type::UByteTyID:   // Unsigned integer types...
  case Type::UShortTyID:
  case Type::UIntTyID:
  case Type::ULongTyID:
    output_vbr(cast<ConstantUInt>(CPV)->getValue(), Out);
    break;

  case Type::SByteTyID:   // Signed integer types...
  case Type::ShortTyID:
  case Type::IntTyID:
  case Type::LongTyID:
    output_vbr(cast<ConstantSInt>(CPV)->getValue(), Out);
    break;

  case Type::TypeTyID:     // Serialize type type
    assert(0 && "Types should not be in the Constant!");
    break;

  case Type::ArrayTyID: {
    const ConstantArray *CPA = cast<ConstantArray>(CPV);
    unsigned size = CPA->getValues().size();
    assert(size == cast<ArrayType>(CPA->getType())->getNumElements()
           && "ConstantArray out of whack!");
    for (unsigned i = 0; i < size; i++) {
      int Slot = Table.getValSlot(CPA->getOperand(i));
      assert(Slot != -1 && "Constant used but not available!!");
      output_vbr((unsigned)Slot, Out);
    }
    break;
  }

  case Type::StructTyID: {
    const ConstantStruct *CPS = cast<ConstantStruct>(CPV);
    const std::vector<Use> &Vals = CPS->getValues();

    for (unsigned i = 0; i < Vals.size(); ++i) {
      int Slot = Table.getValSlot(Vals[i]);
      assert(Slot != -1 && "Constant used but not available!!");
      output_vbr((unsigned)Slot, Out);
    }      
    break;
  }

  case Type::PointerTyID: {
    const ConstantPointer *CPP = cast<ConstantPointer>(CPV);
    assert(!isa<ConstantPointerNull>(CPP) && "Null should be already emitted!");
    const ConstantPointerRef *CPR = cast<ConstantPointerRef>(CPP);
    int Slot = Table.getValSlot((Value*)CPR->getValue());
    assert(Slot != -1 && "Global used but not available!!");
    output_vbr((unsigned)Slot, Out);
    break;
  }

  case Type::FloatTyID: {   // Floating point types...
    float Tmp = (float)cast<ConstantFP>(CPV)->getValue();
    output_data(&Tmp, &Tmp+1, Out);
    break;
  }
  case Type::DoubleTyID: {
    double Tmp = cast<ConstantFP>(CPV)->getValue();
    output_data(&Tmp, &Tmp+1, Out);
    break;
  }

  case Type::VoidTyID: 
  case Type::LabelTyID:
  default:
    std::cerr << __FILE__ << ":" << __LINE__ << ": Don't know how to serialize"
              << " type '" << CPV->getType()->getName() << "'\n";
    break;
  }
  return false;
}
