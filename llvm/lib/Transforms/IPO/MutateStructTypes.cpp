//===- MutateStructTypes.cpp - Change struct defns ------------------------===//
//
// This pass is used to change structure accesses and type definitions in some
// way.  It can be used to arbitrarily permute structure fields, safely, without
// breaking code.  A transformation may only be done on a type if that type has
// been found to be "safe" by the 'FindUnsafePointerTypes' pass.  This pass will
// assert and die if you try to do an illegal transformation.
//
// This is an interprocedural pass that requires the entire program to do a
// transformation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/MutateStructTypes.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"
#include "Support/STLExtras.h"
#include "Support/Debug.h"
#include <algorithm>

// ValuePlaceHolder - A stupid little marker value.  It appears as an
// instruction of type Instruction::UserOp1.
//
struct ValuePlaceHolder : public Instruction {
  ValuePlaceHolder(const Type *Ty) : Instruction(Ty, UserOp1, "") {}

  virtual Instruction *clone() const { abort(); return 0; }
  virtual const char *getOpcodeName() const { return "placeholder"; }
};


// ConvertType - Convert from the old type system to the new one...
const Type *MutateStructTypes::ConvertType(const Type *Ty) {
  if (Ty->isPrimitiveType() ||
      isa<OpaqueType>(Ty)) return Ty;  // Don't convert primitives

  std::map<const Type *, PATypeHolder>::iterator I = TypeMap.find(Ty);
  if (I != TypeMap.end()) return I->second;

  const Type *DestTy = 0;

  PATypeHolder PlaceHolder = OpaqueType::get();
  TypeMap.insert(std::make_pair(Ty, PlaceHolder.get()));

  switch (Ty->getPrimitiveID()) {
  case Type::FunctionTyID: {
    const FunctionType *MT = cast<FunctionType>(Ty);
    const Type *RetTy = ConvertType(MT->getReturnType());
    std::vector<const Type*> ArgTypes;

    for (FunctionType::ParamTypes::const_iterator I = MT->getParamTypes().begin(),
           E = MT->getParamTypes().end(); I != E; ++I)
      ArgTypes.push_back(ConvertType(*I));
    
    DestTy = FunctionType::get(RetTy, ArgTypes, MT->isVarArg());
    break;
  }
  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);
    const StructType::ElementTypes &El = ST->getElementTypes();
    std::vector<const Type *> Types;

    for (StructType::ElementTypes::const_iterator I = El.begin(), E = El.end();
         I != E; ++I)
      Types.push_back(ConvertType(*I));
    DestTy = StructType::get(Types);
    break;
  }
  case Type::ArrayTyID:
    DestTy = ArrayType::get(ConvertType(cast<ArrayType>(Ty)->getElementType()),
                            cast<ArrayType>(Ty)->getNumElements());
    break;

  case Type::PointerTyID:
    DestTy = PointerType::get(
                 ConvertType(cast<PointerType>(Ty)->getElementType()));
    break;
  default:
    assert(0 && "Unknown type!");
    return 0;
  }

  assert(DestTy && "Type didn't get created!?!?");

  // Refine our little placeholder value into a real type...
  ((DerivedType*)PlaceHolder.get())->refineAbstractTypeTo(DestTy);
  TypeMap.insert(std::make_pair(Ty, PlaceHolder.get()));

  return PlaceHolder.get();
}


// AdjustIndices - Convert the indices specified by Idx to the new changed form
// using the specified OldTy as the base type being indexed into.
//
void MutateStructTypes::AdjustIndices(const CompositeType *OldTy,
                                      std::vector<Value*> &Idx,
                                      unsigned i) {
  assert(i < Idx.size() && "i out of range!");
  const CompositeType *NewCT = cast<CompositeType>(ConvertType(OldTy));
  if (NewCT == OldTy) return;  // No adjustment unless type changes

  if (const StructType *OldST = dyn_cast<StructType>(OldTy)) {
    // Figure out what the current index is...
    unsigned ElNum = cast<ConstantUInt>(Idx[i])->getValue();
    assert(ElNum < OldST->getElementTypes().size());

    std::map<const StructType*, TransformType>::iterator
      I = Transforms.find(OldST);
    if (I != Transforms.end()) {
      assert(ElNum < I->second.second.size());
      // Apply the XForm specified by Transforms map...
      unsigned NewElNum = I->second.second[ElNum];
      Idx[i] = ConstantUInt::get(Type::UByteTy, NewElNum);
    }
  }

  // Recursively process subtypes...
  if (i+1 < Idx.size())
    AdjustIndices(cast<CompositeType>(OldTy->getTypeAtIndex(Idx[i])), Idx, i+1);
}


// ConvertValue - Convert from the old value in the old type system to the new
// type system.
//
Value *MutateStructTypes::ConvertValue(const Value *V) {
  // Ignore null values and simple constants..
  if (V == 0) return 0;

  if (const Constant *CPV = dyn_cast<Constant>(V)) {
    if (V->getType()->isPrimitiveType())
      return (Value*)CPV;

    if (isa<ConstantPointerNull>(CPV))
      return ConstantPointerNull::get(
                      cast<PointerType>(ConvertType(V->getType())));
    assert(0 && "Unable to convert constpool val of this type!");
  }

  // Check to see if this is an out of function reference first...
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    // Check to see if the value is in the map...
    std::map<const GlobalValue*, GlobalValue*>::iterator I = GlobalMap.find(GV);
    if (I == GlobalMap.end())
      return (Value*)GV;  // Not mapped, just return value itself
    return I->second;
  }
  
  std::map<const Value*, Value*>::iterator I = LocalValueMap.find(V);
  if (I != LocalValueMap.end()) return I->second;

  if (const BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
    // Create placeholder block to represent the basic block we haven't seen yet
    // This will be used when the block gets created.
    //
    return LocalValueMap[V] = new BasicBlock(BB->getName());
  }

  DEBUG(std::cerr << "NPH: " << V << "\n");

  // Otherwise make a constant to represent it
  return LocalValueMap[V] = new ValuePlaceHolder(ConvertType(V->getType()));
}


// setTransforms - Take a map that specifies what transformation to do for each
// field of the specified structure types.  There is one element of the vector
// for each field of the structure.  The value specified indicates which slot of
// the destination structure the field should end up in.  A negative value 
// indicates that the field should be deleted entirely.
//
void MutateStructTypes::setTransforms(const TransformsType &XForm) {

  // Loop over the types and insert dummy entries into the type map so that 
  // recursive types are resolved properly...
  for (std::map<const StructType*, std::vector<int> >::const_iterator
         I = XForm.begin(), E = XForm.end(); I != E; ++I) {
    const StructType *OldTy = I->first;
    TypeMap.insert(std::make_pair(OldTy, OpaqueType::get()));
  }

  // Loop over the type specified and figure out what types they should become
  for (std::map<const StructType*, std::vector<int> >::const_iterator
         I = XForm.begin(), E = XForm.end(); I != E; ++I) {
    const StructType  *OldTy = I->first;
    const std::vector<int> &InVec = I->second;

    assert(OldTy->getElementTypes().size() == InVec.size() &&
           "Action not specified for every element of structure type!");

    std::vector<const Type *> NewType;

    // Convert the elements of the type over, including the new position mapping
    int Idx = 0;
    std::vector<int>::const_iterator TI = find(InVec.begin(), InVec.end(), Idx);
    while (TI != InVec.end()) {
      unsigned Offset = TI-InVec.begin();
      const Type *NewEl = ConvertType(OldTy->getContainedType(Offset));
      assert(NewEl && "Element not found!");
      NewType.push_back(NewEl);

      TI = find(InVec.begin(), InVec.end(), ++Idx);
    }

    // Create a new type that corresponds to the destination type
    PATypeHolder NSTy = StructType::get(NewType);

    // Refine the old opaque type to the new type to properly handle recursive
    // types...
    //
    const Type *OldTypeStub = TypeMap.find(OldTy)->second.get();
    ((DerivedType*)OldTypeStub)->refineAbstractTypeTo(NSTy);

    // Add the transformation to the Transforms map.
    Transforms.insert(std::make_pair(OldTy,
                       std::make_pair(cast<StructType>(NSTy.get()), InVec)));

    DEBUG(std::cerr << "Mutate " << OldTy << "\nTo " << NSTy << "\n");
  }
}

void MutateStructTypes::clearTransforms() {
  Transforms.clear();
  TypeMap.clear();
  GlobalMap.clear();
  assert(LocalValueMap.empty() &&
         "Local Value Map should always be empty between transformations!");
}

// processGlobals - This loops over global constants defined in the
// module, converting them to their new type.
//
void MutateStructTypes::processGlobals(Module &M) {
  // Loop through the functions in the module and create a new version of the
  // function to contained the transformed code.  Also, be careful to not
  // process the values that we add.
  //
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal()) {
      const FunctionType *NewMTy = 
        cast<FunctionType>(ConvertType(I->getFunctionType()));
      
      // Create a new function to put stuff into...
      Function *NewMeth = new Function(NewMTy, I->getLinkage(), I->getName());
      if (I->hasName())
        I->setName("OLD."+I->getName());

      // Insert the new function into the function list... to be filled in later
      M.getFunctionList().push_back(NewMeth);
      
      // Keep track of the association...
      GlobalMap[I] = NewMeth;
    }

  // TODO: HANDLE GLOBAL VARIABLES

  // Remap the symbol table to refer to the types in a nice way
  //
  SymbolTable &ST = M.getSymbolTable();
  SymbolTable::iterator I = ST.find(Type::TypeTy);
  if (I != ST.end()) {    // Get the type plane for Type's
    SymbolTable::VarMap &Plane = I->second;
    for (SymbolTable::type_iterator TI = Plane.begin(), TE = Plane.end();
         TI != TE; ++TI) {
      // FIXME: This is gross, I'm reaching right into a symbol table and
      // mucking around with it's internals... but oh well.
      //
      TI->second = (Value*)cast<Type>(ConvertType(cast<Type>(TI->second)));
    }
  }
}


// removeDeadGlobals - For this pass, all this does is remove the old versions
// of the functions and global variables that we no longer need.
void MutateStructTypes::removeDeadGlobals(Module &M) {
  // Prepare for deletion of globals by dropping their interdependencies...
  for(Module::iterator I = M.begin(); I != M.end(); ++I) {
    if (GlobalMap.find(I) != GlobalMap.end())
      I->dropAllReferences();
  }

  // Run through and delete the functions and global variables...
#if 0  // TODO: HANDLE GLOBAL VARIABLES
  M->getGlobalList().delete_span(M.gbegin(), M.gbegin()+NumGVars/2);
#endif
  for(Module::iterator I = M.begin(); I != M.end();) {
    if (GlobalMap.find(I) != GlobalMap.end())
      I = M.getFunctionList().erase(I);
    else
      ++I;
  }
}



// transformFunction - This transforms the instructions of the function to use
// the new types.
//
void MutateStructTypes::transformFunction(Function *m) {
  const Function *M = m;
  std::map<const GlobalValue*, GlobalValue*>::iterator GMI = GlobalMap.find(M);
  if (GMI == GlobalMap.end())
    return;  // Do not affect one of our new functions that we are creating

  Function *NewMeth = cast<Function>(GMI->second);

  // Okay, first order of business, create the arguments...
  for (Function::aiterator I = m->abegin(), E = m->aend(),
         DI = NewMeth->abegin(); I != E; ++I, ++DI) {
    DI->setName(I->getName());
    LocalValueMap[I] = DI; // Keep track of value mapping
  }


  // Loop over all of the basic blocks copying instructions over...
  for (Function::const_iterator BB = M->begin(), BBE = M->end(); BB != BBE;
       ++BB) {
    // Create a new basic block and establish a mapping between the old and new
    BasicBlock *NewBB = cast<BasicBlock>(ConvertValue(BB));
    NewMeth->getBasicBlockList().push_back(NewBB);  // Add block to function

    // Copy over all of the instructions in the basic block...
    for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end();
         II != IE; ++II) {

      const Instruction &I = *II;   // Get the current instruction...
      Instruction *NewI = 0;

      switch (I.getOpcode()) {
        // Terminator Instructions
      case Instruction::Ret:
        NewI = new ReturnInst(
                   ConvertValue(cast<ReturnInst>(I).getReturnValue()));
        break;
      case Instruction::Br: {
        const BranchInst &BI = cast<BranchInst>(I);
        if (BI.isConditional()) {
          NewI =
              new BranchInst(cast<BasicBlock>(ConvertValue(BI.getSuccessor(0))),
                             cast<BasicBlock>(ConvertValue(BI.getSuccessor(1))),
                             ConvertValue(BI.getCondition()));
        } else {
          NewI = 
            new BranchInst(cast<BasicBlock>(ConvertValue(BI.getSuccessor(0))));
        }
        break;
      }
      case Instruction::Switch:
      case Instruction::Invoke:
      case Instruction::Unwind:
        assert(0 && "Insn not implemented!");

        // Binary Instructions
      case Instruction::Add:
      case Instruction::Sub:
      case Instruction::Mul:
      case Instruction::Div:
      case Instruction::Rem:
        // Logical Operations
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:

        // Binary Comparison Instructions
      case Instruction::SetEQ:
      case Instruction::SetNE:
      case Instruction::SetLE:
      case Instruction::SetGE:
      case Instruction::SetLT:
      case Instruction::SetGT:
        NewI = BinaryOperator::create((Instruction::BinaryOps)I.getOpcode(),
                                      ConvertValue(I.getOperand(0)),
                                      ConvertValue(I.getOperand(1)));
        break;

      case Instruction::Shr:
      case Instruction::Shl:
        NewI = new ShiftInst(cast<ShiftInst>(I).getOpcode(),
                             ConvertValue(I.getOperand(0)),
                             ConvertValue(I.getOperand(1)));
        break;


        // Memory Instructions
      case Instruction::Alloca:
        NewI = 
          new MallocInst(
                  ConvertType(cast<PointerType>(I.getType())->getElementType()),
                         I.getNumOperands() ? ConvertValue(I.getOperand(0)) :0);
        break;
      case Instruction::Malloc:
        NewI = 
          new MallocInst(
                  ConvertType(cast<PointerType>(I.getType())->getElementType()),
                         I.getNumOperands() ? ConvertValue(I.getOperand(0)) :0);
        break;

      case Instruction::Free:
        NewI = new FreeInst(ConvertValue(I.getOperand(0)));
        break;

      case Instruction::Load:
        NewI = new LoadInst(ConvertValue(I.getOperand(0)));
        break;
      case Instruction::Store:
        NewI = new StoreInst(ConvertValue(I.getOperand(0)),
                             ConvertValue(I.getOperand(1)));
        break;
      case Instruction::GetElementPtr: {
        const GetElementPtrInst &GEP = cast<GetElementPtrInst>(I);
        std::vector<Value*> Indices(GEP.idx_begin(), GEP.idx_end());
        if (!Indices.empty()) {
          const Type *PTy =
            cast<PointerType>(GEP.getOperand(0)->getType())->getElementType();
          AdjustIndices(cast<CompositeType>(PTy), Indices);
        }

        NewI = new GetElementPtrInst(ConvertValue(GEP.getOperand(0)), Indices);
        break;
      }

        // Miscellaneous Instructions
      case Instruction::PHINode: {
        const PHINode &OldPN = cast<PHINode>(I);
        PHINode *PN = new PHINode(ConvertType(OldPN.getType()));
        for (unsigned i = 0; i < OldPN.getNumIncomingValues(); ++i)
          PN->addIncoming(ConvertValue(OldPN.getIncomingValue(i)),
                    cast<BasicBlock>(ConvertValue(OldPN.getIncomingBlock(i))));
        NewI = PN;
        break;
      }
      case Instruction::Cast:
        NewI = new CastInst(ConvertValue(I.getOperand(0)),
                            ConvertType(I.getType()));
        break;
      case Instruction::Call: {
        Value *Meth = ConvertValue(I.getOperand(0));
        std::vector<Value*> Operands;
        for (unsigned i = 1; i < I.getNumOperands(); ++i)
          Operands.push_back(ConvertValue(I.getOperand(i)));
        NewI = new CallInst(Meth, Operands);
        break;
      }
        
      default:
        assert(0 && "UNKNOWN INSTRUCTION ENCOUNTERED!\n");
        break;
      }

      NewI->setName(I.getName());
      NewBB->getInstList().push_back(NewI);

      // Check to see if we had to make a placeholder for this value...
      std::map<const Value*,Value*>::iterator LVMI = LocalValueMap.find(&I);
      if (LVMI != LocalValueMap.end()) {
        // Yup, make sure it's a placeholder...
        Instruction *I = cast<Instruction>(LVMI->second);
        assert(I->getOpcode() == Instruction::UserOp1 && "Not a placeholder!");

        // Replace all uses of the place holder with the real deal...
        I->replaceAllUsesWith(NewI);
        delete I;                    // And free the placeholder memory
      }

      // Keep track of the fact the the local implementation of this instruction
      // is NewI.
      LocalValueMap[&I] = NewI;
    }
  }

  LocalValueMap.clear();
}


bool MutateStructTypes::run(Module &M) {
  processGlobals(M);

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    transformFunction(I);

  removeDeadGlobals(M);
  return true;
}

