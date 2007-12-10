//===-- Value.cpp - Implement the Value class -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Value and User classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InstrTypes.h"
#include "llvm/Module.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Constants.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/InstrTypes.h"
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                                Value Class
//===----------------------------------------------------------------------===//

static inline const Type *checkType(const Type *Ty) {
  assert(Ty && "Value defined with a null type: Error!");
  return Ty;
}

Value::Value(const Type *ty, unsigned scid)
  : SubclassID(scid), SubclassData(0), Ty(checkType(ty)),
    UseList(0), Name(0) {
  if (!isa<Constant>(this) && !isa<BasicBlock>(this))
    assert((Ty->isFirstClassType() || Ty == Type::VoidTy ||
           isa<OpaqueType>(ty)) &&
           "Cannot create non-first-class values except for constants!");
}

Value::~Value() 
{
  switch(SubclassID)
  {
  case ArgumentVal:
    Argument::destroyThis(cast<Argument>(this));
    break;
  case BasicBlockVal:
    BasicBlock::destroyThis(cast<BasicBlock>(this));
    break;
  case FunctionVal:
    Function::destroyThis(cast<Function>(this));
    break;
  case GlobalAliasVal:
    GlobalAlias::destroyThis(cast<GlobalAlias>(this));
    break;
  case GlobalVariableVal:
    GlobalVariable::destroyThis(cast<GlobalVariable>(this));
    break;
  case UndefValueVal:
    UndefValue::destroyThis(cast<UndefValue>(this));
    break;
  case ConstantExprVal:
    {
      ConstantExpr* CE = dyn_cast<ConstantExpr>(this);
      if(CE->getOpcode() == Instruction::GetElementPtr)
      {
        GetElementPtrConstantExpr* GECE = 
          dyn_cast<GetElementPtrConstantExpr>(CE);
        GetElementPtrConstantExpr::destroyThis(GECE);
      }
      else if(CE->getOpcode() == Instruction::ExtractElement)
      {
        ExtractElementConstantExpr* EECE = 
          dyn_cast<ExtractElementConstantExpr>(CE);
        ExtractElementConstantExpr::destroyThis(EECE);
      }
      else if(CE->getOpcode() == Instruction::InsertElement)
      {
        InsertElementConstantExpr* IECE = 
          dyn_cast<InsertElementConstantExpr>(CE);
        InsertElementConstantExpr::destroyThis(IECE);
      }
      else if(CE->getOpcode() == Instruction::Select)
      {
        SelectConstantExpr* SCE = dyn_cast<SelectConstantExpr>(CE);
        SelectConstantExpr::destroyThis(SCE);
      }
      else if(CE->getOpcode() == Instruction::ShuffleVector)
      {
        ShuffleVectorConstantExpr* SVCE = 
          dyn_cast<ShuffleVectorConstantExpr>(CE);
        ShuffleVectorConstantExpr::destroyThis(SVCE);
      }
      else if(BinaryConstantExpr* BCE = dyn_cast<BinaryConstantExpr>(this))
        BinaryConstantExpr::destroyThis(BCE);
      else if(UnaryConstantExpr* UCE = dyn_cast<UnaryConstantExpr>(this))
        UnaryConstantExpr::destroyThis(UCE);
      else if(CompareConstantExpr* CCE = dyn_cast<CompareConstantExpr>(this))
        CompareConstantExpr::destroyThis(CCE);
      else
        assert(0 && "Unknown ConstantExpr-inherited class in ~Value.");
    }
    break;
  case ConstantAggregateZeroVal:
    ConstantAggregateZero::destroyThis(cast<ConstantAggregateZero>(this));
    break;
  case ConstantIntVal:          
    ConstantInt::destroyThis(cast<ConstantInt>(this));
    break;
  case ConstantFPVal:         
    ConstantFP::destroyThis(cast<ConstantFP>(this));
    break;
  case ConstantArrayVal:      
    ConstantArray::destroyThis(cast<ConstantArray>(this));
    break;
  case ConstantStructVal:       
    ConstantStruct::destroyThis(cast<ConstantStruct>(this));
    break;
  case ConstantVectorVal:     
    ConstantVector::destroyThis(cast<ConstantVector>(this));
    break;
  case ConstantPointerNullVal:   
    ConstantPointerNull::destroyThis(cast<ConstantPointerNull>(this));
    break;
  case InlineAsmVal:         
    InlineAsm::destroyThis(cast<InlineAsm>(this));
    break;

  default:
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(this))
      BinaryOperator::destroyThis(BO);
    else if (CallInst *CI = dyn_cast<CallInst>(this))
      CallInst::destroyThis(CI);
    else if (CmpInst *CI = dyn_cast<CmpInst>(this))
    {
      if (FCmpInst *FCI = dyn_cast<FCmpInst>(this))
        FCmpInst::destroyThis(FCI);
      else if (ICmpInst *ICI = dyn_cast<ICmpInst>(this))
        ICmpInst::destroyThis(ICI);
      else
        assert(0 && "Unknown CmpInst-inherited class in ~Value.");
    }
    else if (ExtractElementInst *EEI = dyn_cast<ExtractElementInst>(this))
      ExtractElementInst::destroyThis(EEI);
    else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(this))
      GetElementPtrInst::destroyThis(GEP);
    else if (InsertElementInst* IE = dyn_cast<InsertElementInst>(this))
      InsertElementInst::destroyThis(IE);
    else if (PHINode *PN = dyn_cast<PHINode>(this))
      PHINode::destroyThis(PN);
    else if (SelectInst *SI = dyn_cast<SelectInst>(this))
      SelectInst::destroyThis(SI);
    else if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(this))
      ShuffleVectorInst::destroyThis(SVI);
    else if (StoreInst *SI = dyn_cast<StoreInst>(this))
      StoreInst::destroyThis(SI);
    else if (TerminatorInst *TI = dyn_cast<TerminatorInst>(this))
    {
      if (BranchInst* BI = dyn_cast<BranchInst>(this))
        BranchInst::destroyThis(BI);
      else if (InvokeInst* II = dyn_cast<InvokeInst>(this))
        InvokeInst::destroyThis(II);
      else if (ReturnInst* RI = dyn_cast<ReturnInst>(this))
        ReturnInst::destroyThis(RI);
      else if (SwitchInst *SI = dyn_cast<SwitchInst>(this))
        SwitchInst::destroyThis(SI);
      else if (UnreachableInst *UI = dyn_cast<UnreachableInst>(this))
        UnreachableInst::destroyThis(UI);
      else if (UnwindInst *UI = dyn_cast<UnwindInst>(this))
        UnwindInst::destroyThis(UI);
      else
        assert(0 && "Unknown TerminatorInst-inherited class in ~Value.");
    }
    else if(UnaryInstruction* UI = dyn_cast<UnaryInstruction>(this))
    {
      if(AllocationInst* AI = dyn_cast<AllocationInst>(this))
      {
        if(AllocaInst* AI = dyn_cast<AllocaInst>(this))
          AllocaInst::destroyThis(AI);
        else if(MallocInst* MI = dyn_cast<MallocInst>(this))
          MallocInst::destroyThis(MI);
        else
          assert(0 && "Unknown AllocationInst-inherited class in ~Value.");
      } else if(CastInst* CI = dyn_cast<CastInst>(this)) {
        if(BitCastInst* BCI = dyn_cast<BitCastInst>(CI))
          BitCastInst::destroyThis(BCI);
        else if(FPExtInst* FPEI = dyn_cast<FPExtInst>(CI))
          FPExtInst::destroyThis(FPEI);
        else if(FPToSIInst* FPSII = dyn_cast<FPToSIInst>(CI))
          FPToSIInst::destroyThis(FPSII);
        else if(FPToUIInst* FPUII = dyn_cast<FPToUIInst>(CI))
          FPToUIInst::destroyThis(FPUII);
        else if(FPTruncInst* FPTI = dyn_cast<FPTruncInst>(CI))
          FPTruncInst::destroyThis(FPTI);
        else if(IntToPtrInst* I2PI = dyn_cast<IntToPtrInst>(CI))
          IntToPtrInst::destroyThis(I2PI);
        else if(PtrToIntInst* P2II = dyn_cast<PtrToIntInst>(CI))
          PtrToIntInst::destroyThis(P2II);
        else if(SExtInst* SEI = dyn_cast<SExtInst>(CI))
          SExtInst::destroyThis(SEI);
        else if(SIToFPInst* SIFPI = dyn_cast<SIToFPInst>(CI))
          SIToFPInst::destroyThis(SIFPI);
        else if(TruncInst* TI = dyn_cast<TruncInst>(CI))
          TruncInst::destroyThis(TI);
        else if(UIToFPInst* UIFPI = dyn_cast<UIToFPInst>(CI))
          UIToFPInst::destroyThis(UIFPI);
        else if(ZExtInst* ZEI = dyn_cast<ZExtInst>(CI))
          ZExtInst::destroyThis(ZEI);
        else
          assert(0 && "Unknown CastInst-inherited class in ~Value.");
      }
      else if(FreeInst* FI = dyn_cast<FreeInst>(this))
        FreeInst::destroyThis(FI);
      else if(LoadInst* LI = dyn_cast<LoadInst>(this))
        LoadInst::destroyThis(LI);
      else if(VAArgInst* VAI = dyn_cast<VAArgInst>(this))
        VAArgInst::destroyThis(VAI);
      else
        assert(0 && "Unknown UnaryInstruction-inherited class in ~Value.");
    }
    else if (DummyInst *DI = dyn_cast<DummyInst>(this))
      DummyInst::destroyThis(DI);
    else
      assert(0 && "Unknown Instruction-inherited class in ~Value.");
    break;
  }
}

void Value::destroyThis(Value*v)
{
#ifndef NDEBUG      // Only in -g mode...
  // Check to make sure that there are no uses of this value that are still
  // around when the value is destroyed.  If there are, then we have a dangling
  // reference and something is wrong.  This code is here to print out what is
  // still being referenced.  The value in question should be printed as
  // a <badref>
  //
  if (!v->use_empty()) {
    DOUT << "While deleting: " << *v->Ty << " %" << v->Name << "\n";
    for (use_iterator I = v->use_begin(), E = v->use_end(); I != E; ++I)
      DOUT << "Use still stuck around after Def is destroyed:"
           << **I << "\n";
  }
#endif
  assert(v->use_empty() && "Uses remain when a value is destroyed!");

  // If this value is named, destroy the name.  This should not be in a symtab
  // at this point.
  if (v->Name)
    v->Name->Destroy();
  
  // There should be no uses of this object anymore, remove it.
  LeakDetector::removeGarbageObject(v);
}

/// hasNUses - Return true if this Value has exactly N users.
///
bool Value::hasNUses(unsigned N) const {
  use_const_iterator UI = use_begin(), E = use_end();

  for (; N; --N, ++UI)
    if (UI == E) return false;  // Too few.
  return UI == E;
}

/// hasNUsesOrMore - Return true if this value has N users or more.  This is
/// logically equivalent to getNumUses() >= N.
///
bool Value::hasNUsesOrMore(unsigned N) const {
  use_const_iterator UI = use_begin(), E = use_end();

  for (; N; --N, ++UI)
    if (UI == E) return false;  // Too few.

  return true;
}


/// getNumUses - This method computes the number of uses of this Value.  This
/// is a linear time operation.  Use hasOneUse or hasNUses to check for specific
/// values.
unsigned Value::getNumUses() const {
  return (unsigned)std::distance(use_begin(), use_end());
}

static bool getSymTab(Value *V, ValueSymbolTable *&ST) {
  ST = 0;
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    if (BasicBlock *P = I->getParent())
      if (Function *PP = P->getParent())
        ST = &PP->getValueSymbolTable();
  } else if (BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
    if (Function *P = BB->getParent()) 
      ST = &P->getValueSymbolTable();
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    if (Module *P = GV->getParent()) 
      ST = &P->getValueSymbolTable();
  } else if (Argument *A = dyn_cast<Argument>(V)) {
    if (Function *P = A->getParent()) 
      ST = &P->getValueSymbolTable();
  } else {
    assert(isa<Constant>(V) && "Unknown value type!");
    return true;  // no name is setable for this.
  }
  return false;
}

/// getNameStart - Return a pointer to a null terminated string for this name.
/// Note that names can have null characters within the string as well as at
/// their end.  This always returns a non-null pointer.
const char *Value::getNameStart() const {
  if (Name == 0) return "";
  return Name->getKeyData();
}

/// getNameLen - Return the length of the string, correctly handling nul
/// characters embedded into them.
unsigned Value::getNameLen() const {
  return Name ? Name->getKeyLength() : 0;
}


std::string Value::getNameStr() const {
  if (Name == 0) return "";
  return std::string(Name->getKeyData(),
                     Name->getKeyData()+Name->getKeyLength());
}

void Value::setName(const std::string &name) {
  setName(&name[0], name.size());
}

void Value::setName(const char *Name) {
  setName(Name, Name ? strlen(Name) : 0);
}

void Value::setName(const char *NameStr, unsigned NameLen) {
  if (NameLen == 0 && !hasName()) return;
  assert(getType() != Type::VoidTy && "Cannot assign a name to void values!");
  
  // Get the symbol table to update for this object.
  ValueSymbolTable *ST;
  if (getSymTab(this, ST))
    return;  // Cannot set a name on this value (e.g. constant).

  if (!ST) { // No symbol table to update?  Just do the change.
    if (NameLen == 0) {
      // Free the name for this value.
      Name->Destroy();
      Name = 0;
      return;
    }
    
    if (Name) {
      // Name isn't changing?
      if (NameLen == Name->getKeyLength() &&
          !memcmp(Name->getKeyData(), NameStr, NameLen))
        return;
      Name->Destroy();
    }
    
    // NOTE: Could optimize for the case the name is shrinking to not deallocate
    // then reallocated.
      
    // Create the new name.
    Name = ValueName::Create(NameStr, NameStr+NameLen);
    Name->setValue(this);
    return;
  }
  
  // NOTE: Could optimize for the case the name is shrinking to not deallocate
  // then reallocated.
  if (hasName()) {
    // Name isn't changing?
    if (NameLen == Name->getKeyLength() &&
        !memcmp(Name->getKeyData(), NameStr, NameLen))
      return;

    // Remove old name.
    ST->removeValueName(Name);
    Name->Destroy();
    Name = 0;

    if (NameLen == 0)
      return;
  }

  // Name is changing to something new.
  Name = ST->createValueName(NameStr, NameLen, this);
}


/// takeName - transfer the name from V to this value, setting V's name to
/// empty.  It is an error to call V->takeName(V). 
void Value::takeName(Value *V) {
  ValueSymbolTable *ST = 0;
  // If this value has a name, drop it.
  if (hasName()) {
    // Get the symtab this is in.
    if (getSymTab(this, ST)) {
      // We can't set a name on this value, but we need to clear V's name if
      // it has one.
      if (V->hasName()) V->setName(0, 0);
      return;  // Cannot set a name on this value (e.g. constant).
    }
    
    // Remove old name.
    if (ST)
      ST->removeValueName(Name);
    Name->Destroy();
    Name = 0;
  } 
  
  // Now we know that this has no name.
  
  // If V has no name either, we're done.
  if (!V->hasName()) return;
   
  // Get this's symtab if we didn't before.
  if (!ST) {
    if (getSymTab(this, ST)) {
      // Clear V's name.
      V->setName(0, 0);
      return;  // Cannot set a name on this value (e.g. constant).
    }
  }
  
  // Get V's ST, this should always succed, because V has a name.
  ValueSymbolTable *VST;
  bool Failure = getSymTab(V, VST);
  assert(!Failure && "V has a name, so it should have a ST!");
  
  // If these values are both in the same symtab, we can do this very fast.
  // This works even if both values have no symtab yet.
  if (ST == VST) {
    // Take the name!
    Name = V->Name;
    V->Name = 0;
    Name->setValue(this);
    return;
  }
  
  // Otherwise, things are slightly more complex.  Remove V's name from VST and
  // then reinsert it into ST.
  
  if (VST)
    VST->removeValueName(V->Name);
  Name = V->Name;
  V->Name = 0;
  Name->setValue(this);
  
  if (ST)
    ST->reinsertValue(this);
}


// uncheckedReplaceAllUsesWith - This is exactly the same as replaceAllUsesWith,
// except that it doesn't have all of the asserts.  The asserts fail because we
// are half-way done resolving types, which causes some types to exist as two
// different Type*'s at the same time.  This is a sledgehammer to work around
// this problem.
//
void Value::uncheckedReplaceAllUsesWith(Value *New) {
  while (!use_empty()) {
    Use &U = *UseList;
    // Must handle Constants specially, we cannot call replaceUsesOfWith on a
    // constant because they are uniqued.
    if (Constant *C = dyn_cast<Constant>(U.getUser())) {
      if (!isa<GlobalValue>(C)) {
        C->replaceUsesOfWithOnConstant(this, New, &U);
        continue;
      }
    }
    
    U.set(New);
  }
}

void Value::replaceAllUsesWith(Value *New) {
  assert(New && "Value::replaceAllUsesWith(<null>) is invalid!");
  assert(New != this && "this->replaceAllUsesWith(this) is NOT valid!");
  assert(New->getType() == getType() &&
         "replaceAllUses of value with new value of different type!");

  uncheckedReplaceAllUsesWith(New);
}

//===----------------------------------------------------------------------===//
//                                 User Class
//===----------------------------------------------------------------------===//

// replaceUsesOfWith - Replaces all references to the "From" definition with
// references to the "To" definition.
//
void User::replaceUsesOfWith(Value *From, Value *To) {
  if (From == To) return;   // Duh what?

  assert(!isa<Constant>(this) || isa<GlobalValue>(this) &&
         "Cannot call User::replaceUsesofWith on a constant!");

  for (unsigned i = 0, E = getNumOperands(); i != E; ++i)
    if (getOperand(i) == From) {  // Is This operand is pointing to oldval?
      // The side effects of this setOperand call include linking to
      // "To", adding "this" to the uses list of To, and
      // most importantly, removing "this" from the use list of "From".
      setOperand(i, To); // Fix it now...
    }
}

