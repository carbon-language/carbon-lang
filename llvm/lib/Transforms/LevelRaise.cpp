//===- LevelRaise.cpp - Code to change LLVM to higher level -----------------=//
//
// This file implements the 'raising' part of the LevelChange API.  This is
// useful because, in general, it makes the LLVM code terser and easier to
// analyze.  Note that it is good to run DCE after doing this transformation.
//
//  Eliminate silly things in the source that do not effect the level, but do
//  clean up the code:
//    * Casts of casts
//    - getelementptr/load & getelementptr/store are folded into a direct
//      load or store
//    - Convert this code (for both alloca and malloc):
//          %reg110 = shl uint %n, ubyte 2          ;;<uint>
//          %reg108 = alloca ubyte, uint %reg110            ;;<ubyte*>
//          %cast76 = cast ubyte* %reg108 to uint*          ;;<uint*>
//      To: %cast76 = alloca uint, uint %n
//   Convert explicit addressing to use getelementptr instruction where possible
//      - ...
//
//   Convert explicit addressing on pointers to use getelementptr instruction.
//    - If a pointer is used by arithmetic operation, insert an array casted
//      version into the source program, only for the following pointer types:
//        * Method argument pointers
//        - Pointers returned by alloca or malloc
//        - Pointers returned by function calls
//    - If a pointer is indexed with a value scaled by a constant size equal
//      to the element size of the array, the expression is replaced with a
//      getelementptr instruction.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/LevelChange.h"
#include "llvm/Method.h"
#include "llvm/Support/STLExtras.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/Target/TargetData.h"
#include <map>
#include <algorithm>

#include "llvm/Assembly/Writer.h"

//#define DEBUG_PEEPHOLE_INSTS 1

#ifdef DEBUG_PEEPHOLE_INSTS
#define PRINT_PEEPHOLE(ID, NUM, I)            \
  cerr << "Inst P/H " << ID << "[" << NUM << "] " << I;
#else
#define PRINT_PEEPHOLE(ID, NUM, I)
#endif

#define PRINT_PEEPHOLE1(ID, I1) do { PRINT_PEEPHOLE(ID, 0, I1); } while (0)
#define PRINT_PEEPHOLE2(ID, I1, I2) \
  do { PRINT_PEEPHOLE(ID, 0, I1); PRINT_PEEPHOLE(ID, 1, I2); } while (0)
#define PRINT_PEEPHOLE3(ID, I1, I2, I3) \
  do { PRINT_PEEPHOLE(ID, 0, I1); PRINT_PEEPHOLE(ID, 1, I2); \
       PRINT_PEEPHOLE(ID, 2, I3); } while (0)


// TargetData Hack: Eventually we will have annotations given to us by the
// backend so that we know stuff about type size and alignments.  For now
// though, just use this, because it happens to match the model that GCC uses.
//
const TargetData TD("LevelRaise: Should be GCC though!");


// losslessCastableTypes - Return true if the types are bitwise equivalent.
// This predicate returns true if it is possible to cast from one type to
// another without gaining or losing precision, or altering the bits in any way.
//
static bool losslessCastableTypes(const Type *T1, const Type *T2) {
  assert(T1->isPrimitiveType() || isa<PointerType>(T1));
  assert(T2->isPrimitiveType() || isa<PointerType>(T2));

  if (T1->getPrimitiveID() == T2->getPrimitiveID())
    return true;  // Handles identity cast, and cast of differing pointer types

  // Now we know that they are two differing primitive or pointer types
  switch (T1->getPrimitiveID()) {
  case Type::UByteTyID:   return T2 == Type::SByteTy;
  case Type::SByteTyID:   return T2 == Type::UByteTy;
  case Type::UShortTyID:  return T2 == Type::ShortTy;
  case Type::ShortTyID:   return T2 == Type::UShortTy;
  case Type::UIntTyID:    return T2 == Type::IntTy;
  case Type::IntTyID:     return T2 == Type::UIntTy;
  case Type::ULongTyID:
  case Type::LongTyID:
  case Type::PointerTyID:
    return T2 == Type::ULongTy || T2 == Type::LongTy ||
           T2->getPrimitiveID() == Type::PointerTyID;
  default:
    return false;  // Other types have no identity values
  }
}


// isReinterpretingCast - Return true if the cast instruction specified will
// cause the operand to be "reinterpreted".  A value is reinterpreted if the
// cast instruction would cause the underlying bits to change.
//
static inline bool isReinterpretingCast(const CastInst *CI) {
  return !losslessCastableTypes(CI->getOperand(0)->getType(), CI->getType());
}


// getPointedToStruct - If the argument is a pointer type, and the pointed to
// value is a struct type, return the struct type, else return null.
//
static const StructType *getPointedToStruct(const Type *Ty) {
  const PointerType *PT = dyn_cast<PointerType>(Ty);
  return PT ? dyn_cast<StructType>(PT->getValueType()) : 0;
}


// getStructOffsetType - Return a vector of offsets that are to be used to index
// into the specified struct type to get as close as possible to index as we
// can.  Note that it is possible that we cannot get exactly to Offset, in which
// case we update offset to be the offset we actually obtained.  The resultant
// leaf type is returned.
//
static const Type *getStructOffsetType(const Type *Ty, unsigned &Offset,
                                       vector<ConstPoolVal*> &Offsets) {
  if (!isa<StructType>(Ty)) {
    Offset = 0;   // Return the offset that we were able to acheive
    return Ty;    // Return the leaf type
  }

  assert(Offset < TD.getTypeSize(Ty) && "Offset not in struct!");
  const StructType *STy = cast<StructType>(Ty);
  const StructLayout *SL = TD.getStructLayout(STy);

  // This loop terminates always on a 0 <= i < MemberOffsets.size()
  unsigned i;
  for (i = 0; i < SL->MemberOffsets.size()-1; ++i)
    if (Offset >= SL->MemberOffsets[i] && Offset <  SL->MemberOffsets[i+1])
      break;
  
  assert(Offset >= SL->MemberOffsets[i] && Offset <  SL->MemberOffsets[i+1]);

  // Make sure to save the current index...
  Offsets.push_back(ConstPoolUInt::get(Type::UByteTy, i));

  unsigned SubOffs = Offset - SL->MemberOffsets[i];
  const Type *LeafTy = getStructOffsetType(STy->getElementTypes()[i], SubOffs,
                                           Offsets);
  Offset = SL->MemberOffsets[i] + SubOffs;
  return LeafTy;
}



// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
// with a value, then remove and delete the original instruction.
//
static void ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                                 BasicBlock::iterator &BI, Value *V) {
  Instruction *I = *BI;
  // Replaces all of the uses of the instruction with uses of the value
  I->replaceAllUsesWith(V);

  // Remove the unneccesary instruction now...
  BIL.remove(BI);

  // Make sure to propogate a name if there is one already...
  if (I->hasName() && !V->hasName())
    V->setName(I->getName(), BIL.getParent()->getSymbolTable());

  // Remove the dead instruction now...
  delete I;
}


// ReplaceInstWithInst - Replace the instruction specified by BI with the
// instruction specified by I.  The original instruction is deleted and BI is
// updated to point to the new instruction.
//
static void ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                                BasicBlock::iterator &BI, Instruction *I) {
  assert(I->getParent() == 0 &&
         "ReplaceInstWithInst: Instruction already inserted into basic block!");

  // Insert the new instruction into the basic block...
  BI = BIL.insert(BI, I)+1;

  // Replace all uses of the old instruction, and delete it.
  ReplaceInstWithValue(BIL, BI, I);

  // Reexamine the instruction just inserted next time around the cleanup pass
  // loop.
  --BI;
}


// ExpressionConvertableToType - Return true if it is possible
static bool ExpressionConvertableToType(Value *V, const Type *Ty) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return false;              // Noninstructions can't convert
  if (I->getType() == Ty) return false;  // Expression already correct type!

  switch (I->getOpcode()) {
  case Instruction::Cast:
    // We can convert the expr if the cast destination type is losslessly
    // convertable to the requested type.
    return losslessCastableTypes(Ty, I->getType());

  case Instruction::Add:
  case Instruction::Sub:
    return ExpressionConvertableToType(I->getOperand(0), Ty) &&
           ExpressionConvertableToType(I->getOperand(1), Ty);
  case Instruction::Shl:
  case Instruction::Shr:
    return ExpressionConvertableToType(I->getOperand(0), Ty);
  }
  return false;
}


static Instruction *ConvertExpressionToType(Value *V, const Type *Ty) {
  Instruction *I = cast<Instruction>(V);
  assert(ExpressionConvertableToType(I, Ty) && "Inst is not convertable!");
  BasicBlock *BB = I->getParent();
  BasicBlock::InstListType &BIL = BB->getInstList();
  string Name = I->getName();  if (!Name.empty()) I->setName("");
  Instruction *Res;     // Result of conversion

  //cerr << endl << endl << "Type:\t" << Ty << "\nInst: " << I << "BB Before: " << BB << endl;

  switch (I->getOpcode()) {
  case Instruction::Cast:
    Res = new CastInst(I->getOperand(0), Ty, Name);
    break;
    
  case Instruction::Add:
  case Instruction::Sub:
    Res = BinaryOperator::create(cast<BinaryOperator>(I)->getOpcode(),
                                 ConvertExpressionToType(I->getOperand(0), Ty),
                                 ConvertExpressionToType(I->getOperand(1), Ty),
                                 Name);
    break;

  case Instruction::Shl:
  case Instruction::Shr:
    Res = new ShiftInst(cast<ShiftInst>(I)->getOpcode(),
                        ConvertExpressionToType(I->getOperand(0), Ty),
                        I->getOperand(1), Name);
    break;

  default:
    assert(0 && "Expression convertable, but don't know how to convert?");
    return 0;
  }

  BasicBlock::iterator It = find(BIL.begin(), BIL.end(), I);
  assert(It != BIL.end() && "Instruction not in own basic block??");
  BIL.insert(It, Res);

  //cerr << "RInst: " << Res << "BB After: " << BB << endl << endl;

  return Res;
}



// DoInsertArrayCast - If the argument value has a pointer type, and if the
// argument value is used as an array, insert a cast before the specified 
// basic block iterator that casts the value to an array pointer.  Return the
// new cast instruction (in the CastResult var), or null if no cast is inserted.
//
static bool DoInsertArrayCast(Method *CurMeth, Value *V, BasicBlock *BB,
			      BasicBlock::iterator &InsertBefore,
			      CastInst *&CastResult) {
  const PointerType *ThePtrType = dyn_cast<PointerType>(V->getType());
  if (!ThePtrType) return false;
  bool InsertCast = false;

  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I) {
    Instruction *Inst = cast<Instruction>(*I);
    switch (Inst->getOpcode()) {
    default: break;                  // Not an interesting use...
    case Instruction::Add:           // It's being used as an array index!
  //case Instruction::Sub:
      InsertCast = true;
      break;
    case Instruction::Cast:          // There is already a cast instruction!
      if (const PointerType *PT = dyn_cast<const PointerType>(Inst->getType()))
	if (const ArrayType *AT = dyn_cast<const ArrayType>(PT->getValueType()))
	  if (AT->getElementType() == ThePtrType->getValueType()) {
	    // Cast already exists! Return the existing one!
	    CastResult = cast<CastInst>(Inst);
	    return false;       // No changes made to program though...
	  }
      break;
    }
  }

  if (!InsertCast) return false;  // There is no reason to insert a cast!

  // Insert a cast!
  const Type *ElTy = ThePtrType->getValueType();
  const PointerType *DestTy = PointerType::get(ArrayType::get(ElTy));

  CastResult = new CastInst(V, DestTy);
  BB->getInstList().insert(InsertBefore, CastResult);
  //cerr << "Inserted cast: " << CastResult;
  return true;            // Made a change!
}


// DoInsertArrayCasts - Loop over all "incoming" values in the specified method,
// inserting a cast for pointer values that are used as arrays. For our
// purposes, an incoming value is considered to be either a value that is 
// either a method parameter, a value created by alloca or malloc, or a value
// returned from a function call.  All casts are kept attached to their original
// values through the PtrCasts map.
//
static bool DoInsertArrayCasts(Method *M, map<Value*, CastInst*> &PtrCasts) {
  assert(!M->isExternal() && "Can't handle external methods!");

  // Insert casts for all arguments to the function...
  bool Changed = false;
  BasicBlock *CurBB = M->front();
  BasicBlock::iterator It = CurBB->begin();
  for (Method::ArgumentListType::iterator AI = M->getArgumentList().begin(), 
	 AE = M->getArgumentList().end(); AI != AE; ++AI) {
    CastInst *TheCast = 0;
    if (DoInsertArrayCast(M, *AI, CurBB, It, TheCast)) {
      It = CurBB->begin();      // We might have just invalidated the iterator!
      Changed = true;           // Yes we made a change
      ++It;                     // Insert next cast AFTER this one...
    }

    if (TheCast)                // Is there a cast associated with this value?
      PtrCasts[*AI] = TheCast;  // Yes, add it to the map...
  }

  // TODO: insert casts for alloca, malloc, and function call results.  Also, 
  // look for pointers that already have casts, to add to the map.

  return Changed;
}




// DoElminatePointerArithmetic - Loop over each incoming pointer variable,
// replacing indexing arithmetic with getelementptr calls.
//
static bool DoEliminatePointerArithmetic(const pair<Value*, CastInst*> &Val) {
  Value    *V  = Val.first;   // The original pointer
  CastInst *CV = Val.second;  // The array casted version of the pointer...

  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I) {
    Instruction *Inst = cast<Instruction>(*I);
    if (Inst->getOpcode() != Instruction::Add) 
      continue;   // We only care about add instructions

    BinaryOperator *Add = cast<BinaryOperator>(Inst);

    // Make sure the array is the first operand of the add expression...
    if (Add->getOperand(0) != V)
      Add->swapOperands();

    // Get the amount added to the pointer value...
    Value *AddAmount = Add->getOperand(1);

    
  }
  return false;
}


// Peephole Malloc instructions: we take a look at the use chain of the
// malloc instruction, and try to find out if the following conditions hold:
//   1. The malloc is of the form: 'malloc [sbyte], uint <constant>'
//   2. The only users of the malloc are cast instructions
//   3. Of the cast instructions, there is only one destination pointer type
//      [RTy] where the size of the pointed to object is equal to the number
//      of bytes allocated.
//
// If these conditions hold, we convert the malloc to allocate an [RTy]
// element.  This should be extended in the future to handle arrays. TODO
//
static bool PeepholeMallocInst(BasicBlock *BB, BasicBlock::iterator &BI) {
  MallocInst *MI = cast<MallocInst>(*BI);
  if (!MI->isArrayAllocation()) return false;    // No array allocation?

  ConstPoolUInt *Amt = dyn_cast<ConstPoolUInt>(MI->getArraySize());
  if (Amt == 0 || MI->getAllocatedType() != ArrayType::get(Type::SByteTy))
    return false;

  // Get the number of bytes allocated...
  unsigned Size = Amt->getValue();
  const Type *ResultTy = 0;

  // Loop over all of the uses of the malloc instruction, inspecting casts.
  for (Value::use_iterator I = MI->use_begin(), E = MI->use_end();
       I != E; ++I) {
    if (!isa<CastInst>(*I)) {
      //cerr << "\tnon" << *I;
      return false;  // A non cast user?
    }
    CastInst *CI = cast<CastInst>(*I);
    //cerr << "\t" << CI;
    
    // We only work on casts to pointer types for sure, be conservative
    if (!isa<PointerType>(CI->getType())) {
      cerr << "Found cast of malloc value to non pointer type:\n" << CI;
      return false;
    }

    const Type *DestTy = cast<PointerType>(CI->getType())->getValueType();
    if (TD.getTypeSize(DestTy) == Size && DestTy != ResultTy) {
      // Does the size of the allocated type match the number of bytes
      // allocated?
      //
      if (ResultTy == 0) {
        ResultTy = DestTy;   // Keep note of this for future uses...
      } else {
        // It's overdefined!  We don't know which type to convert to!
        return false;
      }
    }
  }

  // If we get this far, we have either found, or not, a type that is cast to
  // that is of the same size as the malloc instruction.
  if (!ResultTy) return false;

  PRINT_PEEPHOLE1("mall-refine:in ", MI);
  ReplaceInstWithInst(BB->getInstList(), BI, 
                      MI = new MallocInst(PointerType::get(ResultTy)));
  PRINT_PEEPHOLE1("mall-refine:out", MI);
  return true;
}



static bool PeepholeOptimize(BasicBlock *BB, BasicBlock::iterator &BI) {
  Instruction *I = *BI;
  if (I->use_size() == 0) return false;

  if (CastInst *CI = dyn_cast<CastInst>(I)) {
    Value       *Src    = CI->getOperand(0);
    Instruction *SrcI   = dyn_cast<Instruction>(Src); // Nonnull if instr source
    const Type  *DestTy = CI->getType();

    // Check for a cast of the same type as the destination!
    if (DestTy == Src->getType()) {
      PRINT_PEEPHOLE1("cast-of-self-ty", CI);
      CI->replaceAllUsesWith(Src);
      if (!Src->hasName() && CI->hasName()) {
        string Name = CI->getName();
        CI->setName(""); Src->setName(Name);
      }
      return true;
    }

    // Check for a cast of cast, where no size information is lost...
    if (SrcI)
      if (CastInst *CSrc = dyn_cast<CastInst>(SrcI))
        if (isReinterpretingCast(CI) + isReinterpretingCast(CSrc) < 2) {
          // We can only do c-c elimination if, at most, one cast does a
          // reinterpretation of the input data.
          //
          // If legal, make this cast refer the the original casts argument!
          //
          PRINT_PEEPHOLE2("cast-cast:in ", CI, CSrc);
          CI->setOperand(0, CSrc->getOperand(0));
          PRINT_PEEPHOLE1("cast-cast:out", CI);
          return true;
        }

    // Check to see if it's a cast of an instruction that does not depend on the
    // specific type of the operands to do it's job.
    if (SrcI && !isReinterpretingCast(CI) && 
        ExpressionConvertableToType(SrcI, DestTy)) {
      PRINT_PEEPHOLE2("EXPR-CONV:in ", CI, SrcI);
      CI->setOperand(0, ConvertExpressionToType(SrcI, DestTy));
      BI = BB->begin();  // Rescan basic block.  BI might be invalidated.
      PRINT_PEEPHOLE2("EXPR-CONV:out", CI, CI->getOperand(0));
      return true;
    }

  } else if (MallocInst *MI = dyn_cast<MallocInst>(I)) {
    if (PeepholeMallocInst(BB, BI)) return true;
  } else if (I->getOpcode() == Instruction::Add &&
             isa<CastInst>(I->getOperand(1))) {

    // Peephole optimize the following instructions:
    // %t1 = cast ulong <const int> to {<...>} *
    // %t2 = add {<...>} * %SP, %t1              ;; Constant must be 2nd operand
    //
    //    or
    // %t1 = cast {<...>}* %SP to int*
    // %t5 = cast ulong <const int> to int*
    // %t2 = add int* %t1, %t5                   ;; int is same size as field
    //
    // Into: %t3 = getelementptr {<...>} * %SP, <element indices>
    //       %t2 = cast <eltype> * %t3 to {<...>}*
    //
    Value            *AddOp1  = I->getOperand(0);
    CastInst         *AddOp2  = cast<CastInst>(I->getOperand(1));
    ConstPoolUInt    *OffsetV = dyn_cast<ConstPoolUInt>(AddOp2->getOperand(0));
    unsigned          Offset  = OffsetV ? OffsetV->getValue() : 0;
    Value            *SrcPtr;  // Of type pointer to struct...
    const StructType *StructTy;

    if ((StructTy = getPointedToStruct(AddOp1->getType()))) {
      SrcPtr = AddOp1;                      // Handle the first case...
    } else if (CastInst *AddOp1c = dyn_cast<CastInst>(AddOp1)) {
      SrcPtr = AddOp1c->getOperand(0);      // Handle the second case...
      StructTy = getPointedToStruct(SrcPtr->getType());
    }
    
    // Only proceed if we have detected all of our conditions successfully...
    if (Offset && StructTy && SrcPtr && Offset < TD.getTypeSize(StructTy)) {
      const StructLayout *SL = TD.getStructLayout(StructTy);
      vector<ConstPoolVal*> Offsets;
      unsigned ActualOffset = Offset;
      const Type *ElTy = getStructOffsetType(StructTy, ActualOffset, Offsets);

      if (getPointedToStruct(AddOp1->getType())) {  // case 1
        PRINT_PEEPHOLE2("add-to-gep1:in", AddOp2, I);
      } else {
        PRINT_PEEPHOLE3("add-to-gep2:in", AddOp1, AddOp2, I);
      }

      GetElementPtrInst *GEP = new GetElementPtrInst(SrcPtr, Offsets);
      BI = BB->getInstList().insert(BI, GEP)+1;

      assert(Offset-ActualOffset == 0  &&
             "GEP to middle of element not implemented yet!");

      ReplaceInstWithInst(BB->getInstList(), BI, 
                          I = new CastInst(GEP, I->getType()));
      PRINT_PEEPHOLE2("add-to-gep:out", GEP, I);
      return true;
    }
  }

  return false;
}




static bool DoRaisePass(Method *M) {
  bool Changed = false;
  for (Method::iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI) {
    BasicBlock *BB = *MI;
    BasicBlock::InstListType &BIL = BB->getInstList();

    for (BasicBlock::iterator BI = BB->begin(); BI != BB->end();) {
      if (PeepholeOptimize(BB, BI))
        Changed = true;
      else
        ++BI;
    }
  }
  return Changed;
}


// RaisePointerReferences::doit - Raise a method representation to a higher
// level.
//
bool RaisePointerReferences::doit(Method *M) {
  if (M->isExternal()) return false;
  bool Changed = false;

  while (DoRaisePass(M)) Changed = true;

  // PtrCasts - Keep a mapping between the pointer values (the key of the 
  // map), and the cast to array pointer (the value) in this map.  This is
  // used when converting pointer math into array addressing.
  // 
  map<Value*, CastInst*> PtrCasts;

  // Insert casts for all incoming pointer values.  Keep track of those casts
  // and the identified incoming values in the PtrCasts map.
  //
  Changed |= DoInsertArrayCasts(M, PtrCasts);

  // Loop over each incoming pointer variable, replacing indexing arithmetic
  // with getelementptr calls.
  //
  Changed |= reduce_apply_bool(PtrCasts.begin(), PtrCasts.end(), 
                               ptr_fun(DoEliminatePointerArithmetic));

  return Changed;
}
