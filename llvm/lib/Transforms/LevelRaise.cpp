//===- LevelRaise.cpp - Code to change LLVM to higher level -----------------=//
//
// This file implements the 'raising' part of the LevelChange API.  This is
// useful because, in general, it makes the LLVM code terser and easier to
// analyze.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/LevelChange.h"
#include "TransformInternals.h"
#include "llvm/Method.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"
#include "llvm/ConstantVals.h"
#include "llvm/Optimizations/ConstantHandling.h"
#include "llvm/Optimizations/DCE.h"
#include "llvm/Optimizations/ConstantProp.h"
#include "llvm/Analysis/Expressions.h"
#include "Support/STLExtras.h"
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
#define PRINT_PEEPHOLE4(ID, I1, I2, I3, I4) \
  do { PRINT_PEEPHOLE(ID, 0, I1); PRINT_PEEPHOLE(ID, 1, I2); \
       PRINT_PEEPHOLE(ID, 2, I3); PRINT_PEEPHOLE(ID, 3, I4); } while (0)


// isReinterpretingCast - Return true if the cast instruction specified will
// cause the operand to be "reinterpreted".  A value is reinterpreted if the
// cast instruction would cause the underlying bits to change.
//
static inline bool isReinterpretingCast(const CastInst *CI) {
  return!CI->getOperand(0)->getType()->isLosslesslyConvertableTo(CI->getType());
}



// Peephole optimize the following instructions:
// %t1 = cast ? to x *
// %t2 = add x * %SP, %t1              ;; Constant must be 2nd operand
//
// Into: %t3 = getelementptr {<...>} * %SP, <element indices>
//       %t2 = cast <eltype> * %t3 to {<...>}*
//
static bool HandleCastToPointer(BasicBlock::iterator BI,
                                const PointerType *DestPTy) {
  CastInst *CI = cast<CastInst>(*BI);

  // Scan all of the uses, looking for any uses that are not add
  // instructions.  If we have non-adds, do not make this transformation.
  //
  for (Value::use_iterator I = CI->use_begin(), E = CI->use_end();
       I != E; ++I) {
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(*I)) {
      if (BO->getOpcode() != Instruction::Add)
        return false;
    } else {
      return false;
    }
  }

  vector<Value*> Indices;
  Value *Src = CI->getOperand(0);
  const Type *Result = ConvertableToGEP(DestPTy, Src, Indices, &BI);
  const PointerType *UsedType = DestPTy;

  // If we fail, check to see if we have an pointer source type that, if
  // converted to an unsized array type, would work.  In this case, we propogate
  // the cast forward to the input of the add instruction.
  //
  if (Result == 0 && getPointedToComposite(DestPTy) == 0) {
    // Make an unsized array and try again...
    UsedType = PointerType::get(ArrayType::get(DestPTy->getElementType()));
    Result = ConvertableToGEP(UsedType, Src, Indices, &BI);    
  }

  if (Result == 0) return false;  // Not convertable...

  PRINT_PEEPHOLE2("cast-add-to-gep:in", Src, CI);

  // If we have a getelementptr capability... transform all of the 
  // add instruction uses into getelementptr's.
  for (Value::use_iterator UI = CI->use_begin(), E = CI->use_end();
       UI != E; ++UI) {
    Instruction *I = cast<Instruction>(*UI);
    assert(I->getOpcode() == Instruction::Add && I->getNumOperands() == 2 &&
           "Use is not a valid add instruction!");
    
    // Get the value added to the cast result pointer...
    Value *OtherPtr = I->getOperand((I->getOperand(0) == CI) ? 1 : 0);

    BasicBlock *BB = I->getParent();
    BasicBlock::iterator AddIt = find(BB->getInstList().begin(),
                                      BB->getInstList().end(), I);

    if (UsedType != DestPTy) {
      // Insert a cast of the incoming value to something addressable...
      CastInst *C = new CastInst(OtherPtr, UsedType, OtherPtr->getName());
      AddIt = BB->getInstList().insert(AddIt, C)+1;
      OtherPtr = C;
    }

    GetElementPtrInst *GEP = new GetElementPtrInst(OtherPtr, Indices);

    PRINT_PEEPHOLE1("cast-add-to-gep:i", I);
    
    // Replace the old add instruction with the shiny new GEP inst
    ReplaceInstWithInst(BB->getInstList(), AddIt, GEP);
    PRINT_PEEPHOLE1("cast-add-to-gep:o", GEP);
  }
  return true;
}

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
static bool PeepholeOptimizeAddCast(BasicBlock *BB, BasicBlock::iterator &BI,
                                    Value *AddOp1, CastInst *AddOp2) {
  const CompositeType *CompTy;
  Value *OffsetVal = AddOp2->getOperand(0);
  Value *SrcPtr;  // Of type pointer to struct...

  if ((CompTy = getPointedToComposite(AddOp1->getType()))) {
    SrcPtr = AddOp1;                      // Handle the first case...
  } else if (CastInst *AddOp1c = dyn_cast<CastInst>(AddOp1)) {
    SrcPtr = AddOp1c->getOperand(0);      // Handle the second case...
    CompTy = getPointedToComposite(SrcPtr->getType());
  }

  // Only proceed if we have detected all of our conditions successfully...
  if (!CompTy || !SrcPtr || !OffsetVal->getType()->isIntegral())
    return false;

  vector<Value*> Indices;
  if (!ConvertableToGEP(SrcPtr->getType(), OffsetVal, Indices, &BI))
    return false;  // Not convertable... perhaps next time

  if (getPointedToComposite(AddOp1->getType())) {  // case 1
    PRINT_PEEPHOLE2("add-to-gep1:in", AddOp2, *BI);
  } else {
    PRINT_PEEPHOLE3("add-to-gep2:in", AddOp1, AddOp2, *BI);
  }

  GetElementPtrInst *GEP = new GetElementPtrInst(SrcPtr, Indices,
                                                 AddOp2->getName());
  BI = BB->getInstList().insert(BI, GEP)+1;

  Instruction *NCI = new CastInst(GEP, AddOp1->getType());
  ReplaceInstWithInst(BB->getInstList(), BI, NCI);
  PRINT_PEEPHOLE2("add-to-gep:out", GEP, NCI);
  return true;
}

static bool PeepholeOptimize(BasicBlock *BB, BasicBlock::iterator &BI) {
  Instruction *I = *BI;

  if (CastInst *CI = dyn_cast<CastInst>(I)) {
    Value       *Src    = CI->getOperand(0);
    Instruction *SrcI   = dyn_cast<Instruction>(Src); // Nonnull if instr source
    const Type  *DestTy = CI->getType();

    // Peephole optimize the following instruction:
    // %V2 = cast <ty> %V to <ty>
    //
    // Into: <nothing>
    //
    if (DestTy == Src->getType()) {   // Check for a cast to same type as src!!
      PRINT_PEEPHOLE1("cast-of-self-ty", CI);
      CI->replaceAllUsesWith(Src);
      if (!Src->hasName() && CI->hasName()) {
        string Name = CI->getName();
        CI->setName("");
        Src->setName(Name, BB->getParent()->getSymbolTable());
      }
      return true;
    }

    // Peephole optimize the following instructions:
    // %tmp = cast <ty> %V to <ty2>
    // %V   = cast <ty2> %tmp to <ty3>     ; Where ty & ty2 are same size
    //
    // Into: cast <ty> %V to <ty3>
    //
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
    if (!isReinterpretingCast(CI)) {
      // Check to see if we can convert the source of the cast to match the
      // destination type of the cast...
      //
      ValueTypeCache ConvertedTypes;
      if (ValueConvertableToType(CI, Src->getType(), ConvertedTypes)) {
        PRINT_PEEPHOLE2("CAST-DEST-EXPR-CONV:in ", Src, CI);

#ifdef DEBUG_PEEPHOLE_INSTS
        cerr << "\nCONVERTING EXPR TYPE:\n";
#endif
        ValueMapCache ValueMap;
        ConvertValueToNewType(CI, Src, ValueMap);  // This will delete CI!

        BI = BB->begin();  // Rescan basic block.  BI might be invalidated.
        PRINT_PEEPHOLE1("CAST-DEST-EXPR-CONV:out", Src);
#ifdef DEBUG_PEEPHOLE_INSTS
        cerr << "DONE CONVERTING EXPR TYPE: \n\n";// << BB->getParent();
#endif
        return true;
      }

      // Check to see if we can convert the users of the cast value to match the
      // source type of the cast...
      //
      ConvertedTypes.clear();
      if (ExpressionConvertableToType(Src, DestTy, ConvertedTypes)) {
        PRINT_PEEPHOLE2("CAST-SRC-EXPR-CONV:in ", Src, CI);
          
#ifdef DEBUG_PEEPHOLE_INSTS
        cerr << "\nCONVERTING SRC EXPR TYPE:\n";
#endif
        ValueMapCache ValueMap;
        Value *E = ConvertExpressionToType(Src, DestTy, ValueMap);
        if (Constant *CPV = dyn_cast<Constant>(E))
          CI->replaceAllUsesWith(CPV);

        BI = BB->begin();  // Rescan basic block.  BI might be invalidated.
        PRINT_PEEPHOLE1("CAST-SRC-EXPR-CONV:out", E);
#ifdef DEBUG_PEEPHOLE_INSTS
        cerr << "DONE CONVERTING SRC EXPR TYPE: \n\n";// << BB->getParent();
#endif
        return true;
      }
    }

    // Otherwise find out it this cast is a cast to a pointer type, which is
    // then added to some other pointer, then loaded or stored through.  If
    // so, convert the add into a getelementptr instruction...
    //
    if (const PointerType *DestPTy = dyn_cast<PointerType>(DestTy)) {
      if (HandleCastToPointer(BI, DestPTy)) {
        BI = BB->begin();  // Rescan basic block.  BI might be invalidated.
        return true;
      }
    }

    // Check to see if we are casting from a structure pointer to a pointer to
    // the first element of the structure... to avoid munching other peepholes,
    // we only let this happen if there are no add uses of the cast.
    //
    // Peephole optimize the following instructions:
    // %t1 = cast {<...>} * %StructPtr to <ty> *
    //
    // Into: %t2 = getelementptr {<...>} * %StructPtr, <0, 0, 0, ...>
    //       %t1 = cast <eltype> * %t1 to <ty> *
    //
#if 1
    if (const CompositeType *CTy = getPointedToComposite(Src->getType()))
      if (const PointerType *DestPTy = dyn_cast<PointerType>(DestTy)) {

        // Loop over uses of the cast, checking for add instructions.  If an add
        // exists, this is probably a part of a more complex GEP, so we don't
        // want to mess around with the cast.
        //
        bool HasAddUse = false;
        for (Value::use_iterator I = CI->use_begin(), E = CI->use_end();
             I != E; ++I)
          if (isa<Instruction>(*I) &&
              cast<Instruction>(*I)->getOpcode() == Instruction::Add) {
            HasAddUse = true; break;
          }

        // If it doesn't have an add use, check to see if the dest type is
        // losslessly convertable to one of the types in the start of the struct
        // type.
        //
        if (!HasAddUse) {
          const Type *DestPointedTy = DestPTy->getElementType();
          unsigned Depth = 1;
          const CompositeType *CurCTy = CTy;
          const Type *ElTy = 0;

          // Build the index vector, full of all zeros
          vector<Value*> Indices;

          while (CurCTy) {
            if (const StructType *CurSTy = dyn_cast<StructType>(CurCTy)) {
              // Check for a zero element struct type... if we have one, bail.
              if (CurSTy->getElementTypes().size() == 0) break;
            
              // Grab the first element of the struct type, which must lie at
              // offset zero in the struct.
              //
              ElTy = CurSTy->getElementTypes()[0];
            } else {
              ElTy = cast<ArrayType>(CurCTy)->getElementType();
            }

            // Insert a zero to index through this type...
            Indices.push_back(ConstantUInt::get(CurCTy->getIndexType(), 0));

            // Did we find what we're looking for?
            if (ElTy->isLosslesslyConvertableTo(DestPointedTy)) break;
            
            // Nope, go a level deeper.
            ++Depth;
            CurCTy = dyn_cast<CompositeType>(ElTy);
            ElTy = 0;
          }
          
          // Did we find what we were looking for? If so, do the transformation
          if (ElTy) {
            PRINT_PEEPHOLE1("cast-for-first:in", CI);

            // Insert the new T cast instruction... stealing old T's name
            GetElementPtrInst *GEP = new GetElementPtrInst(Src, Indices,
                                                           CI->getName());
            CI->setName("");
            BI = BB->getInstList().insert(BI, GEP)+1;

            // Make the old cast instruction reference the new GEP instead of
            // the old src value.
            //
            CI->setOperand(0, GEP);
            
            PRINT_PEEPHOLE2("cast-for-first:out", GEP, CI);
            return true;
          }
        }
      }
#endif

#if 1
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Value *Val     = SI->getOperand(0);
    Value *Pointer = SI->getPointerOperand();
    
    // Peephole optimize the following instructions:
    // %t1 = getelementptr {<...>} * %StructPtr, <element indices>
    // store <elementty> %v, <elementty> * %t1
    //
    // Into: store <elementty> %v, {<...>} * %StructPtr, <element indices>
    //
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Pointer)) {
      // Append any indices that the store instruction has onto the end of the
      // ones that the GEP is carrying...
      //
      vector<Value*> Indices(GEP->copyIndices());
      Indices.insert(Indices.end(), SI->idx_begin(), SI->idx_end());

      PRINT_PEEPHOLE2("gep-store:in", GEP, SI);
      ReplaceInstWithInst(BB->getInstList(), BI,
                          SI = new StoreInst(Val, GEP->getPointerOperand(),
                                             Indices));
      PRINT_PEEPHOLE1("gep-store:out", SI);
      return true;
    }
    
    // Peephole optimize the following instructions:
    // %t = cast <T1>* %P to <T2> * ;; If T1 is losslessly convertable to T2
    // store <T2> %V, <T2>* %t
    //
    // Into: 
    // %t = cast <T2> %V to <T1>
    // store <T1> %t2, <T1>* %P
    //
    if (CastInst *CI = dyn_cast<CastInst>(Pointer))
      if (Value *CastSrc = CI->getOperand(0)) // CSPT = CastSrcPointerType
        if (PointerType *CSPT = dyn_cast<PointerType>(CastSrc->getType()))
          // convertable types?
          if (Val->getType()->isLosslesslyConvertableTo(CSPT->getElementType()) &&
              !SI->hasIndices()) {      // No subscripts yet!
            PRINT_PEEPHOLE3("st-src-cast:in ", Pointer, Val, SI);

            // Insert the new T cast instruction... stealing old T's name
            CastInst *NCI = new CastInst(Val, CSPT->getElementType(),
                                         CI->getName());
            CI->setName("");
            BI = BB->getInstList().insert(BI, NCI)+1;

            // Replace the old store with a new one!
            ReplaceInstWithInst(BB->getInstList(), BI,
                                SI = new StoreInst(NCI, CastSrc));
            PRINT_PEEPHOLE3("st-src-cast:out", NCI, CastSrc, SI);
            return true;
          }


  } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
#if 0
    Value *Pointer = LI->getPointerOperand();
    
    // Peephole optimize the following instructions:
    // %t1 = getelementptr {<...>} * %StructPtr, <element indices>
    // %V  = load <elementty> * %t1
    //
    // Into: load {<...>} * %StructPtr, <element indices>
    //
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Pointer)) {
      // Append any indices that the load instruction has onto the end of the
      // ones that the GEP is carrying...
      //
      vector<Value*> Indices(GEP->copyIndices());
      Indices.insert(Indices.end(), LI->idx_begin(), LI->idx_end());

      PRINT_PEEPHOLE2("gep-load:in", GEP, LI);
      ReplaceInstWithInst(BB->getInstList(), BI,
                          LI = new LoadInst(GEP->getPointerOperand(),
                                            Indices));
      PRINT_PEEPHOLE1("gep-load:out", LI);
      return true;
    }
#endif
  } else if (I->getOpcode() == Instruction::Add &&
             isa<CastInst>(I->getOperand(1))) {

    if (PeepholeOptimizeAddCast(BB, BI, I->getOperand(0),
                                cast<CastInst>(I->getOperand(1))))
      return true;

#endif
  }

  return false;
}




static bool DoRaisePass(Method *M) {
  bool Changed = false;
  for (Method::iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI) {
    BasicBlock *BB = *MI;
    BasicBlock::InstListType &BIL = BB->getInstList();

    for (BasicBlock::iterator BI = BB->begin(); BI != BB->end();) {
      if (opt::DeadCodeElimination::dceInstruction(BIL, BI) ||
	  opt::ConstantPropogation::doConstantPropogation(BB, BI)) {
        Changed = true; 
#ifdef DEBUG_PEEPHOLE_INSTS
        cerr << "DeadCode Elinated!\n";
#endif
      } else if (PeepholeOptimize(BB, BI))
        Changed = true;
      else
        ++BI;
    }
  }
  return Changed;
}




// DoInsertArrayCast - If the argument value has a pointer type, and if the
// argument value is used as an array, insert a cast before the specified 
// basic block iterator that casts the value to an array pointer.  Return the
// new cast instruction (in the CastResult var), or null if no cast is inserted.
//
static bool DoInsertArrayCast(Value *V, BasicBlock *BB,
			      BasicBlock::iterator InsertBefore) {
  const PointerType *ThePtrType = dyn_cast<PointerType>(V->getType());
  if (!ThePtrType) return false;

  const Type *ElTy = ThePtrType->getElementType();
  if (isa<MethodType>(ElTy) ||
      (isa<ArrayType>(ElTy) && cast<ArrayType>(ElTy)->isUnsized()))
    return false;

  unsigned ElementSize = TD.getTypeSize(ElTy);
  bool InsertCast = false;

  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I) {
    Instruction *Inst = cast<Instruction>(*I);
    switch (Inst->getOpcode()) {
    case Instruction::Cast:          // There is already a cast instruction!
      if (const PointerType *PT = dyn_cast<const PointerType>(Inst->getType()))
	if (const ArrayType *AT = dyn_cast<const ArrayType>(PT->getElementType()))
	  if (AT->getElementType() == ThePtrType->getElementType()) {
	    // Cast already exists! Don't mess around with it.
	    return false;       // No changes made to program though...
	  }
      break;
    case Instruction::Add: {         // Analyze pointer arithmetic...
      Value *OtherOp = Inst->getOperand(Inst->getOperand(0) == V);
      analysis::ExprType Expr = analysis::ClassifyExpression(OtherOp);

      // This looks like array addressing iff:
      //   A. The constant of the index is larger than the size of the element
      //      type.
      //   B. The scale factor is >= the size of the type.
      //
      if (Expr.Offset && getConstantValue(Expr.Offset) >= (int)ElementSize) // A
        InsertCast = true;

      if (Expr.Scale && getConstantValue(Expr.Scale) >= (int)ElementSize) // B
        InsertCast = true;

      break;
    }
    default: break;                  // Not an interesting use...
    }
  }

  if (!InsertCast) return false;  // There is no reason to insert a cast!

  // Calculate the destination pointer type
  const PointerType *DestTy = PointerType::get(ArrayType::get(ElTy));

  // Check to make sure that all uses of the value can be converted over to use
  // the newly typed value.
  //
  ValueTypeCache ConvertedTypes;
  if (!ValueConvertableToType(V, DestTy, ConvertedTypes)) {
    cerr << "FAILED to convert types of values for " << V << " to " << DestTy << "\n";
    ConvertedTypes.clear();
    ValueConvertableToType(V, DestTy, ConvertedTypes);
    return false;
  }
  ConvertedTypes.clear();

  // Insert a cast!
  CastInst *TheCast = 
    new CastInst(Constant::getNullConstant(V->getType()), DestTy, V->getName());
  BB->getInstList().insert(InsertBefore, TheCast);

#ifdef DEBUG_PEEPHOLE_INSTS
  cerr << "Inserting cast for " << V << endl;
#endif

  // Convert users of the old value over to use the cast result...
  ValueMapCache VMC;
  ConvertValueToNewType(V, TheCast, VMC);

  // The cast is the only thing that is allowed to reference the value...
  TheCast->setOperand(0, V);

#ifdef DEBUG_PEEPHOLE_INSTS
  cerr << "Inserted ptr-array cast: " << TheCast;
#endif
  return true;            // Made a change!
}


// DoInsertArrayCasts - Loop over all "incoming" values in the specified method,
// inserting a cast for pointer values that are used as arrays. For our
// purposes, an incoming value is considered to be either a value that is 
// either a method parameter, or a pointer returned from a function call.
//
static bool DoInsertArrayCasts(Method *M) {
  assert(!M->isExternal() && "Can't handle external methods!");

  // Insert casts for all arguments to the function...
  bool Changed = false;
  BasicBlock *CurBB = M->front();

  for (Method::ArgumentListType::iterator AI = M->getArgumentList().begin(), 
	 AE = M->getArgumentList().end(); AI != AE; ++AI) {

    Changed |= DoInsertArrayCast(*AI, CurBB, CurBB->begin());
  }

  // TODO: insert casts for alloca, malloc, and function call results.  Also, 
  // look for pointers that already have casts, to add to the map.

#ifdef DEBUG_PEEPHOLE_INSTS
  if (Changed) cerr << "Inserted casts:\n" << M;
#endif

  return Changed;
}




// RaisePointerReferences::doit - Raise a method representation to a higher
// level.
//
bool RaisePointerReferences::doit(Method *M) {
  if (M->isExternal()) return false;

#ifdef DEBUG_PEEPHOLE_INSTS
  cerr << "\n\n\nStarting to work on Method '" << M->getName() << "'\n";
#endif

  // Insert casts for all incoming pointer pointer values that are treated as
  // arrays...
  //
  bool Changed = false, LocalChange;
  
  do {
#ifdef DEBUG_PEEPHOLE_INSTS
    cerr << "Looping: \n" << M;
#endif
    LocalChange = DoInsertArrayCasts(M);

    // Iterate over the method, refining it, until it converges on a stable
    // state
    while (DoRaisePass(M)) LocalChange = true;
    Changed |= LocalChange;

  } while (LocalChange);

  return Changed;
}
