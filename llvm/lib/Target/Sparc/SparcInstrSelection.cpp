//===-- SparcInstrSelection.cpp -------------------------------------------===//
//
//  BURS instruction selection for SPARC V9 architecture.      
//
//===----------------------------------------------------------------------===//

#include "SparcInternals.h"
#include "SparcInstrSelectionSupport.h"
#include "SparcRegClassInfo.h"
#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrAnnot.h"
#include "llvm/CodeGen/InstrForest.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/Function.h"
#include "llvm/Constants.h"
#include "llvm/ConstantHandling.h"
#include "llvm/Intrinsics.h"
#include "Support/MathExtras.h"
#include <math.h>

static inline void Add3OperandInstr(unsigned Opcode, InstructionNode* Node,
                                    std::vector<MachineInstr*>& mvec) {
  mvec.push_back(BuildMI(Opcode, 3).addReg(Node->leftChild()->getValue())
                                   .addReg(Node->rightChild()->getValue())
                                   .addRegDef(Node->getValue()));
}



//---------------------------------------------------------------------------
// Function: GetMemInstArgs
// 
// Purpose:
//   Get the pointer value and the index vector for a memory operation
//   (GetElementPtr, Load, or Store).  If all indices of the given memory
//   operation are constant, fold in constant indices in a chain of
//   preceding GetElementPtr instructions (if any), and return the
//   pointer value of the first instruction in the chain.
//   All folded instructions are marked so no code is generated for them.
//
// Return values:
//   Returns the pointer Value to use.
//   Returns the resulting IndexVector in idxVec.
//   Returns true/false in allConstantIndices if all indices are/aren't const.
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
// Function: FoldGetElemChain
// 
// Purpose:
//   Fold a chain of GetElementPtr instructions containing only
//   constant offsets into an equivalent (Pointer, IndexVector) pair.
//   Returns the pointer Value, and stores the resulting IndexVector
//   in argument chainIdxVec. This is a helper function for
//   FoldConstantIndices that does the actual folding. 
//---------------------------------------------------------------------------


// Check for a constant 0.
inline bool
IsZero(Value* idx)
{
  return (idx == ConstantSInt::getNullValue(idx->getType()));
}

static Value*
FoldGetElemChain(InstrTreeNode* ptrNode, std::vector<Value*>& chainIdxVec,
                 bool lastInstHasLeadingNonZero)
{
  InstructionNode* gepNode = dyn_cast<InstructionNode>(ptrNode);
  GetElementPtrInst* gepInst =
    dyn_cast_or_null<GetElementPtrInst>(gepNode ? gepNode->getInstruction() :0);

  // ptr value is not computed in this tree or ptr value does not come from GEP
  // instruction
  if (gepInst == NULL)
    return NULL;

  // Return NULL if we don't fold any instructions in.
  Value* ptrVal = NULL;

  // Now chase the chain of getElementInstr instructions, if any.
  // Check for any non-constant indices and stop there.
  // Also, stop if the first index of child is a non-zero array index
  // and the last index of the current node is a non-array index:
  // in that case, a non-array declared type is being accessed as an array
  // which is not type-safe, but could be legal.
  // 
  InstructionNode* ptrChild = gepNode;
  while (ptrChild && (ptrChild->getOpLabel() == Instruction::GetElementPtr ||
                      ptrChild->getOpLabel() == GetElemPtrIdx))
  {
    // Child is a GetElemPtr instruction
    gepInst = cast<GetElementPtrInst>(ptrChild->getValue());
    User::op_iterator OI, firstIdx = gepInst->idx_begin();
    User::op_iterator lastIdx = gepInst->idx_end();
    bool allConstantOffsets = true;

    // The first index of every GEP must be an array index.
    assert((*firstIdx)->getType() == Type::LongTy &&
           "INTERNAL ERROR: Structure index for a pointer type!");

    // If the last instruction had a leading non-zero index, check if the
    // current one references a sequential (i.e., indexable) type.
    // If not, the code is not type-safe and we would create an illegal GEP
    // by folding them, so don't fold any more instructions.
    // 
    if (lastInstHasLeadingNonZero)
      if (! isa<SequentialType>(gepInst->getType()->getElementType()))
        break;   // cannot fold in any preceding getElementPtr instrs.

    // Check that all offsets are constant for this instruction
    for (OI = firstIdx; allConstantOffsets && OI != lastIdx; ++OI)
      allConstantOffsets = isa<ConstantInt>(*OI);

    if (allConstantOffsets) {
      // Get pointer value out of ptrChild.
      ptrVal = gepInst->getPointerOperand();

      // Remember if it has leading zero index: it will be discarded later.
      lastInstHasLeadingNonZero = ! IsZero(*firstIdx);

      // Insert its index vector at the start, skipping any leading [0]
      chainIdxVec.insert(chainIdxVec.begin(),
                         firstIdx + !lastInstHasLeadingNonZero, lastIdx);

      // Mark the folded node so no code is generated for it.
      ((InstructionNode*) ptrChild)->markFoldedIntoParent();

      // Get the previous GEP instruction and continue trying to fold
      ptrChild = dyn_cast<InstructionNode>(ptrChild->leftChild());
    } else // cannot fold this getElementPtr instr. or any preceding ones
      break;
  }

  // If the first getElementPtr instruction had a leading [0], add it back.
  // Note that this instruction is the *last* one successfully folded above.
  if (ptrVal && ! lastInstHasLeadingNonZero) 
    chainIdxVec.insert(chainIdxVec.begin(), ConstantSInt::get(Type::LongTy,0));

  return ptrVal;
}


//---------------------------------------------------------------------------
// Function: GetGEPInstArgs
// 
// Purpose:
//   Helper function for GetMemInstArgs that handles the final getElementPtr
//   instruction used by (or same as) the memory operation.
//   Extracts the indices of the current instruction and tries to fold in
//   preceding ones if all indices of the current one are constant.
//---------------------------------------------------------------------------

static Value *
GetGEPInstArgs(InstructionNode* gepNode,
               std::vector<Value*>& idxVec,
               bool& allConstantIndices)
{
  allConstantIndices = true;
  GetElementPtrInst* gepI = cast<GetElementPtrInst>(gepNode->getInstruction());

  // Default pointer is the one from the current instruction.
  Value* ptrVal = gepI->getPointerOperand();
  InstrTreeNode* ptrChild = gepNode->leftChild(); 

  // Extract the index vector of the GEP instructin.
  // If all indices are constant and first index is zero, try to fold
  // in preceding GEPs with all constant indices.
  for (User::op_iterator OI=gepI->idx_begin(),  OE=gepI->idx_end();
       allConstantIndices && OI != OE; ++OI)
    if (! isa<Constant>(*OI))
      allConstantIndices = false;     // note: this also terminates loop!

  // If we have only constant indices, fold chains of constant indices
  // in this and any preceding GetElemPtr instructions.
  bool foldedGEPs = false;
  bool leadingNonZeroIdx = gepI && ! IsZero(*gepI->idx_begin());
  if (allConstantIndices)
    if (Value* newPtr = FoldGetElemChain(ptrChild, idxVec, leadingNonZeroIdx)) {
      ptrVal = newPtr;
      foldedGEPs = true;
    }

  // Append the index vector of the current instruction.
  // Skip the leading [0] index if preceding GEPs were folded into this.
  idxVec.insert(idxVec.end(),
                gepI->idx_begin() + (foldedGEPs && !leadingNonZeroIdx),
                gepI->idx_end());

  return ptrVal;
}

//---------------------------------------------------------------------------
// Function: GetMemInstArgs
// 
// Purpose:
//   Get the pointer value and the index vector for a memory operation
//   (GetElementPtr, Load, or Store).  If all indices of the given memory
//   operation are constant, fold in constant indices in a chain of
//   preceding GetElementPtr instructions (if any), and return the
//   pointer value of the first instruction in the chain.
//   All folded instructions are marked so no code is generated for them.
//
// Return values:
//   Returns the pointer Value to use.
//   Returns the resulting IndexVector in idxVec.
//   Returns true/false in allConstantIndices if all indices are/aren't const.
//---------------------------------------------------------------------------

static Value*
GetMemInstArgs(InstructionNode* memInstrNode,
               std::vector<Value*>& idxVec,
               bool& allConstantIndices)
{
  allConstantIndices = false;
  Instruction* memInst = memInstrNode->getInstruction();
  assert(idxVec.size() == 0 && "Need empty vector to return indices");

  // If there is a GetElemPtr instruction to fold in to this instr,
  // it must be in the left child for Load and GetElemPtr, and in the
  // right child for Store instructions.
  InstrTreeNode* ptrChild = (memInst->getOpcode() == Instruction::Store
                             ? memInstrNode->rightChild()
                             : memInstrNode->leftChild()); 
  
  // Default pointer is the one from the current instruction.
  Value* ptrVal = ptrChild->getValue(); 

  // Find the "last" GetElemPtr instruction: this one or the immediate child.
  // There will be none if this is a load or a store from a scalar pointer.
  InstructionNode* gepNode = NULL;
  if (isa<GetElementPtrInst>(memInst))
    gepNode = memInstrNode;
  else if (isa<InstructionNode>(ptrChild) && isa<GetElementPtrInst>(ptrVal)) {
    // Child of load/store is a GEP and memInst is its only use.
    // Use its indices and mark it as folded.
    gepNode = cast<InstructionNode>(ptrChild);
    gepNode->markFoldedIntoParent();
  }

  // If there are no indices, return the current pointer.
  // Else extract the pointer from the GEP and fold the indices.
  return gepNode ? GetGEPInstArgs(gepNode, idxVec, allConstantIndices)
                 : ptrVal;
}


//************************ Internal Functions ******************************/


static inline MachineOpCode 
ChooseBprInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode;
  
  Instruction* setCCInstr =
    ((InstructionNode*) instrNode->leftChild())->getInstruction();
  
  switch(setCCInstr->getOpcode())
  {
  case Instruction::SetEQ: opCode = V9::BRZ;   break;
  case Instruction::SetNE: opCode = V9::BRNZ;  break;
  case Instruction::SetLE: opCode = V9::BRLEZ; break;
  case Instruction::SetGE: opCode = V9::BRGEZ; break;
  case Instruction::SetLT: opCode = V9::BRLZ;  break;
  case Instruction::SetGT: opCode = V9::BRGZ;  break;
  default:
    assert(0 && "Unrecognized VM instruction!");
    opCode = V9::INVALID_OPCODE;
    break; 
  }
  
  return opCode;
}


static inline MachineOpCode 
ChooseBpccInstruction(const InstructionNode* instrNode,
                      const BinaryOperator* setCCInstr)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  bool isSigned = setCCInstr->getOperand(0)->getType()->isSigned();
  
  if (isSigned) {
    switch(setCCInstr->getOpcode())
    {
    case Instruction::SetEQ: opCode = V9::BE;  break;
    case Instruction::SetNE: opCode = V9::BNE; break;
    case Instruction::SetLE: opCode = V9::BLE; break;
    case Instruction::SetGE: opCode = V9::BGE; break;
    case Instruction::SetLT: opCode = V9::BL;  break;
    case Instruction::SetGT: opCode = V9::BG;  break;
    default:
      assert(0 && "Unrecognized VM instruction!");
      break; 
    }
  } else {
    switch(setCCInstr->getOpcode())
    {
    case Instruction::SetEQ: opCode = V9::BE;   break;
    case Instruction::SetNE: opCode = V9::BNE;  break;
    case Instruction::SetLE: opCode = V9::BLEU; break;
    case Instruction::SetGE: opCode = V9::BCC;  break;
    case Instruction::SetLT: opCode = V9::BCS;  break;
    case Instruction::SetGT: opCode = V9::BGU;  break;
    default:
      assert(0 && "Unrecognized VM instruction!");
      break; 
    }
  }
  
  return opCode;
}

static inline MachineOpCode 
ChooseBFpccInstruction(const InstructionNode* instrNode,
                       const BinaryOperator* setCCInstr)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  switch(setCCInstr->getOpcode())
  {
  case Instruction::SetEQ: opCode = V9::FBE;  break;
  case Instruction::SetNE: opCode = V9::FBNE; break;
  case Instruction::SetLE: opCode = V9::FBLE; break;
  case Instruction::SetGE: opCode = V9::FBGE; break;
  case Instruction::SetLT: opCode = V9::FBL;  break;
  case Instruction::SetGT: opCode = V9::FBG;  break;
  default:
    assert(0 && "Unrecognized VM instruction!");
    break; 
  }
  
  return opCode;
}


// Create a unique TmpInstruction for a boolean value,
// representing the CC register used by a branch on that value.
// For now, hack this using a little static cache of TmpInstructions.
// Eventually the entire BURG instruction selection should be put
// into a separate class that can hold such information.
// The static cache is not too bad because the memory for these
// TmpInstructions will be freed along with the rest of the Function anyway.
// 
static TmpInstruction*
GetTmpForCC(Value* boolVal, const Function *F, const Type* ccType)
{
  typedef hash_map<const Value*, TmpInstruction*> BoolTmpCache;
  static BoolTmpCache boolToTmpCache;     // Map boolVal -> TmpInstruction*
  static const Function *lastFunction = 0;// Use to flush cache between funcs
  
  assert(boolVal->getType() == Type::BoolTy && "Weird but ok! Delete assert");
  
  if (lastFunction != F) {
    lastFunction = F;
    boolToTmpCache.clear();
  }
  
  // Look for tmpI and create a new one otherwise.  The new value is
  // directly written to map using the ref returned by operator[].
  TmpInstruction*& tmpI = boolToTmpCache[boolVal];
  if (tmpI == NULL)
    tmpI = new TmpInstruction(ccType, boolVal);
  
  return tmpI;
}


static inline MachineOpCode 
ChooseBccInstruction(const InstructionNode* instrNode,
                     bool& isFPBranch)
{
  InstructionNode* setCCNode = (InstructionNode*) instrNode->leftChild();
  assert(setCCNode->getOpLabel() == SetCCOp);
  BinaryOperator* setCCInstr =cast<BinaryOperator>(setCCNode->getInstruction());
  const Type* setCCType = setCCInstr->getOperand(0)->getType();
  
  isFPBranch = setCCType->isFloatingPoint(); // Return value: don't delete!
  
  if (isFPBranch)
    return ChooseBFpccInstruction(instrNode, setCCInstr);
  else
    return ChooseBpccInstruction(instrNode, setCCInstr);
}


static inline MachineOpCode 
ChooseMovFpccInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  switch(instrNode->getInstruction()->getOpcode())
  {
  case Instruction::SetEQ: opCode = V9::MOVFE;  break;
  case Instruction::SetNE: opCode = V9::MOVFNE; break;
  case Instruction::SetLE: opCode = V9::MOVFLE; break;
  case Instruction::SetGE: opCode = V9::MOVFGE; break;
  case Instruction::SetLT: opCode = V9::MOVFL;  break;
  case Instruction::SetGT: opCode = V9::MOVFG;  break;
  default:
    assert(0 && "Unrecognized VM instruction!");
    break; 
  }
  
  return opCode;
}


// Assumes that SUBcc v1, v2 -> v3 has been executed.
// In most cases, we want to clear v3 and then follow it by instruction
// MOVcc 1 -> v3.
// Set mustClearReg=false if v3 need not be cleared before conditional move.
// Set valueToMove=0 if we want to conditionally move 0 instead of 1
//                      (i.e., we want to test inverse of a condition)
// (The latter two cases do not seem to arise because SetNE needs nothing.)
// 
static MachineOpCode
ChooseMovpccAfterSub(const InstructionNode* instrNode,
                     bool& mustClearReg,
                     int& valueToMove)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  mustClearReg = true;
  valueToMove = 1;
  
  switch(instrNode->getInstruction()->getOpcode())
  {
  case Instruction::SetEQ: opCode = V9::MOVE;  break;
  case Instruction::SetLE: opCode = V9::MOVLE; break;
  case Instruction::SetGE: opCode = V9::MOVGE; break;
  case Instruction::SetLT: opCode = V9::MOVL;  break;
  case Instruction::SetGT: opCode = V9::MOVG;  break;
  case Instruction::SetNE: assert(0 && "No move required!"); break;
  default:		     assert(0 && "Unrecognized VM instr!"); break; 
  }
  
  return opCode;
}

static inline MachineOpCode
ChooseConvertToFloatInstr(OpLabel vopCode, const Type* opType)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  switch(vopCode)
  {
  case ToFloatTy: 
    if (opType == Type::SByteTy || opType == Type::ShortTy ||
        opType == Type::IntTy)
      opCode = V9::FITOS;
    else if (opType == Type::LongTy)
      opCode = V9::FXTOS;
    else if (opType == Type::DoubleTy)
      opCode = V9::FDTOS;
    else if (opType == Type::FloatTy)
      ;
    else
      assert(0 && "Cannot convert this type to FLOAT on SPARC");
    break;
      
  case ToDoubleTy: 
    // This is usually used in conjunction with CreateCodeToCopyIntToFloat().
    // Both functions should treat the integer as a 32-bit value for types
    // of 4 bytes or less, and as a 64-bit value otherwise.
    if (opType == Type::SByteTy || opType == Type::UByteTy ||
        opType == Type::ShortTy || opType == Type::UShortTy ||
        opType == Type::IntTy   || opType == Type::UIntTy)
      opCode = V9::FITOD;
    else if (opType == Type::LongTy || opType == Type::ULongTy)
      opCode = V9::FXTOD;
    else if (opType == Type::FloatTy)
      opCode = V9::FSTOD;
    else if (opType == Type::DoubleTy)
      ;
    else
      assert(0 && "Cannot convert this type to DOUBLE on SPARC");
    break;
      
  default:
    break;
  }
  
  return opCode;
}

static inline MachineOpCode 
ChooseConvertFPToIntInstr(Type::PrimitiveID tid, const Type* opType)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;;

  assert((opType == Type::FloatTy || opType == Type::DoubleTy)
         && "This function should only be called for FLOAT or DOUBLE");

  if (tid == Type::UIntTyID) {
    assert(tid != Type::UIntTyID && "FP-to-uint conversions must be expanded"
           " into FP->long->uint for SPARC v9:  SO RUN PRESELECTION PASS!");
  } else if (tid == Type::SByteTyID || tid == Type::ShortTyID || 
             tid == Type::IntTyID   || tid == Type::UByteTyID ||
             tid == Type::UShortTyID) {
    opCode = (opType == Type::FloatTy)? V9::FSTOI : V9::FDTOI;
  } else if (tid == Type::LongTyID || tid == Type::ULongTyID) {
      opCode = (opType == Type::FloatTy)? V9::FSTOX : V9::FDTOX;
  } else
    assert(0 && "Should not get here, Mo!");

  return opCode;
}

MachineInstr*
CreateConvertFPToIntInstr(Type::PrimitiveID destTID,
                          Value* srcVal, Value* destVal)
{
  MachineOpCode opCode = ChooseConvertFPToIntInstr(destTID, srcVal->getType());
  assert(opCode != V9::INVALID_OPCODE && "Expected to need conversion!");
  return BuildMI(opCode, 2).addReg(srcVal).addRegDef(destVal);
}

// CreateCodeToConvertFloatToInt: Convert FP value to signed or unsigned integer
// The FP value must be converted to the dest type in an FP register,
// and the result is then copied from FP to int register via memory.
//
// Since fdtoi converts to signed integers, any FP value V between MAXINT+1
// and MAXUNSIGNED (i.e., 2^31 <= V <= 2^32-1) would be converted incorrectly
// *only* when converting to an unsigned.  (Unsigned byte, short or long
// don't have this problem.)
// For unsigned int, we therefore have to generate the code sequence:
// 
//      if (V > (float) MAXINT) {
//        unsigned result = (unsigned) (V  - (float) MAXINT);
//        result = result + (unsigned) MAXINT;
//      }
//      else
//        result = (unsigned) V;
// 
static void
CreateCodeToConvertFloatToInt(const TargetMachine& target,
                              Value* opVal,
                              Instruction* destI,
                              std::vector<MachineInstr*>& mvec,
                              MachineCodeForInstruction& mcfi)
{
  // Create a temporary to represent the FP register into which the
  // int value will placed after conversion.  The type of this temporary
  // depends on the type of FP register to use: single-prec for a 32-bit
  // int or smaller; double-prec for a 64-bit int.
  // 
  size_t destSize = target.getTargetData().getTypeSize(destI->getType());
  const Type* destTypeToUse = (destSize > 4)? Type::DoubleTy : Type::FloatTy;
  TmpInstruction* destForCast = new TmpInstruction(destTypeToUse, opVal);
  mcfi.addTemp(destForCast);

  // Create the fp-to-int conversion code
  MachineInstr* M =CreateConvertFPToIntInstr(destI->getType()->getPrimitiveID(),
                                             opVal, destForCast);
  mvec.push_back(M);

  // Create the fpreg-to-intreg copy code
  target.getInstrInfo().
    CreateCodeToCopyFloatToInt(target, destI->getParent()->getParent(),
                               destForCast, destI, mvec, mcfi);
}


static inline MachineOpCode 
ChooseAddInstruction(const InstructionNode* instrNode)
{
  return ChooseAddInstructionByType(instrNode->getInstruction()->getType());
}


static inline MachineInstr* 
CreateMovFloatInstruction(const InstructionNode* instrNode,
                          const Type* resultType)
{
  return BuildMI((resultType == Type::FloatTy) ? V9::FMOVS : V9::FMOVD, 2)
                   .addReg(instrNode->leftChild()->getValue())
                   .addRegDef(instrNode->getValue());
}

static inline MachineInstr* 
CreateAddConstInstruction(const InstructionNode* instrNode)
{
  MachineInstr* minstr = NULL;
  
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(isa<Constant>(constOp));
  
  // Cases worth optimizing are:
  // (1) Add with 0 for float or double: use an FMOV of appropriate type,
  //	 instead of an FADD (1 vs 3 cycles).  There is no integer MOV.
  // 
  if (ConstantFP *FPC = dyn_cast<ConstantFP>(constOp)) {
    double dval = FPC->getValue();
    if (dval == 0.0)
      minstr = CreateMovFloatInstruction(instrNode,
                                        instrNode->getInstruction()->getType());
  }
  
  return minstr;
}


static inline MachineOpCode 
ChooseSubInstructionByType(const Type* resultType)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  if (resultType->isInteger() || isa<PointerType>(resultType)) {
      opCode = V9::SUB;
  } else {
    switch(resultType->getPrimitiveID())
    {
    case Type::FloatTyID:  opCode = V9::FSUBS; break;
    case Type::DoubleTyID: opCode = V9::FSUBD; break;
    default: assert(0 && "Invalid type for SUB instruction"); break; 
    }
  }

  return opCode;
}


static inline MachineInstr* 
CreateSubConstInstruction(const InstructionNode* instrNode)
{
  MachineInstr* minstr = NULL;
  
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(isa<Constant>(constOp));
  
  // Cases worth optimizing are:
  // (1) Sub with 0 for float or double: use an FMOV of appropriate type,
  //	 instead of an FSUB (1 vs 3 cycles).  There is no integer MOV.
  // 
  if (ConstantFP *FPC = dyn_cast<ConstantFP>(constOp)) {
    double dval = FPC->getValue();
    if (dval == 0.0)
      minstr = CreateMovFloatInstruction(instrNode,
                                        instrNode->getInstruction()->getType());
  }
  
  return minstr;
}


static inline MachineOpCode 
ChooseFcmpInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  Value* operand = ((InstrTreeNode*) instrNode->leftChild())->getValue();
  switch(operand->getType()->getPrimitiveID()) {
  case Type::FloatTyID:  opCode = V9::FCMPS; break;
  case Type::DoubleTyID: opCode = V9::FCMPD; break;
  default: assert(0 && "Invalid type for FCMP instruction"); break; 
  }
  
  return opCode;
}


// Assumes that leftArg and rightArg are both cast instructions.
//
static inline bool
BothFloatToDouble(const InstructionNode* instrNode)
{
  InstrTreeNode* leftArg = instrNode->leftChild();
  InstrTreeNode* rightArg = instrNode->rightChild();
  InstrTreeNode* leftArgArg = leftArg->leftChild();
  InstrTreeNode* rightArgArg = rightArg->leftChild();
  assert(leftArg->getValue()->getType() == rightArg->getValue()->getType());
  
  // Check if both arguments are floats cast to double
  return (leftArg->getValue()->getType() == Type::DoubleTy &&
          leftArgArg->getValue()->getType() == Type::FloatTy &&
          rightArgArg->getValue()->getType() == Type::FloatTy);
}


static inline MachineOpCode 
ChooseMulInstructionByType(const Type* resultType)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  if (resultType->isInteger())
    opCode = V9::MULX;
  else
    switch(resultType->getPrimitiveID())
      {
      case Type::FloatTyID:  opCode = V9::FMULS; break;
      case Type::DoubleTyID: opCode = V9::FMULD; break;
      default: assert(0 && "Invalid type for MUL instruction"); break; 
      }
  
  return opCode;
}



static inline MachineInstr*
CreateIntNegInstruction(const TargetMachine& target,
                        Value* vreg)
{
  return BuildMI(V9::SUB, 3).addMReg(target.getRegInfo().getZeroRegNum())
    .addReg(vreg).addRegDef(vreg);
}


// Create instruction sequence for any shift operation.
// SLL or SLLX on an operand smaller than the integer reg. size (64bits)
// requires a second instruction for explicit sign-extension.
// Note that we only have to worry about a sign-bit appearing in the
// most significant bit of the operand after shifting (e.g., bit 32 of
// Int or bit 16 of Short), so we do not have to worry about results
// that are as large as a normal integer register.
// 
static inline void
CreateShiftInstructions(const TargetMachine& target,
                        Function* F,
                        MachineOpCode shiftOpCode,
                        Value* argVal1,
                        Value* optArgVal2, /* Use optArgVal2 if not NULL */
                        unsigned optShiftNum, /* else use optShiftNum */
                        Instruction* destVal,
                        std::vector<MachineInstr*>& mvec,
                        MachineCodeForInstruction& mcfi)
{
  assert((optArgVal2 != NULL || optShiftNum <= 64) &&
         "Large shift sizes unexpected, but can be handled below: "
         "You need to check whether or not it fits in immed field below");
  
  // If this is a logical left shift of a type smaller than the standard
  // integer reg. size, we have to extend the sign-bit into upper bits
  // of dest, so we need to put the result of the SLL into a temporary.
  // 
  Value* shiftDest = destVal;
  unsigned opSize = target.getTargetData().getTypeSize(argVal1->getType());
  if ((shiftOpCode == V9::SLL || shiftOpCode == V9::SLLX) && opSize < 8)
    { // put SLL result into a temporary
      shiftDest = new TmpInstruction(argVal1, optArgVal2, "sllTmp");
      mcfi.addTemp(shiftDest);
    }
  
  MachineInstr* M = (optArgVal2 != NULL)
    ? BuildMI(shiftOpCode, 3).addReg(argVal1).addReg(optArgVal2)
                             .addReg(shiftDest, MOTy::Def)
    : BuildMI(shiftOpCode, 3).addReg(argVal1).addZImm(optShiftNum)
                             .addReg(shiftDest, MOTy::Def);
  mvec.push_back(M);
  
  if (shiftDest != destVal)
    { // extend the sign-bit of the result into all upper bits of dest
      assert(8*opSize <= 32 && "Unexpected type size > 4 and < IntRegSize?");
      target.getInstrInfo().
        CreateSignExtensionInstructions(target, F, shiftDest, destVal,
                                        8*opSize, mvec, mcfi);
    }
}


// Does not create any instructions if we cannot exploit constant to
// create a cheaper instruction.
// This returns the approximate cost of the instructions generated,
// which is used to pick the cheapest when both operands are constant.
static unsigned
CreateMulConstInstruction(const TargetMachine &target, Function* F,
                          Value* lval, Value* rval, Instruction* destVal,
                          std::vector<MachineInstr*>& mvec,
                          MachineCodeForInstruction& mcfi)
{
  /* Use max. multiply cost, viz., cost of MULX */
  unsigned cost = target.getInstrInfo().minLatency(V9::MULX);
  unsigned firstNewInstr = mvec.size();
  
  Value* constOp = rval;
  if (! isa<Constant>(constOp))
    return cost;
  
  // Cases worth optimizing are:
  // (1) Multiply by 0 or 1 for any type: replace with copy (ADD or FMOV)
  // (2) Multiply by 2^x for integer types: replace with Shift
  // 
  const Type* resultType = destVal->getType();
  
  if (resultType->isInteger() || isa<PointerType>(resultType)) {
    bool isValidConst;
    int64_t C = GetConstantValueAsSignedInt(constOp, isValidConst);
    if (isValidConst) {
      unsigned pow;
      bool needNeg = false;
      if (C < 0) {
        needNeg = true;
        C = -C;
      }
          
      if (C == 0 || C == 1) {
        cost = target.getInstrInfo().minLatency(V9::ADD);
        unsigned Zero = target.getRegInfo().getZeroRegNum();
        MachineInstr* M;
        if (C == 0)
          M = BuildMI(V9::ADD,3).addMReg(Zero).addMReg(Zero).addRegDef(destVal);
        else
          M = BuildMI(V9::ADD,3).addReg(lval).addMReg(Zero).addRegDef(destVal);
        mvec.push_back(M);
      }
      else if (isPowerOf2(C, pow)) {
        unsigned opSize = target.getTargetData().getTypeSize(resultType);
        MachineOpCode opCode = (opSize <= 32)? V9::SLL : V9::SLLX;
        CreateShiftInstructions(target, F, opCode, lval, NULL, pow,
                                destVal, mvec, mcfi);
      }
          
      if (mvec.size() > 0 && needNeg)
      { // insert <reg = SUB 0, reg> after the instr to flip the sign
        MachineInstr* M = CreateIntNegInstruction(target, destVal);
        mvec.push_back(M);
      }
    }
  } else {
    if (ConstantFP *FPC = dyn_cast<ConstantFP>(constOp)) {
      double dval = FPC->getValue();
      if (fabs(dval) == 1) {
        MachineOpCode opCode =  (dval < 0)
          ? (resultType == Type::FloatTy? V9::FNEGS : V9::FNEGD)
          : (resultType == Type::FloatTy? V9::FMOVS : V9::FMOVD);
        mvec.push_back(BuildMI(opCode,2).addReg(lval).addRegDef(destVal));
      } 
    }
  }
  
  if (firstNewInstr < mvec.size()) {
    cost = 0;
    for (unsigned i=firstNewInstr; i < mvec.size(); ++i)
      cost += target.getInstrInfo().minLatency(mvec[i]->getOpCode());
  }
  
  return cost;
}


// Does not create any instructions if we cannot exploit constant to
// create a cheaper instruction.
// 
static inline void
CreateCheapestMulConstInstruction(const TargetMachine &target,
                                  Function* F,
                                  Value* lval, Value* rval,
                                  Instruction* destVal,
                                  std::vector<MachineInstr*>& mvec,
                                  MachineCodeForInstruction& mcfi)
{
  Value* constOp;
  if (isa<Constant>(lval) && isa<Constant>(rval))
    { // both operands are constant: evaluate and "set" in dest
      Constant* P = ConstantFoldBinaryInstruction(Instruction::Mul,
                                  cast<Constant>(lval), cast<Constant>(rval));
      target.getInstrInfo().CreateCodeToLoadConst(target,F,P,destVal,mvec,mcfi);
    }
  else if (isa<Constant>(rval))         // rval is constant, but not lval
    CreateMulConstInstruction(target, F, lval, rval, destVal, mvec, mcfi);
  else if (isa<Constant>(lval))         // lval is constant, but not rval
    CreateMulConstInstruction(target, F, lval, rval, destVal, mvec, mcfi);
  
  // else neither is constant
  return;
}

// Return NULL if we cannot exploit constant to create a cheaper instruction
static inline void
CreateMulInstruction(const TargetMachine &target, Function* F,
                     Value* lval, Value* rval, Instruction* destVal,
                     std::vector<MachineInstr*>& mvec,
                     MachineCodeForInstruction& mcfi,
                     MachineOpCode forceMulOp = INVALID_MACHINE_OPCODE)
{
  unsigned L = mvec.size();
  CreateCheapestMulConstInstruction(target,F, lval, rval, destVal, mvec, mcfi);
  if (mvec.size() == L) {
    // no instructions were added so create MUL reg, reg, reg.
    // Use FSMULD if both operands are actually floats cast to doubles.
    // Otherwise, use the default opcode for the appropriate type.
    MachineOpCode mulOp = ((forceMulOp != INVALID_MACHINE_OPCODE)
                           ? forceMulOp 
                           : ChooseMulInstructionByType(destVal->getType()));
    mvec.push_back(BuildMI(mulOp, 3).addReg(lval).addReg(rval)
                   .addRegDef(destVal));
  }
}


// Generate a divide instruction for Div or Rem.
// For Rem, this assumes that the operand type will be signed if the result
// type is signed.  This is correct because they must have the same sign.
// 
static inline MachineOpCode 
ChooseDivInstruction(TargetMachine &target,
                     const InstructionNode* instrNode)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType->isInteger())
    opCode = resultType->isSigned()? V9::SDIVX : V9::UDIVX;
  else
    switch(resultType->getPrimitiveID())
      {
      case Type::FloatTyID:  opCode = V9::FDIVS; break;
      case Type::DoubleTyID: opCode = V9::FDIVD; break;
      default: assert(0 && "Invalid type for DIV instruction"); break; 
      }
  
  return opCode;
}


// Return if we cannot exploit constant to create a cheaper instruction
static void
CreateDivConstInstruction(TargetMachine &target,
                          const InstructionNode* instrNode,
                          std::vector<MachineInstr*>& mvec)
{
  Value* LHS  = instrNode->leftChild()->getValue();
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  if (!isa<Constant>(constOp))
    return;

  Instruction* destVal = instrNode->getInstruction();
  unsigned ZeroReg = target.getRegInfo().getZeroRegNum();
  
  // Cases worth optimizing are:
  // (1) Divide by 1 for any type: replace with copy (ADD or FMOV)
  // (2) Divide by 2^x for integer types: replace with SR[L or A]{X}
  // 
  const Type* resultType = instrNode->getInstruction()->getType();
 
  if (resultType->isInteger())
  {
    unsigned pow;
    bool isValidConst;
    int64_t C = GetConstantValueAsSignedInt(constOp, isValidConst);
    if (isValidConst) {
      bool needNeg = false;
      if (C < 0) {
        needNeg = true;
        C = -C;
      }
          
      if (C == 1) {
        mvec.push_back(BuildMI(V9::ADD, 3).addReg(LHS).addMReg(ZeroReg)
                       .addRegDef(destVal));
      } else if (isPowerOf2(C, pow)) {
        unsigned opCode;
        Value* shiftOperand;

        if (resultType->isSigned()) {
          // The result may be negative and we need to add one before shifting 
          // a negative value.  Use:
          //      srl i0, 31, x0; add x0, i0, i1         (if i0 is <= 32 bits)
          // or
          //      srlx i0, 63, x0; add x0, i0, i1         (if i0 is 64 bits)
          // to compute i1=i0+1 if i0 < 0 and i1=i0 otherwise.  
          // 
          TmpInstruction *srlTmp, *addTmp;
          MachineCodeForInstruction& mcfi
            = MachineCodeForInstruction::get(destVal);
          srlTmp = new TmpInstruction(resultType, LHS, 0, "getSign");
          addTmp = new TmpInstruction(resultType, LHS, srlTmp, "incIfNeg");
          mcfi.addTemp(srlTmp);
          mcfi.addTemp(addTmp);

          // Create the SRL or SRLX instruction to get the sign bit
          mvec.push_back(BuildMI((resultType==Type::LongTy)? V9::SRLX:V9::SRL,3)
                         .addReg(LHS)
                         .addSImm((resultType==Type::LongTy)? 63 : 31)
                         .addRegDef(srlTmp));

          // Create the ADD instruction to add 1 for negative values
          mvec.push_back(BuildMI(V9::ADD, 3).addReg(LHS).addReg(srlTmp)
                         .addRegDef(addTmp));

          // Get the shift operand and "right-shift" opcode to do the divide
          shiftOperand = addTmp;
          opCode = (resultType==Type::LongTy) ? V9::SRAX : V9::SRA;
        }
        else {
          // Get the shift operand and "right-shift" opcode to do the divide
          shiftOperand = LHS;
          opCode = (resultType==Type::LongTy) ? V9::SRLX : V9::SRL;
        }

        // Now do the actual shift!
        mvec.push_back(BuildMI(opCode, 3).addReg(shiftOperand).addZImm(pow)
                       .addRegDef(destVal));
      }
          
      if (needNeg && (C == 1 || isPowerOf2(C, pow))) {
        // insert <reg = SUB 0, reg> after the instr to flip the sign
        mvec.push_back(CreateIntNegInstruction(target, destVal));
      }
    }
  } else {
    if (ConstantFP *FPC = dyn_cast<ConstantFP>(constOp)) {
      double dval = FPC->getValue();
      if (fabs(dval) == 1) {
        unsigned opCode = 
          (dval < 0) ? (resultType == Type::FloatTy? V9::FNEGS : V9::FNEGD)
          : (resultType == Type::FloatTy? V9::FMOVS : V9::FMOVD);
              
        mvec.push_back(BuildMI(opCode, 2).addReg(LHS).addRegDef(destVal));
      } 
    }
  }
}


static void
CreateCodeForVariableSizeAlloca(const TargetMachine& target,
                                Instruction* result,
                                unsigned tsize,
                                Value* numElementsVal,
                                std::vector<MachineInstr*>& getMvec)
{
  Value* totalSizeVal;
  MachineInstr* M;
  MachineCodeForInstruction& mcfi = MachineCodeForInstruction::get(result);
  Function *F = result->getParent()->getParent();

  // Enforce the alignment constraints on the stack pointer at
  // compile time if the total size is a known constant.
  if (isa<Constant>(numElementsVal))
    {
      bool isValid;
      int64_t numElem = GetConstantValueAsSignedInt(numElementsVal, isValid);
      assert(isValid && "Unexpectedly large array dimension in alloca!");
      int64_t total = numElem * tsize;
      if (int extra= total % target.getFrameInfo().getStackFrameSizeAlignment())
        total += target.getFrameInfo().getStackFrameSizeAlignment() - extra;
      totalSizeVal = ConstantSInt::get(Type::IntTy, total);
    }
  else
    {
      // The size is not a constant.  Generate code to compute it and
      // code to pad the size for stack alignment.
      // Create a Value to hold the (constant) element size
      Value* tsizeVal = ConstantSInt::get(Type::IntTy, tsize);

      // Create temporary values to hold the result of MUL, SLL, SRL
      // THIS CASE IS INCOMPLETE AND WILL BE FIXED SHORTLY.
      TmpInstruction* tmpProd = new TmpInstruction(numElementsVal, tsizeVal);
      TmpInstruction* tmpSLL  = new TmpInstruction(numElementsVal, tmpProd);
      TmpInstruction* tmpSRL  = new TmpInstruction(numElementsVal, tmpSLL);
      mcfi.addTemp(tmpProd);
      mcfi.addTemp(tmpSLL);
      mcfi.addTemp(tmpSRL);

      // Instruction 1: mul numElements, typeSize -> tmpProd
      // This will optimize the MUL as far as possible.
      CreateMulInstruction(target, F, numElementsVal, tsizeVal, tmpProd,getMvec,
                           mcfi, INVALID_MACHINE_OPCODE);

      assert(0 && "Need to insert padding instructions here!");

      totalSizeVal = tmpProd;
    }

  // Get the constant offset from SP for dynamically allocated storage
  // and create a temporary Value to hold it.
  MachineFunction& mcInfo = MachineFunction::get(F);
  bool growUp;
  ConstantSInt* dynamicAreaOffset =
    ConstantSInt::get(Type::IntTy,
                     target.getFrameInfo().getDynamicAreaOffset(mcInfo,growUp));
  assert(! growUp && "Has SPARC v9 stack frame convention changed?");

  unsigned SPReg = target.getRegInfo().getStackPointer();

  // Instruction 2: sub %sp, totalSizeVal -> %sp
  getMvec.push_back(BuildMI(V9::SUB, 3).addMReg(SPReg).addReg(totalSizeVal)
                    .addMReg(SPReg,MOTy::Def));

  // Instruction 3: add %sp, frameSizeBelowDynamicArea -> result
  getMvec.push_back(BuildMI(V9::ADD, 3).addMReg(SPReg).addReg(dynamicAreaOffset)
                    .addRegDef(result));
}        


static void
CreateCodeForFixedSizeAlloca(const TargetMachine& target,
                             Instruction* result,
                             unsigned tsize,
                             unsigned numElements,
                             std::vector<MachineInstr*>& getMvec)
{
  assert(tsize > 0 && "Illegal (zero) type size for alloca");
  assert(result && result->getParent() &&
         "Result value is not part of a function?");
  Function *F = result->getParent()->getParent();
  MachineFunction &mcInfo = MachineFunction::get(F);

  // Check if the offset would small enough to use as an immediate in
  // load/stores (check LDX because all load/stores have the same-size immediate
  // field).  If not, put the variable in the dynamically sized area of the
  // frame.
  unsigned paddedSizeIgnored;
  int offsetFromFP = mcInfo.getInfo()->computeOffsetforLocalVar(result,
                                                     paddedSizeIgnored,
                                                     tsize * numElements);
  if (! target.getInstrInfo().constantFitsInImmedField(V9::LDX, offsetFromFP)) {
    CreateCodeForVariableSizeAlloca(target, result, tsize, 
				    ConstantSInt::get(Type::IntTy,numElements),
				    getMvec);
    return;
  }
  
  // else offset fits in immediate field so go ahead and allocate it.
  offsetFromFP = mcInfo.getInfo()->allocateLocalVar(result, tsize *numElements);
  
  // Create a temporary Value to hold the constant offset.
  // This is needed because it may not fit in the immediate field.
  ConstantSInt* offsetVal = ConstantSInt::get(Type::IntTy, offsetFromFP);
  
  // Instruction 1: add %fp, offsetFromFP -> result
  unsigned FPReg = target.getRegInfo().getFramePointer();
  getMvec.push_back(BuildMI(V9::ADD, 3).addMReg(FPReg).addReg(offsetVal)
                    .addRegDef(result));
}


//------------------------------------------------------------------------ 
// Function SetOperandsForMemInstr
//
// Choose addressing mode for the given load or store instruction.
// Use [reg+reg] if it is an indexed reference, and the index offset is
//		 not a constant or if it cannot fit in the offset field.
// Use [reg+offset] in all other cases.
// 
// This assumes that all array refs are "lowered" to one of these forms:
//	%x = load (subarray*) ptr, constant	; single constant offset
//	%x = load (subarray*) ptr, offsetVal	; single non-constant offset
// Generally, this should happen via strength reduction + LICM.
// Also, strength reduction should take care of using the same register for
// the loop index variable and an array index, when that is profitable.
//------------------------------------------------------------------------ 

static void
SetOperandsForMemInstr(unsigned Opcode,
                       std::vector<MachineInstr*>& mvec,
                       InstructionNode* vmInstrNode,
                       const TargetMachine& target)
{
  Instruction* memInst = vmInstrNode->getInstruction();
  // Index vector, ptr value, and flag if all indices are const.
  std::vector<Value*> idxVec;
  bool allConstantIndices;
  Value* ptrVal = GetMemInstArgs(vmInstrNode, idxVec, allConstantIndices);

  // Now create the appropriate operands for the machine instruction.
  // First, initialize so we default to storing the offset in a register.
  int64_t smallConstOffset = 0;
  Value* valueForRegOffset = NULL;
  MachineOperand::MachineOperandType offsetOpType =
    MachineOperand::MO_VirtualRegister;

  // Check if there is an index vector and if so, compute the
  // right offset for structures and for arrays 
  // 
  if (!idxVec.empty())
    {
      const PointerType* ptrType = cast<PointerType>(ptrVal->getType());
      
      // If all indices are constant, compute the combined offset directly.
      if (allConstantIndices)
        {
          // Compute the offset value using the index vector. Create a
          // virtual reg. for it since it may not fit in the immed field.
          uint64_t offset = target.getTargetData().getIndexedOffset(ptrType,idxVec);
          valueForRegOffset = ConstantSInt::get(Type::LongTy, offset);
        }
      else
        {
          // There is at least one non-constant offset.  Therefore, this must
          // be an array ref, and must have been lowered to a single non-zero
          // offset.  (An extra leading zero offset, if any, can be ignored.)
          // Generate code sequence to compute address from index.
          // 
          bool firstIdxIsZero = IsZero(idxVec[0]);
          assert(idxVec.size() == 1U + firstIdxIsZero 
                 && "Array refs must be lowered before Instruction Selection");

          Value* idxVal = idxVec[firstIdxIsZero];

          std::vector<MachineInstr*> mulVec;
          Instruction* addr = new TmpInstruction(Type::ULongTy, memInst);
          MachineCodeForInstruction::get(memInst).addTemp(addr);

          // Get the array type indexed by idxVal, and compute its element size.
          // The call to getTypeSize() will fail if size is not constant.
          const Type* vecType = (firstIdxIsZero
                                 ? GetElementPtrInst::getIndexedType(ptrType,
                                           std::vector<Value*>(1U, idxVec[0]),
                                           /*AllowCompositeLeaf*/ true)
                                 : ptrType);
          const Type* eltType = cast<SequentialType>(vecType)->getElementType();
          ConstantUInt* eltSizeVal = ConstantUInt::get(Type::ULongTy,
                                       target.getTargetData().getTypeSize(eltType));

          // CreateMulInstruction() folds constants intelligently enough.
          CreateMulInstruction(target, memInst->getParent()->getParent(),
                               idxVal,         /* lval, not likely to be const*/
                               eltSizeVal,     /* rval, likely to be constant */
                               addr,           /* result */
                               mulVec, MachineCodeForInstruction::get(memInst),
                               INVALID_MACHINE_OPCODE);

          assert(mulVec.size() > 0 && "No multiply code created?");
          mvec.insert(mvec.end(), mulVec.begin(), mulVec.end());

          valueForRegOffset = addr;
        }
    }
  else
    {
      offsetOpType = MachineOperand::MO_SignExtendedImmed;
      smallConstOffset = 0;
    }

  // For STORE:
  //   Operand 0 is value, operand 1 is ptr, operand 2 is offset
  // For LOAD or GET_ELEMENT_PTR,
  //   Operand 0 is ptr, operand 1 is offset, operand 2 is result.
  // 
  unsigned offsetOpNum, ptrOpNum;
  MachineInstr *MI;
  if (memInst->getOpcode() == Instruction::Store) {
    if (offsetOpType == MachineOperand::MO_VirtualRegister)
      MI = BuildMI(Opcode, 3).addReg(vmInstrNode->leftChild()->getValue())
                             .addReg(ptrVal).addReg(valueForRegOffset);
    else
      MI = BuildMI(Opcode, 3).addReg(vmInstrNode->leftChild()->getValue())
                             .addReg(ptrVal).addSImm(smallConstOffset);
  } else {
    if (offsetOpType == MachineOperand::MO_VirtualRegister)
      MI = BuildMI(Opcode, 3).addReg(ptrVal).addReg(valueForRegOffset)
                             .addRegDef(memInst);
    else
      MI = BuildMI(Opcode, 3).addReg(ptrVal).addSImm(smallConstOffset)
                             .addRegDef(memInst);
  }
  mvec.push_back(MI);
}


// 
// Substitute operand `operandNum' of the instruction in node `treeNode'
// in place of the use(s) of that instruction in node `parent'.
// Check both explicit and implicit operands!
// Also make sure to skip over a parent who:
// (1) is a list node in the Burg tree, or
// (2) itself had its results forwarded to its parent
// 
static void
ForwardOperand(InstructionNode* treeNode,
               InstrTreeNode*   parent,
               int operandNum)
{
  assert(treeNode && parent && "Invalid invocation of ForwardOperand");
  
  Instruction* unusedOp = treeNode->getInstruction();
  Value* fwdOp = unusedOp->getOperand(operandNum);

  // The parent itself may be a list node, so find the real parent instruction
  while (parent->getNodeType() != InstrTreeNode::NTInstructionNode)
    {
      parent = parent->parent();
      assert(parent && "ERROR: Non-instruction node has no parent in tree.");
    }
  InstructionNode* parentInstrNode = (InstructionNode*) parent;
  
  Instruction* userInstr = parentInstrNode->getInstruction();
  MachineCodeForInstruction &mvec = MachineCodeForInstruction::get(userInstr);

  // The parent's mvec would be empty if it was itself forwarded.
  // Recursively call ForwardOperand in that case...
  //
  if (mvec.size() == 0)
    {
      assert(parent->parent() != NULL &&
             "Parent could not have been forwarded, yet has no instructions?");
      ForwardOperand(treeNode, parent->parent(), operandNum);
    }
  else
    {
      for (unsigned i=0, N=mvec.size(); i < N; i++)
        {
          MachineInstr* minstr = mvec[i];
          for (unsigned i=0, numOps=minstr->getNumOperands(); i < numOps; ++i)
            {
              const MachineOperand& mop = minstr->getOperand(i);
              if (mop.getType() == MachineOperand::MO_VirtualRegister &&
                  mop.getVRegValue() == unusedOp)
                minstr->SetMachineOperandVal(i,
                                MachineOperand::MO_VirtualRegister, fwdOp);
            }
          
          for (unsigned i=0,numOps=minstr->getNumImplicitRefs(); i<numOps; ++i)
            if (minstr->getImplicitRef(i) == unusedOp)
              minstr->setImplicitRef(i, fwdOp,
                                     minstr->implicitRefIsDefined(i),
                                     minstr->implicitRefIsDefinedAndUsed(i));
        }
    }
}


inline bool
AllUsesAreBranches(const Instruction* setccI)
{
  for (Value::use_const_iterator UI=setccI->use_begin(), UE=setccI->use_end();
       UI != UE; ++UI)
    if (! isa<TmpInstruction>(*UI)     // ignore tmp instructions here
        && cast<Instruction>(*UI)->getOpcode() != Instruction::Br)
      return false;
  return true;
}

// Generate code for any intrinsic that needs a special code sequence
// instead of a regular call.  If not that kind of intrinsic, do nothing.
// Returns true if code was generated, otherwise false.
// 
bool CodeGenIntrinsic(LLVMIntrinsic::ID iid, CallInst &callInstr,
                      TargetMachine &target,
                      std::vector<MachineInstr*>& mvec)
{
  switch (iid) {
  case LLVMIntrinsic::va_start: {
    // Get the address of the first vararg value on stack and copy it to
    // the argument of va_start(va_list* ap).
    bool ignore;
    Function* func = cast<Function>(callInstr.getParent()->getParent());
    int numFixedArgs   = func->getFunctionType()->getNumParams();
    int fpReg          = target.getFrameInfo().getIncomingArgBaseRegNum();
    int argSize        = target.getFrameInfo().getSizeOfEachArgOnStack();
    int firstVarArgOff = numFixedArgs * argSize + target.getFrameInfo().
      getFirstIncomingArgOffset(MachineFunction::get(func), ignore);
    mvec.push_back(BuildMI(V9::ADD, 3).addMReg(fpReg).addSImm(firstVarArgOff).
                   addReg(callInstr.getOperand(1)));
    return true;
  }

  case LLVMIntrinsic::va_end:
    return true;                        // no-op on Sparc

  case LLVMIntrinsic::va_copy:
    // Simple copy of current va_list (arg2) to new va_list (arg1)
    mvec.push_back(BuildMI(V9::OR, 3).
                   addMReg(target.getRegInfo().getZeroRegNum()).
                   addReg(callInstr.getOperand(2)).
                   addReg(callInstr.getOperand(1)));
    return true;

  default:
    return false;
  }
}

//******************* Externally Visible Functions *************************/

//------------------------------------------------------------------------ 
// External Function: ThisIsAChainRule
//
// Purpose:
//   Check if a given BURG rule is a chain rule.
//------------------------------------------------------------------------ 

extern bool
ThisIsAChainRule(int eruleno)
{
  switch(eruleno)
    {
    case 111:	// stmt:  reg
    case 123:
    case 124:
    case 125:
    case 126:
    case 127:
    case 128:
    case 129:
    case 130:
    case 131:
    case 132:
    case 133:
    case 155:
    case 221:
    case 222:
    case 241:
    case 242:
    case 243:
    case 244:
    case 245:
    case 321:
      return true; break;

    default:
      return false; break;
    }
}


//------------------------------------------------------------------------ 
// External Function: GetInstructionsByRule
//
// Purpose:
//   Choose machine instructions for the SPARC according to the
//   patterns chosen by the BURG-generated parser.
//------------------------------------------------------------------------ 

void
GetInstructionsByRule(InstructionNode* subtreeRoot,
                      int ruleForNode,
                      short* nts,
                      TargetMachine &target,
                      std::vector<MachineInstr*>& mvec)
{
  bool checkCast = false;		// initialize here to use fall-through
  bool maskUnsignedResult = false;
  int nextRule;
  int forwardOperandNum = -1;
  unsigned allocaSize = 0;
  MachineInstr* M, *M2;
  unsigned L;

  mvec.clear(); 
  
  // If the code for this instruction was folded into the parent (user),
  // then do nothing!
  if (subtreeRoot->isFoldedIntoParent())
    return;
  
  // 
  // Let's check for chain rules outside the switch so that we don't have
  // to duplicate the list of chain rule production numbers here again
  // 
  if (ThisIsAChainRule(ruleForNode))
    {
      // Chain rules have a single nonterminal on the RHS.
      // Get the rule that matches the RHS non-terminal and use that instead.
      // 
      assert(nts[0] && ! nts[1]
             && "A chain rule should have only one RHS non-terminal!");
      nextRule = burm_rule(subtreeRoot->state, nts[0]);
      nts = burm_nts[nextRule];
      GetInstructionsByRule(subtreeRoot, nextRule, nts, target, mvec);
    }
  else
    {
      switch(ruleForNode) {
      case 1:	// stmt:   Ret
      case 2:	// stmt:   RetValue(reg)
      {         // NOTE: Prepass of register allocation is responsible
                //	 for moving return value to appropriate register.
                // Mark the return-address register as a hidden virtual reg.
                // Mark the return value   register as an implicit ref of
                // the machine instruction.
         	// Finally put a NOP in the delay slot.
        ReturnInst *returnInstr =
          cast<ReturnInst>(subtreeRoot->getInstruction());
        assert(returnInstr->getOpcode() == Instruction::Ret);
        
        Instruction* returnReg = new TmpInstruction(returnInstr);
        MachineCodeForInstruction::get(returnInstr).addTemp(returnReg);
        
        M = BuildMI(V9::JMPLRET, 3).addReg(returnReg).addSImm(8)
          .addMReg(target.getRegInfo().getZeroRegNum(), MOTy::Def);
        
        if (returnInstr->getReturnValue() != NULL)
          M->addImplicitRef(returnInstr->getReturnValue());
        
        mvec.push_back(M);
        mvec.push_back(BuildMI(V9::NOP, 0));
        
        break;
      }  
        
      case 3:	// stmt:   Store(reg,reg)
      case 4:	// stmt:   Store(reg,ptrreg)
        SetOperandsForMemInstr(ChooseStoreInstruction(
                        subtreeRoot->leftChild()->getValue()->getType()),
                               mvec, subtreeRoot, target);
        break;

      case 5:	// stmt:   BrUncond
        {
          BranchInst *BI = cast<BranchInst>(subtreeRoot->getInstruction());
          mvec.push_back(BuildMI(V9::BA, 1).addPCDisp(BI->getSuccessor(0)));
        
          // delay slot
          mvec.push_back(BuildMI(V9::NOP, 0));
          break;
        }

      case 206:	// stmt:   BrCond(setCCconst)
      { // setCCconst => boolean was computed with `%b = setCC type reg1 const'
        // If the constant is ZERO, we can use the branch-on-integer-register
        // instructions and avoid the SUBcc instruction entirely.
        // Otherwise this is just the same as case 5, so just fall through.
        // 
        InstrTreeNode* constNode = subtreeRoot->leftChild()->rightChild();
        assert(constNode &&
               constNode->getNodeType() ==InstrTreeNode::NTConstNode);
        Constant *constVal = cast<Constant>(constNode->getValue());
        bool isValidConst;
        
        if ((constVal->getType()->isInteger()
             || isa<PointerType>(constVal->getType()))
            && GetConstantValueAsSignedInt(constVal, isValidConst) == 0
            && isValidConst)
          {
            // That constant is a zero after all...
            // Use the left child of setCC as the first argument!
            // Mark the setCC node so that no code is generated for it.
            InstructionNode* setCCNode = (InstructionNode*)
                                         subtreeRoot->leftChild();
            assert(setCCNode->getOpLabel() == SetCCOp);
            setCCNode->markFoldedIntoParent();
            
            BranchInst* brInst=cast<BranchInst>(subtreeRoot->getInstruction());
            
            M = BuildMI(ChooseBprInstruction(subtreeRoot), 2)
                                .addReg(setCCNode->leftChild()->getValue())
                                .addPCDisp(brInst->getSuccessor(0));
            mvec.push_back(M);
            
            // delay slot
            mvec.push_back(BuildMI(V9::NOP, 0));

            // false branch
            mvec.push_back(BuildMI(V9::BA, 1)
                           .addPCDisp(brInst->getSuccessor(1)));
            
            // delay slot
            mvec.push_back(BuildMI(V9::NOP, 0));
            break;
          }
        // ELSE FALL THROUGH
      }

      case 6:	// stmt:   BrCond(setCC)
      { // bool => boolean was computed with SetCC.
        // The branch to use depends on whether it is FP, signed, or unsigned.
        // If it is an integer CC, we also need to find the unique
        // TmpInstruction representing that CC.
        // 
        BranchInst* brInst = cast<BranchInst>(subtreeRoot->getInstruction());
        bool isFPBranch;
        unsigned Opcode = ChooseBccInstruction(subtreeRoot, isFPBranch);
        Value* ccValue = GetTmpForCC(subtreeRoot->leftChild()->getValue(),
                                     brInst->getParent()->getParent(),
                                     isFPBranch? Type::FloatTy : Type::IntTy);
        M = BuildMI(Opcode, 2).addCCReg(ccValue)
                              .addPCDisp(brInst->getSuccessor(0));
        mvec.push_back(M);

        // delay slot
        mvec.push_back(BuildMI(V9::NOP, 0));

        // false branch
        mvec.push_back(BuildMI(V9::BA, 1).addPCDisp(brInst->getSuccessor(1)));

        // delay slot
        mvec.push_back(BuildMI(V9::NOP, 0));
        break;
      }
        
      case 208:	// stmt:   BrCond(boolconst)
      {
        // boolconst => boolean is a constant; use BA to first or second label
        Constant* constVal = 
          cast<Constant>(subtreeRoot->leftChild()->getValue());
        unsigned dest = cast<ConstantBool>(constVal)->getValue()? 0 : 1;
        
        M = BuildMI(V9::BA, 1).addPCDisp(
          cast<BranchInst>(subtreeRoot->getInstruction())->getSuccessor(dest));
        mvec.push_back(M);
        
        // delay slot
        mvec.push_back(BuildMI(V9::NOP, 0));
        break;
      }
        
      case   8:	// stmt:   BrCond(boolreg)
      { // boolreg   => boolean is stored in an existing register.
        // Just use the branch-on-integer-register instruction!
        // 
        BranchInst *BI = cast<BranchInst>(subtreeRoot->getInstruction());
        M = BuildMI(V9::BRNZ, 2).addReg(subtreeRoot->leftChild()->getValue())
          .addPCDisp(BI->getSuccessor(0));
        mvec.push_back(M);

        // delay slot
        mvec.push_back(BuildMI(V9::NOP, 0));

        // false branch
        mvec.push_back(BuildMI(V9::BA, 1).addPCDisp(BI->getSuccessor(1)));
        
        // delay slot
        mvec.push_back(BuildMI(V9::NOP, 0));
        break;
      }  
      
      case 9:	// stmt:   Switch(reg)
        assert(0 && "*** SWITCH instruction is not implemented yet.");
        break;

      case 10:	// reg:   VRegList(reg, reg)
        assert(0 && "VRegList should never be the topmost non-chain rule");
        break;

      case 21:	// bool:  Not(bool,reg): Both these are implemented as:
      case 421:	// reg:   BNot(reg,reg):	reg = reg XOR-NOT 0
      { // First find the unary operand. It may be left or right, usually right.
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(subtreeRoot->getInstruction()));
        unsigned ZeroReg = target.getRegInfo().getZeroRegNum();
        mvec.push_back(BuildMI(V9::XNOR, 3).addReg(notArg).addMReg(ZeroReg)
                                       .addRegDef(subtreeRoot->getValue()));
        break;
      }

      case 22:	// reg:   ToBoolTy(reg):
      {
        const Type* opType = subtreeRoot->leftChild()->getValue()->getType();
        assert(opType->isIntegral() || isa<PointerType>(opType));
        forwardOperandNum = 0;          // forward first operand to user
        break;
      }
      
      case 23:	// reg:   ToUByteTy(reg)
      case 24:	// reg:   ToSByteTy(reg)
      case 25:	// reg:   ToUShortTy(reg)
      case 26:	// reg:   ToShortTy(reg)
      case 27:	// reg:   ToUIntTy(reg)
      case 28:	// reg:   ToIntTy(reg)
      {
        //======================================================================
        // Rules for integer conversions:
        // 
        //--------
        // From ISO 1998 C++ Standard, Sec. 4.7:
        //
        // 2. If the destination type is unsigned, the resulting value is
        // the least unsigned integer congruent to the source integer
        // (modulo 2n where n is the number of bits used to represent the
        // unsigned type). [Note: In a two s complement representation,
        // this conversion is conceptual and there is no change in the
        // bit pattern (if there is no truncation). ]
        // 
        // 3. If the destination type is signed, the value is unchanged if
        // it can be represented in the destination type (and bitfield width);
        // otherwise, the value is implementation-defined.
        //--------
        // 
        // Since we assume 2s complement representations, this implies:
        // 
        // -- if operand is smaller than destination, zero-extend or sign-extend
        //    according to the signedness of the *operand*: source decides.
        //    ==> we have to do nothing here!
        // 
        // -- if operand is same size as or larger than destination, and the
        //    destination is *unsigned*, zero-extend the operand: dest. decides
        // 
        // -- if operand is same size as or larger than destination, and the
        //    destination is *signed*, the choice is implementation defined:
        //    we sign-extend the operand: i.e., again dest. decides.
        //    Note: this matches both Sun's cc and gcc3.2.
        //======================================================================

        Instruction* destI =  subtreeRoot->getInstruction();
        Value* opVal = subtreeRoot->leftChild()->getValue();
        const Type* opType = opVal->getType();
        if (opType->isIntegral() || isa<PointerType>(opType))
          {
            unsigned opSize = target.getTargetData().getTypeSize(opType);
            unsigned destSize = target.getTargetData().getTypeSize(destI->getType());
            if (opSize >= destSize)
              { // Operand is same size as or larger than dest:
                // zero- or sign-extend, according to the signeddness of
                // the destination (see above).
                if (destI->getType()->isSigned())
                  target.getInstrInfo().CreateSignExtensionInstructions(target,
                    destI->getParent()->getParent(), opVal, destI, 8*destSize,
                    mvec, MachineCodeForInstruction::get(destI));
                else
                  target.getInstrInfo().CreateZeroExtensionInstructions(target,
                    destI->getParent()->getParent(), opVal, destI, 8*destSize,
                    mvec, MachineCodeForInstruction::get(destI));
              }
            else
              forwardOperandNum = 0;          // forward first operand to user
          }
        else if (opType->isFloatingPoint())
          {
            CreateCodeToConvertFloatToInt(target, opVal, destI, mvec,
                                         MachineCodeForInstruction::get(destI));
            if (destI->getType()->isUnsigned())
              maskUnsignedResult = true; // not handled by fp->int code
          }
        else
          assert(0 && "Unrecognized operand type for convert-to-unsigned");

        break;
      }

      case 29:	// reg:   ToULongTy(reg)
      case 30:	// reg:   ToLongTy(reg)
      {
        Value* opVal = subtreeRoot->leftChild()->getValue();
        const Type* opType = opVal->getType();
        if (opType->isIntegral() || isa<PointerType>(opType))
          forwardOperandNum = 0;          // forward first operand to user
        else if (opType->isFloatingPoint())
          {
            Instruction* destI =  subtreeRoot->getInstruction();
            CreateCodeToConvertFloatToInt(target, opVal, destI, mvec,
                                         MachineCodeForInstruction::get(destI));
          }
        else
          assert(0 && "Unrecognized operand type for convert-to-signed");
        break;
      }
      
      case  31:	// reg:   ToFloatTy(reg):
      case  32:	// reg:   ToDoubleTy(reg):
      case 232:	// reg:   ToDoubleTy(Constant):
      
        // If this instruction has a parent (a user) in the tree 
        // and the user is translated as an FsMULd instruction,
        // then the cast is unnecessary.  So check that first.
        // In the future, we'll want to do the same for the FdMULq instruction,
        // so do the check here instead of only for ToFloatTy(reg).
        // 
        if (subtreeRoot->parent() != NULL)
          {
            const MachineCodeForInstruction& mcfi =
              MachineCodeForInstruction::get(
                cast<InstructionNode>(subtreeRoot->parent())->getInstruction());
            if (mcfi.size() == 0 || mcfi.front()->getOpCode() == V9::FSMULD)
              forwardOperandNum = 0;    // forward first operand to user
          }

        if (forwardOperandNum != 0)     // we do need the cast
          {
            Value* leftVal = subtreeRoot->leftChild()->getValue();
            const Type* opType = leftVal->getType();
            MachineOpCode opCode=ChooseConvertToFloatInstr(
                                       subtreeRoot->getOpLabel(), opType);
            if (opCode == V9::INVALID_OPCODE)	// no conversion needed
              {
                forwardOperandNum = 0;      // forward first operand to user
              }
            else
              {
                // If the source operand is a non-FP type it must be
                // first copied from int to float register via memory!
                Instruction *dest = subtreeRoot->getInstruction();
                Value* srcForCast;
                int n = 0;
                if (! opType->isFloatingPoint())
                  {
                    // Create a temporary to represent the FP register
                    // into which the integer will be copied via memory.
                    // The type of this temporary will determine the FP
                    // register used: single-prec for a 32-bit int or smaller,
                    // double-prec for a 64-bit int.
                    // 
                    uint64_t srcSize =
                      target.getTargetData().getTypeSize(leftVal->getType());
                    Type* tmpTypeToUse =
                      (srcSize <= 4)? Type::FloatTy : Type::DoubleTy;
                    srcForCast = new TmpInstruction(tmpTypeToUse, dest);
                    MachineCodeForInstruction &destMCFI = 
                      MachineCodeForInstruction::get(dest);
                    destMCFI.addTemp(srcForCast);

                    target.getInstrInfo().CreateCodeToCopyIntToFloat(target,
                         dest->getParent()->getParent(),
                         leftVal, cast<Instruction>(srcForCast),
                         mvec, destMCFI);
                  }
                else
                  srcForCast = leftVal;

                M = BuildMI(opCode, 2).addReg(srcForCast).addRegDef(dest);
                mvec.push_back(M);
              }
          }
        break;

      case 19:	// reg:   ToArrayTy(reg):
      case 20:	// reg:   ToPointerTy(reg):
        forwardOperandNum = 0;          // forward first operand to user
        break;

      case 233:	// reg:   Add(reg, Constant)
        maskUnsignedResult = true;
        M = CreateAddConstInstruction(subtreeRoot);
        if (M != NULL)
          {
            mvec.push_back(M);
            break;
          }
        // ELSE FALL THROUGH
        
      case 33:	// reg:   Add(reg, reg)
        maskUnsignedResult = true;
        Add3OperandInstr(ChooseAddInstruction(subtreeRoot), subtreeRoot, mvec);
        break;

      case 234:	// reg:   Sub(reg, Constant)
        maskUnsignedResult = true;
        M = CreateSubConstInstruction(subtreeRoot);
        if (M != NULL)
          {
            mvec.push_back(M);
            break;
          }
        // ELSE FALL THROUGH
        
      case 34:	// reg:   Sub(reg, reg)
        maskUnsignedResult = true;
        Add3OperandInstr(ChooseSubInstructionByType(
                                   subtreeRoot->getInstruction()->getType()),
                         subtreeRoot, mvec);
        break;

      case 135:	// reg:   Mul(todouble, todouble)
        checkCast = true;
        // FALL THROUGH 

      case 35:	// reg:   Mul(reg, reg)
      {
        maskUnsignedResult = true;
        MachineOpCode forceOp = ((checkCast && BothFloatToDouble(subtreeRoot))
                                 ? V9::FSMULD
                                 : INVALID_MACHINE_OPCODE);
        Instruction* mulInstr = subtreeRoot->getInstruction();
        CreateMulInstruction(target, mulInstr->getParent()->getParent(),
                             subtreeRoot->leftChild()->getValue(),
                             subtreeRoot->rightChild()->getValue(),
                             mulInstr, mvec,
                             MachineCodeForInstruction::get(mulInstr),forceOp);
        break;
      }
      case 335:	// reg:   Mul(todouble, todoubleConst)
        checkCast = true;
        // FALL THROUGH 

      case 235:	// reg:   Mul(reg, Constant)
      {
        maskUnsignedResult = true;
        MachineOpCode forceOp = ((checkCast && BothFloatToDouble(subtreeRoot))
                                 ? V9::FSMULD
                                 : INVALID_MACHINE_OPCODE);
        Instruction* mulInstr = subtreeRoot->getInstruction();
        CreateMulInstruction(target, mulInstr->getParent()->getParent(),
                             subtreeRoot->leftChild()->getValue(),
                             subtreeRoot->rightChild()->getValue(),
                             mulInstr, mvec,
                             MachineCodeForInstruction::get(mulInstr),
                             forceOp);
        break;
      }
      case 236:	// reg:   Div(reg, Constant)
        maskUnsignedResult = true;
        L = mvec.size();
        CreateDivConstInstruction(target, subtreeRoot, mvec);
        if (mvec.size() > L)
          break;
        // ELSE FALL THROUGH
      
      case 36:	// reg:   Div(reg, reg)
        maskUnsignedResult = true;
        Add3OperandInstr(ChooseDivInstruction(target, subtreeRoot),
                         subtreeRoot, mvec);
        break;

      case  37:	// reg:   Rem(reg, reg)
      case 237:	// reg:   Rem(reg, Constant)
      {
        maskUnsignedResult = true;
        Instruction* remInstr = subtreeRoot->getInstruction();
        
        TmpInstruction* quot = new TmpInstruction(
                                        subtreeRoot->leftChild()->getValue(),
                                        subtreeRoot->rightChild()->getValue());
        TmpInstruction* prod = new TmpInstruction(
                                        quot,
                                        subtreeRoot->rightChild()->getValue());
        MachineCodeForInstruction::get(remInstr).addTemp(quot).addTemp(prod); 
        
        M = BuildMI(ChooseDivInstruction(target, subtreeRoot), 3)
                             .addReg(subtreeRoot->leftChild()->getValue())
                             .addReg(subtreeRoot->rightChild()->getValue())
                             .addRegDef(quot);
        mvec.push_back(M);
        
        unsigned MulOpcode =
          ChooseMulInstructionByType(subtreeRoot->getInstruction()->getType());
        Value *MulRHS = subtreeRoot->rightChild()->getValue();
        M = BuildMI(MulOpcode, 3).addReg(quot).addReg(MulRHS).addReg(prod,
                                                                     MOTy::Def);
        mvec.push_back(M);
        
        unsigned Opcode = ChooseSubInstructionByType(
                                   subtreeRoot->getInstruction()->getType());
        M = BuildMI(Opcode, 3).addReg(subtreeRoot->leftChild()->getValue())
                              .addReg(prod).addRegDef(subtreeRoot->getValue());
        mvec.push_back(M);
        break;
      }
      
      case  38:	// bool:   And(bool, bool)
      case 238:	// bool:   And(bool, boolconst)
      case 338:	// reg :   BAnd(reg, reg)
      case 538:	// reg :   BAnd(reg, Constant)
        Add3OperandInstr(V9::AND, subtreeRoot, mvec);
        break;

      case 138:	// bool:   And(bool, not)
      case 438:	// bool:   BAnd(bool, bnot)
      { // Use the argument of NOT as the second argument!
        // Mark the NOT node so that no code is generated for it.
        InstructionNode* notNode = (InstructionNode*) subtreeRoot->rightChild();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(notNode->getInstruction()));
        notNode->markFoldedIntoParent();
        Value *LHS = subtreeRoot->leftChild()->getValue();
        Value *Dest = subtreeRoot->getValue();
        mvec.push_back(BuildMI(V9::ANDN, 3).addReg(LHS).addReg(notArg)
                                       .addReg(Dest, MOTy::Def));
        break;
      }

      case  39:	// bool:   Or(bool, bool)
      case 239:	// bool:   Or(bool, boolconst)
      case 339:	// reg :   BOr(reg, reg)
      case 539:	// reg :   BOr(reg, Constant)
        Add3OperandInstr(V9::OR, subtreeRoot, mvec);
        break;

      case 139:	// bool:   Or(bool, not)
      case 439:	// bool:   BOr(bool, bnot)
      { // Use the argument of NOT as the second argument!
        // Mark the NOT node so that no code is generated for it.
        InstructionNode* notNode = (InstructionNode*) subtreeRoot->rightChild();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(notNode->getInstruction()));
        notNode->markFoldedIntoParent();
        Value *LHS = subtreeRoot->leftChild()->getValue();
        Value *Dest = subtreeRoot->getValue();
        mvec.push_back(BuildMI(V9::ORN, 3).addReg(LHS).addReg(notArg)
                       .addReg(Dest, MOTy::Def));
        break;
      }

      case  40:	// bool:   Xor(bool, bool)
      case 240:	// bool:   Xor(bool, boolconst)
      case 340:	// reg :   BXor(reg, reg)
      case 540:	// reg :   BXor(reg, Constant)
        Add3OperandInstr(V9::XOR, subtreeRoot, mvec);
        break;

      case 140:	// bool:   Xor(bool, not)
      case 440:	// bool:   BXor(bool, bnot)
      { // Use the argument of NOT as the second argument!
        // Mark the NOT node so that no code is generated for it.
        InstructionNode* notNode = (InstructionNode*) subtreeRoot->rightChild();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(notNode->getInstruction()));
        notNode->markFoldedIntoParent();
        Value *LHS = subtreeRoot->leftChild()->getValue();
        Value *Dest = subtreeRoot->getValue();
        mvec.push_back(BuildMI(V9::XNOR, 3).addReg(LHS).addReg(notArg)
                       .addReg(Dest, MOTy::Def));
        break;
      }

      case 41:	// boolconst:   SetCC(reg, Constant)
        // 
        // If the SetCC was folded into the user (parent), it will be
        // caught above.  All other cases are the same as case 42,
        // so just fall through.
        // 
      case 42:	// bool:   SetCC(reg, reg):
      {
        // This generates a SUBCC instruction, putting the difference in
        // a result register, and setting a condition code.
        // 
        // If the boolean result of the SetCC is used by anything other
        // than a branch instruction, or if it is used outside the current
        // basic block, the boolean must be
        // computed and stored in the result register.  Otherwise, discard
        // the difference (by using %g0) and keep only the condition code.
        // 
        // To compute the boolean result in a register we use a conditional
        // move, unless the result of the SUBCC instruction can be used as
        // the bool!  This assumes that zero is FALSE and any non-zero
        // integer is TRUE.
        // 
        InstructionNode* parentNode = (InstructionNode*) subtreeRoot->parent();
        Instruction* setCCInstr = subtreeRoot->getInstruction();
        
        bool keepBoolVal = parentNode == NULL ||
                           ! AllUsesAreBranches(setCCInstr);
        bool subValIsBoolVal = setCCInstr->getOpcode() == Instruction::SetNE;
        bool keepSubVal = keepBoolVal && subValIsBoolVal;
        bool computeBoolVal = keepBoolVal && ! subValIsBoolVal;
        
        bool mustClearReg;
        int valueToMove;
        MachineOpCode movOpCode = 0;
        
        // Mark the 4th operand as being a CC register, and as a def
        // A TmpInstruction is created to represent the CC "result".
        // Unlike other instances of TmpInstruction, this one is used
        // by machine code of multiple LLVM instructions, viz.,
        // the SetCC and the branch.  Make sure to get the same one!
        // Note that we do this even for FP CC registers even though they
        // are explicit operands, because the type of the operand
        // needs to be a floating point condition code, not an integer
        // condition code.  Think of this as casting the bool result to
        // a FP condition code register.
        // 
        Value* leftVal = subtreeRoot->leftChild()->getValue();
        bool isFPCompare = leftVal->getType()->isFloatingPoint();
        
        TmpInstruction* tmpForCC = GetTmpForCC(setCCInstr,
                                     setCCInstr->getParent()->getParent(),
                                     isFPCompare ? Type::FloatTy : Type::IntTy);
        MachineCodeForInstruction::get(setCCInstr).addTemp(tmpForCC);
        
        if (! isFPCompare)
          {
            // Integer condition: dest. should be %g0 or an integer register.
            // If result must be saved but condition is not SetEQ then we need
            // a separate instruction to compute the bool result, so discard
            // result of SUBcc instruction anyway.
            // 
            if (keepSubVal) {
              M = BuildMI(V9::SUBcc, 4)
                .addReg(subtreeRoot->leftChild()->getValue())
                .addReg(subtreeRoot->rightChild()->getValue())
                .addRegDef(subtreeRoot->getValue())
                .addCCReg(tmpForCC, MOTy::Def);
            } else {
              M = BuildMI(V9::SUBcc, 4)
                .addReg(subtreeRoot->leftChild()->getValue())
                .addReg(subtreeRoot->rightChild()->getValue())
                .addMReg(target.getRegInfo().getZeroRegNum(), MOTy::Def)
                .addCCReg(tmpForCC, MOTy::Def);
            }
            mvec.push_back(M);
            
            if (computeBoolVal)
              { // recompute bool using the integer condition codes
                movOpCode =
                  ChooseMovpccAfterSub(subtreeRoot,mustClearReg,valueToMove);
              }
          }
        else
          {
            // FP condition: dest of FCMP should be some FCCn register
            M = BuildMI(ChooseFcmpInstruction(subtreeRoot), 3)
                         .addCCReg(tmpForCC, MOTy::Def)
                         .addReg(subtreeRoot->leftChild()->getValue())
                         .addRegDef(subtreeRoot->rightChild()->getValue());
            mvec.push_back(M);
            
            if (computeBoolVal)
              {// recompute bool using the FP condition codes
                mustClearReg = true;
                valueToMove = 1;
                movOpCode = ChooseMovFpccInstruction(subtreeRoot);
              }
          }
        
        if (computeBoolVal)
          {
            if (mustClearReg)
              {// Unconditionally set register to 0
                M = BuildMI(V9::SETHI, 2).addZImm(0).addRegDef(setCCInstr);
                mvec.push_back(M);
              }
            
            // Now conditionally move `valueToMove' (0 or 1) into the register
            // Mark the register as a use (as well as a def) because the old
            // value should be retained if the condition is false.
            M = BuildMI(movOpCode, 3).addCCReg(tmpForCC).addZImm(valueToMove)
                                     .addReg(setCCInstr, MOTy::UseAndDef);
            mvec.push_back(M);
          }
        break;
      }    

      case 51:	// reg:   Load(reg)
      case 52:	// reg:   Load(ptrreg)
        SetOperandsForMemInstr(ChooseLoadInstruction(
                                   subtreeRoot->getValue()->getType()),
                               mvec, subtreeRoot, target);
        break;

      case 55:	// reg:   GetElemPtr(reg)
      case 56:	// reg:   GetElemPtrIdx(reg,reg)
        // If the GetElemPtr was folded into the user (parent), it will be
        // caught above.  For other cases, we have to compute the address.
        SetOperandsForMemInstr(V9::ADD, mvec, subtreeRoot, target);
        break;

      case 57:	// reg:  Alloca: Implement as 1 instruction:
      {         //	    add %fp, offsetFromFP -> result
        AllocationInst* instr =
          cast<AllocationInst>(subtreeRoot->getInstruction());
        unsigned tsize =
          target.getTargetData().getTypeSize(instr->getAllocatedType());
        assert(tsize != 0);
        CreateCodeForFixedSizeAlloca(target, instr, tsize, 1, mvec);
        break;
      }

      case 58:	// reg:   Alloca(reg): Implement as 3 instructions:
                //	mul num, typeSz -> tmp
                //	sub %sp, tmp    -> %sp
      {         //	add %sp, frameSizeBelowDynamicArea -> result
        AllocationInst* instr =
          cast<AllocationInst>(subtreeRoot->getInstruction());
        const Type* eltType = instr->getAllocatedType();
        
        // If #elements is constant, use simpler code for fixed-size allocas
        int tsize = (int) target.getTargetData().getTypeSize(eltType);
        Value* numElementsVal = NULL;
        bool isArray = instr->isArrayAllocation();
        
        if (!isArray ||
            isa<Constant>(numElementsVal = instr->getArraySize()))
          { // total size is constant: generate code for fixed-size alloca
            unsigned numElements = isArray? 
              cast<ConstantUInt>(numElementsVal)->getValue() : 1;
            CreateCodeForFixedSizeAlloca(target, instr, tsize,
                                         numElements, mvec);
          }
        else // total size is not constant.
          CreateCodeForVariableSizeAlloca(target, instr, tsize,
                                          numElementsVal, mvec);
        break;
      }

      case 61:	// reg:   Call
      {         // Generate a direct (CALL) or indirect (JMPL) call.
                // Mark the return-address register, the indirection
                // register (for indirect calls), the operands of the Call,
                // and the return value (if any) as implicit operands
                // of the machine instruction.
                // 
                // If this is a varargs function, floating point arguments
                // have to passed in integer registers so insert
                // copy-float-to-int instructions for each float operand.
                // 
        CallInst *callInstr = cast<CallInst>(subtreeRoot->getInstruction());
        Value *callee = callInstr->getCalledValue();
        Function* calledFunc = dyn_cast<Function>(callee);

        // Check if this is an intrinsic function that needs a special code
        // sequence (e.g., va_start).  Indirect calls cannot be special.
        // 
        bool specialIntrinsic = false;
        LLVMIntrinsic::ID iid;
        if (calledFunc && (iid=(LLVMIntrinsic::ID)calledFunc->getIntrinsicID()))
          specialIntrinsic = CodeGenIntrinsic(iid, *callInstr, target, mvec);

        // If not, generate the normal call sequence for the function.
        // This can also handle any intrinsics that are just function calls.
        // 
        if (! specialIntrinsic)
          {
            // Create hidden virtual register for return address with type void*
            TmpInstruction* retAddrReg =
              new TmpInstruction(PointerType::get(Type::VoidTy), callInstr);
            MachineCodeForInstruction::get(callInstr).addTemp(retAddrReg);

            // Generate the machine instruction and its operands.
            // Use CALL for direct function calls; this optimistically assumes
            // the PC-relative address fits in the CALL address field (22 bits).
            // Use JMPL for indirect calls.
            // 
            if (calledFunc)             // direct function call
              M = BuildMI(V9::CALL, 1).addPCDisp(callee);
            else                        // indirect function call
              M = BuildMI(V9::JMPLCALL, 3).addReg(callee).addSImm((int64_t)0)
                .addRegDef(retAddrReg);
            mvec.push_back(M);

            const FunctionType* funcType =
              cast<FunctionType>(cast<PointerType>(callee->getType())
                                 ->getElementType());
            bool isVarArgs = funcType->isVarArg();
            bool noPrototype = isVarArgs && funcType->getNumParams() == 0;
        
            // Use a descriptor to pass information about call arguments
            // to the register allocator.  This descriptor will be "owned"
            // and freed automatically when the MachineCodeForInstruction
            // object for the callInstr goes away.
            CallArgsDescriptor* argDesc = new CallArgsDescriptor(callInstr,
                                             retAddrReg, isVarArgs,noPrototype);
            
            assert(callInstr->getOperand(0) == callee
                   && "This is assumed in the loop below!");
            
            for (unsigned i=1, N=callInstr->getNumOperands(); i < N; ++i)
              {
                Value* argVal = callInstr->getOperand(i);
                Instruction* intArgReg = NULL;
            
                // Check for FP arguments to varargs functions.
                // Any such argument in the first $K$ args must be passed in an
                // integer register, where K = #integer argument registers.
                if (isVarArgs && argVal->getType()->isFloatingPoint())
                  {
                    // If it is a function with no prototype, pass value
                    // as an FP value as well as a varargs value
                    if (noPrototype)
                      argDesc->getArgInfo(i-1).setUseFPArgReg();
                
                    // If this arg. is in the first $K$ regs, add a copy
                    // float-to-int instruction to pass the value as an integer.
                    if (i <= target.getRegInfo().getNumOfIntArgRegs())
                      {
                        MachineCodeForInstruction &destMCFI = 
                          MachineCodeForInstruction::get(callInstr);   
                        intArgReg = new TmpInstruction(Type::IntTy, argVal);
                        destMCFI.addTemp(intArgReg);
                    
                        std::vector<MachineInstr*> copyMvec;
                        target.getInstrInfo().CreateCodeToCopyFloatToInt(target,
                                         callInstr->getParent()->getParent(),
                                         argVal, (TmpInstruction*) intArgReg,
                                         copyMvec, destMCFI);
                        mvec.insert(mvec.begin(),copyMvec.begin(),copyMvec.end());
                    
                        argDesc->getArgInfo(i-1).setUseIntArgReg();
                        argDesc->getArgInfo(i-1).setArgCopy(intArgReg);
                      }
                    else
                      // Cannot fit in first $K$ regs so pass arg on stack
                      argDesc->getArgInfo(i-1).setUseStackSlot();
                  }
            
                if (intArgReg)
                  mvec.back()->addImplicitRef(intArgReg);
            
                mvec.back()->addImplicitRef(argVal);
              }
        
            // Add the return value as an implicit ref.  The call operands
            // were added above.
            if (callInstr->getType() != Type::VoidTy)
              mvec.back()->addImplicitRef(callInstr, /*isDef*/ true);
        
            // For the CALL instruction, the ret. addr. reg. is also implicit
            if (isa<Function>(callee))
              mvec.back()->addImplicitRef(retAddrReg, /*isDef*/ true);
        
            // delay slot
            mvec.push_back(BuildMI(V9::NOP, 0));
          }

        break;
      }
      
      case 62:	// reg:   Shl(reg, reg)
      {
        Value* argVal1 = subtreeRoot->leftChild()->getValue();
        Value* argVal2 = subtreeRoot->rightChild()->getValue();
        Instruction* shlInstr = subtreeRoot->getInstruction();
        
        const Type* opType = argVal1->getType();
        assert((opType->isInteger() || isa<PointerType>(opType)) &&
               "Shl unsupported for other types");
        
        CreateShiftInstructions(target, shlInstr->getParent()->getParent(),
                                (opType == Type::LongTy)? V9::SLLX : V9::SLL,
                                argVal1, argVal2, 0, shlInstr, mvec,
                                MachineCodeForInstruction::get(shlInstr));
        break;
      }
      
      case 63:	// reg:   Shr(reg, reg)
      { const Type* opType = subtreeRoot->leftChild()->getValue()->getType();
        assert((opType->isInteger() || isa<PointerType>(opType)) &&
               "Shr unsupported for other types");
        Add3OperandInstr(opType->isSigned()
                               ? (opType == Type::LongTy ? V9::SRAX : V9::SRA)
                               : (opType == Type::LongTy ? V9::SRLX : V9::SRL),
                         subtreeRoot, mvec);
        break;
      }
      
      case 64:	// reg:   Phi(reg,reg)
        break;                          // don't forward the value

      case 65:	// reg:   VaArg(reg)
      {
        // Use value initialized by va_start as pointer to args on the stack.
        // Load argument via current pointer value, then increment pointer.
        int argSize = target.getFrameInfo().getSizeOfEachArgOnStack();
        Instruction* vaArgI = subtreeRoot->getInstruction();
        mvec.push_back(BuildMI(V9::LDX, 3).addReg(vaArgI->getOperand(0)).
                       addSImm(0).addRegDef(vaArgI));
        mvec.push_back(BuildMI(V9::ADD, 3).addReg(vaArgI->getOperand(0)).
                       addSImm(argSize).addRegDef(vaArgI->getOperand(0)));
        break;
      }
      
      case 71:	// reg:     VReg
      case 72:	// reg:     Constant
        break;                          // don't forward the value

      default:
        assert(0 && "Unrecognized BURG rule");
        break;
      }
    }

  if (forwardOperandNum >= 0)
    { // We did not generate a machine instruction but need to use operand.
      // If user is in the same tree, replace Value in its machine operand.
      // If not, insert a copy instruction which should get coalesced away
      // by register allocation.
      if (subtreeRoot->parent() != NULL)
        ForwardOperand(subtreeRoot, subtreeRoot->parent(), forwardOperandNum);
      else
        {
          std::vector<MachineInstr*> minstrVec;
          Instruction* instr = subtreeRoot->getInstruction();
          target.getInstrInfo().
            CreateCopyInstructionsByType(target,
                                         instr->getParent()->getParent(),
                                         instr->getOperand(forwardOperandNum),
                                         instr, minstrVec,
                                        MachineCodeForInstruction::get(instr));
          assert(minstrVec.size() > 0);
          mvec.insert(mvec.end(), minstrVec.begin(), minstrVec.end());
        }
    }

  if (maskUnsignedResult)
    { // If result is unsigned and smaller than int reg size,
      // we need to clear high bits of result value.
      assert(forwardOperandNum < 0 && "Need mask but no instruction generated");
      Instruction* dest = subtreeRoot->getInstruction();
      if (dest->getType()->isUnsigned())
        {
          unsigned destSize=target.getTargetData().getTypeSize(dest->getType());
          if (destSize <= 4)
            { // Mask high bits.  Use a TmpInstruction to represent the
              // intermediate result before masking.  Since those instructions
              // have already been generated, go back and substitute tmpI
              // for dest in the result position of each one of them.
              TmpInstruction *tmpI = new TmpInstruction(dest->getType(), dest,
                                                        NULL, "maskHi");
              MachineCodeForInstruction::get(dest).addTemp(tmpI);

              for (unsigned i=0, N=mvec.size(); i < N; ++i)
                mvec[i]->substituteValue(dest, tmpI);

              M = BuildMI(V9::SRL, 3).addReg(tmpI).addZImm(8*(4-destSize))
                .addReg(dest, MOTy::Def);
              mvec.push_back(M);
            }
          else if (destSize < 8)
            assert(0 && "Unsupported type size: 32 < size < 64 bits");
        }
    }
}
