//===- SparcV9BurgISel.cpp - SparcV9 BURG-based Instruction Selector ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SparcV9 BURG-based instruction selector. It uses the SSA graph to
// construct a forest of BURG instruction trees (class InstrForest) and then
// uses the BURG-generated tree grammar (BURM) to find the optimal instruction
// sequences for the SparcV9.
//
//===----------------------------------------------------------------------===//

#include "MachineInstrAnnot.h"
#include "SparcV9BurgISel.h"
#include "SparcV9InstrForest.h"
#include "SparcV9Internals.h"
#include "SparcV9TmpInstr.h"
#include "SparcV9FrameInfo.h"
#include "SparcV9RegisterInfo.h"
#include "MachineFunctionInfo.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Type.h"
#include "llvm/Config/alloca.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/hash_map"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace llvm;

//==------------------------------------------------------------------------==//
//          InstrForest (V9ISel BURG instruction trees) implementation
//==------------------------------------------------------------------------==//

namespace llvm {

class InstructionNode : public InstrTreeNode {
  bool codeIsFoldedIntoParent;

public:
  InstructionNode(Instruction *_instr);

  Instruction *getInstruction() const {
    assert(treeNodeType == NTInstructionNode);
    return cast<Instruction>(val);
  }

  void markFoldedIntoParent() { codeIsFoldedIntoParent = true; }
  bool isFoldedIntoParent()   { return codeIsFoldedIntoParent; }

  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InstructionNode *N) { return true; }
  static inline bool classof(const InstrTreeNode *N) {
    return N->getNodeType() == InstrTreeNode::NTInstructionNode;
  }

protected:
  virtual void dumpNode(int indent) const;
};

class VRegListNode : public InstrTreeNode {
public:
  VRegListNode() : InstrTreeNode(NTVRegListNode, 0) { opLabel = VRegListOp; }
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VRegListNode  *N) { return true; }
  static inline bool classof(const InstrTreeNode *N) {
    return N->getNodeType() == InstrTreeNode::NTVRegListNode;
  }
protected:
  virtual void dumpNode(int indent) const;
};

class VRegNode : public InstrTreeNode {
public:
  VRegNode(Value* _val) : InstrTreeNode(NTVRegNode, _val) {
    opLabel = VRegNodeOp;
  }
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VRegNode  *N) { return true; }
  static inline bool classof(const InstrTreeNode *N) {
    return N->getNodeType() == InstrTreeNode::NTVRegNode;
  }
protected:
  virtual void dumpNode(int indent) const;
};

class ConstantNode : public InstrTreeNode {
public:
  ConstantNode(Constant *constVal)
    : InstrTreeNode(NTConstNode, (Value*)constVal) {
    opLabel = ConstantNodeOp;
  }
  Constant *getConstVal() const { return (Constant*) val;}
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantNode  *N) { return true; }
  static inline bool classof(const InstrTreeNode *N) {
    return N->getNodeType() == InstrTreeNode::NTConstNode;
  }
protected:
  virtual void dumpNode(int indent) const;
};

class LabelNode : public InstrTreeNode {
public:
  LabelNode(BasicBlock* BB) : InstrTreeNode(NTLabelNode, (Value*)BB) {
    opLabel = LabelNodeOp;
  }
  BasicBlock *getBasicBlock() const { return (BasicBlock*)val;}
  // Methods to support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const LabelNode     *N) { return true; }
  static inline bool classof(const InstrTreeNode *N) {
    return N->getNodeType() == InstrTreeNode::NTLabelNode;
  }
protected:
  virtual void dumpNode(int indent) const;
};

/// InstrForest -  A forest of instruction trees for a single function.
/// The goal of InstrForest is to group instructions into a single
/// tree if one or more of them might be potentially combined into a
/// single complex instruction in the target machine. We group two
/// instructions O and I if: (1) Instruction O computes an operand used
/// by instruction I, and (2) O and I are part of the same basic block,
/// and (3) O has only a single use, viz., I.
///
class InstrForest : private hash_map<const Instruction *, InstructionNode*> {
public:
  // Use a vector for the root set to get a deterministic iterator
  // for stable code generation.  Even though we need to erase nodes
  // during forest construction, a vector should still be efficient
  // because the elements to erase are nearly always near the end.
  typedef std::vector<InstructionNode*> RootSet;
  typedef RootSet::      iterator       root_iterator;
  typedef RootSet::const_iterator const_root_iterator;

private:
  RootSet treeRoots;

public:
  /*ctor*/      InstrForest     (Function *F);
  /*dtor*/      ~InstrForest    ();

  /// getTreeNodeForInstr - Returns the tree node for an Instruction.
  ///
  inline InstructionNode *getTreeNodeForInstr(Instruction* instr) {
    return (*this)[instr];
  }

  /// Iterators for the root nodes for all the trees.
  ///
  const_root_iterator roots_begin() const     { return treeRoots.begin(); }
        root_iterator roots_begin()           { return treeRoots.begin(); }
  const_root_iterator roots_end  () const     { return treeRoots.end();   }
        root_iterator roots_end  ()           { return treeRoots.end();   }

  void dump() const;

private:
  // Methods used to build the instruction forest.
  void eraseRoot    (InstructionNode* node);
  void setLeftChild (InstrTreeNode* parent, InstrTreeNode* child);
  void setRightChild(InstrTreeNode* parent, InstrTreeNode* child);
  void setParent    (InstrTreeNode* child,  InstrTreeNode* parent);
  void noteTreeNodeForInstr(Instruction* instr, InstructionNode* treeNode);
  InstructionNode* buildTreeForInstruction(Instruction* instr);
};

void InstrTreeNode::dump(int dumpChildren, int indent) const {
  dumpNode(indent);

  if (dumpChildren) {
    if (LeftChild)
      LeftChild->dump(dumpChildren, indent+1);
    if (RightChild)
      RightChild->dump(dumpChildren, indent+1);
  }
}

InstructionNode::InstructionNode(Instruction* I)
  : InstrTreeNode(NTInstructionNode, I), codeIsFoldedIntoParent(false) {
  opLabel = I->getOpcode();

  // Distinguish special cases of some instructions such as Ret and Br
  //
  if (opLabel == Instruction::Ret && cast<ReturnInst>(I)->getReturnValue()) {
    opLabel = RetValueOp;                // ret(value) operation
  }
  else if (opLabel ==Instruction::Br && !cast<BranchInst>(I)->isUnconditional())
  {
    opLabel = BrCondOp;         // br(cond) operation
  } else if (opLabel >= Instruction::SetEQ && opLabel <= Instruction::SetGT) {
    opLabel = SetCCOp;          // common label for all SetCC ops
  } else if (opLabel == Instruction::Alloca && I->getNumOperands() > 0) {
    opLabel = AllocaN;           // Alloca(ptr, N) operation
  } else if (opLabel == Instruction::GetElementPtr &&
             cast<GetElementPtrInst>(I)->hasIndices()) {
    opLabel = opLabel + 100;             // getElem with index vector
  } else if (opLabel == Instruction::Xor &&
             BinaryOperator::isNot(I)) {
    opLabel = (I->getType() == Type::BoolTy)?  NotOp  // boolean Not operator
      : BNotOp; // bitwise Not operator
  } else if (opLabel == Instruction::And || opLabel == Instruction::Or ||
             opLabel == Instruction::Xor) {
    // Distinguish bitwise operators from logical operators!
    if (I->getType() != Type::BoolTy)
      opLabel = opLabel + 100;   // bitwise operator
  } else if (opLabel == Instruction::Cast) {
    const Type *ITy = I->getType();
    switch(ITy->getTypeID())
    {
    case Type::BoolTyID:    opLabel = ToBoolTy;    break;
    case Type::UByteTyID:   opLabel = ToUByteTy;   break;
    case Type::SByteTyID:   opLabel = ToSByteTy;   break;
    case Type::UShortTyID:  opLabel = ToUShortTy;  break;
    case Type::ShortTyID:   opLabel = ToShortTy;   break;
    case Type::UIntTyID:    opLabel = ToUIntTy;    break;
    case Type::IntTyID:     opLabel = ToIntTy;     break;
    case Type::ULongTyID:   opLabel = ToULongTy;   break;
    case Type::LongTyID:    opLabel = ToLongTy;    break;
    case Type::FloatTyID:   opLabel = ToFloatTy;   break;
    case Type::DoubleTyID:  opLabel = ToDoubleTy;  break;
    case Type::ArrayTyID:   opLabel = ToArrayTy;   break;
    case Type::PointerTyID: opLabel = ToPointerTy; break;
    default:
      // Just use `Cast' opcode otherwise. It's probably ignored.
      break;
    }
  }
}

void InstructionNode::dumpNode(int indent) const {
  for (int i=0; i < indent; i++)
    std::cerr << "    ";
  std::cerr << getInstruction()->getOpcodeName()
            << " [label " << getOpLabel() << "]" << "\n";
}

void VRegListNode::dumpNode(int indent) const {
  for (int i=0; i < indent; i++)
    std::cerr << "    ";

  std::cerr << "List" << "\n";
}

void VRegNode::dumpNode(int indent) const {
  for (int i=0; i < indent; i++)
    std::cerr << "    ";
    std::cerr << "VReg " << *getValue() << "\n";
}

void ConstantNode::dumpNode(int indent) const {
  for (int i=0; i < indent; i++)
    std::cerr << "    ";
  std::cerr << "Constant " << *getValue() << "\n";
}

void LabelNode::dumpNode(int indent) const {
  for (int i=0; i < indent; i++)
    std::cerr << "    ";

  std::cerr << "Label " << *getValue() << "\n";
}

/// InstrForest ctor - Create a forest of instruction trees for a
/// single function.
///
InstrForest::InstrForest(Function *F) {
  for (Function::iterator BB = F->begin(), FE = F->end(); BB != FE; ++BB) {
    for(BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      buildTreeForInstruction(I);
  }
}

InstrForest::~InstrForest() {
  for_each(treeRoots.begin(), treeRoots.end(), deleter<InstructionNode>);
}

void InstrForest::dump() const {
  for (const_root_iterator I = roots_begin(); I != roots_end(); ++I)
    (*I)->dump(/*dumpChildren*/ 1, /*indent*/ 0);
}

inline void InstrForest::eraseRoot(InstructionNode* node) {
  for (RootSet::reverse_iterator RI=treeRoots.rbegin(), RE=treeRoots.rend();
       RI != RE; ++RI)
    if (*RI == node)
      treeRoots.erase(RI.base()-1);
}

inline void InstrForest::noteTreeNodeForInstr(Instruction *instr,
                                              InstructionNode *treeNode) {
  (*this)[instr] = treeNode;
  treeRoots.push_back(treeNode);        // mark node as root of a new tree
}

inline void InstrForest::setLeftChild(InstrTreeNode *parent,
                                      InstrTreeNode *child) {
  parent->LeftChild = child;
  child->Parent = parent;
  if (InstructionNode* instrNode = dyn_cast<InstructionNode>(child))
    eraseRoot(instrNode); // no longer a tree root
}

inline void InstrForest::setRightChild(InstrTreeNode *parent,
                                       InstrTreeNode *child) {
  parent->RightChild = child;
  child->Parent = parent;
  if (InstructionNode* instrNode = dyn_cast<InstructionNode>(child))
    eraseRoot(instrNode); // no longer a tree root
}

InstructionNode* InstrForest::buildTreeForInstruction(Instruction *instr) {
  InstructionNode *treeNode = getTreeNodeForInstr(instr);
  if (treeNode) {
    // treeNode has already been constructed for this instruction
    assert(treeNode->getInstruction() == instr);
    return treeNode;
  }

  // Otherwise, create a new tree node for this instruction.
  treeNode = new InstructionNode(instr);
  noteTreeNodeForInstr(instr, treeNode);

  if (instr->getOpcode() == Instruction::Call) {
    // Operands of call instruction
    return treeNode;
  }

  // If the instruction has more than 2 instruction operands,
  // then we need to create artificial list nodes to hold them.
  // (Note that we only count operands that get tree nodes, and not
  // others such as branch labels for a branch or switch instruction.)
  // To do this efficiently, we'll walk all operands, build treeNodes
  // for all appropriate operands and save them in an array.  We then
  // insert children at the end, creating list nodes where needed.
  // As a performance optimization, allocate a child array only
  // if a fixed array is too small.
  int numChildren = 0;
  InstrTreeNode** childArray = new InstrTreeNode*[instr->getNumOperands()];

  // Walk the operands of the instruction
  for (Instruction::op_iterator O = instr->op_begin(); O!=instr->op_end();
       ++O) {
      Value* operand = *O;

      // Check if the operand is a data value, not an branch label, type,
      // method or module.  If the operand is an address type (i.e., label
      // or method) that is used in an non-branching operation, e.g., `add'.
      // that should be considered a data value.
      // Check latter condition here just to simplify the next IF.
      bool includeAddressOperand =
        (isa<BasicBlock>(operand) || isa<Function>(operand))
        && !instr->isTerminator();

      if (includeAddressOperand || isa<Instruction>(operand) ||
          isa<Constant>(operand) || isa<Argument>(operand)) {
        // This operand is a data value.
        // An instruction that computes the incoming value is added as a
        // child of the current instruction if:
        //   the value has only a single use
        //   AND both instructions are in the same basic block.
        //   AND the current instruction is not a PHI (because the incoming
        //              value is conceptually in a predecessor block,
        //              even though it may be in the same static block)
        // (Note that if the value has only a single use (viz., `instr'),
        //  the def of the value can be safely moved just before instr
        //  and therefore it is safe to combine these two instructions.)
        // In all other cases, the virtual register holding the value
        // is used directly, i.e., made a child of the instruction node.
        InstrTreeNode* opTreeNode;
        if (isa<Instruction>(operand) && operand->hasOneUse() &&
            cast<Instruction>(operand)->getParent() == instr->getParent() &&
            instr->getOpcode() != Instruction::PHI &&
            instr->getOpcode() != Instruction::Call) {
          // Recursively create a treeNode for it.
          opTreeNode = buildTreeForInstruction((Instruction*)operand);
        } else if (Constant *CPV = dyn_cast<Constant>(operand)) {
          if (isa<GlobalValue>(CPV))
            opTreeNode = new VRegNode(operand);
          else if (isa<UndefValue>(CPV)) {
            opTreeNode = new
               ConstantNode(Constant::getNullValue(CPV->getType()));
          } else {
            // Create a leaf node for a constant
            opTreeNode = new ConstantNode(CPV);
          }
        } else {
          // Create a leaf node for the virtual register
          opTreeNode = new VRegNode(operand);
        }

        childArray[numChildren++] = opTreeNode;
      }
    }

  // Add any selected operands as children in the tree.
  // Certain instructions can have more than 2 in some instances (viz.,
  // a CALL or a memory access -- LOAD, STORE, and GetElemPtr -- to an
  // array or struct). Make the operands of every such instruction into
  // a right-leaning binary tree with the operand nodes at the leaves
  // and VRegList nodes as internal nodes.
  InstrTreeNode *parent = treeNode;

  if (numChildren > 2) {
    unsigned instrOpcode = treeNode->getInstruction()->getOpcode();
    assert(instrOpcode == Instruction::PHI ||
           instrOpcode == Instruction::Call ||
           instrOpcode == Instruction::Load ||
           instrOpcode == Instruction::Store ||
           instrOpcode == Instruction::GetElementPtr);
  }

  // Insert the first child as a direct child
  if (numChildren >= 1)
    setLeftChild(parent, childArray[0]);

  int n;

  // Create a list node for children 2 .. N-1, if any
  for (n = numChildren-1; n >= 2; n--) {
    // We have more than two children
    InstrTreeNode *listNode = new VRegListNode();
    setRightChild(parent, listNode);
    setLeftChild(listNode, childArray[numChildren - n]);
    parent = listNode;
  }

  // Now insert the last remaining child (if any).
  if (numChildren >= 2) {
    assert(n == 1);
    setRightChild(parent, childArray[numChildren - 1]);
  }

  delete [] childArray;
  return treeNode;
}
//==------------------------------------------------------------------------==//
//                V9ISel Command-line options and declarations
//==------------------------------------------------------------------------==//

namespace {
  /// Allow the user to select the amount of debugging information printed
  /// out by V9ISel.
  ///
  enum SelectDebugLevel_t {
    Select_NoDebugInfo,
    Select_PrintMachineCode,
    Select_DebugInstTrees,
    Select_DebugBurgTrees,
  };
  cl::opt<SelectDebugLevel_t>
  SelectDebugLevel("dselect", cl::Hidden,
                   cl::desc("enable instruction selection debug information"),
                   cl::values(
     clEnumValN(Select_NoDebugInfo,      "n", "disable debug output"),
     clEnumValN(Select_PrintMachineCode, "y", "print generated machine code"),
     clEnumValN(Select_DebugInstTrees,   "i",
                "print debugging info for instruction selection"),
     clEnumValN(Select_DebugBurgTrees,   "b", "print burg trees"),
                              clEnumValEnd));


  /// V9ISel - This is the FunctionPass that drives the instruction selection
  /// process on the SparcV9 target.
  ///
  class V9ISel : public FunctionPass {
    TargetMachine &Target;
    void InsertCodeForPhis(Function &F);
    void InsertPhiElimInstructions(BasicBlock *BB,
                                   const std::vector<MachineInstr*>& CpVec);
    void SelectInstructionsForTree(InstrTreeNode* treeRoot, int goalnt);
    void PostprocessMachineCodeForTree(InstructionNode* instrNode,
                                       int ruleForNode, short* nts);
  public:
    V9ISel(TargetMachine &TM) : Target(TM) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }

    bool runOnFunction(Function &F);
    virtual const char *getPassName() const {
      return "SparcV9 BURG Instruction Selector";
    }
  };
}


//==------------------------------------------------------------------------==//
//                     Various V9ISel helper functions
//==------------------------------------------------------------------------==//

static const uint32_t MAXLO   = (1 << 10) - 1; // set bits set by %lo(*)
static const uint32_t MAXSIMM = (1 << 12) - 1; // set bits in simm13 field of OR

/// ConvertConstantToIntType - Function to get the value of an integral
/// constant in the form that must be put into the machine register.  The
/// specified constant is interpreted as (i.e., converted if necessary to) the
/// specified destination type.  The result is always returned as an uint64_t,
/// since the representation of int64_t and uint64_t are identical.  The
/// argument can be any known const.  isValidConstant is set to true if a valid
/// constant was found.
///
uint64_t ConvertConstantToIntType(const TargetMachine &target, const Value *V,
                                  const Type *destType, bool &isValidConstant) {
  isValidConstant = false;
  uint64_t C = 0;

  if (! destType->isIntegral() && ! isa<PointerType>(destType))
    return C;

  if (! isa<Constant>(V) || isa<GlobalValue>(V))
    return C;

  // GlobalValue: no conversions needed: get value and return it
  if (const GlobalValue* GV = dyn_cast<GlobalValue>(V)) {
    isValidConstant = true;             // may be overwritten by recursive call
    return ConvertConstantToIntType(target, GV, destType, isValidConstant);
  }

  // ConstantBool: no conversions needed: get value and return it
  if (const ConstantBool *CB = dyn_cast<ConstantBool>(V)) {
    isValidConstant = true;
    return (uint64_t) CB->getValue();
  }

  // ConstantPointerNull: it's really just a big, shiny version of zero.
  if (isa<ConstantPointerNull>(V)) {
    isValidConstant = true;
    return 0;
  }

  // For other types of constants, some conversion may be needed.
  // First, extract the constant operand according to its own type
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    switch(CE->getOpcode()) {
    case Instruction::Cast:             // recursively get the value as cast
      C = ConvertConstantToIntType(target, CE->getOperand(0), CE->getType(),
                                   isValidConstant);
      break;
    default:                            // not simplifying other ConstantExprs
      break;
    }
  else if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    isValidConstant = true;
    C = CI->getRawValue();
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(V)) {
    isValidConstant = true;
    double fC = CFP->getValue();
    C = (destType->isSigned()? (uint64_t) (int64_t) fC
                             : (uint64_t)           fC);
  } else if (isa<UndefValue>(V)) {
    isValidConstant = true;
    C = 0;
  }

  // Now if a valid value was found, convert it to destType.
  if (isValidConstant) {
    unsigned opSize   = target.getTargetData().getTypeSize(V->getType());
    unsigned destSize = target.getTargetData().getTypeSize(destType);
    uint64_t maskHi   = (destSize < 8)? (1U << 8*destSize) - 1 : ~0;
    assert(opSize <= 8 && destSize <= 8 && ">8-byte int type unexpected");

    if (destType->isSigned()) {
      if (opSize > destSize)            // operand is larger than dest:
        C = C & maskHi;                 // mask high bits

      if (opSize > destSize ||
          (opSize == destSize && ! V->getType()->isSigned()))
        if (C & (1U << (8*destSize - 1)))
          C =  C | ~maskHi;             // sign-extend from destSize to 64 bits
    }
    else {
      if (opSize > destSize || (V->getType()->isSigned() && destSize < 8)) {
        // operand is larger than dest,
        //    OR both are equal but smaller than the full register size
        //       AND operand is signed, so it may have extra sign bits:
        // mask high bits
        C = C & maskHi;
      }
    }
  }

  return C;
}

/// CreateSETUWConst - Copy a 32-bit unsigned constant into the register
/// `dest', using SETHI, OR in the worst case.  This function correctly emulates
/// the SETUW pseudo-op for SPARC v9 (if argument isSigned == false). The
/// isSigned=true case is used to implement SETSW without duplicating code. It
/// optimizes some common cases:
/// (1) Small value that fits in simm13 field of OR: don't need SETHI.
/// (2) isSigned = true and C is a small negative signed value, i.e.,
///     high bits are 1, and the remaining bits fit in simm13(OR).
static inline void
CreateSETUWConst(uint32_t C,
                 Instruction* dest, std::vector<MachineInstr*>& mvec,
                 MachineCodeForInstruction& mcfi, Value* val, bool isSigned = false) {
  MachineInstr *miSETHI = NULL, *miOR = NULL;

  // In order to get efficient code, we should not generate the SETHI if
  // all high bits are 1 (i.e., this is a small signed value that fits in
  // the simm13 field of OR).  So we check for and handle that case specially.
  // NOTE: The value C = 0x80000000 is bad: sC < 0 *and* -sC < 0.
  //       In fact, sC == -sC, so we have to check for this explicitly.
  int32_t sC = (int32_t) C;
  bool smallNegValue =isSigned && sC < 0 && sC != -sC && -sC < (int32_t)MAXSIMM;

  //Create TmpInstruction for intermediate values
  TmpInstruction *tmpReg = 0;

  // Set the high 22 bits in dest if non-zero and simm13 field of OR not enough
  if (!smallNegValue && (C & ~MAXLO) && C > MAXSIMM) {
    tmpReg = new TmpInstruction(mcfi, PointerType::get(val->getType()), (Instruction*) val);
    miSETHI = BuildMI(V9::SETHI, 2).addZImm(C).addRegDef(tmpReg);
    miSETHI->getOperand(0).markHi32();
    mvec.push_back(miSETHI);
  }

  // Set the low 10 or 12 bits in dest.  This is necessary if no SETHI
  // was generated, or if the low 10 bits are non-zero.
  if (miSETHI==NULL || C & MAXLO) {
    if (miSETHI) {
      // unsigned value with high-order bits set using SETHI
      miOR = BuildMI(V9::ORi,3).addReg(tmpReg).addZImm(C).addRegDef(dest);
      miOR->getOperand(1).markLo32();
    } else {
      // unsigned or small signed value that fits in simm13 field of OR
      assert(smallNegValue || (C & ~MAXSIMM) == 0);
      miOR = BuildMI(V9::ORi, 3).addMReg(SparcV9::g0)
        .addSImm(sC).addRegDef(dest);
    }
    mvec.push_back(miOR);
  }
  else
    mvec.push_back(BuildMI(V9::ORr,3).addReg(tmpReg).addMReg(SparcV9::g0).addRegDef(dest));

  assert((miSETHI || miOR) && "Oops, no code was generated!");
}

/// CreateSETSWConst - Set a 32-bit signed constant in the register `dest',
/// with sign-extension to 64 bits.  This uses SETHI, OR, SRA in the worst case.
/// This function correctly emulates the SETSW pseudo-op for SPARC v9.  It
/// optimizes the same cases as SETUWConst, plus:
/// (1) SRA is not needed for positive or small negative values.
///
static inline void
CreateSETSWConst(int32_t C,
                 Instruction* dest, std::vector<MachineInstr*>& mvec,
                 MachineCodeForInstruction& mcfi, Value* val) {

  //TmpInstruction for intermediate values
  TmpInstruction *tmpReg = new TmpInstruction(mcfi, (Instruction*) val);

  // Set the low 32 bits of dest
  CreateSETUWConst((uint32_t) C,  tmpReg, mvec, mcfi, val, /*isSigned*/true);

  // Sign-extend to the high 32 bits if needed.
  // NOTE: The value C = 0x80000000 is bad: -C == C and so -C is < MAXSIMM
  if (C < 0 && (C == -C || -C > (int32_t) MAXSIMM))
    mvec.push_back(BuildMI(V9::SRAi5,3).addReg(tmpReg).addZImm(0).addRegDef(dest));
  else
    mvec.push_back(BuildMI(V9::ORr,3).addReg(tmpReg).addMReg(SparcV9::g0).addRegDef(dest));
}

/// CreateSETXConst - Set a 64-bit signed or unsigned constant in the
/// register `dest'.  Use SETUWConst for each 32 bit word, plus a
/// left-shift-by-32 in between.  This function correctly emulates the SETX
/// pseudo-op for SPARC v9.  It optimizes the same cases as SETUWConst for each
/// 32 bit word.
///
static inline void
CreateSETXConst(uint64_t C,
                Instruction* tmpReg, Instruction* dest,
                std::vector<MachineInstr*>& mvec,
                MachineCodeForInstruction& mcfi, Value* val) {
  assert(C > (unsigned int) ~0 && "Use SETUW/SETSW for 32-bit values!");

  MachineInstr* MI;

  // Code to set the upper 32 bits of the value in register `tmpReg'
  CreateSETUWConst((C >> 32), tmpReg, mvec, mcfi, val);

  //TmpInstruction for intermediate values
  TmpInstruction *tmpReg2 = new TmpInstruction(mcfi, (Instruction*) val);

  // Shift tmpReg left by 32 bits
  mvec.push_back(BuildMI(V9::SLLXi6, 3).addReg(tmpReg).addZImm(32)
                 .addRegDef(tmpReg2));

  //TmpInstruction for intermediate values
  TmpInstruction *tmpReg3 = new TmpInstruction(mcfi, (Instruction*) val);

  // Code to set the low 32 bits of the value in register `dest'
  CreateSETUWConst(C, tmpReg3, mvec, mcfi, val);

  // dest = OR(tmpReg, dest)
  mvec.push_back(BuildMI(V9::ORr,3).addReg(tmpReg3).addReg(tmpReg2).addRegDef(dest));
}

/// CreateSETUWLabel - Set a 32-bit constant (given by a symbolic label) in
/// the register `dest'.
///
static inline void
CreateSETUWLabel(Value* val,
                 Instruction* dest, std::vector<MachineInstr*>& mvec) {
  MachineInstr* MI;

  MachineCodeForInstruction &mcfi = MachineCodeForInstruction::get((Instruction*) val);
  TmpInstruction* tmpReg = new TmpInstruction(mcfi, val);

  // Set the high 22 bits in dest
  MI = BuildMI(V9::SETHI, 2).addReg(val).addRegDef(tmpReg);
  MI->getOperand(0).markHi32();
  mvec.push_back(MI);

  // Set the low 10 bits in dest
  MI = BuildMI(V9::ORr, 3).addReg(tmpReg).addReg(val).addRegDef(dest);
  MI->getOperand(1).markLo32();
  mvec.push_back(MI);
}

/// CreateSETXLabel - Set a 64-bit constant (given by a symbolic label) in the
/// register `dest'.
///
static inline void
CreateSETXLabel(Value* val, Instruction* tmpReg,
                Instruction* dest, std::vector<MachineInstr*>& mvec,
                MachineCodeForInstruction& mcfi) {
  assert(isa<Constant>(val) &&
         "I only know about constant values and global addresses");

  MachineInstr* MI;

  MI = BuildMI(V9::SETHI, 2).addPCDisp(val).addRegDef(tmpReg);
  MI->getOperand(0).markHi64();
  mvec.push_back(MI);

  TmpInstruction* tmpReg2 =
        new TmpInstruction(mcfi, PointerType::get(val->getType()), val);

  MI = BuildMI(V9::ORi, 3).addReg(tmpReg).addPCDisp(val).addRegDef(tmpReg2);
  MI->getOperand(1).markLo64();
  mvec.push_back(MI);


  TmpInstruction* tmpReg3 =
        new TmpInstruction(mcfi, PointerType::get(val->getType()), val);

  mvec.push_back(BuildMI(V9::SLLXi6, 3).addReg(tmpReg2).addZImm(32)
                 .addRegDef(tmpReg3));


  TmpInstruction* tmpReg4 =
        new TmpInstruction(mcfi, PointerType::get(val->getType()), val);
  MI = BuildMI(V9::SETHI, 2).addPCDisp(val).addRegDef(tmpReg4);
  MI->getOperand(0).markHi32();
  mvec.push_back(MI);

    TmpInstruction* tmpReg5 =
        new TmpInstruction(mcfi, PointerType::get(val->getType()), val);
  MI = BuildMI(V9::ORr, 3).addReg(tmpReg4).addReg(tmpReg3).addRegDef(tmpReg5);
  mvec.push_back(MI);

  MI = BuildMI(V9::ORi, 3).addReg(tmpReg5).addPCDisp(val).addRegDef(dest);
  MI->getOperand(1).markLo32();
  mvec.push_back(MI);
}

/// CreateUIntSetInstruction - Create code to Set an unsigned constant in the
/// register `dest'.  Uses CreateSETUWConst, CreateSETSWConst or CreateSETXConst
/// as needed.  CreateSETSWConst is an optimization for the case that the
/// unsigned value has all ones in the 33 high bits (so that sign-extension sets
/// them all).
///
static inline void
CreateUIntSetInstruction(uint64_t C, Instruction* dest,
                         std::vector<MachineInstr*>& mvec,
                         MachineCodeForInstruction& mcfi, Value* val) {
  static const uint64_t lo32 = (uint32_t) ~0;
  if (C <= lo32)                        // High 32 bits are 0.  Set low 32 bits.
    CreateSETUWConst((uint32_t) C, dest, mvec, mcfi, val);
  else if ((C & ~lo32) == ~lo32 && (C & (1U << 31))) {
    // All high 33 (not 32) bits are 1s: sign-extension will take care
    // of high 32 bits, so use the sequence for signed int
    CreateSETSWConst((int32_t) C, dest, mvec, mcfi, val);
  } else if (C > lo32) {
    // C does not fit in 32 bits
    TmpInstruction* tmpReg = new TmpInstruction(mcfi, Type::IntTy);
    CreateSETXConst(C, tmpReg, dest, mvec, mcfi, val);
  }
}

/// CreateIntSetInstruction - Create code to Set a signed constant in the
/// register `dest'.  Really the same as CreateUIntSetInstruction.
///
static inline void
CreateIntSetInstruction(int64_t C, Instruction* dest,
                        std::vector<MachineInstr*>& mvec,
                        MachineCodeForInstruction& mcfi, Value* val) {
  CreateUIntSetInstruction((uint64_t) C, dest, mvec, mcfi, val);
}

/// MaxConstantsTableTy - Table mapping LLVM opcodes to the max. immediate
/// constant usable for that operation in the SparcV9 backend. Used by
/// ConstantMayNotFitInImmedField().
///
struct MaxConstantsTableTy {
  // Entry == 0 ==> no immediate constant field exists at all.
  // Entry >  0 ==> abs(immediate constant) <= Entry
  std::vector<int> tbl;

  int getMaxConstantForInstr (unsigned llvmOpCode);
  MaxConstantsTableTy ();
  unsigned size() const            { return tbl.size (); }
  int &operator[] (unsigned index) { return tbl[index];  }
};

int MaxConstantsTableTy::getMaxConstantForInstr(unsigned llvmOpCode) {
  int modelOpCode = -1;

  if (llvmOpCode >= Instruction::BinaryOpsBegin &&
      llvmOpCode <  Instruction::BinaryOpsEnd)
    modelOpCode = V9::ADDi;
  else
    switch(llvmOpCode) {
    case Instruction::Ret:   modelOpCode = V9::JMPLCALLi; break;

    case Instruction::Malloc:
    case Instruction::Alloca:
    case Instruction::GetElementPtr:
    case Instruction::PHI:
    case Instruction::Cast:
    case Instruction::Call:  modelOpCode = V9::ADDi; break;

    case Instruction::Shl:
    case Instruction::Shr:   modelOpCode = V9::SLLXi6; break;

    default: break;
    };

  return (modelOpCode < 0)? 0: SparcV9MachineInstrDesc[modelOpCode].maxImmedConst;
}

MaxConstantsTableTy::MaxConstantsTableTy () : tbl (Instruction::OtherOpsEnd) {
  unsigned op;
  assert(tbl.size() == Instruction::OtherOpsEnd &&
         "assignments below will be illegal!");
  for (op = Instruction::TermOpsBegin; op < Instruction::TermOpsEnd; ++op)
    tbl[op] = getMaxConstantForInstr(op);
  for (op = Instruction::BinaryOpsBegin; op < Instruction::BinaryOpsEnd; ++op)
    tbl[op] = getMaxConstantForInstr(op);
  for (op = Instruction::MemoryOpsBegin; op < Instruction::MemoryOpsEnd; ++op)
    tbl[op] = getMaxConstantForInstr(op);
  for (op = Instruction::OtherOpsBegin; op < Instruction::OtherOpsEnd; ++op)
    tbl[op] = getMaxConstantForInstr(op);
}

bool ConstantMayNotFitInImmedField(const Constant* CV, const Instruction* I) {
  // The one and only MaxConstantsTable, used only by this function.
  static MaxConstantsTableTy MaxConstantsTable;

  if (I->getOpcode() >= MaxConstantsTable.size()) // user-defined op (or bug!)
    return true;

  // can always use %g0
  if (isa<ConstantPointerNull>(CV) || isa<UndefValue>(CV))
    return false;

  if (isa<SwitchInst>(I)) // Switch instructions will be lowered!
    return false;

  if (const ConstantInt* CI = dyn_cast<ConstantInt>(CV))
    return labs((int64_t)CI->getRawValue()) > MaxConstantsTable[I->getOpcode()];

  if (isa<ConstantBool>(CV))
    return 1 > MaxConstantsTable[I->getOpcode()];

  return true;
}

/// ChooseLoadInstruction - Return the appropriate load instruction opcode
/// based on the given LLVM value type.
///
static inline MachineOpCode ChooseLoadInstruction(const Type *DestTy) {
  switch (DestTy->getTypeID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:   return V9::LDUBr;
  case Type::SByteTyID:   return V9::LDSBr;
  case Type::UShortTyID:  return V9::LDUHr;
  case Type::ShortTyID:   return V9::LDSHr;
  case Type::UIntTyID:    return V9::LDUWr;
  case Type::IntTyID:     return V9::LDSWr;
  case Type::PointerTyID:
  case Type::ULongTyID:
  case Type::LongTyID:    return V9::LDXr;
  case Type::FloatTyID:   return V9::LDFr;
  case Type::DoubleTyID:  return V9::LDDFr;
  default: assert(0 && "Invalid type for Load instruction");
  }
  return 0;
}

/// ChooseStoreInstruction - Return the appropriate store instruction opcode
/// based on the given LLVM value type.
///
static inline MachineOpCode ChooseStoreInstruction(const Type *DestTy) {
  switch (DestTy->getTypeID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:
  case Type::SByteTyID:   return V9::STBr;
  case Type::UShortTyID:
  case Type::ShortTyID:   return V9::STHr;
  case Type::UIntTyID:
  case Type::IntTyID:     return V9::STWr;
  case Type::PointerTyID:
  case Type::ULongTyID:
  case Type::LongTyID:    return V9::STXr;
  case Type::FloatTyID:   return V9::STFr;
  case Type::DoubleTyID:  return V9::STDFr;
  default: assert(0 && "Invalid type for Store instruction");
  }
  return 0;
}

static inline MachineOpCode ChooseAddInstructionByType(const Type* resultType) {
  MachineOpCode opCode = V9::INVALID_OPCODE;
  if (resultType->isIntegral() || isa<PointerType>(resultType)
      || isa<FunctionType>(resultType) || resultType == Type::LabelTy) {
    opCode = V9::ADDr;
  } else
    switch(resultType->getTypeID()) {
    case Type::FloatTyID:  opCode = V9::FADDS; break;
    case Type::DoubleTyID: opCode = V9::FADDD; break;
    default: assert(0 && "Invalid type for ADD instruction"); break;
    }

  return opCode;
}

/// convertOpcodeFromRegToImm - Because the SparcV9 instruction selector likes
/// to re-write operands to instructions, making them change from a Value*
/// (virtual register) to a Constant* (making an immediate field), we need to
/// change the opcode from a register-based instruction to an immediate-based
/// instruction, hence this mapping.
///
static unsigned convertOpcodeFromRegToImm(unsigned Opcode) {
  switch (Opcode) {
    /* arithmetic */
  case V9::ADDr:     return V9::ADDi;
  case V9::ADDccr:   return V9::ADDcci;
  case V9::ADDCr:    return V9::ADDCi;
  case V9::ADDCccr:  return V9::ADDCcci;
  case V9::SUBr:     return V9::SUBi;
  case V9::SUBccr:   return V9::SUBcci;
  case V9::SUBCr:    return V9::SUBCi;
  case V9::SUBCccr:  return V9::SUBCcci;
  case V9::MULXr:    return V9::MULXi;
  case V9::SDIVXr:   return V9::SDIVXi;
  case V9::UDIVXr:   return V9::UDIVXi;

    /* logical */
  case V9::ANDr:    return V9::ANDi;
  case V9::ANDccr:  return V9::ANDcci;
  case V9::ANDNr:   return V9::ANDNi;
  case V9::ANDNccr: return V9::ANDNcci;
  case V9::ORr:     return V9::ORi;
  case V9::ORccr:   return V9::ORcci;
  case V9::ORNr:    return V9::ORNi;
  case V9::ORNccr:  return V9::ORNcci;
  case V9::XORr:    return V9::XORi;
  case V9::XORccr:  return V9::XORcci;
  case V9::XNORr:   return V9::XNORi;
  case V9::XNORccr: return V9::XNORcci;

    /* shift */
  case V9::SLLr5:   return V9::SLLi5;
  case V9::SRLr5:   return V9::SRLi5;
  case V9::SRAr5:   return V9::SRAi5;
  case V9::SLLXr6:  return V9::SLLXi6;
  case V9::SRLXr6:  return V9::SRLXi6;
  case V9::SRAXr6:  return V9::SRAXi6;

    /* Conditional move on int comparison with zero */
  case V9::MOVRZr:   return V9::MOVRZi;
  case V9::MOVRLEZr: return V9::MOVRLEZi;
  case V9::MOVRLZr:  return V9::MOVRLZi;
  case V9::MOVRNZr:  return V9::MOVRNZi;
  case V9::MOVRGZr:  return V9::MOVRGZi;
  case V9::MOVRGEZr: return V9::MOVRGEZi;


    /* Conditional move on int condition code */
  case V9::MOVAr:   return V9::MOVAi;
  case V9::MOVNr:   return V9::MOVNi;
  case V9::MOVNEr:  return V9::MOVNEi;
  case V9::MOVEr:   return V9::MOVEi;
  case V9::MOVGr:   return V9::MOVGi;
  case V9::MOVLEr:  return V9::MOVLEi;
  case V9::MOVGEr:  return V9::MOVGEi;
  case V9::MOVLr:   return V9::MOVLi;
  case V9::MOVGUr:  return V9::MOVGUi;
  case V9::MOVLEUr: return V9::MOVLEUi;
  case V9::MOVCCr:  return V9::MOVCCi;
  case V9::MOVCSr:  return V9::MOVCSi;
  case V9::MOVPOSr: return V9::MOVPOSi;
  case V9::MOVNEGr: return V9::MOVNEGi;
  case V9::MOVVCr:  return V9::MOVVCi;
  case V9::MOVVSr:  return V9::MOVVSi;

    /* Conditional move of int reg on fp condition code */
  case V9::MOVFAr:   return V9::MOVFAi;
  case V9::MOVFNr:   return V9::MOVFNi;
  case V9::MOVFUr:   return V9::MOVFUi;
  case V9::MOVFGr:   return V9::MOVFGi;
  case V9::MOVFUGr:  return V9::MOVFUGi;
  case V9::MOVFLr:   return V9::MOVFLi;
  case V9::MOVFULr:  return V9::MOVFULi;
  case V9::MOVFLGr:  return V9::MOVFLGi;
  case V9::MOVFNEr:  return V9::MOVFNEi;
  case V9::MOVFEr:   return V9::MOVFEi;
  case V9::MOVFUEr:  return V9::MOVFUEi;
  case V9::MOVFGEr:  return V9::MOVFGEi;
  case V9::MOVFUGEr: return V9::MOVFUGEi;
  case V9::MOVFLEr:  return V9::MOVFLEi;
  case V9::MOVFULEr: return V9::MOVFULEi;
  case V9::MOVFOr:   return V9::MOVFOi;

    /* load */
  case V9::LDSBr:   return V9::LDSBi;
  case V9::LDSHr:   return V9::LDSHi;
  case V9::LDSWr:   return V9::LDSWi;
  case V9::LDUBr:   return V9::LDUBi;
  case V9::LDUHr:   return V9::LDUHi;
  case V9::LDUWr:   return V9::LDUWi;
  case V9::LDXr:    return V9::LDXi;
  case V9::LDFr:    return V9::LDFi;
  case V9::LDDFr:   return V9::LDDFi;
  case V9::LDQFr:   return V9::LDQFi;
  case V9::LDFSRr:  return V9::LDFSRi;
  case V9::LDXFSRr: return V9::LDXFSRi;

    /* store */
  case V9::STBr:    return V9::STBi;
  case V9::STHr:    return V9::STHi;
  case V9::STWr:    return V9::STWi;
  case V9::STXr:    return V9::STXi;
  case V9::STFr:    return V9::STFi;
  case V9::STDFr:   return V9::STDFi;
  case V9::STFSRr:  return V9::STFSRi;
  case V9::STXFSRr: return V9::STXFSRi;

    /* jump & return */
  case V9::JMPLCALLr: return V9::JMPLCALLi;
  case V9::JMPLRETr:  return V9::JMPLRETi;

  /* save and restore */
  case V9::SAVEr:     return V9::SAVEi;
  case V9::RESTOREr:  return V9::RESTOREi;

  default:
    // It's already in correct format
    // Or, it's just not handled yet, but an assert() would break LLC
#if 0
    std::cerr << "Unhandled opcode in convertOpcodeFromRegToImm(): " << Opcode
              << "\n";
#endif
    return Opcode;
  }
}

/// CreateCodeToLoadConst - Create an instruction sequence to put the
/// constant `val' into the virtual register `dest'.  `val' may be a Constant or
/// a GlobalValue, viz., the constant address of a global variable or function.
/// The generated instructions are returned in `mvec'. Any temp. registers
/// (TmpInstruction) created are recorded in mcfi. Any stack space required is
/// allocated via MachineFunction.
///
void CreateCodeToLoadConst(const TargetMachine& target, Function* F,
                           Value* val, Instruction* dest,
                           std::vector<MachineInstr*>& mvec,
                           MachineCodeForInstruction& mcfi) {
  assert(isa<Constant>(val) &&
         "I only know about constant values and global addresses");

  // Use a "set" instruction for known constants or symbolic constants (labels)
  // that can go in an integer reg.
  // We have to use a "load" instruction for all other constants,
  // in particular, floating point constants.
  const Type* valType = val->getType();

  if (isa<GlobalValue>(val)) {
      TmpInstruction* tmpReg =
        new TmpInstruction(mcfi, PointerType::get(val->getType()), val);
      CreateSETXLabel(val, tmpReg, dest, mvec, mcfi);
      return;
  }

  bool isValid;
  uint64_t C = ConvertConstantToIntType(target, val, dest->getType(), isValid);
  if (isValid) {
    if (dest->getType()->isSigned())
      CreateUIntSetInstruction(C, dest, mvec, mcfi, val);
    else
      CreateIntSetInstruction((int64_t) C, dest, mvec, mcfi, val);

  } else {
    // Make an instruction sequence to load the constant, viz:
    //            SETX <addr-of-constant>, tmpReg, addrReg
    //            LOAD  /*addr*/ addrReg, /*offset*/ 0, dest
    // First, create a tmp register to be used by the SETX sequence.
    TmpInstruction* tmpReg =
      new TmpInstruction(mcfi, PointerType::get(val->getType()));

    // Create another TmpInstruction for the address register
    TmpInstruction* addrReg =
      new TmpInstruction(mcfi, PointerType::get(val->getType()));

    // Get the constant pool index for this constant
    MachineConstantPool *CP = MachineFunction::get(F).getConstantPool();
    Constant *C = cast<Constant>(val);
    unsigned Align = target.getTargetData().getTypeAlignmentShift(C->getType());
    unsigned CPI = CP->getConstantPoolIndex(C, Align);

    // Put the address of the constant into a register
    MachineInstr* MI;

    MI = BuildMI(V9::SETHI, 2).addConstantPoolIndex(CPI).addRegDef(tmpReg);
    MI->getOperand(0).markHi64();
    mvec.push_back(MI);

    //Create another tmp register for the SETX sequence to preserve SSA
    TmpInstruction* tmpReg2 =
      new TmpInstruction(mcfi, PointerType::get(val->getType()));

    MI = BuildMI(V9::ORi, 3).addReg(tmpReg).addConstantPoolIndex(CPI)
      .addRegDef(tmpReg2);
    MI->getOperand(1).markLo64();
    mvec.push_back(MI);

    //Create another tmp register for the SETX sequence to preserve SSA
    TmpInstruction* tmpReg3 =
      new TmpInstruction(mcfi, PointerType::get(val->getType()));

    mvec.push_back(BuildMI(V9::SLLXi6, 3).addReg(tmpReg2).addZImm(32)
                   .addRegDef(tmpReg3));
    MI = BuildMI(V9::SETHI, 2).addConstantPoolIndex(CPI).addRegDef(addrReg);
    MI->getOperand(0).markHi32();
    mvec.push_back(MI);

    // Create another TmpInstruction for the address register
    TmpInstruction* addrReg2 =
      new TmpInstruction(mcfi, PointerType::get(val->getType()));


    MI = BuildMI(V9::ORr, 3).addReg(addrReg).addReg(tmpReg3).addRegDef(addrReg2);
    mvec.push_back(MI);

    // Create another TmpInstruction for the address register
    TmpInstruction* addrReg3 =
      new TmpInstruction(mcfi, PointerType::get(val->getType()));

    MI = BuildMI(V9::ORi, 3).addReg(addrReg2).addConstantPoolIndex(CPI)
      .addRegDef(addrReg3);
    MI->getOperand(1).markLo32();
    mvec.push_back(MI);

    // Now load the constant from out ConstantPool label
    unsigned Opcode = ChooseLoadInstruction(val->getType());
    Opcode = convertOpcodeFromRegToImm(Opcode);
    mvec.push_back(BuildMI(Opcode, 3)
                   .addReg(addrReg3).addSImm((int64_t)0).addRegDef(dest));
  }
}

/// CreateCodeToCopyFloatToInt - Similarly, create an instruction sequence
/// to copy an FP register `val' to an integer register `dest' by copying to
/// memory and back.  The generated instructions are returned in `mvec'.  Any
/// temp. virtual registers (TmpInstruction) created are recorded in mcfi.
/// Temporary stack space required is allocated via MachineFunction.
///
void CreateCodeToCopyFloatToInt(const TargetMachine& target, Function* F,
                                Value* val, Instruction* dest,
                                std::vector<MachineInstr*>& mvec,
                                MachineCodeForInstruction& mcfi) {
  const Type* opTy   = val->getType();
  const Type* destTy = dest->getType();
  assert(opTy->isFloatingPoint() && "Source type must be float/double");
  assert((destTy->isIntegral() || isa<PointerType>(destTy))
         && "Dest type must be integer, bool or pointer");

  // FIXME: For now, we allocate permanent space because the stack frame
  // manager does not allow locals to be allocated (e.g., for alloca) after
  // a temp is allocated!
  int offset = MachineFunction::get(F).getInfo<SparcV9FunctionInfo>()->allocateLocalVar(val);

  unsigned FPReg = target.getRegInfo()->getFramePointer();

  // Store instruction stores `val' to [%fp+offset].
  // The store opCode is based only the source value being copied.
  unsigned StoreOpcode = ChooseStoreInstruction(opTy);
  StoreOpcode = convertOpcodeFromRegToImm(StoreOpcode);
  mvec.push_back(BuildMI(StoreOpcode, 3)
                 .addReg(val).addMReg(FPReg).addSImm(offset));

  // Load instruction loads [%fp+offset] to `dest'.
  // The type of the load opCode is the integer type that matches the
  // source type in size:
  // On SparcV9: int for float, long for double.
  // Note that we *must* use signed loads even for unsigned dest types, to
  // ensure correct sign-extension for UByte, UShort or UInt:
  const Type* loadTy = (opTy == Type::FloatTy)? Type::IntTy : Type::LongTy;
  unsigned LoadOpcode = ChooseLoadInstruction(loadTy);
  LoadOpcode = convertOpcodeFromRegToImm(LoadOpcode);
  mvec.push_back(BuildMI(LoadOpcode, 3).addMReg(FPReg)
                 .addSImm(offset).addRegDef(dest));
}

/// CreateBitExtensionInstructions - Helper function for sign-extension and
/// zero-extension. For SPARC v9, we sign-extend the given operand using SLL;
/// SRA/SRL.
///
inline void
CreateBitExtensionInstructions(bool signExtend, const TargetMachine& target,
                               Function* F, Value* srcVal, Value* destVal,
                               unsigned int numLowBits,
                               std::vector<MachineInstr*>& mvec,
                               MachineCodeForInstruction& mcfi) {
  MachineInstr* M;

  assert(numLowBits <= 32 && "Otherwise, nothing should be done here!");

  if (numLowBits < 32) {
    // SLL is needed since operand size is < 32 bits.
    TmpInstruction *tmpI = new TmpInstruction(mcfi, destVal->getType(),
                                              srcVal, destVal, "make32");
    mvec.push_back(BuildMI(V9::SLLXi6, 3).addReg(srcVal)
                   .addZImm(32-numLowBits).addRegDef(tmpI));
    srcVal = tmpI;
  }

  mvec.push_back(BuildMI(signExtend? V9::SRAi5 : V9::SRLi5, 3)
                 .addReg(srcVal).addZImm(32-numLowBits).addRegDef(destVal));
}

/// CreateSignExtensionInstructions - Create instruction sequence to produce
/// a sign-extended register value from an arbitrary-sized integer value (sized
/// in bits, not bytes). The generated instructions are returned in `mvec'. Any
/// temp. registers (TmpInstruction) created are recorded in mcfi. Any stack
/// space required is allocated via MachineFunction.
///
void CreateSignExtensionInstructions(const TargetMachine& target,
                                     Function* F, Value* srcVal, Value* destVal,
                                     unsigned int numLowBits,
                                     std::vector<MachineInstr*>& mvec,
                                     MachineCodeForInstruction& mcfi) {
  CreateBitExtensionInstructions(/*signExtend*/ true, target, F, srcVal,
                                 destVal, numLowBits, mvec, mcfi);
}

/// CreateZeroExtensionInstructions - Create instruction sequence to produce
/// a zero-extended register value from an arbitrary-sized integer value (sized
/// in bits, not bytes).  For SPARC v9, we sign-extend the given operand using
/// SLL; SRL.  The generated instructions are returned in `mvec'.  Any temp.
/// registers (TmpInstruction) created are recorded in mcfi.  Any stack space
/// required is allocated via MachineFunction.
///
void CreateZeroExtensionInstructions(const TargetMachine& target,
                                     Function* F, Value* srcVal, Value* destVal,
                                     unsigned int numLowBits,
                                     std::vector<MachineInstr*>& mvec,
                                     MachineCodeForInstruction& mcfi) {
  CreateBitExtensionInstructions(/*signExtend*/ false, target, F, srcVal,
                                 destVal, numLowBits, mvec, mcfi);
}

/// CreateCodeToCopyIntToFloat - Create an instruction sequence to copy an
/// integer register `val' to a floating point register `dest' by copying to
/// memory and back. val must be an integral type.  dest must be a Float or
/// Double. The generated instructions are returned in `mvec'. Any temp.
/// registers (TmpInstruction) created are recorded in mcfi. Any stack space
/// required is allocated via MachineFunction.
///
void CreateCodeToCopyIntToFloat(const TargetMachine& target,
                                Function* F, Value* val, Instruction* dest,
                                std::vector<MachineInstr*>& mvec,
                                MachineCodeForInstruction& mcfi) {
  assert((val->getType()->isIntegral() || isa<PointerType>(val->getType()))
         && "Source type must be integral (integer or bool) or pointer");
  assert(dest->getType()->isFloatingPoint()
         && "Dest type must be float/double");

  // Get a stack slot to use for the copy
  int offset = MachineFunction::get(F).getInfo<SparcV9FunctionInfo>()->allocateLocalVar(val);

  // Get the size of the source value being copied.
  size_t srcSize = target.getTargetData().getTypeSize(val->getType());

  // Store instruction stores `val' to [%fp+offset].
  // The store and load opCodes are based on the size of the source value.
  // If the value is smaller than 32 bits, we must sign- or zero-extend it
  // to 32 bits since the load-float will load 32 bits.
  // Note that the store instruction is the same for signed and unsigned ints.
  const Type* storeType = (srcSize <= 4)? Type::IntTy : Type::LongTy;
  Value* storeVal = val;
  if (srcSize < target.getTargetData().getTypeSize(Type::FloatTy)) {
    // sign- or zero-extend respectively
    storeVal = new TmpInstruction(mcfi, storeType, val);
    if (val->getType()->isSigned())
      CreateSignExtensionInstructions(target, F, val, storeVal, 8*srcSize,
                                      mvec, mcfi);
    else
      CreateZeroExtensionInstructions(target, F, val, storeVal, 8*srcSize,
                                      mvec, mcfi);
  }

  unsigned FPReg = target.getRegInfo()->getFramePointer();
  unsigned StoreOpcode = ChooseStoreInstruction(storeType);
  StoreOpcode = convertOpcodeFromRegToImm(StoreOpcode);
  mvec.push_back(BuildMI(StoreOpcode, 3)
                 .addReg(storeVal).addMReg(FPReg).addSImm(offset));

  // Load instruction loads [%fp+offset] to `dest'.
  // The type of the load opCode is the floating point type that matches the
  // stored type in size:
  // On SparcV9: float for int or smaller, double for long.
  const Type* loadType = (srcSize <= 4)? Type::FloatTy : Type::DoubleTy;
  unsigned LoadOpcode = ChooseLoadInstruction(loadType);
  LoadOpcode = convertOpcodeFromRegToImm(LoadOpcode);
  mvec.push_back(BuildMI(LoadOpcode, 3)
                 .addMReg(FPReg).addSImm(offset).addRegDef(dest));
}

/// InsertCodeToLoadConstant - Generates code to load the constant
/// into a TmpInstruction (virtual reg) and returns the virtual register.
///
static TmpInstruction*
InsertCodeToLoadConstant(Function *F, Value* opValue, Instruction* vmInstr,
                         std::vector<MachineInstr*>& loadConstVec,
                         TargetMachine& target) {
  // Create a tmp virtual register to hold the constant.
  MachineCodeForInstruction &mcfi = MachineCodeForInstruction::get(vmInstr);
  TmpInstruction* tmpReg = new TmpInstruction(mcfi, opValue);

  CreateCodeToLoadConst(target, F, opValue, tmpReg, loadConstVec, mcfi);

  // Record the mapping from the tmp VM instruction to machine instruction.
  // Do this for all machine instructions that were not mapped to any
  // other temp values created by
  // tmpReg->addMachineInstruction(loadConstVec.back());
  return tmpReg;
}

MachineOperand::MachineOperandType
ChooseRegOrImmed(int64_t intValue, bool isSigned,
                 MachineOpCode opCode, const TargetMachine& target,
                 bool canUseImmed, unsigned int& getMachineRegNum,
                 int64_t& getImmedValue) {
  MachineOperand::MachineOperandType opType=MachineOperand::MO_VirtualRegister;
  getMachineRegNum = 0;
  getImmedValue = 0;

  if (canUseImmed &&
      target.getInstrInfo()->constantFitsInImmedField(opCode, intValue)) {
      opType = isSigned? MachineOperand::MO_SignExtendedImmed
                       : MachineOperand::MO_UnextendedImmed;
      getImmedValue = intValue;
  } else if (intValue == 0 &&
             target.getRegInfo()->getZeroRegNum() != (unsigned)-1) {
    opType = MachineOperand::MO_MachineRegister;
    getMachineRegNum = target.getRegInfo()->getZeroRegNum();
  }

  return opType;
}

MachineOperand::MachineOperandType
ChooseRegOrImmed(Value* val,
                 MachineOpCode opCode, const TargetMachine& target,
                 bool canUseImmed, unsigned int& getMachineRegNum,
                 int64_t& getImmedValue) {
  getMachineRegNum = 0;
  getImmedValue = 0;

  // To use reg or immed, constant needs to be integer, bool, or a NULL pointer.
  // ConvertConstantToIntType() does the right conversions.
  bool isValidConstant;
  uint64_t valueToUse =
    ConvertConstantToIntType(target, val, val->getType(), isValidConstant);
  if (! isValidConstant)
    return MachineOperand::MO_VirtualRegister;

  // Now check if the constant value fits in the IMMED field.
  return ChooseRegOrImmed((int64_t) valueToUse, val->getType()->isSigned(),
                          opCode, target, canUseImmed,
                          getMachineRegNum, getImmedValue);
}

/// CreateCopyInstructionsByType - Create instruction(s) to copy src to dest,
/// for arbitrary types. The generated instructions are returned in `mvec'. Any
/// temp. registers (TmpInstruction) created are recorded in mcfi. Any stack
/// space required is allocated via MachineFunction.
///
void CreateCopyInstructionsByType(const TargetMachine& target,
                                  Function *F, Value* src, Instruction* dest,
                                  std::vector<MachineInstr*>& mvec,
                                  MachineCodeForInstruction& mcfi) {
  bool loadConstantToReg = false;
  const Type* resultType = dest->getType();
  MachineOpCode opCode = ChooseAddInstructionByType(resultType);
  assert (opCode != V9::INVALID_OPCODE
          && "Unsupported result type in CreateCopyInstructionsByType()");

  // If `src' is a constant that doesn't fit in the immed field or if it is
  // a global variable (i.e., a constant address), generate a load
  // instruction instead of an add.
  if (isa<GlobalValue>(src))
    loadConstantToReg = true;
  else if (isa<Constant>(src)) {
    unsigned int machineRegNum;
    int64_t immedValue;
    MachineOperand::MachineOperandType opType =
      ChooseRegOrImmed(src, opCode, target, /*canUseImmed*/ true,
                       machineRegNum, immedValue);

    if (opType == MachineOperand::MO_VirtualRegister)
      loadConstantToReg = true;
  }

  if (loadConstantToReg) {
    // `src' is constant and cannot fit in immed field for the ADD.
    // Insert instructions to "load" the constant into a register.
    CreateCodeToLoadConst(target, F, src, dest, mvec, mcfi);
  } else {
    // Create a reg-to-reg copy instruction for the given type:
    // -- For FP values, create a FMOVS or FMOVD instruction
    // -- For non-FP values, create an add-with-0 instruction (opCode as above)
    // Make `src' the second operand, in case it is a small constant!
    MachineInstr* MI;
    if (resultType->isFloatingPoint())
      MI = (BuildMI(resultType == Type::FloatTy? V9::FMOVS : V9::FMOVD, 2)
            .addReg(src).addRegDef(dest));
    else {
        const Type* Ty =isa<PointerType>(resultType)? Type::ULongTy :resultType;
        MI = (BuildMI(opCode, 3)
              .addSImm((int64_t) 0).addReg(src).addRegDef(dest));
    }
    mvec.push_back(MI);
  }
}

/// FixConstantOperandsForInstr - Make a machine instruction use its constant
/// operands more efficiently.  If the constant is 0, then use the hardwired 0
/// register, if any.  Else, if the constant fits in the IMMEDIATE field, then
/// use that field.  Otherwise, else create instructions to put the constant
/// into a register, either directly or by loading explicitly from the constant
/// pool.  In the first 2 cases, the operand of `minstr' is modified in place.
/// Returns a vector of machine instructions generated for operands that fall
/// under case 3; these must be inserted before `minstr'.
///
std::vector<MachineInstr*>
FixConstantOperandsForInstr(Instruction* vmInstr, MachineInstr* minstr,
                            TargetMachine& target) {
  std::vector<MachineInstr*> MVec;

  MachineOpCode opCode = minstr->getOpcode();
  const TargetInstrInfo& instrInfo = *target.getInstrInfo();
  int resultPos = instrInfo.get(opCode).resultPos;
  int immedPos = instrInfo.getImmedConstantPos(opCode);

  Function *F = vmInstr->getParent()->getParent();

  for (unsigned op=0; op < minstr->getNumOperands(); op++) {
      const MachineOperand& mop = minstr->getOperand(op);

      // Skip the result position, preallocated machine registers, or operands
      // that cannot be constants (CC regs or PC-relative displacements)
      if (resultPos == (int)op ||
          mop.getType() == MachineOperand::MO_MachineRegister ||
          mop.getType() == MachineOperand::MO_CCRegister ||
          mop.getType() == MachineOperand::MO_PCRelativeDisp)
        continue;

      bool constantThatMustBeLoaded = false;
      unsigned int machineRegNum = 0;
      int64_t immedValue = 0;
      Value* opValue = NULL;
      MachineOperand::MachineOperandType opType =
        MachineOperand::MO_VirtualRegister;

      // Operand may be a virtual register or a compile-time constant
      if (mop.getType() == MachineOperand::MO_VirtualRegister) {
        assert(mop.getVRegValue() != NULL);
        opValue = mop.getVRegValue();
        if (Constant *opConst = dyn_cast<Constant>(opValue))
          if (!isa<GlobalValue>(opConst)) {
            opType = ChooseRegOrImmed(opConst, opCode, target,
                                      (immedPos == (int)op), machineRegNum,
                                      immedValue);
            if (opType == MachineOperand::MO_VirtualRegister)
              constantThatMustBeLoaded = true;
          }
      } else {
        // If the operand is from the constant pool, don't try to change it.
        if (mop.getType() == MachineOperand::MO_ConstantPoolIndex) {
          continue;
        }
        assert(mop.isImmediate());
        bool isSigned = mop.getType() == MachineOperand::MO_SignExtendedImmed;

        // Bit-selection flags indicate an instruction that is extracting
        // bits from its operand so ignore this even if it is a big constant.
        if (mop.isHiBits32() || mop.isLoBits32() ||
            mop.isHiBits64() || mop.isLoBits64())
          continue;

        opType = ChooseRegOrImmed(mop.getImmedValue(), isSigned,
                                  opCode, target, (immedPos == (int)op),
                                  machineRegNum, immedValue);

        if (opType == MachineOperand::MO_SignExtendedImmed ||
            opType == MachineOperand::MO_UnextendedImmed) {
          // The optype is an immediate value
          // This means we need to change the opcode, e.g. ADDr -> ADDi
          unsigned newOpcode = convertOpcodeFromRegToImm(opCode);
          minstr->setOpcode(newOpcode);
        }

        if (opType == mop.getType())
          continue;           // no change: this is the most common case

        if (opType == MachineOperand::MO_VirtualRegister) {
          constantThatMustBeLoaded = true;
          opValue = isSigned
            ? (Value*)ConstantSInt::get(Type::LongTy, immedValue)
            : (Value*)ConstantUInt::get(Type::ULongTy,(uint64_t)immedValue);
        }
      }

      if (opType == MachineOperand::MO_MachineRegister)
        minstr->SetMachineOperandReg(op, machineRegNum);
      else if (opType == MachineOperand::MO_SignExtendedImmed ||
               opType == MachineOperand::MO_UnextendedImmed) {
        minstr->SetMachineOperandConst(op, opType, immedValue);
        // The optype is or has become an immediate
        // This means we need to change the opcode, e.g. ADDr -> ADDi
        unsigned newOpcode = convertOpcodeFromRegToImm(opCode);
        minstr->setOpcode(newOpcode);
      } else if (constantThatMustBeLoaded ||
               (opValue && isa<GlobalValue>(opValue)))
        { // opValue is a constant that must be explicitly loaded into a reg
          assert(opValue);
          TmpInstruction* tmpReg = InsertCodeToLoadConstant(F, opValue, vmInstr,
                                                            MVec, target);
          minstr->SetMachineOperandVal(op, MachineOperand::MO_VirtualRegister,
                                       tmpReg);
        }
    }

  // Also, check for implicit operands used by the machine instruction
  // (no need to check those defined since they cannot be constants).
  // These include:
  // -- arguments to a Call
  // -- return value of a Return
  // Any such operand that is a constant value needs to be fixed also.
  // The current instructions with implicit refs (viz., Call and Return)
  // have no immediate fields, so the constant always needs to be loaded
  // into a register.
  bool isCall = instrInfo.isCall(opCode);
  unsigned lastCallArgNum = 0;          // unused if not a call
  CallArgsDescriptor* argDesc = NULL;   // unused if not a call
  if (isCall)
    argDesc = CallArgsDescriptor::get(minstr);

  for (unsigned i=0, N=minstr->getNumImplicitRefs(); i < N; ++i)
    if (isa<Constant>(minstr->getImplicitRef(i))) {
        Value* oldVal = minstr->getImplicitRef(i);
        TmpInstruction* tmpReg =
          InsertCodeToLoadConstant(F, oldVal, vmInstr, MVec, target);
        minstr->setImplicitRef(i, tmpReg);

        if (isCall) {
          // find and replace the argument in the CallArgsDescriptor
          unsigned i=lastCallArgNum;
          while (argDesc->getArgInfo(i).getArgVal() != oldVal)
            ++i;
          assert(i < argDesc->getNumArgs() &&
                 "Constant operands to a call *must* be in the arg list");
          lastCallArgNum = i;
          argDesc->getArgInfo(i).replaceArgVal(tmpReg);
        }
      }

  return MVec;
}

static inline void Add3OperandInstr(unsigned Opcode, InstructionNode* Node,
                                    std::vector<MachineInstr*>& mvec) {
  mvec.push_back(BuildMI(Opcode, 3).addReg(Node->leftChild()->getValue())
                                   .addReg(Node->rightChild()->getValue())
                                   .addRegDef(Node->getValue()));
}

/// IsZero - Check for a constant 0.
///
static inline bool IsZero(Value* idx) {
  return (isa<Constant>(idx) && cast<Constant>(idx)->isNullValue()) ||
         isa<UndefValue>(idx);
}

/// FoldGetElemChain - Fold a chain of GetElementPtr instructions containing
/// only constant offsets into an equivalent (Pointer, IndexVector) pair.
/// Returns the pointer Value, and stores the resulting IndexVector in argument
/// chainIdxVec. This is a helper function for FoldConstantIndices that does the
/// actual folding.
//
static Value*
FoldGetElemChain(InstrTreeNode* ptrNode, std::vector<Value*>& chainIdxVec,
                 bool lastInstHasLeadingNonZero) {
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
  InstructionNode* ptrChild = gepNode;
  while (ptrChild && (ptrChild->getOpLabel() == Instruction::GetElementPtr ||
                      ptrChild->getOpLabel() == GetElemPtrIdx)) {
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
    if (lastInstHasLeadingNonZero)
      if (! isa<SequentialType>(gepInst->getType()->getElementType()))
        break;   // cannot fold in any preceding getElementPtr instrs.

    // Check that all offsets are constant for this instruction
    for (OI = firstIdx; allConstantOffsets && OI != lastIdx; ++OI)
      allConstantOffsets = isa<ConstantInt>(*OI);

    if (allConstantOffsets) {
      // Get pointer value out of ptrChild.
      ptrVal = gepInst->getPointerOperand();

      // Insert its index vector at the start, skipping any leading [0]
      // Remember the old size to check if anything was inserted.
      unsigned oldSize = chainIdxVec.size();
      int firstIsZero = IsZero(*firstIdx);
      chainIdxVec.insert(chainIdxVec.begin(), firstIdx + firstIsZero, lastIdx);

      // Remember if it has leading zero index: it will be discarded later.
      if (oldSize < chainIdxVec.size())
        lastInstHasLeadingNonZero = !firstIsZero;

      // Mark the folded node so no code is generated for it.
      ((InstructionNode*) ptrChild)->markFoldedIntoParent();

      // Get the previous GEP instruction and continue trying to fold
      ptrChild = dyn_cast<InstructionNode>(ptrChild->leftChild());
    } else // cannot fold this getElementPtr instr. or any preceding ones
      break;
  }

  // If the first getElementPtr instruction had a leading [0], add it back.
  // Note that this instruction is the *last* one that was successfully
  // folded *and* contributed any indices, in the loop above.
  if (ptrVal && ! lastInstHasLeadingNonZero)
    chainIdxVec.insert(chainIdxVec.begin(), ConstantSInt::get(Type::LongTy,0));

  return ptrVal;
}

/// GetGEPInstArgs - Helper function for GetMemInstArgs that handles the
/// final getElementPtr instruction used by (or same as) the memory operation.
/// Extracts the indices of the current instruction and tries to fold in
/// preceding ones if all indices of the current one are constant.
///
static Value *GetGEPInstArgs(InstructionNode *gepNode,
                             std::vector<Value *> &idxVec,
                             bool &allConstantIndices) {
  allConstantIndices = true;
  GetElementPtrInst* gepI = cast<GetElementPtrInst>(gepNode->getInstruction());

  // Default pointer is the one from the current instruction.
  Value* ptrVal = gepI->getPointerOperand();
  InstrTreeNode* ptrChild = gepNode->leftChild();

  // Extract the index vector of the GEP instruction.
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
  if (allConstantIndices && !leadingNonZeroIdx)
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

/// GetMemInstArgs - Get the pointer value and the index vector for a memory
/// operation (GetElementPtr, Load, or Store).  If all indices of the given
/// memory operation are constant, fold in constant indices in a chain of
/// preceding GetElementPtr instructions (if any), and return the pointer value
/// of the first instruction in the chain. All folded instructions are marked so
/// no code is generated for them. Returns the pointer Value to use, and
/// returns the resulting IndexVector in idxVec. Sets allConstantIndices
/// to true/false if all indices are/aren't const.
///
static Value *GetMemInstArgs(InstructionNode *memInstrNode,
                             std::vector<Value*> &idxVec,
                             bool& allConstantIndices) {
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

static inline MachineOpCode
ChooseBprInstruction(const InstructionNode* instrNode) {
  MachineOpCode opCode;

  Instruction* setCCInstr =
    ((InstructionNode*) instrNode->leftChild())->getInstruction();

  switch(setCCInstr->getOpcode()) {
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
                      const BinaryOperator* setCCInstr) {
  MachineOpCode opCode = V9::INVALID_OPCODE;

  bool isSigned = setCCInstr->getOperand(0)->getType()->isSigned();

  if (isSigned) {
    switch(setCCInstr->getOpcode()) {
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
    switch(setCCInstr->getOpcode()) {
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
                       const BinaryOperator* setCCInstr) {
  MachineOpCode opCode = V9::INVALID_OPCODE;

  switch(setCCInstr->getOpcode()) {
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

// GetTmpForCC - Create a unique TmpInstruction for a boolean value,
// representing the CC register used by a branch on that value.
// For now, hack this using a little static cache of TmpInstructions.
// Eventually the entire BURG instruction selection should be put
// into a separate class that can hold such information.
// The static cache is not too bad because the memory for these
// TmpInstructions will be freed along with the rest of the Function anyway.
//
static TmpInstruction *GetTmpForCC (Value* boolVal, const Function *F,
                                    const Type* ccType,
                                    MachineCodeForInstruction& mcfi) {
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
    tmpI = new TmpInstruction(mcfi, ccType, boolVal);

  return tmpI;
}

static inline MachineOpCode
ChooseBccInstruction(const InstructionNode* instrNode, const Type*& setCCType) {
  InstructionNode* setCCNode = (InstructionNode*) instrNode->leftChild();
  assert(setCCNode->getOpLabel() == SetCCOp);
  BinaryOperator* setCCInstr =cast<BinaryOperator>(setCCNode->getInstruction());
  setCCType = setCCInstr->getOperand(0)->getType();

  if (setCCType->isFloatingPoint())
    return ChooseBFpccInstruction(instrNode, setCCInstr);
  else
    return ChooseBpccInstruction(instrNode, setCCInstr);
}

/// ChooseMovFpcciInstruction - WARNING: since this function has only one
/// caller, it always returns the opcode that expects an immediate and a
/// register. If this function is ever used in cases where an opcode that takes
/// two registers is required, then modify this function and use
/// convertOpcodeFromRegToImm() where required. It will be necessary to expand
/// convertOpcodeFromRegToImm() to handle the new cases of opcodes.
///
static inline MachineOpCode
ChooseMovFpcciInstruction(const InstructionNode* instrNode) {
  MachineOpCode opCode = V9::INVALID_OPCODE;

  switch(instrNode->getInstruction()->getOpcode()) {
  case Instruction::SetEQ: opCode = V9::MOVFEi;  break;
  case Instruction::SetNE: opCode = V9::MOVFNEi; break;
  case Instruction::SetLE: opCode = V9::MOVFLEi; break;
  case Instruction::SetGE: opCode = V9::MOVFGEi; break;
  case Instruction::SetLT: opCode = V9::MOVFLi;  break;
  case Instruction::SetGT: opCode = V9::MOVFGi;  break;
  default:
    assert(0 && "Unrecognized VM instruction!");
    break;
  }

  return opCode;
}

/// ChooseMovpcciForSetCC -- Choose a conditional-move instruction
/// based on the type of SetCC operation.
///
/// WARNING: like the previous function, this function always returns
/// the opcode that expects an immediate and a register.  See above.
///
static MachineOpCode ChooseMovpcciForSetCC(const InstructionNode* instrNode) {
  MachineOpCode opCode = V9::INVALID_OPCODE;

  const Type* opType = instrNode->leftChild()->getValue()->getType();
  assert(opType->isIntegral() || isa<PointerType>(opType));
  bool noSign = opType->isUnsigned() || isa<PointerType>(opType);

  switch(instrNode->getInstruction()->getOpcode()) {
  case Instruction::SetEQ: opCode = V9::MOVEi;                        break;
  case Instruction::SetLE: opCode = noSign? V9::MOVLEUi : V9::MOVLEi; break;
  case Instruction::SetGE: opCode = noSign? V9::MOVCCi  : V9::MOVGEi; break;
  case Instruction::SetLT: opCode = noSign? V9::MOVCSi  : V9::MOVLi;  break;
  case Instruction::SetGT: opCode = noSign? V9::MOVGUi  : V9::MOVGi;  break;
  case Instruction::SetNE: opCode = V9::MOVNEi;                       break;
  default: assert(0 && "Unrecognized LLVM instr!"); break;
  }

  return opCode;
}

/// ChooseMovpregiForSetCC -- Choose a conditional-move-on-register-value
/// instruction based on the type of SetCC operation.  These instructions
/// compare a register with 0 and perform the move is the comparison is true.
///
/// WARNING: like the previous function, this function it always returns
/// the opcode that expects an immediate and a register.  See above.
///
static MachineOpCode ChooseMovpregiForSetCC(const InstructionNode* instrNode) {
  MachineOpCode opCode = V9::INVALID_OPCODE;

  switch(instrNode->getInstruction()->getOpcode()) {
  case Instruction::SetEQ: opCode = V9::MOVRZi;  break;
  case Instruction::SetLE: opCode = V9::MOVRLEZi; break;
  case Instruction::SetGE: opCode = V9::MOVRGEZi; break;
  case Instruction::SetLT: opCode = V9::MOVRLZi;  break;
  case Instruction::SetGT: opCode = V9::MOVRGZi;  break;
  case Instruction::SetNE: opCode = V9::MOVRNZi; break;
  default: assert(0 && "Unrecognized VM instr!"); break;
  }

  return opCode;
}

static inline MachineOpCode
ChooseConvertToFloatInstr(const TargetMachine& target,
                          OpLabel vopCode, const Type* opType) {
  assert((vopCode == ToFloatTy || vopCode == ToDoubleTy) &&
         "Unrecognized convert-to-float opcode!");
  assert((opType->isIntegral() || opType->isFloatingPoint() ||
          isa<PointerType>(opType))
         && "Trying to convert a non-scalar type to FLOAT/DOUBLE?");

  MachineOpCode opCode = V9::INVALID_OPCODE;

  unsigned opSize = target.getTargetData().getTypeSize(opType);

  if (opType == Type::FloatTy)
    opCode = (vopCode == ToFloatTy? V9::NOP : V9::FSTOD);
  else if (opType == Type::DoubleTy)
    opCode = (vopCode == ToFloatTy? V9::FDTOS : V9::NOP);
  else if (opSize <= 4)
    opCode = (vopCode == ToFloatTy? V9::FITOS : V9::FITOD);
  else {
    assert(opSize == 8 && "Unrecognized type size > 4 and < 8!");
    opCode = (vopCode == ToFloatTy? V9::FXTOS : V9::FXTOD);
  }

  return opCode;
}

static inline MachineOpCode
ChooseConvertFPToIntInstr(const TargetMachine& target,
                          const Type* destType, const Type* opType) {
  assert((opType == Type::FloatTy || opType == Type::DoubleTy)
         && "This function should only be called for FLOAT or DOUBLE");
  assert((destType->isIntegral() || isa<PointerType>(destType))
         && "Trying to convert FLOAT/DOUBLE to a non-scalar type?");

  MachineOpCode opCode = V9::INVALID_OPCODE;

  unsigned destSize = target.getTargetData().getTypeSize(destType);

  if (destType == Type::UIntTy)
    assert(destType != Type::UIntTy && "Expand FP-to-uint beforehand.");
  else if (destSize <= 4)
    opCode = (opType == Type::FloatTy)? V9::FSTOI : V9::FDTOI;
  else {
    assert(destSize == 8 && "Unrecognized type size > 4 and < 8!");
    opCode = (opType == Type::FloatTy)? V9::FSTOX : V9::FDTOX;
  }

  return opCode;
}

static MachineInstr*
CreateConvertFPToIntInstr(const TargetMachine& target, Value* srcVal,
                          Value* destVal, const Type* destType) {
  MachineOpCode opCode = ChooseConvertFPToIntInstr(target, destType,
                                                   srcVal->getType());
  assert(opCode != V9::INVALID_OPCODE && "Expected to need conversion!");
  return BuildMI(opCode, 2).addReg(srcVal).addRegDef(destVal);
}

/// CreateCodeToConvertFloatToInt: Convert FP value to signed or unsigned
/// integer.  The FP value must be converted to the dest type in an FP register,
/// and the result is then copied from FP to int register via memory.  SPARC
/// does not have a float-to-uint conversion, only a float-to-int (fdtoi).
/// Since fdtoi converts to signed integers, any FP value V between MAXINT+1 and
/// MAXUNSIGNED (i.e., 2^31 <= V <= 2^32-1) would be converted incorrectly.
/// Therefore, for converting an FP value to uint32_t, we first need to convert
/// to uint64_t and then to uint32_t.
///
static void
CreateCodeToConvertFloatToInt(const TargetMachine& target,
                              Value* opVal, Instruction* destI,
                              std::vector<MachineInstr*>& mvec,
                              MachineCodeForInstruction& mcfi) {
  Function* F = destI->getParent()->getParent();

  // Create a temporary to represent the FP register into which the
  // int value will placed after conversion.  The type of this temporary
  // depends on the type of FP register to use: single-prec for a 32-bit
  // int or smaller; double-prec for a 64-bit int.
  size_t destSize = target.getTargetData().getTypeSize(destI->getType());

  const Type* castDestType = destI->getType(); // type for the cast instr result
  const Type* castDestRegType;          // type for cast instruction result reg
  TmpInstruction* destForCast;          // dest for cast instruction
  Instruction* fpToIntCopyDest = destI; // dest for fp-reg-to-int-reg copy instr

  // For converting an FP value to uint32_t, we first need to convert to
  // uint64_t and then to uint32_t, as explained above.
  if (destI->getType() == Type::UIntTy) {
    castDestType    = Type::ULongTy;       // use this instead of type of destI
    castDestRegType = Type::DoubleTy;      // uint64_t needs 64-bit FP register.
    destForCast     = new TmpInstruction(mcfi, castDestRegType, opVal);
    fpToIntCopyDest = new TmpInstruction(mcfi, castDestType, destForCast);
  } else {
    castDestRegType = (destSize > 4)? Type::DoubleTy : Type::FloatTy;
    destForCast = new TmpInstruction(mcfi, castDestRegType, opVal);
  }

  // Create the fp-to-int conversion instruction (src and dest regs are FP regs)
  mvec.push_back(CreateConvertFPToIntInstr(target, opVal, destForCast,
                                           castDestType));

  // Create the fpreg-to-intreg copy code
  CreateCodeToCopyFloatToInt(target, F, destForCast, fpToIntCopyDest, mvec,
                             mcfi);

  // Create the uint64_t to uint32_t conversion, if needed
  if (destI->getType() == Type::UIntTy)
    CreateZeroExtensionInstructions(target, F, fpToIntCopyDest, destI,
                                    /*numLowBits*/ 32, mvec, mcfi);
}

static inline MachineOpCode
ChooseAddInstruction(const InstructionNode* instrNode) {
  return ChooseAddInstructionByType(instrNode->getInstruction()->getType());
}

static inline MachineInstr*
CreateMovFloatInstruction(const InstructionNode* instrNode,
                          const Type* resultType) {
  return BuildMI((resultType == Type::FloatTy) ? V9::FMOVS : V9::FMOVD, 2)
                   .addReg(instrNode->leftChild()->getValue())
                   .addRegDef(instrNode->getValue());
}

static inline MachineInstr*
CreateAddConstInstruction(const InstructionNode* instrNode) {
  MachineInstr* minstr = NULL;

  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(isa<Constant>(constOp));

  // Cases worth optimizing are:
  // (1) Add with 0 for float or double: use an FMOV of appropriate type,
  //     instead of an FADD (1 vs 3 cycles).  There is no integer MOV.
  if (ConstantFP *FPC = dyn_cast<ConstantFP>(constOp)) {
    double dval = FPC->getValue();
    if (dval == 0.0)
      minstr = CreateMovFloatInstruction(instrNode,
                                        instrNode->getInstruction()->getType());
  }

  return minstr;
}

static inline MachineOpCode ChooseSubInstructionByType(const Type* resultType) {
  MachineOpCode opCode = V9::INVALID_OPCODE;

  if (resultType->isInteger() || isa<PointerType>(resultType)) {
      opCode = V9::SUBr;
  } else {
    switch(resultType->getTypeID()) {
    case Type::FloatTyID:  opCode = V9::FSUBS; break;
    case Type::DoubleTyID: opCode = V9::FSUBD; break;
    default: assert(0 && "Invalid type for SUB instruction"); break;
    }
  }

  return opCode;
}

static inline MachineInstr*
CreateSubConstInstruction(const InstructionNode* instrNode) {
  MachineInstr* minstr = NULL;

  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(isa<Constant>(constOp));

  // Cases worth optimizing are:
  // (1) Sub with 0 for float or double: use an FMOV of appropriate type,
  //     instead of an FSUB (1 vs 3 cycles).  There is no integer MOV.
  if (ConstantFP *FPC = dyn_cast<ConstantFP>(constOp)) {
    double dval = FPC->getValue();
    if (dval == 0.0)
      minstr = CreateMovFloatInstruction(instrNode,
                                        instrNode->getInstruction()->getType());
  }

  return minstr;
}

static inline MachineOpCode
ChooseFcmpInstruction(const InstructionNode* instrNode) {
  MachineOpCode opCode = V9::INVALID_OPCODE;

  Value* operand = ((InstrTreeNode*) instrNode->leftChild())->getValue();
  switch(operand->getType()->getTypeID()) {
  case Type::FloatTyID:  opCode = V9::FCMPS; break;
  case Type::DoubleTyID: opCode = V9::FCMPD; break;
  default: assert(0 && "Invalid type for FCMP instruction"); break;
  }

  return opCode;
}

/// BothFloatToDouble - Assumes that leftArg and rightArg of instrNode are both
/// cast instructions. Returns true if both are floats cast to double.
///
static inline bool BothFloatToDouble(const InstructionNode* instrNode) {
  InstrTreeNode* leftArg = instrNode->leftChild();
  InstrTreeNode* rightArg = instrNode->rightChild();
  InstrTreeNode* leftArgArg = leftArg->leftChild();
  InstrTreeNode* rightArgArg = rightArg->leftChild();
  assert(leftArg->getValue()->getType() == rightArg->getValue()->getType());
  return (leftArg->getValue()->getType() == Type::DoubleTy &&
          leftArgArg->getValue()->getType() == Type::FloatTy &&
          rightArgArg->getValue()->getType() == Type::FloatTy);
}

static inline MachineOpCode ChooseMulInstructionByType(const Type* resultType) {
  MachineOpCode opCode = V9::INVALID_OPCODE;

  if (resultType->isInteger())
    opCode = V9::MULXr;
  else
    switch(resultType->getTypeID()) {
    case Type::FloatTyID:  opCode = V9::FMULS; break;
    case Type::DoubleTyID: opCode = V9::FMULD; break;
    default: assert(0 && "Invalid type for MUL instruction"); break;
    }

  return opCode;
}

static inline MachineInstr*
CreateIntNegInstruction(const TargetMachine& target, Value* vreg) {
  return BuildMI(V9::SUBr, 3).addMReg(target.getRegInfo()->getZeroRegNum())
    .addReg(vreg).addRegDef(vreg);
}

static inline MachineInstr*
CreateIntNegInstruction(const TargetMachine& target, Value* vreg, Value *destreg) {
  return BuildMI(V9::SUBr, 3).addMReg(target.getRegInfo()->getZeroRegNum())
    .addReg(vreg).addRegDef(destreg);
}

/// CreateShiftInstructions - Create instruction sequence for any shift
/// operation. SLL or SLLX on an operand smaller than the integer reg. size
/// (64bits) requires a second instruction for explicit sign-extension. Note
/// that we only have to worry about a sign-bit appearing in the most
/// significant bit of the operand after shifting (e.g., bit 32 of Int or bit 16
/// of Short), so we do not have to worry about results that are as large as a
/// normal integer register.
///
static inline void
CreateShiftInstructions(const TargetMachine& target, Function* F,
                        MachineOpCode shiftOpCode, Value* argVal1,
                        Value* optArgVal2, /* Use optArgVal2 if not NULL */
                        unsigned optShiftNum, /* else use optShiftNum */
                        Instruction* destVal, std::vector<MachineInstr*>& mvec,
                        MachineCodeForInstruction& mcfi) {
  assert((optArgVal2 != NULL || optShiftNum <= 64) &&
         "Large shift sizes unexpected, but can be handled below: "
         "You need to check whether or not it fits in immed field below");

  // If this is a logical left shift of a type smaller than the standard
  // integer reg. size, we have to extend the sign-bit into upper bits
  // of dest, so we need to put the result of the SLL into a temporary.
  Value* shiftDest = destVal;
  unsigned opSize = target.getTargetData().getTypeSize(argVal1->getType());

  if ((shiftOpCode == V9::SLLr5 || shiftOpCode == V9::SLLXr6) && opSize < 8) {
    // put SLL result into a temporary
    shiftDest = new TmpInstruction(mcfi, argVal1, optArgVal2, "sllTmp");
  }

  MachineInstr* M = (optArgVal2 != NULL)
    ? BuildMI(shiftOpCode, 3).addReg(argVal1).addReg(optArgVal2)
                             .addReg(shiftDest, MachineOperand::Def)
    : BuildMI(shiftOpCode, 3).addReg(argVal1).addZImm(optShiftNum)
                             .addReg(shiftDest, MachineOperand::Def);
  mvec.push_back(M);

  if (shiftDest != destVal) {
    // extend the sign-bit of the result into all upper bits of dest
    assert(8*opSize <= 32 && "Unexpected type size > 4 and < IntRegSize?");
    CreateSignExtensionInstructions(target, F, shiftDest, destVal, 8*opSize,
                                    mvec, mcfi);
  }
}

/// CreateMulConstInstruction - Does not create any instructions if we
/// cannot exploit constant to create a cheaper instruction. This returns the
/// approximate cost of the instructions generated, which is used to pick the
/// cheapest when both operands are constant.
///
static unsigned
CreateMulConstInstruction(const TargetMachine &target, Function* F,
                          Value* lval, Value* rval, Instruction* destVal,
                          std::vector<MachineInstr*>& mvec,
                          MachineCodeForInstruction& mcfi) {
  // Use max. multiply cost, viz., cost of MULX
  unsigned cost = target.getInstrInfo()->minLatency(V9::MULXr);
  unsigned firstNewInstr = mvec.size();

  Value* constOp = rval;
  if (! isa<Constant>(constOp))
    return cost;

  // Cases worth optimizing are:
  // (1) Multiply by 0 or 1 for any type: replace with copy (ADD or FMOV)
  // (2) Multiply by 2^x for integer types: replace with Shift
  const Type* resultType = destVal->getType();

  if (resultType->isInteger() || isa<PointerType>(resultType)) {
    bool isValidConst;
    int64_t C = (int64_t) ConvertConstantToIntType(target, constOp,
                                                   constOp->getType(),
                                                   isValidConst);
    if (isValidConst) {
      bool needNeg = false;
      if (C < 0) {
        needNeg = true;
        C = -C;
      }
      TmpInstruction *tmpNeg = 0;

      if (C == 0 || C == 1) {
        cost = target.getInstrInfo()->minLatency(V9::ADDr);
        unsigned Zero = target.getRegInfo()->getZeroRegNum();
        MachineInstr* M;
        if (C == 0)
          M =BuildMI(V9::ADDr,3).addMReg(Zero).addMReg(Zero).addRegDef(destVal);
        else
          M = BuildMI(V9::ADDr,3).addReg(lval).addMReg(Zero).addRegDef(destVal);
        mvec.push_back(M);
      } else if (isPowerOf2_64(C)) {
        unsigned pow = Log2_64(C);
        if(!needNeg) {
        unsigned opSize = target.getTargetData().getTypeSize(resultType);
        MachineOpCode opCode = (opSize <= 32)? V9::SLLr5 : V9::SLLXr6;
        CreateShiftInstructions(target, F, opCode, lval, NULL, pow,
                                destVal, mvec, mcfi);
        }
        else {
          //Create tmp instruction to hold intermeidate value, since we need
          //to negate the result
          tmpNeg = new TmpInstruction(mcfi, lval);
          unsigned opSize = target.getTargetData().getTypeSize(resultType);
          MachineOpCode opCode = (opSize <= 32)? V9::SLLr5 : V9::SLLXr6;
          CreateShiftInstructions(target, F, opCode, lval, NULL, pow,
                                  tmpNeg, mvec, mcfi);
        }

      }

      if (mvec.size() > 0 && needNeg) {
        MachineInstr* M = 0;
        if(tmpNeg)
        // insert <reg = SUB 0, reg> after the instr to flip the sign
          M = CreateIntNegInstruction(target, tmpNeg, destVal);
        else
          M = CreateIntNegInstruction(target, destVal);
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
      cost += target.getInstrInfo()->minLatency(mvec[i]->getOpcode());
  }

  return cost;
}

/// CreateCheapestMulConstInstruction - Does not create any instructions
/// if we cannot exploit constant to create a cheaper instruction.
///
static inline void
CreateCheapestMulConstInstruction(const TargetMachine &target, Function* F,
                                  Value* lval, Value* rval,
                                  Instruction* destVal,
                                  std::vector<MachineInstr*>& mvec,
                                  MachineCodeForInstruction& mcfi) {
  Value* constOp;
  if (isa<Constant>(lval) && isa<Constant>(rval)) {
    // both operands are constant: evaluate and "set" in dest
    Constant* P = ConstantExpr::get(Instruction::Mul,
                                    cast<Constant>(lval),
                                    cast<Constant>(rval));
    CreateCodeToLoadConst (target, F, P, destVal, mvec, mcfi);
  }
  else if (isa<Constant>(rval))         // rval is constant, but not lval
    CreateMulConstInstruction(target, F, lval, rval, destVal, mvec, mcfi);
  else if (isa<Constant>(lval))         // lval is constant, but not rval
    CreateMulConstInstruction(target, F, lval, rval, destVal, mvec, mcfi);

  // else neither is constant
  return;
}

/// CreateMulInstruction - Returns NULL if we cannot exploit constant
/// to create a cheaper instruction.
///
static inline void
CreateMulInstruction(const TargetMachine &target, Function* F,
                     Value* lval, Value* rval, Instruction* destVal,
                     std::vector<MachineInstr*>& mvec,
                     MachineCodeForInstruction& mcfi,
                     MachineOpCode forceMulOp = -1) {
  unsigned L = mvec.size();
  CreateCheapestMulConstInstruction(target,F, lval, rval, destVal, mvec, mcfi);
  if (mvec.size() == L) {
    // no instructions were added so create MUL reg, reg, reg.
    // Use FSMULD if both operands are actually floats cast to doubles.
    // Otherwise, use the default opcode for the appropriate type.
    MachineOpCode mulOp = ((forceMulOp != -1)
                           ? forceMulOp
                           : ChooseMulInstructionByType(destVal->getType()));
    mvec.push_back(BuildMI(mulOp, 3).addReg(lval).addReg(rval)
                   .addRegDef(destVal));
  }
}

/// ChooseDivInstruction - Generate a divide instruction for Div or Rem.
/// For Rem, this assumes that the operand type will be signed if the result
/// type is signed.  This is correct because they must have the same sign.
///
static inline MachineOpCode
ChooseDivInstruction(TargetMachine &target, const InstructionNode* instrNode) {
  MachineOpCode opCode = V9::INVALID_OPCODE;

  const Type* resultType = instrNode->getInstruction()->getType();

  if (resultType->isInteger())
    opCode = resultType->isSigned()? V9::SDIVXr : V9::UDIVXr;
  else
    switch(resultType->getTypeID()) {
      case Type::FloatTyID:  opCode = V9::FDIVS; break;
      case Type::DoubleTyID: opCode = V9::FDIVD; break;
      default: assert(0 && "Invalid type for DIV instruction"); break;
      }

  return opCode;
}

/// CreateDivConstInstruction - Return if we cannot exploit constant to create
/// a cheaper instruction.
///
static void CreateDivConstInstruction(TargetMachine &target,
                                      const InstructionNode* instrNode,
                                      std::vector<MachineInstr*>& mvec) {
  Value* LHS  = instrNode->leftChild()->getValue();
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  if (!isa<Constant>(constOp))
    return;

  Instruction* destVal = instrNode->getInstruction();
  unsigned ZeroReg = target.getRegInfo()->getZeroRegNum();

  // Cases worth optimizing are:
  // (1) Divide by 1 for any type: replace with copy (ADD or FMOV)
  // (2) Divide by 2^x for integer types: replace with SR[L or A]{X}
  const Type* resultType = instrNode->getInstruction()->getType();

  if (resultType->isInteger()) {
    bool isValidConst;
    int64_t C = (int64_t) ConvertConstantToIntType(target, constOp,
                                                   constOp->getType(),
                                                   isValidConst);
    if (isValidConst) {
      bool needNeg = false;
      if (C < 0) {
        needNeg = true;
        C = -C;
      }

      if (C == 1) {
        mvec.push_back(BuildMI(V9::ADDr, 3).addReg(LHS).addMReg(ZeroReg)
                       .addRegDef(destVal));
      } else if (isPowerOf2_64(C)) {
        unsigned pow = Log2_64(C);
        unsigned opCode;
        Value* shiftOperand;
        unsigned opSize = target.getTargetData().getTypeSize(resultType);

        if (resultType->isSigned()) {
          // For N / 2^k, if the operand N is negative,
          // we need to add (2^k - 1) before right-shifting by k, i.e.,
          //
          //    (N / 2^k) = N >> k,               if N >= 0;
          //                (N + 2^k - 1) >> k,   if N < 0
          //
          // If N is <= 32 bits, use:
          //    sra N, 31, t1           // t1 = ~0,         if N < 0,  0 else
          //    srl t1, 32-k, t2        // t2 = 2^k - 1,    if N < 0,  0 else
          //    add t2, N, t3           // t3 = N + 2^k -1, if N < 0,  N else
          //    sra t3, k, result       // result = N / 2^k
          //
          // If N is 64 bits, use:
          //    srax N,  k-1,  t1       // t1 = sign bit in high k positions
          //    srlx t1, 64-k, t2       // t2 = 2^k - 1,    if N < 0,  0 else
          //    add t2, N, t3           // t3 = N + 2^k -1, if N < 0,  N else
          //    sra t3, k, result       // result = N / 2^k
          TmpInstruction *sraTmp, *srlTmp, *addTmp;
          MachineCodeForInstruction& mcfi
            = MachineCodeForInstruction::get(destVal);
          sraTmp = new TmpInstruction(mcfi, resultType, LHS, 0, "getSign");
          srlTmp = new TmpInstruction(mcfi, resultType, LHS, 0, "getPlus2km1");
          addTmp = new TmpInstruction(mcfi, resultType, LHS, srlTmp,"incIfNeg");

          // Create the SRA or SRAX instruction to get the sign bit
          mvec.push_back(BuildMI((opSize > 4)? V9::SRAXi6 : V9::SRAi5, 3)
                         .addReg(LHS)
                         .addSImm((resultType==Type::LongTy)? pow-1 : 31)
                         .addRegDef(sraTmp));

          // Create the SRL or SRLX instruction to get the sign bit
          mvec.push_back(BuildMI((opSize > 4)? V9::SRLXi6 : V9::SRLi5, 3)
                         .addReg(sraTmp)
                         .addSImm((resultType==Type::LongTy)? 64-pow : 32-pow)
                         .addRegDef(srlTmp));

          // Create the ADD instruction to add 2^pow-1 for negative values
          mvec.push_back(BuildMI(V9::ADDr, 3).addReg(LHS).addReg(srlTmp)
                         .addRegDef(addTmp));

          // Get the shift operand and "right-shift" opcode to do the divide
          shiftOperand = addTmp;
          opCode = (opSize > 4)? V9::SRAXi6 : V9::SRAi5;
        } else {
          // Get the shift operand and "right-shift" opcode to do the divide
          shiftOperand = LHS;
          opCode = (opSize > 4)? V9::SRLXi6 : V9::SRLi5;
        }

        // Now do the actual shift!
        mvec.push_back(BuildMI(opCode, 3).addReg(shiftOperand).addZImm(pow)
                       .addRegDef(destVal));
      }

      if (needNeg && (C == 1 || isPowerOf2_64(C))) {
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

static void CreateCodeForVariableSizeAlloca(const TargetMachine& target,
                                            Instruction* result, unsigned tsize,
                                            Value* numElementsVal,
                                            std::vector<MachineInstr*>& getMvec)
{
  Value* totalSizeVal;
  MachineInstr* M;
  MachineCodeForInstruction& mcfi = MachineCodeForInstruction::get(result);
  Function *F = result->getParent()->getParent();

  // Enforce the alignment constraints on the stack pointer at
  // compile time if the total size is a known constant.
  if (isa<Constant>(numElementsVal)) {
    bool isValid;
    int64_t numElem = (int64_t)
      ConvertConstantToIntType(target, numElementsVal,
                               numElementsVal->getType(), isValid);
    assert(isValid && "Unexpectedly large array dimension in alloca!");
    int64_t total = numElem * tsize;
    if (int extra= total % SparcV9FrameInfo::StackFrameSizeAlignment)
      total += SparcV9FrameInfo::StackFrameSizeAlignment - extra;
    totalSizeVal = ConstantSInt::get(Type::IntTy, total);
  } else {
    // The size is not a constant.  Generate code to compute it and
    // code to pad the size for stack alignment.
    // Create a Value to hold the (constant) element size
    Value* tsizeVal = ConstantSInt::get(Type::IntTy, tsize);

    // Create temporary values to hold the result of MUL, SLL, SRL
    // To pad `size' to next smallest multiple of 16:
    //          size = (size + 15) & (-16 = 0xfffffffffffffff0)
    TmpInstruction* tmpProd = new TmpInstruction(mcfi,numElementsVal, tsizeVal);
    TmpInstruction* tmpAdd15= new TmpInstruction(mcfi,numElementsVal, tmpProd);
    TmpInstruction* tmpAndf0= new TmpInstruction(mcfi,numElementsVal, tmpAdd15);

    // Instruction 1: mul numElements, typeSize -> tmpProd
    // This will optimize the MUL as far as possible.
    CreateMulInstruction(target, F, numElementsVal, tsizeVal, tmpProd, getMvec,
                         mcfi, -1);

    // Instruction 2: andn tmpProd, 0x0f -> tmpAndn
    getMvec.push_back(BuildMI(V9::ADDi, 3).addReg(tmpProd).addSImm(15)
                      .addReg(tmpAdd15, MachineOperand::Def));

    // Instruction 3: add tmpAndn, 0x10 -> tmpAdd16
    getMvec.push_back(BuildMI(V9::ANDi, 3).addReg(tmpAdd15).addSImm(-16)
                      .addReg(tmpAndf0, MachineOperand::Def));

    totalSizeVal = tmpAndf0;
  }

  // Get the constant offset from SP for dynamically allocated storage
  // and create a temporary Value to hold it.
  MachineFunction& mcInfo = MachineFunction::get(F);
  bool growUp;
  ConstantSInt* dynamicAreaOffset =
    ConstantSInt::get(Type::IntTy,
                    target.getFrameInfo()->getDynamicAreaOffset(mcInfo,growUp));
  assert(! growUp && "Has SPARC v9 stack frame convention changed?");

  unsigned SPReg = target.getRegInfo()->getStackPointer();

  // Instruction 2: sub %sp, totalSizeVal -> %sp
  getMvec.push_back(BuildMI(V9::SUBr, 3).addMReg(SPReg).addReg(totalSizeVal)
                    .addMReg(SPReg,MachineOperand::Def));

  // Instruction 3: add %sp, frameSizeBelowDynamicArea -> result
  getMvec.push_back(BuildMI(V9::ADDr,3).addMReg(SPReg).addReg(dynamicAreaOffset)
                    .addRegDef(result));
}

static void
CreateCodeForFixedSizeAlloca(const TargetMachine& target,
                             Instruction* result, unsigned tsize,
                             unsigned numElements,
                             std::vector<MachineInstr*>& getMvec) {
  assert(result && result->getParent() &&
         "Result value is not part of a function?");
  Function *F = result->getParent()->getParent();
  MachineFunction &mcInfo = MachineFunction::get(F);

  // If the alloca is of zero bytes (which is perfectly legal) we bump it up to
  // one byte.  This is unnecessary, but I really don't want to break any
  // fragile logic in this code.  FIXME.
  if (tsize == 0)
    tsize = 1;

  // Put the variable in the dynamically sized area of the frame if either:
  // (a) The offset is too large to use as an immediate in load/stores
  //     (check LDX because all load/stores have the same-size immed. field).
  // (b) The object is "large", so it could cause many other locals,
  //     spills, and temporaries to have large offsets.
  //     NOTE: We use LARGE = 8 * argSlotSize = 64 bytes.
  // You've gotta love having only 13 bits for constant offset values :-|.
  //
  unsigned paddedSize;
  int offsetFromFP = mcInfo.getInfo<SparcV9FunctionInfo>()->computeOffsetforLocalVar(result,
                                                                paddedSize,
                                                         tsize * numElements);

  if (((int)paddedSize) > 8 * SparcV9FrameInfo::SizeOfEachArgOnStack ||
      !target.getInstrInfo()->constantFitsInImmedField(V9::LDXi,offsetFromFP)) {
    CreateCodeForVariableSizeAlloca(target, result, tsize,
                                    ConstantSInt::get(Type::IntTy,numElements),
                                    getMvec);
    return;
  }

  // else offset fits in immediate field so go ahead and allocate it.
  offsetFromFP = mcInfo.getInfo<SparcV9FunctionInfo>()->allocateLocalVar(result, tsize *numElements);

  // Create a temporary Value to hold the constant offset.
  // This is needed because it may not fit in the immediate field.
  ConstantSInt* offsetVal = ConstantSInt::get(Type::IntTy, offsetFromFP);

  // Instruction 1: add %fp, offsetFromFP -> result
  unsigned FPReg = target.getRegInfo()->getFramePointer();
  getMvec.push_back(BuildMI(V9::ADDr, 3).addMReg(FPReg).addReg(offsetVal)
                    .addRegDef(result));
}

/// SetOperandsForMemInstr - Choose addressing mode for the given load or store
/// instruction.  Use [reg+reg] if it is an indexed reference, and the index
/// offset is not a constant or if it cannot fit in the offset field.  Use
/// [reg+offset] in all other cases.  This assumes that all array refs are
/// "lowered" to one of these forms:
///    %x = load (subarray*) ptr, constant      ; single constant offset
///    %x = load (subarray*) ptr, offsetVal     ; single non-constant offset
/// Generally, this should happen via strength reduction + LICM.  Also, strength
/// reduction should take care of using the same register for the loop index
/// variable and an array index, when that is profitable.
///
static void SetOperandsForMemInstr(unsigned Opcode,
                                   std::vector<MachineInstr*>& mvec,
                                   InstructionNode* vmInstrNode,
                                   const TargetMachine& target) {
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
  if (!idxVec.empty()) {
    const PointerType* ptrType = cast<PointerType>(ptrVal->getType());

    // If all indices are constant, compute the combined offset directly.
    if (allConstantIndices) {
      // Compute the offset value using the index vector. Create a
      // virtual reg. for it since it may not fit in the immed field.
      uint64_t offset = target.getTargetData().getIndexedOffset(ptrType,idxVec);
      valueForRegOffset = ConstantSInt::get(Type::LongTy, offset);
    } else {
      // There is at least one non-constant offset.  Therefore, this must
      // be an array ref, and must have been lowered to a single non-zero
      // offset.  (An extra leading zero offset, if any, can be ignored.)
      // Generate code sequence to compute address from index.
      bool firstIdxIsZero = IsZero(idxVec[0]);
      assert(idxVec.size() == 1U + firstIdxIsZero
             && "Array refs must be lowered before Instruction Selection");

      Value* idxVal = idxVec[firstIdxIsZero];

      std::vector<MachineInstr*> mulVec;
      Instruction* addr =
        new TmpInstruction(MachineCodeForInstruction::get(memInst),
                           Type::ULongTy, memInst);

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
                           -1);

      assert(mulVec.size() > 0 && "No multiply code created?");
      mvec.insert(mvec.end(), mulVec.begin(), mulVec.end());

      valueForRegOffset = addr;
    }
  } else {
    offsetOpType = MachineOperand::MO_SignExtendedImmed;
    smallConstOffset = 0;
  }

  // For STORE:
  //   Operand 0 is value, operand 1 is ptr, operand 2 is offset
  // For LOAD or GET_ELEMENT_PTR,
  //   Operand 0 is ptr, operand 1 is offset, operand 2 is result.
  unsigned offsetOpNum, ptrOpNum;
  MachineInstr *MI;
  if (memInst->getOpcode() == Instruction::Store) {
    if (offsetOpType == MachineOperand::MO_VirtualRegister) {
      MI = BuildMI(Opcode, 3).addReg(vmInstrNode->leftChild()->getValue())
                             .addReg(ptrVal).addReg(valueForRegOffset);
    } else {
      Opcode = convertOpcodeFromRegToImm(Opcode);
      MI = BuildMI(Opcode, 3).addReg(vmInstrNode->leftChild()->getValue())
                             .addReg(ptrVal).addSImm(smallConstOffset);
    }
  } else {
    if (offsetOpType == MachineOperand::MO_VirtualRegister) {
      MI = BuildMI(Opcode, 3).addReg(ptrVal).addReg(valueForRegOffset)
                             .addRegDef(memInst);
    } else {
      Opcode = convertOpcodeFromRegToImm(Opcode);
      MI = BuildMI(Opcode, 3).addReg(ptrVal).addSImm(smallConstOffset)
                             .addRegDef(memInst);
    }
  }
  mvec.push_back(MI);
}

/// ForwardOperand - Substitute operand `operandNum' of the instruction in
/// node `treeNode' in place of the use(s) of that instruction in node `parent'.
/// Check both explicit and implicit operands!  Also make sure to skip over a
/// parent who: (1) is a list node in the Burg tree, or (2) itself had its
/// results forwarded to its parent.
///
static void ForwardOperand (InstructionNode *treeNode, InstrTreeNode *parent,
                            int operandNum) {
  assert(treeNode && parent && "Invalid invocation of ForwardOperand");

  Instruction* unusedOp = treeNode->getInstruction();
  Value* fwdOp = unusedOp->getOperand(operandNum);

  // The parent itself may be a list node, so find the real parent instruction
  while (parent->getNodeType() != InstrTreeNode::NTInstructionNode) {
    parent = parent->parent();
    assert(parent && "ERROR: Non-instruction node has no parent in tree.");
  }
  InstructionNode* parentInstrNode = (InstructionNode*) parent;

  Instruction* userInstr = parentInstrNode->getInstruction();
  MachineCodeForInstruction &mvec = MachineCodeForInstruction::get(userInstr);

  // The parent's mvec would be empty if it was itself forwarded.
  // Recursively call ForwardOperand in that case...
  //
  if (mvec.size() == 0) {
    assert(parent->parent() != NULL &&
           "Parent could not have been forwarded, yet has no instructions?");
    ForwardOperand(treeNode, parent->parent(), operandNum);
  } else {
    for (unsigned i=0, N=mvec.size(); i < N; i++) {
      MachineInstr* minstr = mvec[i];
      for (unsigned i=0, numOps=minstr->getNumOperands(); i < numOps; ++i) {
        const MachineOperand& mop = minstr->getOperand(i);
        if (mop.getType() == MachineOperand::MO_VirtualRegister &&
            mop.getVRegValue() == unusedOp) {
          minstr->SetMachineOperandVal(i, MachineOperand::MO_VirtualRegister,
                                       fwdOp);
        }
      }

      for (unsigned i=0,numOps=minstr->getNumImplicitRefs(); i<numOps; ++i)
        if (minstr->getImplicitRef(i) == unusedOp)
          minstr->setImplicitRef(i, fwdOp);
    }
  }
}

/// AllUsesAreBranches - Returns true if all the uses of I are
/// Branch instructions, false otherwise.
///
inline bool AllUsesAreBranches(const Instruction* I) {
  for (Value::use_const_iterator UI=I->use_begin(), UE=I->use_end();
       UI != UE; ++UI)
    if (! isa<TmpInstruction>(*UI)     // ignore tmp instructions here
        && cast<Instruction>(*UI)->getOpcode() != Instruction::Br)
      return false;
  return true;
}

/// CodeGenIntrinsic - Generate code for any intrinsic that needs a special
/// code sequence instead of a regular call.  If not that kind of intrinsic, do
/// nothing. Returns true if code was generated, otherwise false.
///
static bool CodeGenIntrinsic(Intrinsic::ID iid, CallInst &callInstr,
                             TargetMachine &target,
                             std::vector<MachineInstr*>& mvec) {
  switch (iid) {
  default:
    assert(0 && "Unknown intrinsic function call should have been lowered!");
  case Intrinsic::vastart: {
    // Get the address of the first incoming vararg argument on the stack
    Function* func = cast<Function>(callInstr.getParent()->getParent());
    int numFixedArgs   = func->getFunctionType()->getNumParams();
    int fpReg          = SparcV9::i6;
    int firstVarArgOff = numFixedArgs * 8 +
                         SparcV9FrameInfo::FirstIncomingArgOffsetFromFP;
    //What oh what do we pass to TmpInstruction?
    MachineCodeForInstruction& m = MachineCodeForInstruction::get(&callInstr);
    TmpInstruction* T = new TmpInstruction(m, callInstr.getOperand(1)->getType());
    mvec.push_back(BuildMI(V9::ADDi, 3).addMReg(fpReg).addSImm(firstVarArgOff).addRegDef(T));
    mvec.push_back(BuildMI(V9::STXr, 3).addReg(T).addReg(callInstr.getOperand(1)).addSImm(0));
    return true;
  }

  case Intrinsic::vaend:
    return true;                        // no-op on SparcV9

  case Intrinsic::vacopy:
    {
      MachineCodeForInstruction& m1 = MachineCodeForInstruction::get(&callInstr);
      TmpInstruction* VReg =
        new TmpInstruction(m1, callInstr.getOperand(1)->getType());

      // Simple store of current va_list (arg2) to new va_list (arg1)
      mvec.push_back(BuildMI(V9::LDXi, 3).
                     addReg(callInstr.getOperand(2)).addSImm(0).addRegDef(VReg));
      mvec.push_back(BuildMI(V9::STXi, 3).
                     addReg(VReg).addReg(callInstr.getOperand(1)).addSImm(0));
      return true;
    }
  }
}

/// ThisIsAChainRule - returns true if the given  BURG rule is a chain rule.
///
extern bool ThisIsAChainRule(int eruleno) {
  switch(eruleno) {
    case 111:   // stmt:  reg
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
      break;
    }
  return false;
}

/// GetInstructionsByRule - Choose machine instructions for the
/// SPARC V9 according to the patterns chosen by the BURG-generated parser.
/// This is where most of the work in the V9 instruction selector gets done.
///
void GetInstructionsByRule(InstructionNode* subtreeRoot, int ruleForNode,
                           short* nts, TargetMachine &target,
                           std::vector<MachineInstr*>& mvec) {
  bool checkCast = false;               // initialize here to use fall-through
  bool maskUnsignedResult = false;
  int nextRule;
  int forwardOperandNum = -1;
  unsigned allocaSize = 0;
  MachineInstr* M, *M2;
  unsigned L;
  bool foldCase = false;

  mvec.clear();

  // If the code for this instruction was folded into the parent (user),
  // then do nothing!
  if (subtreeRoot->isFoldedIntoParent())
    return;

  // Let's check for chain rules outside the switch so that we don't have
  // to duplicate the list of chain rule production numbers here again
  if (ThisIsAChainRule(ruleForNode)) {
    // Chain rules have a single nonterminal on the RHS.
    // Get the rule that matches the RHS non-terminal and use that instead.
    assert(nts[0] && ! nts[1]
           && "A chain rule should have only one RHS non-terminal!");
    nextRule = burm_rule(subtreeRoot->state, nts[0]);
    nts = burm_nts[nextRule];
    GetInstructionsByRule(subtreeRoot, nextRule, nts, target, mvec);
  } else {
    switch(ruleForNode) {
      case 1:   // stmt:   Ret
      case 2:   // stmt:   RetValue(reg)
      {         // NOTE: Prepass of register allocation is responsible
                //       for moving return value to appropriate register.
                // Copy the return value to the required return register.
                // Mark the return Value as an implicit ref of the RET instr..
                // Mark the return-address register as a hidden virtual reg.
                // Finally put a NOP in the delay slot.
        ReturnInst *returnInstr=cast<ReturnInst>(subtreeRoot->getInstruction());
        Value* retVal = returnInstr->getReturnValue();
        MachineCodeForInstruction& mcfi =
          MachineCodeForInstruction::get(returnInstr);

        // Create a hidden virtual reg to represent the return address register
        // used by the machine instruction but not represented in LLVM.
        Instruction* returnAddrTmp = new TmpInstruction(mcfi, returnInstr);

        MachineInstr* retMI =
          BuildMI(V9::JMPLRETi, 3).addReg(returnAddrTmp).addSImm(8)
          .addMReg(target.getRegInfo()->getZeroRegNum(), MachineOperand::Def);

        // If there is a value to return, we need to:
        // (a) Sign-extend the value if it is smaller than 8 bytes (reg size)
        // (b) Insert a copy to copy the return value to the appropriate reg.
        //     -- For FP values, create a FMOVS or FMOVD instruction
        //     -- For non-FP values, create an add-with-0 instruction
        if (retVal != NULL) {
          const SparcV9RegInfo& regInfo =
            (SparcV9RegInfo&) *target.getRegInfo();
          const Type* retType = retVal->getType();
          unsigned regClassID = regInfo.getRegClassIDOfType(retType);
          unsigned retRegNum = (retType->isFloatingPoint()
                                ? (unsigned) SparcV9FloatRegClass::f0
                                : (unsigned) SparcV9IntRegClass::i0);
          retRegNum = regInfo.getUnifiedRegNum(regClassID, retRegNum);

          // Insert sign-extension instructions for small signed values.
          Value* retValToUse = retVal;
          if (retType->isIntegral() && retType->isSigned()) {
            unsigned retSize = target.getTargetData().getTypeSize(retType);
            if (retSize <= 4) {
              // Create a temporary virtual reg. to hold the sign-extension.
              retValToUse = new TmpInstruction(mcfi, retVal);

              // Sign-extend retVal and put the result in the temporary reg.
              CreateSignExtensionInstructions
                (target, returnInstr->getParent()->getParent(),
                 retVal, retValToUse, 8*retSize, mvec, mcfi);
            }
          }

          // (b) Now, insert a copy to to the appropriate register:
          //     -- For FP values, create a FMOVS or FMOVD instruction
          //     -- For non-FP values, create an add-with-0 instruction
          // First, create a virtual register to represent the register and
          // mark this vreg as being an implicit operand of the ret MI.
          TmpInstruction* retVReg =
            new TmpInstruction(mcfi, retValToUse, NULL, "argReg");

          retMI->addImplicitRef(retVReg);

          if (retType->isFloatingPoint())
            M = (BuildMI(retType==Type::FloatTy? V9::FMOVS : V9::FMOVD, 2)
                 .addReg(retValToUse).addReg(retVReg, MachineOperand::Def));
          else
            M = (BuildMI(ChooseAddInstructionByType(retType), 3)
                 .addReg(retValToUse).addSImm((int64_t) 0)
                 .addReg(retVReg, MachineOperand::Def));

          // Mark the operand with the register it should be assigned
          M->SetRegForOperand(M->getNumOperands()-1, retRegNum);
          retMI->SetRegForImplicitRef(retMI->getNumImplicitRefs()-1, retRegNum);

          mvec.push_back(M);
        }

        // Now insert the RET instruction and a NOP for the delay slot
        mvec.push_back(retMI);
        mvec.push_back(BuildMI(V9::NOP, 0));

        break;
      }

      case 3:   // stmt:   Store(reg,reg)
      case 4:   // stmt:   Store(reg,ptrreg)
        SetOperandsForMemInstr(ChooseStoreInstruction(
                        subtreeRoot->leftChild()->getValue()->getType()),
                               mvec, subtreeRoot, target);
        break;

      case 5:   // stmt:   BrUncond
        {
          BranchInst *BI = cast<BranchInst>(subtreeRoot->getInstruction());
          mvec.push_back(BuildMI(V9::BA, 1).addPCDisp(BI->getSuccessor(0)));

          // delay slot
          mvec.push_back(BuildMI(V9::NOP, 0));
          break;
        }

      case 206: // stmt:   BrCond(setCCconst)
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
            && ConvertConstantToIntType(target,
                             constVal, constVal->getType(), isValidConst) == 0
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

      case 6:   // stmt:   BrCond(setCC)
      { // bool => boolean was computed with SetCC.
        // The branch to use depends on whether it is FP, signed, or unsigned.
        // If it is an integer CC, we also need to find the unique
        // TmpInstruction representing that CC.
        //
        BranchInst* brInst = cast<BranchInst>(subtreeRoot->getInstruction());
        const Type* setCCType;
        unsigned Opcode = ChooseBccInstruction(subtreeRoot, setCCType);
        Value* ccValue = GetTmpForCC(subtreeRoot->leftChild()->getValue(),
                                     brInst->getParent()->getParent(),
                                     setCCType,
                                     MachineCodeForInstruction::get(brInst));
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

      case 208: // stmt:   BrCond(boolconst)
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

      case   8: // stmt:   BrCond(boolreg)
      { // boolreg   => boolean is recorded in an integer register.
        //              Use branch-on-integer-register instruction.
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

      case 9:   // stmt:   Switch(reg)
        assert(0 && "*** SWITCH instruction is not implemented yet.");
        break;

      case 10:  // reg:   VRegList(reg, reg)
        assert(0 && "VRegList should never be the topmost non-chain rule");
        break;

      case 21:  // bool:  Not(bool,reg): Compute with a conditional-move-on-reg
      { // First find the unary operand. It may be left or right, usually right.
        Instruction* notI = subtreeRoot->getInstruction();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(subtreeRoot->getInstruction()));
        unsigned ZeroReg = target.getRegInfo()->getZeroRegNum();

        // Unconditionally set register to 0
        mvec.push_back(BuildMI(V9::SETHI, 2).addZImm(0).addRegDef(notI));

        // Now conditionally move 1 into the register.
        // Mark the register as a use (as well as a def) because the old
        // value will be retained if the condition is false.
        mvec.push_back(BuildMI(V9::MOVRZi, 3).addReg(notArg).addZImm(1)
                       .addReg(notI, MachineOperand::UseAndDef));

        break;
      }

      case 421: // reg:   BNot(reg,reg): Compute as reg = reg XOR-NOT 0
      { // First find the unary operand. It may be left or right, usually right.
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(subtreeRoot->getInstruction()));
        unsigned ZeroReg = target.getRegInfo()->getZeroRegNum();
        mvec.push_back(BuildMI(V9::XNORr, 3).addReg(notArg).addMReg(ZeroReg)
                                       .addRegDef(subtreeRoot->getValue()));
        break;
      }

      case 322: // reg:   Not(tobool, reg):
        // Fold CAST-TO-BOOL with NOT by inverting the sense of cast-to-bool
        foldCase = true;
        // Just fall through!

      case 22:  // reg:   ToBoolTy(reg):
      {
        Instruction* castI = subtreeRoot->getInstruction();
        Value* opVal = subtreeRoot->leftChild()->getValue();
        MachineCodeForInstruction &mcfi = MachineCodeForInstruction::get(castI);
        TmpInstruction* tempReg =
          new TmpInstruction(mcfi, opVal);



        assert(opVal->getType()->isIntegral() ||
               isa<PointerType>(opVal->getType()));

        // Unconditionally set register to 0
        mvec.push_back(BuildMI(V9::SETHI, 2).addZImm(0).addRegDef(castI));

        // Now conditionally move 1 into the register.
        // Mark the register as a use (as well as a def) because the old
        // value will be retained if the condition is false.
        MachineOpCode opCode = foldCase? V9::MOVRZi : V9::MOVRNZi;
        mvec.push_back(BuildMI(opCode, 3).addReg(opVal).addZImm(1)
                       .addReg(castI, MachineOperand::UseAndDef));

        break;
      }

      case 23:  // reg:   ToUByteTy(reg)
      case 24:  // reg:   ToSByteTy(reg)
      case 25:  // reg:   ToUShortTy(reg)
      case 26:  // reg:   ToShortTy(reg)
      case 27:  // reg:   ToUIntTy(reg)
      case 28:  // reg:   ToIntTy(reg)
      case 29:  // reg:   ToULongTy(reg)
      case 30:  // reg:   ToLongTy(reg)
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
        // -- If operand is smaller than destination, zero-extend or sign-extend
        //    according to the signedness of the *operand*: source decides:
        //    (1) If operand is signed, sign-extend it.
        //        If dest is unsigned, zero-ext the result!
        //    (2) If operand is unsigned, our current invariant is that
        //        it's high bits are correct, so zero-extension is not needed.
        //
        // -- If operand is same size as or larger than destination,
        //    zero-extend or sign-extend according to the signedness of
        //    the *destination*: destination decides:
        //    (1) If destination is signed, sign-extend (truncating if needed)
        //        This choice is implementation defined.  We sign-extend the
        //        operand, which matches both Sun's cc and gcc3.2.
        //    (2) If destination is unsigned, zero-extend (truncating if needed)
        //======================================================================

        Instruction* destI =  subtreeRoot->getInstruction();
        Function* currentFunc = destI->getParent()->getParent();
        MachineCodeForInstruction& mcfi=MachineCodeForInstruction::get(destI);

        Value* opVal = subtreeRoot->leftChild()->getValue();
        const Type* opType = opVal->getType();
        const Type* destType = destI->getType();
        unsigned opSize   = target.getTargetData().getTypeSize(opType);
        unsigned destSize = target.getTargetData().getTypeSize(destType);

        bool isIntegral = opType->isIntegral() || isa<PointerType>(opType);

        if (opType == Type::BoolTy ||
            opType == destType ||
            isIntegral && opSize == destSize && opSize == 8) {
          // nothing to do in all these cases
          forwardOperandNum = 0;          // forward first operand to user

        } else if (opType->isFloatingPoint()) {

          CreateCodeToConvertFloatToInt(target, opVal, destI, mvec, mcfi);
          if (destI->getType()->isUnsigned() && destI->getType() !=Type::UIntTy)
            maskUnsignedResult = true; // not handled by fp->int code

        } else if (isIntegral) {

          bool opSigned     = opType->isSigned();
          bool destSigned   = destType->isSigned();
          unsigned extSourceInBits = 8 * std::min<unsigned>(opSize, destSize);

          assert(! (opSize == destSize && opSigned == destSigned) &&
                 "How can different int types have same size and signedness?");

          bool signExtend = (opSize <  destSize && opSigned ||
                             opSize >= destSize && destSigned);

          bool signAndZeroExtend = (opSize < destSize && destSize < 8u &&
                                    opSigned && !destSigned);
          assert(!signAndZeroExtend || signExtend);

          bool zeroExtendOnly = opSize >= destSize && !destSigned;
          assert(!zeroExtendOnly || !signExtend);

          if (signExtend) {
            Value* signExtDest = (signAndZeroExtend
                                  ? new TmpInstruction(mcfi, destType, opVal)
                                  : destI);

            CreateSignExtensionInstructions
              (target, currentFunc,opVal,signExtDest,extSourceInBits,mvec,mcfi);

            if (signAndZeroExtend)
              CreateZeroExtensionInstructions
              (target, currentFunc, signExtDest, destI, 8*destSize, mvec, mcfi);
          }
          else if (zeroExtendOnly) {
            CreateZeroExtensionInstructions
              (target, currentFunc, opVal, destI, extSourceInBits, mvec, mcfi);
          }
          else
            forwardOperandNum = 0;          // forward first operand to user

        } else
          assert(0 && "Unrecognized operand type for convert-to-integer");

        break;
      }

      case  31: // reg:   ToFloatTy(reg):
      case  32: // reg:   ToDoubleTy(reg):
      case 232: // reg:   ToDoubleTy(Constant):

        // If this instruction has a parent (a user) in the tree
        // and the user is translated as an FsMULd instruction,
        // then the cast is unnecessary.  So check that first.
        // In the future, we'll want to do the same for the FdMULq instruction,
        // so do the check here instead of only for ToFloatTy(reg).
        //
        if (subtreeRoot->parent() != NULL) {
          const MachineCodeForInstruction& mcfi =
            MachineCodeForInstruction::get(
                cast<InstructionNode>(subtreeRoot->parent())->getInstruction());
          if (mcfi.size() == 0 || mcfi.front()->getOpcode() == V9::FSMULD)
            forwardOperandNum = 0;    // forward first operand to user
        }

        if (forwardOperandNum != 0) {    // we do need the cast
          Value* leftVal = subtreeRoot->leftChild()->getValue();
          const Type* opType = leftVal->getType();
          MachineOpCode opCode=ChooseConvertToFloatInstr(target,
                                       subtreeRoot->getOpLabel(), opType);
          if (opCode == V9::NOP) {      // no conversion needed
            forwardOperandNum = 0;      // forward first operand to user
          } else {
            // If the source operand is a non-FP type it must be
            // first copied from int to float register via memory!
            Instruction *dest = subtreeRoot->getInstruction();
            Value* srcForCast;
            int n = 0;
            if (! opType->isFloatingPoint()) {
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
              MachineCodeForInstruction &destMCFI =
                MachineCodeForInstruction::get(dest);
              srcForCast = new TmpInstruction(destMCFI, tmpTypeToUse, dest);

              CreateCodeToCopyIntToFloat(target,
                         dest->getParent()->getParent(),
                         leftVal, cast<Instruction>(srcForCast),
                         mvec, destMCFI);
            } else
              srcForCast = leftVal;

            M = BuildMI(opCode, 2).addReg(srcForCast).addRegDef(dest);
            mvec.push_back(M);
          }
        }
        break;

      case 19:  // reg:   ToArrayTy(reg):
      case 20:  // reg:   ToPointerTy(reg):
        forwardOperandNum = 0;          // forward first operand to user
        break;

      case 233: // reg:   Add(reg, Constant)
        maskUnsignedResult = true;
        M = CreateAddConstInstruction(subtreeRoot);
        if (M != NULL) {
          mvec.push_back(M);
          break;
        }
        // ELSE FALL THROUGH

      case 33:  // reg:   Add(reg, reg)
        maskUnsignedResult = true;
        Add3OperandInstr(ChooseAddInstruction(subtreeRoot), subtreeRoot, mvec);
        break;

      case 234: // reg:   Sub(reg, Constant)
        maskUnsignedResult = true;
        M = CreateSubConstInstruction(subtreeRoot);
        if (M != NULL) {
          mvec.push_back(M);
          break;
        }
        // ELSE FALL THROUGH

      case 34:  // reg:   Sub(reg, reg)
        maskUnsignedResult = true;
        Add3OperandInstr(ChooseSubInstructionByType(
                                   subtreeRoot->getInstruction()->getType()),
                         subtreeRoot, mvec);
        break;

      case 135: // reg:   Mul(todouble, todouble)
        checkCast = true;
        // FALL THROUGH

      case 35:  // reg:   Mul(reg, reg)
      {
        maskUnsignedResult = true;
        MachineOpCode forceOp = ((checkCast && BothFloatToDouble(subtreeRoot))
                                 ? (MachineOpCode)V9::FSMULD
                                 : -1);
        Instruction* mulInstr = subtreeRoot->getInstruction();
        CreateMulInstruction(target, mulInstr->getParent()->getParent(),
                             subtreeRoot->leftChild()->getValue(),
                             subtreeRoot->rightChild()->getValue(),
                             mulInstr, mvec,
                             MachineCodeForInstruction::get(mulInstr),forceOp);
        break;
      }
      case 335: // reg:   Mul(todouble, todoubleConst)
        checkCast = true;
        // FALL THROUGH

      case 235: // reg:   Mul(reg, Constant)
      {
        maskUnsignedResult = true;
        MachineOpCode forceOp = ((checkCast && BothFloatToDouble(subtreeRoot))
                                 ? (MachineOpCode)V9::FSMULD
                                 : -1);
        Instruction* mulInstr = subtreeRoot->getInstruction();
        CreateMulInstruction(target, mulInstr->getParent()->getParent(),
                             subtreeRoot->leftChild()->getValue(),
                             subtreeRoot->rightChild()->getValue(),
                             mulInstr, mvec,
                             MachineCodeForInstruction::get(mulInstr),
                             forceOp);
        break;
      }
      case 236: // reg:   Div(reg, Constant)
        maskUnsignedResult = true;
        L = mvec.size();
        CreateDivConstInstruction(target, subtreeRoot, mvec);
        if (mvec.size() > L)
          break;
        // ELSE FALL THROUGH

      case 36:  // reg:   Div(reg, reg)
      {
        maskUnsignedResult = true;

        // If either operand of divide is smaller than 64 bits, we have
        // to make sure the unused top bits are correct because they affect
        // the result.  These bits are already correct for unsigned values.
        // They may be incorrect for signed values, so sign extend to fill in.
        Instruction* divI = subtreeRoot->getInstruction();
        Value* divOp1 = subtreeRoot->leftChild()->getValue();
        Value* divOp2 = subtreeRoot->rightChild()->getValue();
        Value* divOp1ToUse = divOp1;
        Value* divOp2ToUse = divOp2;
        if (divI->getType()->isSigned()) {
          unsigned opSize=target.getTargetData().getTypeSize(divI->getType());
          if (opSize < 8) {
            MachineCodeForInstruction& mcfi=MachineCodeForInstruction::get(divI);
            divOp1ToUse = new TmpInstruction(mcfi, divOp1);
            divOp2ToUse = new TmpInstruction(mcfi, divOp2);
            CreateSignExtensionInstructions(target,
                                              divI->getParent()->getParent(),
                                              divOp1, divOp1ToUse,
                                              8*opSize, mvec, mcfi);
            CreateSignExtensionInstructions(target,
                                              divI->getParent()->getParent(),
                                              divOp2, divOp2ToUse,
                                              8*opSize, mvec, mcfi);
          }
        }

        mvec.push_back(BuildMI(ChooseDivInstruction(target, subtreeRoot), 3)
                       .addReg(divOp1ToUse)
                       .addReg(divOp2ToUse)
                       .addRegDef(divI));

        break;
      }

      case  37: // reg:   Rem(reg, reg)
      case 237: // reg:   Rem(reg, Constant)
      {
        maskUnsignedResult = true;

        Instruction* remI   = subtreeRoot->getInstruction();
        Value* divOp1 = subtreeRoot->leftChild()->getValue();
        Value* divOp2 = subtreeRoot->rightChild()->getValue();

        MachineCodeForInstruction& mcfi = MachineCodeForInstruction::get(remI);

        // If second operand of divide is smaller than 64 bits, we have
        // to make sure the unused top bits are correct because they affect
        // the result.  These bits are already correct for unsigned values.
        // They may be incorrect for signed values, so sign extend to fill in.
        //
        Value* divOpToUse = divOp2;
        if (divOp2->getType()->isSigned()) {
          unsigned opSize=target.getTargetData().getTypeSize(divOp2->getType());
          if (opSize < 8) {
            divOpToUse = new TmpInstruction(mcfi, divOp2);
            CreateSignExtensionInstructions(target,
                                              remI->getParent()->getParent(),
                                              divOp2, divOpToUse,
                                              8*opSize, mvec, mcfi);
          }
        }

        // Now compute: result = rem V1, V2 as:
        //      result = V1 - (V1 / signExtend(V2)) * signExtend(V2)
        //
        TmpInstruction* quot = new TmpInstruction(mcfi, divOp1, divOpToUse);
        TmpInstruction* prod = new TmpInstruction(mcfi, quot, divOpToUse);

        mvec.push_back(BuildMI(ChooseDivInstruction(target, subtreeRoot), 3)
                       .addReg(divOp1).addReg(divOpToUse).addRegDef(quot));

        mvec.push_back(BuildMI(ChooseMulInstructionByType(remI->getType()), 3)
                       .addReg(quot).addReg(divOpToUse).addRegDef(prod));

        mvec.push_back(BuildMI(ChooseSubInstructionByType(remI->getType()), 3)
                       .addReg(divOp1).addReg(prod).addRegDef(remI));

        break;
      }

      case  38: // bool:   And(bool, bool)
      case 138: // bool:   And(bool, not)
      case 238: // bool:   And(bool, boolconst)
      case 338: // reg :   BAnd(reg, reg)
      case 538: // reg :   BAnd(reg, Constant)
        Add3OperandInstr(V9::ANDr, subtreeRoot, mvec);
        break;

      case 438: // bool:   BAnd(bool, bnot)
      { // Use the argument of NOT as the second argument!
        // Mark the NOT node so that no code is generated for it.
        // If the type is boolean, set 1 or 0 in the result register.
        InstructionNode* notNode = (InstructionNode*) subtreeRoot->rightChild();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(notNode->getInstruction()));
        notNode->markFoldedIntoParent();
        Value *lhs = subtreeRoot->leftChild()->getValue();
        Value *dest = subtreeRoot->getValue();
        mvec.push_back(BuildMI(V9::ANDNr, 3).addReg(lhs).addReg(notArg)
                                       .addReg(dest, MachineOperand::Def));

        if (notArg->getType() == Type::BoolTy) {
          // set 1 in result register if result of above is non-zero
          mvec.push_back(BuildMI(V9::MOVRNZi, 3).addReg(dest).addZImm(1)
                         .addReg(dest, MachineOperand::UseAndDef));
        }

        break;
      }

      case  39: // bool:   Or(bool, bool)
      case 139: // bool:   Or(bool, not)
      case 239: // bool:   Or(bool, boolconst)
      case 339: // reg :   BOr(reg, reg)
      case 539: // reg :   BOr(reg, Constant)
        Add3OperandInstr(V9::ORr, subtreeRoot, mvec);
        break;

      case 439: // bool:   BOr(bool, bnot)
      { // Use the argument of NOT as the second argument!
        // Mark the NOT node so that no code is generated for it.
        // If the type is boolean, set 1 or 0 in the result register.
        InstructionNode* notNode = (InstructionNode*) subtreeRoot->rightChild();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(notNode->getInstruction()));
        notNode->markFoldedIntoParent();
        Value *lhs = subtreeRoot->leftChild()->getValue();
        Value *dest = subtreeRoot->getValue();

        mvec.push_back(BuildMI(V9::ORNr, 3).addReg(lhs).addReg(notArg)
                       .addReg(dest, MachineOperand::Def));

        if (notArg->getType() == Type::BoolTy) {
          // set 1 in result register if result of above is non-zero
          mvec.push_back(BuildMI(V9::MOVRNZi, 3).addReg(dest).addZImm(1)
                         .addReg(dest, MachineOperand::UseAndDef));
        }

        break;
      }

      case  40: // bool:   Xor(bool, bool)
      case 140: // bool:   Xor(bool, not)
      case 240: // bool:   Xor(bool, boolconst)
      case 340: // reg :   BXor(reg, reg)
      case 540: // reg :   BXor(reg, Constant)
        Add3OperandInstr(V9::XORr, subtreeRoot, mvec);
        break;

      case 440: // bool:   BXor(bool, bnot)
      { // Use the argument of NOT as the second argument!
        // Mark the NOT node so that no code is generated for it.
        // If the type is boolean, set 1 or 0 in the result register.
        InstructionNode* notNode = (InstructionNode*) subtreeRoot->rightChild();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(notNode->getInstruction()));
        notNode->markFoldedIntoParent();
        Value *lhs = subtreeRoot->leftChild()->getValue();
        Value *dest = subtreeRoot->getValue();
        mvec.push_back(BuildMI(V9::XNORr, 3).addReg(lhs).addReg(notArg)
                       .addReg(dest, MachineOperand::Def));

        if (notArg->getType() == Type::BoolTy) {
          // set 1 in result register if result of above is non-zero
          mvec.push_back(BuildMI(V9::MOVRNZi, 3).addReg(dest).addZImm(1)
                         .addReg(dest, MachineOperand::UseAndDef));
        }
        break;
      }

      case 41:  // setCCconst:   SetCC(reg, Constant)
      { // Comparison is with a constant:
        //
        // If the bool result must be computed into a register (see below),
        // and the constant is int ZERO, we can use the MOVR[op] instructions
        // and avoid the SUBcc instruction entirely.
        // Otherwise this is just the same as case 42, so just fall through.
        //
        // The result of the SetCC must be computed and stored in a register if
        // it is used outside the current basic block (so it must be computed
        // as a boolreg) or it is used by anything other than a branch.
        // We will use a conditional move to do this.
        //
        Instruction* setCCInstr = subtreeRoot->getInstruction();
        bool computeBoolVal = (subtreeRoot->parent() == NULL ||
                               ! AllUsesAreBranches(setCCInstr));

        if (computeBoolVal) {
          InstrTreeNode* constNode = subtreeRoot->rightChild();
          assert(constNode &&
                 constNode->getNodeType() ==InstrTreeNode::NTConstNode);
          Constant *constVal = cast<Constant>(constNode->getValue());
          bool isValidConst;

          if ((constVal->getType()->isInteger()
               || isa<PointerType>(constVal->getType()))
              && ConvertConstantToIntType(target,
                             constVal, constVal->getType(), isValidConst) == 0
              && isValidConst)
          {
            // That constant is an integer zero after all...
            // Use a MOVR[op] to compute the boolean result
            // Unconditionally set register to 0
            mvec.push_back(BuildMI(V9::SETHI, 2).addZImm(0)
                           .addRegDef(setCCInstr));

            // Now conditionally move 1 into the register.
            // Mark the register as a use (as well as a def) because the old
            // value will be retained if the condition is false.
            MachineOpCode movOpCode = ChooseMovpregiForSetCC(subtreeRoot);
            mvec.push_back(BuildMI(movOpCode, 3)
                           .addReg(subtreeRoot->leftChild()->getValue())
                           .addZImm(1)
                           .addReg(setCCInstr, MachineOperand::UseAndDef));

            break;
          }
        }
        // ELSE FALL THROUGH
      }

      case 42:  // bool:   SetCC(reg, reg):
      {
        // This generates a SUBCC instruction, putting the difference in a
        // result reg. if needed, and/or setting a condition code if needed.
        //
        Instruction* setCCInstr = subtreeRoot->getInstruction();
        Value* leftVal  = subtreeRoot->leftChild()->getValue();
        Value* rightVal = subtreeRoot->rightChild()->getValue();
        const Type* opType = leftVal->getType();
        bool isFPCompare = opType->isFloatingPoint();

        // If the boolean result of the SetCC is used outside the current basic
        // block (so it must be computed as a boolreg) or is used by anything
        // other than a branch, the boolean must be computed and stored
        // in a result register.  We will use a conditional move to do this.
        //
        bool computeBoolVal = (subtreeRoot->parent() == NULL ||
                               ! AllUsesAreBranches(setCCInstr));

        // A TmpInstruction is created to represent the CC "result".
        // Unlike other instances of TmpInstruction, this one is used
        // by machine code of multiple LLVM instructions, viz.,
        // the SetCC and the branch.  Make sure to get the same one!
        // Note that we do this even for FP CC registers even though they
        // are explicit operands, because the type of the operand
        // needs to be a floating point condition code, not an integer
        // condition code.  Think of this as casting the bool result to
        // a FP condition code register.
        // Later, we mark the 4th operand as being a CC register, and as a def.
        //
        TmpInstruction* tmpForCC = GetTmpForCC(setCCInstr,
                                    setCCInstr->getParent()->getParent(),
                                    leftVal->getType(),
                                    MachineCodeForInstruction::get(setCCInstr));

        // If the operands are signed values smaller than 4 bytes, then they
        // must be sign-extended in order to do a valid 32-bit comparison
        // and get the right result in the 32-bit CC register (%icc).
        //
        Value* leftOpToUse  = leftVal;
        Value* rightOpToUse = rightVal;
        if (opType->isIntegral() && opType->isSigned()) {
          unsigned opSize = target.getTargetData().getTypeSize(opType);
          if (opSize < 4) {
            MachineCodeForInstruction& mcfi =
              MachineCodeForInstruction::get(setCCInstr);

            // create temporary virtual regs. to hold the sign-extensions
            leftOpToUse  = new TmpInstruction(mcfi, leftVal);
            rightOpToUse = new TmpInstruction(mcfi, rightVal);

            // sign-extend each operand and put the result in the temporary reg.
            CreateSignExtensionInstructions
              (target, setCCInstr->getParent()->getParent(),
               leftVal, leftOpToUse, 8*opSize, mvec, mcfi);
            CreateSignExtensionInstructions
              (target, setCCInstr->getParent()->getParent(),
               rightVal, rightOpToUse, 8*opSize, mvec, mcfi);
          }
        }

        if (! isFPCompare) {
          // Integer condition: set CC and discard result.
          mvec.push_back(BuildMI(V9::SUBccr, 4)
                         .addReg(leftOpToUse)
                         .addReg(rightOpToUse)
                         .addMReg(target.getRegInfo()->
                                   getZeroRegNum(), MachineOperand::Def)
                         .addCCReg(tmpForCC, MachineOperand::Def));
        } else {
          // FP condition: dest of FCMP should be some FCCn register
          mvec.push_back(BuildMI(ChooseFcmpInstruction(subtreeRoot), 3)
                         .addCCReg(tmpForCC, MachineOperand::Def)
                         .addReg(leftOpToUse)
                         .addReg(rightOpToUse));
        }

        if (computeBoolVal) {
          MachineOpCode movOpCode = (isFPCompare
                                     ? ChooseMovFpcciInstruction(subtreeRoot)
                                     : ChooseMovpcciForSetCC(subtreeRoot));

          // Unconditionally set register to 0
          M = BuildMI(V9::SETHI, 2).addZImm(0).addRegDef(setCCInstr);
          mvec.push_back(M);

          // Now conditionally move 1 into the register.
          // Mark the register as a use (as well as a def) because the old
          // value will be retained if the condition is false.
          M = (BuildMI(movOpCode, 3).addCCReg(tmpForCC).addZImm(1)
               .addReg(setCCInstr, MachineOperand::UseAndDef));
          mvec.push_back(M);
        }
        break;
      }

      case 51:  // reg:   Load(reg)
      case 52:  // reg:   Load(ptrreg)
        SetOperandsForMemInstr(ChooseLoadInstruction(
                                   subtreeRoot->getValue()->getType()),
                               mvec, subtreeRoot, target);
        break;

      case 55:  // reg:   GetElemPtr(reg)
      case 56:  // reg:   GetElemPtrIdx(reg,reg)
        // If the GetElemPtr was folded into the user (parent), it will be
        // caught above.  For other cases, we have to compute the address.
        SetOperandsForMemInstr(V9::ADDr, mvec, subtreeRoot, target);
        break;

      case 57:  // reg:  Alloca: Implement as 1 instruction:
      {         //          add %fp, offsetFromFP -> result
        AllocationInst* instr =
          cast<AllocationInst>(subtreeRoot->getInstruction());
        unsigned tsize =
          target.getTargetData().getTypeSize(instr->getAllocatedType());
        assert(tsize != 0);
        CreateCodeForFixedSizeAlloca(target, instr, tsize, 1, mvec);
        break;
      }

      case 58:  // reg:   Alloca(reg): Implement as 3 instructions:
                //      mul num, typeSz -> tmp
                //      sub %sp, tmp    -> %sp
      {         //      add %sp, frameSizeBelowDynamicArea -> result
        AllocationInst* instr =
          cast<AllocationInst>(subtreeRoot->getInstruction());
        const Type* eltType = instr->getAllocatedType();

        // If #elements is constant, use simpler code for fixed-size allocas
        int tsize = (int) target.getTargetData().getTypeSize(eltType);
        Value* numElementsVal = NULL;
        bool isArray = instr->isArrayAllocation();

        if (!isArray || isa<Constant>(numElementsVal = instr->getArraySize())) {
          // total size is constant: generate code for fixed-size alloca
          unsigned numElements = isArray?
            cast<ConstantUInt>(numElementsVal)->getValue() : 1;
          CreateCodeForFixedSizeAlloca(target, instr, tsize,
                                       numElements, mvec);
        } else {
          // total size is not constant.
          CreateCodeForVariableSizeAlloca(target, instr, tsize,
                                          numElementsVal, mvec);
        }
        break;
      }

      case 61:  // reg:   Call
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
        Intrinsic::ID iid;
        if (calledFunc && (iid=(Intrinsic::ID)calledFunc->getIntrinsicID()))
          specialIntrinsic = CodeGenIntrinsic(iid, *callInstr, target, mvec);

        // If not, generate the normal call sequence for the function.
        // This can also handle any intrinsics that are just function calls.
        //
        if (! specialIntrinsic) {
          Function* currentFunc = callInstr->getParent()->getParent();
          MachineFunction& MF = MachineFunction::get(currentFunc);
          MachineCodeForInstruction& mcfi =
            MachineCodeForInstruction::get(callInstr);
          const SparcV9RegInfo& regInfo =
            (SparcV9RegInfo&) *target.getRegInfo();
          const TargetFrameInfo& frameInfo = *target.getFrameInfo();

          // Create hidden virtual register for return address with type void*
          TmpInstruction* retAddrReg =
            new TmpInstruction(mcfi, PointerType::get(Type::VoidTy), callInstr);

          // Generate the machine instruction and its operands.
          // Use CALL for direct function calls; this optimistically assumes
          // the PC-relative address fits in the CALL address field (22 bits).
          // Use JMPL for indirect calls.
          // This will be added to mvec later, after operand copies.
          //
          MachineInstr* callMI;
          if (calledFunc)             // direct function call
            callMI = BuildMI(V9::CALL, 1).addPCDisp(callee);
          else                        // indirect function call
            callMI = (BuildMI(V9::JMPLCALLi,3).addReg(callee)
                      .addSImm((int64_t)0).addRegDef(retAddrReg));

          const FunctionType* funcType =
            cast<FunctionType>(cast<PointerType>(callee->getType())
                               ->getElementType());
          bool isVarArgs = funcType->isVarArg();
          bool noPrototype = isVarArgs && funcType->getNumParams() == 0;

          // Use a descriptor to pass information about call arguments
          // to the register allocator.  This descriptor will be "owned"
          // and freed automatically when the MachineCodeForInstruction
          // object for the callInstr goes away.
          CallArgsDescriptor* argDesc =
            new CallArgsDescriptor(callInstr, retAddrReg,isVarArgs,noPrototype);
          assert(callInstr->getOperand(0) == callee
                 && "This is assumed in the loop below!");

          // Insert sign-extension instructions for small signed values,
          // if this is an unknown function (i.e., called via a funcptr)
          // or an external one (i.e., which may not be compiled by llc).
          //
          if (calledFunc == NULL || calledFunc->isExternal()) {
            for (unsigned i=1, N=callInstr->getNumOperands(); i < N; ++i) {
              Value* argVal = callInstr->getOperand(i);
              const Type* argType = argVal->getType();
              if (argType->isIntegral() && argType->isSigned()) {
                unsigned argSize = target.getTargetData().getTypeSize(argType);
                if (argSize <= 4) {
                  // create a temporary virtual reg. to hold the sign-extension
                  TmpInstruction* argExtend = new TmpInstruction(mcfi, argVal);

                  // sign-extend argVal and put the result in the temporary reg.
                  CreateSignExtensionInstructions
                    (target, currentFunc, argVal, argExtend,
                     8*argSize, mvec, mcfi);

                  // replace argVal with argExtend in CallArgsDescriptor
                  argDesc->getArgInfo(i-1).replaceArgVal(argExtend);
                }
              }
            }
          }

          // Insert copy instructions to get all the arguments into
          // all the places that they need to be.
          //
          for (unsigned i=1, N=callInstr->getNumOperands(); i < N; ++i) {
            int argNo = i-1;
            CallArgInfo& argInfo = argDesc->getArgInfo(argNo);
            Value* argVal = argInfo.getArgVal(); // don't use callInstr arg here
            const Type* argType = argVal->getType();
            unsigned regType = regInfo.getRegTypeForDataType(argType);
            unsigned argSize = target.getTargetData().getTypeSize(argType);
            int regNumForArg = SparcV9RegInfo::getInvalidRegNum();
            unsigned regClassIDOfArgReg;

            // Check for FP arguments to varargs functions.
            // Any such argument in the first $K$ args must be passed in an
            // integer register.  If there is no prototype, it must also
            // be passed as an FP register.
            // K = #integer argument registers.
            bool isFPArg = argVal->getType()->isFloatingPoint();
            if (isVarArgs && isFPArg) {

              if (noPrototype) {
                // It is a function with no prototype: pass value
                // as an FP value as well as a varargs value.  The FP value
                // may go in a register or on the stack.  The copy instruction
                // to the outgoing reg/stack is created by the normal argument
                // handling code since this is the "normal" passing mode.
                //
                regNumForArg = regInfo.regNumForFPArg(regType,
                                                      false, false, argNo,
                                                      regClassIDOfArgReg);
                if (regNumForArg == regInfo.getInvalidRegNum())
                  argInfo.setUseStackSlot();
                else
                  argInfo.setUseFPArgReg();
              }

              // If this arg. is in the first $K$ regs, add special copy-
              // float-to-int instructions to pass the value as an int.
              // To check if it is in the first $K$, get the register
              // number for the arg #i.  These copy instructions are
              // generated here because they are extra cases and not needed
              // for the normal argument handling (some code reuse is
              // possible though -- later).
              //
              int copyRegNum = regInfo.regNumForIntArg(false, false, argNo,
                                                       regClassIDOfArgReg);
              if (copyRegNum != regInfo.getInvalidRegNum()) {
                // Create a virtual register to represent copyReg. Mark
                // this vreg as being an implicit operand of the call MI
                const Type* loadTy = (argType == Type::FloatTy
                                      ? Type::IntTy : Type::LongTy);
                TmpInstruction* argVReg = new TmpInstruction(mcfi, loadTy,
                                                             argVal, NULL,
                                                             "argRegCopy");
                callMI->addImplicitRef(argVReg);

                // Get a temp stack location to use to copy
                // float-to-int via the stack.
                //
                // FIXME: For now, we allocate permanent space because
                // the stack frame manager does not allow locals to be
                // allocated (e.g., for alloca) after a temp is
                // allocated!
                //
                // int tmpOffset = MF.getInfo<SparcV9FunctionInfo>()->pushTempValue(argSize);
                int tmpOffset = MF.getInfo<SparcV9FunctionInfo>()->allocateLocalVar(argVReg);

                // Generate the store from FP reg to stack
                unsigned StoreOpcode = ChooseStoreInstruction(argType);
                M = BuildMI(convertOpcodeFromRegToImm(StoreOpcode), 3)
                  .addReg(argVal).addMReg(regInfo.getFramePointer())
                  .addSImm(tmpOffset);
                mvec.push_back(M);

                // Generate the load from stack to int arg reg
                unsigned LoadOpcode = ChooseLoadInstruction(loadTy);
                M = BuildMI(convertOpcodeFromRegToImm(LoadOpcode), 3)
                  .addMReg(regInfo.getFramePointer()).addSImm(tmpOffset)
                  .addReg(argVReg, MachineOperand::Def);

                // Mark operand with register it should be assigned
                // both for copy and for the callMI
                M->SetRegForOperand(M->getNumOperands()-1, copyRegNum);
                callMI->SetRegForImplicitRef(callMI->getNumImplicitRefs()-1,
                                             copyRegNum);
                mvec.push_back(M);

                // Add info about the argument to the CallArgsDescriptor
                argInfo.setUseIntArgReg();
                argInfo.setArgCopy(copyRegNum);
              } else {
                // Cannot fit in first $K$ regs so pass arg on stack
                argInfo.setUseStackSlot();
              }
            } else if (isFPArg) {
              // Get the outgoing arg reg to see if there is one.
              regNumForArg = regInfo.regNumForFPArg(regType, false, false,
                                                    argNo, regClassIDOfArgReg);
              if (regNumForArg == regInfo.getInvalidRegNum())
                argInfo.setUseStackSlot();
              else {
                argInfo.setUseFPArgReg();
                regNumForArg =regInfo.getUnifiedRegNum(regClassIDOfArgReg,
                                                       regNumForArg);
              }
            } else {
              // Get the outgoing arg reg to see if there is one.
              regNumForArg = regInfo.regNumForIntArg(false,false,
                                                     argNo, regClassIDOfArgReg);
              if (regNumForArg == regInfo.getInvalidRegNum())
                argInfo.setUseStackSlot();
              else {
                argInfo.setUseIntArgReg();
                regNumForArg =regInfo.getUnifiedRegNum(regClassIDOfArgReg,
                                                       regNumForArg);
              }
            }

            //
            // Now insert copy instructions to stack slot or arg. register
            //
            if (argInfo.usesStackSlot()) {
              // Get the stack offset for this argument slot.
              // FP args on stack are right justified so adjust offset!
              // int arguments are also right justified but they are
              // always loaded as a full double-word so the offset does
              // not need to be adjusted.
              int argOffset = frameInfo.getOutgoingArgOffset(MF, argNo);
              if (argType->isFloatingPoint()) {
                unsigned slotSize = SparcV9FrameInfo::SizeOfEachArgOnStack;
                assert(argSize <= slotSize && "Insufficient slot size!");
                argOffset += slotSize - argSize;
              }

              // Now generate instruction to copy argument to stack
              MachineOpCode storeOpCode =
                (argType->isFloatingPoint()
                 ? ((argSize == 4)? V9::STFi : V9::STDFi) : V9::STXi);

              M = BuildMI(storeOpCode, 3).addReg(argVal)
                .addMReg(regInfo.getStackPointer()).addSImm(argOffset);
              mvec.push_back(M);
            }
            else if (regNumForArg != regInfo.getInvalidRegNum()) {

              // Create a virtual register to represent the arg reg. Mark
              // this vreg as being an implicit operand of the call MI.
              TmpInstruction* argVReg =
                new TmpInstruction(mcfi, argVal, NULL, "argReg");

              callMI->addImplicitRef(argVReg);

              // Generate the reg-to-reg copy into the outgoing arg reg.
              // -- For FP values, create a FMOVS or FMOVD instruction
              // -- For non-FP values, create an add-with-0 instruction
              if (argType->isFloatingPoint())
                M=(BuildMI(argType==Type::FloatTy? V9::FMOVS :V9::FMOVD,2)
                   .addReg(argVal).addReg(argVReg, MachineOperand::Def));
              else
                M = (BuildMI(ChooseAddInstructionByType(argType), 3)
                     .addReg(argVal).addSImm((int64_t) 0)
                     .addReg(argVReg, MachineOperand::Def));

              // Mark the operand with the register it should be assigned
              M->SetRegForOperand(M->getNumOperands()-1, regNumForArg);
              callMI->SetRegForImplicitRef(callMI->getNumImplicitRefs()-1,
                                           regNumForArg);

              mvec.push_back(M);
            }
            else
              assert(argInfo.getArgCopy() != regInfo.getInvalidRegNum() &&
                     "Arg. not in stack slot, primary or secondary register?");
          }

          // add call instruction and delay slot before copying return value
          mvec.push_back(callMI);
          mvec.push_back(BuildMI(V9::NOP, 0));

          // Add the return value as an implicit ref.  The call operands
          // were added above.  Also, add code to copy out the return value.
          // This is always register-to-register for int or FP return values.
          //
          if (callInstr->getType() != Type::VoidTy) {
            // Get the return value reg.
            const Type* retType = callInstr->getType();

            int regNum = (retType->isFloatingPoint()
                          ? (unsigned) SparcV9FloatRegClass::f0
                          : (unsigned) SparcV9IntRegClass::o0);
            unsigned regClassID = regInfo.getRegClassIDOfType(retType);
            regNum = regInfo.getUnifiedRegNum(regClassID, regNum);

            // Create a virtual register to represent it and mark
            // this vreg as being an implicit operand of the call MI
            TmpInstruction* retVReg =
              new TmpInstruction(mcfi, callInstr, NULL, "argReg");

            callMI->addImplicitRef(retVReg, /*isDef*/ true);

            // Generate the reg-to-reg copy from the return value reg.
            // -- For FP values, create a FMOVS or FMOVD instruction
            // -- For non-FP values, create an add-with-0 instruction
            if (retType->isFloatingPoint())
              M = (BuildMI(retType==Type::FloatTy? V9::FMOVS : V9::FMOVD, 2)
                   .addReg(retVReg).addReg(callInstr, MachineOperand::Def));
            else
              M = (BuildMI(ChooseAddInstructionByType(retType), 3)
                   .addReg(retVReg).addSImm((int64_t) 0)
                   .addReg(callInstr, MachineOperand::Def));

            // Mark the operand with the register it should be assigned
            // Also mark the implicit ref of the call defining this operand
            M->SetRegForOperand(0, regNum);
            callMI->SetRegForImplicitRef(callMI->getNumImplicitRefs()-1,regNum);

            mvec.push_back(M);
          }

          // For the CALL instruction, the ret. addr. reg. is also implicit
          if (isa<Function>(callee))
            callMI->addImplicitRef(retAddrReg, /*isDef*/ true);

          MF.getInfo<SparcV9FunctionInfo>()->popAllTempValues();  // free temps used for this inst
        }

        break;
      }

      case 62:  // reg:   Shl(reg, reg)
      {
        Value* argVal1 = subtreeRoot->leftChild()->getValue();
        Value* argVal2 = subtreeRoot->rightChild()->getValue();
        Instruction* shlInstr = subtreeRoot->getInstruction();

        const Type* opType = argVal1->getType();
        assert((opType->isInteger() || isa<PointerType>(opType)) &&
               "Shl unsupported for other types");
        unsigned opSize = target.getTargetData().getTypeSize(opType);

        CreateShiftInstructions(target, shlInstr->getParent()->getParent(),
                                (opSize > 4)? V9::SLLXr6:V9::SLLr5,
                                argVal1, argVal2, 0, shlInstr, mvec,
                                MachineCodeForInstruction::get(shlInstr));
        break;
      }

      case 63:  // reg:   Shr(reg, reg)
      {
        const Type* opType = subtreeRoot->leftChild()->getValue()->getType();
        assert((opType->isInteger() || isa<PointerType>(opType)) &&
               "Shr unsupported for other types");
        unsigned opSize = target.getTargetData().getTypeSize(opType);
        Add3OperandInstr(opType->isSigned()
                         ? (opSize > 4? V9::SRAXr6 : V9::SRAr5)
                         : (opSize > 4? V9::SRLXr6 : V9::SRLr5),
                         subtreeRoot, mvec);
        break;
      }

      case 64:  // reg:   Phi(reg,reg)
        break;                          // don't forward the value

      case 66:  // reg:   VAArg (reg): the va_arg instruction
      { // Load argument from stack using current va_list pointer value.
        // Use 64-bit load for all non-FP args, and LDDF or double for FP.
        Instruction* vaArgI = subtreeRoot->getInstruction();
        //but first load the va_list pointer
        // Create a virtual register to represent it
        //What oh what do we pass to TmpInstruction?
        MachineCodeForInstruction& m1 = MachineCodeForInstruction::get(vaArgI);
        TmpInstruction* VReg = new TmpInstruction(m1, vaArgI->getOperand(0)->getType());
        mvec.push_back(BuildMI(V9::LDXi, 3).addReg(vaArgI->getOperand(0))
                       .addSImm(0).addRegDef(VReg));
        //OK, now do the load
        MachineOpCode loadOp = (vaArgI->getType()->isFloatingPoint()
                                ? (vaArgI->getType() == Type::FloatTy
                                   ? V9::LDFi : V9::LDDFi)
                                : V9::LDXi);
        mvec.push_back(BuildMI(loadOp, 3).addReg(VReg).
                       addSImm(0).addRegDef(vaArgI));
        //Also increment the pointer
        MachineCodeForInstruction& m2 = MachineCodeForInstruction::get(vaArgI);
        TmpInstruction* VRegA = new TmpInstruction(m2, vaArgI->getOperand(0)->getType());
        unsigned argSize = SparcV9FrameInfo::SizeOfEachArgOnStack;
        mvec.push_back(BuildMI(V9::ADDi, 3).addReg(VReg).
                       addSImm(argSize).addRegDef(VRegA));
        //And store
        mvec.push_back(BuildMI(V9::STXr, 3).addReg(VRegA).
                       addReg(vaArgI->getOperand(0)).addSImm(0));
        break;
      }

      case 71:  // reg:     VReg
      case 72:  // reg:     Constant
        break;                          // don't forward the value

      default:
        assert(0 && "Unrecognized BURG rule");
        break;
      }
    }

  if (forwardOperandNum >= 0) {
    // We did not generate a machine instruction but need to use operand.
    // If user is in the same tree, replace Value in its machine operand.
    // If not, insert a copy instruction which should get coalesced away
    // by register allocation.
    if (subtreeRoot->parent() != NULL)
      ForwardOperand(subtreeRoot, subtreeRoot->parent(), forwardOperandNum);
    else {
      std::vector<MachineInstr*> minstrVec;
      Instruction* instr = subtreeRoot->getInstruction();
      CreateCopyInstructionsByType(target,
                                     instr->getParent()->getParent(),
                                     instr->getOperand(forwardOperandNum),
                                     instr, minstrVec,
                                     MachineCodeForInstruction::get(instr));
      assert(minstrVec.size() > 0);
      mvec.insert(mvec.end(), minstrVec.begin(), minstrVec.end());
    }
  }

  if (maskUnsignedResult) {
    // If result is unsigned and smaller than int reg size,
    // we need to clear high bits of result value.
    assert(forwardOperandNum < 0 && "Need mask but no instruction generated");
    Instruction* dest = subtreeRoot->getInstruction();
    if (dest->getType()->isUnsigned()) {
      unsigned destSize=target.getTargetData().getTypeSize(dest->getType());
      if (destSize <= 4) {
        // Mask high 64 - N bits, where N = 4*destSize.

        // Use a TmpInstruction to represent the
        // intermediate result before masking.  Since those instructions
        // have already been generated, go back and substitute tmpI
        // for dest in the result position of each one of them.
        //
        MachineCodeForInstruction& mcfi = MachineCodeForInstruction::get(dest);
        TmpInstruction *tmpI = new TmpInstruction(mcfi, dest->getType(),
                                                  dest, NULL, "maskHi");
        Value* srlArgToUse = tmpI;

        unsigned numSubst = 0;
        for (unsigned i=0, N=mvec.size(); i < N; ++i) {

          // Make sure we substitute all occurrences of dest in these instrs.
          // Otherwise, we will have bogus code.
          bool someArgsWereIgnored = false;

          // Make sure not to substitute an upwards-exposed use -- that would
          // introduce a use of `tmpI' with no preceding def.  Therefore,
          // substitute a use or def-and-use operand only if a previous def
          // operand has already been substituted (i.e., numSubst > 0).
          //
          numSubst += mvec[i]->substituteValue(dest, tmpI,
                                               /*defsOnly*/ numSubst == 0,
                                               /*notDefsAndUses*/ numSubst > 0,
                                               someArgsWereIgnored);
          assert(!someArgsWereIgnored &&
                 "Operand `dest' exists but not replaced: probably bogus!");
        }
        assert(numSubst > 0 && "Operand `dest' not replaced: probably bogus!");

        // Left shift 32-N if size (N) is less than 32 bits.
        // Use another tmp. virtual register to represent this result.
        if (destSize < 4) {
          srlArgToUse = new TmpInstruction(mcfi, dest->getType(),
                                           tmpI, NULL, "maskHi2");
          mvec.push_back(BuildMI(V9::SLLXi6, 3).addReg(tmpI)
                         .addZImm(8*(4-destSize))
                         .addReg(srlArgToUse, MachineOperand::Def));
        }

        // Logical right shift 32-N to get zero extension in top 64-N bits.
        mvec.push_back(BuildMI(V9::SRLi5, 3).addReg(srlArgToUse)
                         .addZImm(8*(4-destSize))
                         .addReg(dest, MachineOperand::Def));

      } else if (destSize < 8) {
        assert(0 && "Unsupported type size: 32 < size < 64 bits");
      }
    }
  }
}

} // End llvm namespace

//==------------------------------------------------------------------------==//
//                     Class V9ISel Implementation
//==------------------------------------------------------------------------==//

bool V9ISel::runOnFunction(Function &F) {
  DefaultIntrinsicLowering IL;
  // First pass - Walk the function, lowering any calls to intrinsic functions
  // which the instruction selector cannot handle.
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(I++))
        if (Function *F = CI->getCalledFunction())
          switch (F->getIntrinsicID()) {
          case Intrinsic::not_intrinsic:
          case Intrinsic::vastart:
          case Intrinsic::vacopy:
          case Intrinsic::vaend:
            // We directly implement these intrinsics.  Note that this knowledge
            // is incestuously entangled with the code in
            // SparcInstrSelection.cpp and must be updated when it is updated.
            // Since ALL of the code in this library is incestuously intertwined
            // with it already and sparc specific, we will live with this.
            break;
          default:
            // All other intrinsic calls we must lower.
            Instruction *Before = CI->getPrev();
            IL.LowerIntrinsicCall(CI);
            if (Before) {        // Move iterator to instruction after call
              I = Before;  ++I;
            } else {
              I = BB->begin();
            }
          }

  // Build the instruction trees to be given as inputs to BURG.
  InstrForest instrForest(&F);
  if (SelectDebugLevel >= Select_DebugInstTrees) {
    std::cerr << "\n\n*** Input to instruction selection for function "
              << F.getName() << "\n\n" << F
              << "\n\n*** Instruction trees for function "
              << F.getName() << "\n\n";
    instrForest.dump();
  }

  // Invoke BURG instruction selection for each tree
  for (InstrForest::const_root_iterator RI = instrForest.roots_begin();
       RI != instrForest.roots_end(); ++RI) {
    InstructionNode* basicNode = *RI;
    assert(basicNode->parent() == NULL && "A `root' node has a parent?");

    // Invoke BURM to label each tree node with a state
    burm_label(basicNode);
    if (SelectDebugLevel >= Select_DebugBurgTrees) {
      printcover(basicNode, 1, 0);
      std::cerr << "\nCover cost == " << treecost(basicNode, 1, 0) <<"\n\n";
      printMatches(basicNode);
    }

    // Then recursively walk the tree to select instructions
    SelectInstructionsForTree(basicNode, /*goalnt*/1);
  }

  // Create the MachineBasicBlocks and add all of the MachineInstrs
  // defined in the MachineCodeForInstruction objects to the MachineBasicBlocks.
  MachineFunction &MF = MachineFunction::get(&F);
  std::map<const BasicBlock *, MachineBasicBlock *> MBBMap;
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
    MachineBasicBlock *MBB = new MachineBasicBlock(BI);
    MF.getBasicBlockList().push_back(MBB);
    MBBMap[BI] = MBB;

    for (BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
      MachineCodeForInstruction &mvec = MachineCodeForInstruction::get(II);
      MBB->insert(MBB->end(), mvec.begin(), mvec.end());
    }
  }

  // Initialize Machine-CFG for the function.
  for (MachineFunction::iterator i = MF.begin (), e = MF.end (); i != e; ++i) {
    MachineBasicBlock &MBB = *i;
    const BasicBlock *BB = MBB.getBasicBlock ();
    // for each successor S of BB, add MBBMap[S] as a successor of MBB.
    for (succ_const_iterator si = succ_begin(BB), se = succ_end(BB); si != se;
         ++si) {
      MachineBasicBlock *succMBB = MBBMap[*si];
      assert (succMBB && "Can't find MachineBasicBlock for this successor");
      MBB.addSuccessor (succMBB);
    }
  }

  // Insert phi elimination code
  InsertCodeForPhis(F);

  if (SelectDebugLevel >= Select_PrintMachineCode) {
    std::cerr << "\n*** Machine instructions after INSTRUCTION SELECTION\n";
    MachineFunction::get(&F).dump();
  }

  return true;
}

/// InsertCodeForPhis - This method inserts Phi elimination code for
/// all Phi nodes in the given function.  After this method is called,
/// the Phi nodes still exist in the LLVM code, but copies are added to the
/// machine code.
///
void V9ISel::InsertCodeForPhis(Function &F) {
  // Iterate over every Phi node PN in F:
  MachineFunction &MF = MachineFunction::get(&F);
  for (MachineFunction::iterator BB = MF.begin(); BB != MF.end(); ++BB) {
    for (BasicBlock::const_iterator IIt = BB->getBasicBlock()->begin();
         const PHINode *PN = dyn_cast<PHINode>(IIt); ++IIt) {
      // Create a new temporary register to hold the result of the Phi copy.
      // The leak detector shouldn't track these nodes.  They are not garbage,
      // even though their parent field is never filled in.
      Value *PhiCpRes = new PHINode(PN->getType(), PN->getName() + ":PhiCp");
      LeakDetector::removeGarbageObject(PhiCpRes);

      // For each of PN's incoming values, insert a copy in the corresponding
      // predecessor block.
      MachineCodeForInstruction &MCforPN = MachineCodeForInstruction::get (PN);
      for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i) {
        std::vector<MachineInstr*> mvec, CpVec;
        Target.getRegInfo()->cpValue2Value(PN->getIncomingValue(i),
                                           PhiCpRes, mvec);
        for (std::vector<MachineInstr*>::iterator MI=mvec.begin();
             MI != mvec.end(); ++MI) {
          std::vector<MachineInstr*> CpVec2 =
            FixConstantOperandsForInstr(const_cast<PHINode*>(PN), *MI, Target);
          CpVec2.push_back(*MI);
          CpVec.insert(CpVec.end(), CpVec2.begin(), CpVec2.end());
        }
        // Insert the copy instructions into the predecessor BB.
        InsertPhiElimInstructions(PN->getIncomingBlock(i), CpVec);
        MCforPN.insert (MCforPN.end (), CpVec.begin (), CpVec.end ());
      }
      // Insert a copy instruction from PhiCpRes to PN.
      std::vector<MachineInstr*> mvec;
      Target.getRegInfo()->cpValue2Value(PhiCpRes, const_cast<PHINode*>(PN),
                                        mvec);
      BB->insert(BB->begin(), mvec.begin(), mvec.end());
      MCforPN.insert (MCforPN.end (), mvec.begin (), mvec.end ());
    }  // for each Phi Instr in BB
  } // for all BBs in function
}

/// InsertPhiElimInstructions - Inserts the instructions in CpVec into the
/// MachineBasicBlock corresponding to BB, just before its terminator
/// instruction. This is used by InsertCodeForPhis() to insert copies, above.
///
void V9ISel::InsertPhiElimInstructions(BasicBlock *BB,
                                       const std::vector<MachineInstr*>& CpVec)
{
  Instruction *TermInst = (Instruction*)BB->getTerminator();
  MachineCodeForInstruction &MC4Term = MachineCodeForInstruction::get(TermInst);
  MachineInstr *FirstMIOfTerm = MC4Term.front();
  assert (FirstMIOfTerm && "No Machine Instrs for terminator");

  MachineBasicBlock *MBB = FirstMIOfTerm->getParent();
  assert(MBB && "Machine BB for predecessor's terminator not found");
  MachineBasicBlock::iterator MCIt = FirstMIOfTerm;
  assert(MCIt != MBB->end() && "Start inst of terminator not found");

  // Insert the copy instructions just before the first machine instruction
  // generated for the terminator.
  MBB->insert(MCIt, CpVec.begin(), CpVec.end());
}

/// SelectInstructionsForTree - Recursively walk the tree to select
/// instructions. Do this top-down so that child instructions can exploit
/// decisions made at the child instructions.
///
/// E.g., if br(setle(reg,const)) decides the constant is 0 and uses
/// a branch-on-integer-register instruction, then the setle node
/// can use that information to avoid generating the SUBcc instruction.
///
/// Note that this cannot be done bottom-up because setle must do this
/// only if it is a child of the branch (otherwise, the result of setle
/// may be used by multiple instructions).
///
void V9ISel::SelectInstructionsForTree(InstrTreeNode* treeRoot, int goalnt) {
  // Get the rule that matches this node.
  int ruleForNode = burm_rule(treeRoot->state, goalnt);

  if (ruleForNode == 0) {
    std::cerr << "Could not match instruction tree for instr selection\n";
    abort();
  }

  // Get this rule's non-terminals and the corresponding child nodes (if any)
  short *nts = burm_nts[ruleForNode];

  // First, select instructions for the current node and rule.
  // (If this is a list node, not an instruction, then skip this step).
  // This function is specific to the target architecture.
  if (treeRoot->opLabel != VRegListOp) {
    std::vector<MachineInstr*> minstrVec;
    InstructionNode* instrNode = (InstructionNode*)treeRoot;
    assert(instrNode->getNodeType() == InstrTreeNode::NTInstructionNode);
    GetInstructionsByRule(instrNode, ruleForNode, nts, Target, minstrVec);
    MachineCodeForInstruction &mvec =
      MachineCodeForInstruction::get(instrNode->getInstruction());
    mvec.insert(mvec.end(), minstrVec.begin(), minstrVec.end());
  }

  // Then, recursively compile the child nodes, if any.
  //
  if (nts[0]) {
    // i.e., there is at least one kid
    InstrTreeNode* kids[2];
    int currentRule = ruleForNode;
    burm_kids(treeRoot, currentRule, kids);

    // First skip over any chain rules so that we don't visit
    // the current node again.
    while (ThisIsAChainRule(currentRule)) {
      currentRule = burm_rule(treeRoot->state, nts[0]);
      nts = burm_nts[currentRule];
      burm_kids(treeRoot, currentRule, kids);
    }

    // Now we have the first non-chain rule so we have found
    // the actual child nodes.  Recursively compile them.
    for (unsigned i = 0; nts[i]; i++) {
      assert(i < 2);
      InstrTreeNode::InstrTreeNodeType nodeType = kids[i]->getNodeType();
      if (nodeType == InstrTreeNode::NTVRegListNode ||
          nodeType == InstrTreeNode::NTInstructionNode)
        SelectInstructionsForTree(kids[i], nts[i]);
    }
  }

  // Finally, do any post-processing on this node after its children
  // have been translated.
  if (treeRoot->opLabel != VRegListOp)
    PostprocessMachineCodeForTree((InstructionNode*)treeRoot, ruleForNode, nts);
}

/// PostprocessMachineCodeForTree - Apply any final cleanups to
/// machine code for the root of a subtree after selection for all its
/// children has been completed.
///
void V9ISel::PostprocessMachineCodeForTree(InstructionNode *instrNode,
                                           int ruleForNode, short *nts) {
  // Fix up any constant operands in the machine instructions to either
  // use an immediate field or to load the constant into a register.
  // Walk backwards and use direct indexes to allow insertion before current.
  Instruction* vmInstr = instrNode->getInstruction();
  MachineCodeForInstruction &mvec = MachineCodeForInstruction::get(vmInstr);
  for (unsigned i = mvec.size(); i != 0; --i) {
    std::vector<MachineInstr*> loadConstVec =
      FixConstantOperandsForInstr(vmInstr, mvec[i-1], Target);
    mvec.insert(mvec.begin()+i-1, loadConstVec.begin(), loadConstVec.end());
  }
}

/// createSparcV9BurgInstSelector - Creates and returns a new SparcV9
/// BURG-based instruction selection pass.
///
FunctionPass *llvm::createSparcV9BurgInstSelector(TargetMachine &TM) {
  return new V9ISel(TM);
}
