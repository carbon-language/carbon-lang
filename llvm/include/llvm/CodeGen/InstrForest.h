//===-- llvm/CodeGen/InstForest.h -------------------------------*- C++ -*-===//
//
// Purpose:
//	Convert SSA graph to instruction trees for instruction selection.
// 
// Strategy:
//  The basic idea is that we would like to group instructions into a single
//  tree if one or more of them might be potentially combined into a single
//  complex instruction in the target machine.
//  Since this grouping is completely machine-independent, it is as
//  aggressive as possible.  In particular, we group two instructions
//  O and I if:
//  (1) Instruction O computes an operand of instruction I, and
//  (2) O and I are part of the same basic block, and
//  (3) O has only a single use, viz., I.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_INSTRFOREST_H
#define LLVM_CODEGEN_INSTRFOREST_H

#include "llvm/Instruction.h"
#include "Support/hash_map"

class Constant;
class Function;
class InstrTreeNode;
class InstrForest;

//--------------------------------------------------------------------------
// OpLabel values for special-case nodes created for instruction selection.
// All op-labels not defined here are identical to the instruction
// opcode returned by Instruction::getOpcode()
//--------------------------------------------------------------------------

const int  InvalidOp	=  -1;
const int  VRegListOp   =  97;
const int  VRegNodeOp	=  98;
const int  ConstantNodeOp= 99;
const int  LabelNodeOp	= 100;

const int  RetValueOp	= 100 + Instruction::Ret;               // 101
const int  BrCondOp	= 100 + Instruction::Br;                // 102

const int  BAndOp       = 100 + Instruction::And;               // 111
const int  BOrOp        = 100 + Instruction::Or;                // 112
const int  BXorOp       = 100 + Instruction::Xor;               // 113
const int  BNotOp       = 200 + Instruction::Xor;               // 213
const int   NotOp       = 300 + Instruction::Xor;               // 313

const int  SetCCOp	= 100 + Instruction::SetEQ;             // 114

const int  AllocaN	= 100 + Instruction::Alloca;		// 122
const int  LoadIdx	= 100 + Instruction::Load;		// 123
const int  GetElemPtrIdx= 100 + Instruction::GetElementPtr;	// 125

const int  ToBoolTy	= 100 + Instruction::Cast;		// 127
const int  ToUByteTy	= ToBoolTy +  1;
const int  ToSByteTy	= ToBoolTy +  2;
const int  ToUShortTy	= ToBoolTy +  3;
const int  ToShortTy	= ToBoolTy +  4;
const int  ToUIntTy	= ToBoolTy +  5;
const int  ToIntTy	= ToBoolTy +  6;
const int  ToULongTy	= ToBoolTy +  7;
const int  ToLongTy	= ToBoolTy +  8;
const int  ToFloatTy	= ToBoolTy +  9;
const int  ToDoubleTy	= ToBoolTy + 10;
const int  ToArrayTy	= ToBoolTy + 11;
const int  ToPointerTy	= ToBoolTy + 12;

//-------------------------------------------------------------------------
// Data types needed by BURG and implemented by us
//-------------------------------------------------------------------------

typedef int OpLabel;
typedef int StateLabel;

//-------------------------------------------------------------------------
// Declarations of data and functions created by BURG
//-------------------------------------------------------------------------

extern short*		burm_nts[];
  
extern StateLabel	burm_label	(InstrTreeNode* p);
  
extern StateLabel	burm_state	(OpLabel op, StateLabel leftState,
					 StateLabel rightState);

extern StateLabel	burm_rule	(StateLabel state, int goalNT);
  
extern InstrTreeNode**  burm_kids	(InstrTreeNode* p, int eruleno,
					 InstrTreeNode* kids[]);
  
extern void		printcover	(InstrTreeNode*, int, int);
extern void		printtree	(InstrTreeNode*);
extern int		treecost	(InstrTreeNode*, int, int);
extern void		printMatches	(InstrTreeNode*);


//------------------------------------------------------------------------ 
// class InstrTreeNode
// 
// A single tree node in the instruction tree used for
// instruction selection via BURG.
//------------------------------------------------------------------------ 

class InstrTreeNode {
  InstrTreeNode(const InstrTreeNode &);   // DO NOT IMPLEMENT
  void operator=(const InstrTreeNode &);  // DO NOT IMPLEMENT
public:
  enum InstrTreeNodeType { NTInstructionNode,
			   NTVRegListNode,
			   NTVRegNode,
			   NTConstNode,
			   NTLabelNode };
  
  // BASIC TREE NODE START
  InstrTreeNode* LeftChild;
  InstrTreeNode* RightChild;
  InstrTreeNode* Parent;
  OpLabel        opLabel;
  StateLabel     state;
  // BASIC TREE NODE END

protected:
  InstrTreeNodeType treeNodeType;
  Value*	   val;
  
public:
  InstrTreeNode(InstrTreeNodeType nodeType, Value* _val)
    : treeNodeType(nodeType), val(_val) {
    LeftChild = RightChild = Parent = 0;
    opLabel   = InvalidOp;
  }
  virtual ~InstrTreeNode() {
    delete LeftChild;
    delete RightChild;
  }
  
  InstrTreeNodeType	getNodeType	() const { return treeNodeType; }
  
  Value*		getValue	() const { return val; }
  
  inline OpLabel	getOpLabel	() const { return opLabel; }
  
  inline InstrTreeNode*	leftChild() const {
    return LeftChild;
  }
  
  // If right child is a list node, recursively get its *left* child
  inline InstrTreeNode* rightChild() const {
    return (!RightChild ? 0 : 
	    (RightChild->getOpLabel() == VRegListOp
	     ? RightChild->LeftChild : RightChild));
  }
  
  inline InstrTreeNode *parent() const {
    return Parent;
  }
  
  void dump(int dumpChildren, int indent) const;

protected:
  virtual void dumpNode(int indent) const = 0;

  friend class InstrForest;
};


class InstructionNode : public InstrTreeNode {
private:
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
  VRegListNode() : InstrTreeNode(NTVRegListNode, 0) {
    opLabel = VRegListOp;
  }

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


//------------------------------------------------------------------------
// class InstrForest
// 
// A forest of instruction trees, usually for a single method.
//
// Methods:
//     buildTreesForMethod()	Builds the forest of trees for a method
//     getTreeNodeForInstr()	Returns the tree node for an Instruction
//     getRootSet()		Returns a set of root nodes for all the trees
// 
//------------------------------------------------------------------------ 

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
  /*ctor*/	InstrForest	(Function *F);
  /*dtor*/	~InstrForest	();
  
  inline InstructionNode *getTreeNodeForInstr(Instruction* instr) {
    return (*this)[instr];
  }
  
  const_root_iterator roots_begin() const     { return treeRoots.begin(); }
        root_iterator roots_begin()           { return treeRoots.begin(); }
  const_root_iterator roots_end  () const     { return treeRoots.end();   }
        root_iterator roots_end  ()           { return treeRoots.end();   }
  
  void dump() const;
  
private:
  //
  // Private methods for buidling the instruction forest
  //
  void eraseRoot    (InstructionNode* node);
  void setLeftChild (InstrTreeNode* parent, InstrTreeNode* child);
  void setRightChild(InstrTreeNode* parent, InstrTreeNode* child);
  void setParent    (InstrTreeNode* child,  InstrTreeNode* parent);
  void noteTreeNodeForInstr(Instruction* instr, InstructionNode* treeNode);
  
  InstructionNode* buildTreeForInstruction(Instruction* instr);
};

#endif
