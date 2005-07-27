//===- SparcV9InstrForest.h - SparcV9 BURG Instruction Selector Trees -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A forest of BURG instruction trees (class InstrForest) which represents
// a function to the BURG-based instruction selector, and a bunch of constants
// and declarations used by the generated BURG code.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV9INSTRFOREST_H
#define SPARCV9INSTRFOREST_H

#include "llvm/Instruction.h"
using namespace llvm;

/// OpLabel values for special-case nodes created for instruction selection.
/// All op-labels not defined here are identical to the instruction
/// opcode returned by Instruction::getOpcode().
///
static const int
 InvalidOp      = -1,
 VRegListOp     = 97,
 VRegNodeOp     = 98,
 ConstantNodeOp = 99,
 LabelNodeOp    = 100,
 RetValueOp     = 100 + Instruction::Ret,               // 101
 BrCondOp       = 100 + Instruction::Br,                // 102
 BAndOp         = 100 + Instruction::And,               // 111
 BOrOp          = 100 + Instruction::Or,                // 112
 BXorOp         = 100 + Instruction::Xor,               // 113
 BNotOp         = 200 + Instruction::Xor,               // 213
  NotOp         = 300 + Instruction::Xor,               // 313
 SetCCOp        = 100 + Instruction::SetEQ,             // 114
 AllocaN        = 100 + Instruction::Alloca,            // 122
 LoadIdx        = 100 + Instruction::Load,              // 123
 GetElemPtrIdx  = 100 + Instruction::GetElementPtr,     // 125
 ToBoolTy       = 100 + Instruction::Cast;              // 127
static const int
 ToUByteTy      = ToBoolTy +  1,
 ToSByteTy      = ToBoolTy +  2,
 ToUShortTy     = ToBoolTy +  3,
 ToShortTy      = ToBoolTy +  4,
 ToUIntTy       = ToBoolTy +  5,
 ToIntTy        = ToBoolTy +  6,
 ToULongTy      = ToBoolTy +  7,
 ToLongTy       = ToBoolTy +  8,
 ToFloatTy      = ToBoolTy +  9,
 ToDoubleTy     = ToBoolTy + 10,
 ToArrayTy      = ToBoolTy + 11,
 ToPointerTy    = ToBoolTy + 12;

/// Data types needed by BURG
///
typedef int OpLabel;
typedef int StateLabel;

/// Declarations of data and functions created by BURG
///
namespace llvm {
  class InstrTreeNode;
};
extern short*           burm_nts[];
extern StateLabel       burm_label      (InstrTreeNode* p);
extern StateLabel       burm_state      (OpLabel op, StateLabel leftState,
                                         StateLabel rightState);
extern StateLabel       burm_rule       (StateLabel state, int goalNT);
extern InstrTreeNode**  burm_kids       (InstrTreeNode* p, int eruleno,
                                         InstrTreeNode* kids[]);
extern void             printcover      (InstrTreeNode*, int, int);
extern void             printtree       (InstrTreeNode*);
extern int              treecost        (InstrTreeNode*, int, int);
extern void             printMatches    (InstrTreeNode*);

namespace llvm {

/// InstrTreeNode - A single tree node in the instruction tree used for
/// instruction selection via BURG.
///
class InstrTreeNode {
  InstrTreeNode(const InstrTreeNode &);   // DO NOT IMPLEMENT
  void operator=(const InstrTreeNode &);  // DO NOT IMPLEMENT
public:
  enum InstrTreeNodeType { NTInstructionNode,
                           NTVRegListNode,
                           NTVRegNode,
                           NTConstNode,
                           NTLabelNode };
  InstrTreeNode* LeftChild;
  InstrTreeNode* RightChild;
  InstrTreeNode* Parent;
  OpLabel        opLabel;
  StateLabel     state;

protected:
  InstrTreeNodeType treeNodeType;
  Value*           val;

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
  InstrTreeNodeType     getNodeType     () const { return treeNodeType; }
  Value*                getValue        () const { return val; }
  inline OpLabel        getOpLabel      () const { return opLabel; }
  inline InstrTreeNode *leftChild       () const { return LeftChild; }
  inline InstrTreeNode *parent          () const { return Parent; }

  // If right child is a list node, recursively get its *left* child
  inline InstrTreeNode* rightChild() const {
    return (!RightChild ? 0 :
            (RightChild->getOpLabel() == VRegListOp
             ? RightChild->LeftChild : RightChild));
  }
  void dump(int dumpChildren, int indent) const;
protected:
  virtual void dumpNode(int indent) const = 0;
  friend class InstrForest;
};

} // end namespace llvm.

#endif
