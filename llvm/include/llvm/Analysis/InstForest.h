//===- llvm/Analysis/InstForest.h - Partition Func into forest --*- C++ -*-===//
//
// This interface is used to partition a method into a forest of instruction
// trees, where the following invariants hold:
//
// 1. The instructions in a tree are all related to each other through use
//    relationships.
// 2. All instructions in a tree are members of the same basic block
// 3. All instructions in a tree (with the exception of the root), may have only
//    a single user.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_INSTFOREST_H
#define LLVM_ANALYSIS_INSTFOREST_H

#include "llvm/Function.h"
#include "Support/Tree.h"
#include <map>

template<class Payload> class InstTreeNode;
template<class Payload> class InstForest;

//===----------------------------------------------------------------------===//
//  Class InstTreeNode
//===----------------------------------------------------------------------===//
//
// There is an instance of this class for each node in the instruction forest.
// There should be a node for every instruction in the tree, as well as
// Temporary nodes that correspond to other trees in the forest and to variables
// and global variables.  Constants have their own special node.
//
template<class Payload>
class InstTreeNode : 
    public Tree<InstTreeNode<Payload>, 
                std::pair<std::pair<Value*, char>, Payload> > {

  friend class InstForest<Payload>;
  typedef Tree<InstTreeNode<Payload>,
               std::pair<std::pair<Value*, char>, Payload> > super;

  // Constants used for the node type value
  enum NodeTypeTy {
    ConstNode        = Value::ConstantVal,
    BasicBlockNode   = Value::BasicBlockVal,
    InstructionNode  = Value::InstructionVal,
    TemporaryNode    = -1
  };

  // Helper functions to make accessing our data nicer...
  const Value *getValue() const { return getTreeData().first.first; }
        Value *getValue()       { return getTreeData().first.first; }
  enum NodeTypeTy getNodeType() const {
    return (enum NodeTypeTy)getTreeData().first.second;
  }

  InstTreeNode(const InstTreeNode &);     // Do not implement
  void operator=(const InstTreeNode &);   // Do not implement

  // Only creatable by InstForest
  InstTreeNode(InstForest<Payload> &IF, Value *V, InstTreeNode *Parent);
  bool CanMergeInstIntoTree(Instruction *Inst);
public:
  // Accessor functions...
  inline       Payload &getData()       { return getTreeData().second; }
  inline const Payload &getData() const { return getTreeData().second; }

  // Type checking functions...
  inline bool isConstant()    const { return getNodeType() == ConstNode; }
  inline bool isBasicBlock()  const { return getNodeType() == BasicBlockNode; }
  inline bool isInstruction() const { return getNodeType() == InstructionNode; }
  inline bool isTemporary()   const { return getNodeType() == TemporaryNode; }

  // Accessors for different node types...
  inline Constant *getConstant() {
    return cast<Constant>(getValue());
  }
  inline const Constant *getConstant() const {
    return cast<Constant>(getValue());
  }
  inline BasicBlock *getBasicBlock() {
    return cast<BasicBlock>(getValue());
  }
  inline const BasicBlock *getBasicBlock() const {
    return cast<BasicBlock>(getValue());
  }
  inline Instruction *getInstruction() {
    assert(isInstruction() && "getInstruction() on non instruction node!");
    return cast<Instruction>(getValue());
  }
  inline const Instruction *getInstruction() const {
    assert(isInstruction() && "getInstruction() on non instruction node!");
    return cast<Instruction>(getValue());
  }
  inline Instruction *getTemporary() {
    assert(isTemporary() && "getTemporary() on non temporary node!");
    return cast<Instruction>(getValue());
  }
  inline const Instruction *getTemporary() const {
    assert(isTemporary() && "getTemporary() on non temporary node!");
    return cast<Instruction>(getValue());
  }

public:
  // print - Called by operator<< below...
  void print(std::ostream &o, unsigned Indent) const {
    o << std::string(Indent*2, ' ');
    switch (getNodeType()) {
    case ConstNode      : o << "Constant   : "; break;
    case BasicBlockNode : o << "BasicBlock : " << getValue()->getName() << "\n";
      return;
    case InstructionNode: o << "Instruction: "; break;
    case TemporaryNode  : o << "Temporary  : "; break;
    default: o << "UNKNOWN NODE TYPE: " << getNodeType() << "\n"; abort();
    }

    o << getValue();
    if (!isa<Instruction>(getValue())) o << "\n";

    for (unsigned i = 0; i < getNumChildren(); ++i)
      getChild(i)->print(o, Indent+1);
  }
};

template<class Payload>
inline std::ostream &operator<<(std::ostream &o,
                                const InstTreeNode<Payload> *N) {
  N->print(o, 0); return o;
}

//===----------------------------------------------------------------------===//
//  Class InstForest
//===----------------------------------------------------------------------===//
//
// This class represents the instruction forest itself.  It exposes iterators
// to an underlying vector of Instruction Trees.  Each root of the tree is 
// guaranteed to be an instruction node.  The constructor builds the forest.
//
template<class Payload>
class InstForest : public std::vector<InstTreeNode<Payload> *> {
  friend class InstTreeNode<Payload>;

  typedef typename std::vector<InstTreeNode<Payload> *>::const_iterator const_iterator;

  // InstMap - Map contains entries for ALL instructions in the method and the
  // InstTreeNode that they correspond to.
  //
  std::map<Instruction*, InstTreeNode<Payload> *> InstMap;

  void addInstMapping(Instruction *I, InstTreeNode<Payload> *IN) {
    InstMap.insert(std::make_pair(I, IN));
  }

  void removeInstFromRootList(Instruction *I) {
    for (unsigned i = size(); i > 0; --i)
      if (operator[](i-1)->getValue() == I) {
	erase(begin()+i-1);
	return;
      }
  }

public:
  // ctor - Create an instruction forest for the specified method...
  InstForest(Function *F) {
    for (Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB)
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
        if (!getInstNode(I)) {  // Do we already have a tree for this inst?
          // No, create one!  InstTreeNode ctor automatically adds the
          // created node into our InstMap
          push_back(new InstTreeNode<Payload>(*this, I, 0));
        }
  }

  // dtor - Free the trees...
  ~InstForest() {
    for (unsigned i = size(); i != 0; --i)
      delete operator[](i-1);
  }

  // getInstNode - Return the instruction node that corresponds to the specified
  // instruction...  This node may be embeded in a larger tree, in which case
  // the parent pointer can be used to find the root of the tree.
  //
  inline InstTreeNode<Payload> *getInstNode(Instruction *Inst) {
    typename std::map<Instruction*, InstTreeNode<Payload> *>::iterator I =
      InstMap.find(Inst);
    if (I != InstMap.end()) return I->second;
    return 0;
  }
  inline const InstTreeNode<Payload> *getInstNode(const Instruction *Inst)const{
    typename std::map<Instruction*, InstTreeNode<Payload>*>::const_iterator I = 
      InstMap.find(Inst);
    if (I != InstMap.end()) return I->second;
    return 0;
  }

  // print - Called by operator<< below...
  void print(std::ostream &out) const {
    for (const_iterator I = begin(), E = end(); I != E; ++I)
      out << *I;
  }
};

template<class Payload>
inline std::ostream &operator<<(std::ostream &o, const InstForest<Payload> &IF){
  IF.print(o); return o;
}


//===----------------------------------------------------------------------===//
//  Method Implementations
//===----------------------------------------------------------------------===//

// CanMergeInstIntoTree - Return true if it is allowed to merge the specified
// instruction into 'this' instruction tree.  This is allowed iff:
//   1. The instruction is in the same basic block as the current one
//   2. The instruction has only one use
//
template <class Payload>
bool InstTreeNode<Payload>::CanMergeInstIntoTree(Instruction *I) {
  if (!I->use_empty() && !I->hasOneUse()) return false;
  return I->getParent() == cast<Instruction>(getValue())->getParent();
}


// InstTreeNode ctor - This constructor creates the instruction tree for the
// specified value.  If the value is an instruction, it recursively creates the 
// internal/child nodes and adds them to the instruction forest.
//
template <class Payload>
InstTreeNode<Payload>::InstTreeNode(InstForest<Payload> &IF, Value *V,
				    InstTreeNode *Parent) : super(Parent) {
  getTreeData().first.first = V;   // Save tree node
 
  if (!isa<Instruction>(V)) {
    assert((isa<Constant>(V) || isa<BasicBlock>(V) ||
	    isa<Argument>(V) || isa<GlobalValue>(V)) &&
	   "Unrecognized value type for InstForest Partition!");
    if (isa<Constant>(V))
      getTreeData().first.second = ConstNode;
    else if (isa<BasicBlock>(V))
      getTreeData().first.second = BasicBlockNode;
    else 
      getTreeData().first.second = TemporaryNode;
      
    return;
  }

  // Must be an instruction then... see if we can include it in this tree!
  Instruction *I = cast<Instruction>(V);
  if (Parent && !Parent->CanMergeInstIntoTree(I)) {
    // Not root node of tree, but mult uses?
    getTreeData().first.second = TemporaryNode;   // Must be a temporary!
    return;
  }

  // Otherwise, we are an internal instruction node.  We must process our
  // uses and add them as children of this node.
  //
  std::vector<InstTreeNode*> Children;

  // Make sure that the forest knows about us!
  IF.addInstMapping(I, this);
    
  // Walk the operands of the instruction adding children for all of the uses
  // of the instruction...
  // 
  for (Instruction::op_iterator OI = I->op_begin(); OI != I->op_end(); ++OI) {
    Value *Operand = *OI;
    InstTreeNode<Payload> *IN = IF.getInstNode(dyn_cast<Instruction>(Operand));
    if (IN && CanMergeInstIntoTree(cast<Instruction>(Operand))) {
      Children.push_back(IN);
      IF.removeInstFromRootList(cast<Instruction>(Operand));
    } else {
      // No node for this child yet... create one now!
      Children.push_back(new InstTreeNode(IF, *OI, this));
    }
  }

  setChildren(Children);
  getTreeData().first.second = InstructionNode;
}

#endif

