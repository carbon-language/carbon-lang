//===- DAGISelEmitter.cpp - Generate an instruction selector --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a DAG instruction selector.
//
//===----------------------------------------------------------------------===//

#include "DAGISelEmitter.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Helpers for working with extended types.

/// FilterVTs - Filter a list of VT's according to a predicate.
///
template<typename T>
static std::vector<MVT::ValueType> 
FilterVTs(const std::vector<MVT::ValueType> &InVTs, T Filter) {
  std::vector<MVT::ValueType> Result;
  for (unsigned i = 0, e = InVTs.size(); i != e; ++i)
    if (Filter(InVTs[i]))
      Result.push_back(InVTs[i]);
  return Result;
}

/// isExtIntegerVT - Return true if the specified extended value type is
/// integer, or isInt.
static bool isExtIntegerVT(unsigned char VT) {
  return VT == MVT::isInt ||
        (VT < MVT::LAST_VALUETYPE && MVT::isInteger((MVT::ValueType)VT));
}

/// isExtFloatingPointVT - Return true if the specified extended value type is
/// floating point, or isFP.
static bool isExtFloatingPointVT(unsigned char VT) {
  return VT == MVT::isFP ||
        (VT < MVT::LAST_VALUETYPE && MVT::isFloatingPoint((MVT::ValueType)VT));
}

//===----------------------------------------------------------------------===//
// SDTypeConstraint implementation
//

SDTypeConstraint::SDTypeConstraint(Record *R) {
  OperandNo = R->getValueAsInt("OperandNum");
  
  if (R->isSubClassOf("SDTCisVT")) {
    ConstraintType = SDTCisVT;
    x.SDTCisVT_Info.VT = getValueType(R->getValueAsDef("VT"));
  } else if (R->isSubClassOf("SDTCisPtrTy")) {
    ConstraintType = SDTCisPtrTy;
  } else if (R->isSubClassOf("SDTCisInt")) {
    ConstraintType = SDTCisInt;
  } else if (R->isSubClassOf("SDTCisFP")) {
    ConstraintType = SDTCisFP;
  } else if (R->isSubClassOf("SDTCisSameAs")) {
    ConstraintType = SDTCisSameAs;
    x.SDTCisSameAs_Info.OtherOperandNum = R->getValueAsInt("OtherOperandNum");
  } else if (R->isSubClassOf("SDTCisVTSmallerThanOp")) {
    ConstraintType = SDTCisVTSmallerThanOp;
    x.SDTCisVTSmallerThanOp_Info.OtherOperandNum = 
      R->getValueAsInt("OtherOperandNum");
  } else if (R->isSubClassOf("SDTCisOpSmallerThanOp")) {
    ConstraintType = SDTCisOpSmallerThanOp;
    x.SDTCisOpSmallerThanOp_Info.BigOperandNum = 
      R->getValueAsInt("BigOperandNum");
  } else {
    std::cerr << "Unrecognized SDTypeConstraint '" << R->getName() << "'!\n";
    exit(1);
  }
}

/// getOperandNum - Return the node corresponding to operand #OpNo in tree
/// N, which has NumResults results.
TreePatternNode *SDTypeConstraint::getOperandNum(unsigned OpNo,
                                                 TreePatternNode *N,
                                                 unsigned NumResults) const {
  assert(NumResults <= 1 &&
         "We only work with nodes with zero or one result so far!");
  
  if (OpNo < NumResults)
    return N;  // FIXME: need value #
  else
    return N->getChild(OpNo-NumResults);
}

/// ApplyTypeConstraint - Given a node in a pattern, apply this type
/// constraint to the nodes operands.  This returns true if it makes a
/// change, false otherwise.  If a type contradiction is found, throw an
/// exception.
bool SDTypeConstraint::ApplyTypeConstraint(TreePatternNode *N,
                                           const SDNodeInfo &NodeInfo,
                                           TreePattern &TP) const {
  unsigned NumResults = NodeInfo.getNumResults();
  assert(NumResults <= 1 &&
         "We only work with nodes with zero or one result so far!");
  
  // Check that the number of operands is sane.
  if (NodeInfo.getNumOperands() >= 0) {
    if (N->getNumChildren() != (unsigned)NodeInfo.getNumOperands())
      TP.error(N->getOperator()->getName() + " node requires exactly " +
               itostr(NodeInfo.getNumOperands()) + " operands!");
  }

  const CodeGenTarget &CGT = TP.getDAGISelEmitter().getTargetInfo();
  
  TreePatternNode *NodeToApply = getOperandNum(OperandNo, N, NumResults);
  
  switch (ConstraintType) {
  default: assert(0 && "Unknown constraint type!");
  case SDTCisVT:
    // Operand must be a particular type.
    return NodeToApply->UpdateNodeType(x.SDTCisVT_Info.VT, TP);
  case SDTCisPtrTy: {
    // Operand must be same as target pointer type.
    return NodeToApply->UpdateNodeType(CGT.getPointerType(), TP);
  }
  case SDTCisInt: {
    // If there is only one integer type supported, this must be it.
    std::vector<MVT::ValueType> IntVTs =
      FilterVTs(CGT.getLegalValueTypes(), MVT::isInteger);

    // If we found exactly one supported integer type, apply it.
    if (IntVTs.size() == 1)
      return NodeToApply->UpdateNodeType(IntVTs[0], TP);
    return NodeToApply->UpdateNodeType(MVT::isInt, TP);
  }
  case SDTCisFP: {
    // If there is only one FP type supported, this must be it.
    std::vector<MVT::ValueType> FPVTs =
      FilterVTs(CGT.getLegalValueTypes(), MVT::isFloatingPoint);
        
    // If we found exactly one supported FP type, apply it.
    if (FPVTs.size() == 1)
      return NodeToApply->UpdateNodeType(FPVTs[0], TP);
    return NodeToApply->UpdateNodeType(MVT::isFP, TP);
  }
  case SDTCisSameAs: {
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisSameAs_Info.OtherOperandNum, N, NumResults);
    return NodeToApply->UpdateNodeType(OtherNode->getExtType(), TP) |
           OtherNode->UpdateNodeType(NodeToApply->getExtType(), TP);
  }
  case SDTCisVTSmallerThanOp: {
    // The NodeToApply must be a leaf node that is a VT.  OtherOperandNum must
    // have an integer type that is smaller than the VT.
    if (!NodeToApply->isLeaf() ||
        !dynamic_cast<DefInit*>(NodeToApply->getLeafValue()) ||
        !static_cast<DefInit*>(NodeToApply->getLeafValue())->getDef()
               ->isSubClassOf("ValueType"))
      TP.error(N->getOperator()->getName() + " expects a VT operand!");
    MVT::ValueType VT =
     getValueType(static_cast<DefInit*>(NodeToApply->getLeafValue())->getDef());
    if (!MVT::isInteger(VT))
      TP.error(N->getOperator()->getName() + " VT operand must be integer!");
    
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisVTSmallerThanOp_Info.OtherOperandNum, N,NumResults);
    
    // It must be integer.
    bool MadeChange = false;
    MadeChange |= OtherNode->UpdateNodeType(MVT::isInt, TP);
    
    if (OtherNode->hasTypeSet() && OtherNode->getType() <= VT)
      OtherNode->UpdateNodeType(MVT::Other, TP);  // Throw an error.
    return false;
  }
  case SDTCisOpSmallerThanOp: {
    TreePatternNode *BigOperand =
      getOperandNum(x.SDTCisOpSmallerThanOp_Info.BigOperandNum, N, NumResults);

    // Both operands must be integer or FP, but we don't care which.
    bool MadeChange = false;
    
    if (isExtIntegerVT(NodeToApply->getExtType()))
      MadeChange |= BigOperand->UpdateNodeType(MVT::isInt, TP);
    else if (isExtFloatingPointVT(NodeToApply->getExtType()))
      MadeChange |= BigOperand->UpdateNodeType(MVT::isFP, TP);
    if (isExtIntegerVT(BigOperand->getExtType()))
      MadeChange |= NodeToApply->UpdateNodeType(MVT::isInt, TP);
    else if (isExtFloatingPointVT(BigOperand->getExtType()))
      MadeChange |= NodeToApply->UpdateNodeType(MVT::isFP, TP);

    std::vector<MVT::ValueType> VTs = CGT.getLegalValueTypes();
    
    if (isExtIntegerVT(NodeToApply->getExtType())) {
      VTs = FilterVTs(VTs, MVT::isInteger);
    } else if (isExtFloatingPointVT(NodeToApply->getExtType())) {
      VTs = FilterVTs(VTs, MVT::isFloatingPoint);
    } else {
      VTs.clear();
    }

    switch (VTs.size()) {
    default:         // Too many VT's to pick from.
    case 0: break;   // No info yet.
    case 1: 
      // Only one VT of this flavor.  Cannot ever satisify the constraints.
      return NodeToApply->UpdateNodeType(MVT::Other, TP);  // throw
    case 2:
      // If we have exactly two possible types, the little operand must be the
      // small one, the big operand should be the big one.  Common with 
      // float/double for example.
      assert(VTs[0] < VTs[1] && "Should be sorted!");
      MadeChange |= NodeToApply->UpdateNodeType(VTs[0], TP);
      MadeChange |= BigOperand->UpdateNodeType(VTs[1], TP);
      break;
    }    
    return MadeChange;
  }
  }  
  return false;
}


//===----------------------------------------------------------------------===//
// SDNodeInfo implementation
//
SDNodeInfo::SDNodeInfo(Record *R) : Def(R) {
  EnumName    = R->getValueAsString("Opcode");
  SDClassName = R->getValueAsString("SDClass");
  Record *TypeProfile = R->getValueAsDef("TypeProfile");
  NumResults = TypeProfile->getValueAsInt("NumResults");
  NumOperands = TypeProfile->getValueAsInt("NumOperands");
  
  // Parse the properties.
  Properties = 0;
  std::vector<Record*> PropList = R->getValueAsListOfDefs("Properties");
  for (unsigned i = 0, e = PropList.size(); i != e; ++i) {
    if (PropList[i]->getName() == "SDNPCommutative") {
      Properties |= 1 << SDNPCommutative;
    } else if (PropList[i]->getName() == "SDNPAssociative") {
      Properties |= 1 << SDNPAssociative;
    } else if (PropList[i]->getName() == "SDNPHasChain") {
      Properties |= 1 << SDNPHasChain;
    } else {
      std::cerr << "Unknown SD Node property '" << PropList[i]->getName()
                << "' on node '" << R->getName() << "'!\n";
      exit(1);
    }
  }
  
  
  // Parse the type constraints.
  std::vector<Record*> ConstraintList =
    TypeProfile->getValueAsListOfDefs("Constraints");
  TypeConstraints.assign(ConstraintList.begin(), ConstraintList.end());
}

//===----------------------------------------------------------------------===//
// TreePatternNode implementation
//

TreePatternNode::~TreePatternNode() {
#if 0 // FIXME: implement refcounted tree nodes!
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    delete getChild(i);
#endif
}

/// UpdateNodeType - Set the node type of N to VT if VT contains
/// information.  If N already contains a conflicting type, then throw an
/// exception.  This returns true if any information was updated.
///
bool TreePatternNode::UpdateNodeType(unsigned char VT, TreePattern &TP) {
  if (VT == MVT::isUnknown || getExtType() == VT) return false;
  if (getExtType() == MVT::isUnknown) {
    setType(VT);
    return true;
  }
  
  // If we are told this is to be an int or FP type, and it already is, ignore
  // the advice.
  if ((VT == MVT::isInt && isExtIntegerVT(getExtType())) ||
      (VT == MVT::isFP  && isExtFloatingPointVT(getExtType())))
    return false;
      
  // If we know this is an int or fp type, and we are told it is a specific one,
  // take the advice.
  if ((getExtType() == MVT::isInt && isExtIntegerVT(VT)) ||
      (getExtType() == MVT::isFP  && isExtFloatingPointVT(VT))) {
    setType(VT);
    return true;
  }      

  if (isLeaf()) {
    dump();
    TP.error("Type inference contradiction found in node!");
  } else {
    TP.error("Type inference contradiction found in node " + 
             getOperator()->getName() + "!");
  }
  return true; // unreachable
}


void TreePatternNode::print(std::ostream &OS) const {
  if (isLeaf()) {
    OS << *getLeafValue();
  } else {
    OS << "(" << getOperator()->getName();
  }
  
  switch (getExtType()) {
  case MVT::Other: OS << ":Other"; break;
  case MVT::isInt: OS << ":isInt"; break;
  case MVT::isFP : OS << ":isFP"; break;
  case MVT::isUnknown: ; /*OS << ":?";*/ break;
  default:  OS << ":" << getType(); break;
  }

  if (!isLeaf()) {
    if (getNumChildren() != 0) {
      OS << " ";
      getChild(0)->print(OS);
      for (unsigned i = 1, e = getNumChildren(); i != e; ++i) {
        OS << ", ";
        getChild(i)->print(OS);
      }
    }
    OS << ")";
  }
  
  if (!PredicateFn.empty())
    OS << "<<P:" << PredicateFn << ">>";
  if (TransformFn)
    OS << "<<X:" << TransformFn->getName() << ">>";
  if (!getName().empty())
    OS << ":$" << getName();

}
void TreePatternNode::dump() const {
  print(std::cerr);
}

/// isIsomorphicTo - Return true if this node is recursively isomorphic to
/// the specified node.  For this comparison, all of the state of the node
/// is considered, except for the assigned name.  Nodes with differing names
/// that are otherwise identical are considered isomorphic.
bool TreePatternNode::isIsomorphicTo(const TreePatternNode *N) const {
  if (N == this) return true;
  if (N->isLeaf() != isLeaf() || getExtType() != N->getExtType() ||
      getPredicateFn() != N->getPredicateFn() ||
      getTransformFn() != N->getTransformFn())
    return false;

  if (isLeaf()) {
    if (DefInit *DI = dynamic_cast<DefInit*>(getLeafValue()))
      if (DefInit *NDI = dynamic_cast<DefInit*>(N->getLeafValue()))
        return DI->getDef() == NDI->getDef();
    return getLeafValue() == N->getLeafValue();
  }
  
  if (N->getOperator() != getOperator() ||
      N->getNumChildren() != getNumChildren()) return false;
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (!getChild(i)->isIsomorphicTo(N->getChild(i)))
      return false;
  return true;
}

/// clone - Make a copy of this tree and all of its children.
///
TreePatternNode *TreePatternNode::clone() const {
  TreePatternNode *New;
  if (isLeaf()) {
    New = new TreePatternNode(getLeafValue());
  } else {
    std::vector<TreePatternNode*> CChildren;
    CChildren.reserve(Children.size());
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      CChildren.push_back(getChild(i)->clone());
    New = new TreePatternNode(getOperator(), CChildren);
  }
  New->setName(getName());
  New->setType(getExtType());
  New->setPredicateFn(getPredicateFn());
  New->setTransformFn(getTransformFn());
  return New;
}

/// SubstituteFormalArguments - Replace the formal arguments in this tree
/// with actual values specified by ArgMap.
void TreePatternNode::
SubstituteFormalArguments(std::map<std::string, TreePatternNode*> &ArgMap) {
  if (isLeaf()) return;
  
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = getChild(i);
    if (Child->isLeaf()) {
      Init *Val = Child->getLeafValue();
      if (dynamic_cast<DefInit*>(Val) &&
          static_cast<DefInit*>(Val)->getDef()->getName() == "node") {
        // We found a use of a formal argument, replace it with its value.
        Child = ArgMap[Child->getName()];
        assert(Child && "Couldn't find formal argument!");
        setChild(i, Child);
      }
    } else {
      getChild(i)->SubstituteFormalArguments(ArgMap);
    }
  }
}


/// InlinePatternFragments - If this pattern refers to any pattern
/// fragments, inline them into place, giving us a pattern without any
/// PatFrag references.
TreePatternNode *TreePatternNode::InlinePatternFragments(TreePattern &TP) {
  if (isLeaf()) return this;  // nothing to do.
  Record *Op = getOperator();
  
  if (!Op->isSubClassOf("PatFrag")) {
    // Just recursively inline children nodes.
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      setChild(i, getChild(i)->InlinePatternFragments(TP));
    return this;
  }

  // Otherwise, we found a reference to a fragment.  First, look up its
  // TreePattern record.
  TreePattern *Frag = TP.getDAGISelEmitter().getPatternFragment(Op);
  
  // Verify that we are passing the right number of operands.
  if (Frag->getNumArgs() != Children.size())
    TP.error("'" + Op->getName() + "' fragment requires " +
             utostr(Frag->getNumArgs()) + " operands!");

  TreePatternNode *FragTree = Frag->getOnlyTree()->clone();

  // Resolve formal arguments to their actual value.
  if (Frag->getNumArgs()) {
    // Compute the map of formal to actual arguments.
    std::map<std::string, TreePatternNode*> ArgMap;
    for (unsigned i = 0, e = Frag->getNumArgs(); i != e; ++i)
      ArgMap[Frag->getArgName(i)] = getChild(i)->InlinePatternFragments(TP);
  
    FragTree->SubstituteFormalArguments(ArgMap);
  }
  
  FragTree->setName(getName());
  FragTree->UpdateNodeType(getExtType(), TP);
  
  // Get a new copy of this fragment to stitch into here.
  //delete this;    // FIXME: implement refcounting!
  return FragTree;
}

/// getIntrinsicType - Check to see if the specified record has an intrinsic
/// type which should be applied to it.  This infer the type of register
/// references from the register file information, for example.
///
static unsigned char getIntrinsicType(Record *R, bool NotRegisters,
                                      TreePattern &TP) {
  // Check to see if this is a register or a register class...
  if (R->isSubClassOf("RegisterClass")) {
    if (NotRegisters) return MVT::isUnknown;
    const CodeGenRegisterClass &RC = 
      TP.getDAGISelEmitter().getTargetInfo().getRegisterClass(R);
    return RC.getValueTypeNum(0);
  } else if (R->isSubClassOf("PatFrag")) {
    // Pattern fragment types will be resolved when they are inlined.
    return MVT::isUnknown;
  } else if (R->isSubClassOf("Register")) {
    // If the register appears in exactly one regclass, and the regclass has one
    // value type, use it as the known type.
    const CodeGenTarget &T = TP.getDAGISelEmitter().getTargetInfo();
    if (const CodeGenRegisterClass *RC = T.getRegisterClassForRegister(R))
      if (RC->getNumValueTypes() == 1)
        return RC->getValueTypeNum(0);
    return MVT::isUnknown;
  } else if (R->isSubClassOf("ValueType") || R->isSubClassOf("CondCode")) {
    // Using a VTSDNode or CondCodeSDNode.
    return MVT::Other;
  } else if (R->isSubClassOf("ComplexPattern")) {
    return TP.getDAGISelEmitter().getComplexPattern(R).getValueType();
  } else if (R->getName() == "node") {
    // Placeholder.
    return MVT::isUnknown;
  }
  
  TP.error("Unknown node flavor used in pattern: " + R->getName());
  return MVT::Other;
}

/// ApplyTypeConstraints - Apply all of the type constraints relevent to
/// this node and its children in the tree.  This returns true if it makes a
/// change, false otherwise.  If a type contradiction is found, throw an
/// exception.
bool TreePatternNode::ApplyTypeConstraints(TreePattern &TP, bool NotRegisters) {
  if (isLeaf()) {
    if (DefInit *DI = dynamic_cast<DefInit*>(getLeafValue())) {
      // If it's a regclass or something else known, include the type.
      return UpdateNodeType(getIntrinsicType(DI->getDef(), NotRegisters, TP),
                            TP);
    } else if (IntInit *II = dynamic_cast<IntInit*>(getLeafValue())) {
      // Int inits are always integers. :)
      bool MadeChange = UpdateNodeType(MVT::isInt, TP);
      
      if (hasTypeSet()) {
        unsigned Size = MVT::getSizeInBits(getType());
        // Make sure that the value is representable for this type.
        if (Size < 32) {
          int Val = (II->getValue() << (32-Size)) >> (32-Size);
          if (Val != II->getValue())
            TP.error("Sign-extended integer value '" + itostr(II->getValue()) +
                     "' is out of range for type 'MVT::" + 
                     getEnumName(getType()) + "'!");
        }
      }
      
      return MadeChange;
    }
    return false;
  }
  
  // special handling for set, which isn't really an SDNode.
  if (getOperator()->getName() == "set") {
    assert (getNumChildren() == 2 && "Only handle 2 operand set's for now!");
    bool MadeChange = getChild(0)->ApplyTypeConstraints(TP, NotRegisters);
    MadeChange |= getChild(1)->ApplyTypeConstraints(TP, NotRegisters);
    
    // Types of operands must match.
    MadeChange |= getChild(0)->UpdateNodeType(getChild(1)->getExtType(), TP);
    MadeChange |= getChild(1)->UpdateNodeType(getChild(0)->getExtType(), TP);
    MadeChange |= UpdateNodeType(MVT::isVoid, TP);
    return MadeChange;
  } else if (getOperator()->isSubClassOf("SDNode")) {
    const SDNodeInfo &NI = TP.getDAGISelEmitter().getSDNodeInfo(getOperator());
    
    bool MadeChange = NI.ApplyTypeConstraints(this, TP);
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    // Branch, etc. do not produce results and top-level forms in instr pattern
    // must have void types.
    if (NI.getNumResults() == 0)
      MadeChange |= UpdateNodeType(MVT::isVoid, TP);
    return MadeChange;  
  } else if (getOperator()->isSubClassOf("Instruction")) {
    const DAGInstruction &Inst =
      TP.getDAGISelEmitter().getInstruction(getOperator());
    bool MadeChange = false;
    unsigned NumResults = Inst.getNumResults();
    
    assert(NumResults <= 1 &&
           "Only supports zero or one result instrs!");
    // Apply the result type to the node
    if (NumResults == 0) {
      MadeChange = UpdateNodeType(MVT::isVoid, TP);
    } else {
      Record *ResultNode = Inst.getResult(0);
      assert(ResultNode->isSubClassOf("RegisterClass") &&
             "Operands should be register classes!");

      const CodeGenRegisterClass &RC = 
        TP.getDAGISelEmitter().getTargetInfo().getRegisterClass(ResultNode);

      // Get the first ValueType in the RegClass, it's as good as any.
      MadeChange = UpdateNodeType(RC.getValueTypeNum(0), TP);
    }

    if (getNumChildren() != Inst.getNumOperands())
      TP.error("Instruction '" + getOperator()->getName() + " expects " +
               utostr(Inst.getNumOperands()) + " operands, not " +
               utostr(getNumChildren()) + " operands!");
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i) {
      Record *OperandNode = Inst.getOperand(i);
      MVT::ValueType VT;
      if (OperandNode->isSubClassOf("RegisterClass")) {
        const CodeGenRegisterClass &RC = 
          TP.getDAGISelEmitter().getTargetInfo().getRegisterClass(OperandNode);
        VT = RC.getValueTypeNum(0);
      } else if (OperandNode->isSubClassOf("Operand")) {
        VT = getValueType(OperandNode->getValueAsDef("Type"));
      } else {
        assert(0 && "Unknown operand type!");
        abort();
      }
      
      MadeChange |= getChild(i)->UpdateNodeType(VT, TP);
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    }
    return MadeChange;
  } else {
    assert(getOperator()->isSubClassOf("SDNodeXForm") && "Unknown node type!");
    
    // Node transforms always take one operand, and take and return the same
    // type.
    if (getNumChildren() != 1)
      TP.error("Node transform '" + getOperator()->getName() +
               "' requires one operand!");
    bool MadeChange = UpdateNodeType(getChild(0)->getExtType(), TP);
    MadeChange |= getChild(0)->UpdateNodeType(getExtType(), TP);
    return MadeChange;
  }
}

/// canPatternMatch - If it is impossible for this pattern to match on this
/// target, fill in Reason and return false.  Otherwise, return true.  This is
/// used as a santity check for .td files (to prevent people from writing stuff
/// that can never possibly work), and to prevent the pattern permuter from
/// generating stuff that is useless.
bool TreePatternNode::canPatternMatch(std::string &Reason, DAGISelEmitter &ISE){
  if (isLeaf()) return true;

  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (!getChild(i)->canPatternMatch(Reason, ISE))
      return false;

  // If this node is a commutative operator, check that the LHS isn't an
  // immediate.
  const SDNodeInfo &NodeInfo = ISE.getSDNodeInfo(getOperator());
  if (NodeInfo.hasProperty(SDNodeInfo::SDNPCommutative)) {
    // Scan all of the operands of the node and make sure that only the last one
    // is a constant node.
    for (unsigned i = 0, e = getNumChildren()-1; i != e; ++i)
      if (!getChild(i)->isLeaf() && 
          getChild(i)->getOperator()->getName() == "imm") {
        Reason = "Immediate value must be on the RHS of commutative operators!";
        return false;
      }
  }
  
  return true;
}

//===----------------------------------------------------------------------===//
// TreePattern implementation
//

TreePattern::TreePattern(Record *TheRec, ListInit *RawPat, bool isInput,
                         DAGISelEmitter &ise) : TheRecord(TheRec), ISE(ise) {
   isInputPattern = isInput;
   for (unsigned i = 0, e = RawPat->getSize(); i != e; ++i)
     Trees.push_back(ParseTreePattern((DagInit*)RawPat->getElement(i)));
}

TreePattern::TreePattern(Record *TheRec, DagInit *Pat, bool isInput,
                         DAGISelEmitter &ise) : TheRecord(TheRec), ISE(ise) {
  isInputPattern = isInput;
  Trees.push_back(ParseTreePattern(Pat));
}

TreePattern::TreePattern(Record *TheRec, TreePatternNode *Pat, bool isInput,
                         DAGISelEmitter &ise) : TheRecord(TheRec), ISE(ise) {
  isInputPattern = isInput;
  Trees.push_back(Pat);
}



void TreePattern::error(const std::string &Msg) const {
  dump();
  throw "In " + TheRecord->getName() + ": " + Msg;
}

TreePatternNode *TreePattern::ParseTreePattern(DagInit *Dag) {
  Record *Operator = Dag->getNodeType();
  
  if (Operator->isSubClassOf("ValueType")) {
    // If the operator is a ValueType, then this must be "type cast" of a leaf
    // node.
    if (Dag->getNumArgs() != 1)
      error("Type cast only takes one operand!");
    
    Init *Arg = Dag->getArg(0);
    TreePatternNode *New;
    if (DefInit *DI = dynamic_cast<DefInit*>(Arg)) {
      Record *R = DI->getDef();
      if (R->isSubClassOf("SDNode") || R->isSubClassOf("PatFrag")) {
        Dag->setArg(0, new DagInit(R,
                                std::vector<std::pair<Init*, std::string> >()));
        return ParseTreePattern(Dag);
      }
      New = new TreePatternNode(DI);
    } else if (DagInit *DI = dynamic_cast<DagInit*>(Arg)) {
      New = ParseTreePattern(DI);
    } else if (IntInit *II = dynamic_cast<IntInit*>(Arg)) {
      New = new TreePatternNode(II);
      if (!Dag->getArgName(0).empty())
        error("Constant int argument should not have a name!");
    } else {
      Arg->dump();
      error("Unknown leaf value for tree pattern!");
      return 0;
    }
    
    // Apply the type cast.
    New->UpdateNodeType(getValueType(Operator), *this);
    New->setName(Dag->getArgName(0));
    return New;
  }
  
  // Verify that this is something that makes sense for an operator.
  if (!Operator->isSubClassOf("PatFrag") && !Operator->isSubClassOf("SDNode") &&
      !Operator->isSubClassOf("Instruction") && 
      !Operator->isSubClassOf("SDNodeXForm") &&
      Operator->getName() != "set")
    error("Unrecognized node '" + Operator->getName() + "'!");
  
  //  Check to see if this is something that is illegal in an input pattern.
  if (isInputPattern && (Operator->isSubClassOf("Instruction") ||
      Operator->isSubClassOf("SDNodeXForm")))
    error("Cannot use '" + Operator->getName() + "' in an input pattern!");
  
  std::vector<TreePatternNode*> Children;
  
  for (unsigned i = 0, e = Dag->getNumArgs(); i != e; ++i) {
    Init *Arg = Dag->getArg(i);
    if (DagInit *DI = dynamic_cast<DagInit*>(Arg)) {
      Children.push_back(ParseTreePattern(DI));
      if (Children.back()->getName().empty())
        Children.back()->setName(Dag->getArgName(i));
    } else if (DefInit *DefI = dynamic_cast<DefInit*>(Arg)) {
      Record *R = DefI->getDef();
      // Direct reference to a leaf DagNode or PatFrag?  Turn it into a
      // TreePatternNode if its own.
      if (R->isSubClassOf("SDNode") || R->isSubClassOf("PatFrag")) {
        Dag->setArg(i, new DagInit(R,
                              std::vector<std::pair<Init*, std::string> >()));
        --i;  // Revisit this node...
      } else {
        TreePatternNode *Node = new TreePatternNode(DefI);
        Node->setName(Dag->getArgName(i));
        Children.push_back(Node);
        
        // Input argument?
        if (R->getName() == "node") {
          if (Dag->getArgName(i).empty())
            error("'node' argument requires a name to match with operand list");
          Args.push_back(Dag->getArgName(i));
        }
      }
    } else if (IntInit *II = dynamic_cast<IntInit*>(Arg)) {
      TreePatternNode *Node = new TreePatternNode(II);
      if (!Dag->getArgName(i).empty())
        error("Constant int argument should not have a name!");
      Children.push_back(Node);
    } else {
      std::cerr << '"';
      Arg->dump();
      std::cerr << "\": ";
      error("Unknown leaf value for tree pattern!");
    }
  }
  
  return new TreePatternNode(Operator, Children);
}

/// InferAllTypes - Infer/propagate as many types throughout the expression
/// patterns as possible.  Return true if all types are infered, false
/// otherwise.  Throw an exception if a type contradiction is found.
bool TreePattern::InferAllTypes() {
  bool MadeChange = true;
  while (MadeChange) {
    MadeChange = false;
    for (unsigned i = 0, e = Trees.size(); i != e; ++i)
      MadeChange |= Trees[i]->ApplyTypeConstraints(*this, false);
  }
  
  bool HasUnresolvedTypes = false;
  for (unsigned i = 0, e = Trees.size(); i != e; ++i)
    HasUnresolvedTypes |= Trees[i]->ContainsUnresolvedType();
  return !HasUnresolvedTypes;
}

void TreePattern::print(std::ostream &OS) const {
  OS << getRecord()->getName();
  if (!Args.empty()) {
    OS << "(" << Args[0];
    for (unsigned i = 1, e = Args.size(); i != e; ++i)
      OS << ", " << Args[i];
    OS << ")";
  }
  OS << ": ";
  
  if (Trees.size() > 1)
    OS << "[\n";
  for (unsigned i = 0, e = Trees.size(); i != e; ++i) {
    OS << "\t";
    Trees[i]->print(OS);
    OS << "\n";
  }

  if (Trees.size() > 1)
    OS << "]\n";
}

void TreePattern::dump() const { print(std::cerr); }



//===----------------------------------------------------------------------===//
// DAGISelEmitter implementation
//

// Parse all of the SDNode definitions for the target, populating SDNodes.
void DAGISelEmitter::ParseNodeInfo() {
  std::vector<Record*> Nodes = Records.getAllDerivedDefinitions("SDNode");
  while (!Nodes.empty()) {
    SDNodes.insert(std::make_pair(Nodes.back(), Nodes.back()));
    Nodes.pop_back();
  }
}

/// ParseNodeTransforms - Parse all SDNodeXForm instances into the SDNodeXForms
/// map, and emit them to the file as functions.
void DAGISelEmitter::ParseNodeTransforms(std::ostream &OS) {
  OS << "\n// Node transformations.\n";
  std::vector<Record*> Xforms = Records.getAllDerivedDefinitions("SDNodeXForm");
  while (!Xforms.empty()) {
    Record *XFormNode = Xforms.back();
    Record *SDNode = XFormNode->getValueAsDef("Opcode");
    std::string Code = XFormNode->getValueAsCode("XFormFunction");
    SDNodeXForms.insert(std::make_pair(XFormNode,
                                       std::make_pair(SDNode, Code)));

    if (!Code.empty()) {
      std::string ClassName = getSDNodeInfo(SDNode).getSDClassName();
      const char *C2 = ClassName == "SDNode" ? "N" : "inN";

      OS << "inline SDOperand Transform_" << XFormNode->getName()
         << "(SDNode *" << C2 << ") {\n";
      if (ClassName != "SDNode")
        OS << "  " << ClassName << " *N = cast<" << ClassName << ">(inN);\n";
      OS << Code << "\n}\n";
    }

    Xforms.pop_back();
  }
}

void DAGISelEmitter::ParseComplexPatterns() {
  std::vector<Record*> AMs = Records.getAllDerivedDefinitions("ComplexPattern");
  while (!AMs.empty()) {
    ComplexPatterns.insert(std::make_pair(AMs.back(), AMs.back()));
    AMs.pop_back();
  }
}


/// ParsePatternFragments - Parse all of the PatFrag definitions in the .td
/// file, building up the PatternFragments map.  After we've collected them all,
/// inline fragments together as necessary, so that there are no references left
/// inside a pattern fragment to a pattern fragment.
///
/// This also emits all of the predicate functions to the output file.
///
void DAGISelEmitter::ParsePatternFragments(std::ostream &OS) {
  std::vector<Record*> Fragments = Records.getAllDerivedDefinitions("PatFrag");
  
  // First step, parse all of the fragments and emit predicate functions.
  OS << "\n// Predicate functions.\n";
  for (unsigned i = 0, e = Fragments.size(); i != e; ++i) {
    DagInit *Tree = Fragments[i]->getValueAsDag("Fragment");
    TreePattern *P = new TreePattern(Fragments[i], Tree, true, *this);
    PatternFragments[Fragments[i]] = P;
    
    // Validate the argument list, converting it to map, to discard duplicates.
    std::vector<std::string> &Args = P->getArgList();
    std::set<std::string> OperandsMap(Args.begin(), Args.end());
    
    if (OperandsMap.count(""))
      P->error("Cannot have unnamed 'node' values in pattern fragment!");
    
    // Parse the operands list.
    DagInit *OpsList = Fragments[i]->getValueAsDag("Operands");
    if (OpsList->getNodeType()->getName() != "ops")
      P->error("Operands list should start with '(ops ... '!");
    
    // Copy over the arguments.       
    Args.clear();
    for (unsigned j = 0, e = OpsList->getNumArgs(); j != e; ++j) {
      if (!dynamic_cast<DefInit*>(OpsList->getArg(j)) ||
          static_cast<DefInit*>(OpsList->getArg(j))->
          getDef()->getName() != "node")
        P->error("Operands list should all be 'node' values.");
      if (OpsList->getArgName(j).empty())
        P->error("Operands list should have names for each operand!");
      if (!OperandsMap.count(OpsList->getArgName(j)))
        P->error("'" + OpsList->getArgName(j) +
                 "' does not occur in pattern or was multiply specified!");
      OperandsMap.erase(OpsList->getArgName(j));
      Args.push_back(OpsList->getArgName(j));
    }
    
    if (!OperandsMap.empty())
      P->error("Operands list does not contain an entry for operand '" +
               *OperandsMap.begin() + "'!");

    // If there is a code init for this fragment, emit the predicate code and
    // keep track of the fact that this fragment uses it.
    std::string Code = Fragments[i]->getValueAsCode("Predicate");
    if (!Code.empty()) {
      assert(!P->getOnlyTree()->isLeaf() && "Can't be a leaf!");
      std::string ClassName =
        getSDNodeInfo(P->getOnlyTree()->getOperator()).getSDClassName();
      const char *C2 = ClassName == "SDNode" ? "N" : "inN";
      
      OS << "inline bool Predicate_" << Fragments[i]->getName()
         << "(SDNode *" << C2 << ") {\n";
      if (ClassName != "SDNode")
        OS << "  " << ClassName << " *N = cast<" << ClassName << ">(inN);\n";
      OS << Code << "\n}\n";
      P->getOnlyTree()->setPredicateFn("Predicate_"+Fragments[i]->getName());
    }
    
    // If there is a node transformation corresponding to this, keep track of
    // it.
    Record *Transform = Fragments[i]->getValueAsDef("OperandTransform");
    if (!getSDNodeTransform(Transform).second.empty())    // not noop xform?
      P->getOnlyTree()->setTransformFn(Transform);
  }
  
  OS << "\n\n";

  // Now that we've parsed all of the tree fragments, do a closure on them so
  // that there are not references to PatFrags left inside of them.
  for (std::map<Record*, TreePattern*>::iterator I = PatternFragments.begin(),
       E = PatternFragments.end(); I != E; ++I) {
    TreePattern *ThePat = I->second;
    ThePat->InlinePatternFragments();
        
    // Infer as many types as possible.  Don't worry about it if we don't infer
    // all of them, some may depend on the inputs of the pattern.
    try {
      ThePat->InferAllTypes();
    } catch (...) {
      // If this pattern fragment is not supported by this target (no types can
      // satisfy its constraints), just ignore it.  If the bogus pattern is
      // actually used by instructions, the type consistency error will be
      // reported there.
    }
    
    // If debugging, print out the pattern fragment result.
    DEBUG(ThePat->dump());
  }
}

/// HandleUse - Given "Pat" a leaf in the pattern, check to see if it is an
/// instruction input.  Return true if this is a real use.
static bool HandleUse(TreePattern *I, TreePatternNode *Pat,
                      std::map<std::string, TreePatternNode*> &InstInputs) {
  // No name -> not interesting.
  if (Pat->getName().empty()) {
    if (Pat->isLeaf()) {
      DefInit *DI = dynamic_cast<DefInit*>(Pat->getLeafValue());
      if (DI && DI->getDef()->isSubClassOf("RegisterClass"))
        I->error("Input " + DI->getDef()->getName() + " must be named!");

    }
    return false;
  }

  Record *Rec;
  if (Pat->isLeaf()) {
    DefInit *DI = dynamic_cast<DefInit*>(Pat->getLeafValue());
    if (!DI) I->error("Input $" + Pat->getName() + " must be an identifier!");
    Rec = DI->getDef();
  } else {
    assert(Pat->getNumChildren() == 0 && "can't be a use with children!");
    Rec = Pat->getOperator();
  }

  TreePatternNode *&Slot = InstInputs[Pat->getName()];
  if (!Slot) {
    Slot = Pat;
  } else {
    Record *SlotRec;
    if (Slot->isLeaf()) {
      SlotRec = dynamic_cast<DefInit*>(Slot->getLeafValue())->getDef();
    } else {
      assert(Slot->getNumChildren() == 0 && "can't be a use with children!");
      SlotRec = Slot->getOperator();
    }
    
    // Ensure that the inputs agree if we've already seen this input.
    if (Rec != SlotRec)
      I->error("All $" + Pat->getName() + " inputs must agree with each other");
    if (Slot->getExtType() != Pat->getExtType())
      I->error("All $" + Pat->getName() + " inputs must agree with each other");
  }
  return true;
}

/// FindPatternInputsAndOutputs - Scan the specified TreePatternNode (which is
/// part of "I", the instruction), computing the set of inputs and outputs of
/// the pattern.  Report errors if we see anything naughty.
void DAGISelEmitter::
FindPatternInputsAndOutputs(TreePattern *I, TreePatternNode *Pat,
                            std::map<std::string, TreePatternNode*> &InstInputs,
                            std::map<std::string, Record*> &InstResults) {
  if (Pat->isLeaf()) {
    bool isUse = HandleUse(I, Pat, InstInputs);
    if (!isUse && Pat->getTransformFn())
      I->error("Cannot specify a transform function for a non-input value!");
    return;
  } else if (Pat->getOperator()->getName() != "set") {
    // If this is not a set, verify that the children nodes are not void typed,
    // and recurse.
    for (unsigned i = 0, e = Pat->getNumChildren(); i != e; ++i) {
      if (Pat->getChild(i)->getExtType() == MVT::isVoid)
        I->error("Cannot have void nodes inside of patterns!");
      FindPatternInputsAndOutputs(I, Pat->getChild(i), InstInputs, InstResults);
    }
    
    // If this is a non-leaf node with no children, treat it basically as if
    // it were a leaf.  This handles nodes like (imm).
    bool isUse = false;
    if (Pat->getNumChildren() == 0)
      isUse = HandleUse(I, Pat, InstInputs);
    
    if (!isUse && Pat->getTransformFn())
      I->error("Cannot specify a transform function for a non-input value!");
    return;
  } 
  
  // Otherwise, this is a set, validate and collect instruction results.
  if (Pat->getNumChildren() == 0)
    I->error("set requires operands!");
  else if (Pat->getNumChildren() & 1)
    I->error("set requires an even number of operands");
  
  if (Pat->getTransformFn())
    I->error("Cannot specify a transform function on a set node!");
  
  // Check the set destinations.
  unsigned NumValues = Pat->getNumChildren()/2;
  for (unsigned i = 0; i != NumValues; ++i) {
    TreePatternNode *Dest = Pat->getChild(i);
    if (!Dest->isLeaf())
      I->error("set destination should be a register!");
    
    DefInit *Val = dynamic_cast<DefInit*>(Dest->getLeafValue());
    if (!Val)
      I->error("set destination should be a register!");
    
    if (!Val->getDef()->isSubClassOf("RegisterClass") &&
        !Val->getDef()->isSubClassOf("Register"))
      I->error("set destination should be a register!");
    if (Dest->getName().empty())
      I->error("set destination must have a name!");
    if (InstResults.count(Dest->getName()))
      I->error("cannot set '" + Dest->getName() +"' multiple times");
    InstResults[Dest->getName()] = Val->getDef();

    // Verify and collect info from the computation.
    FindPatternInputsAndOutputs(I, Pat->getChild(i+NumValues),
                                InstInputs, InstResults);
  }
}

/// NodeHasChain - return true if TreePatternNode has the property
/// 'hasChain', meaning it reads a ctrl-flow chain operand and writes
/// a chain result.
static bool NodeHasChain(TreePatternNode *N, DAGISelEmitter &ISE)
{
  if (N->isLeaf()) return false;
  Record *Operator = N->getOperator();
  if (!Operator->isSubClassOf("SDNode")) return false;

  const SDNodeInfo &NodeInfo = ISE.getSDNodeInfo(Operator);
  return NodeInfo.hasProperty(SDNodeInfo::SDNPHasChain);
}

static bool PatternHasCtrlDep(TreePatternNode *N, DAGISelEmitter &ISE)
{
  if (NodeHasChain(N, ISE))
    return true;
  else {
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
      TreePatternNode *Child = N->getChild(i);
      if (PatternHasCtrlDep(Child, ISE))
        return true;
    }
  }

  return false;
}


/// ParseInstructions - Parse all of the instructions, inlining and resolving
/// any fragments involved.  This populates the Instructions list with fully
/// resolved instructions.
void DAGISelEmitter::ParseInstructions() {
  std::vector<Record*> Instrs = Records.getAllDerivedDefinitions("Instruction");
  
  for (unsigned i = 0, e = Instrs.size(); i != e; ++i) {
    ListInit *LI = 0;
    
    if (dynamic_cast<ListInit*>(Instrs[i]->getValueInit("Pattern")))
      LI = Instrs[i]->getValueAsListInit("Pattern");
    
    // If there is no pattern, only collect minimal information about the
    // instruction for its operand list.  We have to assume that there is one
    // result, as we have no detailed info.
    if (!LI || LI->getSize() == 0) {
      std::vector<Record*> Results;
      std::vector<Record*> Operands;
      
      CodeGenInstruction &InstInfo =Target.getInstruction(Instrs[i]->getName());
      
      // Doesn't even define a result?
      if (InstInfo.OperandList.size() == 0)
        continue;

      // FIXME: temporary hack...
      if (InstInfo.isReturn || InstInfo.isBranch || InstInfo.isCall ||
          InstInfo.isStore) {
        // These produce no results
        for (unsigned j = 0, e = InstInfo.OperandList.size(); j != e; ++j)
          Operands.push_back(InstInfo.OperandList[j].Rec);
      } else {
        // Assume the first operand is the result.
        Results.push_back(InstInfo.OperandList[0].Rec);
      
        // The rest are inputs.
        for (unsigned j = 1, e = InstInfo.OperandList.size(); j != e; ++j)
          Operands.push_back(InstInfo.OperandList[j].Rec);
      }
      
      // Create and insert the instruction.
      Instructions.insert(std::make_pair(Instrs[i], 
                            DAGInstruction(0, Results, Operands)));
      continue;  // no pattern.
    }
    
    // Parse the instruction.
    TreePattern *I = new TreePattern(Instrs[i], LI, true, *this);
    // Inline pattern fragments into it.
    I->InlinePatternFragments();
    
    // Infer as many types as possible.  If we cannot infer all of them, we can
    // never do anything with this instruction pattern: report it to the user.
    if (!I->InferAllTypes())
      I->error("Could not infer all types in pattern!");
    
    // InstInputs - Keep track of all of the inputs of the instruction, along 
    // with the record they are declared as.
    std::map<std::string, TreePatternNode*> InstInputs;
    
    // InstResults - Keep track of all the virtual registers that are 'set'
    // in the instruction, including what reg class they are.
    std::map<std::string, Record*> InstResults;
    
    // Verify that the top-level forms in the instruction are of void type, and
    // fill in the InstResults map.
    for (unsigned j = 0, e = I->getNumTrees(); j != e; ++j) {
      TreePatternNode *Pat = I->getTree(j);
      if (Pat->getExtType() != MVT::isVoid)
        I->error("Top-level forms in instruction pattern should have"
                 " void types");

      // Find inputs and outputs, and verify the structure of the uses/defs.
      FindPatternInputsAndOutputs(I, Pat, InstInputs, InstResults);
    }

    // Now that we have inputs and outputs of the pattern, inspect the operands
    // list for the instruction.  This determines the order that operands are
    // added to the machine instruction the node corresponds to.
    unsigned NumResults = InstResults.size();

    // Parse the operands list from the (ops) list, validating it.
    std::vector<std::string> &Args = I->getArgList();
    assert(Args.empty() && "Args list should still be empty here!");
    CodeGenInstruction &CGI = Target.getInstruction(Instrs[i]->getName());

    // Check that all of the results occur first in the list.
    std::vector<Record*> Results;
    for (unsigned i = 0; i != NumResults; ++i) {
      if (i == CGI.OperandList.size())
        I->error("'" + InstResults.begin()->first +
                 "' set but does not appear in operand list!");
      const std::string &OpName = CGI.OperandList[i].Name;
      
      // Check that it exists in InstResults.
      Record *R = InstResults[OpName];
      if (R == 0)
        I->error("Operand $" + OpName + " should be a set destination: all "
                 "outputs must occur before inputs in operand list!");
      
      if (CGI.OperandList[i].Rec != R)
        I->error("Operand $" + OpName + " class mismatch!");
      
      // Remember the return type.
      Results.push_back(CGI.OperandList[i].Rec);
      
      // Okay, this one checks out.
      InstResults.erase(OpName);
    }

    // Loop over the inputs next.  Make a copy of InstInputs so we can destroy
    // the copy while we're checking the inputs.
    std::map<std::string, TreePatternNode*> InstInputsCheck(InstInputs);

    std::vector<TreePatternNode*> ResultNodeOperands;
    std::vector<Record*> Operands;
    for (unsigned i = NumResults, e = CGI.OperandList.size(); i != e; ++i) {
      const std::string &OpName = CGI.OperandList[i].Name;
      if (OpName.empty())
        I->error("Operand #" + utostr(i) + " in operands list has no name!");

      if (!InstInputsCheck.count(OpName))
        I->error("Operand $" + OpName +
                 " does not appear in the instruction pattern");
      TreePatternNode *InVal = InstInputsCheck[OpName];
      InstInputsCheck.erase(OpName);   // It occurred, remove from map.
      
      if (InVal->isLeaf() &&
          dynamic_cast<DefInit*>(InVal->getLeafValue())) {
        Record *InRec = static_cast<DefInit*>(InVal->getLeafValue())->getDef();
        if (CGI.OperandList[i].Rec != InRec &&
            !InRec->isSubClassOf("ComplexPattern"))
          I->error("Operand $" + OpName +
                   "'s register class disagrees between the operand and pattern");
      }
      Operands.push_back(CGI.OperandList[i].Rec);
      
      // Construct the result for the dest-pattern operand list.
      TreePatternNode *OpNode = InVal->clone();
      
      // No predicate is useful on the result.
      OpNode->setPredicateFn("");
      
      // Promote the xform function to be an explicit node if set.
      if (Record *Xform = OpNode->getTransformFn()) {
        OpNode->setTransformFn(0);
        std::vector<TreePatternNode*> Children;
        Children.push_back(OpNode);
        OpNode = new TreePatternNode(Xform, Children);
      }
      
      ResultNodeOperands.push_back(OpNode);
    }
    
    if (!InstInputsCheck.empty())
      I->error("Input operand $" + InstInputsCheck.begin()->first +
               " occurs in pattern but not in operands list!");

    TreePatternNode *ResultPattern =
      new TreePatternNode(I->getRecord(), ResultNodeOperands);

    // Create and insert the instruction.
    DAGInstruction TheInst(I, Results, Operands);
    Instructions.insert(std::make_pair(I->getRecord(), TheInst));

    // Use a temporary tree pattern to infer all types and make sure that the
    // constructed result is correct.  This depends on the instruction already
    // being inserted into the Instructions map.
    TreePattern Temp(I->getRecord(), ResultPattern, false, *this);
    Temp.InferAllTypes();

    DAGInstruction &TheInsertedInst = Instructions.find(I->getRecord())->second;
    TheInsertedInst.setResultPattern(Temp.getOnlyTree());
    
    DEBUG(I->dump());
  }
   
  // If we can, convert the instructions to be patterns that are matched!
  for (std::map<Record*, DAGInstruction>::iterator II = Instructions.begin(),
       E = Instructions.end(); II != E; ++II) {
    DAGInstruction &TheInst = II->second;
    TreePattern *I = TheInst.getPattern();
    if (I == 0) continue;  // No pattern.

    if (I->getNumTrees() != 1) {
      std::cerr << "CANNOT HANDLE: " << I->getRecord()->getName() << " yet!";
      continue;
    }
    TreePatternNode *Pattern = I->getTree(0);
    TreePatternNode *SrcPattern;
    if (TheInst.getNumResults() == 0) {
      SrcPattern = Pattern;
    } else {
      if (Pattern->getOperator()->getName() != "set")
        continue;  // Not a set (store or something?)
    
      if (Pattern->getNumChildren() != 2)
        continue;  // Not a set of a single value (not handled so far)

      SrcPattern = Pattern->getChild(1)->clone();    
    }
    
    std::string Reason;
    if (!SrcPattern->canPatternMatch(Reason, *this))
      I->error("Instruction can never match: " + Reason);
    
    TreePatternNode *DstPattern = TheInst.getResultPattern();
    PatternsToMatch.push_back(std::make_pair(SrcPattern, DstPattern));

    if (PatternHasCtrlDep(Pattern, *this)) {
      Record *Instr = II->first;
      CodeGenInstruction &InstInfo = Target.getInstruction(Instr->getName());
      InstInfo.hasCtrlDep = true;
    }
  }
}

void DAGISelEmitter::ParsePatterns() {
  std::vector<Record*> Patterns = Records.getAllDerivedDefinitions("Pattern");

  for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
    DagInit *Tree = Patterns[i]->getValueAsDag("PatternToMatch");
    TreePattern *Pattern = new TreePattern(Patterns[i], Tree, true, *this);

    // Inline pattern fragments into it.
    Pattern->InlinePatternFragments();
    
    // Infer as many types as possible.  If we cannot infer all of them, we can
    // never do anything with this pattern: report it to the user.
    if (!Pattern->InferAllTypes())
      Pattern->error("Could not infer all types in pattern!");

    // Validate that the input pattern is correct.
    {
      std::map<std::string, TreePatternNode*> InstInputs;
      std::map<std::string, Record*> InstResults;
      FindPatternInputsAndOutputs(Pattern, Pattern->getOnlyTree(),
                                  InstInputs, InstResults);
    }
    
    ListInit *LI = Patterns[i]->getValueAsListInit("ResultInstrs");
    if (LI->getSize() == 0) continue;  // no pattern.
    
    // Parse the instruction.
    TreePattern *Result = new TreePattern(Patterns[i], LI, false, *this);
    
    // Inline pattern fragments into it.
    Result->InlinePatternFragments();
    
    // Infer as many types as possible.  If we cannot infer all of them, we can
    // never do anything with this pattern: report it to the user.
    if (!Result->InferAllTypes())
      Result->error("Could not infer all types in pattern result!");
   
    if (Result->getNumTrees() != 1)
      Result->error("Cannot handle instructions producing instructions "
                    "with temporaries yet!");

    std::string Reason;
    if (!Pattern->getOnlyTree()->canPatternMatch(Reason, *this))
      Pattern->error("Pattern can never match: " + Reason);
    
    PatternsToMatch.push_back(std::make_pair(Pattern->getOnlyTree(),
                                             Result->getOnlyTree()));
  }
}

/// CombineChildVariants - Given a bunch of permutations of each child of the
/// 'operator' node, put them together in all possible ways.
static void CombineChildVariants(TreePatternNode *Orig, 
               const std::vector<std::vector<TreePatternNode*> > &ChildVariants,
                                 std::vector<TreePatternNode*> &OutVariants,
                                 DAGISelEmitter &ISE) {
  // Make sure that each operand has at least one variant to choose from.
  for (unsigned i = 0, e = ChildVariants.size(); i != e; ++i)
    if (ChildVariants[i].empty())
      return;
        
  // The end result is an all-pairs construction of the resultant pattern.
  std::vector<unsigned> Idxs;
  Idxs.resize(ChildVariants.size());
  bool NotDone = true;
  while (NotDone) {
    // Create the variant and add it to the output list.
    std::vector<TreePatternNode*> NewChildren;
    for (unsigned i = 0, e = ChildVariants.size(); i != e; ++i)
      NewChildren.push_back(ChildVariants[i][Idxs[i]]);
    TreePatternNode *R = new TreePatternNode(Orig->getOperator(), NewChildren);
    
    // Copy over properties.
    R->setName(Orig->getName());
    R->setPredicateFn(Orig->getPredicateFn());
    R->setTransformFn(Orig->getTransformFn());
    R->setType(Orig->getExtType());
    
    // If this pattern cannot every match, do not include it as a variant.
    std::string ErrString;
    if (!R->canPatternMatch(ErrString, ISE)) {
      delete R;
    } else {
      bool AlreadyExists = false;
      
      // Scan to see if this pattern has already been emitted.  We can get
      // duplication due to things like commuting:
      //   (and GPRC:$a, GPRC:$b) -> (and GPRC:$b, GPRC:$a)
      // which are the same pattern.  Ignore the dups.
      for (unsigned i = 0, e = OutVariants.size(); i != e; ++i)
        if (R->isIsomorphicTo(OutVariants[i])) {
          AlreadyExists = true;
          break;
        }
      
      if (AlreadyExists)
        delete R;
      else
        OutVariants.push_back(R);
    }
    
    // Increment indices to the next permutation.
    NotDone = false;
    // Look for something we can increment without causing a wrap-around.
    for (unsigned IdxsIdx = 0; IdxsIdx != Idxs.size(); ++IdxsIdx) {
      if (++Idxs[IdxsIdx] < ChildVariants[IdxsIdx].size()) {
        NotDone = true;   // Found something to increment.
        break;
      }
      Idxs[IdxsIdx] = 0;
    }
  }
}

/// CombineChildVariants - A helper function for binary operators.
///
static void CombineChildVariants(TreePatternNode *Orig, 
                                 const std::vector<TreePatternNode*> &LHS,
                                 const std::vector<TreePatternNode*> &RHS,
                                 std::vector<TreePatternNode*> &OutVariants,
                                 DAGISelEmitter &ISE) {
  std::vector<std::vector<TreePatternNode*> > ChildVariants;
  ChildVariants.push_back(LHS);
  ChildVariants.push_back(RHS);
  CombineChildVariants(Orig, ChildVariants, OutVariants, ISE);
}  


static void GatherChildrenOfAssociativeOpcode(TreePatternNode *N,
                                     std::vector<TreePatternNode *> &Children) {
  assert(N->getNumChildren()==2 &&"Associative but doesn't have 2 children!");
  Record *Operator = N->getOperator();
  
  // Only permit raw nodes.
  if (!N->getName().empty() || !N->getPredicateFn().empty() ||
      N->getTransformFn()) {
    Children.push_back(N);
    return;
  }

  if (N->getChild(0)->isLeaf() || N->getChild(0)->getOperator() != Operator)
    Children.push_back(N->getChild(0));
  else
    GatherChildrenOfAssociativeOpcode(N->getChild(0), Children);

  if (N->getChild(1)->isLeaf() || N->getChild(1)->getOperator() != Operator)
    Children.push_back(N->getChild(1));
  else
    GatherChildrenOfAssociativeOpcode(N->getChild(1), Children);
}

/// GenerateVariantsOf - Given a pattern N, generate all permutations we can of
/// the (potentially recursive) pattern by using algebraic laws.
///
static void GenerateVariantsOf(TreePatternNode *N,
                               std::vector<TreePatternNode*> &OutVariants,
                               DAGISelEmitter &ISE) {
  // We cannot permute leaves.
  if (N->isLeaf()) {
    OutVariants.push_back(N);
    return;
  }

  // Look up interesting info about the node.
  const SDNodeInfo &NodeInfo = ISE.getSDNodeInfo(N->getOperator());

  // If this node is associative, reassociate.
  if (NodeInfo.hasProperty(SDNodeInfo::SDNPAssociative)) {
    // Reassociate by pulling together all of the linked operators 
    std::vector<TreePatternNode*> MaximalChildren;
    GatherChildrenOfAssociativeOpcode(N, MaximalChildren);

    // Only handle child sizes of 3.  Otherwise we'll end up trying too many
    // permutations.
    if (MaximalChildren.size() == 3) {
      // Find the variants of all of our maximal children.
      std::vector<TreePatternNode*> AVariants, BVariants, CVariants;
      GenerateVariantsOf(MaximalChildren[0], AVariants, ISE);
      GenerateVariantsOf(MaximalChildren[1], BVariants, ISE);
      GenerateVariantsOf(MaximalChildren[2], CVariants, ISE);
      
      // There are only two ways we can permute the tree:
      //   (A op B) op C    and    A op (B op C)
      // Within these forms, we can also permute A/B/C.
      
      // Generate legal pair permutations of A/B/C.
      std::vector<TreePatternNode*> ABVariants;
      std::vector<TreePatternNode*> BAVariants;
      std::vector<TreePatternNode*> ACVariants;
      std::vector<TreePatternNode*> CAVariants;
      std::vector<TreePatternNode*> BCVariants;
      std::vector<TreePatternNode*> CBVariants;
      CombineChildVariants(N, AVariants, BVariants, ABVariants, ISE);
      CombineChildVariants(N, BVariants, AVariants, BAVariants, ISE);
      CombineChildVariants(N, AVariants, CVariants, ACVariants, ISE);
      CombineChildVariants(N, CVariants, AVariants, CAVariants, ISE);
      CombineChildVariants(N, BVariants, CVariants, BCVariants, ISE);
      CombineChildVariants(N, CVariants, BVariants, CBVariants, ISE);

      // Combine those into the result: (x op x) op x
      CombineChildVariants(N, ABVariants, CVariants, OutVariants, ISE);
      CombineChildVariants(N, BAVariants, CVariants, OutVariants, ISE);
      CombineChildVariants(N, ACVariants, BVariants, OutVariants, ISE);
      CombineChildVariants(N, CAVariants, BVariants, OutVariants, ISE);
      CombineChildVariants(N, BCVariants, AVariants, OutVariants, ISE);
      CombineChildVariants(N, CBVariants, AVariants, OutVariants, ISE);

      // Combine those into the result: x op (x op x)
      CombineChildVariants(N, CVariants, ABVariants, OutVariants, ISE);
      CombineChildVariants(N, CVariants, BAVariants, OutVariants, ISE);
      CombineChildVariants(N, BVariants, ACVariants, OutVariants, ISE);
      CombineChildVariants(N, BVariants, CAVariants, OutVariants, ISE);
      CombineChildVariants(N, AVariants, BCVariants, OutVariants, ISE);
      CombineChildVariants(N, AVariants, CBVariants, OutVariants, ISE);
      return;
    }
  }
  
  // Compute permutations of all children.
  std::vector<std::vector<TreePatternNode*> > ChildVariants;
  ChildVariants.resize(N->getNumChildren());
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
    GenerateVariantsOf(N->getChild(i), ChildVariants[i], ISE);

  // Build all permutations based on how the children were formed.
  CombineChildVariants(N, ChildVariants, OutVariants, ISE);

  // If this node is commutative, consider the commuted order.
  if (NodeInfo.hasProperty(SDNodeInfo::SDNPCommutative)) {
    assert(N->getNumChildren()==2 &&"Commutative but doesn't have 2 children!");
    // Consider the commuted order.
    CombineChildVariants(N, ChildVariants[1], ChildVariants[0],
                         OutVariants, ISE);
  }
}


// GenerateVariants - Generate variants.  For example, commutative patterns can
// match multiple ways.  Add them to PatternsToMatch as well.
void DAGISelEmitter::GenerateVariants() {
  
  DEBUG(std::cerr << "Generating instruction variants.\n");
  
  // Loop over all of the patterns we've collected, checking to see if we can
  // generate variants of the instruction, through the exploitation of
  // identities.  This permits the target to provide agressive matching without
  // the .td file having to contain tons of variants of instructions.
  //
  // Note that this loop adds new patterns to the PatternsToMatch list, but we
  // intentionally do not reconsider these.  Any variants of added patterns have
  // already been added.
  //
  for (unsigned i = 0, e = PatternsToMatch.size(); i != e; ++i) {
    std::vector<TreePatternNode*> Variants;
    GenerateVariantsOf(PatternsToMatch[i].first, Variants, *this);

    assert(!Variants.empty() && "Must create at least original variant!");
    Variants.erase(Variants.begin());  // Remove the original pattern.

    if (Variants.empty())  // No variants for this pattern.
      continue;

    DEBUG(std::cerr << "FOUND VARIANTS OF: ";
          PatternsToMatch[i].first->dump();
          std::cerr << "\n");

    for (unsigned v = 0, e = Variants.size(); v != e; ++v) {
      TreePatternNode *Variant = Variants[v];

      DEBUG(std::cerr << "  VAR#" << v <<  ": ";
            Variant->dump();
            std::cerr << "\n");
      
      // Scan to see if an instruction or explicit pattern already matches this.
      bool AlreadyExists = false;
      for (unsigned p = 0, e = PatternsToMatch.size(); p != e; ++p) {
        // Check to see if this variant already exists.
        if (Variant->isIsomorphicTo(PatternsToMatch[p].first)) {
          DEBUG(std::cerr << "  *** ALREADY EXISTS, ignoring variant.\n");
          AlreadyExists = true;
          break;
        }
      }
      // If we already have it, ignore the variant.
      if (AlreadyExists) continue;

      // Otherwise, add it to the list of patterns we have.
      PatternsToMatch.push_back(std::make_pair(Variant, 
                                               PatternsToMatch[i].second));
    }

    DEBUG(std::cerr << "\n");
  }
}


// NodeIsComplexPattern - return true if N is a leaf node and a subclass of
// ComplexPattern.
static bool NodeIsComplexPattern(TreePatternNode *N)
{
  return (N->isLeaf() &&
          dynamic_cast<DefInit*>(N->getLeafValue()) &&
          static_cast<DefInit*>(N->getLeafValue())->getDef()->
          isSubClassOf("ComplexPattern"));
}

// NodeGetComplexPattern - return the pointer to the ComplexPattern if N
// is a leaf node and a subclass of ComplexPattern, else it returns NULL.
static const ComplexPattern *NodeGetComplexPattern(TreePatternNode *N,
                                                   DAGISelEmitter &ISE)
{
  if (N->isLeaf() &&
      dynamic_cast<DefInit*>(N->getLeafValue()) &&
      static_cast<DefInit*>(N->getLeafValue())->getDef()->
      isSubClassOf("ComplexPattern")) {
    return &ISE.getComplexPattern(static_cast<DefInit*>(N->getLeafValue())
                                  ->getDef());
  }
  return NULL;
}

/// getPatternSize - Return the 'size' of this pattern.  We want to match large
/// patterns before small ones.  This is used to determine the size of a
/// pattern.
static unsigned getPatternSize(TreePatternNode *P, DAGISelEmitter &ISE) {
  assert(isExtIntegerVT(P->getExtType()) || 
         isExtFloatingPointVT(P->getExtType()) ||
         P->getExtType() == MVT::isVoid && "Not a valid pattern node to size!");
  unsigned Size = 1;  // The node itself.

  // FIXME: This is a hack to statically increase the priority of patterns
  // which maps a sub-dag to a complex pattern. e.g. favors LEA over ADD.
  // Later we can allow complexity / cost for each pattern to be (optionally)
  // specified. To get best possible pattern match we'll need to dynamically
  // calculate the complexity of all patterns a dag can potentially map to.
  const ComplexPattern *AM = NodeGetComplexPattern(P, ISE);
  if (AM)
    Size += AM->getNumOperands();
    
  // Count children in the count if they are also nodes.
  for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = P->getChild(i);
    if (!Child->isLeaf() && Child->getExtType() != MVT::Other)
      Size += getPatternSize(Child, ISE);
    else if (Child->isLeaf()) {
      if (dynamic_cast<IntInit*>(Child->getLeafValue())) 
        ++Size;  // Matches a ConstantSDNode.
      else if (NodeIsComplexPattern(Child))
        Size += getPatternSize(Child, ISE);
    }
  }
  
  return Size;
}

/// getResultPatternCost - Compute the number of instructions for this pattern.
/// This is a temporary hack.  We should really include the instruction
/// latencies in this calculation.
static unsigned getResultPatternCost(TreePatternNode *P) {
  if (P->isLeaf()) return 0;
  
  unsigned Cost = P->getOperator()->isSubClassOf("Instruction");
  for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i)
    Cost += getResultPatternCost(P->getChild(i));
  return Cost;
}

// PatternSortingPredicate - return true if we prefer to match LHS before RHS.
// In particular, we want to match maximal patterns first and lowest cost within
// a particular complexity first.
struct PatternSortingPredicate {
  PatternSortingPredicate(DAGISelEmitter &ise) : ISE(ise) {};
  DAGISelEmitter &ISE;

  bool operator()(DAGISelEmitter::PatternToMatch *LHS,
                  DAGISelEmitter::PatternToMatch *RHS) {
    unsigned LHSSize = getPatternSize(LHS->first, ISE);
    unsigned RHSSize = getPatternSize(RHS->first, ISE);
    if (LHSSize > RHSSize) return true;   // LHS -> bigger -> less cost
    if (LHSSize < RHSSize) return false;
    
    // If the patterns have equal complexity, compare generated instruction cost
    return getResultPatternCost(LHS->second) <getResultPatternCost(RHS->second);
  }
};

/// getRegisterValueType - Look up and return the first ValueType of specified 
/// RegisterClass record
static MVT::ValueType getRegisterValueType(Record *R, const CodeGenTarget &T) {
  if (const CodeGenRegisterClass *RC = T.getRegisterClassForRegister(R))
    return RC->getValueTypeNum(0);
  return MVT::Other;
}


/// RemoveAllTypes - A quick recursive walk over a pattern which removes all
/// type information from it.
static void RemoveAllTypes(TreePatternNode *N) {
  N->setType(MVT::isUnknown);
  if (!N->isLeaf())
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
      RemoveAllTypes(N->getChild(i));
}

Record *DAGISelEmitter::getSDNodeNamed(const std::string &Name) const {
  Record *N = Records.getDef(Name);
  assert(N && N->isSubClassOf("SDNode") && "Bad argument");
  return N;
}

class PatternCodeEmitter {
private:
  DAGISelEmitter &ISE;

  // LHS of the pattern being matched
  TreePatternNode *LHS;
  unsigned PatternNo;
  std::ostream &OS;
  // Node to name mapping
  std::map<std::string,std::string> VariableMap;
  // Names of all the folded nodes which produce chains.
  std::vector<std::string> FoldedChains;
  bool FoundChain;
  bool InFlag;
  unsigned TmpNo;

public:
  PatternCodeEmitter(DAGISelEmitter &ise, TreePatternNode *lhs,
                     unsigned PatNum, std::ostream &os) :
    ISE(ise), LHS(lhs), PatternNo(PatNum), OS(os),
    FoundChain(false), InFlag(false), TmpNo(0) {};

  /// EmitMatchCode - Emit a matcher for N, going to the label for PatternNo
  /// if the match fails. At this point, we already know that the opcode for N
  /// matches, and the SDNode for the result has the RootName specified name.
  void EmitMatchCode(TreePatternNode *N, const std::string &RootName,
                     bool isRoot = false) {
    if (N->isLeaf()) {
      if (IntInit *II = dynamic_cast<IntInit*>(N->getLeafValue())) {
        OS << "      if (cast<ConstantSDNode>(" << RootName
           << ")->getSignExtended() != " << II->getValue() << ")\n"
           << "        goto P" << PatternNo << "Fail;\n";
        return;
      } else if (!NodeIsComplexPattern(N)) {
        assert(0 && "Cannot match this as a leaf value!");
        abort();
      }
    }
  
    // If this node has a name associated with it, capture it in VariableMap.  If
    // we already saw this in the pattern, emit code to verify dagness.
    if (!N->getName().empty()) {
      std::string &VarMapEntry = VariableMap[N->getName()];
      if (VarMapEntry.empty()) {
        VarMapEntry = RootName;
      } else {
        // If we get here, this is a second reference to a specific name.  Since
        // we already have checked that the first reference is valid, we don't
        // have to recursively match it, just check that it's the same as the
        // previously named thing.
        OS << "      if (" << VarMapEntry << " != " << RootName
           << ") goto P" << PatternNo << "Fail;\n";
        return;
      }
    }


    // Emit code to load the child nodes and match their contents recursively.
    unsigned OpNo = 0;
    bool HasChain = NodeHasChain(N, ISE);
    if (HasChain) {
      OpNo = 1;
      if (!isRoot) {
        const SDNodeInfo &CInfo = ISE.getSDNodeInfo(N->getOperator());
        OS << "      if (!" << RootName << ".hasOneUse()) goto P"
           << PatternNo << "Fail;   // Multiple uses of actual result?\n";
        OS << "      if (CodeGenMap.count(" << RootName
           << ".getValue(" << CInfo.getNumResults() << "))) goto P"
           << PatternNo << "Fail;   // Already selected for a chain use?\n";
      }
    }

    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i, ++OpNo) {
      OS << "      SDOperand " << RootName << OpNo <<" = " << RootName
         << ".getOperand(" << OpNo << ");\n";
      TreePatternNode *Child = N->getChild(i);
    
      if (!Child->isLeaf()) {
        // If it's not a leaf, recursively match.
        const SDNodeInfo &CInfo = ISE.getSDNodeInfo(Child->getOperator());
        OS << "      if (" << RootName << OpNo << ".getOpcode() != "
           << CInfo.getEnumName() << ") goto P" << PatternNo << "Fail;\n";
        EmitMatchCode(Child, RootName + utostr(OpNo));
        if (NodeHasChain(Child, ISE))
          FoldedChains.push_back(RootName + utostr(OpNo));
      } else {
        // If this child has a name associated with it, capture it in VarMap.  If
        // we already saw this in the pattern, emit code to verify dagness.
        if (!Child->getName().empty()) {
          std::string &VarMapEntry = VariableMap[Child->getName()];
          if (VarMapEntry.empty()) {
            VarMapEntry = RootName + utostr(OpNo);
          } else {
            // If we get here, this is a second reference to a specific name.  Since
            // we already have checked that the first reference is valid, we don't
            // have to recursively match it, just check that it's the same as the
            // previously named thing.
            OS << "      if (" << VarMapEntry << " != " << RootName << OpNo
               << ") goto P" << PatternNo << "Fail;\n";
            continue;
          }
        }
      
        // Handle leaves of various types.
        if (DefInit *DI = dynamic_cast<DefInit*>(Child->getLeafValue())) {
          Record *LeafRec = DI->getDef();
          if (LeafRec->isSubClassOf("RegisterClass")) {
            // Handle register references.  Nothing to do here.
          } else if (LeafRec->isSubClassOf("Register")) {
            if (!InFlag) {
              OS << "      SDOperand InFlag = SDOperand(0,0);\n";
              InFlag = true;
            }
          } else if (LeafRec->isSubClassOf("ComplexPattern")) {
            // Handle complex pattern. Nothing to do here.
          } else if (LeafRec->isSubClassOf("ValueType")) {
            // Make sure this is the specified value type.
            OS << "      if (cast<VTSDNode>(" << RootName << OpNo << ")->getVT() != "
               << "MVT::" << LeafRec->getName() << ") goto P" << PatternNo
               << "Fail;\n";
          } else if (LeafRec->isSubClassOf("CondCode")) {
            // Make sure this is the specified cond code.
            OS << "      if (cast<CondCodeSDNode>(" << RootName << OpNo
               << ")->get() != " << "ISD::" << LeafRec->getName()
               << ") goto P" << PatternNo << "Fail;\n";
          } else {
            Child->dump();
            assert(0 && "Unknown leaf type!");
          }
        } else if (IntInit *II = dynamic_cast<IntInit*>(Child->getLeafValue())) {
          OS << "      if (!isa<ConstantSDNode>(" << RootName << OpNo << ") ||\n"
             << "          cast<ConstantSDNode>(" << RootName << OpNo
             << ")->getSignExtended() != " << II->getValue() << ")\n"
             << "        goto P" << PatternNo << "Fail;\n";
        } else {
          Child->dump();
          assert(0 && "Unknown leaf type!");
        }
      }
    }

    if (HasChain) {
      if (!FoundChain) {
        OS << "      SDOperand Chain = " << RootName << ".getOperand(0);\n";
        FoundChain = true;
      }
    }

    // If there is a node predicate for this, emit the call.
    if (!N->getPredicateFn().empty())
      OS << "      if (!" << N->getPredicateFn() << "(" << RootName
         << ".Val)) goto P" << PatternNo << "Fail;\n";
  }

  /// EmitResultCode - Emit the action for a pattern.  Now that it has matched
  /// we actually have to build a DAG!
  std::pair<unsigned, unsigned>
  EmitResultCode(TreePatternNode *N, bool isRoot = false) {
    // This is something selected from the pattern we matched.
    if (!N->getName().empty()) {
      assert(!isRoot && "Root of pattern cannot be a leaf!");
      std::string &Val = VariableMap[N->getName()];
      assert(!Val.empty() &&
             "Variable referenced but not defined and not caught earlier!");
      if (Val[0] == 'T' && Val[1] == 'm' && Val[2] == 'p') {
        // Already selected this operand, just return the tmpval.
        return std::make_pair(1, atoi(Val.c_str()+3));
      }

      const ComplexPattern *CP;
      unsigned ResNo = TmpNo++;
      unsigned NumRes = 1;
      if (!N->isLeaf() && N->getOperator()->getName() == "imm") {
        switch (N->getType()) {
          default: assert(0 && "Unknown type for constant node!");
          case MVT::i1:  OS << "      bool Tmp"; break;
          case MVT::i8:  OS << "      unsigned char Tmp"; break;
          case MVT::i16: OS << "      unsigned short Tmp"; break;
          case MVT::i32: OS << "      unsigned Tmp"; break;
          case MVT::i64: OS << "      uint64_t Tmp"; break;
        }
        OS << ResNo << "C = cast<ConstantSDNode>(" << Val << ")->getValue();\n";
        OS << "      SDOperand Tmp" << ResNo << " = CurDAG->getTargetConstant(Tmp"
           << ResNo << "C, MVT::" << getEnumName(N->getType()) << ");\n";
      } else if (!N->isLeaf() && N->getOperator()->getName() == "tglobaladdr") {
        OS << "      SDOperand Tmp" << ResNo << " = " << Val << ";\n";
      } else if (!N->isLeaf() && N->getOperator()->getName() == "tconstpool") {
        OS << "      SDOperand Tmp" << ResNo << " = " << Val << ";\n";
      } else if (N->isLeaf() && (CP = NodeGetComplexPattern(N, ISE))) {
        std::string Fn = CP->getSelectFunc();
        NumRes = CP->getNumOperands();
        OS << "      SDOperand ";
        for (unsigned i = 0; i < NumRes; i++) {
          if (i != 0) OS << ", ";
          OS << "Tmp" << i + ResNo;
        }
        OS << ";\n";
        OS << "      if (!" << Fn << "(" << Val;
        for (unsigned i = 0; i < NumRes; i++)
          OS << " , Tmp" << i + ResNo;
        OS << ")) goto P" << PatternNo << "Fail;\n";
        TmpNo = ResNo + NumRes;
      } else {
        OS << "      SDOperand Tmp" << ResNo << " = Select(" << Val << ");\n";
      }
      // Add Tmp<ResNo> to VariableMap, so that we don't multiply select this
      // value if used multiple times by this pattern result.
      Val = "Tmp"+utostr(ResNo);
      return std::make_pair(NumRes, ResNo);
    }
  
    if (N->isLeaf()) {
      // If this is an explicit register reference, handle it.
      if (DefInit *DI = dynamic_cast<DefInit*>(N->getLeafValue())) {
        unsigned ResNo = TmpNo++;
        if (DI->getDef()->isSubClassOf("Register")) {
          OS << "      SDOperand Tmp" << ResNo << " = CurDAG->getRegister("
             << ISE.getQualifiedName(DI->getDef()) << ", MVT::"
             << getEnumName(N->getType())
             << ");\n";
          return std::make_pair(1, ResNo);
        }
      } else if (IntInit *II = dynamic_cast<IntInit*>(N->getLeafValue())) {
        unsigned ResNo = TmpNo++;
        OS << "      SDOperand Tmp" << ResNo << " = CurDAG->getTargetConstant("
           << II->getValue() << ", MVT::"
           << getEnumName(N->getType())
           << ");\n";
        return std::make_pair(1, ResNo);
      }
    
      N->dump();
      assert(0 && "Unknown leaf type!");
      return std::make_pair(1, ~0U);
    }

    Record *Op = N->getOperator();
    if (Op->isSubClassOf("Instruction")) {
      // Determine operand emission order. Complex pattern first.
      std::vector<std::pair<unsigned, TreePatternNode*> > EmitOrder;
      std::vector<std::pair<unsigned, TreePatternNode*> >::iterator OI;
      for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
        TreePatternNode *Child = N->getChild(i);
        if (i == 0) {
          EmitOrder.push_back(std::make_pair(i, Child));
          OI = EmitOrder.begin();
        } else if (NodeIsComplexPattern(Child)) {
          OI = EmitOrder.insert(OI, std::make_pair(i, Child));
        } else {
          EmitOrder.push_back(std::make_pair(i, Child));
        }
      }

      // Emit all of the operands.
      std::vector<std::pair<unsigned, unsigned> > NumTemps(EmitOrder.size());
      for (unsigned i = 0, e = EmitOrder.size(); i != e; ++i) {
        unsigned OpOrder       = EmitOrder[i].first;
        TreePatternNode *Child = EmitOrder[i].second;
        std::pair<unsigned, unsigned> NumTemp =  EmitResultCode(Child);
        NumTemps[OpOrder] = NumTemp;
      }

      // List all the operands in the right order.
      std::vector<unsigned> Ops;
      for (unsigned i = 0, e = NumTemps.size(); i != e; i++) {
        for (unsigned j = 0; j < NumTemps[i].first; j++)
          Ops.push_back(NumTemps[i].second + j);
      }

      CodeGenInstruction &II =
        ISE.getTargetInfo().getInstruction(Op->getName());

      // Emit all the chain and CopyToReg stuff.
      if (II.hasCtrlDep)
        OS << "      Chain = Select(Chain);\n";
      EmitCopyToRegs(LHS, "N", II.hasCtrlDep);

      const DAGInstruction &Inst = ISE.getInstruction(Op);
      unsigned NumResults = Inst.getNumResults();    
      unsigned ResNo = TmpNo++;
      if (!isRoot) {
        OS << "      SDOperand Tmp" << ResNo << " = CurDAG->getTargetNode("
           << II.Namespace << "::" << II.TheDef->getName() << ", MVT::"
           << getEnumName(N->getType());
        unsigned LastOp = 0;
        for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
          LastOp = Ops[i];
          OS << ", Tmp" << LastOp;
        }
        OS << ");\n";
        if (II.hasCtrlDep) {
          // Must have at least one result
          OS << "      Chain = Tmp" << LastOp << ".getValue("
             << NumResults << ");\n";
        }
      } else if (II.hasCtrlDep) {
        OS << "      SDOperand Result = ";
        OS << "CurDAG->getTargetNode("
           << II.Namespace << "::" << II.TheDef->getName();
        if (NumResults > 0) 
          OS << ", MVT::" << getEnumName(N->getType()); // TODO: multiple results?
        OS << ", MVT::Other";
        for (unsigned i = 0, e = Ops.size(); i != e; ++i)
          OS << ", Tmp" << Ops[i];
        OS << ", Chain";
        if (InFlag)
          OS << ", InFlag";
        OS << ");\n";
        if (NumResults != 0) {
          OS << "      CodeGenMap[N.getValue(0)] = Result;\n";
        }
        OS << "      Chain ";
        if (NodeHasChain(LHS, ISE))
          OS << "= CodeGenMap[N.getValue(" << NumResults << ")] ";
        for (unsigned j = 0, e = FoldedChains.size(); j < e; j++)
          OS << "= CodeGenMap[" << FoldedChains[j] << ".getValue("
             << NumResults << ")] ";
        OS << "= Result.getValue(" << NumResults << ");\n";
        if (NumResults == 0)
          OS << "      return Chain;\n";
        else
          OS << "      return (N.ResNo) ? Chain : Result.getValue(0);\n";
      } else {
        // If this instruction is the root, and if there is only one use of it,
        // use SelectNodeTo instead of getTargetNode to avoid an allocation.
        OS << "      if (N.Val->hasOneUse()) {\n";
        OS << "        return CurDAG->SelectNodeTo(N.Val, "
           << II.Namespace << "::" << II.TheDef->getName() << ", MVT::"
           << getEnumName(N->getType());
        for (unsigned i = 0, e = Ops.size(); i != e; ++i)
          OS << ", Tmp" << Ops[i];
        if (InFlag)
          OS << ", InFlag";
        OS << ");\n";
        OS << "      } else {\n";
        OS << "        return CodeGenMap[N] = CurDAG->getTargetNode("
           << II.Namespace << "::" << II.TheDef->getName() << ", MVT::"
           << getEnumName(N->getType());
        for (unsigned i = 0, e = Ops.size(); i != e; ++i)
          OS << ", Tmp" << Ops[i];
        if (InFlag)
          OS << ", InFlag";
        OS << ");\n";
        OS << "      }\n";
      }
      return std::make_pair(1, ResNo);
    } else if (Op->isSubClassOf("SDNodeXForm")) {
      assert(N->getNumChildren() == 1 && "node xform should have one child!");
      unsigned OpVal = EmitResultCode(N->getChild(0))
        .second;
    
      unsigned ResNo = TmpNo++;
      OS << "      SDOperand Tmp" << ResNo << " = Transform_" << Op->getName()
         << "(Tmp" << OpVal << ".Val);\n";
      if (isRoot) {
        OS << "      CodeGenMap[N] = Tmp" << ResNo << ";\n";
        OS << "      return Tmp" << ResNo << ";\n";
      }
      return std::make_pair(1, ResNo);
    } else {
      N->dump();
      assert(0 && "Unknown node in result pattern!");
      return std::make_pair(1, ~0U);
    }
  }

  /// InsertOneTypeCheck - Insert a type-check for an unresolved type in 'Pat' and
  /// add it to the tree.  'Pat' and 'Other' are isomorphic trees except that 
  /// 'Pat' may be missing types.  If we find an unresolved type to add a check
  /// for, this returns true otherwise false if Pat has all types.
  bool InsertOneTypeCheck(TreePatternNode *Pat, TreePatternNode *Other,
                          const std::string &Prefix) {
    // Did we find one?
    if (!Pat->hasTypeSet()) {
      // Move a type over from 'other' to 'pat'.
      Pat->setType(Other->getType());
      OS << "      if (" << Prefix << ".Val->getValueType(0) != MVT::"
         << getName(Pat->getType()) << ") goto P" << PatternNo << "Fail;\n";
      return true;
    }
  
    unsigned OpNo = (unsigned) NodeHasChain(Pat, ISE);
    for (unsigned i = 0, e = Pat->getNumChildren(); i != e; ++i, ++OpNo)
      if (InsertOneTypeCheck(Pat->getChild(i), Other->getChild(i),
                             Prefix + utostr(OpNo)))
        return true;
    return false;
  }

private:
  /// EmitCopyToRegs - Emit the flag operands for the DAG that is
  /// being built.
  void EmitCopyToRegs(TreePatternNode *N, const std::string &RootName,
                      bool HasCtrlDep) {
    const CodeGenTarget &T = ISE.getTargetInfo();
    unsigned OpNo = (unsigned) NodeHasChain(N, ISE);
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i, ++OpNo) {
      TreePatternNode *Child = N->getChild(i);
      if (!Child->isLeaf()) {
        EmitCopyToRegs(Child, RootName + utostr(OpNo), HasCtrlDep);
      } else {
        if (DefInit *DI = dynamic_cast<DefInit*>(Child->getLeafValue())) {
          Record *RR = DI->getDef();
          if (RR->isSubClassOf("Register")) {
            MVT::ValueType RVT = getRegisterValueType(RR, T);
            if (HasCtrlDep) {
              OS << "      SDOperand " << RootName << "CR" << i << ";\n";
              OS << "      " << RootName << "CR" << i
                 << "  = CurDAG->getCopyToReg(Chain, CurDAG->getRegister("
                 << ISE.getQualifiedName(RR) << ", MVT::"
                 << getEnumName(RVT) << ")"
                 << ", Select(" << RootName << OpNo << "), InFlag);\n";
              OS << "      Chain  = " << RootName << "CR" << i
                 << ".getValue(0);\n";
              OS << "      InFlag = " << RootName << "CR" << i
                 << ".getValue(1);\n";
            } else {
              OS << "      InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode()"
                 << ", CurDAG->getRegister(" << ISE.getQualifiedName(RR)
                 << ", MVT::" << getEnumName(RVT) << ")"
                 << ", Select(" << RootName << OpNo
                 << "), InFlag).getValue(1);\n";
            }
          }
        }
      }
    }
  }
};

/// EmitCodeForPattern - Given a pattern to match, emit code to the specified
/// stream to match the pattern, and generate the code for the match if it
/// succeeds.
void DAGISelEmitter::EmitCodeForPattern(PatternToMatch &Pattern,
                                        std::ostream &OS) {
  static unsigned PatternCount = 0;
  unsigned PatternNo = PatternCount++;
  OS << "    { // Pattern #" << PatternNo << ": ";
  Pattern.first->print(OS);
  OS << "\n      // Emits: ";
  Pattern.second->print(OS);
  OS << "\n";
  OS << "      // Pattern complexity = " << getPatternSize(Pattern.first, *this)
     << "  cost = " << getResultPatternCost(Pattern.second) << "\n";

  PatternCodeEmitter Emitter(*this, Pattern.first, PatternNo, OS);

  // Emit the matcher, capturing named arguments in VariableMap.
  Emitter.EmitMatchCode(Pattern.first, "N", true /*the root*/);

  // TP - Get *SOME* tree pattern, we don't care which.
  TreePattern &TP = *PatternFragments.begin()->second;
  
  // At this point, we know that we structurally match the pattern, but the
  // types of the nodes may not match.  Figure out the fewest number of type 
  // comparisons we need to emit.  For example, if there is only one integer
  // type supported by a target, there should be no type comparisons at all for
  // integer patterns!
  //
  // To figure out the fewest number of type checks needed, clone the pattern,
  // remove the types, then perform type inference on the pattern as a whole.
  // If there are unresolved types, emit an explicit check for those types,
  // apply the type to the tree, then rerun type inference.  Iterate until all
  // types are resolved.
  //
  TreePatternNode *Pat = Pattern.first->clone();
  RemoveAllTypes(Pat);
  
  do {
    // Resolve/propagate as many types as possible.
    try {
      bool MadeChange = true;
      while (MadeChange)
        MadeChange = Pat->ApplyTypeConstraints(TP,true/*Ignore reg constraints*/);
    } catch (...) {
      assert(0 && "Error: could not find consistent types for something we"
             " already decided was ok!");
      abort();
    }

    // Insert a check for an unresolved type and add it to the tree.  If we find
    // an unresolved type to add a check for, this returns true and we iterate,
    // otherwise we are done.
  } while (Emitter.InsertOneTypeCheck(Pat, Pattern.first, "N"));

  Emitter.EmitResultCode(Pattern.second, true /*the root*/);

  delete Pat;
  
  OS << "    }\n  P" << PatternNo << "Fail:\n";
}


namespace {
  /// CompareByRecordName - An ordering predicate that implements less-than by
  /// comparing the names records.
  struct CompareByRecordName {
    bool operator()(const Record *LHS, const Record *RHS) const {
      // Sort by name first.
      if (LHS->getName() < RHS->getName()) return true;
      // If both names are equal, sort by pointer.
      return LHS->getName() == RHS->getName() && LHS < RHS;
    }
  };
}

void DAGISelEmitter::EmitInstructionSelector(std::ostream &OS) {
  std::string InstNS = Target.inst_begin()->second.Namespace;
  if (!InstNS.empty()) InstNS += "::";
  
  // Emit boilerplate.
  OS << "// The main instruction selector code.\n"
     << "SDOperand SelectCode(SDOperand N) {\n"
     << "  if (N.getOpcode() >= ISD::BUILTIN_OP_END &&\n"
     << "      N.getOpcode() < (ISD::BUILTIN_OP_END+" << InstNS
     << "INSTRUCTION_LIST_END))\n"
     << "    return N;   // Already selected.\n\n"
  << "  std::map<SDOperand, SDOperand>::iterator CGMI = CodeGenMap.find(N);\n"
     << "  if (CGMI != CodeGenMap.end()) return CGMI->second;\n"
     << "  switch (N.getOpcode()) {\n"
     << "  default: break;\n"
     << "  case ISD::EntryToken:       // These leaves remain the same.\n"
     << "    return N;\n"
     << "  case ISD::AssertSext:\n"
     << "  case ISD::AssertZext: {\n"
     << "    SDOperand Tmp0 = Select(N.getOperand(0));\n"
     << "    if (!N.Val->hasOneUse()) CodeGenMap[N] = Tmp0;\n"
     << "    return Tmp0;\n"
     << "  }\n"
     << "  case ISD::TokenFactor:\n"
     << "    if (N.getNumOperands() == 2) {\n"
     << "      SDOperand Op0 = Select(N.getOperand(0));\n"
     << "      SDOperand Op1 = Select(N.getOperand(1));\n"
     << "      return CodeGenMap[N] =\n"
     << "          CurDAG->getNode(ISD::TokenFactor, MVT::Other, Op0, Op1);\n"
     << "    } else {\n"
     << "      std::vector<SDOperand> Ops;\n"
     << "      for (unsigned i = 0, e = N.getNumOperands(); i != e; ++i)\n"
     << "        Ops.push_back(Select(N.getOperand(i)));\n"
     << "       return CodeGenMap[N] = \n"
     << "               CurDAG->getNode(ISD::TokenFactor, MVT::Other, Ops);\n"
     << "    }\n"
     << "  case ISD::CopyFromReg: {\n"
     << "    SDOperand Chain = Select(N.getOperand(0));\n"
     << "    if (Chain == N.getOperand(0)) return N; // No change\n"
     << "    SDOperand New = CurDAG->getCopyFromReg(Chain,\n"
     << "                    cast<RegisterSDNode>(N.getOperand(1))->getReg(),\n"
     << "                                         N.Val->getValueType(0));\n"
     << "    return New.getValue(N.ResNo);\n"
     << "  }\n"
     << "  case ISD::CopyToReg: {\n"
     << "    SDOperand Chain = Select(N.getOperand(0));\n"
     << "    SDOperand Reg = N.getOperand(1);\n"
     << "    SDOperand Val = Select(N.getOperand(2));\n"
     << "    return CodeGenMap[N] = \n"
     << "                   CurDAG->getNode(ISD::CopyToReg, MVT::Other,\n"
     << "                                   Chain, Reg, Val);\n"
     << "  }\n";
    
  // Group the patterns by their top-level opcodes.
  std::map<Record*, std::vector<PatternToMatch*>,
           CompareByRecordName> PatternsByOpcode;
  for (unsigned i = 0, e = PatternsToMatch.size(); i != e; ++i) {
    TreePatternNode *Node = PatternsToMatch[i].first;
    if (!Node->isLeaf()) {
      PatternsByOpcode[Node->getOperator()].push_back(&PatternsToMatch[i]);
    } else {
      const ComplexPattern *CP;
      if (IntInit *II = 
             dynamic_cast<IntInit*>(Node->getLeafValue())) {
        PatternsByOpcode[getSDNodeNamed("imm")].push_back(&PatternsToMatch[i]);
      } else if ((CP = NodeGetComplexPattern(Node, *this))) {
        std::vector<Record*> OpNodes = CP->getRootNodes();
        for (unsigned j = 0, e = OpNodes.size(); j != e; j++) {
          PatternsByOpcode[OpNodes[j]].insert(PatternsByOpcode[OpNodes[j]].begin(),
                                              &PatternsToMatch[i]);
        }
      } else {
        std::cerr << "Unrecognized opcode '";
        Node->dump();
        std::cerr << "' on tree pattern '";
        std::cerr << PatternsToMatch[i].second->getOperator()->getName();
        std::cerr << "'!\n";
        exit(1);
      }
    }
  }
  
  // Loop over all of the case statements.
  for (std::map<Record*, std::vector<PatternToMatch*>,
                CompareByRecordName>::iterator PBOI = PatternsByOpcode.begin(),
       E = PatternsByOpcode.end(); PBOI != E; ++PBOI) {
    const SDNodeInfo &OpcodeInfo = getSDNodeInfo(PBOI->first);
    std::vector<PatternToMatch*> &Patterns = PBOI->second;
    
    OS << "  case " << OpcodeInfo.getEnumName() << ":\n";

    // We want to emit all of the matching code now.  However, we want to emit
    // the matches in order of minimal cost.  Sort the patterns so the least
    // cost one is at the start.
    std::stable_sort(Patterns.begin(), Patterns.end(),
                     PatternSortingPredicate(*this));
    
    for (unsigned i = 0, e = Patterns.size(); i != e; ++i)
      EmitCodeForPattern(*Patterns[i], OS);
    OS << "    break;\n\n";
  }
  

  OS << "  } // end of big switch.\n\n"
     << "  std::cerr << \"Cannot yet select: \";\n"
     << "  N.Val->dump();\n"
     << "  std::cerr << '\\n';\n"
     << "  abort();\n"
     << "}\n";
}

void DAGISelEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("DAG Instruction Selector for the " + Target.getName() +
                       " target", OS);
  
  OS << "// *** NOTE: This file is #included into the middle of the target\n"
     << "// *** instruction selector class.  These functions are really "
     << "methods.\n\n";
  
  OS << "// Instance var to keep track of multiply used nodes that have \n"
     << "// already been selected.\n"
     << "std::map<SDOperand, SDOperand> CodeGenMap;\n";
  
  ParseNodeInfo();
  ParseNodeTransforms(OS);
  ParseComplexPatterns();
  ParsePatternFragments(OS);
  ParseInstructions();
  ParsePatterns();
  
  // Generate variants.  For example, commutative patterns can match
  // multiple ways.  Add them to PatternsToMatch as well.
  GenerateVariants();

  
  DEBUG(std::cerr << "\n\nALL PATTERNS TO MATCH:\n\n";
        for (unsigned i = 0, e = PatternsToMatch.size(); i != e; ++i) {
          std::cerr << "PATTERN: ";  PatternsToMatch[i].first->dump();
          std::cerr << "\nRESULT:  ";PatternsToMatch[i].second->dump();
          std::cerr << "\n";
        });
  
  // At this point, we have full information about the 'Patterns' we need to
  // parse, both implicitly from instructions as well as from explicit pattern
  // definitions.  Emit the resultant instruction selector.
  EmitInstructionSelector(OS);  
  
  for (std::map<Record*, TreePattern*>::iterator I = PatternFragments.begin(),
       E = PatternFragments.end(); I != E; ++I)
    delete I->second;
  PatternFragments.clear();

  Instructions.clear();
}
