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
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
// SDTypeConstraint implementation
//

SDTypeConstraint::SDTypeConstraint(Record *R) {
  OperandNo = R->getValueAsInt("OperandNum");
  
  if (R->isSubClassOf("SDTCisVT")) {
    ConstraintType = SDTCisVT;
    x.SDTCisVT_Info.VT = getValueType(R->getValueAsDef("VT"));
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
  assert(NumResults == 1 && "We only work with single result nodes so far!");
  
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
  assert(NumResults == 1 && "We only work with single result nodes so far!");
  
  // Check that the number of operands is sane.
  if (NodeInfo.getNumOperands() >= 0) {
    if (N->getNumChildren() != (unsigned)NodeInfo.getNumOperands())
      TP.error(N->getOperator()->getName() + " node requires exactly " +
               itostr(NodeInfo.getNumOperands()) + " operands!");
  }
  
  TreePatternNode *NodeToApply = getOperandNum(OperandNo, N, NumResults);
  
  switch (ConstraintType) {
  default: assert(0 && "Unknown constraint type!");
  case SDTCisVT:
    // Operand must be a particular type.
    return NodeToApply->UpdateNodeType(x.SDTCisVT_Info.VT, TP);
  case SDTCisInt:
    if (NodeToApply->hasTypeSet() && !MVT::isInteger(NodeToApply->getType()))
      NodeToApply->UpdateNodeType(MVT::i1, TP);  // throw an error.

    // FIXME: can tell from the target if there is only one Int type supported.
    return false;
  case SDTCisFP:
    if (NodeToApply->hasTypeSet() &&
        !MVT::isFloatingPoint(NodeToApply->getType()))
      NodeToApply->UpdateNodeType(MVT::f32, TP);  // throw an error.
    // FIXME: can tell from the target if there is only one FP type supported.
    return false;
  case SDTCisSameAs: {
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisSameAs_Info.OtherOperandNum, N, NumResults);
    return NodeToApply->UpdateNodeType(OtherNode->getType(), TP) |
           OtherNode->UpdateNodeType(NodeToApply->getType(), TP);
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
    if (OtherNode->hasTypeSet() &&
        (!MVT::isInteger(OtherNode->getType()) ||
         OtherNode->getType() <= VT))
      OtherNode->UpdateNodeType(MVT::Other, TP);  // Throw an error.
    return false;
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
  
  // Parse the type constraints.
  ListInit *Constraints = TypeProfile->getValueAsListInit("Constraints");
  for (unsigned i = 0, e = Constraints->getSize(); i != e; ++i) {
    assert(dynamic_cast<DefInit*>(Constraints->getElement(i)) &&
           "Constraints list should contain constraint definitions!");
    Record *Constraint = 
      static_cast<DefInit*>(Constraints->getElement(i))->getDef();
    TypeConstraints.push_back(Constraint);
  }
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
bool TreePatternNode::UpdateNodeType(MVT::ValueType VT, TreePattern &TP) {
  if (VT == MVT::LAST_VALUETYPE || getType() == VT) return false;
  if (getType() == MVT::LAST_VALUETYPE) {
    setType(VT);
    return true;
  }
  
  TP.error("Type inference contradiction found in node " + 
           getOperator()->getName() + "!");
  return true; // unreachable
}


void TreePatternNode::print(std::ostream &OS) const {
  if (isLeaf()) {
    OS << *getLeafValue();
  } else {
    OS << "(" << getOperator()->getName();
  }
  
  if (getType() == MVT::Other)
    OS << ":Other";
  else if (getType() == MVT::LAST_VALUETYPE)
    ;//OS << ":?";
  else
    OS << ":" << getType();

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
    OS << "<<" << PredicateFn << ">>";
  if (!getName().empty())
    OS << ":$" << getName();

}
void TreePatternNode::dump() const {
  print(std::cerr);
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
  New->setType(getType());
  New->setPredicateFn(getPredicateFn());
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
  
  // Get a new copy of this fragment to stitch into here.
  //delete this;    // FIXME: implement refcounting!
  return FragTree;
}

/// ApplyTypeConstraints - Apply all of the type constraints relevent to
/// this node and its children in the tree.  This returns true if it makes a
/// change, false otherwise.  If a type contradiction is found, throw an
/// exception.
bool TreePatternNode::ApplyTypeConstraints(TreePattern &TP) {
  if (isLeaf()) return false;
  
  // special handling for set, which isn't really an SDNode.
  if (getOperator()->getName() == "set") {
    assert (getNumChildren() == 2 && "Only handle 2 operand set's for now!");
    bool MadeChange = getChild(0)->ApplyTypeConstraints(TP);
    MadeChange |= getChild(1)->ApplyTypeConstraints(TP);
    
    // Types of operands must match.
    MadeChange |= getChild(0)->UpdateNodeType(getChild(1)->getType(), TP);
    MadeChange |= getChild(1)->UpdateNodeType(getChild(0)->getType(), TP);
    MadeChange |= UpdateNodeType(MVT::isVoid, TP);
    return MadeChange;
  }
  
  const SDNodeInfo &NI = TP.getDAGISelEmitter().getSDNodeInfo(getOperator());
  
  bool MadeChange = NI.ApplyTypeConstraints(this, TP);
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    MadeChange |= getChild(i)->ApplyTypeConstraints(TP);
  return MadeChange;  
}


//===----------------------------------------------------------------------===//
// TreePattern implementation
//

TreePattern::TreePattern(PatternType pty, Record *TheRec,
                         const std::vector<DagInit *> &RawPat,
                         DAGISelEmitter &ise)
  : PTy(pty), TheRecord(TheRec), ISE(ise) {

  for (unsigned i = 0, e = RawPat.size(); i != e; ++i)
    Trees.push_back(ParseTreePattern(RawPat[i]));
  
  // Sanity checks and cleanup.
  switch (PTy) {
  case PatFrag: {
    assert(Trees.size() == 1 && "How can we have more than one pattern here?");
    
    // Validate arguments list, convert it to map, to discard duplicates.
    std::set<std::string> OperandsMap(Args.begin(), Args.end());

    if (OperandsMap.count(""))
      error("Cannot have unnamed 'node' values in pattern fragment!");
      
    // Parse the operands list.
    DagInit *OpsList = TheRec->getValueAsDag("Operands");
    if (OpsList->getNodeType()->getName() != "ops")
      error("Operands list should start with '(ops ... '!");
    
    // Copy over the arguments.       
    Args.clear();
    for (unsigned i = 0, e = OpsList->getNumArgs(); i != e; ++i) {
      if (!dynamic_cast<DefInit*>(OpsList->getArg(i)) ||
          static_cast<DefInit*>(OpsList->getArg(i))->
                          getDef()->getName() != "node")
        error("Operands list should all be 'node' values.");
      if (OpsList->getArgName(i).empty())
        error("Operands list should have names for each operand!");
      if (!OperandsMap.count(OpsList->getArgName(i)))
        error("'" + OpsList->getArgName(i) +
              "' does not occur in pattern or was multiply specified!");
      OperandsMap.erase(OpsList->getArgName(i));
      Args.push_back(OpsList->getArgName(i));
    }
    
    if (!OperandsMap.empty())
      error("Operands list does not contain an entry for operand '" +
            *OperandsMap.begin() + "'!");
    
    break;
  }
  default:
    if (!Args.empty())
      error("Only pattern fragments can have operands (use 'node' values)!");
    break;
  }
}

void TreePattern::error(const std::string &Msg) const {
  std::string M = "In ";
  switch (PTy) {
    case PatFrag:     M += "patfrag "; break;
    case Instruction: M += "instruction "; break;
  }
  throw M + TheRecord->getName() + ": " + Msg;
}

/// getIntrinsicType - Check to see if the specified record has an intrinsic
/// type which should be applied to it.  This infer the type of register
/// references from the register file information, for example.
///
MVT::ValueType TreePattern::getIntrinsicType(Record *R) const {
  // Check to see if this is a register or a register class...
  if (R->isSubClassOf("RegisterClass"))
    return getValueType(R->getValueAsDef("RegType"));
  else if (R->isSubClassOf("PatFrag")) {
    // Pattern fragment types will be resolved when they are inlined.
    return MVT::LAST_VALUETYPE;
  } else if (R->isSubClassOf("Register")) {
    assert(0 && "Explicit registers not handled here yet!\n");
    return MVT::LAST_VALUETYPE;
  } else if (R->isSubClassOf("ValueType")) {
    // Using a VTSDNode.
    return MVT::Other;
  } else if (R->getName() == "node") {
    // Placeholder.
    return MVT::LAST_VALUETYPE;
  }
  
  error("Unknown value used: " + R->getName());
  return MVT::Other;
}

TreePatternNode *TreePattern::ParseTreePattern(DagInit *Dag) {
  Record *Operator = Dag->getNodeType();
  
  if (Operator->isSubClassOf("ValueType")) {
    // If the operator is a ValueType, then this must be "type cast" of a leaf
    // node.
    if (Dag->getNumArgs() != 1)
      error("Type cast only valid for a leaf node!");
    
    Init *Arg = Dag->getArg(0);
    TreePatternNode *New;
    if (DefInit *DI = dynamic_cast<DefInit*>(Arg)) {
      New = new TreePatternNode(DI);
      // If it's a regclass or something else known, set the type.
      New->setType(getIntrinsicType(DI->getDef()));
    } else if (DagInit *DI = dynamic_cast<DagInit*>(Arg)) {
      New = ParseTreePattern(DI);
    } else {
      Arg->dump();
      error("Unknown leaf value for tree pattern!");
      return 0;
    }
    
    // Apply the type cast.
    New->UpdateNodeType(getValueType(Operator), *this);
    return New;
  }
  
  // Verify that this is something that makes sense for an operator.
  if (!Operator->isSubClassOf("PatFrag") && !Operator->isSubClassOf("SDNode") &&
      Operator->getName() != "set")
    error("Unrecognized node '" + Operator->getName() + "'!");
  
  std::vector<TreePatternNode*> Children;
  
  for (unsigned i = 0, e = Dag->getNumArgs(); i != e; ++i) {
    Init *Arg = Dag->getArg(i);
    if (DagInit *DI = dynamic_cast<DagInit*>(Arg)) {
      Children.push_back(ParseTreePattern(DI));
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
        
        // If it's a regclass or something else known, set the type.
        Node->setType(getIntrinsicType(R));
        
        // Input argument?
        if (R->getName() == "node") {
          if (Dag->getArgName(i).empty())
            error("'node' argument requires a name to match with operand list");
          Args.push_back(Dag->getArgName(i));
        }
      }
    } else {
      Arg->dump();
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
      MadeChange |= Trees[i]->ApplyTypeConstraints(*this);
  }
  
  bool HasUnresolvedTypes = false;
  for (unsigned i = 0, e = Trees.size(); i != e; ++i)
    HasUnresolvedTypes |= Trees[i]->ContainsUnresolvedType();
  return !HasUnresolvedTypes;
}

void TreePattern::print(std::ostream &OS) const {
  switch (getPatternType()) {
  case TreePattern::PatFrag:     OS << "PatFrag pattern "; break;
  case TreePattern::Instruction: OS << "Inst pattern "; break;
  }
  
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

/// ParseAndResolvePatternFragments - Parse all of the PatFrag definitions in
/// the .td file, building up the PatternFragments map.  After we've collected
/// them all, inline fragments together as necessary, so that there are no
/// references left inside a pattern fragment to a pattern fragment.
///
/// This also emits all of the predicate functions to the output file.
///
void DAGISelEmitter::ParseAndResolvePatternFragments(std::ostream &OS) {
  std::vector<Record*> Fragments = Records.getAllDerivedDefinitions("PatFrag");
  
  // First step, parse all of the fragments and emit predicate functions.
  OS << "\n// Predicate functions.\n";
  for (unsigned i = 0, e = Fragments.size(); i != e; ++i) {
    std::vector<DagInit*> Trees;
    Trees.push_back(Fragments[i]->getValueAsDag("Fragment"));
    TreePattern *P = new TreePattern(TreePattern::PatFrag, Fragments[i],
                                     Trees, *this);
    PatternFragments[Fragments[i]] = P;

    // If there is a code init for this fragment, emit the predicate code and
    // keep track of the fact that this fragment uses it.
    CodeInit *CI =
      dynamic_cast<CodeInit*>(Fragments[i]->getValueInit("Predicate"));
    if (!CI->getValue().empty()) {
      assert(!P->getOnlyTree()->isLeaf() && "Can't be a leaf!");
      std::string ClassName =
        getSDNodeInfo(P->getOnlyTree()->getOperator()).getSDClassName();
      const char *C2 = ClassName == "SDNode" ? "N" : "inN";
      
      OS << "static inline bool Predicate_" << Fragments[i]->getName()
         << "(SDNode *" << C2 << ") {\n";
      if (ClassName != "SDNode")
        OS << "  " << ClassName << " *N = cast<" << ClassName << ">(inN);\n";
      OS << CI->getValue() << "\n}\n";
      P->getOnlyTree()->setPredicateFn("Predicate_"+Fragments[i]->getName());
    }
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

/// ParseAndResolveInstructions - Parse all of the instructions, inlining and
/// resolving any fragments involved.  This populates the Instructions list with
/// fully resolved instructions.
void DAGISelEmitter::ParseAndResolveInstructions() {
  std::vector<Record*> Instrs = Records.getAllDerivedDefinitions("Instruction");
  
  for (unsigned i = 0, e = Instrs.size(); i != e; ++i) {
    if (!dynamic_cast<ListInit*>(Instrs[i]->getValueInit("Pattern")))
      continue; // no pattern yet, ignore it.
    
    ListInit *LI = Instrs[i]->getValueAsListInit("Pattern");
    if (LI->getSize() == 0) continue;  // no pattern.
    
    std::vector<DagInit*> Trees;
    for (unsigned j = 0, e = LI->getSize(); j != e; ++j)
      Trees.push_back((DagInit*)LI->getElement(j));

    // Parse the instruction.
    TreePattern *I = new TreePattern(TreePattern::Instruction, Instrs[i],
                                     Trees, *this);
    // Inline pattern fragments into it.
    I->InlinePatternFragments();
    
    // Infer as many types as possible.  If we cannot infer all of them, we can
    // never do anything with this instruction pattern: report it to the user.
    if (!I->InferAllTypes()) {
      I->dump();
      I->error("Could not infer all types in pattern!");
    }
    
    // Verify that the top-level forms in the instruction are of void type.
    for (unsigned j = 0, e = I->getNumTrees(); j != e; ++j)
      if (I->getTree(j)->getType() != MVT::isVoid) {
        I->dump();
        I->error("Top-level forms in instruction pattern should have"
                 " void types");
      }

    DEBUG(I->dump());
    Instructions.push_back(I);
  }
}

void DAGISelEmitter::EmitInstructionSelector(std::ostream &OS) {
  // Emit boilerplate.
  OS << "// The main instruction selector code.\n"
     << "SDOperand " << Target.getName()
     << "DAGToDAGISel::SelectCode(SDOperand Op) {\n"
     << "  SDNode *N = Op.Val;\n"
     << "  if (N->getOpcode() >= ISD::BUILTIN_OP_END &&\n"
     << "      N->getOpcode() < PPCISD::FIRST_NUMBER)\n"
     << "    return Op;   // Already selected.\n\n"
     << "  switch (N->getOpcode()) {\n"
     << "  default: break;\n"
     << "  case ISD::EntryToken:       // These leaves remain the same.\n"
     << "    return Op;\n"
     << "  case ISD::AssertSext:\n"
     << "  case ISD::AssertZext:\n"
     << "    return Select(N->getOperand(0));\n";
    

  
  OS << "  } // end of big switch.\n\n"
     << "  std::cerr << \"Cannot yet select: \";\n"
     << "  N->dump();\n"
     << "  std::cerr << '\\n';\n"
     << "  abort();\n"
     << "}\n";
}


void DAGISelEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("DAG Instruction Selector for the " + Target.getName() +
                       " target", OS);
  
  ParseNodeInfo();
  ParseAndResolvePatternFragments(OS);
  ParseAndResolveInstructions();
  
  // TODO: convert some instructions to expanders if needed or something.
  
  EmitInstructionSelector(OS);  
  
  for (std::map<Record*, TreePattern*>::iterator I = PatternFragments.begin(),
       E = PatternFragments.end(); I != E; ++I)
    delete I->second;
  PatternFragments.clear();

  for (unsigned i = 0, e = Instructions.size(); i != e; ++i)
    delete Instructions[i];
  Instructions.clear();
}
