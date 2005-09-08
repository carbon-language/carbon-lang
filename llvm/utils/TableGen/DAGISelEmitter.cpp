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
// SDNodeInfo implementation
//
SDNodeInfo::SDNodeInfo(Record *R) : Def(R) {
  EnumName    = R->getValueAsString("Opcode");
  SDClassName = R->getValueAsString("SDClass");
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

  TreePatternNode *FragTree = Frag->getTrees()[0]->clone();

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
    //return ISE.ReadNonterminal(R)->getTree()->getType();
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
    
    // Apply the type cast...
    assert(0 && "unimp yet");
    //New->updateNodeType(getValueType(Operator), TheRecord->getName());
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
      assert(!P->getTrees()[0]->isLeaf() && "Can't be a leaf!");
      std::string ClassName =
          P->getTrees()[0]->getOperator()->getValueAsString("SDClass");
      const char *C2 = ClassName == "SDNode" ? "N" : "inN";
      
      OS << "static inline bool Predicate_" << Fragments[i]->getName()
         << "(SDNode *" << C2 << ") {\n";
      if (ClassName != "SDNode")
        OS << "  " << ClassName << " *N = cast<" << ClassName << ">(inN);\n";
      OS << CI->getValue() << "\n}\n";
      P->getTrees()[0]->setPredicateFn("Predicate_"+Fragments[i]->getName());
    }
  }
  
  OS << "\n\n";

  // Now that we've parsed all of the tree fragments, do a closure on them so
  // that there are not references to PatFrags left inside of them.
  for (std::map<Record*, TreePattern*>::iterator I = PatternFragments.begin(),
       E = PatternFragments.end(); I != E; ++I) {
    I->second->InlinePatternFragments();
    // If debugging, print out the pattern fragment result.
    DEBUG(I->second->dump());
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
    Instructions.push_back(new TreePattern(TreePattern::Instruction, Instrs[i],
                                           Trees, *this));
    // Inline pattern fragments into it.
    Instructions.back()->InlinePatternFragments();
    
    DEBUG(Instructions.back()->dump());
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
