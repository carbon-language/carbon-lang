//===- InstrInfoEmitter.cpp - Generate a Instruction Set Desc. ------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#include "InstrSelectorEmitter.h"
#include "CodeGenWrappers.h"
#include "Record.h"
#include "Support/Debug.h"
#include "Support/StringExtras.h"
#include <set>

NodeType::ArgResultTypes NodeType::Translate(Record *R) {
  const std::string &Name = R->getName();
  if (Name == "DNVT_any")  return Any;
  if (Name == "DNVT_void") return Void;
  if (Name == "DNVT_val" ) return Val;
  if (Name == "DNVT_arg0") return Arg0;
  if (Name == "DNVT_arg1") return Arg1;
  if (Name == "DNVT_ptr" ) return Ptr;
  if (Name == "DNVT_i8"  ) return I8;
  throw "Unknown DagNodeValType '" + Name + "'!";
}


//===----------------------------------------------------------------------===//
// TreePatternNode implementation
//

/// getValueRecord - Returns the value of this tree node as a record.  For now
/// we only allow DefInit's as our leaf values, so this is used.
Record *TreePatternNode::getValueRecord() const {
  DefInit *DI = dynamic_cast<DefInit*>(getValue());
  assert(DI && "Instruction Selector does not yet support non-def leaves!");
  return DI->getDef();
}


// updateNodeType - Set the node type of N to VT if VT contains information.  If
// N already contains a conflicting type, then throw an exception
//
bool TreePatternNode::updateNodeType(MVT::ValueType VT,
                                     const std::string &RecName) {
  if (VT == MVT::Other || getType() == VT) return false;
  if (getType() == MVT::Other) {
    setType(VT);
    return true;
  }

  throw "Type inferfence contradiction found for pattern " + RecName;
}

/// InstantiateNonterminals - If this pattern refers to any nonterminals which
/// are not themselves completely resolved, clone the nonterminal and resolve it
/// with the using context we provide.
///
void TreePatternNode::InstantiateNonterminals(InstrSelectorEmitter &ISE) {
  if (!isLeaf()) {
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      getChild(i)->InstantiateNonterminals(ISE);
    return;
  }
  
  // If this is a leaf, it might be a reference to a nonterminal!  Check now.
  Record *R = getValueRecord();
  if (R->isSubClassOf("Nonterminal")) {
    Pattern *NT = ISE.getPattern(R);
    if (!NT->isResolved()) {
      // We found an unresolved nonterminal reference.  Ask the ISE to clone
      // it for us, then update our reference to the fresh, new, resolved,
      // nonterminal.
      
      Value = new DefInit(ISE.InstantiateNonterminal(NT, getType()));
    }
  }
}


/// clone - Make a copy of this tree and all of its children.
///
TreePatternNode *TreePatternNode::clone() const {
  TreePatternNode *New;
  if (isLeaf()) {
    New = new TreePatternNode(Value);
  } else {
    std::vector<std::pair<TreePatternNode*, std::string> > CChildren;
    CChildren.reserve(Children.size());
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      CChildren.push_back(std::make_pair(getChild(i)->clone(),getChildName(i)));
    New = new TreePatternNode(Operator, CChildren);
  }
  New->setType(Type);
  return New;
}

std::ostream &operator<<(std::ostream &OS, const TreePatternNode &N) {
  if (N.isLeaf())
    return OS << N.getType() << ":" << *N.getValue();
  OS << "(" << N.getType() << ":";
  OS << N.getOperator()->getName();
  
  if (N.getNumChildren() != 0) {
    OS << " " << *N.getChild(0);
    for (unsigned i = 1, e = N.getNumChildren(); i != e; ++i)
      OS << ", " << *N.getChild(i);
  }  
  return OS << ")";
}

void TreePatternNode::dump() const { std::cerr << *this; }

//===----------------------------------------------------------------------===//
// Pattern implementation
//

// Parse the specified DagInit into a TreePattern which we can use.
//
Pattern::Pattern(PatternType pty, DagInit *RawPat, Record *TheRec,
                 InstrSelectorEmitter &ise)
  : PTy(pty), ResultNode(0), TheRecord(TheRec), ISE(ise) {

  // First, parse the pattern...
  Tree = ParseTreePattern(RawPat);

  // Run the type-inference engine...
  InferAllTypes();

  if (PTy == Instruction || PTy == Expander) {
    // Check to make sure there is not any unset types in the tree pattern...
    if (!isResolved()) {
      std::cerr << "In pattern: " << *Tree << "\n";
      error("Could not infer all types!");
    }

    // Check to see if we have a top-level (set) of a register.
    if (Tree->getOperator()->getName() == "set") {
      assert(Tree->getNumChildren() == 2 && "Set with != 2 arguments?");
      if (!Tree->getChild(0)->isLeaf())
        error("Arg #0 of set should be a register or register class!");
      ResultNode = Tree->getChild(0);
      ResultName = Tree->getChildName(0);
      Tree = Tree->getChild(1);
    }
  }

  calculateArgs(Tree, "");
}

void Pattern::error(const std::string &Msg) const {
  std::string M = "In ";
  switch (PTy) {
  case Nonterminal: M += "nonterminal "; break;
  case Instruction: M += "instruction "; break;
  case Expander   : M += "expander "; break;
  }
  throw M + TheRecord->getName() + ": " + Msg;  
}

/// calculateArgs - Compute the list of all of the arguments to this pattern,
/// which are the non-void leaf nodes in this pattern.
///
void Pattern::calculateArgs(TreePatternNode *N, const std::string &Name) {
  if (N->isLeaf() || N->getNumChildren() == 0) {
    if (N->getType() != MVT::isVoid)
      Args.push_back(std::make_pair(N, Name));
  } else {
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
      calculateArgs(N->getChild(i), N->getChildName(i));
  }
}

/// getIntrinsicType - Check to see if the specified record has an intrinsic
/// type which should be applied to it.  This infer the type of register
/// references from the register file information, for example.
///
MVT::ValueType Pattern::getIntrinsicType(Record *R) const {
  // Check to see if this is a register or a register class...
  if (R->isSubClassOf("RegisterClass"))
    return getValueType(R->getValueAsDef("RegType"));
  else if (R->isSubClassOf("Nonterminal"))
    return ISE.ReadNonterminal(R)->getTree()->getType();
  else if (R->isSubClassOf("Register")) {
    std::cerr << "WARNING: Explicit registers not handled yet!\n";
    return MVT::Other;
  }

  error("Unknown value used: " + R->getName());
  return MVT::Other;
}

TreePatternNode *Pattern::ParseTreePattern(DagInit *Dag) {
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
    }

    // Apply the type cast...
    New->updateNodeType(getValueType(Operator), TheRecord->getName());
    return New;
  }

  if (!ISE.getNodeTypes().count(Operator))
    error("Unrecognized node '" + Operator->getName() + "'!");

  std::vector<std::pair<TreePatternNode*, std::string> > Children;
  
  for (unsigned i = 0, e = Dag->getNumArgs(); i != e; ++i) {
    Init *Arg = Dag->getArg(i);
    if (DagInit *DI = dynamic_cast<DagInit*>(Arg)) {
      Children.push_back(std::make_pair(ParseTreePattern(DI),
                                        Dag->getArgName(i)));
    } else if (DefInit *DefI = dynamic_cast<DefInit*>(Arg)) {
      Record *R = DefI->getDef();
      // Direct reference to a leaf DagNode?  Turn it into a DagNode if its own.
      if (R->isSubClassOf("DagNode")) {
        Dag->setArg(i, new DagInit(R,
                                std::vector<std::pair<Init*, std::string> >()));
        --i;  // Revisit this node...
      } else {
        Children.push_back(std::make_pair(new TreePatternNode(DefI),
                                          Dag->getArgName(i)));
        // If it's a regclass or something else known, set the type.
        Children.back().first->setType(getIntrinsicType(R));
      }
    } else {
      Arg->dump();
      error("Unknown leaf value for tree pattern!");
    }
  }

  return new TreePatternNode(Operator, Children);
}

void Pattern::InferAllTypes() {
  bool MadeChange, AnyUnset;
  do {
    MadeChange = false;
    AnyUnset = InferTypes(Tree, MadeChange);
  } while ((AnyUnset || MadeChange) && !(AnyUnset && !MadeChange));
  Resolved = !AnyUnset;
}


// InferTypes - Perform type inference on the tree, returning true if there
// are any remaining untyped nodes and setting MadeChange if any changes were
// made.
bool Pattern::InferTypes(TreePatternNode *N, bool &MadeChange) {
  if (N->isLeaf()) return N->getType() == MVT::Other;

  bool AnyUnset = false;
  Record *Operator = N->getOperator();
  const NodeType &NT = ISE.getNodeType(Operator);

  // Check to see if we can infer anything about the argument types from the
  // return types...
  if (N->getNumChildren() != NT.ArgTypes.size())
    error("Incorrect number of children for " + Operator->getName() + " node!");

  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = N->getChild(i);
    AnyUnset |= InferTypes(Child, MadeChange);

    switch (NT.ArgTypes[i]) {
    case NodeType::Any: break;
    case NodeType::I8:
      MadeChange |= Child->updateNodeType(MVT::i1, TheRecord->getName());
      break;
    case NodeType::Arg0:
      MadeChange |= Child->updateNodeType(N->getChild(0)->getType(),
                                          TheRecord->getName());
      break;
    case NodeType::Arg1:
      MadeChange |= Child->updateNodeType(N->getChild(1)->getType(),
                                          TheRecord->getName());
      break;
    case NodeType::Val:
      if (Child->getType() == MVT::isVoid)
        error("Inferred a void node in an illegal place!");
      break;
    case NodeType::Ptr:
      MadeChange |= Child->updateNodeType(ISE.getTarget().getPointerType(),
                                          TheRecord->getName());
      break;
    case NodeType::Void:
      MadeChange |= Child->updateNodeType(MVT::isVoid, TheRecord->getName());
      break;
    default: assert(0 && "Invalid argument ArgType!");
    }
  }

  // See if we can infer anything about the return type now...
  switch (NT.ResultType) {
  case NodeType::Any: break;
  case NodeType::Void:
    MadeChange |= N->updateNodeType(MVT::isVoid, TheRecord->getName());
    break;
  case NodeType::I8:
    MadeChange |= N->updateNodeType(MVT::i1, TheRecord->getName());
    break;
  case NodeType::Arg0:
    MadeChange |= N->updateNodeType(N->getChild(0)->getType(),
                                    TheRecord->getName());
    break;
  case NodeType::Arg1:
    MadeChange |= N->updateNodeType(N->getChild(1)->getType(),
                                    TheRecord->getName());
    break;
  case NodeType::Ptr:
    MadeChange |= N->updateNodeType(ISE.getTarget().getPointerType(),
                                    TheRecord->getName());
    break;
  case NodeType::Val:
    if (N->getType() == MVT::isVoid)
      error("Inferred a void node in an illegal place!");
    break;
  default:
    assert(0 && "Unhandled type constraint!");
    break;
  }

  return AnyUnset | N->getType() == MVT::Other;
}

/// clone - This method is used to make an exact copy of the current pattern,
/// then change the "TheRecord" instance variable to the specified record.
///
Pattern *Pattern::clone(Record *R) const {
  assert(PTy == Nonterminal && "Can only clone nonterminals");
  return new Pattern(Tree->clone(), R, Resolved, ISE);
}



std::ostream &operator<<(std::ostream &OS, const Pattern &P) {
  switch (P.getPatternType()) {
  case Pattern::Nonterminal: OS << "Nonterminal pattern "; break;
  case Pattern::Instruction: OS << "Instruction pattern "; break;
  case Pattern::Expander:    OS << "Expander pattern    "; break;
  }

  OS << P.getRecord()->getName() << ":\t";

  if (Record *Result = P.getResult())
    OS << Result->getName() << " = ";
  OS << *P.getTree();

  if (!P.isResolved())
    OS << " [not completely resolved]";
  return OS;
}

void Pattern::dump() const { std::cerr << *this; }



/// getSlotName - If this is a leaf node, return the slot name that the operand
/// will update.
std::string Pattern::getSlotName() const {
  if (getPatternType() == Pattern::Nonterminal) {
    // Just use the nonterminal name, which will already include the type if
    // it has been cloned.
    return getRecord()->getName();
  } else {
    std::string SlotName;
    if (getResult())
      SlotName = getResult()->getName()+"_";
    else
      SlotName = "Void_";
    return SlotName + getName(getTree()->getType());
  }
}

/// getSlotName - If this is a leaf node, return the slot name that the
/// operand will update.
std::string Pattern::getSlotName(Record *R) {
  if (R->isSubClassOf("Nonterminal")) {
    // Just use the nonterminal name, which will already include the type if
    // it has been cloned.
    return R->getName();
  } else if (R->isSubClassOf("RegisterClass")) {
    MVT::ValueType Ty = getValueType(R->getValueAsDef("RegType"));
    return R->getName() + "_" + getName(Ty);
  } else {
    assert(0 && "Don't know how to get a slot name for this!");
  }
}

//===----------------------------------------------------------------------===//
// PatternOrganizer implementation
//

/// addPattern - Add the specified pattern to the appropriate location in the
/// collection.
void PatternOrganizer::addPattern(Pattern *P) {
  NodesForSlot &Nodes = AllPatterns[P->getSlotName()];
  if (!P->getTree()->isLeaf())
    Nodes[P->getTree()->getOperator()].push_back(P);
  else {
    // Right now we only support DefInit's with node types...
    Nodes[P->getTree()->getValueRecord()].push_back(P);
  }
}



//===----------------------------------------------------------------------===//
// InstrSelectorEmitter implementation
//

/// ReadNodeTypes - Read in all of the node types in the current RecordKeeper,
/// turning them into the more accessible NodeTypes data structure.
///
void InstrSelectorEmitter::ReadNodeTypes() {
  std::vector<Record*> Nodes = Records.getAllDerivedDefinitions("DagNode");
  DEBUG(std::cerr << "Getting node types: ");
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
    Record *Node = Nodes[i];
    
    // Translate the return type...
    NodeType::ArgResultTypes RetTy =
      NodeType::Translate(Node->getValueAsDef("RetType"));

    // Translate the arguments...
    ListInit *Args = Node->getValueAsListInit("ArgTypes");
    std::vector<NodeType::ArgResultTypes> ArgTypes;

    for (unsigned a = 0, e = Args->getSize(); a != e; ++a) {
      if (DefInit *DI = dynamic_cast<DefInit*>(Args->getElement(a)))
        ArgTypes.push_back(NodeType::Translate(DI->getDef()));
      else
        throw "In node " + Node->getName() + ", argument is not a Def!";

      if (a == 0 && ArgTypes.back() == NodeType::Arg0)
        throw "In node " + Node->getName() + ", arg 0 cannot have type 'arg0'!";
      if (a == 1 && ArgTypes.back() == NodeType::Arg1)
        throw "In node " + Node->getName() + ", arg 1 cannot have type 'arg1'!";
    }
    if ((RetTy == NodeType::Arg0 && Args->getSize() == 0) ||
        (RetTy == NodeType::Arg1 && Args->getSize() < 2))
      throw "In node " + Node->getName() +
            ", invalid return type for node with this many operands!";

    // Add the node type mapping now...
    NodeTypes[Node] = NodeType(RetTy, ArgTypes);
    DEBUG(std::cerr << Node->getName() << ", ");
  }
  DEBUG(std::cerr << "DONE!\n");
}

Pattern *InstrSelectorEmitter::ReadNonterminal(Record *R) {
  Pattern *&P = Patterns[R];
  if (P) return P;  // Don't reread it!

  DagInit *DI = R->getValueAsDag("Pattern");
  P = new Pattern(Pattern::Nonterminal, DI, R, *this);
  DEBUG(std::cerr << "Parsed " << *P << "\n");
  return P;
}


// ReadNonTerminals - Read in all nonterminals and incorporate them into our
// pattern database.
void InstrSelectorEmitter::ReadNonterminals() {
  std::vector<Record*> NTs = Records.getAllDerivedDefinitions("Nonterminal");
  for (unsigned i = 0, e = NTs.size(); i != e; ++i)
    ReadNonterminal(NTs[i]);
}


/// ReadInstructionPatterns - Read in all subclasses of Instruction, and process
/// those with a useful Pattern field.
///
void InstrSelectorEmitter::ReadInstructionPatterns() {
  std::vector<Record*> Insts = Records.getAllDerivedDefinitions("Instruction");
  for (unsigned i = 0, e = Insts.size(); i != e; ++i) {
    Record *Inst = Insts[i];
    if (DagInit *DI = dynamic_cast<DagInit*>(Inst->getValueInit("Pattern"))) {
      Patterns[Inst] = new Pattern(Pattern::Instruction, DI, Inst, *this);
      DEBUG(std::cerr << "Parsed " << *Patterns[Inst] << "\n");
    }
  }
}

/// ReadExpanderPatterns - Read in all expander patterns...
///
void InstrSelectorEmitter::ReadExpanderPatterns() {
  std::vector<Record*> Expanders = Records.getAllDerivedDefinitions("Expander");
  for (unsigned i = 0, e = Expanders.size(); i != e; ++i) {
    Record *Expander = Expanders[i];
    DagInit *DI = Expander->getValueAsDag("Pattern");
    Patterns[Expander] = new Pattern(Pattern::Expander, DI, Expander, *this);
    DEBUG(std::cerr << "Parsed " << *Patterns[Expander] << "\n");
  }
}


// InstantiateNonterminals - Instantiate any unresolved nonterminals with
// information from the context that they are used in.
//
void InstrSelectorEmitter::InstantiateNonterminals() {
  DEBUG(std::cerr << "Instantiating nonterminals:\n");
  for (std::map<Record*, Pattern*>::iterator I = Patterns.begin(),
         E = Patterns.end(); I != E; ++I)
    if (I->second->isResolved())
      I->second->InstantiateNonterminals();
}

/// InstantiateNonterminal - This method takes the nonterminal specified by
/// NT, which should not be completely resolved, clones it, applies ResultTy
/// to its root, then runs the type inference stuff on it.  This should
/// produce a newly resolved nonterminal, which we make a record for and
/// return.  To be extra fancy and efficient, this only makes one clone for
/// each type it is instantiated with.
Record *InstrSelectorEmitter::InstantiateNonterminal(Pattern *NT,
                                                     MVT::ValueType ResultTy) {
  assert(!NT->isResolved() && "Nonterminal is already resolved!");

  // Check to see if we have already instantiated this pair...
  Record* &Slot = InstantiatedNTs[std::make_pair(NT, ResultTy)];
  if (Slot) return Slot;
  
  Record *New = new Record(NT->getRecord()->getName()+"_"+getName(ResultTy));

  // Copy over the superclasses...
  const std::vector<Record*> &SCs = NT->getRecord()->getSuperClasses();
  for (unsigned i = 0, e = SCs.size(); i != e; ++i)
    New->addSuperClass(SCs[i]);

  DEBUG(std::cerr << "  Nonterminal '" << NT->getRecord()->getName()
                  << "' for type '" << getName(ResultTy) << "', producing '"
                  << New->getName() << "'\n");

  // Copy the pattern...
  Pattern *NewPat = NT->clone(New);

  // Apply the type to the root...
  NewPat->getTree()->updateNodeType(ResultTy, New->getName());

  // Infer types...
  NewPat->InferAllTypes();

  // Make sure everything is good to go now...
  if (!NewPat->isResolved())
    NewPat->error("Instantiating nonterminal did not resolve all types!");

  // Add the pattern to the patterns map, add the record to the RecordKeeper,
  // return the new record.
  Patterns[New] = NewPat;
  Records.addDef(New);
  return Slot = New;
}

// CalculateComputableValues - Fill in the ComputableValues map through
// analysis of the patterns we are playing with.
void InstrSelectorEmitter::CalculateComputableValues() {
  // Loop over all of the patterns, adding them to the ComputableValues map
  for (std::map<Record*, Pattern*>::iterator I = Patterns.begin(),
         E = Patterns.end(); I != E; ++I)
    if (I->second->isResolved()) {
      // We don't want to add patterns like R32 = R32.  This is a hack working
      // around a special case of a general problem, but for now we explicitly
      // forbid these patterns.  They can never match anyway.
      Pattern *P = I->second;
      if (!P->getResult() || !P->getTree()->isLeaf() ||
          P->getResult() != P->getTree()->getValueRecord())
        ComputableValues.addPattern(P);
    }
}

#if 0
// MoveIdenticalPatterns - Given a tree pattern 'P', move all of the tree
// patterns which have the same top-level structure as P from the 'From' list to
// the 'To' list.
static void MoveIdenticalPatterns(TreePatternNode *P,
                    std::vector<std::pair<Pattern*, TreePatternNode*> > &From,
                    std::vector<std::pair<Pattern*, TreePatternNode*> > &To) {
  assert(!P->isLeaf() && "All leaves are identical!");

  const std::vector<TreePatternNode*> &PChildren = P->getChildren();
  for (unsigned i = 0; i != From.size(); ++i) {
    TreePatternNode *N = From[i].second;
    assert(P->getOperator() == N->getOperator() &&"Differing operators?");
    assert(PChildren.size() == N->getChildren().size() &&
           "Nodes with different arity??");
    bool isDifferent = false;
    for (unsigned c = 0, e = PChildren.size(); c != e; ++c) {
      TreePatternNode *PC = PChildren[c];
      TreePatternNode *NC = N->getChild(c);
      if (PC->isLeaf() != NC->isLeaf()) {
        isDifferent = true;
        break;
      }

      if (!PC->isLeaf()) {
        if (PC->getOperator() != NC->getOperator()) {
          isDifferent = true;
          break;
        }
      } else {  // It's a leaf!
        if (PC->getValueRecord() != NC->getValueRecord()) {
          isDifferent = true;
          break;
        }
      }
    }
    // If it's the same as the reference one, move it over now...
    if (!isDifferent) {
      To.push_back(std::make_pair(From[i].first, N));
      From.erase(From.begin()+i);
      --i;   // Don't skip an entry...
    }
  }
}
#endif

static std::string getNodeName(Record *R) {
  RecordVal *RV = R->getValue("EnumName");
  if (RV)
    if (Init *I = RV->getValue())
      if (StringInit *SI = dynamic_cast<StringInit*>(I))
        return SI->getValue();
  return R->getName();
}


static void EmitPatternPredicates(TreePatternNode *Tree,
                                  const std::string &VarName, std::ostream &OS){
  OS << " && " << VarName << "->getNodeType() == ISD::"
     << getNodeName(Tree->getOperator());

  for (unsigned c = 0, e = Tree->getNumChildren(); c != e; ++c)
    if (!Tree->getChild(c)->isLeaf())
      EmitPatternPredicates(Tree->getChild(c),
                            VarName + "->getUse(" + utostr(c)+")", OS);
}

static void EmitPatternCosts(TreePatternNode *Tree, const std::string &VarName,
                             std::ostream &OS) {
  for (unsigned c = 0, e = Tree->getNumChildren(); c != e; ++c)
    if (Tree->getChild(c)->isLeaf()) {
      OS << " + Match_"
         << Pattern::getSlotName(Tree->getChild(c)->getValueRecord()) << "("
         << VarName << "->getUse(" << c << "))";
    } else {
      EmitPatternCosts(Tree->getChild(c),
                       VarName + "->getUse(" + utostr(c) + ")", OS);
    }
}


// EmitMatchCosters - Given a list of patterns, which all have the same root
// pattern operator, emit an efficient decision tree to decide which one to
// pick.  This is structured this way to avoid reevaluations of non-obvious
// subexpressions.
void InstrSelectorEmitter::EmitMatchCosters(std::ostream &OS,
           const std::vector<std::pair<Pattern*, TreePatternNode*> > &Patterns,
                                            const std::string &VarPrefix,
                                            unsigned IndentAmt) {
  assert(!Patterns.empty() && "No patterns to emit matchers for!");
  std::string Indent(IndentAmt, ' ');
  
  // Load all of the operands of the root node into scalars for fast access
  const NodeType &ONT = getNodeType(Patterns[0].second->getOperator());
  for (unsigned i = 0, e = ONT.ArgTypes.size(); i != e; ++i)
    OS << Indent << "SelectionDAGNode *" << VarPrefix << "_Op" << i
       << " = N->getUse(" << i << ");\n";

  // Compute the costs of computing the various nonterminals/registers, which
  // are directly used at this level.
  OS << "\n" << Indent << "// Operand matching costs...\n";
  std::set<std::string> ComputedValues;   // Avoid duplicate computations...
  for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
    TreePatternNode *NParent = Patterns[i].second;
    for (unsigned c = 0, e = NParent->getNumChildren(); c != e; ++c) {
      TreePatternNode *N = NParent->getChild(c);
      if (N->isLeaf()) {
        Record *VR = N->getValueRecord();
        const std::string &LeafName = VR->getName();
        std::string OpName  = VarPrefix + "_Op" + utostr(c);
        std::string ValName = OpName + "_" + LeafName + "_Cost";
        if (!ComputedValues.count(ValName)) {
          OS << Indent << "unsigned " << ValName << " = Match_"
             << Pattern::getSlotName(VR) << "(" << OpName << ");\n";
          ComputedValues.insert(ValName);
        }
      }
    }
  }
  OS << "\n";


  std::string LocCostName = VarPrefix + "_Cost";
  OS << Indent << "unsigned " << LocCostName << "Min = ~0U >> 1;\n"
     << Indent << "unsigned " << VarPrefix << "_PatternMin = NoMatchPattern;\n";
  
#if 0
  // Separate out all of the patterns into groups based on what their top-level
  // signature looks like...
  std::vector<std::pair<Pattern*, TreePatternNode*> > PatternsLeft(Patterns);
  while (!PatternsLeft.empty()) {
    // Process all of the patterns that have the same signature as the last
    // element...
    std::vector<std::pair<Pattern*, TreePatternNode*> > Group;
    MoveIdenticalPatterns(PatternsLeft.back().second, PatternsLeft, Group);
    assert(!Group.empty() && "Didn't at least pick the source pattern?");

#if 0
    OS << "PROCESSING GROUP:\n";
    for (unsigned i = 0, e = Group.size(); i != e; ++i)
      OS << "  " << *Group[i].first << "\n";
    OS << "\n\n";
#endif

    OS << Indent << "{ // ";

    if (Group.size() != 1) {
      OS << Group.size() << " size group...\n";
      OS << Indent << "  unsigned " << VarPrefix << "_Pattern = NoMatch;\n";
    } else {
      OS << *Group[0].first << "\n";
      OS << Indent << "  unsigned " << VarPrefix << "_Pattern = "
         << Group[0].first->getRecord()->getName() << "_Pattern;\n";
    }

    OS << Indent << "  unsigned " << LocCostName << " = ";
    if (Group.size() == 1)
      OS << "1;\n";    // Add inst cost if at individual rec
    else
      OS << "0;\n";

    // Loop over all of the operands, adding in their costs...
    TreePatternNode *N = Group[0].second;
    const std::vector<TreePatternNode*> &Children = N->getChildren();

    // If necessary, emit conditionals to check for the appropriate tree
    // structure here...
    for (unsigned i = 0, e = Children.size(); i != e; ++i) {
      TreePatternNode *C = Children[i];
      if (C->isLeaf()) {
        // We already calculated the cost for this leaf, add it in now...
        OS << Indent << "  " << LocCostName << " += "
           << VarPrefix << "_Op" << utostr(i) << "_"
           << C->getValueRecord()->getName() << "_Cost;\n";
      } else {
        // If it's not a leaf, we have to check to make sure that the current
        // node has the appropriate structure, then recurse into it...
        OS << Indent << "  if (" << VarPrefix << "_Op" << i
           << "->getNodeType() == ISD::" << getNodeName(C->getOperator())
           << ") {\n";
        std::vector<std::pair<Pattern*, TreePatternNode*> > SubPatterns;
        for (unsigned n = 0, e = Group.size(); n != e; ++n)
          SubPatterns.push_back(std::make_pair(Group[n].first,
                                               Group[n].second->getChild(i)));
        EmitMatchCosters(OS, SubPatterns, VarPrefix+"_Op"+utostr(i),
                         IndentAmt + 4);
        OS << Indent << "  }\n";
      }
    }

    // If the cost for this match is less than the minimum computed cost so far,
    // update the minimum cost and selected pattern.
    OS << Indent << "  if (" << LocCostName << " < " << LocCostName << "Min) { "
       << LocCostName << "Min = " << LocCostName << "; " << VarPrefix
       << "_PatternMin = " << VarPrefix << "_Pattern; }\n";
    
    OS << Indent << "}\n";
  }
#endif

  for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
    Pattern *P = Patterns[i].first;
    TreePatternNode *PTree = P->getTree();
    unsigned PatternCost = 1;

    // Check to see if there are any non-leaf elements in the pattern.  If so,
    // we need to emit a predicate for this match.
    bool AnyNonLeaf = false;
    for (unsigned c = 0, e = PTree->getNumChildren(); c != e; ++c)
      if (!PTree->getChild(c)->isLeaf()) {
        AnyNonLeaf = true;
        break;
      }

    if (!AnyNonLeaf) {   // No predicate necessary, just output a scope...
      OS << "  {// " << *P << "\n";
    } else {
      // We need to emit a predicate to make sure the tree pattern matches, do
      // so now...
      OS << "  if (1";
      for (unsigned c = 0, e = PTree->getNumChildren(); c != e; ++c)
        if (!PTree->getChild(c)->isLeaf())
          EmitPatternPredicates(PTree->getChild(c),
                                VarPrefix + "_Op" + utostr(c), OS);

      OS << ") {\n    // " << *P << "\n";
    }

    OS << "    unsigned PatCost = " << PatternCost;

    for (unsigned c = 0, e = PTree->getNumChildren(); c != e; ++c)
      if (PTree->getChild(c)->isLeaf()) {
        OS << " + " << VarPrefix << "_Op" << c << "_"
           << PTree->getChild(c)->getValueRecord()->getName() << "_Cost";
      } else {
        EmitPatternCosts(PTree->getChild(c), VarPrefix + "_Op" + utostr(c), OS);
      }
    OS << ";\n";
    OS << "    if (PatCost < MinCost) { MinCost = PatCost; Pattern = "
       << P->getRecord()->getName() << "_Pattern; }\n"
       << "  }\n";
  }
}

static void ReduceAllOperands(TreePatternNode *N, const std::string &Name,
             std::vector<std::pair<TreePatternNode*, std::string> > &Operands,
                              std::ostream &OS) {
  if (N->isLeaf()) {
    // If this is a leaf, register or nonterminal reference...
    std::string SlotName = Pattern::getSlotName(N->getValueRecord());
    OS << "    ReducedValue_" << SlotName << " *" << Name << "Val = Reduce_"
       << SlotName << "(" << Name << ", MBB);\n";
    Operands.push_back(std::make_pair(N, Name+"Val"));
  } else if (N->getNumChildren() == 0) {
    // This is a reference to a leaf tree node, like an immediate or frame
    // index.
    if (N->getType() != MVT::isVoid) {
      std::string SlotName =
        getNodeName(N->getOperator()) + "_" + getName(N->getType());
      OS << "    ReducedValue_" << SlotName << " *" << Name << "Val = "
         << Name << "->getValue<ReducedValue_" << SlotName << ">(ISD::"
         << SlotName << "_Slot);\n";
      Operands.push_back(std::make_pair(N, Name+"Val"));
    }
  } else {
    // Otherwise this is an interior node...
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
      std::string ChildName = Name + "_Op" + utostr(i);
      OS << "    SelectionDAGNode *" << ChildName << " = " << Name
         << "->getUse(" << i << ");\n";
      ReduceAllOperands(N->getChild(i), ChildName, Operands, OS);
    }
  }
}

/// PrintExpanderOperand - Print out Arg as part of the instruction emission
/// process for the expander pattern P.  This argument may be referencing some
/// values defined in P, or may just be physical register references or
/// something like that.  If PrintArg is true, we are printing out arguments to
/// the BuildMI call.  If it is false, we are printing the result register
/// name.
void InstrSelectorEmitter::PrintExpanderOperand(Init *Arg,
                                                const std::string &NameVar,
                                                TreePatternNode *ArgDeclNode,
                                                Pattern *P, bool PrintArg,
                                                std::ostream &OS) {
  if (DefInit *DI = dynamic_cast<DefInit*>(Arg)) {
    Record *Arg = DI->getDef();
    if (!ArgDeclNode->isLeaf() && ArgDeclNode->getNumChildren() != 0)
      P->error("Expected leaf node as argument!");
    Record *ArgDecl = ArgDeclNode->isLeaf() ? ArgDeclNode->getValueRecord() :
                      ArgDeclNode->getOperator();
    if (Arg->isSubClassOf("Register")) {
      // This is a physical register reference... make sure that the instruction
      // requested a register!
      if (!ArgDecl->isSubClassOf("RegisterClass"))
        P->error("Argument mismatch for instruction pattern!");

      // FIXME: This should check to see if the register is in the specified
      // register class!
      if (PrintArg) OS << ".addReg(";
      OS << getQualifiedName(Arg);
      if (PrintArg) OS << ")";
      return;
    } else if (Arg->isSubClassOf("RegisterClass")) {
      // If this is a symbolic register class reference, we must be using a
      // named value.
      if (NameVar.empty()) P->error("Did not specify WHICH register to pass!");
      if (Arg != ArgDecl) P->error("Instruction pattern mismatch!");

      if (PrintArg) OS << ".addReg(";
      OS << NameVar;
      if (PrintArg) OS << ")";
      return;
    } else if (Arg->getName() == "frameidx") {
      if (!PrintArg) P->error("Cannot define a new frameidx value!");
      OS << ".addFrameIndex(" << NameVar << ")";
      return;
    } else if (Arg->getName() == "basicblock") {
      if (!PrintArg) P->error("Cannot define a new basicblock value!");
      OS << ".addMBB(" << NameVar << ")";
      return;
    }
    P->error("Unknown operand type '" + Arg->getName() + "' to expander!");
  } else if (IntInit *II = dynamic_cast<IntInit*>(Arg)) {
    if (!NameVar.empty())
      P->error("Illegal to specify a name for a constant initializer arg!");

    // Hack this check to allow R32 values with 0 as the initializer for memory
    // references... FIXME!
    if (ArgDeclNode->isLeaf() && II->getValue() == 0 &&
        ArgDeclNode->getValueRecord()->getName() == "R32") {
      OS << ".addReg(0)";
    } else {
      if (ArgDeclNode->isLeaf() || ArgDeclNode->getOperator()->getName()!="imm")
        P->error("Illegal immediate int value '" + itostr(II->getValue()) +
                 "' operand!");
      OS << ".addZImm(" << II->getValue() << ")";
    }
    return;
  }
  P->error("Unknown operand type to expander!");
}

static std::string getArgName(Pattern *P, const std::string &ArgName, 
       const std::vector<std::pair<TreePatternNode*, std::string> > &Operands) {
  assert(P->getNumArgs() == Operands.size() &&"Argument computation mismatch!");
  if (ArgName.empty()) return "";

  for (unsigned i = 0, e = P->getNumArgs(); i != e; ++i)
    if (P->getArgName(i) == ArgName)
      return Operands[i].second + "->Val";

  if (ArgName == P->getResultName())
    return "NewReg";
  P->error("Pattern does not define a value named $" + ArgName + "!");
  return "";
}


void InstrSelectorEmitter::run(std::ostream &OS) {
  // Type-check all of the node types to ensure we "understand" them.
  ReadNodeTypes();
  
  // Read in all of the nonterminals, instructions, and expanders...
  ReadNonterminals();
  ReadInstructionPatterns();
  ReadExpanderPatterns();

  // Instantiate any unresolved nonterminals with information from the context
  // that they are used in.
  InstantiateNonterminals();

  // Clear InstantiatedNTs, we don't need it anymore...
  InstantiatedNTs.clear();

  DEBUG(std::cerr << "Patterns acquired:\n");
  for (std::map<Record*, Pattern*>::iterator I = Patterns.begin(),
         E = Patterns.end(); I != E; ++I)
    if (I->second->isResolved())
      DEBUG(std::cerr << "  " << *I->second << "\n");

  CalculateComputableValues();
  
  EmitSourceFileHeader("Instruction Selector for the " + Target.getName() +
                       " target", OS);
  OS << "#include \"llvm/CodeGen/MachineInstrBuilder.h\"\n";

  // Output the slot number enums...
  OS << "\nenum { // Slot numbers...\n"
     << "  LastBuiltinSlot = ISD::NumBuiltinSlots-1, // Start numbering here\n";
  for (PatternOrganizer::iterator I = ComputableValues.begin(),
         E = ComputableValues.end(); I != E; ++I)
    OS << "  " << I->first << "_Slot,\n";
  OS << "  NumSlots\n};\n\n// Reduction value typedefs...\n";

  // Output the reduction value typedefs...
  for (PatternOrganizer::iterator I = ComputableValues.begin(),
         E = ComputableValues.end(); I != E; ++I) {

    OS << "typedef ReducedValue<unsigned, " << I->first
       << "_Slot> ReducedValue_" << I->first << ";\n";
  }

  // Output the pattern enums...
  OS << "\n\n"
     << "enum { // Patterns...\n"
     << "  NotComputed = 0,\n"
     << "  NoMatchPattern, \n";
  for (PatternOrganizer::iterator I = ComputableValues.begin(),
         E = ComputableValues.end(); I != E; ++I) {
    OS << "  // " << I->first << " patterns...\n";
    for (PatternOrganizer::NodesForSlot::iterator J = I->second.begin(),
           E = I->second.end(); J != E; ++J)
      for (unsigned i = 0, e = J->second.size(); i != e; ++i)
        OS << "  " << J->second[i]->getRecord()->getName() << "_Pattern,\n";
  }
  OS << "};\n\n";

  //===--------------------------------------------------------------------===//
  // Emit the class definition...
  //
  OS << "namespace {\n"
     << "  class " << Target.getName() << "ISel {\n"
     << "    SelectionDAG &DAG;\n"
     << "  public:\n"
     << "    X86ISel(SelectionDAG &D) : DAG(D) {}\n"
     << "    void generateCode();\n"
     << "  private:\n"
     << "    unsigned makeAnotherReg(const TargetRegisterClass *RC) {\n"
     << "      return DAG.getMachineFunction().getSSARegMap()->createVirt"
                                       "ualRegister(RC);\n"
     << "    }\n\n"
     << "    // DAG matching methods for classes... all of these methods"
                                       " return the cost\n"
     << "    // of producing a value of the specified class and type, which"
                                       " also gets\n"
     << "    // added to the DAG node.\n";

  // Output all of the matching prototypes for slots...
  for (PatternOrganizer::iterator I = ComputableValues.begin(),
         E = ComputableValues.end(); I != E; ++I)
    OS << "    unsigned Match_" << I->first << "(SelectionDAGNode *N);\n";
  OS << "\n    // DAG matching methods for DAG nodes...\n";

  // Output all of the matching prototypes for slot/node pairs
  for (PatternOrganizer::iterator I = ComputableValues.begin(),
         E = ComputableValues.end(); I != E; ++I)
    for (PatternOrganizer::NodesForSlot::iterator J = I->second.begin(),
           E = I->second.end(); J != E; ++J)
      OS << "    unsigned Match_" << I->first << "_" << getNodeName(J->first)
         << "(SelectionDAGNode *N);\n";

  // Output all of the dag reduction methods prototypes...
  OS << "\n    // DAG reduction methods...\n";
  for (PatternOrganizer::iterator I = ComputableValues.begin(),
         E = ComputableValues.end(); I != E; ++I)
    OS << "    ReducedValue_" << I->first << " *Reduce_" << I->first
       << "(SelectionDAGNode *N,\n" << std::string(27+2*I->first.size(), ' ')
       << "MachineBasicBlock *MBB);\n";
  OS << "  };\n}\n\n";

  // Emit the generateCode entry-point...
  OS << "void X86ISel::generateCode() {\n"
     << "  SelectionDAGNode *Root = DAG.getRoot();\n"
     << "  assert(Root->getValueType() == MVT::isVoid && "
                                       "\"Root of DAG produces value??\");\n\n"
     << "  std::cerr << \"\\n\";\n"
     << "  unsigned Cost = Match_Void_void(Root);\n"
     << "  if (Cost >= ~0U >> 1) {\n"
     << "    std::cerr << \"Match failed!\\n\";\n"
     << "    Root->dump();\n"
     << "    abort();\n"
     << "  }\n\n"
     << "  std::cerr << \"Total DAG Cost: \" << Cost << \"\\n\\n\";\n\n"
     << "  Reduce_Void_void(Root, 0);\n"
     << "}\n\n"
     << "//===" << std::string(70, '-') << "===//\n"
     << "//  Matching methods...\n"
     << "//\n\n";

  //===--------------------------------------------------------------------===//
  // Emit all of the matcher methods...
  //
  for (PatternOrganizer::iterator I = ComputableValues.begin(),
         E = ComputableValues.end(); I != E; ++I) {
    const std::string &SlotName = I->first;
    OS << "unsigned " << Target.getName() << "ISel::Match_" << SlotName
       << "(SelectionDAGNode *N) {\n"
       << "  assert(N->getValueType() == MVT::"
       << getEnumName((*I->second.begin()).second[0]->getTree()->getType())
       << ");\n" << "  // If we already have a cost available for " << SlotName
       << " use it!\n"
       << "  if (N->getPatternFor(" << SlotName << "_Slot))\n"
       << "    return N->getCostFor(" << SlotName << "_Slot);\n\n"
       << "  unsigned Cost;\n"
       << "  switch (N->getNodeType()) {\n"
       << "  default: Cost = ~0U >> 1;   // Match failed\n"
       << "           N->setPatternCostFor(" << SlotName << "_Slot, NoMatchPattern, Cost, NumSlots);\n"
       << "           break;\n";

    for (PatternOrganizer::NodesForSlot::iterator J = I->second.begin(),
           E = I->second.end(); J != E; ++J)
      if (!J->first->isSubClassOf("Nonterminal"))
        OS << "  case ISD::" << getNodeName(J->first) << ":\tCost = Match_"
           << SlotName << "_" << getNodeName(J->first) << "(N); break;\n";
    OS << "  }\n";  // End of the switch statement

    // Emit any patterns which have a nonterminal leaf as the RHS.  These may
    // match multiple root nodes, so they cannot be handled with the switch...
    for (PatternOrganizer::NodesForSlot::iterator J = I->second.begin(),
           E = I->second.end(); J != E; ++J)
      if (J->first->isSubClassOf("Nonterminal")) {
        OS << "  unsigned " << J->first->getName() << "_Cost = Match_"
           << getNodeName(J->first) << "(N);\n"
           << "  if (" << getNodeName(J->first) << "_Cost < Cost) Cost = "
           << getNodeName(J->first) << "_Cost;\n";
      }

    OS << "  return Cost;\n}\n\n";

    for (PatternOrganizer::NodesForSlot::iterator J = I->second.begin(),
           E = I->second.end(); J != E; ++J) {
      Record *Operator = J->first;
      bool isNonterm = Operator->isSubClassOf("Nonterminal");
      if (!isNonterm) {
        OS << "unsigned " << Target.getName() << "ISel::Match_";
        if (!isNonterm) OS << SlotName << "_";
        OS << getNodeName(Operator) << "(SelectionDAGNode *N) {\n"
           << "  unsigned Pattern = NoMatchPattern;\n"
           << "  unsigned MinCost = ~0U >> 1;\n";
        
        std::vector<std::pair<Pattern*, TreePatternNode*> > Patterns;
        for (unsigned i = 0, e = J->second.size(); i != e; ++i)
          Patterns.push_back(std::make_pair(J->second[i],
                                            J->second[i]->getTree()));
        EmitMatchCosters(OS, Patterns, "N", 2);
        
        OS << "\n  N->setPatternCostFor(" << SlotName
           << "_Slot, Pattern, MinCost, NumSlots);\n"
           << "  return MinCost;\n"
           << "}\n";
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Emit all of the reducer methods...
  //
  OS << "\n\n//===" << std::string(70, '-') << "===//\n"
     << "// Reducer methods...\n"
     << "//\n";

  for (PatternOrganizer::iterator I = ComputableValues.begin(),
         E = ComputableValues.end(); I != E; ++I) {
    const std::string &SlotName = I->first;
    OS << "ReducedValue_" << SlotName << " *" << Target.getName()
       << "ISel::Reduce_" << SlotName
       << "(SelectionDAGNode *N, MachineBasicBlock *MBB) {\n"
       << "  ReducedValue_" << SlotName << " *Val = N->hasValue<ReducedValue_"
       << SlotName << ">(" << SlotName << "_Slot);\n"
       << "  if (Val) return Val;\n"
       << "  if (N->getBB()) MBB = N->getBB();\n\n"
       << "  switch (N->getPatternFor(" << SlotName << "_Slot)) {\n";

    // Loop over all of the patterns that can produce a value for this slot...
    PatternOrganizer::NodesForSlot &NodesForSlot = I->second;
    for (PatternOrganizer::NodesForSlot::iterator J = NodesForSlot.begin(),
           E = NodesForSlot.end(); J != E; ++J)
      for (unsigned i = 0, e = J->second.size(); i != e; ++i) {
        Pattern *P = J->second[i];
        OS << "  case " << P->getRecord()->getName() << "_Pattern: {\n"
           << "    // " << *P << "\n";
        // Loop over the operands, reducing them...
        std::vector<std::pair<TreePatternNode*, std::string> > Operands;
        ReduceAllOperands(P->getTree(), "N", Operands, OS);
        
        // Now that we have reduced all of our operands, and have the values
        // that reduction produces, perform the reduction action for this
        // pattern.
        std::string Result;

        // If the pattern produces a register result, generate a new register
        // now.
        if (Record *R = P->getResult()) {
          assert(R->isSubClassOf("RegisterClass") &&
                 "Only handle register class results so far!");
          OS << "    unsigned NewReg = makeAnotherReg(" << Target.getName()
             << "::" << R->getName() << "RegisterClass);\n";
          Result = "NewReg";
          DEBUG(OS << "    std::cerr << \"%reg\" << NewReg << \" =\t\";\n");
        } else {
          DEBUG(OS << "    std::cerr << \"\t\t\";\n");
          Result = "0";
        }

        // Print out the pattern that matched...
        DEBUG(OS << "    std::cerr << \"  " << P->getRecord()->getName() <<'"');
        DEBUG(for (unsigned i = 0, e = Operands.size(); i != e; ++i)
                if (Operands[i].first->isLeaf()) {
                  Record *RV = Operands[i].first->getValueRecord();
                  assert(RV->isSubClassOf("RegisterClass") &&
                         "Only handles registers here so far!");
                  OS << " << \" %reg\" << " << Operands[i].second
                     << "->Val";
                } else {
                  OS << " << ' ' << " << Operands[i].second
                     << "->Val";
                });
        DEBUG(OS << " << \"\\n\";\n");
        
        // Generate the reduction code appropriate to the particular type of
        // pattern that this is...
        switch (P->getPatternType()) {
        case Pattern::Instruction:
          // Instruction patterns just emit a single MachineInstr, using BuildMI
          OS << "    BuildMI(MBB, " << Target.getName() << "::"
             << P->getRecord()->getName() << ", " << Operands.size();
          if (P->getResult()) OS << ", NewReg";
          OS << ")";

          for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
            TreePatternNode *Op = Operands[i].first;
            if (Op->isLeaf()) {
              Record *RV = Op->getValueRecord();
              assert(RV->isSubClassOf("RegisterClass") &&
                     "Only handles registers here so far!");
              OS << ".addReg(" << Operands[i].second << "->Val)";
            } else if (Op->getOperator()->getName() == "imm") {
              OS << ".addZImm(" << Operands[i].second << "->Val)";
            } else if (Op->getOperator()->getName() == "basicblock") {
              OS << ".addMBB(" << Operands[i].second << "->Val)";
            } else {
              assert(0 && "Unknown value type!");
            }
          }
          OS << ";\n";
          break;
        case Pattern::Expander: {
          // Expander patterns emit one machine instr for each instruction in
          // the list of instructions expanded to.
          ListInit *Insts = P->getRecord()->getValueAsListInit("Result");
          for (unsigned IN = 0, e = Insts->getSize(); IN != e; ++IN) {
            DagInit *DIInst = dynamic_cast<DagInit*>(Insts->getElement(IN));
            if (!DIInst) P->error("Result list must contain instructions!");
            Record *InstRec  = DIInst->getNodeType();
            Pattern *InstPat = getPattern(InstRec);
            if (!InstPat || InstPat->getPatternType() != Pattern::Instruction)
              P->error("Instruction list must contain Instruction patterns!");
            
            bool hasResult = InstPat->getResult() != 0;
            if (InstPat->getNumArgs() != DIInst->getNumArgs()-hasResult) {
              P->error("Incorrect number of arguments specified for inst '" +
                       InstPat->getRecord()->getName() + "' in result list!");
            }

            // Start emission of the instruction...
            OS << "    BuildMI(MBB, " << Target.getName() << "::"
               << InstRec->getName() << ", "
               << DIInst->getNumArgs()-hasResult;
            // Emit register result if necessary..
            if (hasResult) {
              std::string ArgNameVal =
                getArgName(P, DIInst->getArgName(0), Operands);
              PrintExpanderOperand(DIInst->getArg(0), ArgNameVal,
                                   InstPat->getResultNode(), P, false,
                                   OS << ", ");
            }
            OS << ")";

            for (unsigned i = hasResult, e = DIInst->getNumArgs(); i != e; ++i){
              std::string ArgNameVal =
                getArgName(P, DIInst->getArgName(i), Operands);

              PrintExpanderOperand(DIInst->getArg(i), ArgNameVal,
                                   InstPat->getArg(i-hasResult), P, true, OS);
            }

            OS << ";\n";
          }
          break;
        }
        default:
          assert(0 && "Reduction of this type of pattern not implemented!");
        }

        OS << "    Val = new ReducedValue_" << SlotName << "(" << Result<<");\n"
           << "    break;\n"
           << "  }\n";
      }
    
    
    OS << "  default: assert(0 && \"Unknown " << SlotName << " pattern!\");\n"
       << "  }\n\n  N->addValue(Val);  // Do not ever recalculate this\n"
       << "  return Val;\n}\n\n";
  }
}

