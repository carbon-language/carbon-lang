//===- InstrInfoEmitter.cpp - Generate a Instruction Set Desc. ------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#include "InstrSelectorEmitter.h"
#include "Record.h"

NodeType::ArgResultTypes NodeType::Translate(Record *R) {
  const std::string &Name = R->getName();
  if (Name == "DNRT_void") return Void;
  if (Name == "DNRT_val" || Name == "DNAT_val") return Val;
  if (Name == "DNRT_arg0" || Name == "DNAT_arg0") return Arg0;
  if (Name == "DNAT_ptr") return Ptr;
  throw "Unknown DagNodeResult Type '" + Name + "'!";
}


/// ProcessNodeTypes - Process all of the node types in the current
/// RecordKeeper, turning them into the more accessible NodeTypes data
/// structure.
///
void InstrSelectorEmitter::ProcessNodeTypes() {
  std::vector<Record*> Nodes = Records.getAllDerivedDefinitions("DagNode");

  for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
    Record *Node = Nodes[i];
    
    // Translate the return type...
    NodeType::ArgResultTypes RetTy =
      NodeType::Translate(Node->getValueAsDef("RetType"));

    // Translate the arguments...
    ListInit *Args = Node->getValueAsListInit("ArgTypes");
    std::vector<NodeType::ArgResultTypes> ArgTypes;

    for (unsigned a = 0, e = Args->getSize(); a != e; ++a)
      if (DefInit *DI = dynamic_cast<DefInit*>(Args->getElement(a)))
        ArgTypes.push_back(NodeType::Translate(DI->getDef()));
      else
        throw "In node " + Node->getName() + ", argument is not a Def!";

    // Add the node type mapping now...
    NodeTypes[Node] = NodeType(RetTy, ArgTypes);
  }  
}

void InstrSelectorEmitter::run(std::ostream &OS) {
  // Type-check all of the node types to ensure we "understand" them.
  ProcessNodeTypes();
  
  
  
}
