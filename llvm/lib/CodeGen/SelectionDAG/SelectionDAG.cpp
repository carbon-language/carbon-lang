//===-- SelectionDAG.cpp - Implement the SelectionDAG* classes ------------===//
//
// This file implements the SelectionDAG* classes, which are used to perform
// DAG-based instruction selection in a target-specific manner.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Type.h"

SelectionDAG::~SelectionDAG() {
  for (unsigned i = 0, e = AllNodes.size(); i != e; ++i)
    delete AllNodes[i];
}


/// dump - Print out the current Selection DAG...
void SelectionDAG::dump() const {
  Root->dump();  // Print from the root...
}

/// getValueType - Return the ValueType for the specified LLVM type.  This
/// method works on all scalar LLVM types.
///
MVT::ValueType SelectionDAG::getValueType(const Type *Ty) const {
  switch (Ty->getPrimitiveID()) {
  case Type::VoidTyID: assert(0 && "Void type object in getValueType!");
  default: assert(0 && "Unknown type in DAGBuilder!\n");
  case Type::BoolTyID:    return MVT::i1;
  case Type::SByteTyID:
  case Type::UByteTyID:   return MVT::i8;
  case Type::ShortTyID:
  case Type::UShortTyID:  return MVT::i16;
  case Type::IntTyID:
  case Type::UIntTyID:    return MVT::i32;
  case Type::LongTyID:
  case Type::ULongTyID:   return MVT::i64;
  case Type::FloatTyID:   return MVT::f32;
  case Type::DoubleTyID:  return MVT::f64;
  case Type::PointerTyID: return PointerType;
  }
}

void SelectionDAGNode::dump() const {
  // Print out the DAG in post-order
  std::map<const SelectionDAGNode*, unsigned> NodeIDs;
  unsigned ID = 0;
  printit(0, ID, NodeIDs);
}

void SelectionDAGNode::printit(unsigned Offset, unsigned &LastID,
                               std::map<const SelectionDAGNode*,
                                        unsigned> &NodeIDs) const {
  if (!NodeIDs.count(this)) {
    // Emit all of the uses first...
    for (unsigned i = 0, e = Uses.size(); i != e; ++i)
      Uses[i]->printit(Offset+1, LastID, NodeIDs);

    NodeIDs[this] = LastID++;

    std::cerr << std::string(Offset, ' ') << "#" << LastID-1 << " ";
  } else {
    // Node has already been emitted...
    std::cerr << std::string(Offset, ' ') << "#" << NodeIDs[this] << " ";
  }

  switch (ValueType) {
  case MVT::isVoid: std::cerr << "V:"; break;
  case MVT::i1:   std::cerr << "i1:"; break;
  case MVT::i8:   std::cerr << "i8:"; break;
  case MVT::i16:  std::cerr << "i16:"; break;
  case MVT::i32:  std::cerr << "i32:"; break;
  case MVT::i64:  std::cerr << "i64:"; break;
  case MVT::f32:  std::cerr << "f32:"; break;
  case MVT::f64:  std::cerr << "f64:"; break;
  default: assert(0 && "Invalid node ValueType!");
  }
  switch (NodeType) {
  case ISD::ChainNode:      std::cerr << "ChainNode"; break;
  case ISD::BlockChainNode: std::cerr << "BlockChainNode"; break;
  case ISD::ProtoNode:      std::cerr << "ProtoNode"; break;
  case ISD::Constant:       std::cerr << "Constant"; break;
  case ISD::FrameIndex:     std::cerr << "FrameIndex"; break;
  case ISD::Plus:           std::cerr << "Plus"; break;
  case ISD::Minus:          std::cerr << "Minus"; break;
  case ISD::Times:          std::cerr << "Times"; break;
  case ISD::SDiv:           std::cerr << "SDiv"; break;
  case ISD::UDiv:           std::cerr << "UDiv"; break;
  case ISD::SRem:           std::cerr << "SRem"; break;
  case ISD::URem:           std::cerr << "URem"; break;
  case ISD::And:            std::cerr << "And"; break;
  case ISD::Or:             std::cerr << "Or"; break;
  case ISD::Xor:            std::cerr << "Xor"; break;
  case ISD::Br:             std::cerr << "Br"; break;
  case ISD::Switch:         std::cerr << "Switch"; break;
  case ISD::Ret:            std::cerr << "Ret"; break;
  case ISD::RetVoid:        std::cerr << "RetVoid"; break;
  case ISD::Load:           std::cerr << "Load"; break;
  case ISD::Store:          std::cerr << "Store"; break;
  case ISD::PHI:            std::cerr << "PHI"; break;
  case ISD::Call:           std::cerr << "Call"; break;
  }

  std::cerr << "\n";
}
