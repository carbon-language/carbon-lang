//===- InstrInfoEmitter.h - Generate a Instruction Set Desc. ----*- C++ -*-===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef INSTRSELECTOR_EMITTER_H
#define INSTRSELECTOR_EMITTER_H

#include "TableGenBackend.h"
#include "CodeGenWrappers.h"
#include <vector>
#include <map>
class DagInit;
class Init;

struct NodeType {
  enum ArgResultTypes {
    // Both argument and return types...
    Val,            // A non-void type
    Arg0,           // Value matches the type of Arg0
    Ptr,            // Tree node is the type of the target pointer

    // Return types
    Void,           // Tree node always returns void
  };

  ArgResultTypes ResultType;
  std::vector<ArgResultTypes> ArgTypes;

  NodeType(ArgResultTypes RT, std::vector<ArgResultTypes> &AT) : ResultType(RT){
    AT.swap(ArgTypes);
  }

  NodeType() : ResultType(Val) {}
  NodeType(const NodeType &N) : ResultType(N.ResultType), ArgTypes(N.ArgTypes){}

  static ArgResultTypes Translate(Record *R);
};

class TreePatternNode {
  /// Operator - The operation that this node represents... this is null if this
  /// is a leaf.
  Record *Operator;

  /// Type - The inferred value type...
  MVT::ValueType                Type;

  /// Children - If this is not a leaf (Operator != 0), this is the subtrees
  /// that we contain.
  std::vector<TreePatternNode*> Children;

  /// Value - If this node is a leaf, this indicates what the thing is.
  Init *Value;
public:
  TreePatternNode(Record *o, const std::vector<TreePatternNode*> &c)
    : Operator(o), Type(MVT::Other), Children(c), Value(0) {}
  TreePatternNode(Init *V) : Operator(0), Type(MVT::Other), Value(V) {}

  Record *getOperator() const { return Operator; }
  MVT::ValueType getType() const { return Type; }
  void setType(MVT::ValueType T) { Type = T; }

  bool isLeaf() const { return Operator == 0; }

  const std::vector<TreePatternNode*> &getChildren() const {
    assert(Operator != 0 && "This is a leaf node!");
    return Children;
  }
  Init *getValue() const {
    assert(Operator == 0 && "This is not a leaf node!");
    return Value;
  }

  void dump() const;
};

std::ostream &operator<<(std::ostream &OS, const TreePatternNode &N);



class InstrSelectorEmitter : public TableGenBackend {
  RecordKeeper &Records;
  CodeGenTarget Target;

  std::map<Record*, NodeType> NodeTypes;
public:
  InstrSelectorEmitter(RecordKeeper &R) : Records(R) {}
  
  // run - Output the instruction set description, returning true on failure.
  void run(std::ostream &OS);

private:
  // ProcessNodeTypes - Process all of the node types in the current
  // RecordKeeper, turning them into the more accessible NodeTypes data
  // structure.
  void ProcessNodeTypes();

  // ProcessNonTerminals - Read in all nonterminals and incorporate them into
  // our pattern database.
  void ProcessNonTerminals();

  // ProcessInstructionPatterns - Read in all subclasses of Instruction, and
  // process those with a useful Pattern field.
  void ProcessInstructionPatterns();

  // ParseTreePattern - Parse the specified DagInit into a TreePattern which we
  // can use.
  //
  TreePatternNode *ParseTreePattern(DagInit *DI, const std::string &RecName);

  // InferTypes - Perform type inference on the tree, returning true if there
  // are any remaining untyped nodes and setting MadeChange if any changes were
  // made.
  bool InferTypes(TreePatternNode *N, const std::string &RecName,
                  bool &MadeChange);

  // ReadAndCheckPattern - Parse the specified DagInit into a pattern and then
  // perform full type inference.
  TreePatternNode *ReadAndCheckPattern(DagInit *DI, const std::string &RecName);
};

#endif
