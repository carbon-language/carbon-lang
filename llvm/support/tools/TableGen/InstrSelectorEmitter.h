//===- InstrInfoEmitter.h - Generate a Instruction Set Desc. ----*- C++ -*-===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef INSTRSELECTOR_EMITTER_H
#define INSTRSELECTOR_EMITTER_H

#include "TableGenBackend.h"
#include <vector>
#include <map>

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

class InstrSelectorEmitter : public TableGenBackend {
  RecordKeeper &Records;

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

  // ProcessInstructionPatterns - Read in all subclasses of Instruction, and
  // process those with a useful Pattern field.
  void ProcessInstructionPatterns();
};

#endif
