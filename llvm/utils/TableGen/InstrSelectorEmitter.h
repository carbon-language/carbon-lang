//===- InstrInfoEmitter.h - Generate a Instruction Set Desc. ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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
#include <cassert>

class DagInit;
class Init;
class InstrSelectorEmitter;

/// NodeType - Represents Information parsed from the DagNode entries.
///
struct NodeType {
  enum ArgResultTypes {
    Any,            // No constraint on type
    Val,            // A non-void type
    Arg0,           // Value matches the type of Arg0
    Arg1,           // Value matches the type of Arg1
    Ptr,            // Tree node is the type of the target pointer
    I8,             // Always bool
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



/// TreePatternNode - Represent a node of the tree patterns.
///
class TreePatternNode {
  /// Operator - The operation that this node represents... this is null if this
  /// is a leaf.
  Record *Operator;

  /// Type - The inferred value type...
  ///
  MVT::ValueType                Type;

  /// Children - If this is not a leaf (Operator != 0), this is the subtrees
  /// that we contain.
  std::vector<std::pair<TreePatternNode*, std::string> > Children;

  /// Value - If this node is a leaf, this indicates what the thing is.
  ///
  Init *Value;
public:
  TreePatternNode(Record *o, const std::vector<std::pair<TreePatternNode*,
                                                         std::string> > &c)
    : Operator(o), Type(MVT::Other), Children(c), Value(0) {}
  TreePatternNode(Init *V) : Operator(0), Type(MVT::Other), Value(V) {}

  Record *getOperator() const {
    assert(Operator && "This is a leaf node!");
    return Operator;
  }
  MVT::ValueType getType() const { return Type; }
  void setType(MVT::ValueType T) { Type = T; }

  bool isLeaf() const { return Operator == 0; }

  unsigned getNumChildren() const { return Children.size(); }
  TreePatternNode *getChild(unsigned c) const {
    assert(Operator != 0 && "This is a leaf node!");
    assert(c < Children.size() && "Child access out of range!");
    return Children[c].first;
  }
  const std::string &getChildName(unsigned c) const {
    assert(Operator != 0 && "This is a leaf node!");
    assert(c < Children.size() && "Child access out of range!");
    return Children[c].second;
  }

  Init *getValue() const {
    assert(Operator == 0 && "This is not a leaf node!");
    return Value;
  }

  /// getValueRecord - Returns the value of this tree node as a record.  For now
  /// we only allow DefInit's as our leaf values, so this is used.
  Record *getValueRecord() const;

  /// clone - Make a copy of this tree and all of its children.
  ///
  TreePatternNode *clone() const;

  void dump() const;

  /// InstantiateNonterminals - If this pattern refers to any nonterminals which
  /// are not themselves completely resolved, clone the nonterminal and resolve
  /// it with the using context we provide.
  void InstantiateNonterminals(InstrSelectorEmitter &ISE);

  /// UpdateNodeType - Set the node type of N to VT if VT contains information.
  /// If N already contains a conflicting type, then throw an exception.  This
  /// returns true if any information was updated.
  ///
  bool updateNodeType(MVT::ValueType VT, const std::string &RecName);
};

std::ostream &operator<<(std::ostream &OS, const TreePatternNode &N);



/// Pattern - Represent a pattern of one form or another.  Currently, three
/// types of patterns are possible: Instruction's, Nonterminals, and Expanders.
///
struct Pattern {
  enum PatternType {
    Nonterminal, Instruction, Expander
  };
private:
  /// PTy - The type of pattern this is.
  ///
  PatternType PTy;

  /// Tree - The tree pattern which corresponds to this pattern.  Note that if
  /// there was a (set) node on the outside level that it has been stripped off.
  ///
  TreePatternNode *Tree;
  
  /// Result - If this is an instruction or expander pattern, this is the
  /// register result, specified with a (set) in the pattern.
  ///
  std::string ResultName;      // The name of the result value...
  TreePatternNode *ResultNode; // The leaf node for the result register...

  /// TheRecord - The actual TableGen record corresponding to this pattern.
  ///
  Record *TheRecord;

  /// Resolved - This is true of the pattern is useful in practice.  In
  /// particular, some non-terminals will have non-resolvable types.  When a
  /// user of the non-terminal is later found, they will have inferred a type
  /// for the result of the non-terminal, which cause a clone of an unresolved
  /// nonterminal to be made which is "resolved".
  ///
  bool Resolved;

  /// Args - This is a list of all of the arguments to this pattern, which are
  /// the non-void leaf nodes in this pattern.
  std::vector<std::pair<TreePatternNode*, std::string> > Args;

  /// ISE - the instruction selector emitter coordinating this madness.
  ///
  InstrSelectorEmitter &ISE;
public:

  /// Pattern constructor - Parse the specified DagInitializer into the current
  /// record.
  Pattern(PatternType pty, DagInit *RawPat, Record *TheRec,
          InstrSelectorEmitter &ise);

  /// Pattern - Constructor used for cloning nonterminal patterns
  Pattern(TreePatternNode *tree, Record *rec, bool res,
          InstrSelectorEmitter &ise)
    : PTy(Nonterminal), Tree(tree), ResultNode(0), TheRecord(rec),
      Resolved(res), ISE(ise) {
    calculateArgs(Tree, "");
  }

  /// getPatternType - Return what flavor of Record this pattern originated from
  ///
  PatternType getPatternType() const { return PTy; }

  /// getTree - Return the tree pattern which corresponds to this pattern.
  ///
  TreePatternNode *getTree() const { return Tree; }
  
  Record *getResult() const {
    return ResultNode ? ResultNode->getValueRecord() : 0;
  }
  const std::string &getResultName() const { return ResultName; }
  TreePatternNode *getResultNode() const { return ResultNode; }

  /// getRecord - Return the actual TableGen record corresponding to this
  /// pattern.
  ///
  Record *getRecord() const { return TheRecord; }

  unsigned getNumArgs() const { return Args.size(); }
  TreePatternNode *getArg(unsigned i) const {
    assert(i < Args.size() && "Argument reference out of range!");
    return Args[i].first;
  }
  Record *getArgRec(unsigned i) const {
    return getArg(i)->getValueRecord();
  }
  Init *getArgVal(unsigned i) const {
    return getArg(i)->getValue();
  }
  const std::string &getArgName(unsigned i) const {
    assert(i < Args.size() && "Argument reference out of range!");
    return Args[i].second;
  }

  bool isResolved() const { return Resolved; }

  /// InferAllTypes - Runs the type inference engine on the current pattern,
  /// stopping when nothing can be inferred, then updating the Resolved field.
  void InferAllTypes();

  /// InstantiateNonterminals - If this pattern refers to any nonterminals which
  /// are not themselves completely resolved, clone the nonterminal and resolve
  /// it with the using context we provide.
  void InstantiateNonterminals() {
    Tree->InstantiateNonterminals(ISE);
  }

  /// clone - This method is used to make an exact copy of the current pattern,
  /// then change the "TheRecord" instance variable to the specified record.
  ///
  Pattern *clone(Record *R) const;

  /// error - Throw an exception, prefixing it with information about this
  /// pattern.
  void error(const std::string &Msg) const;

  /// getSlotName - If this is a leaf node, return the slot name that the
  /// operand will update.
  std::string getSlotName() const;
  static std::string getSlotName(Record *R);

  void dump() const;

private:
  void calculateArgs(TreePatternNode *N, const std::string &Name);
  MVT::ValueType getIntrinsicType(Record *R) const;
  TreePatternNode *ParseTreePattern(DagInit *DI);
  bool InferTypes(TreePatternNode *N, bool &MadeChange);
};

std::ostream &operator<<(std::ostream &OS, const Pattern &P);


/// PatternOrganizer - This class represents all of the patterns which are
/// useful for the instruction selector, neatly catagorized in a hierarchical
/// structure.
struct PatternOrganizer {
  /// PatternsForNode - The list of patterns which can produce a value of a
  /// particular slot type, given a particular root node in the tree.  All of
  /// the patterns in this vector produce the same value type and have the same
  /// root DAG node.
  typedef std::vector<Pattern*> PatternsForNode;

  /// NodesForSlot - This map keeps track of all of the root DAG nodes which can
  /// lead to the production of a value for this slot.  All of the patterns in
  /// this data structure produces values of the same slot.
  typedef std::map<Record*, PatternsForNode> NodesForSlot;

  /// AllPatterns - This data structure contains all patterns in the instruction
  /// selector.
  std::map<std::string, NodesForSlot> AllPatterns;

  // Forwarding functions...
  typedef std::map<std::string, NodesForSlot>::iterator iterator;
  iterator begin() { return AllPatterns.begin(); }
  iterator end()   { return AllPatterns.end(); }


  /// addPattern - Add the specified pattern to the appropriate location in the
  /// collection.
  void addPattern(Pattern *P);
};


/// InstrSelectorEmitter - The top-level class which coordinates construction
/// and emission of the instruction selector.
///
class InstrSelectorEmitter : public TableGenBackend {
  RecordKeeper &Records;
  CodeGenTarget Target;

  std::map<Record*, NodeType> NodeTypes;

  /// Patterns - a list of all of the patterns defined by the target description
  ///
  std::map<Record*, Pattern*> Patterns;

  /// InstantiatedNTs - A data structure to keep track of which nonterminals
  /// have been instantiated already...
  ///
  std::map<std::pair<Pattern*,MVT::ValueType>, Record*> InstantiatedNTs;

  /// ComputableValues - This map indicates which patterns can be used to
  /// generate a value that is used by the selector.  The keys of this map
  /// implicitly define the values that are used by the selector.
  ///
  PatternOrganizer ComputableValues;

public:
  InstrSelectorEmitter(RecordKeeper &R) : Records(R) {}
  
  // run - Output the instruction set description, returning true on failure.
  void run(std::ostream &OS);

  const CodeGenTarget &getTarget() const { return Target; }
  std::map<Record*, NodeType> &getNodeTypes() { return NodeTypes; }
  const NodeType &getNodeType(Record *R) const {
    std::map<Record*, NodeType>::const_iterator I = NodeTypes.find(R);
    assert(I != NodeTypes.end() && "Unknown node type!");
    return I->second;
  }

  /// getPattern - return the pattern corresponding to the specified record, or
  /// null if there is none.
  Pattern *getPattern(Record *R) const {
    std::map<Record*, Pattern*>::const_iterator I = Patterns.find(R);
    return I != Patterns.end() ? I->second : 0;
  }

  /// ReadNonterminal - This method parses the specified record as a
  /// nonterminal, but only if it hasn't been read in already.
  Pattern *ReadNonterminal(Record *R);

  /// InstantiateNonterminal - This method takes the nonterminal specified by
  /// NT, which should not be completely resolved, clones it, applies ResultTy
  /// to its root, then runs the type inference stuff on it.  This should
  /// produce a newly resolved nonterminal, which we make a record for and
  /// return.  To be extra fancy and efficient, this only makes one clone for
  /// each type it is instantiated with.
  Record *InstantiateNonterminal(Pattern *NT, MVT::ValueType ResultTy);

private:
  // ReadNodeTypes - Read in all of the node types in the current RecordKeeper,
  // turning them into the more accessible NodeTypes data structure.
  void ReadNodeTypes();

  // ReadNonTerminals - Read in all nonterminals and incorporate them into our
  // pattern database.
  void ReadNonterminals();

  // ReadInstructionPatterns - Read in all subclasses of Instruction, and
  // process those with a useful Pattern field.
  void ReadInstructionPatterns();

  // ReadExpanderPatterns - Read in all of the expanded patterns.
  void ReadExpanderPatterns();

  // InstantiateNonterminals - Instantiate any unresolved nonterminals with
  // information from the context that they are used in.
  void InstantiateNonterminals();
  
  // CalculateComputableValues - Fill in the ComputableValues map through
  // analysis of the patterns we are playing with.
  void CalculateComputableValues();

  // EmitMatchCosters - Given a list of patterns, which all have the same root
  // pattern operator, emit an efficient decision tree to decide which one to
  // pick.  This is structured this way to avoid reevaluations of non-obvious
  // subexpressions.
  void EmitMatchCosters(std::ostream &OS,
            const std::vector<std::pair<Pattern*, TreePatternNode*> > &Patterns,
                        const std::string &VarPrefix, unsigned Indent);
  
  /// PrintExpanderOperand - Print out Arg as part of the instruction emission
  /// process for the expander pattern P.  This argument may be referencing some
  /// values defined in P, or may just be physical register references or
  /// something like that.  If PrintArg is true, we are printing out arguments
  /// to the BuildMI call.  If it is false, we are printing the result register
  /// name.
  void PrintExpanderOperand(Init *Arg, const std::string &NameVar,
                            TreePatternNode *ArgDecl, Pattern *P,
                            bool PrintArg, std::ostream &OS);
};

#endif
