//===- DAGISelEmitter.h - Generate an instruction selector ------*- C++ -*-===//
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

#ifndef DAGISEL_EMITTER_H
#define DAGISEL_EMITTER_H

#include "TableGenBackend.h"
#include "CodeGenTarget.h"

namespace llvm {
  class Record;
  struct Init;
  class ListInit;
  class DagInit;
  class SDNodeInfo;
  class TreePattern;
  class TreePatternNode;
  class DAGISelEmitter;
  
  /// SDTypeConstraint - This is a discriminated union of constraints,
  /// corresponding to the SDTypeConstraint tablegen class in Target.td.
  struct SDTypeConstraint {
    SDTypeConstraint(Record *R);
    
    unsigned OperandNo;   // The operand # this constraint applies to.
    enum { 
      SDTCisVT, SDTCisInt, SDTCisFP, SDTCisSameAs, SDTCisVTSmallerThanOp
    } ConstraintType;
    
    union {   // The discriminated union.
      struct {
        MVT::ValueType VT;
      } SDTCisVT_Info;
      struct {
        unsigned OtherOperandNum;
      } SDTCisSameAs_Info;
      struct {
        unsigned OtherOperandNum;
      } SDTCisVTSmallerThanOp_Info;
    } x;

    /// ApplyTypeConstraint - Given a node in a pattern, apply this type
    /// constraint to the nodes operands.  This returns true if it makes a
    /// change, false otherwise.  If a type contradiction is found, throw an
    /// exception.
    bool ApplyTypeConstraint(TreePatternNode *N, const SDNodeInfo &NodeInfo,
                             TreePattern &TP) const;
    
    /// getOperandNum - Return the node corresponding to operand #OpNo in tree
    /// N, which has NumResults results.
    TreePatternNode *getOperandNum(unsigned OpNo, TreePatternNode *N,
                                   unsigned NumResults) const;
  };
  
  /// SDNodeInfo - One of these records is created for each SDNode instance in
  /// the target .td file.  This represents the various dag nodes we will be
  /// processing.
  class SDNodeInfo {
    Record *Def;
    std::string EnumName;
    std::string SDClassName;
    unsigned NumResults;
    int NumOperands;
    std::vector<SDTypeConstraint> TypeConstraints;
  public:
    SDNodeInfo(Record *R);  // Parse the specified record.
    
    unsigned getNumResults() const { return NumResults; }
    int getNumOperands() const { return NumOperands; }
    Record *getRecord() const { return Def; }
    const std::string &getEnumName() const { return EnumName; }
    const std::string &getSDClassName() const { return SDClassName; }
    
    const std::vector<SDTypeConstraint> &getTypeConstraints() const {
      return TypeConstraints;
    }

    /// ApplyTypeConstraints - Given a node in a pattern, apply the type
    /// constraints for this node to the operands of the node.  This returns
    /// true if it makes a change, false otherwise.  If a type contradiction is
    /// found, throw an exception.
    bool ApplyTypeConstraints(TreePatternNode *N, TreePattern &TP) const {
      bool MadeChange = false;
      for (unsigned i = 0, e = TypeConstraints.size(); i != e; ++i)
        MadeChange |= TypeConstraints[i].ApplyTypeConstraint(N, *this, TP);
      return MadeChange;
    }
  };

  /// FIXME: TreePatternNode's can be shared in some cases (due to dag-shaped
  /// patterns), and as such should be ref counted.  We currently just leak all
  /// TreePatternNode objects!
  class TreePatternNode {
    /// The inferred type for this node, or MVT::LAST_VALUETYPE if it hasn't
    /// been determined yet.
    MVT::ValueType Ty;

    /// Operator - The Record for the operator if this is an interior node (not
    /// a leaf).
    Record *Operator;
    
    /// Val - The init value (e.g. the "GPRC" record, or "7") for a leaf.
    ///
    Init *Val;
    
    /// Name - The name given to this node with the :$foo notation.
    ///
    std::string Name;
    
    /// PredicateFn - The predicate function to execute on this node to check
    /// for a match.  If this string is empty, no predicate is involved.
    std::string PredicateFn;
    
    /// TransformFn - The transformation function to execute on this node before
    /// it can be substituted into the resulting instruction on a pattern match.
    Record *TransformFn;
    
    std::vector<TreePatternNode*> Children;
  public:
    TreePatternNode(Record *Op, const std::vector<TreePatternNode*> &Ch) 
      : Ty(MVT::LAST_VALUETYPE), Operator(Op), Val(0), TransformFn(0),
        Children(Ch) {}
    TreePatternNode(Init *val)    // leaf ctor
      : Ty(MVT::LAST_VALUETYPE), Operator(0), Val(val), TransformFn(0) {}
    ~TreePatternNode();
    
    const std::string &getName() const { return Name; }
    void setName(const std::string &N) { Name = N; }
    
    bool isLeaf() const { return Val != 0; }
    bool hasTypeSet() const { return Ty != MVT::LAST_VALUETYPE; }
    MVT::ValueType getType() const { return Ty; }
    void setType(MVT::ValueType VT) { Ty = VT; }
    
    Init *getLeafValue() const { assert(isLeaf()); return Val; }
    Record *getOperator() const { assert(!isLeaf()); return Operator; }
    
    unsigned getNumChildren() const { return Children.size(); }
    TreePatternNode *getChild(unsigned N) const { return Children[N]; }
    void setChild(unsigned i, TreePatternNode *N) {
      Children[i] = N;
    }
    
    const std::string &getPredicateFn() const { return PredicateFn; }
    void setPredicateFn(const std::string &Fn) { PredicateFn = Fn; }

    Record *getTransformFn() const { return TransformFn; }
    void setTransformFn(Record *Fn) { TransformFn = Fn; }
    
    void print(std::ostream &OS) const;
    void dump() const;
    
  public:   // Higher level manipulation routines.

    /// clone - Return a new copy of this tree.
    ///
    TreePatternNode *clone() const;
    
    /// SubstituteFormalArguments - Replace the formal arguments in this tree
    /// with actual values specified by ArgMap.
    void SubstituteFormalArguments(std::map<std::string,
                                            TreePatternNode*> &ArgMap);

    /// InlinePatternFragments - If this pattern refers to any pattern
    /// fragments, inline them into place, giving us a pattern without any
    /// PatFrag references.
    TreePatternNode *InlinePatternFragments(TreePattern &TP);
    
    /// ApplyTypeConstraints - Apply all of the type constraints relevent to
    /// this node and its children in the tree.  This returns true if it makes a
    /// change, false otherwise.  If a type contradiction is found, throw an
    /// exception.
    bool ApplyTypeConstraints(TreePattern &TP);
    
    /// UpdateNodeType - Set the node type of N to VT if VT contains
    /// information.  If N already contains a conflicting type, then throw an
    /// exception.  This returns true if any information was updated.
    ///
    bool UpdateNodeType(MVT::ValueType VT, TreePattern &TP);
    
    /// ContainsUnresolvedType - Return true if this tree contains any
    /// unresolved types.
    bool ContainsUnresolvedType() const {
      if (Ty == MVT::LAST_VALUETYPE) return true;
      for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
        if (getChild(i)->ContainsUnresolvedType()) return true;
      return false;
    }
  };
  
  
  /// TreePattern - Represent a pattern, used for instructions, pattern
  /// fragments, etc.
  ///
  class TreePattern {
    /// Trees - The list of pattern trees which corresponds to this pattern.
    /// Note that PatFrag's only have a single tree.
    ///
    std::vector<TreePatternNode*> Trees;
    
    /// TheRecord - The actual TableGen record corresponding to this pattern.
    ///
    Record *TheRecord;
      
    /// Args - This is a list of all of the arguments to this pattern (for
    /// PatFrag patterns), which are the 'node' markers in this pattern.
    std::vector<std::string> Args;
    
    /// ISE - the DAG isel emitter coordinating this madness.
    ///
    DAGISelEmitter &ISE;
  public:
      
    /// TreePattern constructor - Parse the specified DagInits into the
    /// current record.
    TreePattern(Record *TheRec, ListInit *RawPat, DAGISelEmitter &ise);
    TreePattern(Record *TheRec, DagInit *Pat, DAGISelEmitter &ise);
    TreePattern(Record *TheRec, TreePatternNode *Pat, DAGISelEmitter &ise);
        
    /// getTrees - Return the tree patterns which corresponds to this pattern.
    ///
    const std::vector<TreePatternNode*> &getTrees() const { return Trees; }
    unsigned getNumTrees() const { return Trees.size(); }
    TreePatternNode *getTree(unsigned i) const { return Trees[i]; }
    TreePatternNode *getOnlyTree() const {
      assert(Trees.size() == 1 && "Doesn't have exactly one pattern!");
      return Trees[0];
    }
        
    /// getRecord - Return the actual TableGen record corresponding to this
    /// pattern.
    ///
    Record *getRecord() const { return TheRecord; }
    
    unsigned getNumArgs() const { return Args.size(); }
    const std::string &getArgName(unsigned i) const {
      assert(i < Args.size() && "Argument reference out of range!");
      return Args[i];
    }
    std::vector<std::string> &getArgList() { return Args; }
    
    DAGISelEmitter &getDAGISelEmitter() const { return ISE; }

    /// InlinePatternFragments - If this pattern refers to any pattern
    /// fragments, inline them into place, giving us a pattern without any
    /// PatFrag references.
    void InlinePatternFragments() {
      for (unsigned i = 0, e = Trees.size(); i != e; ++i)
        Trees[i] = Trees[i]->InlinePatternFragments(*this);
    }
    
    /// InferAllTypes - Infer/propagate as many types throughout the expression
    /// patterns as possible.  Return true if all types are infered, false
    /// otherwise.  Throw an exception if a type contradiction is found.
    bool InferAllTypes();
    
    /// error - Throw an exception, prefixing it with information about this
    /// pattern.
    void error(const std::string &Msg) const;
    
    void print(std::ostream &OS) const;
    void dump() const;
    
  private:
    MVT::ValueType getIntrinsicType(Record *R) const;
    TreePatternNode *ParseTreePattern(DagInit *DI);
  };


  class DAGInstruction {
    TreePattern *Pattern;
    unsigned NumResults;
    unsigned NumOperands;
    std::vector<MVT::ValueType> ResultTypes;
    std::vector<MVT::ValueType> OperandTypes;
    TreePatternNode *ResultPattern;
  public:
    DAGInstruction(TreePattern *TP,
                   const std::vector<MVT::ValueType> &resultTypes,
                   const std::vector<MVT::ValueType> &operandTypes)
      : Pattern(TP), ResultTypes(resultTypes), OperandTypes(operandTypes), 
        ResultPattern(0) {}

    TreePattern *getPattern() const { return Pattern; }
    unsigned getNumResults() const { return ResultTypes.size(); }
    unsigned getNumOperands() const { return OperandTypes.size(); }
    
    void setResultPattern(TreePatternNode *R) { ResultPattern = R; }
    
    MVT::ValueType getResultType(unsigned RN) const {
      assert(RN < ResultTypes.size());
      return ResultTypes[RN];
    }
    
    MVT::ValueType getOperandType(unsigned ON) const {
      assert(ON < OperandTypes.size());
      return OperandTypes[ON];
    }
    TreePatternNode *getResultPattern() const { return ResultPattern; }
  };
  
  
/// InstrSelectorEmitter - The top-level class which coordinates construction
/// and emission of the instruction selector.
///
class DAGISelEmitter : public TableGenBackend {
  RecordKeeper &Records;
  CodeGenTarget Target;

  std::map<Record*, SDNodeInfo> SDNodes;
  std::map<Record*, std::pair<Record*, std::string> > SDNodeXForms;
  std::map<Record*, TreePattern*> PatternFragments;
  std::map<Record*, DAGInstruction> Instructions;
  
  /// PatternsToMatch - All of the things we are matching on the DAG.  The first
  /// value is the pattern to match, the second pattern is the result to
  /// emit.
  typedef std::pair<TreePatternNode*, TreePatternNode*> PatternToMatch;
  std::vector<PatternToMatch> PatternsToMatch;
public:
  DAGISelEmitter(RecordKeeper &R) : Records(R) {}

  // run - Output the isel, returning true on failure.
  void run(std::ostream &OS);
  
  const SDNodeInfo &getSDNodeInfo(Record *R) const {
    assert(SDNodes.count(R) && "Unknown node!");
    return SDNodes.find(R)->second;
  }

  TreePattern *getPatternFragment(Record *R) const {
    assert(PatternFragments.count(R) && "Invalid pattern fragment request!");
    return PatternFragments.find(R)->second;
  }
  
  const std::pair<Record*, std::string> &getSDNodeTransform(Record *R) const {
    assert(SDNodeXForms.count(R) && "Invalid transform!");
    return SDNodeXForms.find(R)->second;
  }
  
  const DAGInstruction &getInstruction(Record *R) const {
    assert(Instructions.count(R) && "Unknown instruction!");
    return Instructions.find(R)->second;
  }
  
private:
  void ParseNodeInfo();
  void ParseNodeTransforms(std::ostream &OS);
  void ParsePatternFragments(std::ostream &OS);
  void ParseInstructions();
  void ParsePatterns();
  void FindPatternInputsAndOutputs(TreePattern *I, TreePatternNode *Pat,
                                   std::map<std::string,
                                            TreePatternNode*> &InstInputs,
                                   std::map<std::string, Record*> &InstResults);
  void EmitInstructionSelector(std::ostream &OS);
};

} // End llvm namespace

#endif
