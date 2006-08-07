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
#include "CodeGenIntrinsics.h"
#include <set>

namespace llvm {
  class Record;
  struct Init;
  class ListInit;
  class DagInit;
  class SDNodeInfo;
  class TreePattern;
  class TreePatternNode;
  class DAGISelEmitter;
  class ComplexPattern;
  
  /// MVT::DAGISelGenValueType - These are some extended forms of MVT::ValueType
  /// that we use as lattice values during type inferrence.
  namespace MVT {
    enum DAGISelGenValueType {
      isFP  = MVT::LAST_VALUETYPE,
      isInt,
      isUnknown
    };
  }
  
  /// SDTypeConstraint - This is a discriminated union of constraints,
  /// corresponding to the SDTypeConstraint tablegen class in Target.td.
  struct SDTypeConstraint {
    SDTypeConstraint(Record *R);
    
    unsigned OperandNo;   // The operand # this constraint applies to.
    enum { 
      SDTCisVT, SDTCisPtrTy, SDTCisInt, SDTCisFP, SDTCisSameAs, 
      SDTCisVTSmallerThanOp, SDTCisOpSmallerThanOp, SDTCisIntVectorOfSameSize
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
      struct {
        unsigned BigOperandNum;
      } SDTCisOpSmallerThanOp_Info;
      struct {
        unsigned OtherOperandNum;
      } SDTCisIntVectorOfSameSize_Info;
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
    unsigned Properties;
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
    
    // SelectionDAG node properties.
    enum SDNP { SDNPCommutative, SDNPAssociative, SDNPHasChain,
                SDNPOutFlag, SDNPInFlag, SDNPOptInFlag  };

    /// hasProperty - Return true if this node has the specified property.
    ///
    bool hasProperty(enum SDNP Prop) const { return Properties & (1 << Prop); }

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
    /// The inferred type for this node, or MVT::isUnknown if it hasn't
    /// been determined yet.
    std::vector<unsigned char> Types;
    
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
      : Types(), Operator(Op), Val(0), TransformFn(0),
      Children(Ch) { Types.push_back(MVT::isUnknown); }
    TreePatternNode(Init *val)    // leaf ctor
      : Types(), Operator(0), Val(val), TransformFn(0) {
      Types.push_back(MVT::isUnknown);
    }
    ~TreePatternNode();
    
    const std::string &getName() const { return Name; }
    void setName(const std::string &N) { Name = N; }
    
    bool isLeaf() const { return Val != 0; }
    bool hasTypeSet() const {
      return (Types[0] < MVT::LAST_VALUETYPE) || (Types[0] == MVT::iPTR);
    }
    bool isTypeCompletelyUnknown() const {
      return Types[0] == MVT::isUnknown;
    }
    bool isTypeDynamicallyResolved() const {
      return Types[0] == MVT::iPTR;
    }
    MVT::ValueType getTypeNum(unsigned Num) const {
      assert(hasTypeSet() && "Doesn't have a type yet!");
      assert(Types.size() > Num && "Type num out of range!");
      return (MVT::ValueType)Types[Num];
    }
    unsigned char getExtTypeNum(unsigned Num) const { 
      assert(Types.size() > Num && "Extended type num out of range!");
      return Types[Num]; 
    }
    const std::vector<unsigned char> &getExtTypes() const { return Types; }
    void setTypes(const std::vector<unsigned char> &T) { Types = T; }
    void removeTypes() { Types = std::vector<unsigned char>(1,MVT::isUnknown); }
    
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
    
    /// isIsomorphicTo - Return true if this node is recursively isomorphic to
    /// the specified node.  For this comparison, all of the state of the node
    /// is considered, except for the assigned name.  Nodes with differing names
    /// that are otherwise identical are considered isomorphic.
    bool isIsomorphicTo(const TreePatternNode *N) const;
    
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
    bool ApplyTypeConstraints(TreePattern &TP, bool NotRegisters);
    
    /// UpdateNodeType - Set the node type of N to VT if VT contains
    /// information.  If N already contains a conflicting type, then throw an
    /// exception.  This returns true if any information was updated.
    ///
    bool UpdateNodeType(const std::vector<unsigned char> &ExtVTs,
                        TreePattern &TP);
    bool UpdateNodeType(unsigned char ExtVT, TreePattern &TP) {
      std::vector<unsigned char> ExtVTs(1, ExtVT);
      return UpdateNodeType(ExtVTs, TP);
    }
    
    /// ContainsUnresolvedType - Return true if this tree contains any
    /// unresolved types.
    bool ContainsUnresolvedType() const {
      if (!hasTypeSet() && !isTypeDynamicallyResolved()) return true;
      for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
        if (getChild(i)->ContainsUnresolvedType()) return true;
      return false;
    }
    
    /// canPatternMatch - If it is impossible for this pattern to match on this
    /// target, fill in Reason and return false.  Otherwise, return true.
    bool canPatternMatch(std::string &Reason, DAGISelEmitter &ISE);
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

    /// isInputPattern - True if this is an input pattern, something to match.
    /// False if this is an output pattern, something to emit.
    bool isInputPattern;
  public:
      
    /// TreePattern constructor - Parse the specified DagInits into the
    /// current record.
    TreePattern(Record *TheRec, ListInit *RawPat, bool isInput,
                DAGISelEmitter &ise);
    TreePattern(Record *TheRec, DagInit *Pat, bool isInput,
                DAGISelEmitter &ise);
    TreePattern(Record *TheRec, TreePatternNode *Pat, bool isInput,
                DAGISelEmitter &ise);
        
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
    TreePatternNode *ParseTreePattern(DagInit *DI);
  };


  class DAGInstruction {
    TreePattern *Pattern;
    std::vector<Record*> Results;
    std::vector<Record*> Operands;
    std::vector<Record*> ImpResults;
    std::vector<Record*> ImpOperands;
    TreePatternNode *ResultPattern;
  public:
    DAGInstruction(TreePattern *TP,
                   const std::vector<Record*> &results,
                   const std::vector<Record*> &operands,
                   const std::vector<Record*> &impresults,
                   const std::vector<Record*> &impoperands)
      : Pattern(TP), Results(results), Operands(operands), 
        ImpResults(impresults), ImpOperands(impoperands),
        ResultPattern(0) {}

    TreePattern *getPattern() const { return Pattern; }
    unsigned getNumResults() const { return Results.size(); }
    unsigned getNumOperands() const { return Operands.size(); }
    unsigned getNumImpResults() const { return ImpResults.size(); }
    unsigned getNumImpOperands() const { return ImpOperands.size(); }
    
    void setResultPattern(TreePatternNode *R) { ResultPattern = R; }
    
    Record *getResult(unsigned RN) const {
      assert(RN < Results.size());
      return Results[RN];
    }
    
    Record *getOperand(unsigned ON) const {
      assert(ON < Operands.size());
      return Operands[ON];
    }

    Record *getImpResult(unsigned RN) const {
      assert(RN < ImpResults.size());
      return ImpResults[RN];
    }
    
    Record *getImpOperand(unsigned ON) const {
      assert(ON < ImpOperands.size());
      return ImpOperands[ON];
    }

    TreePatternNode *getResultPattern() const { return ResultPattern; }
  };
  
/// PatternToMatch - Used by DAGISelEmitter to keep tab of patterns processed
/// to produce isel.
struct PatternToMatch {
  PatternToMatch(ListInit *preds,
                 TreePatternNode *src, TreePatternNode *dst,
                 unsigned complexity):
    Predicates(preds), SrcPattern(src), DstPattern(dst),
    AddedComplexity(complexity) {};

  ListInit        *Predicates;  // Top level predicate conditions to match.
  TreePatternNode *SrcPattern;  // Source pattern to match.
  TreePatternNode *DstPattern;  // Resulting pattern.
  unsigned         AddedComplexity; // Add to matching pattern complexity.

  ListInit        *getPredicates() const { return Predicates; }
  TreePatternNode *getSrcPattern() const { return SrcPattern; }
  TreePatternNode *getDstPattern() const { return DstPattern; }
  unsigned         getAddedComplexity() const { return AddedComplexity; }
};

/// DAGISelEmitter - The top-level class which coordinates construction
/// and emission of the instruction selector.
///
class DAGISelEmitter : public TableGenBackend {
private:
  RecordKeeper &Records;
  CodeGenTarget Target;
  std::vector<CodeGenIntrinsic> Intrinsics;
  
  std::map<Record*, SDNodeInfo> SDNodes;
  std::map<Record*, std::pair<Record*, std::string> > SDNodeXForms;
  std::map<Record*, ComplexPattern> ComplexPatterns;
  std::map<Record*, TreePattern*> PatternFragments;
  std::map<Record*, DAGInstruction> Instructions;
  
  // Specific SDNode definitions:
  Record *intrinsic_void_sdnode;
  Record *intrinsic_w_chain_sdnode, *intrinsic_wo_chain_sdnode;
  
  /// PatternsToMatch - All of the things we are matching on the DAG.  The first
  /// value is the pattern to match, the second pattern is the result to
  /// emit.
  std::vector<PatternToMatch> PatternsToMatch;
public:
  DAGISelEmitter(RecordKeeper &R) : Records(R) {}

  // run - Output the isel, returning true on failure.
  void run(std::ostream &OS);
  
  const CodeGenTarget &getTargetInfo() const { return Target; }
  
  Record *getSDNodeNamed(const std::string &Name) const;
  
  const SDNodeInfo &getSDNodeInfo(Record *R) const {
    assert(SDNodes.count(R) && "Unknown node!");
    return SDNodes.find(R)->second;
  }

  const std::pair<Record*, std::string> &getSDNodeTransform(Record *R) const {
    assert(SDNodeXForms.count(R) && "Invalid transform!");
    return SDNodeXForms.find(R)->second;
  }

  const ComplexPattern &getComplexPattern(Record *R) const {
    assert(ComplexPatterns.count(R) && "Unknown addressing mode!");
    return ComplexPatterns.find(R)->second;
  }
  
  const CodeGenIntrinsic &getIntrinsic(Record *R) const {
    for (unsigned i = 0, e = Intrinsics.size(); i != e; ++i)
      if (Intrinsics[i].TheDef == R) return Intrinsics[i];
    assert(0 && "Unknown intrinsic!");
    abort();
  }
  
  const CodeGenIntrinsic &getIntrinsicInfo(unsigned IID) const {
    assert(IID-1 < Intrinsics.size() && "Bad intrinsic ID!");
    return Intrinsics[IID-1];
  }
  
  unsigned getIntrinsicID(Record *R) const {
    for (unsigned i = 0, e = Intrinsics.size(); i != e; ++i)
      if (Intrinsics[i].TheDef == R) return i;
    assert(0 && "Unknown intrinsic!");
    abort();
  }
  
  TreePattern *getPatternFragment(Record *R) const {
    assert(PatternFragments.count(R) && "Invalid pattern fragment request!");
    return PatternFragments.find(R)->second;
  }
  
  const DAGInstruction &getInstruction(Record *R) const {
    assert(Instructions.count(R) && "Unknown instruction!");
    return Instructions.find(R)->second;
  }
  
  Record *get_intrinsic_void_sdnode() const {
    return intrinsic_void_sdnode;
  }
  Record *get_intrinsic_w_chain_sdnode() const {
    return intrinsic_w_chain_sdnode;
  }
  Record *get_intrinsic_wo_chain_sdnode() const {
    return intrinsic_wo_chain_sdnode;
  }

  
private:
  void ParseNodeInfo();
  void ParseNodeTransforms(std::ostream &OS);
  void ParseComplexPatterns();
  void ParsePatternFragments(std::ostream &OS);
  void ParseInstructions();
  void ParsePatterns();
  void GenerateVariants();
  void FindPatternInputsAndOutputs(TreePattern *I, TreePatternNode *Pat,
                                   std::map<std::string,
                                            TreePatternNode*> &InstInputs,
                                   std::map<std::string,
                                            TreePatternNode*> &InstResults,
                                   std::vector<Record*> &InstImpInputs,
                                   std::vector<Record*> &InstImpResults);
  void GenerateCodeForPattern(PatternToMatch &Pattern,
                      std::vector<std::pair<bool, std::string> > &GeneratedCode,
                     std::set<std::pair<unsigned, std::string> > &GeneratedDecl,
                              std::vector<std::string> &TargetOpcodes,
                              std::vector<std::string> &TargetVTs);
  void EmitPatterns(std::vector<std::pair<PatternToMatch*, 
                    std::vector<std::pair<bool, std::string> > > > &Patterns, 
                    unsigned Indent, std::ostream &OS);
  void EmitInstructionSelector(std::ostream &OS);
};

} // End llvm namespace

#endif
