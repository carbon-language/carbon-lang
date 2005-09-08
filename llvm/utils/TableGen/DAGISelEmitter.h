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
  class Init;
  class DagInit;
  class TreePattern;
  class DAGISelEmitter;
  
  /// SDNodeInfo - One of these records is created for each SDNode instance in
  /// the target .td file.  This represents the various dag nodes we will be
  /// processing.
  class SDNodeInfo {
    Record *Def;
    std::string EnumName;
    std::string SDClassName;
  public:
    SDNodeInfo(Record *R);  // Parse the specified record.
    
    Record *getRecord() const { return Def; }
    const std::string &getEnumName() const { return EnumName; }
    const std::string &getSDClassName() const { return SDClassName; }
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
    
    std::vector<TreePatternNode*> Children;
  public:
    TreePatternNode(Record *Op, const std::vector<TreePatternNode*> &Ch) 
      : Ty(MVT::LAST_VALUETYPE), Operator(Op), Val(0), Children(Ch) {}
    TreePatternNode(Init *val)    // leaf ctor
      : Ty(MVT::LAST_VALUETYPE), Operator(0), Val(val) {}
    ~TreePatternNode();
    
    const std::string &getName() const { return Name; }
    void setName(const std::string &N) { Name = N; }
    
    bool isLeaf() const { return Val != 0; }
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
    
    void print(std::ostream &OS) const;
    void dump() const;
    
  public:   // Higher level manipulation routines.

    /// clone - Return a new copy of this tree.
    ///
    TreePatternNode *clone() const;
    
    void SubstituteFormalArguments(std::map<std::string,
                                            TreePatternNode*> &ArgMap);

    /// InlinePatternFragments - If this pattern refers to any pattern
    /// fragments, inline them into place, giving us a pattern without any
    /// PatFrag references.
    TreePatternNode *InlinePatternFragments(TreePattern &TP);
        
  };
  
  
  /// TreePattern - Represent a pattern of one form or another.  Currently, two
  /// types of patterns are possible: Instructions and PatFrags.
  ///
  class TreePattern {
  public:
    enum PatternType {
      PatFrag, Instruction
    };
  private:
    /// PTy - The type of pattern this is.
    ///
    PatternType PTy;
    
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
    TreePattern(PatternType pty, Record *TheRec,
                const std::vector<DagInit *> &RawPat, DAGISelEmitter &ise);
        
    /// getPatternType - Return what flavor of Record this pattern originated from
    ///
    PatternType getPatternType() const { return PTy; }
    
    /// getTrees - Return the tree patterns which corresponds to this pattern.
    ///
    const std::vector<TreePatternNode*> &getTrees() const { return Trees; }
        
    /// getRecord - Return the actual TableGen record corresponding to this
    /// pattern.
    ///
    Record *getRecord() const { return TheRecord; }
    
    unsigned getNumArgs() const { return Args.size(); }
    const std::string &getArgName(unsigned i) const {
      assert(i < Args.size() && "Argument reference out of range!");
      return Args[i];
    }
    
    DAGISelEmitter &getDAGISelEmitter() const { return ISE; }

    /// InlinePatternFragments - If this pattern refers to any pattern
    /// fragments, inline them into place, giving us a pattern without any
    /// PatFrag references.
    void InlinePatternFragments() {
      for (unsigned i = 0, e = Trees.size(); i != e; ++i)
        Trees[i] = Trees[i]->InlinePatternFragments(*this);
    }
    
    /// error - Throw an exception, prefixing it with information about this
    /// pattern.
    void error(const std::string &Msg) const;
    
    void print(std::ostream &OS) const;
    void dump() const;
    
  private:
    MVT::ValueType getIntrinsicType(Record *R) const;
    TreePatternNode *ParseTreePattern(DagInit *DI);
  };
  
  
  
/// InstrSelectorEmitter - The top-level class which coordinates construction
/// and emission of the instruction selector.
///
class DAGISelEmitter : public TableGenBackend {
  RecordKeeper &Records;
  CodeGenTarget Target;

  std::map<Record*, SDNodeInfo> SDNodes;
  std::map<Record*, TreePattern*> PatternFragments;
  std::vector<TreePattern*> Instructions;
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
  
private:
  void ParseNodeInfo();
  void ParseAndResolvePatternFragments(std::ostream &OS);
  void ParseAndResolveInstructions();
  void EmitInstructionSelector(std::ostream &OS);
};

} // End llvm namespace

#endif
