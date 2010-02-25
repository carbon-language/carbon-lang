//===- DAGISelMatcher.h - Representation of DAG pattern matcher -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef TBLGEN_DAGISELMATCHER_H
#define TBLGEN_DAGISELMATCHER_H

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace llvm {
  class CodeGenDAGPatterns;
  class MatcherNode;
  class PatternToMatch;
  class raw_ostream;
  class ComplexPattern;
  class Record;

MatcherNode *ConvertPatternToMatcher(const PatternToMatch &Pattern,
                                     const CodeGenDAGPatterns &CGP);
MatcherNode *OptimizeMatcher(MatcherNode *Matcher);
void EmitMatcherTable(const MatcherNode *Matcher, raw_ostream &OS);

  
/// MatcherNode - Base class for all the the DAG ISel Matcher representation
/// nodes.
class MatcherNode {
  // The next matcher node that is executed after this one.  Null if this is the
  // last stage of a match.
  OwningPtr<MatcherNode> Next;
public:
  enum KindTy {
    // Matcher state manipulation.
    Scope,                // Push a checking scope.
    RecordNode,           // Record the current node.
    RecordChild,          // Record a child of the current node.
    RecordMemRef,         // Record the memref in the current node.
    CaptureFlagInput,     // If the current node has an input flag, save it.
    MoveChild,            // Move current node to specified child.
    MoveParent,           // Move current node to parent.
    
    // Predicate checking.
    CheckSame,            // Fail if not same as prev match.
    CheckPatternPredicate,
    CheckPredicate,       // Fail if node predicate fails.
    CheckOpcode,          // Fail if not opcode.
    CheckMultiOpcode,     // Fail if not in opcode list.
    CheckType,            // Fail if not correct type.
    CheckChildType,       // Fail if child has wrong type.
    CheckInteger,         // Fail if wrong val.
    CheckCondCode,        // Fail if not condcode.
    CheckValueType,
    CheckComplexPat,
    CheckAndImm,
    CheckOrImm,
    CheckFoldableChainNode,
    CheckChainCompatible,
    
    // Node creation/emisssion.
    EmitInteger,          // Create a TargetConstant
    EmitStringInteger,    // Create a TargetConstant from a string.
    EmitRegister,         // Create a register.
    EmitConvertToTarget,  // Convert a imm/fpimm to target imm/fpimm
    EmitMergeInputChains, // Merge together a chains for an input.
    EmitCopyToReg,        // Emit a copytoreg into a physreg.
    EmitNode,             // Create a DAG node
    EmitNodeXForm,        // Run a SDNodeXForm
    MarkFlagResults,      // Indicate which interior nodes have flag results.
    CompleteMatch         // Finish a match and update the results.
  };
  const KindTy Kind;

protected:
  MatcherNode(KindTy K) : Kind(K) {}
public:
  virtual ~MatcherNode() {}
  
  KindTy getKind() const { return Kind; }

  MatcherNode *getNext() { return Next.get(); }
  const MatcherNode *getNext() const { return Next.get(); }
  void setNext(MatcherNode *C) { Next.reset(C); }
  MatcherNode *takeNext() { return Next.take(); }

  OwningPtr<MatcherNode> &getNextPtr() { return Next; }
  
  static inline bool classof(const MatcherNode *) { return true; }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const = 0;
  void dump() const;
protected:
  void printNext(raw_ostream &OS, unsigned indent) const;
};
  
/// ScopeMatcherNode - This pushes a failure scope on the stack and evaluates
/// 'Check'.  If 'Check' fails to match, it pops its scope and continues on to
/// 'Next'.
class ScopeMatcherNode : public MatcherNode {
  OwningPtr<MatcherNode> Check;
public:
  ScopeMatcherNode(MatcherNode *check = 0, MatcherNode *next = 0)
    : MatcherNode(Scope), Check(check) {
    setNext(next);
  }
  
  MatcherNode *getCheck() { return Check.get(); }
  const MatcherNode *getCheck() const { return Check.get(); }
  void setCheck(MatcherNode *N) { Check.reset(N); }
  OwningPtr<MatcherNode> &getCheckPtr() { return Check; }

  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == Scope;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};

/// RecordMatcherNode - Save the current node in the operand list.
class RecordMatcherNode : public MatcherNode {
  /// WhatFor - This is a string indicating why we're recording this.  This
  /// should only be used for comment generation not anything semantic.
  std::string WhatFor;
public:
  RecordMatcherNode(const std::string &whatfor)
    : MatcherNode(RecordNode), WhatFor(whatfor) {}
  
  const std::string &getWhatFor() const { return WhatFor; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == RecordNode;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// RecordChildMatcherNode - Save a numbered child of the current node, or fail
/// the match if it doesn't exist.  This is logically equivalent to:
///    MoveChild N + RecordNode + MoveParent.
class RecordChildMatcherNode : public MatcherNode {
  unsigned ChildNo;
  
  /// WhatFor - This is a string indicating why we're recording this.  This
  /// should only be used for comment generation not anything semantic.
  std::string WhatFor;
public:
  RecordChildMatcherNode(unsigned childno, const std::string &whatfor)
  : MatcherNode(RecordChild), ChildNo(childno), WhatFor(whatfor) {}
  
  unsigned getChildNo() const { return ChildNo; }
  const std::string &getWhatFor() const { return WhatFor; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == RecordChild;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// RecordMemRefMatcherNode - Save the current node's memref.
class RecordMemRefMatcherNode : public MatcherNode {
public:
  RecordMemRefMatcherNode() : MatcherNode(RecordMemRef) {}
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == RecordMemRef;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};

  
/// CaptureFlagInputMatcherNode - If the current record has a flag input, record
/// it so that it is used as an input to the generated code.
class CaptureFlagInputMatcherNode : public MatcherNode {
public:
  CaptureFlagInputMatcherNode()
    : MatcherNode(CaptureFlagInput) {}
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CaptureFlagInput;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// MoveChildMatcherNode - This tells the interpreter to move into the
/// specified child node.
class MoveChildMatcherNode : public MatcherNode {
  unsigned ChildNo;
public:
  MoveChildMatcherNode(unsigned childNo)
  : MatcherNode(MoveChild), ChildNo(childNo) {}
  
  unsigned getChildNo() const { return ChildNo; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == MoveChild;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// MoveParentMatcherNode - This tells the interpreter to move to the parent
/// of the current node.
class MoveParentMatcherNode : public MatcherNode {
public:
  MoveParentMatcherNode()
  : MatcherNode(MoveParent) {}
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == MoveParent;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};

/// CheckSameMatcherNode - This checks to see if this node is exactly the same
/// node as the specified match that was recorded with 'Record'.  This is used
/// when patterns have the same name in them, like '(mul GPR:$in, GPR:$in)'.
class CheckSameMatcherNode : public MatcherNode {
  unsigned MatchNumber;
public:
  CheckSameMatcherNode(unsigned matchnumber)
  : MatcherNode(CheckSame), MatchNumber(matchnumber) {}
  
  unsigned getMatchNumber() const { return MatchNumber; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckSame;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// CheckPatternPredicateMatcherNode - This checks the target-specific predicate
/// to see if the entire pattern is capable of matching.  This predicate does
/// not take a node as input.  This is used for subtarget feature checks etc.
class CheckPatternPredicateMatcherNode : public MatcherNode {
  std::string Predicate;
public:
  CheckPatternPredicateMatcherNode(StringRef predicate)
  : MatcherNode(CheckPatternPredicate), Predicate(predicate) {}
  
  StringRef getPredicate() const { return Predicate; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckPatternPredicate;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// CheckPredicateMatcherNode - This checks the target-specific predicate to
/// see if the node is acceptable.
class CheckPredicateMatcherNode : public MatcherNode {
  StringRef PredName;
public:
  CheckPredicateMatcherNode(StringRef predname)
    : MatcherNode(CheckPredicate), PredName(predname) {}
  
  StringRef getPredicateName() const { return PredName; }

  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckPredicate;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
  
/// CheckOpcodeMatcherNode - This checks to see if the current node has the
/// specified opcode, if not it fails to match.
class CheckOpcodeMatcherNode : public MatcherNode {
  StringRef OpcodeName;
public:
  CheckOpcodeMatcherNode(StringRef opcodename)
    : MatcherNode(CheckOpcode), OpcodeName(opcodename) {}
  
  StringRef getOpcodeName() const { return OpcodeName; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckOpcode;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// CheckMultiOpcodeMatcherNode - This checks to see if the current node has one
/// of the specified opcode, if not it fails to match.
class CheckMultiOpcodeMatcherNode : public MatcherNode {
  SmallVector<StringRef, 4> OpcodeNames;
public:
  CheckMultiOpcodeMatcherNode(const StringRef *opcodes, unsigned numops)
  : MatcherNode(CheckMultiOpcode), OpcodeNames(opcodes, opcodes+numops) {}
  
  unsigned getNumOpcodeNames() const { return OpcodeNames.size(); }
  StringRef getOpcodeName(unsigned i) const { return OpcodeNames[i]; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckMultiOpcode;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
  
  
/// CheckTypeMatcherNode - This checks to see if the current node has the
/// specified type, if not it fails to match.
class CheckTypeMatcherNode : public MatcherNode {
  MVT::SimpleValueType Type;
public:
  CheckTypeMatcherNode(MVT::SimpleValueType type)
    : MatcherNode(CheckType), Type(type) {}
  
  MVT::SimpleValueType getType() const { return Type; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckType;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// CheckChildTypeMatcherNode - This checks to see if a child node has the
/// specified type, if not it fails to match.
class CheckChildTypeMatcherNode : public MatcherNode {
  unsigned ChildNo;
  MVT::SimpleValueType Type;
public:
  CheckChildTypeMatcherNode(unsigned childno, MVT::SimpleValueType type)
    : MatcherNode(CheckChildType), ChildNo(childno), Type(type) {}
  
  unsigned getChildNo() const { return ChildNo; }
  MVT::SimpleValueType getType() const { return Type; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckChildType;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  

/// CheckIntegerMatcherNode - This checks to see if the current node is a
/// ConstantSDNode with the specified integer value, if not it fails to match.
class CheckIntegerMatcherNode : public MatcherNode {
  int64_t Value;
public:
  CheckIntegerMatcherNode(int64_t value)
    : MatcherNode(CheckInteger), Value(value) {}
  
  int64_t getValue() const { return Value; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckInteger;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// CheckCondCodeMatcherNode - This checks to see if the current node is a
/// CondCodeSDNode with the specified condition, if not it fails to match.
class CheckCondCodeMatcherNode : public MatcherNode {
  StringRef CondCodeName;
public:
  CheckCondCodeMatcherNode(StringRef condcodename)
  : MatcherNode(CheckCondCode), CondCodeName(condcodename) {}
  
  StringRef getCondCodeName() const { return CondCodeName; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckCondCode;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// CheckValueTypeMatcherNode - This checks to see if the current node is a
/// VTSDNode with the specified type, if not it fails to match.
class CheckValueTypeMatcherNode : public MatcherNode {
  StringRef TypeName;
public:
  CheckValueTypeMatcherNode(StringRef type_name)
  : MatcherNode(CheckValueType), TypeName(type_name) {}
  
  StringRef getTypeName() const { return TypeName; }

  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckValueType;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
  
  
/// CheckComplexPatMatcherNode - This node runs the specified ComplexPattern on
/// the current node.
class CheckComplexPatMatcherNode : public MatcherNode {
  const ComplexPattern &Pattern;
public:
  CheckComplexPatMatcherNode(const ComplexPattern &pattern)
  : MatcherNode(CheckComplexPat), Pattern(pattern) {}
  
  const ComplexPattern &getPattern() const { return Pattern; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckComplexPat;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// CheckAndImmMatcherNode - This checks to see if the current node is an 'and'
/// with something equivalent to the specified immediate.
class CheckAndImmMatcherNode : public MatcherNode {
  int64_t Value;
public:
  CheckAndImmMatcherNode(int64_t value)
  : MatcherNode(CheckAndImm), Value(value) {}
  
  int64_t getValue() const { return Value; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckAndImm;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};

/// CheckOrImmMatcherNode - This checks to see if the current node is an 'and'
/// with something equivalent to the specified immediate.
class CheckOrImmMatcherNode : public MatcherNode {
  int64_t Value;
public:
  CheckOrImmMatcherNode(int64_t value)
    : MatcherNode(CheckOrImm), Value(value) {}
  
  int64_t getValue() const { return Value; }

  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckOrImm;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};

/// CheckFoldableChainNodeMatcherNode - This checks to see if the current node
/// (which defines a chain operand) is safe to fold into a larger pattern.
class CheckFoldableChainNodeMatcherNode : public MatcherNode {
public:
  CheckFoldableChainNodeMatcherNode()
    : MatcherNode(CheckFoldableChainNode) {}
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckFoldableChainNode;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};

/// CheckChainCompatibleMatcherNode - Verify that the current node's chain
/// operand is 'compatible' with the specified recorded node's.
class CheckChainCompatibleMatcherNode : public MatcherNode {
  unsigned PreviousOp;
public:
  CheckChainCompatibleMatcherNode(unsigned previousop)
    : MatcherNode(CheckChainCompatible), PreviousOp(previousop) {}
  
  unsigned getPreviousOp() const { return PreviousOp; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CheckChainCompatible;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// EmitIntegerMatcherNode - This creates a new TargetConstant.
class EmitIntegerMatcherNode : public MatcherNode {
  int64_t Val;
  MVT::SimpleValueType VT;
public:
  EmitIntegerMatcherNode(int64_t val, MVT::SimpleValueType vt)
  : MatcherNode(EmitInteger), Val(val), VT(vt) {}
  
  int64_t getValue() const { return Val; }
  MVT::SimpleValueType getVT() const { return VT; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == EmitInteger;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};

/// EmitStringIntegerMatcherNode - A target constant whose value is represented
/// by a string.
class EmitStringIntegerMatcherNode : public MatcherNode {
  std::string Val;
  MVT::SimpleValueType VT;
public:
  EmitStringIntegerMatcherNode(const std::string &val, MVT::SimpleValueType vt)
    : MatcherNode(EmitStringInteger), Val(val), VT(vt) {}
  
  const std::string &getValue() const { return Val; }
  MVT::SimpleValueType getVT() const { return VT; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == EmitStringInteger;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// EmitRegisterMatcherNode - This creates a new TargetConstant.
class EmitRegisterMatcherNode : public MatcherNode {
  /// Reg - The def for the register that we're emitting.  If this is null, then
  /// this is a reference to zero_reg.
  Record *Reg;
  MVT::SimpleValueType VT;
public:
  EmitRegisterMatcherNode(Record *reg, MVT::SimpleValueType vt)
    : MatcherNode(EmitRegister), Reg(reg), VT(vt) {}
  
  Record *getReg() const { return Reg; }
  MVT::SimpleValueType getVT() const { return VT; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == EmitRegister;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};

/// EmitConvertToTargetMatcherNode - Emit an operation that reads a specified
/// recorded node and converts it from being a ISD::Constant to
/// ISD::TargetConstant, likewise for ConstantFP.
class EmitConvertToTargetMatcherNode : public MatcherNode {
  unsigned Slot;
public:
  EmitConvertToTargetMatcherNode(unsigned slot)
    : MatcherNode(EmitConvertToTarget), Slot(slot) {}
  
  unsigned getSlot() const { return Slot; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == EmitConvertToTarget;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// EmitMergeInputChainsMatcherNode - Emit a node that merges a list of input
/// chains together with a token factor.  The list of nodes are the nodes in the
/// matched pattern that have chain input/outputs.  This node adds all input
/// chains of these nodes if they are not themselves a node in the pattern.
class EmitMergeInputChainsMatcherNode : public MatcherNode {
  SmallVector<unsigned, 3> ChainNodes;
public:
  EmitMergeInputChainsMatcherNode(const unsigned *nodes, unsigned NumNodes)
  : MatcherNode(EmitMergeInputChains), ChainNodes(nodes, nodes+NumNodes) {}
  
  unsigned getNumNodes() const { return ChainNodes.size(); }
  
  unsigned getNode(unsigned i) const {
    assert(i < ChainNodes.size());
    return ChainNodes[i];
  }  
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == EmitMergeInputChains;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// EmitCopyToRegMatcherNode - Emit a CopyToReg node from a value to a physreg,
/// pushing the chain and flag results.
///
class EmitCopyToRegMatcherNode : public MatcherNode {
  unsigned SrcSlot; // Value to copy into the physreg.
  Record *DestPhysReg;
public:
  EmitCopyToRegMatcherNode(unsigned srcSlot, Record *destPhysReg)
  : MatcherNode(EmitCopyToReg), SrcSlot(srcSlot), DestPhysReg(destPhysReg) {}
  
  unsigned getSrcSlot() const { return SrcSlot; }
  Record *getDestPhysReg() const { return DestPhysReg; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == EmitCopyToReg;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
    
  
/// EmitNodeXFormMatcherNode - Emit an operation that runs an SDNodeXForm on a
/// recorded node and records the result.
class EmitNodeXFormMatcherNode : public MatcherNode {
  unsigned Slot;
  Record *NodeXForm;
public:
  EmitNodeXFormMatcherNode(unsigned slot, Record *nodeXForm)
  : MatcherNode(EmitNodeXForm), Slot(slot), NodeXForm(nodeXForm) {}
  
  unsigned getSlot() const { return Slot; }
  Record *getNodeXForm() const { return NodeXForm; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == EmitNodeXForm;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// EmitNodeMatcherNode - This signals a successful match and generates a node.
class EmitNodeMatcherNode : public MatcherNode {
  std::string OpcodeName;
  const SmallVector<MVT::SimpleValueType, 3> VTs;
  const SmallVector<unsigned, 6> Operands;
  bool HasChain, HasFlag, HasMemRefs;
  
  /// NumFixedArityOperands - If this is a fixed arity node, this is set to -1.
  /// If this is a varidic node, this is set to the number of fixed arity
  /// operands in the root of the pattern.  The rest are appended to this node.
  int NumFixedArityOperands;
public:
  EmitNodeMatcherNode(const std::string &opcodeName,
                      const MVT::SimpleValueType *vts, unsigned numvts,
                      const unsigned *operands, unsigned numops,
                      bool hasChain, bool hasFlag, bool hasmemrefs,
                      int numfixedarityoperands)
    : MatcherNode(EmitNode), OpcodeName(opcodeName),
      VTs(vts, vts+numvts), Operands(operands, operands+numops),
      HasChain(hasChain), HasFlag(hasFlag), HasMemRefs(hasmemrefs),
      NumFixedArityOperands(numfixedarityoperands) {}
  
  const std::string &getOpcodeName() const { return OpcodeName; }
  
  unsigned getNumVTs() const { return VTs.size(); }
  MVT::SimpleValueType getVT(unsigned i) const {
    assert(i < VTs.size());
    return VTs[i];
  }
  
  unsigned getNumOperands() const { return Operands.size(); }
  unsigned getOperand(unsigned i) const {
    assert(i < Operands.size());
    return Operands[i];
  }  
  
  bool hasChain() const { return HasChain; }
  bool hasFlag() const { return HasFlag; }
  bool hasMemRefs() const { return HasMemRefs; }
  int getNumFixedArityOperands() const { return NumFixedArityOperands; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == EmitNode;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
  
/// MarkFlagResultsMatcherNode - This node indicates which non-root nodes in the
/// pattern produce flags.  This allows CompleteMatchMatcherNode to update them
/// with the output flag of the resultant code.
class MarkFlagResultsMatcherNode : public MatcherNode {
  SmallVector<unsigned, 3> FlagResultNodes;
public:
  MarkFlagResultsMatcherNode(const unsigned *nodes, unsigned NumNodes)
  : MatcherNode(MarkFlagResults), FlagResultNodes(nodes, nodes+NumNodes) {}
  
  unsigned getNumNodes() const { return FlagResultNodes.size(); }
  
  unsigned getNode(unsigned i) const {
    assert(i < FlagResultNodes.size());
    return FlagResultNodes[i];
  }  
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == MarkFlagResults;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};

/// CompleteMatchMatcherNode - Complete a match by replacing the results of the
/// pattern with the newly generated nodes.  This also prints a comment
/// indicating the source and dest patterns.
class CompleteMatchMatcherNode : public MatcherNode {
  SmallVector<unsigned, 2> Results;
  const PatternToMatch &Pattern;
public:
  CompleteMatchMatcherNode(const unsigned *results, unsigned numresults,
                           const PatternToMatch &pattern)
  : MatcherNode(CompleteMatch), Results(results, results+numresults),
    Pattern(pattern) {}

  unsigned getNumResults() const { return Results.size(); }
  unsigned getResult(unsigned R) const { return Results[R]; }
  const PatternToMatch &getPattern() const { return Pattern; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == CompleteMatch;
  }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};
 
  
} // end namespace llvm

#endif
