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
#include "llvm/Support/Casting.h"

namespace llvm {
  class CodeGenDAGPatterns;
  class MatcherNode;
  class PatternToMatch;
  class raw_ostream;
  class ComplexPattern;

MatcherNode *ConvertPatternToMatcher(const PatternToMatch &Pattern,
                                     const CodeGenDAGPatterns &CGP);

void EmitMatcherTable(const MatcherNode *Matcher, raw_ostream &OS);

  
/// MatcherNode - Base class for all the the DAG ISel Matcher representation
/// nodes.
class MatcherNode {
  // The next matcher node that is executed after this one.  Null if this is the
  // last stage of a match.
  OwningPtr<MatcherNode> Next;
public:
  enum KindTy {
    EmitNode,
    Push,           // [Push, Dest0, Dest1, Dest2, Dest3]
    Record,         // [Record]
    MoveChild,      // [MoveChild, Child#]
    MoveParent,     // [MoveParent]
    
    CheckSame,      // [CheckSame, N]         Fail if not same as prev match.
    CheckPatternPredicate,
    CheckPredicate, // [CheckPredicate, P]    Fail if predicate fails.
    CheckOpcode,    // [CheckOpcode, Opcode]  Fail if not opcode.
    CheckType,      // [CheckType, MVT]       Fail if not correct type.
    CheckInteger,   // [CheckInteger, int0,int1,int2,...int7] Fail if wrong val.
    CheckCondCode,  // [CheckCondCode, CondCode] Fail if not condcode.
    CheckValueType,
    CheckComplexPat,
    CheckAndImm,
    CheckOrImm,
    CheckFoldableChainNode,
    CheckChainCompatible
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
  
  static inline bool classof(const MatcherNode *) { return true; }
  
  virtual void print(raw_ostream &OS, unsigned indent = 0) const = 0;
  void dump() const;
protected:
  void printNext(raw_ostream &OS, unsigned indent) const;
};
  
/// EmitNodeMatcherNode - This signals a successful match and generates a node.
class EmitNodeMatcherNode : public MatcherNode {
  const PatternToMatch &Pattern;
public:
  EmitNodeMatcherNode(const PatternToMatch &pattern)
    : MatcherNode(EmitNode), Pattern(pattern) {}

  const PatternToMatch &getPattern() const { return Pattern; }

  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == EmitNode;
  }

  virtual void print(raw_ostream &OS, unsigned indent = 0) const;
};


/// PushMatcherNode - This pushes a failure scope on the stack and evaluates
/// 'Next'.  If 'Next' fails to match, it pops its scope and attempts to
/// match 'Failure'.
class PushMatcherNode : public MatcherNode {
  OwningPtr<MatcherNode> Failure;
public:
  PushMatcherNode(MatcherNode *next = 0, MatcherNode *failure = 0)
    : MatcherNode(Push), Failure(failure) {
    setNext(next);
  }
  
  MatcherNode *getFailure() { return Failure.get(); }
  const MatcherNode *getFailure() const { return Failure.get(); }
  void setFailure(MatcherNode *N) { Failure.reset(N); }

  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == Push;
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
    : MatcherNode(Record), WhatFor(whatfor) {}
  
  const std::string &getWhatFor() const { return WhatFor; }
  
  static inline bool classof(const MatcherNode *N) {
    return N->getKind() == Record;
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
  
  

} // end namespace llvm

#endif
