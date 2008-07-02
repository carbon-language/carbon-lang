//===- DAGISelEmitter.cpp - Generate an instruction selector --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a DAG instruction selector.
//
//===----------------------------------------------------------------------===//

#include "DAGISelEmitter.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Streams.h"
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
// DAGISelEmitter Helper methods
//

/// NodeIsComplexPattern - return true if N is a leaf node and a subclass of
/// ComplexPattern.
static bool NodeIsComplexPattern(TreePatternNode *N) {
  return (N->isLeaf() &&
          dynamic_cast<DefInit*>(N->getLeafValue()) &&
          static_cast<DefInit*>(N->getLeafValue())->getDef()->
          isSubClassOf("ComplexPattern"));
}

/// NodeGetComplexPattern - return the pointer to the ComplexPattern if N
/// is a leaf node and a subclass of ComplexPattern, else it returns NULL.
static const ComplexPattern *NodeGetComplexPattern(TreePatternNode *N,
                                                   CodeGenDAGPatterns &CGP) {
  if (N->isLeaf() &&
      dynamic_cast<DefInit*>(N->getLeafValue()) &&
      static_cast<DefInit*>(N->getLeafValue())->getDef()->
      isSubClassOf("ComplexPattern")) {
    return &CGP.getComplexPattern(static_cast<DefInit*>(N->getLeafValue())
                                       ->getDef());
  }
  return NULL;
}

/// getPatternSize - Return the 'size' of this pattern.  We want to match large
/// patterns before small ones.  This is used to determine the size of a
/// pattern.
static unsigned getPatternSize(TreePatternNode *P, CodeGenDAGPatterns &CGP) {
  assert((EMVT::isExtIntegerInVTs(P->getExtTypes()) ||
          EMVT::isExtFloatingPointInVTs(P->getExtTypes()) ||
          P->getExtTypeNum(0) == MVT::isVoid ||
          P->getExtTypeNum(0) == MVT::Flag ||
          P->getExtTypeNum(0) == MVT::iPTR) && 
         "Not a valid pattern node to size!");
  unsigned Size = 3;  // The node itself.
  // If the root node is a ConstantSDNode, increases its size.
  // e.g. (set R32:$dst, 0).
  if (P->isLeaf() && dynamic_cast<IntInit*>(P->getLeafValue()))
    Size += 2;

  // FIXME: This is a hack to statically increase the priority of patterns
  // which maps a sub-dag to a complex pattern. e.g. favors LEA over ADD.
  // Later we can allow complexity / cost for each pattern to be (optionally)
  // specified. To get best possible pattern match we'll need to dynamically
  // calculate the complexity of all patterns a dag can potentially map to.
  const ComplexPattern *AM = NodeGetComplexPattern(P, CGP);
  if (AM)
    Size += AM->getNumOperands() * 3;

  // If this node has some predicate function that must match, it adds to the
  // complexity of this node.
  if (!P->getPredicateFn().empty())
    ++Size;
  
  // Count children in the count if they are also nodes.
  for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = P->getChild(i);
    if (!Child->isLeaf() && Child->getExtTypeNum(0) != MVT::Other)
      Size += getPatternSize(Child, CGP);
    else if (Child->isLeaf()) {
      if (dynamic_cast<IntInit*>(Child->getLeafValue())) 
        Size += 5;  // Matches a ConstantSDNode (+3) and a specific value (+2).
      else if (NodeIsComplexPattern(Child))
        Size += getPatternSize(Child, CGP);
      else if (!Child->getPredicateFn().empty())
        ++Size;
    }
  }
  
  return Size;
}

/// getResultPatternCost - Compute the number of instructions for this pattern.
/// This is a temporary hack.  We should really include the instruction
/// latencies in this calculation.
static unsigned getResultPatternCost(TreePatternNode *P,
                                     CodeGenDAGPatterns &CGP) {
  if (P->isLeaf()) return 0;
  
  unsigned Cost = 0;
  Record *Op = P->getOperator();
  if (Op->isSubClassOf("Instruction")) {
    Cost++;
    CodeGenInstruction &II = CGP.getTargetInfo().getInstruction(Op->getName());
    if (II.usesCustomDAGSchedInserter)
      Cost += 10;
  }
  for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i)
    Cost += getResultPatternCost(P->getChild(i), CGP);
  return Cost;
}

/// getResultPatternCodeSize - Compute the code size of instructions for this
/// pattern.
static unsigned getResultPatternSize(TreePatternNode *P, 
                                     CodeGenDAGPatterns &CGP) {
  if (P->isLeaf()) return 0;

  unsigned Cost = 0;
  Record *Op = P->getOperator();
  if (Op->isSubClassOf("Instruction")) {
    Cost += Op->getValueAsInt("CodeSize");
  }
  for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i)
    Cost += getResultPatternSize(P->getChild(i), CGP);
  return Cost;
}

// PatternSortingPredicate - return true if we prefer to match LHS before RHS.
// In particular, we want to match maximal patterns first and lowest cost within
// a particular complexity first.
struct PatternSortingPredicate {
  PatternSortingPredicate(CodeGenDAGPatterns &cgp) : CGP(cgp) {}
  CodeGenDAGPatterns &CGP;

  bool operator()(const PatternToMatch *LHS,
                  const PatternToMatch *RHS) {
    unsigned LHSSize = getPatternSize(LHS->getSrcPattern(), CGP);
    unsigned RHSSize = getPatternSize(RHS->getSrcPattern(), CGP);
    LHSSize += LHS->getAddedComplexity();
    RHSSize += RHS->getAddedComplexity();
    if (LHSSize > RHSSize) return true;   // LHS -> bigger -> less cost
    if (LHSSize < RHSSize) return false;
    
    // If the patterns have equal complexity, compare generated instruction cost
    unsigned LHSCost = getResultPatternCost(LHS->getDstPattern(), CGP);
    unsigned RHSCost = getResultPatternCost(RHS->getDstPattern(), CGP);
    if (LHSCost < RHSCost) return true;
    if (LHSCost > RHSCost) return false;

    return getResultPatternSize(LHS->getDstPattern(), CGP) <
      getResultPatternSize(RHS->getDstPattern(), CGP);
  }
};

/// getRegisterValueType - Look up and return the first ValueType of specified 
/// RegisterClass record
static MVT::SimpleValueType getRegisterValueType(Record *R, const CodeGenTarget &T) {
  if (const CodeGenRegisterClass *RC = T.getRegisterClassForRegister(R))
    return RC->getValueTypeNum(0);
  return MVT::Other;
}


/// RemoveAllTypes - A quick recursive walk over a pattern which removes all
/// type information from it.
static void RemoveAllTypes(TreePatternNode *N) {
  N->removeTypes();
  if (!N->isLeaf())
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
      RemoveAllTypes(N->getChild(i));
}

/// NodeHasProperty - return true if TreePatternNode has the specified
/// property.
static bool NodeHasProperty(TreePatternNode *N, SDNP Property,
                            CodeGenDAGPatterns &CGP) {
  if (N->isLeaf()) {
    const ComplexPattern *CP = NodeGetComplexPattern(N, CGP);
    if (CP)
      return CP->hasProperty(Property);
    return false;
  }
  Record *Operator = N->getOperator();
  if (!Operator->isSubClassOf("SDNode")) return false;

  return CGP.getSDNodeInfo(Operator).hasProperty(Property);
}

static bool PatternHasProperty(TreePatternNode *N, SDNP Property,
                               CodeGenDAGPatterns &CGP) {
  if (NodeHasProperty(N, Property, CGP))
    return true;

  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = N->getChild(i);
    if (PatternHasProperty(Child, Property, CGP))
      return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Node Transformation emitter implementation.
//
void DAGISelEmitter::EmitNodeTransforms(std::ostream &OS) {
  // Walk the pattern fragments, adding them to a map, which sorts them by
  // name.
  typedef std::map<std::string, CodeGenDAGPatterns::NodeXForm> NXsByNameTy;
  NXsByNameTy NXsByName;

  for (CodeGenDAGPatterns::nx_iterator I = CGP.nx_begin(), E = CGP.nx_end();
       I != E; ++I)
    NXsByName.insert(std::make_pair(I->first->getName(), I->second));
  
  OS << "\n// Node transformations.\n";
  
  for (NXsByNameTy::iterator I = NXsByName.begin(), E = NXsByName.end();
       I != E; ++I) {
    Record *SDNode = I->second.first;
    std::string Code = I->second.second;
    
    if (Code.empty()) continue;  // Empty code?  Skip it.
    
    std::string ClassName = CGP.getSDNodeInfo(SDNode).getSDClassName();
    const char *C2 = ClassName == "SDNode" ? "N" : "inN";
    
    OS << "inline SDOperand Transform_" << I->first << "(SDNode *" << C2
       << ") {\n";
    if (ClassName != "SDNode")
      OS << "  " << ClassName << " *N = cast<" << ClassName << ">(inN);\n";
    OS << Code << "\n}\n";
  }
}

//===----------------------------------------------------------------------===//
// Predicate emitter implementation.
//

void DAGISelEmitter::EmitPredicateFunctions(std::ostream &OS) {
  OS << "\n// Predicate functions.\n";

  // Walk the pattern fragments, adding them to a map, which sorts them by
  // name.
  typedef std::map<std::string, std::pair<Record*, TreePattern*> > PFsByNameTy;
  PFsByNameTy PFsByName;

  for (CodeGenDAGPatterns::pf_iterator I = CGP.pf_begin(), E = CGP.pf_end();
       I != E; ++I)
    PFsByName.insert(std::make_pair(I->first->getName(), *I));

  
  for (PFsByNameTy::iterator I = PFsByName.begin(), E = PFsByName.end();
       I != E; ++I) {
    Record *PatFragRecord = I->second.first;// Record that derives from PatFrag.
    TreePattern *P = I->second.second;
    
    // If there is a code init for this fragment, emit the predicate code.
    std::string Code = PatFragRecord->getValueAsCode("Predicate");
    if (Code.empty()) continue;
    
    if (P->getOnlyTree()->isLeaf())
      OS << "inline bool Predicate_" << PatFragRecord->getName()
      << "(SDNode *N) {\n";
    else {
      std::string ClassName =
        CGP.getSDNodeInfo(P->getOnlyTree()->getOperator()).getSDClassName();
      const char *C2 = ClassName == "SDNode" ? "N" : "inN";
      
      OS << "inline bool Predicate_" << PatFragRecord->getName()
         << "(SDNode *" << C2 << ") {\n";
      if (ClassName != "SDNode")
        OS << "  " << ClassName << " *N = cast<" << ClassName << ">(inN);\n";
    }
    OS << Code << "\n}\n";
  }
  
  OS << "\n\n";
}


//===----------------------------------------------------------------------===//
// PatternCodeEmitter implementation.
//
class PatternCodeEmitter {
private:
  CodeGenDAGPatterns &CGP;

  // Predicates.
  ListInit *Predicates;
  // Pattern cost.
  unsigned Cost;
  // Instruction selector pattern.
  TreePatternNode *Pattern;
  // Matched instruction.
  TreePatternNode *Instruction;
  
  // Node to name mapping
  std::map<std::string, std::string> VariableMap;
  // Node to operator mapping
  std::map<std::string, Record*> OperatorMap;
  // Name of the folded node which produces a flag.
  std::pair<std::string, unsigned> FoldedFlag;
  // Names of all the folded nodes which produce chains.
  std::vector<std::pair<std::string, unsigned> > FoldedChains;
  // Original input chain(s).
  std::vector<std::pair<std::string, std::string> > OrigChains;
  std::set<std::string> Duplicates;

  /// LSI - Load/Store information.
  /// Save loads/stores matched by a pattern, and generate a MemOperandSDNode
  /// for each memory access. This facilitates the use of AliasAnalysis in
  /// the backend.
  std::vector<std::string> LSI;

  /// GeneratedCode - This is the buffer that we emit code to.  The first int
  /// indicates whether this is an exit predicate (something that should be
  /// tested, and if true, the match fails) [when 1], or normal code to emit
  /// [when 0], or initialization code to emit [when 2].
  std::vector<std::pair<unsigned, std::string> > &GeneratedCode;
  /// GeneratedDecl - This is the set of all SDOperand declarations needed for
  /// the set of patterns for each top-level opcode.
  std::set<std::string> &GeneratedDecl;
  /// TargetOpcodes - The target specific opcodes used by the resulting
  /// instructions.
  std::vector<std::string> &TargetOpcodes;
  std::vector<std::string> &TargetVTs;
  /// OutputIsVariadic - Records whether the instruction output pattern uses
  /// variable_ops.  This requires that the Emit function be passed an
  /// additional argument to indicate where the input varargs operands
  /// begin.
  bool &OutputIsVariadic;
  /// NumInputRootOps - Records the number of operands the root node of the
  /// input pattern has.  This information is used in the generated code to
  /// pass to Emit functions when variable_ops processing is needed.
  unsigned &NumInputRootOps;

  std::string ChainName;
  unsigned TmpNo;
  unsigned OpcNo;
  unsigned VTNo;
  
  void emitCheck(const std::string &S) {
    if (!S.empty())
      GeneratedCode.push_back(std::make_pair(1, S));
  }
  void emitCode(const std::string &S) {
    if (!S.empty())
      GeneratedCode.push_back(std::make_pair(0, S));
  }
  void emitInit(const std::string &S) {
    if (!S.empty())
      GeneratedCode.push_back(std::make_pair(2, S));
  }
  void emitDecl(const std::string &S) {
    assert(!S.empty() && "Invalid declaration");
    GeneratedDecl.insert(S);
  }
  void emitOpcode(const std::string &Opc) {
    TargetOpcodes.push_back(Opc);
    OpcNo++;
  }
  void emitVT(const std::string &VT) {
    TargetVTs.push_back(VT);
    VTNo++;
  }
public:
  PatternCodeEmitter(CodeGenDAGPatterns &cgp, ListInit *preds,
                     TreePatternNode *pattern, TreePatternNode *instr,
                     std::vector<std::pair<unsigned, std::string> > &gc,
                     std::set<std::string> &gd,
                     std::vector<std::string> &to,
                     std::vector<std::string> &tv,
                     bool &oiv,
                     unsigned &niro)
  : CGP(cgp), Predicates(preds), Pattern(pattern), Instruction(instr),
    GeneratedCode(gc), GeneratedDecl(gd),
    TargetOpcodes(to), TargetVTs(tv),
    OutputIsVariadic(oiv), NumInputRootOps(niro),
    TmpNo(0), OpcNo(0), VTNo(0) {}

  /// EmitMatchCode - Emit a matcher for N, going to the label for PatternNo
  /// if the match fails. At this point, we already know that the opcode for N
  /// matches, and the SDNode for the result has the RootName specified name.
  void EmitMatchCode(TreePatternNode *N, TreePatternNode *P,
                     const std::string &RootName, const std::string &ChainSuffix,
                     bool &FoundChain) {

    // Save loads/stores matched by a pattern.
    if (!N->isLeaf() && N->getName().empty()) {
      if (NodeHasProperty(N, SDNPMemOperand, CGP))
        LSI.push_back(RootName);
    }

    bool isRoot = (P == NULL);
    // Emit instruction predicates. Each predicate is just a string for now.
    if (isRoot) {
      // Record input varargs info.
      NumInputRootOps = N->getNumChildren();

      std::string PredicateCheck;
      for (unsigned i = 0, e = Predicates->getSize(); i != e; ++i) {
        if (DefInit *Pred = dynamic_cast<DefInit*>(Predicates->getElement(i))) {
          Record *Def = Pred->getDef();
          if (!Def->isSubClassOf("Predicate")) {
#ifndef NDEBUG
            Def->dump();
#endif
            assert(0 && "Unknown predicate type!");
          }
          if (!PredicateCheck.empty())
            PredicateCheck += " && ";
          PredicateCheck += "(" + Def->getValueAsString("CondString") + ")";
        }
      }
      
      emitCheck(PredicateCheck);
    }

    if (N->isLeaf()) {
      if (IntInit *II = dynamic_cast<IntInit*>(N->getLeafValue())) {
        emitCheck("cast<ConstantSDNode>(" + RootName +
                  ")->getSignExtended() == " + itostr(II->getValue()));
        return;
      } else if (!NodeIsComplexPattern(N)) {
        assert(0 && "Cannot match this as a leaf value!");
        abort();
      }
    }
  
    // If this node has a name associated with it, capture it in VariableMap. If
    // we already saw this in the pattern, emit code to verify dagness.
    if (!N->getName().empty()) {
      std::string &VarMapEntry = VariableMap[N->getName()];
      if (VarMapEntry.empty()) {
        VarMapEntry = RootName;
      } else {
        // If we get here, this is a second reference to a specific name.  Since
        // we already have checked that the first reference is valid, we don't
        // have to recursively match it, just check that it's the same as the
        // previously named thing.
        emitCheck(VarMapEntry + " == " + RootName);
        return;
      }

      if (!N->isLeaf())
        OperatorMap[N->getName()] = N->getOperator();
    }


    // Emit code to load the child nodes and match their contents recursively.
    unsigned OpNo = 0;
    bool NodeHasChain = NodeHasProperty   (N, SDNPHasChain, CGP);
    bool HasChain     = PatternHasProperty(N, SDNPHasChain, CGP);
    bool EmittedUseCheck = false;
    if (HasChain) {
      if (NodeHasChain)
        OpNo = 1;
      if (!isRoot) {
        // Multiple uses of actual result?
        emitCheck(RootName + ".hasOneUse()");
        EmittedUseCheck = true;
        if (NodeHasChain) {
          // If the immediate use can somehow reach this node through another
          // path, then can't fold it either or it will create a cycle.
          // e.g. In the following diagram, XX can reach ld through YY. If
          // ld is folded into XX, then YY is both a predecessor and a successor
          // of XX.
          //
          //         [ld]
          //         ^  ^
          //         |  |
          //        /   \---
          //      /        [YY]
          //      |         ^
          //     [XX]-------|
          bool NeedCheck = false;
          if (P != Pattern)
            NeedCheck = true;
          else {
            const SDNodeInfo &PInfo = CGP.getSDNodeInfo(P->getOperator());
            NeedCheck =
              P->getOperator() == CGP.get_intrinsic_void_sdnode() ||
              P->getOperator() == CGP.get_intrinsic_w_chain_sdnode() ||
              P->getOperator() == CGP.get_intrinsic_wo_chain_sdnode() ||
              PInfo.getNumOperands() > 1 ||
              PInfo.hasProperty(SDNPHasChain) ||
              PInfo.hasProperty(SDNPInFlag) ||
              PInfo.hasProperty(SDNPOptInFlag);
          }

          if (NeedCheck) {
            std::string ParentName(RootName.begin(), RootName.end()-1);
            emitCheck("CanBeFoldedBy(" + RootName + ".Val, " + ParentName +
                      ".Val, N.Val)");
          }
        }
      }

      if (NodeHasChain) {
        if (FoundChain) {
          emitCheck("(" + ChainName + ".Val == " + RootName + ".Val || "
                    "IsChainCompatible(" + ChainName + ".Val, " +
                    RootName + ".Val))");
          OrigChains.push_back(std::make_pair(ChainName, RootName));
        } else
          FoundChain = true;
        ChainName = "Chain" + ChainSuffix;
        emitInit("SDOperand " + ChainName + " = " + RootName +
                 ".getOperand(0);");
      }
    }

    // Don't fold any node which reads or writes a flag and has multiple uses.
    // FIXME: We really need to separate the concepts of flag and "glue". Those
    // real flag results, e.g. X86CMP output, can have multiple uses.
    // FIXME: If the optional incoming flag does not exist. Then it is ok to
    // fold it.
    if (!isRoot &&
        (PatternHasProperty(N, SDNPInFlag, CGP) ||
         PatternHasProperty(N, SDNPOptInFlag, CGP) ||
         PatternHasProperty(N, SDNPOutFlag, CGP))) {
      if (!EmittedUseCheck) {
        // Multiple uses of actual result?
        emitCheck(RootName + ".hasOneUse()");
      }
    }

    // If there is a node predicate for this, emit the call.
    if (!N->getPredicateFn().empty())
      emitCheck(N->getPredicateFn() + "(" + RootName + ".Val)");

    
    // If this is an 'and R, 1234' where the operation is AND/OR and the RHS is
    // a constant without a predicate fn that has more that one bit set, handle
    // this as a special case.  This is usually for targets that have special
    // handling of certain large constants (e.g. alpha with it's 8/16/32-bit
    // handling stuff).  Using these instructions is often far more efficient
    // than materializing the constant.  Unfortunately, both the instcombiner
    // and the dag combiner can often infer that bits are dead, and thus drop
    // them from the mask in the dag.  For example, it might turn 'AND X, 255'
    // into 'AND X, 254' if it knows the low bit is set.  Emit code that checks
    // to handle this.
    if (!N->isLeaf() && 
        (N->getOperator()->getName() == "and" || 
         N->getOperator()->getName() == "or") &&
        N->getChild(1)->isLeaf() &&
        N->getChild(1)->getPredicateFn().empty()) {
      if (IntInit *II = dynamic_cast<IntInit*>(N->getChild(1)->getLeafValue())) {
        if (!isPowerOf2_32(II->getValue())) {  // Don't bother with single bits.
          emitInit("SDOperand " + RootName + "0" + " = " +
                   RootName + ".getOperand(" + utostr(0) + ");");
          emitInit("SDOperand " + RootName + "1" + " = " +
                   RootName + ".getOperand(" + utostr(1) + ");");

          emitCheck("isa<ConstantSDNode>(" + RootName + "1)");
          const char *MaskPredicate = N->getOperator()->getName() == "or"
            ? "CheckOrMask(" : "CheckAndMask(";
          emitCheck(MaskPredicate + RootName + "0, cast<ConstantSDNode>(" +
                    RootName + "1), " + itostr(II->getValue()) + ")");
          
          EmitChildMatchCode(N->getChild(0), N, RootName + utostr(0), RootName,
                             ChainSuffix + utostr(0), FoundChain);
          return;
        }
      }
    }
    
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i, ++OpNo) {
      emitInit("SDOperand " + RootName + utostr(OpNo) + " = " +
               RootName + ".getOperand(" +utostr(OpNo) + ");");

      EmitChildMatchCode(N->getChild(i), N, RootName + utostr(OpNo), RootName,
                         ChainSuffix + utostr(OpNo), FoundChain);
    }

    // Handle cases when root is a complex pattern.
    const ComplexPattern *CP;
    if (isRoot && N->isLeaf() && (CP = NodeGetComplexPattern(N, CGP))) {
      std::string Fn = CP->getSelectFunc();
      unsigned NumOps = CP->getNumOperands();
      for (unsigned i = 0; i < NumOps; ++i) {
        emitDecl("CPTmp" + utostr(i));
        emitCode("SDOperand CPTmp" + utostr(i) + ";");
      }
      if (CP->hasProperty(SDNPHasChain)) {
        emitDecl("CPInChain");
        emitDecl("Chain" + ChainSuffix);
        emitCode("SDOperand CPInChain;");
        emitCode("SDOperand Chain" + ChainSuffix + ";");
      }

      std::string Code = Fn + "(" + RootName + ", " + RootName;
      for (unsigned i = 0; i < NumOps; i++)
        Code += ", CPTmp" + utostr(i);
      if (CP->hasProperty(SDNPHasChain)) {
        ChainName = "Chain" + ChainSuffix;
        Code += ", CPInChain, Chain" + ChainSuffix;
      }
      emitCheck(Code + ")");
    }
  }

  void EmitChildMatchCode(TreePatternNode *Child, TreePatternNode *Parent,
                          const std::string &RootName, 
                          const std::string &ParentRootName,
                          const std::string &ChainSuffix, bool &FoundChain) {
    if (!Child->isLeaf()) {
      // If it's not a leaf, recursively match.
      const SDNodeInfo &CInfo = CGP.getSDNodeInfo(Child->getOperator());
      emitCheck(RootName + ".getOpcode() == " +
                CInfo.getEnumName());
      EmitMatchCode(Child, Parent, RootName, ChainSuffix, FoundChain);
      bool HasChain = false;
      if (NodeHasProperty(Child, SDNPHasChain, CGP)) {
        HasChain = true;
        FoldedChains.push_back(std::make_pair(RootName, CInfo.getNumResults()));
      }
      if (NodeHasProperty(Child, SDNPOutFlag, CGP)) {
        assert(FoldedFlag.first == "" && FoldedFlag.second == 0 &&
               "Pattern folded multiple nodes which produce flags?");
        FoldedFlag = std::make_pair(RootName,
                                    CInfo.getNumResults() + (unsigned)HasChain);
      }
    } else {
      // If this child has a name associated with it, capture it in VarMap. If
      // we already saw this in the pattern, emit code to verify dagness.
      if (!Child->getName().empty()) {
        std::string &VarMapEntry = VariableMap[Child->getName()];
        if (VarMapEntry.empty()) {
          VarMapEntry = RootName;
        } else {
          // If we get here, this is a second reference to a specific name.
          // Since we already have checked that the first reference is valid,
          // we don't have to recursively match it, just check that it's the
          // same as the previously named thing.
          emitCheck(VarMapEntry + " == " + RootName);
          Duplicates.insert(RootName);
          return;
        }
      }
      
      // Handle leaves of various types.
      if (DefInit *DI = dynamic_cast<DefInit*>(Child->getLeafValue())) {
        Record *LeafRec = DI->getDef();
        if (LeafRec->isSubClassOf("RegisterClass") || 
            LeafRec->getName() == "ptr_rc") {
          // Handle register references.  Nothing to do here.
        } else if (LeafRec->isSubClassOf("Register")) {
          // Handle register references.
        } else if (LeafRec->isSubClassOf("ComplexPattern")) {
          // Handle complex pattern.
          const ComplexPattern *CP = NodeGetComplexPattern(Child, CGP);
          std::string Fn = CP->getSelectFunc();
          unsigned NumOps = CP->getNumOperands();
          for (unsigned i = 0; i < NumOps; ++i) {
            emitDecl("CPTmp" + utostr(i));
            emitCode("SDOperand CPTmp" + utostr(i) + ";");
          }
          if (CP->hasProperty(SDNPHasChain)) {
            const SDNodeInfo &PInfo = CGP.getSDNodeInfo(Parent->getOperator());
            FoldedChains.push_back(std::make_pair("CPInChain",
                                                  PInfo.getNumResults()));
            ChainName = "Chain" + ChainSuffix;
            emitDecl("CPInChain");
            emitDecl(ChainName);
            emitCode("SDOperand CPInChain;");
            emitCode("SDOperand " + ChainName + ";");
          }
          
          std::string Code = Fn + "(";
          if (CP->hasAttribute(CPAttrParentAsRoot)) {
            Code += ParentRootName + ", ";
          } else {
            Code += "N, ";
          }
          if (CP->hasProperty(SDNPHasChain)) {
            std::string ParentName(RootName.begin(), RootName.end()-1);
            Code += ParentName + ", ";
          }
          Code += RootName;
          for (unsigned i = 0; i < NumOps; i++)
            Code += ", CPTmp" + utostr(i);
          if (CP->hasProperty(SDNPHasChain))
            Code += ", CPInChain, Chain" + ChainSuffix;
          emitCheck(Code + ")");
        } else if (LeafRec->getName() == "srcvalue") {
          // Place holder for SRCVALUE nodes. Nothing to do here.
        } else if (LeafRec->isSubClassOf("ValueType")) {
          // Make sure this is the specified value type.
          emitCheck("cast<VTSDNode>(" + RootName +
                    ")->getVT() == MVT::" + LeafRec->getName());
        } else if (LeafRec->isSubClassOf("CondCode")) {
          // Make sure this is the specified cond code.
          emitCheck("cast<CondCodeSDNode>(" + RootName +
                    ")->get() == ISD::" + LeafRec->getName());
        } else {
#ifndef NDEBUG
          Child->dump();
          cerr << " ";
#endif
          assert(0 && "Unknown leaf type!");
        }
        
        // If there is a node predicate for this, emit the call.
        if (!Child->getPredicateFn().empty())
          emitCheck(Child->getPredicateFn() + "(" + RootName +
                    ".Val)");
      } else if (IntInit *II =
                 dynamic_cast<IntInit*>(Child->getLeafValue())) {
        emitCheck("isa<ConstantSDNode>(" + RootName + ")");
        unsigned CTmp = TmpNo++;
        emitCode("int64_t CN"+utostr(CTmp)+" = cast<ConstantSDNode>("+
                 RootName + ")->getSignExtended();");
        
        emitCheck("CN" + utostr(CTmp) + " == " +itostr(II->getValue()));
      } else {
#ifndef NDEBUG
        Child->dump();
#endif
        assert(0 && "Unknown leaf type!");
      }
    }
  }

  /// EmitResultCode - Emit the action for a pattern.  Now that it has matched
  /// we actually have to build a DAG!
  std::vector<std::string>
  EmitResultCode(TreePatternNode *N, std::vector<Record*> DstRegs,
                 bool InFlagDecled, bool ResNodeDecled,
                 bool LikeLeaf = false, bool isRoot = false) {
    // List of arguments of getTargetNode() or SelectNodeTo().
    std::vector<std::string> NodeOps;
    // This is something selected from the pattern we matched.
    if (!N->getName().empty()) {
      const std::string &VarName = N->getName();
      std::string Val = VariableMap[VarName];
      bool ModifiedVal = false;
      if (Val.empty()) {
        cerr << "Variable '" << VarName << " referenced but not defined "
             << "and not caught earlier!\n";
        abort();
      }
      if (Val[0] == 'T' && Val[1] == 'm' && Val[2] == 'p') {
        // Already selected this operand, just return the tmpval.
        NodeOps.push_back(Val);
        return NodeOps;
      }

      const ComplexPattern *CP;
      unsigned ResNo = TmpNo++;
      if (!N->isLeaf() && N->getOperator()->getName() == "imm") {
        assert(N->getExtTypes().size() == 1 && "Multiple types not handled!");
        std::string CastType;
        std::string TmpVar =  "Tmp" + utostr(ResNo);
        switch (N->getTypeNum(0)) {
        default:
          cerr << "Cannot handle " << getEnumName(N->getTypeNum(0))
               << " type as an immediate constant. Aborting\n";
          abort();
        case MVT::i1:  CastType = "bool"; break;
        case MVT::i8:  CastType = "unsigned char"; break;
        case MVT::i16: CastType = "unsigned short"; break;
        case MVT::i32: CastType = "unsigned"; break;
        case MVT::i64: CastType = "uint64_t"; break;
        }
        emitCode("SDOperand " + TmpVar + 
                 " = CurDAG->getTargetConstant(((" + CastType +
                 ") cast<ConstantSDNode>(" + Val + ")->getValue()), " +
                 getEnumName(N->getTypeNum(0)) + ");");
        // Add Tmp<ResNo> to VariableMap, so that we don't multiply select this
        // value if used multiple times by this pattern result.
        Val = TmpVar;
        ModifiedVal = true;
        NodeOps.push_back(Val);
      } else if (!N->isLeaf() && N->getOperator()->getName() == "fpimm") {
        assert(N->getExtTypes().size() == 1 && "Multiple types not handled!");
        std::string TmpVar =  "Tmp" + utostr(ResNo);
        emitCode("SDOperand " + TmpVar + 
                 " = CurDAG->getTargetConstantFP(cast<ConstantFPSDNode>(" + 
                 Val + ")->getValueAPF(), cast<ConstantFPSDNode>(" + Val +
                 ")->getValueType(0));");
        // Add Tmp<ResNo> to VariableMap, so that we don't multiply select this
        // value if used multiple times by this pattern result.
        Val = TmpVar;
        ModifiedVal = true;
        NodeOps.push_back(Val);
      } else if (!N->isLeaf() && N->getOperator()->getName() == "texternalsym"){
        Record *Op = OperatorMap[N->getName()];
        // Transform ExternalSymbol to TargetExternalSymbol
        if (Op && Op->getName() == "externalsym") {
          std::string TmpVar = "Tmp"+utostr(ResNo);
          emitCode("SDOperand " + TmpVar + " = CurDAG->getTarget"
                   "ExternalSymbol(cast<ExternalSymbolSDNode>(" +
                   Val + ")->getSymbol(), " +
                   getEnumName(N->getTypeNum(0)) + ");");
          // Add Tmp<ResNo> to VariableMap, so that we don't multiply select
          // this value if used multiple times by this pattern result.
          Val = TmpVar;
          ModifiedVal = true;
        }
        NodeOps.push_back(Val);
      } else if (!N->isLeaf() && (N->getOperator()->getName() == "tglobaladdr"
                 || N->getOperator()->getName() == "tglobaltlsaddr")) {
        Record *Op = OperatorMap[N->getName()];
        // Transform GlobalAddress to TargetGlobalAddress
        if (Op && (Op->getName() == "globaladdr" ||
                   Op->getName() == "globaltlsaddr")) {
          std::string TmpVar = "Tmp" + utostr(ResNo);
          emitCode("SDOperand " + TmpVar + " = CurDAG->getTarget"
                   "GlobalAddress(cast<GlobalAddressSDNode>(" + Val +
                   ")->getGlobal(), " + getEnumName(N->getTypeNum(0)) +
                   ");");
          // Add Tmp<ResNo> to VariableMap, so that we don't multiply select
          // this value if used multiple times by this pattern result.
          Val = TmpVar;
          ModifiedVal = true;
        }
        NodeOps.push_back(Val);
      } else if (!N->isLeaf()
                 && (N->getOperator()->getName() == "texternalsym"
                      || N->getOperator()->getName() == "tconstpool")) {
        // Do not rewrite the variable name, since we don't generate a new
        // temporary.
        NodeOps.push_back(Val);
      } else if (N->isLeaf() && (CP = NodeGetComplexPattern(N, CGP))) {
        for (unsigned i = 0; i < CP->getNumOperands(); ++i) {
          emitCode("AddToISelQueue(CPTmp" + utostr(i) + ");");
          NodeOps.push_back("CPTmp" + utostr(i));
        }
      } else {
        // This node, probably wrapped in a SDNodeXForm, behaves like a leaf
        // node even if it isn't one. Don't select it.
        if (!LikeLeaf) {
          emitCode("AddToISelQueue(" + Val + ");");
          if (isRoot && N->isLeaf()) {
            emitCode("ReplaceUses(N, " + Val + ");");
            emitCode("return NULL;");
          }
        }
        NodeOps.push_back(Val);
      }

      if (ModifiedVal) {
        VariableMap[VarName] = Val;
      }
      return NodeOps;
    }
    if (N->isLeaf()) {
      // If this is an explicit register reference, handle it.
      if (DefInit *DI = dynamic_cast<DefInit*>(N->getLeafValue())) {
        unsigned ResNo = TmpNo++;
        if (DI->getDef()->isSubClassOf("Register")) {
          emitCode("SDOperand Tmp" + utostr(ResNo) + " = CurDAG->getRegister(" +
                   getQualifiedName(DI->getDef()) + ", " +
                   getEnumName(N->getTypeNum(0)) + ");");
          NodeOps.push_back("Tmp" + utostr(ResNo));
          return NodeOps;
        } else if (DI->getDef()->getName() == "zero_reg") {
          emitCode("SDOperand Tmp" + utostr(ResNo) +
                   " = CurDAG->getRegister(0, " +
                   getEnumName(N->getTypeNum(0)) + ");");
          NodeOps.push_back("Tmp" + utostr(ResNo));
          return NodeOps;
        }
      } else if (IntInit *II = dynamic_cast<IntInit*>(N->getLeafValue())) {
        unsigned ResNo = TmpNo++;
        assert(N->getExtTypes().size() == 1 && "Multiple types not handled!");
        emitCode("SDOperand Tmp" + utostr(ResNo) + 
                 " = CurDAG->getTargetConstant(0x" + itohexstr(II->getValue()) +
                 "ULL, " + getEnumName(N->getTypeNum(0)) + ");");
        NodeOps.push_back("Tmp" + utostr(ResNo));
        return NodeOps;
      }
    
#ifndef NDEBUG
      N->dump();
#endif
      assert(0 && "Unknown leaf type!");
      return NodeOps;
    }

    Record *Op = N->getOperator();
    if (Op->isSubClassOf("Instruction")) {
      const CodeGenTarget &CGT = CGP.getTargetInfo();
      CodeGenInstruction &II = CGT.getInstruction(Op->getName());
      const DAGInstruction &Inst = CGP.getInstruction(Op);
      const TreePattern *InstPat = Inst.getPattern();
      // FIXME: Assume actual pattern comes before "implicit".
      TreePatternNode *InstPatNode =
        isRoot ? (InstPat ? InstPat->getTree(0) : Pattern)
               : (InstPat ? InstPat->getTree(0) : NULL);
      if (InstPatNode && InstPatNode->getOperator()->getName() == "set") {
        InstPatNode = InstPatNode->getChild(InstPatNode->getNumChildren()-1);
      }
      bool IsVariadic = isRoot && II.isVariadic;
      // FIXME: fix how we deal with physical register operands.
      bool HasImpInputs  = isRoot && Inst.getNumImpOperands() > 0;
      bool HasImpResults = isRoot && DstRegs.size() > 0;
      bool NodeHasOptInFlag = isRoot &&
        PatternHasProperty(Pattern, SDNPOptInFlag, CGP);
      bool NodeHasInFlag  = isRoot &&
        PatternHasProperty(Pattern, SDNPInFlag, CGP);
      bool NodeHasOutFlag = isRoot &&
        PatternHasProperty(Pattern, SDNPOutFlag, CGP);
      bool NodeHasChain = InstPatNode &&
        PatternHasProperty(InstPatNode, SDNPHasChain, CGP);
      bool InputHasChain = isRoot &&
        NodeHasProperty(Pattern, SDNPHasChain, CGP);
      unsigned NumResults = Inst.getNumResults();    
      unsigned NumDstRegs = HasImpResults ? DstRegs.size() : 0;

      // Record output varargs info.
      OutputIsVariadic = IsVariadic;

      if (NodeHasOptInFlag) {
        emitCode("bool HasInFlag = "
           "(N.getOperand(N.getNumOperands()-1).getValueType() == MVT::Flag);");
      }
      if (IsVariadic)
        emitCode("SmallVector<SDOperand, 8> Ops" + utostr(OpcNo) + ";");

      // How many results is this pattern expected to produce?
      unsigned NumPatResults = 0;
      for (unsigned i = 0, e = Pattern->getExtTypes().size(); i != e; i++) {
        MVT::SimpleValueType VT = Pattern->getTypeNum(i);
        if (VT != MVT::isVoid && VT != MVT::Flag)
          NumPatResults++;
      }

      if (OrigChains.size() > 0) {
        // The original input chain is being ignored. If it is not just
        // pointing to the op that's being folded, we should create a
        // TokenFactor with it and the chain of the folded op as the new chain.
        // We could potentially be doing multiple levels of folding, in that
        // case, the TokenFactor can have more operands.
        emitCode("SmallVector<SDOperand, 8> InChains;");
        for (unsigned i = 0, e = OrigChains.size(); i < e; ++i) {
          emitCode("if (" + OrigChains[i].first + ".Val != " +
                   OrigChains[i].second + ".Val) {");
          emitCode("  AddToISelQueue(" + OrigChains[i].first + ");");
          emitCode("  InChains.push_back(" + OrigChains[i].first + ");");
          emitCode("}");
        }
        emitCode("AddToISelQueue(" + ChainName + ");");
        emitCode("InChains.push_back(" + ChainName + ");");
        emitCode(ChainName + " = CurDAG->getNode(ISD::TokenFactor, MVT::Other, "
                 "&InChains[0], InChains.size());");
      }

      // Loop over all of the operands of the instruction pattern, emitting code
      // to fill them all in.  The node 'N' usually has number children equal to
      // the number of input operands of the instruction.  However, in cases
      // where there are predicate operands for an instruction, we need to fill
      // in the 'execute always' values.  Match up the node operands to the
      // instruction operands to do this.
      std::vector<std::string> AllOps;
      for (unsigned ChildNo = 0, InstOpNo = NumResults;
           InstOpNo != II.OperandList.size(); ++InstOpNo) {
        std::vector<std::string> Ops;
        
        // Determine what to emit for this operand.
        Record *OperandNode = II.OperandList[InstOpNo].Rec;
        if ((OperandNode->isSubClassOf("PredicateOperand") ||
             OperandNode->isSubClassOf("OptionalDefOperand")) &&
            !CGP.getDefaultOperand(OperandNode).DefaultOps.empty()) {
          // This is a predicate or optional def operand; emit the
          // 'default ops' operands.
          const DAGDefaultOperand &DefaultOp =
            CGP.getDefaultOperand(II.OperandList[InstOpNo].Rec);
          for (unsigned i = 0, e = DefaultOp.DefaultOps.size(); i != e; ++i) {
            Ops = EmitResultCode(DefaultOp.DefaultOps[i], DstRegs,
                                 InFlagDecled, ResNodeDecled);
            AllOps.insert(AllOps.end(), Ops.begin(), Ops.end());
          }
        } else {
          // Otherwise this is a normal operand or a predicate operand without
          // 'execute always'; emit it.
          Ops = EmitResultCode(N->getChild(ChildNo), DstRegs,
                               InFlagDecled, ResNodeDecled);
          AllOps.insert(AllOps.end(), Ops.begin(), Ops.end());
          ++ChildNo;
        }
      }

      // Emit all the chain and CopyToReg stuff.
      bool ChainEmitted = NodeHasChain;
      if (NodeHasChain)
        emitCode("AddToISelQueue(" + ChainName + ");");
      if (NodeHasInFlag || HasImpInputs)
        EmitInFlagSelectCode(Pattern, "N", ChainEmitted,
                             InFlagDecled, ResNodeDecled, true);
      if (NodeHasOptInFlag || NodeHasInFlag || HasImpInputs) {
        if (!InFlagDecled) {
          emitCode("SDOperand InFlag(0, 0);");
          InFlagDecled = true;
        }
        if (NodeHasOptInFlag) {
          emitCode("if (HasInFlag) {");
          emitCode("  InFlag = N.getOperand(N.getNumOperands()-1);");
          emitCode("  AddToISelQueue(InFlag);");
          emitCode("}");
        }
      }

      unsigned ResNo = TmpNo++;
      if (!isRoot || InputHasChain || NodeHasChain || NodeHasOutFlag ||
          NodeHasOptInFlag || HasImpResults) {
        std::string Code;
        std::string Code2;
        std::string NodeName;
        if (!isRoot) {
          NodeName = "Tmp" + utostr(ResNo);
          Code2 = "SDOperand " + NodeName + "(";
        } else {
          NodeName = "ResNode";
          if (!ResNodeDecled) {
            Code2 = "SDNode *" + NodeName + " = ";
            ResNodeDecled = true;
          } else
            Code2 = NodeName + " = ";
        }

        Code += "CurDAG->getTargetNode(Opc" + utostr(OpcNo);
        unsigned OpsNo = OpcNo;
        emitOpcode(II.Namespace + "::" + II.TheDef->getName());

        // Output order: results, chain, flags
        // Result types.
        if (NumResults > 0 && N->getTypeNum(0) != MVT::isVoid) {
          Code += ", VT" + utostr(VTNo);
          emitVT(getEnumName(N->getTypeNum(0)));
        }
        // Add types for implicit results in physical registers, scheduler will
        // care of adding copyfromreg nodes.
        for (unsigned i = 0; i < NumDstRegs; i++) {
          Record *RR = DstRegs[i];
          if (RR->isSubClassOf("Register")) {
            MVT::SimpleValueType RVT = getRegisterValueType(RR, CGT);
            Code += ", " + getEnumName(RVT);
          }
        }
        if (NodeHasChain)
          Code += ", MVT::Other";
        if (NodeHasOutFlag)
          Code += ", MVT::Flag";

        // Inputs.
        if (IsVariadic) {
          for (unsigned i = 0, e = AllOps.size(); i != e; ++i)
            emitCode("Ops" + utostr(OpsNo) + ".push_back(" + AllOps[i] + ");");
          AllOps.clear();

          // Figure out whether any operands at the end of the op list are not
          // part of the variable section.
          std::string EndAdjust;
          if (NodeHasInFlag || HasImpInputs)
            EndAdjust = "-1";  // Always has one flag.
          else if (NodeHasOptInFlag)
            EndAdjust = "-(HasInFlag?1:0)"; // May have a flag.

          emitCode("for (unsigned i = NumInputRootOps + " + utostr(NodeHasChain) +
                   ", e = N.getNumOperands()" + EndAdjust + "; i != e; ++i) {");

          emitCode("  AddToISelQueue(N.getOperand(i));");
          emitCode("  Ops" + utostr(OpsNo) + ".push_back(N.getOperand(i));");
          emitCode("}");
        }

        // Generate MemOperandSDNodes nodes for each memory accesses covered by 
        // this pattern.
        if (II.isSimpleLoad | II.mayLoad | II.mayStore) {
          std::vector<std::string>::const_iterator mi, mie;
          for (mi = LSI.begin(), mie = LSI.end(); mi != mie; ++mi) {
            emitCode("SDOperand LSI_" + *mi + " = "
                     "CurDAG->getMemOperand(cast<MemSDNode>(" +
                     *mi + ")->getMemOperand());");
            if (IsVariadic)
              emitCode("Ops" + utostr(OpsNo) + ".push_back(LSI_" + *mi + ");");
            else
              AllOps.push_back("LSI_" + *mi);
          }
        }

        if (NodeHasChain) {
          if (IsVariadic)
            emitCode("Ops" + utostr(OpsNo) + ".push_back(" + ChainName + ");");
          else
            AllOps.push_back(ChainName);
        }

        if (IsVariadic) {
          if (NodeHasInFlag || HasImpInputs)
            emitCode("Ops" + utostr(OpsNo) + ".push_back(InFlag);");
          else if (NodeHasOptInFlag) {
            emitCode("if (HasInFlag)");
            emitCode("  Ops" + utostr(OpsNo) + ".push_back(InFlag);");
          }
          Code += ", &Ops" + utostr(OpsNo) + "[0], Ops" + utostr(OpsNo) +
            ".size()";
        } else if (NodeHasInFlag || NodeHasOptInFlag || HasImpInputs)
          AllOps.push_back("InFlag");

        unsigned NumOps = AllOps.size();
        if (NumOps) {
          if (!NodeHasOptInFlag && NumOps < 4) {
            for (unsigned i = 0; i != NumOps; ++i)
              Code += ", " + AllOps[i];
          } else {
            std::string OpsCode = "SDOperand Ops" + utostr(OpsNo) + "[] = { ";
            for (unsigned i = 0; i != NumOps; ++i) {
              OpsCode += AllOps[i];
              if (i != NumOps-1)
                OpsCode += ", ";
            }
            emitCode(OpsCode + " };");
            Code += ", Ops" + utostr(OpsNo) + ", ";
            if (NodeHasOptInFlag) {
              Code += "HasInFlag ? ";
              Code += utostr(NumOps) + " : " + utostr(NumOps-1);
            } else
              Code += utostr(NumOps);
          }
        }
            
        if (!isRoot)
          Code += "), 0";
        emitCode(Code2 + Code + ");");

        if (NodeHasChain) {
          // Remember which op produces the chain.
          if (!isRoot)
            emitCode(ChainName + " = SDOperand(" + NodeName +
                     ".Val, " + utostr(NumResults+NumDstRegs) + ");");
          else
            emitCode(ChainName + " = SDOperand(" + NodeName +
                     ", " + utostr(NumResults+NumDstRegs) + ");");
        }

        if (!isRoot) {
          NodeOps.push_back("Tmp" + utostr(ResNo));
          return NodeOps;
        }

        bool NeedReplace = false;
        if (NodeHasOutFlag) {
          if (!InFlagDecled) {
            emitCode("SDOperand InFlag(ResNode, " + 
                   utostr(NumResults+NumDstRegs+(unsigned)NodeHasChain) + ");");
            InFlagDecled = true;
          } else
            emitCode("InFlag = SDOperand(ResNode, " + 
                   utostr(NumResults+NumDstRegs+(unsigned)NodeHasChain) + ");");
        }

        if (FoldedChains.size() > 0) {
          std::string Code;
          for (unsigned j = 0, e = FoldedChains.size(); j < e; j++)
            emitCode("ReplaceUses(SDOperand(" +
                     FoldedChains[j].first + ".Val, " + 
                     utostr(FoldedChains[j].second) + "), SDOperand(ResNode, " +
                     utostr(NumResults+NumDstRegs) + "));");
          NeedReplace = true;
        }

        if (NodeHasOutFlag) {
          if (FoldedFlag.first != "") {
            emitCode("ReplaceUses(SDOperand(" + FoldedFlag.first + ".Val, " +
                     utostr(FoldedFlag.second) + "), InFlag);");
          } else {
            assert(NodeHasProperty(Pattern, SDNPOutFlag, CGP));
            emitCode("ReplaceUses(SDOperand(N.Val, " +
                     utostr(NumPatResults + (unsigned)InputHasChain)
                     +"), InFlag);");
          }
          NeedReplace = true;
        }

        if (NeedReplace && InputHasChain)
          emitCode("ReplaceUses(SDOperand(N.Val, " + 
                   utostr(NumPatResults) + "), SDOperand(" + ChainName
                   + ".Val, " + ChainName + ".ResNo" + "));");

        // User does not expect the instruction would produce a chain!
        if ((!InputHasChain && NodeHasChain) && NodeHasOutFlag) {
          ;
        } else if (InputHasChain && !NodeHasChain) {
          // One of the inner node produces a chain.
          if (NodeHasOutFlag)
            emitCode("ReplaceUses(SDOperand(N.Val, " + utostr(NumPatResults+1) +
                     "), SDOperand(ResNode, N.ResNo-1));");
          emitCode("ReplaceUses(SDOperand(N.Val, " + utostr(NumPatResults) +
                   "), " + ChainName + ");");
        }

        emitCode("return ResNode;");
      } else {
        std::string Code = "return CurDAG->SelectNodeTo(N.Val, Opc" +
          utostr(OpcNo);
        if (N->getTypeNum(0) != MVT::isVoid)
          Code += ", VT" + utostr(VTNo);
        if (NodeHasOutFlag)
          Code += ", MVT::Flag";

        if (NodeHasInFlag || NodeHasOptInFlag || HasImpInputs)
          AllOps.push_back("InFlag");

        unsigned NumOps = AllOps.size();
        if (NumOps) {
          if (!NodeHasOptInFlag && NumOps < 4) {
            for (unsigned i = 0; i != NumOps; ++i)
              Code += ", " + AllOps[i];
          } else {
            std::string OpsCode = "SDOperand Ops" + utostr(OpcNo) + "[] = { ";
            for (unsigned i = 0; i != NumOps; ++i) {
              OpsCode += AllOps[i];
              if (i != NumOps-1)
                OpsCode += ", ";
            }
            emitCode(OpsCode + " };");
            Code += ", Ops" + utostr(OpcNo) + ", ";
            Code += utostr(NumOps);
          }
        }
        emitCode(Code + ");");
        emitOpcode(II.Namespace + "::" + II.TheDef->getName());
        if (N->getTypeNum(0) != MVT::isVoid)
          emitVT(getEnumName(N->getTypeNum(0)));
      }

      return NodeOps;
    } else if (Op->isSubClassOf("SDNodeXForm")) {
      assert(N->getNumChildren() == 1 && "node xform should have one child!");
      // PatLeaf node - the operand may or may not be a leaf node. But it should
      // behave like one.
      std::vector<std::string> Ops =
        EmitResultCode(N->getChild(0), DstRegs, InFlagDecled,
                       ResNodeDecled, true);
      unsigned ResNo = TmpNo++;
      emitCode("SDOperand Tmp" + utostr(ResNo) + " = Transform_" + Op->getName()
               + "(" + Ops.back() + ".Val);");
      NodeOps.push_back("Tmp" + utostr(ResNo));
      if (isRoot)
        emitCode("return Tmp" + utostr(ResNo) + ".Val;");
      return NodeOps;
    } else {
      N->dump();
      cerr << "\n";
      throw std::string("Unknown node in result pattern!");
    }
  }

  /// InsertOneTypeCheck - Insert a type-check for an unresolved type in 'Pat'
  /// and add it to the tree. 'Pat' and 'Other' are isomorphic trees except that 
  /// 'Pat' may be missing types.  If we find an unresolved type to add a check
  /// for, this returns true otherwise false if Pat has all types.
  bool InsertOneTypeCheck(TreePatternNode *Pat, TreePatternNode *Other,
                          const std::string &Prefix, bool isRoot = false) {
    // Did we find one?
    if (Pat->getExtTypes() != Other->getExtTypes()) {
      // Move a type over from 'other' to 'pat'.
      Pat->setTypes(Other->getExtTypes());
      // The top level node type is checked outside of the select function.
      if (!isRoot)
        emitCheck(Prefix + ".Val->getValueType(0) == " +
                  getName(Pat->getTypeNum(0)));
      return true;
    }
  
    unsigned OpNo =
      (unsigned) NodeHasProperty(Pat, SDNPHasChain, CGP);
    for (unsigned i = 0, e = Pat->getNumChildren(); i != e; ++i, ++OpNo)
      if (InsertOneTypeCheck(Pat->getChild(i), Other->getChild(i),
                             Prefix + utostr(OpNo)))
        return true;
    return false;
  }

private:
  /// EmitInFlagSelectCode - Emit the flag operands for the DAG that is
  /// being built.
  void EmitInFlagSelectCode(TreePatternNode *N, const std::string &RootName,
                            bool &ChainEmitted, bool &InFlagDecled,
                            bool &ResNodeDecled, bool isRoot = false) {
    const CodeGenTarget &T = CGP.getTargetInfo();
    unsigned OpNo =
      (unsigned) NodeHasProperty(N, SDNPHasChain, CGP);
    bool HasInFlag = NodeHasProperty(N, SDNPInFlag, CGP);
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i, ++OpNo) {
      TreePatternNode *Child = N->getChild(i);
      if (!Child->isLeaf()) {
        EmitInFlagSelectCode(Child, RootName + utostr(OpNo), ChainEmitted,
                             InFlagDecled, ResNodeDecled);
      } else {
        if (DefInit *DI = dynamic_cast<DefInit*>(Child->getLeafValue())) {
          if (!Child->getName().empty()) {
            std::string Name = RootName + utostr(OpNo);
            if (Duplicates.find(Name) != Duplicates.end())
              // A duplicate! Do not emit a copy for this node.
              continue;
          }

          Record *RR = DI->getDef();
          if (RR->isSubClassOf("Register")) {
            MVT::SimpleValueType RVT = getRegisterValueType(RR, T);
            if (RVT == MVT::Flag) {
              if (!InFlagDecled) {
                emitCode("SDOperand InFlag = " + RootName + utostr(OpNo) + ";");
                InFlagDecled = true;
              } else
                emitCode("InFlag = " + RootName + utostr(OpNo) + ";");
              emitCode("AddToISelQueue(InFlag);");
            } else {
              if (!ChainEmitted) {
                emitCode("SDOperand Chain = CurDAG->getEntryNode();");
                ChainName = "Chain";
                ChainEmitted = true;
              }
              emitCode("AddToISelQueue(" + RootName + utostr(OpNo) + ");");
              if (!InFlagDecled) {
                emitCode("SDOperand InFlag(0, 0);");
                InFlagDecled = true;
              }
              std::string Decl = (!ResNodeDecled) ? "SDNode *" : "";
              emitCode(Decl + "ResNode = CurDAG->getCopyToReg(" + ChainName +
                       ", " + getQualifiedName(RR) +
                       ", " +  RootName + utostr(OpNo) + ", InFlag).Val;");
              ResNodeDecled = true;
              emitCode(ChainName + " = SDOperand(ResNode, 0);");
              emitCode("InFlag = SDOperand(ResNode, 1);");
            }
          }
        }
      }
    }

    if (HasInFlag) {
      if (!InFlagDecled) {
        emitCode("SDOperand InFlag = " + RootName +
               ".getOperand(" + utostr(OpNo) + ");");
        InFlagDecled = true;
      } else
        emitCode("InFlag = " + RootName +
               ".getOperand(" + utostr(OpNo) + ");");
      emitCode("AddToISelQueue(InFlag);");
    }
  }
};

/// EmitCodeForPattern - Given a pattern to match, emit code to the specified
/// stream to match the pattern, and generate the code for the match if it
/// succeeds.  Returns true if the pattern is not guaranteed to match.
void DAGISelEmitter::GenerateCodeForPattern(const PatternToMatch &Pattern,
                  std::vector<std::pair<unsigned, std::string> > &GeneratedCode,
                                           std::set<std::string> &GeneratedDecl,
                                        std::vector<std::string> &TargetOpcodes,
                                            std::vector<std::string> &TargetVTs,
                                            bool &OutputIsVariadic,
                                            unsigned &NumInputRootOps) {
  OutputIsVariadic = false;
  NumInputRootOps = 0;

  PatternCodeEmitter Emitter(CGP, Pattern.getPredicates(),
                             Pattern.getSrcPattern(), Pattern.getDstPattern(),
                             GeneratedCode, GeneratedDecl,
                             TargetOpcodes, TargetVTs,
                             OutputIsVariadic, NumInputRootOps);

  // Emit the matcher, capturing named arguments in VariableMap.
  bool FoundChain = false;
  Emitter.EmitMatchCode(Pattern.getSrcPattern(), NULL, "N", "", FoundChain);

  // TP - Get *SOME* tree pattern, we don't care which.
  TreePattern &TP = *CGP.pf_begin()->second;
  
  // At this point, we know that we structurally match the pattern, but the
  // types of the nodes may not match.  Figure out the fewest number of type 
  // comparisons we need to emit.  For example, if there is only one integer
  // type supported by a target, there should be no type comparisons at all for
  // integer patterns!
  //
  // To figure out the fewest number of type checks needed, clone the pattern,
  // remove the types, then perform type inference on the pattern as a whole.
  // If there are unresolved types, emit an explicit check for those types,
  // apply the type to the tree, then rerun type inference.  Iterate until all
  // types are resolved.
  //
  TreePatternNode *Pat = Pattern.getSrcPattern()->clone();
  RemoveAllTypes(Pat);
  
  do {
    // Resolve/propagate as many types as possible.
    try {
      bool MadeChange = true;
      while (MadeChange)
        MadeChange = Pat->ApplyTypeConstraints(TP,
                                               true/*Ignore reg constraints*/);
    } catch (...) {
      assert(0 && "Error: could not find consistent types for something we"
             " already decided was ok!");
      abort();
    }

    // Insert a check for an unresolved type and add it to the tree.  If we find
    // an unresolved type to add a check for, this returns true and we iterate,
    // otherwise we are done.
  } while (Emitter.InsertOneTypeCheck(Pat, Pattern.getSrcPattern(), "N", true));

  Emitter.EmitResultCode(Pattern.getDstPattern(), Pattern.getDstRegs(),
                         false, false, false, true);
  delete Pat;
}

/// EraseCodeLine - Erase one code line from all of the patterns.  If removing
/// a line causes any of them to be empty, remove them and return true when
/// done.
static bool EraseCodeLine(std::vector<std::pair<const PatternToMatch*, 
                          std::vector<std::pair<unsigned, std::string> > > >
                          &Patterns) {
  bool ErasedPatterns = false;
  for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
    Patterns[i].second.pop_back();
    if (Patterns[i].second.empty()) {
      Patterns.erase(Patterns.begin()+i);
      --i; --e;
      ErasedPatterns = true;
    }
  }
  return ErasedPatterns;
}

/// EmitPatterns - Emit code for at least one pattern, but try to group common
/// code together between the patterns.
void DAGISelEmitter::EmitPatterns(std::vector<std::pair<const PatternToMatch*, 
                              std::vector<std::pair<unsigned, std::string> > > >
                                  &Patterns, unsigned Indent,
                                  std::ostream &OS) {
  typedef std::pair<unsigned, std::string> CodeLine;
  typedef std::vector<CodeLine> CodeList;
  typedef std::vector<std::pair<const PatternToMatch*, CodeList> > PatternList;
  
  if (Patterns.empty()) return;
  
  // Figure out how many patterns share the next code line.  Explicitly copy
  // FirstCodeLine so that we don't invalidate a reference when changing
  // Patterns.
  const CodeLine FirstCodeLine = Patterns.back().second.back();
  unsigned LastMatch = Patterns.size()-1;
  while (LastMatch != 0 && Patterns[LastMatch-1].second.back() == FirstCodeLine)
    --LastMatch;
  
  // If not all patterns share this line, split the list into two pieces.  The
  // first chunk will use this line, the second chunk won't.
  if (LastMatch != 0) {
    PatternList Shared(Patterns.begin()+LastMatch, Patterns.end());
    PatternList Other(Patterns.begin(), Patterns.begin()+LastMatch);
    
    // FIXME: Emit braces?
    if (Shared.size() == 1) {
      const PatternToMatch &Pattern = *Shared.back().first;
      OS << "\n" << std::string(Indent, ' ') << "// Pattern: ";
      Pattern.getSrcPattern()->print(OS);
      OS << "\n" << std::string(Indent, ' ') << "// Emits: ";
      Pattern.getDstPattern()->print(OS);
      OS << "\n";
      unsigned AddedComplexity = Pattern.getAddedComplexity();
      OS << std::string(Indent, ' ') << "// Pattern complexity = "
         << getPatternSize(Pattern.getSrcPattern(), CGP) + AddedComplexity
         << "  cost = "
         << getResultPatternCost(Pattern.getDstPattern(), CGP)
         << "  size = "
         << getResultPatternSize(Pattern.getDstPattern(), CGP) << "\n";
    }
    if (FirstCodeLine.first != 1) {
      OS << std::string(Indent, ' ') << "{\n";
      Indent += 2;
    }
    EmitPatterns(Shared, Indent, OS);
    if (FirstCodeLine.first != 1) {
      Indent -= 2;
      OS << std::string(Indent, ' ') << "}\n";
    }
    
    if (Other.size() == 1) {
      const PatternToMatch &Pattern = *Other.back().first;
      OS << "\n" << std::string(Indent, ' ') << "// Pattern: ";
      Pattern.getSrcPattern()->print(OS);
      OS << "\n" << std::string(Indent, ' ') << "// Emits: ";
      Pattern.getDstPattern()->print(OS);
      OS << "\n";
      unsigned AddedComplexity = Pattern.getAddedComplexity();
      OS << std::string(Indent, ' ') << "// Pattern complexity = "
         << getPatternSize(Pattern.getSrcPattern(), CGP) + AddedComplexity
         << "  cost = "
         << getResultPatternCost(Pattern.getDstPattern(), CGP)
         << "  size = "
         << getResultPatternSize(Pattern.getDstPattern(), CGP) << "\n";
    }
    EmitPatterns(Other, Indent, OS);
    return;
  }
  
  // Remove this code from all of the patterns that share it.
  bool ErasedPatterns = EraseCodeLine(Patterns);
  
  bool isPredicate = FirstCodeLine.first == 1;
  
  // Otherwise, every pattern in the list has this line.  Emit it.
  if (!isPredicate) {
    // Normal code.
    OS << std::string(Indent, ' ') << FirstCodeLine.second << "\n";
  } else {
    OS << std::string(Indent, ' ') << "if (" << FirstCodeLine.second;
    
    // If the next code line is another predicate, and if all of the pattern
    // in this group share the same next line, emit it inline now.  Do this
    // until we run out of common predicates.
    while (!ErasedPatterns && Patterns.back().second.back().first == 1) {
      // Check that all of fhe patterns in Patterns end with the same predicate.
      bool AllEndWithSamePredicate = true;
      for (unsigned i = 0, e = Patterns.size(); i != e; ++i)
        if (Patterns[i].second.back() != Patterns.back().second.back()) {
          AllEndWithSamePredicate = false;
          break;
        }
      // If all of the predicates aren't the same, we can't share them.
      if (!AllEndWithSamePredicate) break;
      
      // Otherwise we can.  Emit it shared now.
      OS << " &&\n" << std::string(Indent+4, ' ')
         << Patterns.back().second.back().second;
      ErasedPatterns = EraseCodeLine(Patterns);
    }
    
    OS << ") {\n";
    Indent += 2;
  }
  
  EmitPatterns(Patterns, Indent, OS);
  
  if (isPredicate)
    OS << std::string(Indent-2, ' ') << "}\n";
}

static std::string getOpcodeName(Record *Op, CodeGenDAGPatterns &CGP) {
  return CGP.getSDNodeInfo(Op).getEnumName();
}

static std::string getLegalCName(std::string OpName) {
  std::string::size_type pos = OpName.find("::");
  if (pos != std::string::npos)
    OpName.replace(pos, 2, "_");
  return OpName;
}

void DAGISelEmitter::EmitInstructionSelector(std::ostream &OS) {
  const CodeGenTarget &Target = CGP.getTargetInfo();
  
  // Get the namespace to insert instructions into.  Make sure not to pick up
  // "TargetInstrInfo" by accidentally getting the namespace off the PHI
  // instruction or something.
  std::string InstNS;
  for (CodeGenTarget::inst_iterator i = Target.inst_begin(),
       e = Target.inst_end(); i != e; ++i) {
    InstNS = i->second.Namespace;
    if (InstNS != "TargetInstrInfo")
      break;
  }
  
  if (!InstNS.empty()) InstNS += "::";
  
  // Group the patterns by their top-level opcodes.
  std::map<std::string, std::vector<const PatternToMatch*> > PatternsByOpcode;
  // All unique target node emission functions.
  std::map<std::string, unsigned> EmitFunctions;
  for (CodeGenDAGPatterns::ptm_iterator I = CGP.ptm_begin(),
       E = CGP.ptm_end(); I != E; ++I) {
    const PatternToMatch &Pattern = *I;

    TreePatternNode *Node = Pattern.getSrcPattern();
    if (!Node->isLeaf()) {
      PatternsByOpcode[getOpcodeName(Node->getOperator(), CGP)].
        push_back(&Pattern);
    } else {
      const ComplexPattern *CP;
      if (dynamic_cast<IntInit*>(Node->getLeafValue())) {
        PatternsByOpcode[getOpcodeName(CGP.getSDNodeNamed("imm"), CGP)].
          push_back(&Pattern);
      } else if ((CP = NodeGetComplexPattern(Node, CGP))) {
        std::vector<Record*> OpNodes = CP->getRootNodes();
        for (unsigned j = 0, e = OpNodes.size(); j != e; j++) {
          PatternsByOpcode[getOpcodeName(OpNodes[j], CGP)]
            .insert(PatternsByOpcode[getOpcodeName(OpNodes[j], CGP)].begin(),
                    &Pattern);
        }
      } else {
        cerr << "Unrecognized opcode '";
        Node->dump();
        cerr << "' on tree pattern '";
        cerr << Pattern.getDstPattern()->getOperator()->getName() << "'!\n";
        exit(1);
      }
    }
  }

  // For each opcode, there might be multiple select functions, one per
  // ValueType of the node (or its first operand if it doesn't produce a
  // non-chain result.
  std::map<std::string, std::vector<std::string> > OpcodeVTMap;

  // Emit one Select_* method for each top-level opcode.  We do this instead of
  // emitting one giant switch statement to support compilers where this will
  // result in the recursive functions taking less stack space.
  for (std::map<std::string, std::vector<const PatternToMatch*> >::iterator
         PBOI = PatternsByOpcode.begin(), E = PatternsByOpcode.end();
       PBOI != E; ++PBOI) {
    const std::string &OpName = PBOI->first;
    std::vector<const PatternToMatch*> &PatternsOfOp = PBOI->second;
    assert(!PatternsOfOp.empty() && "No patterns but map has entry?");

    // We want to emit all of the matching code now.  However, we want to emit
    // the matches in order of minimal cost.  Sort the patterns so the least
    // cost one is at the start.
    std::stable_sort(PatternsOfOp.begin(), PatternsOfOp.end(),
                     PatternSortingPredicate(CGP));

    // Split them into groups by type.
    std::map<MVT::SimpleValueType,
             std::vector<const PatternToMatch*> > PatternsByType;
    for (unsigned i = 0, e = PatternsOfOp.size(); i != e; ++i) {
      const PatternToMatch *Pat = PatternsOfOp[i];
      TreePatternNode *SrcPat = Pat->getSrcPattern();
      MVT::SimpleValueType VT = SrcPat->getTypeNum(0);
      std::map<MVT::SimpleValueType,
               std::vector<const PatternToMatch*> >::iterator TI = 
        PatternsByType.find(VT);
      if (TI != PatternsByType.end())
        TI->second.push_back(Pat);
      else {
        std::vector<const PatternToMatch*> PVec;
        PVec.push_back(Pat);
        PatternsByType.insert(std::make_pair(VT, PVec));
      }
    }

    for (std::map<MVT::SimpleValueType,
                  std::vector<const PatternToMatch*> >::iterator
           II = PatternsByType.begin(), EE = PatternsByType.end(); II != EE;
         ++II) {
      MVT::SimpleValueType OpVT = II->first;
      std::vector<const PatternToMatch*> &Patterns = II->second;
      typedef std::vector<std::pair<unsigned,std::string> > CodeList;
      typedef std::vector<std::pair<unsigned,std::string> >::iterator CodeListI;
    
      std::vector<std::pair<const PatternToMatch*, CodeList> > CodeForPatterns;
      std::vector<std::vector<std::string> > PatternOpcodes;
      std::vector<std::vector<std::string> > PatternVTs;
      std::vector<std::set<std::string> > PatternDecls;
      std::vector<bool> OutputIsVariadicFlags;
      std::vector<unsigned> NumInputRootOpsCounts;
      for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
        CodeList GeneratedCode;
        std::set<std::string> GeneratedDecl;
        std::vector<std::string> TargetOpcodes;
        std::vector<std::string> TargetVTs;
        bool OutputIsVariadic;
        unsigned NumInputRootOps;
        GenerateCodeForPattern(*Patterns[i], GeneratedCode, GeneratedDecl,
                               TargetOpcodes, TargetVTs,
                               OutputIsVariadic, NumInputRootOps);
        CodeForPatterns.push_back(std::make_pair(Patterns[i], GeneratedCode));
        PatternDecls.push_back(GeneratedDecl);
        PatternOpcodes.push_back(TargetOpcodes);
        PatternVTs.push_back(TargetVTs);
        OutputIsVariadicFlags.push_back(OutputIsVariadic);
        NumInputRootOpsCounts.push_back(NumInputRootOps);
      }
    
      // Scan the code to see if all of the patterns are reachable and if it is
      // possible that the last one might not match.
      bool mightNotMatch = true;
      for (unsigned i = 0, e = CodeForPatterns.size(); i != e; ++i) {
        CodeList &GeneratedCode = CodeForPatterns[i].second;
        mightNotMatch = false;

        for (unsigned j = 0, e = GeneratedCode.size(); j != e; ++j) {
          if (GeneratedCode[j].first == 1) { // predicate.
            mightNotMatch = true;
            break;
          }
        }
      
        // If this pattern definitely matches, and if it isn't the last one, the
        // patterns after it CANNOT ever match.  Error out.
        if (mightNotMatch == false && i != CodeForPatterns.size()-1) {
          cerr << "Pattern '";
          CodeForPatterns[i].first->getSrcPattern()->print(*cerr.stream());
          cerr << "' is impossible to select!\n";
          exit(1);
        }
      }

      // Factor target node emission code (emitted by EmitResultCode) into
      // separate functions. Uniquing and share them among all instruction
      // selection routines.
      for (unsigned i = 0, e = CodeForPatterns.size(); i != e; ++i) {
        CodeList &GeneratedCode = CodeForPatterns[i].second;
        std::vector<std::string> &TargetOpcodes = PatternOpcodes[i];
        std::vector<std::string> &TargetVTs = PatternVTs[i];
        std::set<std::string> Decls = PatternDecls[i];
        bool OutputIsVariadic = OutputIsVariadicFlags[i];
        unsigned NumInputRootOps = NumInputRootOpsCounts[i];
        std::vector<std::string> AddedInits;
        int CodeSize = (int)GeneratedCode.size();
        int LastPred = -1;
        for (int j = CodeSize-1; j >= 0; --j) {
          if (LastPred == -1 && GeneratedCode[j].first == 1)
            LastPred = j;
          else if (LastPred != -1 && GeneratedCode[j].first == 2)
            AddedInits.push_back(GeneratedCode[j].second);
        }

        std::string CalleeCode = "(const SDOperand &N";
        std::string CallerCode = "(N";
        for (unsigned j = 0, e = TargetOpcodes.size(); j != e; ++j) {
          CalleeCode += ", unsigned Opc" + utostr(j);
          CallerCode += ", " + TargetOpcodes[j];
        }
        for (unsigned j = 0, e = TargetVTs.size(); j != e; ++j) {
          CalleeCode += ", MVT VT" + utostr(j);
          CallerCode += ", " + TargetVTs[j];
        }
        for (std::set<std::string>::iterator
               I = Decls.begin(), E = Decls.end(); I != E; ++I) {
          std::string Name = *I;
          CalleeCode += ", SDOperand &" + Name;
          CallerCode += ", " + Name;
        }

        if (OutputIsVariadic) {
          CalleeCode += ", unsigned NumInputRootOps";
          CallerCode += ", " + utostr(NumInputRootOps);
        }

        CallerCode += ");";
        CalleeCode += ") ";
        // Prevent emission routines from being inlined to reduce selection
        // routines stack frame sizes.
        CalleeCode += "DISABLE_INLINE ";
        CalleeCode += "{\n";

        for (std::vector<std::string>::const_reverse_iterator
               I = AddedInits.rbegin(), E = AddedInits.rend(); I != E; ++I)
          CalleeCode += "  " + *I + "\n";

        for (int j = LastPred+1; j < CodeSize; ++j)
          CalleeCode += "  " + GeneratedCode[j].second + "\n";
        for (int j = LastPred+1; j < CodeSize; ++j)
          GeneratedCode.pop_back();
        CalleeCode += "}\n";

        // Uniquing the emission routines.
        unsigned EmitFuncNum;
        std::map<std::string, unsigned>::iterator EFI =
          EmitFunctions.find(CalleeCode);
        if (EFI != EmitFunctions.end()) {
          EmitFuncNum = EFI->second;
        } else {
          EmitFuncNum = EmitFunctions.size();
          EmitFunctions.insert(std::make_pair(CalleeCode, EmitFuncNum));
          OS << "SDNode *Emit_" << utostr(EmitFuncNum) << CalleeCode;
        }

        // Replace the emission code within selection routines with calls to the
        // emission functions.
        CallerCode = "return Emit_" + utostr(EmitFuncNum) + CallerCode;
        GeneratedCode.push_back(std::make_pair(false, CallerCode));
      }

      // Print function.
      std::string OpVTStr;
      if (OpVT == MVT::iPTR) {
        OpVTStr = "_iPTR";
      } else if (OpVT == MVT::isVoid) {
        // Nodes with a void result actually have a first result type of either
        // Other (a chain) or Flag.  Since there is no one-to-one mapping from
        // void to this case, we handle it specially here.
      } else {
        OpVTStr = "_" + getEnumName(OpVT).substr(5);  // Skip 'MVT::'
      }
      std::map<std::string, std::vector<std::string> >::iterator OpVTI =
        OpcodeVTMap.find(OpName);
      if (OpVTI == OpcodeVTMap.end()) {
        std::vector<std::string> VTSet;
        VTSet.push_back(OpVTStr);
        OpcodeVTMap.insert(std::make_pair(OpName, VTSet));
      } else
        OpVTI->second.push_back(OpVTStr);

      OS << "SDNode *Select_" << getLegalCName(OpName)
         << OpVTStr << "(const SDOperand &N) {\n";    

      // Loop through and reverse all of the CodeList vectors, as we will be
      // accessing them from their logical front, but accessing the end of a
      // vector is more efficient.
      for (unsigned i = 0, e = CodeForPatterns.size(); i != e; ++i) {
        CodeList &GeneratedCode = CodeForPatterns[i].second;
        std::reverse(GeneratedCode.begin(), GeneratedCode.end());
      }
    
      // Next, reverse the list of patterns itself for the same reason.
      std::reverse(CodeForPatterns.begin(), CodeForPatterns.end());
    
      // Emit all of the patterns now, grouped together to share code.
      EmitPatterns(CodeForPatterns, 2, OS);
    
      // If the last pattern has predicates (which could fail) emit code to
      // catch the case where nothing handles a pattern.
      if (mightNotMatch) {
        OS << "  cerr << \"Cannot yet select: \";\n";
        if (OpName != "ISD::INTRINSIC_W_CHAIN" &&
            OpName != "ISD::INTRINSIC_WO_CHAIN" &&
            OpName != "ISD::INTRINSIC_VOID") {
          OS << "  N.Val->dump(CurDAG);\n";
        } else {
          OS << "  unsigned iid = cast<ConstantSDNode>(N.getOperand("
            "N.getOperand(0).getValueType() == MVT::Other))->getValue();\n"
             << "  cerr << \"intrinsic %\"<< "
            "Intrinsic::getName((Intrinsic::ID)iid);\n";
        }
        OS << "  cerr << '\\n';\n"
           << "  abort();\n"
           << "  return NULL;\n";
      }
      OS << "}\n\n";
    }
  }
  
  // Emit boilerplate.
  OS << "SDNode *Select_INLINEASM(SDOperand N) {\n"
     << "  std::vector<SDOperand> Ops(N.Val->op_begin(), N.Val->op_end());\n"
     << "  SelectInlineAsmMemoryOperands(Ops, *CurDAG);\n\n"
    
     << "  // Ensure that the asm operands are themselves selected.\n"
     << "  for (unsigned j = 0, e = Ops.size(); j != e; ++j)\n"
     << "    AddToISelQueue(Ops[j]);\n\n"
    
     << "  std::vector<MVT> VTs;\n"
     << "  VTs.push_back(MVT::Other);\n"
     << "  VTs.push_back(MVT::Flag);\n"
     << "  SDOperand New = CurDAG->getNode(ISD::INLINEASM, VTs, &Ops[0], "
                 "Ops.size());\n"
     << "  return New.Val;\n"
     << "}\n\n";

  OS << "SDNode *Select_UNDEF(const SDOperand &N) {\n"
     << "  return CurDAG->SelectNodeTo(N.Val, TargetInstrInfo::IMPLICIT_DEF,\n"
     << "                              N.getValueType());\n"
     << "}\n\n";

  OS << "SDNode *Select_DBG_LABEL(const SDOperand &N) {\n"
     << "  SDOperand Chain = N.getOperand(0);\n"
     << "  unsigned C = cast<LabelSDNode>(N)->getLabelID();\n"
     << "  SDOperand Tmp = CurDAG->getTargetConstant(C, MVT::i32);\n"
     << "  AddToISelQueue(Chain);\n"
     << "  return CurDAG->SelectNodeTo(N.Val, TargetInstrInfo::DBG_LABEL,\n"
     << "                              MVT::Other, Tmp, Chain);\n"
     << "}\n\n";

  OS << "SDNode *Select_EH_LABEL(const SDOperand &N) {\n"
     << "  SDOperand Chain = N.getOperand(0);\n"
     << "  unsigned C = cast<LabelSDNode>(N)->getLabelID();\n"
     << "  SDOperand Tmp = CurDAG->getTargetConstant(C, MVT::i32);\n"
     << "  AddToISelQueue(Chain);\n"
     << "  return CurDAG->SelectNodeTo(N.Val, TargetInstrInfo::EH_LABEL,\n"
     << "                              MVT::Other, Tmp, Chain);\n"
     << "}\n\n";

  OS << "SDNode *Select_DECLARE(const SDOperand &N) {\n"
     << "  SDOperand Chain = N.getOperand(0);\n"
     << "  SDOperand N1 = N.getOperand(1);\n"
     << "  SDOperand N2 = N.getOperand(2);\n"
     << "  if (!isa<FrameIndexSDNode>(N1) || !isa<GlobalAddressSDNode>(N2)) {\n"
     << "    cerr << \"Cannot yet select llvm.dbg.declare: \";\n"
     << "    N.Val->dump(CurDAG);\n"
     << "    abort();\n"
     << "  }\n"
     << "  int FI = cast<FrameIndexSDNode>(N1)->getIndex();\n"
     << "  GlobalValue *GV = cast<GlobalAddressSDNode>(N2)->getGlobal();\n"
     << "  SDOperand Tmp1 = "
     << "CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());\n"
     << "  SDOperand Tmp2 = "
     << "CurDAG->getTargetGlobalAddress(GV, TLI.getPointerTy());\n"
     << "  AddToISelQueue(Chain);\n"
     << "  return CurDAG->SelectNodeTo(N.Val, TargetInstrInfo::DECLARE,\n"
     << "                              MVT::Other, Tmp1, Tmp2, Chain);\n"
     << "}\n\n";

  OS << "SDNode *Select_EXTRACT_SUBREG(const SDOperand &N) {\n"
     << "  SDOperand N0 = N.getOperand(0);\n"
     << "  SDOperand N1 = N.getOperand(1);\n"
     << "  unsigned C = cast<ConstantSDNode>(N1)->getValue();\n"
     << "  SDOperand Tmp = CurDAG->getTargetConstant(C, MVT::i32);\n"
     << "  AddToISelQueue(N0);\n"
     << "  return CurDAG->SelectNodeTo(N.Val, TargetInstrInfo::EXTRACT_SUBREG,\n"
     << "                              N.getValueType(), N0, Tmp);\n"
     << "}\n\n";

  OS << "SDNode *Select_INSERT_SUBREG(const SDOperand &N) {\n"
     << "  SDOperand N0 = N.getOperand(0);\n"
     << "  SDOperand N1 = N.getOperand(1);\n"
     << "  SDOperand N2 = N.getOperand(2);\n"
     << "  unsigned C = cast<ConstantSDNode>(N2)->getValue();\n"
     << "  SDOperand Tmp = CurDAG->getTargetConstant(C, MVT::i32);\n"
     << "  AddToISelQueue(N1);\n"
     << "  AddToISelQueue(N0);\n"
     << "  return CurDAG->SelectNodeTo(N.Val, TargetInstrInfo::INSERT_SUBREG,\n"
     << "                              N.getValueType(), N0, N1, Tmp);\n"
     << "}\n\n";

  OS << "// The main instruction selector code.\n"
     << "SDNode *SelectCode(SDOperand N) {\n"
     << "  if (N.getOpcode() >= ISD::BUILTIN_OP_END &&\n"
     << "      N.getOpcode() < (ISD::BUILTIN_OP_END+" << InstNS
     << "INSTRUCTION_LIST_END)) {\n"
     << "    return NULL;   // Already selected.\n"
     << "  }\n\n"
     << "  MVT::SimpleValueType NVT = N.Val->getValueType(0).getSimpleVT();\n"
     << "  switch (N.getOpcode()) {\n"
     << "  default: break;\n"
     << "  case ISD::EntryToken:       // These leaves remain the same.\n"
     << "  case ISD::BasicBlock:\n"
     << "  case ISD::Register:\n"
     << "  case ISD::HANDLENODE:\n"
     << "  case ISD::TargetConstant:\n"
     << "  case ISD::TargetConstantFP:\n"
     << "  case ISD::TargetConstantPool:\n"
     << "  case ISD::TargetFrameIndex:\n"
     << "  case ISD::TargetExternalSymbol:\n"
     << "  case ISD::TargetJumpTable:\n"
     << "  case ISD::TargetGlobalTLSAddress:\n"
     << "  case ISD::TargetGlobalAddress: {\n"
     << "    return NULL;\n"
     << "  }\n"
     << "  case ISD::AssertSext:\n"
     << "  case ISD::AssertZext: {\n"
     << "    AddToISelQueue(N.getOperand(0));\n"
     << "    ReplaceUses(N, N.getOperand(0));\n"
     << "    return NULL;\n"
     << "  }\n"
     << "  case ISD::TokenFactor:\n"
     << "  case ISD::CopyFromReg:\n"
     << "  case ISD::CopyToReg: {\n"
     << "    for (unsigned i = 0, e = N.getNumOperands(); i != e; ++i)\n"
     << "      AddToISelQueue(N.getOperand(i));\n"
     << "    return NULL;\n"
     << "  }\n"
     << "  case ISD::INLINEASM: return Select_INLINEASM(N);\n"
     << "  case ISD::DBG_LABEL: return Select_DBG_LABEL(N);\n"
     << "  case ISD::EH_LABEL: return Select_EH_LABEL(N);\n"
     << "  case ISD::DECLARE: return Select_DECLARE(N);\n"
     << "  case ISD::EXTRACT_SUBREG: return Select_EXTRACT_SUBREG(N);\n"
     << "  case ISD::INSERT_SUBREG: return Select_INSERT_SUBREG(N);\n"
     << "  case ISD::UNDEF: return Select_UNDEF(N);\n";

    
  // Loop over all of the case statements, emiting a call to each method we
  // emitted above.
  for (std::map<std::string, std::vector<const PatternToMatch*> >::iterator
         PBOI = PatternsByOpcode.begin(), E = PatternsByOpcode.end();
       PBOI != E; ++PBOI) {
    const std::string &OpName = PBOI->first;
    // Potentially multiple versions of select for this opcode. One for each
    // ValueType of the node (or its first true operand if it doesn't produce a
    // result.
    std::map<std::string, std::vector<std::string> >::iterator OpVTI =
      OpcodeVTMap.find(OpName);
    std::vector<std::string> &OpVTs = OpVTI->second;
    OS << "  case " << OpName << ": {\n";
    // Keep track of whether we see a pattern that has an iPtr result.
    bool HasPtrPattern = false;
    bool HasDefaultPattern = false;
      
    OS << "    switch (NVT) {\n";
    for (unsigned i = 0, e = OpVTs.size(); i < e; ++i) {
      std::string &VTStr = OpVTs[i];
      if (VTStr.empty()) {
        HasDefaultPattern = true;
        continue;
      }

      // If this is a match on iPTR: don't emit it directly, we need special
      // code.
      if (VTStr == "_iPTR") {
        HasPtrPattern = true;
        continue;
      }
      OS << "    case MVT::" << VTStr.substr(1) << ":\n"
         << "      return Select_" << getLegalCName(OpName)
         << VTStr << "(N);\n";
    }
    OS << "    default:\n";
      
    // If there is an iPTR result version of this pattern, emit it here.
    if (HasPtrPattern) {
      OS << "      if (TLI.getPointerTy() == NVT)\n";
      OS << "        return Select_" << getLegalCName(OpName) <<"_iPTR(N);\n";
    }
    if (HasDefaultPattern) {
      OS << "      return Select_" << getLegalCName(OpName) << "(N);\n";
    }
    OS << "      break;\n";
    OS << "    }\n";
    OS << "    break;\n";
    OS << "  }\n";
  }

  OS << "  } // end of big switch.\n\n"
     << "  cerr << \"Cannot yet select: \";\n"
     << "  if (N.getOpcode() != ISD::INTRINSIC_W_CHAIN &&\n"
     << "      N.getOpcode() != ISD::INTRINSIC_WO_CHAIN &&\n"
     << "      N.getOpcode() != ISD::INTRINSIC_VOID) {\n"
     << "    N.Val->dump(CurDAG);\n"
     << "  } else {\n"
     << "    unsigned iid = cast<ConstantSDNode>(N.getOperand("
               "N.getOperand(0).getValueType() == MVT::Other))->getValue();\n"
     << "    cerr << \"intrinsic %\"<< "
               "Intrinsic::getName((Intrinsic::ID)iid);\n"
     << "  }\n"
     << "  cerr << '\\n';\n"
     << "  abort();\n"
     << "  return NULL;\n"
     << "}\n";
}

void DAGISelEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("DAG Instruction Selector for the " +
                       CGP.getTargetInfo().getName() + " target", OS);
  
  OS << "// *** NOTE: This file is #included into the middle of the target\n"
     << "// *** instruction selector class.  These functions are really "
     << "methods.\n\n";

  OS << "// Include standard, target-independent definitions and methods used\n"
     << "// by the instruction selector.\n";
  OS << "#include <llvm/CodeGen/DAGISelHeader.h>\n\n";
  
  EmitNodeTransforms(OS);
  EmitPredicateFunctions(OS);
  
  DOUT << "\n\nALL PATTERNS TO MATCH:\n\n";
  for (CodeGenDAGPatterns::ptm_iterator I = CGP.ptm_begin(), E = CGP.ptm_end();
       I != E; ++I) {
    DOUT << "PATTERN: ";   DEBUG(I->getSrcPattern()->dump());
    DOUT << "\nRESULT:  "; DEBUG(I->getDstPattern()->dump());
    DOUT << "\n";
  }
  
  // At this point, we have full information about the 'Patterns' we need to
  // parse, both implicitly from instructions as well as from explicit pattern
  // definitions.  Emit the resultant instruction selector.
  EmitInstructionSelector(OS);  
  
}
