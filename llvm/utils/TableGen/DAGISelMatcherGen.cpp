//===- DAGISelMatcherGen.cpp - Matcher generator --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DAGISelMatcher.h"
#include "CodeGenDAGPatterns.h"
#include "Record.h"
#include "llvm/ADT/StringMap.h"
using namespace llvm;

namespace {
  class MatcherGen {
    const PatternToMatch &Pattern;
    const CodeGenDAGPatterns &CGP;
    
    /// PatWithNoTypes - This is a clone of Pattern.getSrcPattern() that starts
    /// out with all of the types removed.  This allows us to insert type checks
    /// as we scan the tree.
    TreePatternNode *PatWithNoTypes;
    
    /// VariableMap - A map from variable names ('$dst') to the recorded operand
    /// number that they were captured as.  These are biased by 1 to make
    /// insertion easier.
    StringMap<unsigned> VariableMap;
    unsigned NextRecordedOperandNo;
    
    MatcherNodeWithChild *Matcher;
    MatcherNodeWithChild *CurPredicate;
  public:
    MatcherGen(const PatternToMatch &pattern, const CodeGenDAGPatterns &cgp);
    
    ~MatcherGen() {
      delete PatWithNoTypes;
    }
    
    void EmitMatcherCode();
    
    MatcherNodeWithChild *GetMatcher() const { return Matcher; }
    MatcherNodeWithChild *GetCurPredicate() const { return CurPredicate; }
  private:
    void AddMatcherNode(MatcherNodeWithChild *NewNode);
    void InferPossibleTypes();
    void EmitMatchCode(const TreePatternNode *N, TreePatternNode *NodeNoTypes);
    void EmitLeafMatchCode(const TreePatternNode *N);
    void EmitOperatorMatchCode(const TreePatternNode *N,
                               TreePatternNode *NodeNoTypes);
  };
  
} // end anon namespace.

MatcherGen::MatcherGen(const PatternToMatch &pattern,
                       const CodeGenDAGPatterns &cgp)
: Pattern(pattern), CGP(cgp), NextRecordedOperandNo(0),
  Matcher(0), CurPredicate(0) {
  // We need to produce the matcher tree for the patterns source pattern.  To do
  // this we need to match the structure as well as the types.  To do the type
  // matching, we want to figure out the fewest number of type checks we need to
  // emit.  For example, if there is only one integer type supported by a
  // target, there should be no type comparisons at all for integer patterns!
  //
  // To figure out the fewest number of type checks needed, clone the pattern,
  // remove the types, then perform type inference on the pattern as a whole.
  // If there are unresolved types, emit an explicit check for those types,
  // apply the type to the tree, then rerun type inference.  Iterate until all
  // types are resolved.
  //
  PatWithNoTypes = Pattern.getSrcPattern()->clone();
  PatWithNoTypes->RemoveAllTypes();
    
  // If there are types that are manifestly known, infer them.
  InferPossibleTypes();
}

/// InferPossibleTypes - As we emit the pattern, we end up generating type
/// checks and applying them to the 'PatWithNoTypes' tree.  As we do this, we
/// want to propagate implied types as far throughout the tree as possible so
/// that we avoid doing redundant type checks.  This does the type propagation.
void MatcherGen::InferPossibleTypes() {
  // TP - Get *SOME* tree pattern, we don't care which.  It is only used for
  // diagnostics, which we know are impossible at this point.
  TreePattern &TP = *CGP.pf_begin()->second;
  
  try {
    bool MadeChange = true;
    while (MadeChange)
      MadeChange = PatWithNoTypes->ApplyTypeConstraints(TP,
                                                true/*Ignore reg constraints*/);
  } catch (...) {
    errs() << "Type constraint application shouldn't fail!";
    abort();
  }
}


/// AddMatcherNode - Add a matcher node to the current graph we're building. 
void MatcherGen::AddMatcherNode(MatcherNodeWithChild *NewNode) {
  if (CurPredicate != 0)
    CurPredicate->setChild(NewNode);
  else
    Matcher = NewNode;
  CurPredicate = NewNode;
}



/// EmitLeafMatchCode - Generate matching code for leaf nodes.
void MatcherGen::EmitLeafMatchCode(const TreePatternNode *N) {
  assert(N->isLeaf() && "Not a leaf?");
  // Direct match against an integer constant.
  if (IntInit *II = dynamic_cast<IntInit*>(N->getLeafValue()))
    return AddMatcherNode(new CheckIntegerMatcherNode(II->getValue()));
  
  DefInit *DI = dynamic_cast<DefInit*>(N->getLeafValue());
  if (DI == 0) {
    errs() << "Unknown leaf kind: " << *DI << "\n";
    abort();
  }
  
  Record *LeafRec = DI->getDef();
  if (// Handle register references.  Nothing to do here, they always match.
      LeafRec->isSubClassOf("RegisterClass") || 
      LeafRec->isSubClassOf("PointerLikeRegClass") ||
      LeafRec->isSubClassOf("Register") ||
      // Place holder for SRCVALUE nodes. Nothing to do here.
      LeafRec->getName() == "srcvalue")
    return;
  
  if (LeafRec->isSubClassOf("ValueType"))
    return AddMatcherNode(new CheckValueTypeMatcherNode(LeafRec->getName()));
  
  if (LeafRec->isSubClassOf("CondCode"))
    return AddMatcherNode(new CheckCondCodeMatcherNode(LeafRec->getName()));
  
  if (LeafRec->isSubClassOf("ComplexPattern")) {
    if (!N->getName().empty()) {
      errs() << "We expect complex pattern uses to have names: " << *N << "\n";
      exit(1);
    }
    
    // Handle complex pattern.
    const ComplexPattern &CP = CGP.getComplexPattern(LeafRec);
    return AddMatcherNode(new CheckComplexPatMatcherNode(CP));
  }
  
  errs() << "Unknown leaf kind: " << *N << "\n";
  abort();
}

void MatcherGen::EmitOperatorMatchCode(const TreePatternNode *N,
                                       TreePatternNode *NodeNoTypes) {
  assert(!N->isLeaf() && "Not an operator?");
  const SDNodeInfo &CInfo = CGP.getSDNodeInfo(N->getOperator());
  
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
  if ((N->getOperator()->getName() == "and" || 
       N->getOperator()->getName() == "or") &&
      N->getChild(1)->isLeaf() && N->getChild(1)->getPredicateFns().empty()) {
    if (IntInit *II = dynamic_cast<IntInit*>(N->getChild(1)->getLeafValue())) {
      if (!isPowerOf2_32(II->getValue())) {  // Don't bother with single bits.
        if (N->getOperator()->getName() == "and")
          AddMatcherNode(new CheckAndImmMatcherNode(II->getValue()));
        else
          AddMatcherNode(new CheckOrImmMatcherNode(II->getValue()));

        // Match the LHS of the AND as appropriate.
        AddMatcherNode(new MoveChildMatcherNode(0));
        EmitMatchCode(N->getChild(0), NodeNoTypes->getChild(0));
        AddMatcherNode(new MoveParentMatcherNode());
        return;
      }
    }
  }
  
  // Check that the current opcode lines up.
  AddMatcherNode(new CheckOpcodeMatcherNode(CInfo.getEnumName()));
  
  // If this node has a chain, then the chain is operand #0 is the SDNode, and
  // the child numbers of the node are all offset by one.
  unsigned OpNo = 0;
  if (N->NodeHasProperty(SDNPHasChain, CGP)) {
    OpNo = 1;

    // If this node is not the root and the subtree underneath it produces a
    // chain, then the result of matching the node is also produce a chain.
    // Beyond that, this means that we're also folding (at least) the root node
    // into the node that produce the chain (for example, matching
    // "(add reg, (load ptr))" as a add_with_memory on X86).  This is
    // problematic, if the 'reg' node also uses the load (say, its chain).
    // Graphically:
    //
    //         [LD]
    //         ^  ^
    //         |  \                              DAG's like cheese.
    //        /    |
    //       /    [YY]
    //       |     ^
    //      [XX]--/
    //
    // It would be invalid to fold XX and LD.  In this case, folding the two
    // nodes together would induce a cycle in the DAG, making it a 'cyclic DAG'
    // To prevent this, we emit a dynamic check for legality before allowing
    // this to be folded.
    //
    const TreePatternNode *Root = Pattern.getSrcPattern();
    if (N != Root) {                             // Not the root of the pattern.
      // If there is a node between the root and this node, then we definitely
      // need to emit the check.
      bool NeedCheck = !Root->hasChild(N);
      
      // If it *is* an immediate child of the root, we can still need a check if
      // the root SDNode has multiple inputs.  For us, this means that it is an
      // intrinsic, has multiple operands, or has other inputs like chain or
      // flag).
      if (!NeedCheck) {
        const SDNodeInfo &PInfo = CGP.getSDNodeInfo(Root->getOperator());
        NeedCheck =
          Root->getOperator() == CGP.get_intrinsic_void_sdnode() ||
          Root->getOperator() == CGP.get_intrinsic_w_chain_sdnode() ||
          Root->getOperator() == CGP.get_intrinsic_wo_chain_sdnode() ||
          PInfo.getNumOperands() > 1 ||
          PInfo.hasProperty(SDNPHasChain) ||
          PInfo.hasProperty(SDNPInFlag) ||
          PInfo.hasProperty(SDNPOptInFlag);
      }
      
      if (NeedCheck)
        AddMatcherNode(new CheckFoldableChainNodeMatcherNode());
    }
  }
  
  // FIXME: Need to generate IsChainCompatible checks.
      
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i, ++OpNo) {
    // Get the code suitable for matching this child.  Move to the child, check
    // it then move back to the parent.
    AddMatcherNode(new MoveChildMatcherNode(i));
    EmitMatchCode(N->getChild(i), NodeNoTypes->getChild(i));
    AddMatcherNode(new MoveParentMatcherNode());
  }
}


void MatcherGen::EmitMatchCode(const TreePatternNode *N,
                               TreePatternNode *NodeNoTypes) {
  // If N and NodeNoTypes don't agree on a type, then this is a case where we
  // need to do a type check.  Emit the check, apply the tyep to NodeNoTypes and
  // reinfer any correlated types.
  if (NodeNoTypes->getExtTypes() != N->getExtTypes()) {
    AddMatcherNode(new CheckTypeMatcherNode(N->getTypeNum(0)));
    NodeNoTypes->setTypes(N->getExtTypes());
    InferPossibleTypes();
  }
  
  
  // If this node has a name associated with it, capture it in VariableMap. If
  // we already saw this in the pattern, emit code to verify dagness.
  if (!N->getName().empty()) {
    unsigned &VarMapEntry = VariableMap[N->getName()];
    if (VarMapEntry == 0) {
      VarMapEntry = ++NextRecordedOperandNo;
      
      // If this is a complex pattern, the match operation for it will
      // implicitly record all of the outputs of it (which may be more than
      // one).
      if (const ComplexPattern *AM = N->getComplexPatternInfo(CGP)) {
        // Record the right number of operands.
        // FIXME: Does this include chain?
        VarMapEntry += AM->getNumOperands()-1;
      } else {
        // If it is a normal named node, we must emit a 'Record' opcode.
        AddMatcherNode(new RecordMatcherNode());
      }
      
    } else {
      // If we get here, this is a second reference to a specific name.  Since
      // we already have checked that the first reference is valid, we don't
      // have to recursively match it, just check that it's the same as the
      // previously named thing.
      AddMatcherNode(new CheckSameMatcherNode(VarMapEntry-1));
      return;
    }
  }
  
  // If there are node predicates for this node, generate their checks.
  for (unsigned i = 0, e = N->getPredicateFns().size(); i != e; ++i)
    AddMatcherNode(new CheckPredicateMatcherNode(N->getPredicateFns()[i]));

  if (N->isLeaf())
    EmitLeafMatchCode(N);
  else
    EmitOperatorMatchCode(N, NodeNoTypes);
}

void MatcherGen::EmitMatcherCode() {
  // If the pattern has a predicate on it (e.g. only enabled when a subtarget
  // feature is around, do the check).
  if (!Pattern.getPredicateCheck().empty())
    AddMatcherNode(new 
                 CheckPatternPredicateMatcherNode(Pattern.getPredicateCheck()));
  
  // Emit the matcher for the pattern structure and types.
  EmitMatchCode(Pattern.getSrcPattern(), PatWithNoTypes);
}


MatcherNode *llvm::ConvertPatternToMatcher(const PatternToMatch &Pattern,
                                           const CodeGenDAGPatterns &CGP) {
  MatcherGen Gen(Pattern, CGP);

  // Generate the code for the matcher.
  Gen.EmitMatcherCode();
  
  // If the match succeeds, then we generate Pattern.
  EmitNodeMatcherNode *Result = new EmitNodeMatcherNode(Pattern);
  
  // Link it into the pattern.
  if (MatcherNodeWithChild *Pred = Gen.GetCurPredicate()) {
    Pred->setChild(Result);
    return Gen.GetMatcher();
  }

  // Unconditional match.
  return Result;
}



