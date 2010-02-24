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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <utility>
using namespace llvm;


/// getRegisterValueType - Look up and return the ValueType of the specified
/// register. If the register is a member of multiple register classes which
/// have different associated types, return MVT::Other.
static MVT::SimpleValueType getRegisterValueType(Record *R,
                                                 const CodeGenTarget &T) {
  bool FoundRC = false;
  MVT::SimpleValueType VT = MVT::Other;
  const std::vector<CodeGenRegisterClass> &RCs = T.getRegisterClasses();
  std::vector<Record*>::const_iterator Element;
  
  for (unsigned rc = 0, e = RCs.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = RCs[rc];
    if (!std::count(RC.Elements.begin(), RC.Elements.end(), R))
      continue;
    
    if (!FoundRC) {
      FoundRC = true;
      VT = RC.getValueTypeNum(0);
      continue;
    }
    
    // In multiple RC's.  If the Types of the RC's do not agree, return
    // MVT::Other. The target is responsible for handling this.
    if (VT != RC.getValueTypeNum(0))
      // FIXME2: when does this happen?  Abort?
      return MVT::Other;
  }
  return VT;
}


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
    
    /// NextRecordedOperandNo - As we emit opcodes to record matched values in
    /// the RecordedNodes array, this keeps track of which slot will be next to
    /// record into.
    unsigned NextRecordedOperandNo;
    
    /// MatchedChainNodes - This maintains the position in the recorded nodes
    /// array of all of the recorded input nodes that have chains.
    SmallVector<unsigned, 2> MatchedChainNodes;

    /// MatchedFlagResultNodes - This maintains the position in the recorded
    /// nodes array of all of the recorded input nodes that have flag results.
    SmallVector<unsigned, 2> MatchedFlagResultNodes;
    
    /// PhysRegInputs - List list has an entry for each explicitly specified
    /// physreg input to the pattern.  The first elt is the Register node, the
    /// second is the recorded slot number the input pattern match saved it in.
    SmallVector<std::pair<Record*, unsigned>, 2> PhysRegInputs;
    
    /// EmittedMergeInputChains - For nodes that match patterns involving
    /// chains, is set to true if we emitted the "MergeInputChains" operation.
    bool EmittedMergeInputChains;
    
    /// Matcher - This is the top level of the generated matcher, the result.
    MatcherNode *Matcher;
    
    /// CurPredicate - As we emit matcher nodes, this points to the latest check
    /// which should have future checks stuck into its Next position.
    MatcherNode *CurPredicate;
  public:
    MatcherGen(const PatternToMatch &pattern, const CodeGenDAGPatterns &cgp);
    
    ~MatcherGen() {
      delete PatWithNoTypes;
    }
    
    void EmitMatcherCode();
    void EmitResultCode();
    
    MatcherNode *GetMatcher() const { return Matcher; }
    MatcherNode *GetCurPredicate() const { return CurPredicate; }
  private:
    void AddMatcherNode(MatcherNode *NewNode);
    void InferPossibleTypes();
    
    // Matcher Generation.
    void EmitMatchCode(const TreePatternNode *N, TreePatternNode *NodeNoTypes);
    void EmitLeafMatchCode(const TreePatternNode *N);
    void EmitOperatorMatchCode(const TreePatternNode *N,
                               TreePatternNode *NodeNoTypes);
    
    // Result Code Generation.
    unsigned getNamedArgumentSlot(StringRef Name) {
      unsigned VarMapEntry = VariableMap[Name];
      assert(VarMapEntry != 0 &&
             "Variable referenced but not defined and not caught earlier!");
      return VarMapEntry-1;
    }

    /// GetInstPatternNode - Get the pattern for an instruction.
    const TreePatternNode *GetInstPatternNode(const DAGInstruction &Ins,
                                              const TreePatternNode *N);
    
    void EmitResultOperand(const TreePatternNode *N,
                           SmallVectorImpl<unsigned> &ResultOps);
    void EmitResultOfNamedOperand(const TreePatternNode *N,
                                  SmallVectorImpl<unsigned> &ResultOps);
    void EmitResultLeafAsOperand(const TreePatternNode *N,
                                 SmallVectorImpl<unsigned> &ResultOps);
    void EmitResultInstructionAsOperand(const TreePatternNode *N,
                                        SmallVectorImpl<unsigned> &ResultOps);
    void EmitResultSDNodeXFormAsOperand(const TreePatternNode *N,
                                        SmallVectorImpl<unsigned> &ResultOps);
    };
  
} // end anon namespace.

MatcherGen::MatcherGen(const PatternToMatch &pattern,
                       const CodeGenDAGPatterns &cgp)
: Pattern(pattern), CGP(cgp), NextRecordedOperandNo(0),
  EmittedMergeInputChains(false), Matcher(0), CurPredicate(0) {
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
void MatcherGen::AddMatcherNode(MatcherNode *NewNode) {
  if (CurPredicate != 0)
    CurPredicate->setNext(NewNode);
  else
    Matcher = NewNode;
  CurPredicate = NewNode;
}


//===----------------------------------------------------------------------===//
// Pattern Match Generation
//===----------------------------------------------------------------------===//

/// EmitLeafMatchCode - Generate matching code for leaf nodes.
void MatcherGen::EmitLeafMatchCode(const TreePatternNode *N) {
  assert(N->isLeaf() && "Not a leaf?");
  
  // If there are node predicates for this node, generate their checks.
  for (unsigned i = 0, e = N->getPredicateFns().size(); i != e; ++i)
    AddMatcherNode(new CheckPredicateMatcherNode(N->getPredicateFns()[i]));
  
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
      // Place holder for SRCVALUE nodes. Nothing to do here.
      LeafRec->getName() == "srcvalue")
    return;

  // If we have a physreg reference like (mul gpr:$src, EAX) then we need to
  // record the register 
  if (LeafRec->isSubClassOf("Register")) {
    AddMatcherNode(new RecordMatcherNode("physreg input "+LeafRec->getName()));
    PhysRegInputs.push_back(std::make_pair(LeafRec, NextRecordedOperandNo++));
    return;
  }
  
  if (LeafRec->isSubClassOf("ValueType"))
    return AddMatcherNode(new CheckValueTypeMatcherNode(LeafRec->getName()));
  
  if (LeafRec->isSubClassOf("CondCode"))
    return AddMatcherNode(new CheckCondCodeMatcherNode(LeafRec->getName()));
  
  if (LeafRec->isSubClassOf("ComplexPattern")) {
    // We can't model ComplexPattern uses that don't have their name taken yet.
    // The OPC_CheckComplexPattern operation implicitly records the results.
    if (N->getName().empty()) {
      errs() << "We expect complex pattern uses to have names: " << *N << "\n";
      exit(1);
    }

    // Handle complex pattern.
    const ComplexPattern &CP = CGP.getComplexPattern(LeafRec);

    // If we're at the root of the pattern, we have to check that the opcode
    // is a one of the ones requested to be matched.
    if (N == Pattern.getSrcPattern()) {
      const std::vector<Record*> &OpNodes = CP.getRootNodes();
      if (OpNodes.size() == 1) {
        StringRef OpName = CGP.getSDNodeInfo(OpNodes[0]).getEnumName();
        AddMatcherNode(new CheckOpcodeMatcherNode(OpName));
      } else if (!OpNodes.empty()) {
        SmallVector<StringRef, 4> OpNames;
        for (unsigned i = 0, e = OpNodes.size(); i != e; i++)
          OpNames.push_back(CGP.getSDNodeInfo(OpNodes[i]).getEnumName());
        AddMatcherNode(new CheckMultiOpcodeMatcherNode(OpNames.data(),
                                                       OpNames.size()));
      }
    }
    
    // Emit a CheckComplexPat operation, which does the match (aborting if it
    // fails) and pushes the matched operands onto the recorded nodes list.
    AddMatcherNode(new CheckComplexPatMatcherNode(CP));
    
    // Record the right number of operands.
    NextRecordedOperandNo += CP.getNumOperands();
    if (CP.hasProperty(SDNPHasChain))
      ++NextRecordedOperandNo; // Chained node operand.
    
    // If the complex pattern has a chain, then we need to keep track of the
    // fact that we just recorded a chain input.  The chain input will be
    // matched as the last operand of the predicate if it was successful.
    if (CP.hasProperty(SDNPHasChain)) {
      // It is the last operand recorded.
      assert(NextRecordedOperandNo > 1 &&
             "Should have recorded input/result chains at least!");
      MatchedChainNodes.push_back(NextRecordedOperandNo-1);

      // If we need to check chains, do so, see comment for
      // "NodeHasProperty(SDNPHasChain" below.
      if (MatchedChainNodes.size() > 1) {
        // FIXME2: This is broken, we should eliminate this nonsense completely,
        // but we want to produce the same selections that the old matcher does
        // for now.
        unsigned PrevOp = MatchedChainNodes[MatchedChainNodes.size()-2];
        AddMatcherNode(new CheckChainCompatibleMatcherNode(PrevOp));
      }
    }
    
    // TODO: Complex patterns can't have output flags, if they did, we'd want
    // to record them.
    return;
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
      N->getChild(1)->isLeaf() && N->getChild(1)->getPredicateFns().empty() &&
      N->getPredicateFns().empty()) {
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
  
  // If there are node predicates for this node, generate their checks.
  for (unsigned i = 0, e = N->getPredicateFns().size(); i != e; ++i)
    AddMatcherNode(new CheckPredicateMatcherNode(N->getPredicateFns()[i]));
  
  
  // If this node has memory references (i.e. is a load or store), tell the
  // interpreter to capture them in the memref array.
  if (N->NodeHasProperty(SDNPMemOperand, CGP))
    AddMatcherNode(new RecordMemRefMatcherNode());
  
  // If this node has a chain, then the chain is operand #0 is the SDNode, and
  // the child numbers of the node are all offset by one.
  unsigned OpNo = 0;
  if (N->NodeHasProperty(SDNPHasChain, CGP)) {
    // Record the node and remember it in our chained nodes list.
    AddMatcherNode(new RecordMatcherNode("'" + N->getOperator()->getName() +
                                         "' chained node"));
    // Remember all of the input chains our pattern will match.
    MatchedChainNodes.push_back(NextRecordedOperandNo++);
    
    // If this is the second (e.g. indbr(load) or store(add(load))) or third
    // input chain (e.g. (store (add (load, load))) from msp430) we need to make
    // sure that folding the chain won't induce cycles in the DAG.  This could
    // happen if there were an intermediate node between the indbr and load, for
    // example.
    if (MatchedChainNodes.size() > 1) {
      // FIXME2: This is broken, we should eliminate this nonsense completely,
      // but we want to produce the same selections that the old matcher does
      // for now.
      unsigned PrevOp = MatchedChainNodes[MatchedChainNodes.size()-2];
      AddMatcherNode(new CheckChainCompatibleMatcherNode(PrevOp));
    }
    
    // Don't look at the input chain when matching the tree pattern to the
    // SDNode.
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

  // If this node has an output flag and isn't the root, remember it.
  if (N->NodeHasProperty(SDNPOutFlag, CGP) && 
      N != Pattern.getSrcPattern()) {
    // TODO: This redundantly records nodes with both flags and chains.
    
    // Record the node and remember it in our chained nodes list.
    AddMatcherNode(new RecordMatcherNode("'" + N->getOperator()->getName() +
                                         "' flag output node"));
    // Remember all of the nodes with output flags our pattern will match.
    MatchedFlagResultNodes.push_back(NextRecordedOperandNo++);
  }
  
  // If this node is known to have an input flag or if it *might* have an input
  // flag, capture it as the flag input of the pattern.
  if (N->NodeHasProperty(SDNPOptInFlag, CGP) ||
      N->NodeHasProperty(SDNPInFlag, CGP))
    AddMatcherNode(new CaptureFlagInputMatcherNode());
      
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i, ++OpNo) {
    // Get the code suitable for matching this child.  Move to the child, check
    // it then move back to the parent.
    AddMatcherNode(new MoveChildMatcherNode(OpNo));
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
      // If it is a named node, we must emit a 'Record' opcode.
      VarMapEntry = ++NextRecordedOperandNo;
      AddMatcherNode(new RecordMatcherNode("$" + N->getName()));
    } else {
      // If we get here, this is a second reference to a specific name.  Since
      // we already have checked that the first reference is valid, we don't
      // have to recursively match it, just check that it's the same as the
      // previously named thing.
      AddMatcherNode(new CheckSameMatcherNode(VarMapEntry-1));
      return;
    }
  }
  
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


//===----------------------------------------------------------------------===//
// Node Result Generation
//===----------------------------------------------------------------------===//

void MatcherGen::EmitResultOfNamedOperand(const TreePatternNode *N,
                                          SmallVectorImpl<unsigned> &ResultOps){
  assert(!N->getName().empty() && "Operand not named!");
  
  unsigned SlotNo = getNamedArgumentSlot(N->getName());
  
  // A reference to a complex pattern gets all of the results of the complex
  // pattern's match.
  if (const ComplexPattern *CP = N->getComplexPatternInfo(CGP)) {
    // The first slot entry is the node itself, the subsequent entries are the
    // matched values.
    for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
      ResultOps.push_back(SlotNo+i+1);
    return;
  }

  // If this is an 'imm' or 'fpimm' node, make sure to convert it to the target
  // version of the immediate so that it doesn't get selected due to some other
  // node use.
  if (!N->isLeaf()) {
    StringRef OperatorName = N->getOperator()->getName();
    if (OperatorName == "imm" || OperatorName == "fpimm") {
      AddMatcherNode(new EmitConvertToTargetMatcherNode(SlotNo));
      ResultOps.push_back(NextRecordedOperandNo++);
      return;
    }
  }
  
  ResultOps.push_back(SlotNo);
}

void MatcherGen::EmitResultLeafAsOperand(const TreePatternNode *N,
                                         SmallVectorImpl<unsigned> &ResultOps) {
  assert(N->isLeaf() && "Must be a leaf");
  
  if (IntInit *II = dynamic_cast<IntInit*>(N->getLeafValue())) {
    AddMatcherNode(new EmitIntegerMatcherNode(II->getValue(),N->getTypeNum(0)));
    ResultOps.push_back(NextRecordedOperandNo++);
    return;
  }
  
  // If this is an explicit register reference, handle it.
  if (DefInit *DI = dynamic_cast<DefInit*>(N->getLeafValue())) {
    if (DI->getDef()->isSubClassOf("Register")) {
      AddMatcherNode(new EmitRegisterMatcherNode(DI->getDef(),
                                                 N->getTypeNum(0)));
      ResultOps.push_back(NextRecordedOperandNo++);
      return;
    }
    
    if (DI->getDef()->getName() == "zero_reg") {
      AddMatcherNode(new EmitRegisterMatcherNode(0, N->getTypeNum(0)));
      ResultOps.push_back(NextRecordedOperandNo++);
      return;
    }
    
    // Handle a reference to a register class. This is used
    // in COPY_TO_SUBREG instructions.
    if (DI->getDef()->isSubClassOf("RegisterClass")) {
      std::string Value = getQualifiedName(DI->getDef()) + "RegClassID";
      AddMatcherNode(new EmitStringIntegerMatcherNode(Value, MVT::i32));
      ResultOps.push_back(NextRecordedOperandNo++);
      return;
    }
  }
  
  errs() << "unhandled leaf node: \n";
  N->dump();
}

/// GetInstPatternNode - Get the pattern for an instruction.
/// 
const TreePatternNode *MatcherGen::
GetInstPatternNode(const DAGInstruction &Inst, const TreePatternNode *N) {
  const TreePattern *InstPat = Inst.getPattern();
  
  // FIXME2?: Assume actual pattern comes before "implicit".
  TreePatternNode *InstPatNode;
  if (InstPat)
    InstPatNode = InstPat->getTree(0);
  else if (/*isRoot*/ N == Pattern.getDstPattern())
    InstPatNode = Pattern.getSrcPattern();
  else
    return 0;
  
  if (InstPatNode && !InstPatNode->isLeaf() &&
      InstPatNode->getOperator()->getName() == "set")
    InstPatNode = InstPatNode->getChild(InstPatNode->getNumChildren()-1);
  
  return InstPatNode;
}

void MatcherGen::
EmitResultInstructionAsOperand(const TreePatternNode *N,
                               SmallVectorImpl<unsigned> &OutputOps) {
  Record *Op = N->getOperator();
  const CodeGenTarget &CGT = CGP.getTargetInfo();
  CodeGenInstruction &II = CGT.getInstruction(Op->getName());
  const DAGInstruction &Inst = CGP.getInstruction(Op);
  
  // If we can, get the pattern for the instruction we're generating.  We derive
  // a variety of information from this pattern, such as whether it has a chain.
  //
  // FIXME2: This is extremely dubious for several reasons, not the least of
  // which it gives special status to instructions with patterns that Pat<>
  // nodes can't duplicate.
  const TreePatternNode *InstPatNode = GetInstPatternNode(Inst, N);

  // NodeHasChain - Whether the instruction node we're creating takes chains.  
  bool NodeHasChain = InstPatNode &&
                      InstPatNode->TreeHasProperty(SDNPHasChain, CGP);
  
  bool isRoot = N == Pattern.getDstPattern();

  // TreeHasOutFlag - True if this tree has a flag.
  bool TreeHasInFlag = false, TreeHasOutFlag = false;
  if (isRoot) {
    const TreePatternNode *SrcPat = Pattern.getSrcPattern();
    TreeHasInFlag = SrcPat->TreeHasProperty(SDNPOptInFlag, CGP) ||
                    SrcPat->TreeHasProperty(SDNPInFlag, CGP);
  
    // FIXME2: this is checking the entire pattern, not just the node in
    // question, doing this just for the root seems like a total hack.
    TreeHasOutFlag = SrcPat->TreeHasProperty(SDNPOutFlag, CGP);
  }

  // NumResults - This is the number of results produced by the instruction in
  // the "outs" list.
  unsigned NumResults = Inst.getNumResults();    

  // Loop over all of the operands of the instruction pattern, emitting code
  // to fill them all in.  The node 'N' usually has number children equal to
  // the number of input operands of the instruction.  However, in cases
  // where there are predicate operands for an instruction, we need to fill
  // in the 'execute always' values.  Match up the node operands to the
  // instruction operands to do this.
  SmallVector<unsigned, 8> InstOps;
  for (unsigned ChildNo = 0, InstOpNo = NumResults, e = II.OperandList.size();
       InstOpNo != e; ++InstOpNo) {
    
    // Determine what to emit for this operand.
    Record *OperandNode = II.OperandList[InstOpNo].Rec;
    if ((OperandNode->isSubClassOf("PredicateOperand") ||
         OperandNode->isSubClassOf("OptionalDefOperand")) &&
        !CGP.getDefaultOperand(OperandNode).DefaultOps.empty()) {
      // This is a predicate or optional def operand; emit the
      // 'default ops' operands.
      const DAGDefaultOperand &DefaultOp =
        CGP.getDefaultOperand(II.OperandList[InstOpNo].Rec);
      for (unsigned i = 0, e = DefaultOp.DefaultOps.size(); i != e; ++i)
        EmitResultOperand(DefaultOp.DefaultOps[i], InstOps);
      continue;
    }
    
    // Otherwise this is a normal operand or a predicate operand without
    // 'execute always'; emit it.
    EmitResultOperand(N->getChild(ChildNo), InstOps);
    ++ChildNo;
  }
  
  // Nodes that match patterns with (potentially multiple) chain inputs have to
  // merge them together into a token factor.
  if (NodeHasChain && !EmittedMergeInputChains) {
    // FIXME2: Move this out of emitresult to a top level place.
    assert(!MatchedChainNodes.empty() &&
           "How can this node have chain if no inputs do?");
    // Otherwise, we have to emit an operation to merge the input chains and
    // set this as the current input chain.
    AddMatcherNode(new EmitMergeInputChainsMatcherNode
                        (MatchedChainNodes.data(), MatchedChainNodes.size()));
    EmittedMergeInputChains = true;
  }
  
  // If this node has an input flag or explicitly specified input physregs, we
  // need to add chained and flagged copyfromreg nodes and materialize the flag
  // input.
  if (isRoot && !PhysRegInputs.empty()) {
    // Emit all of the CopyToReg nodes for the input physical registers.  These
    // occur in patterns like (mul:i8 AL:i8, GR8:i8:$src).
    for (unsigned i = 0, e = PhysRegInputs.size(); i != e; ++i)
      AddMatcherNode(new EmitCopyToRegMatcherNode(PhysRegInputs[i].second,
                                                  PhysRegInputs[i].first));
    // Even if the node has no other flag inputs, the resultant node must be
    // flagged to the CopyFromReg nodes we just generated.
    TreeHasInFlag = true;
  }
  
  // Result order: node results, chain, flags
  
  // Determine the result types.
  SmallVector<MVT::SimpleValueType, 4> ResultVTs;
  if (NumResults != 0 && N->getTypeNum(0) != MVT::isVoid) {
    // FIXME2: If the node has multiple results, we should add them.  For now,
    // preserve existing behavior?!
    ResultVTs.push_back(N->getTypeNum(0));
  }

  
  // If this is the root instruction of a pattern that has physical registers in
  // its result pattern, add output VTs for them.  For example, X86 has:
  //   (set AL, (mul ...))
  // This also handles implicit results like:
  //   (implicit EFLAGS)
  if (isRoot && Pattern.getDstRegs().size() != 0) {
    for (unsigned i = 0; i != Pattern.getDstRegs().size(); ++i)
      if (Pattern.getDstRegs()[i]->isSubClassOf("Register"))
        ResultVTs.push_back(getRegisterValueType(Pattern.getDstRegs()[i], CGT));
  }
  if (NodeHasChain)
    ResultVTs.push_back(MVT::Other);
  if (TreeHasOutFlag)
    ResultVTs.push_back(MVT::Flag);

  // FIXME2: Instead of using the isVariadic flag on the instruction, we should
  // have an SDNP that indicates variadicism.  The TargetInstrInfo isVariadic
  // property should be inferred from this when an instruction has a pattern.
  int NumFixedArityOperands = -1;
  if (isRoot && II.isVariadic)
    NumFixedArityOperands = Pattern.getSrcPattern()->getNumChildren();
  
  // If this is the root node and any of the nodes matched nodes in the input
  // pattern have MemRefs in them, have the interpreter collect them and plop
  // them onto this node.
  //
  // FIXME3: This is actively incorrect for result patterns where the root of
  // the pattern is not the memory reference and is also incorrect when the
  // result pattern has multiple memory-referencing instructions.  For example,
  // in the X86 backend, this pattern causes the memrefs to get attached to the
  // CVTSS2SDrr instead of the MOVSSrm:
  //
  //  def : Pat<(extloadf32 addr:$src),
  //            (CVTSS2SDrr (MOVSSrm addr:$src))>;
  //
  bool NodeHasMemRefs =
    isRoot && Pattern.getSrcPattern()->TreeHasProperty(SDNPMemOperand, CGP);

  // FIXME: Eventually add a SelectNodeTo form.  It works if the new node has a
  // superset of the results of the old node, in the same places.  E.g. turning
  // (add (load)) -> add32rm is ok because result #0 is the result and result #1
  // is new.
  AddMatcherNode(new EmitNodeMatcherNode(II.Namespace+"::"+II.TheDef->getName(),
                                         ResultVTs.data(), ResultVTs.size(),
                                         InstOps.data(), InstOps.size(),
                                         NodeHasChain, TreeHasInFlag,
                                         NodeHasMemRefs,NumFixedArityOperands));
  
  // The non-chain and non-flag results of the newly emitted node get recorded.
  for (unsigned i = 0, e = ResultVTs.size(); i != e; ++i) {
    if (ResultVTs[i] == MVT::Other || ResultVTs[i] == MVT::Flag) break;
    OutputOps.push_back(NextRecordedOperandNo++);
  }
  
  // FIXME2: Kill off all the SelectionDAG::SelectNodeTo and getMachineNode
  // variants.  Call MorphNodeTo instead of SelectNodeTo.
}

void MatcherGen::
EmitResultSDNodeXFormAsOperand(const TreePatternNode *N,
                               SmallVectorImpl<unsigned> &ResultOps) {
  assert(N->getOperator()->isSubClassOf("SDNodeXForm") && "Not SDNodeXForm?");

  // Emit the operand.
  SmallVector<unsigned, 8> InputOps;
  
  // FIXME2: Could easily generalize this to support multiple inputs and outputs
  // to the SDNodeXForm.  For now we just support one input and one output like
  // the old instruction selector.
  assert(N->getNumChildren() == 1);
  EmitResultOperand(N->getChild(0), InputOps);

  // The input currently must have produced exactly one result.
  assert(InputOps.size() == 1 && "Unexpected input to SDNodeXForm");

  AddMatcherNode(new EmitNodeXFormMatcherNode(InputOps[0], N->getOperator()));
  ResultOps.push_back(NextRecordedOperandNo++);
}

void MatcherGen::EmitResultOperand(const TreePatternNode *N,
                                   SmallVectorImpl<unsigned> &ResultOps) {
  // This is something selected from the pattern we matched.
  if (!N->getName().empty())
    return EmitResultOfNamedOperand(N, ResultOps);

  if (N->isLeaf())
    return EmitResultLeafAsOperand(N, ResultOps);

  Record *OpRec = N->getOperator();
  if (OpRec->isSubClassOf("Instruction"))
    return EmitResultInstructionAsOperand(N, ResultOps);
  if (OpRec->isSubClassOf("SDNodeXForm"))
    return EmitResultSDNodeXFormAsOperand(N, ResultOps);
  errs() << "Unknown result node to emit code for: " << *N << '\n';
  throw std::string("Unknown node in result pattern!");
}

void MatcherGen::EmitResultCode() {
  // Codegen the root of the result pattern, capturing the resulting values.
  SmallVector<unsigned, 8> Ops;
  EmitResultOperand(Pattern.getDstPattern(), Ops);

  // At this point, we have however many values the result pattern produces.
  // However, the input pattern might not need all of these.  If there are
  // excess values at the end (such as condition codes etc) just lop them off.
  // This doesn't need to worry about flags or chains, just explicit results.
  //
  // FIXME2: This doesn't work because there is currently no way to get an
  // accurate count of the # results the source pattern sets.  This is because
  // of the "parallel" construct in X86 land, which looks like this:
  //
  //def : Pat<(parallel (X86and_flag GR8:$src1, GR8:$src2),
  //           (implicit EFLAGS)),
  //  (AND8rr GR8:$src1, GR8:$src2)>;
  //
  // This idiom means to match the two-result node X86and_flag (which is
  // declared as returning a single result, because we can't match multi-result
  // nodes yet).  In this case, we would have to know that the input has two
  // results.  However, mul8r is modelled exactly the same way, but without
  // implicit defs included.  The fix is to support multiple results directly
  // and eliminate 'parallel'.
  //
  // FIXME2: When this is fixed, we should revert the terrible hack in the
  // OPC_EmitNode code in the interpreter.
#if 0
  const TreePatternNode *Src = Pattern.getSrcPattern();
  unsigned NumSrcResults = Src->getTypeNum(0) != MVT::isVoid ? 1 : 0;
  NumSrcResults += Pattern.getDstRegs().size();
  assert(Ops.size() >= NumSrcResults && "Didn't provide enough results");
  Ops.resize(NumSrcResults);
#endif

  // If the matched pattern covers nodes which define a flag result, emit a node
  // that tells the matcher about them so that it can update their results.
  if (!MatchedFlagResultNodes.empty())
    AddMatcherNode(new MarkFlagResultsMatcherNode(MatchedFlagResultNodes.data(),
                                                MatchedFlagResultNodes.size()));
  
  
  // We know that the resulting pattern has exactly one result/
  // FIXME2: why?  what about something like (set a,b,c, (complexpat))
  // FIXME2: Implicit results should be pushed here I guess?
  AddMatcherNode(new CompleteMatchMatcherNode(Ops.data(), Ops.size(), Pattern));
}


MatcherNode *llvm::ConvertPatternToMatcher(const PatternToMatch &Pattern,
                                           const CodeGenDAGPatterns &CGP) {
  MatcherGen Gen(Pattern, CGP);

  // Generate the code for the matcher.
  Gen.EmitMatcherCode();
  
  
  // FIXME2: Kill extra MoveParent commands at the end of the matcher sequence.
  // FIXME2: Split result code out to another table, and make the matcher end
  // with an "Emit <index>" command.  This allows result generation stuff to be
  // shared and factored?
  
  // If the match succeeds, then we generate Pattern.
  Gen.EmitResultCode();

  // Unconditional match.
  return Gen.GetMatcher();
}



