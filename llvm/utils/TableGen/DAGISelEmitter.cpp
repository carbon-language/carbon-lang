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
#include "DAGISelMatcher.h"
#include "Record.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// DAGISelEmitter Helper methods
//

/// getPatternSize - Return the 'size' of this pattern.  We want to match large
/// patterns before small ones.  This is used to determine the size of a
/// pattern.
static unsigned getPatternSize(TreePatternNode *P, CodeGenDAGPatterns &CGP) {
  assert((EEVT::isExtIntegerInVTs(P->getExtTypes()) ||
          EEVT::isExtFloatingPointInVTs(P->getExtTypes()) ||
          P->getExtTypeNum(0) == MVT::isVoid ||
          P->getExtTypeNum(0) == MVT::Flag ||
          P->getExtTypeNum(0) == MVT::iPTR ||
          P->getExtTypeNum(0) == MVT::iPTRAny) && 
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
  const ComplexPattern *AM = P->getComplexPatternInfo(CGP);
  if (AM)
    Size += AM->getNumOperands() * 3;

  // If this node has some predicate function that must match, it adds to the
  // complexity of this node.
  if (!P->getPredicateFns().empty())
    ++Size;
  
  // Count children in the count if they are also nodes.
  for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = P->getChild(i);
    if (!Child->isLeaf() && Child->getExtTypeNum(0) != MVT::Other)
      Size += getPatternSize(Child, CGP);
    else if (Child->isLeaf()) {
      if (dynamic_cast<IntInit*>(Child->getLeafValue())) 
        Size += 5;  // Matches a ConstantSDNode (+3) and a specific value (+2).
      else if (Child->getComplexPatternInfo(CGP))
        Size += getPatternSize(Child, CGP);
      else if (!Child->getPredicateFns().empty())
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
    if (II.usesCustomInserter)
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

//===----------------------------------------------------------------------===//
// Predicate emitter implementation.
//

void DAGISelEmitter::EmitPredicateFunctions(raw_ostream &OS) {
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
      << "(SDNode *N) const {\n";
    else {
      std::string ClassName =
        CGP.getSDNodeInfo(P->getOnlyTree()->getOperator()).getSDClassName();
      const char *C2 = ClassName == "SDNode" ? "N" : "inN";
      
      OS << "inline bool Predicate_" << PatFragRecord->getName()
         << "(SDNode *" << C2 << ") const {\n";
      if (ClassName != "SDNode")
        OS << "  " << ClassName << " *N = cast<" << ClassName << ">(inN);\n";
    }
    OS << Code << "\n}\n";
  }
  
  OS << "\n\n";
}

namespace {
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
    
    unsigned LHSPatSize = getResultPatternSize(LHS->getDstPattern(), CGP);
    unsigned RHSPatSize = getResultPatternSize(RHS->getDstPattern(), CGP);
    if (LHSPatSize < RHSPatSize) return true;
    if (LHSPatSize > RHSPatSize) return false;
    
    // Sort based on the UID of the pattern, giving us a deterministic ordering.
    assert(LHS->ID != RHS->ID);
    return LHS->ID < RHS->ID;
  }
};
}


void DAGISelEmitter::run(raw_ostream &OS) {
  EmitSourceFileHeader("DAG Instruction Selector for the " +
                       CGP.getTargetInfo().getName() + " target", OS);
  
  OS << "// *** NOTE: This file is #included into the middle of the target\n"
     << "// *** instruction selector class.  These functions are really "
     << "methods.\n\n";

  OS << "// Include standard, target-independent definitions and methods used\n"
     << "// by the instruction selector.\n";
  OS << "#include \"llvm/CodeGen/DAGISelHeader.h\"\n\n";
  
  DEBUG(errs() << "\n\nALL PATTERNS TO MATCH:\n\n";
        for (CodeGenDAGPatterns::ptm_iterator I = CGP.ptm_begin(),
             E = CGP.ptm_end(); I != E; ++I) {
          errs() << "PATTERN: ";   I->getSrcPattern()->dump();
          errs() << "\nRESULT:  "; I->getDstPattern()->dump();
          errs() << "\n";
        });

  // FIXME: These are being used by hand written code, gross.
  EmitPredicateFunctions(OS);

  // Add all the patterns to a temporary list so we can sort them.
  std::vector<const PatternToMatch*> Patterns;
  for (CodeGenDAGPatterns::ptm_iterator I = CGP.ptm_begin(), E = CGP.ptm_end();
       I != E; ++I)
    Patterns.push_back(&*I);

  // We want to process the matches in order of minimal cost.  Sort the patterns
  // so the least cost one is at the start.
  std::stable_sort(Patterns.begin(), Patterns.end(),
                   PatternSortingPredicate(CGP));
  
  
  // Convert each variant of each pattern into a Matcher.
  std::vector<Matcher*> PatternMatchers;
  for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
    for (unsigned Variant = 0; ; ++Variant) {
      if (Matcher *M = ConvertPatternToMatcher(*Patterns[i], Variant, CGP))
        PatternMatchers.push_back(M);
      else
        break;
    }
  }
          
  Matcher *TheMatcher = new ScopeMatcher(&PatternMatchers[0],
                                         PatternMatchers.size());

  TheMatcher = OptimizeMatcher(TheMatcher, CGP);
  //Matcher->dump();
  EmitMatcherTable(TheMatcher, CGP, OS);
  delete TheMatcher;
}
