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
    CodeGenInstruction &II = CGP.getTargetInfo().getInstruction(Op);
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
  
  bool operator()(const PatternToMatch *LHS, const PatternToMatch *RHS) {
    const TreePatternNode *LHSSrc = LHS->getSrcPattern();
    const TreePatternNode *RHSSrc = RHS->getSrcPattern();
    
    if (LHSSrc->getNumTypes() != 0 && RHSSrc->getNumTypes() != 0 &&
        LHSSrc->getType(0) != RHSSrc->getType(0)) {
      MVT::SimpleValueType V1 = LHSSrc->getType(0), V2 = RHSSrc->getType(0);
      if (MVT(V1).isVector() != MVT(V2).isVector())
        return MVT(V2).isVector();
      
      if (MVT(V1).isFloatingPoint() != MVT(V2).isFloatingPoint())
        return MVT(V2).isFloatingPoint();
    }
    
    // Otherwise, if the patterns might both match, sort based on complexity,
    // which means that we prefer to match patterns that cover more nodes in the
    // input over nodes that cover fewer.
    unsigned LHSSize = LHS->getPatternComplexity(CGP);
    unsigned RHSSize = RHS->getPatternComplexity(CGP);
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
    
    // Sort based on the UID of the pattern, giving us a deterministic ordering
    // if all other sorting conditions fail.
    assert(LHS == RHS || LHS->ID != RHS->ID);
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
  std::sort(Patterns.begin(), Patterns.end(), PatternSortingPredicate(CGP));
  
  
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
