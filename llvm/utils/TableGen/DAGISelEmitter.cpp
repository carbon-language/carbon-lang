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

/// CouldMatchSameInput - Return true if it is possible for these two patterns
/// to match the same input.  For example, (add reg, reg) and
///   (add reg, (mul ...)) could both match the same input.  Where this is
/// conservative, it falls back to returning true.
static bool CouldMatchSameInput(const TreePatternNode *N1,
                                const TreePatternNode *N2) {
  // If the types of the two nodes differ, they can't match the same thing.
  if (N1->getNumTypes() != N2->getNumTypes()) return false;
  for (unsigned i = 0, e = N1->getNumTypes(); i != e; ++i)
    if (N1->getType(i) != N2->getType(i))
      return false;
  
  // Handle the case when at least one is a leaf.
  if (N1->isLeaf()) {
    if (N2->isLeaf()) {
      // Handle leaf/leaf cases.  Register operands can match just about
      // anything, so we can only disambiguate a few things here.
      
      // If both operands are leaf integer nodes with different values, they
      // can't match the same thing.
      if (IntInit *II1 = dynamic_cast<IntInit*>(N1->getLeafValue()))
        if (IntInit *II2 = dynamic_cast<IntInit*>(N2->getLeafValue()))
          return II1->getValue() == II2->getValue();
      
      DefInit *DI1 = dynamic_cast<DefInit*>(N1->getLeafValue());
      DefInit *DI2 = dynamic_cast<DefInit*>(N2->getLeafValue());
      if (DI1 != 0 && DI2 != 0) {
        if (DI1->getDef()->isSubClassOf("ValueType") &&
            DI2->getDef()->isSubClassOf("ValueType"))
          return DI1 == DI2;
        if (DI1->getDef()->isSubClassOf("CondCode") &&
            DI2->getDef()->isSubClassOf("CondCode"))
          return DI1 == DI2;
      }

      // TODO: Regclass cannot match a condcode etc.
      
      // Otherwise, complex pattern could match anything, so just return a
      // conservative response.
      return true;
    }
    
    // Conservatively return true.  (imm) could match "7" for example, and GPR
    // can match anything.
    // TODO: could handle (add ...)  != "1" if we cared.
    return true;
  }
  
  // If N2 is a leaf and N1 isn't, check the other way.
  if (N2->isLeaf())
    return CouldMatchSameInput(N2, N1);
  
  // Now we know neither node is a leaf.  If the two patterns have different
  // number of children or different operators, they can't both match.
  Record *Op1 = N1->getOperator(), *Op2 = N1->getOperator();
  
  if (Op1 != Op2 || N1->getNumChildren() != N2->getNumChildren())
    return false;

  // If a child prevents the two patterns from matching, use that.
  for (unsigned i = 0, e = N1->getNumChildren(); i != e; ++i)
    if (!CouldMatchSameInput(N1->getChild(i), N2->getChild(i)))
      return false;
  
  // Otherwise, it looks like they could both match the same thing.
  return true;
}

/// GetSourceMatchPreferenceOrdering - The two input patterns are guaranteed to
/// not match the same input.  Decide which pattern we'd prefer to match first
/// in order to reduce compile time.  This sorting predicate is used to improve
/// compile time so that we try to match scalar operations before vector
/// operations since scalar operations are much more common in practice.
///
/// This returns -1 if we prefer to match N1 before N2, 1 if we prefer to match
/// N2 before N1 or 0 if no preference.
///
static int GetSourceMatchPreferenceOrdering(const TreePatternNode *N1,
                                            const TreePatternNode *N2) {
  // The primary thing we sort on here is to get ints before floats and scalars
  // before vectors.
  for (unsigned i = 0, e = std::min(N1->getNumTypes(), N2->getNumTypes());
       i != e; ++i)
    if (N1->getType(i) != N2->getType(i)) {
      MVT::SimpleValueType V1 = N1->getType(i), V2 = N2->getType(i);
      if (MVT(V1).isVector() != MVT(V2).isVector())
        return MVT(V1).isVector() ? 1 : -1;
      
      if (MVT(V1).isFloatingPoint() != MVT(V2).isFloatingPoint())
        return MVT(V1).isFloatingPoint() ? 1 : -1;
    }
  
  for (unsigned i = 0, e = std::min(N1->getNumChildren(), N2->getNumChildren());
       i != e; ++i)
    if (int Res = GetSourceMatchPreferenceOrdering(N1->getChild(i),
                                                   N2->getChild(i)))
      return Res;
  return 0;
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
    
    // If the patterns are guaranteed to not match at the same time and we
    // prefer to match one before the other (for compile time reasons) use this
    // preference as our discriminator.
    if (0 && !CouldMatchSameInput(LHSSrc, RHSSrc)) {
      int Ordering = GetSourceMatchPreferenceOrdering(LHSSrc, RHSSrc);
      if (Ordering != 0) {
        if (Ordering == -1) {
          errs() << "SORT: " << *LHSSrc << "\n";
          errs() << "NEXT: " << *RHSSrc << "\n\n";
        } else {
          errs() << "SORT: " << *RHSSrc << "\n";
          errs() << "NEXT: " << *LHSSrc << "\n\n";
        }
      }
      
      if (Ordering == -1) return true;
      if (Ordering == 1) return false;
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
