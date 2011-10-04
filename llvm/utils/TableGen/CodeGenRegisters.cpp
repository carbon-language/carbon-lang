//===- CodeGenRegisters.cpp - Register and RegisterClass Info -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate information gleaned from the
// target register and register class definitions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenRegisters.h"
#include "CodeGenTarget.h"
#include "llvm/TableGen/Error.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//                              CodeGenRegister
//===----------------------------------------------------------------------===//

CodeGenRegister::CodeGenRegister(Record *R, unsigned Enum)
  : TheDef(R),
    EnumValue(Enum),
    CostPerUse(R->getValueAsInt("CostPerUse")),
    SubRegsComplete(false)
{}

const std::string &CodeGenRegister::getName() const {
  return TheDef->getName();
}

namespace {
  struct Orphan {
    CodeGenRegister *SubReg;
    Record *First, *Second;
    Orphan(CodeGenRegister *r, Record *a, Record *b)
      : SubReg(r), First(a), Second(b) {}
  };
}

const CodeGenRegister::SubRegMap &
CodeGenRegister::getSubRegs(CodeGenRegBank &RegBank) {
  // Only compute this map once.
  if (SubRegsComplete)
    return SubRegs;
  SubRegsComplete = true;

  std::vector<Record*> SubList = TheDef->getValueAsListOfDefs("SubRegs");
  std::vector<Record*> Indices = TheDef->getValueAsListOfDefs("SubRegIndices");
  if (SubList.size() != Indices.size())
    throw TGError(TheDef->getLoc(), "Register " + getName() +
                  " SubRegIndices doesn't match SubRegs");

  // First insert the direct subregs and make sure they are fully indexed.
  for (unsigned i = 0, e = SubList.size(); i != e; ++i) {
    CodeGenRegister *SR = RegBank.getReg(SubList[i]);
    if (!SubRegs.insert(std::make_pair(Indices[i], SR)).second)
      throw TGError(TheDef->getLoc(), "SubRegIndex " + Indices[i]->getName() +
                    " appears twice in Register " + getName());
  }

  // Keep track of inherited subregs and how they can be reached.
  SmallVector<Orphan, 8> Orphans;

  // Clone inherited subregs and place duplicate entries on Orphans.
  // Here the order is important - earlier subregs take precedence.
  for (unsigned i = 0, e = SubList.size(); i != e; ++i) {
    CodeGenRegister *SR = RegBank.getReg(SubList[i]);
    const SubRegMap &Map = SR->getSubRegs(RegBank);

    // Add this as a super-register of SR now all sub-registers are in the list.
    // This creates a topological ordering, the exact order depends on the
    // order getSubRegs is called on all registers.
    SR->SuperRegs.push_back(this);

    for (SubRegMap::const_iterator SI = Map.begin(), SE = Map.end(); SI != SE;
         ++SI) {
      if (!SubRegs.insert(*SI).second)
        Orphans.push_back(Orphan(SI->second, Indices[i], SI->first));

      // Noop sub-register indexes are possible, so avoid duplicates.
      if (SI->second != SR)
        SI->second->SuperRegs.push_back(this);
    }
  }

  // Process the composites.
  ListInit *Comps = TheDef->getValueAsListInit("CompositeIndices");
  for (unsigned i = 0, e = Comps->size(); i != e; ++i) {
    DagInit *Pat = dynamic_cast<DagInit*>(Comps->getElement(i));
    if (!Pat)
      throw TGError(TheDef->getLoc(), "Invalid dag '" +
                    Comps->getElement(i)->getAsString() +
                    "' in CompositeIndices");
    DefInit *BaseIdxInit = dynamic_cast<DefInit*>(Pat->getOperator());
    if (!BaseIdxInit || !BaseIdxInit->getDef()->isSubClassOf("SubRegIndex"))
      throw TGError(TheDef->getLoc(), "Invalid SubClassIndex in " +
                    Pat->getAsString());

    // Resolve list of subreg indices into R2.
    CodeGenRegister *R2 = this;
    for (DagInit::const_arg_iterator di = Pat->arg_begin(),
         de = Pat->arg_end(); di != de; ++di) {
      DefInit *IdxInit = dynamic_cast<DefInit*>(*di);
      if (!IdxInit || !IdxInit->getDef()->isSubClassOf("SubRegIndex"))
        throw TGError(TheDef->getLoc(), "Invalid SubClassIndex in " +
                      Pat->getAsString());
      const SubRegMap &R2Subs = R2->getSubRegs(RegBank);
      SubRegMap::const_iterator ni = R2Subs.find(IdxInit->getDef());
      if (ni == R2Subs.end())
        throw TGError(TheDef->getLoc(), "Composite " + Pat->getAsString() +
                      " refers to bad index in " + R2->getName());
      R2 = ni->second;
    }

    // Insert composite index. Allow overriding inherited indices etc.
    SubRegs[BaseIdxInit->getDef()] = R2;

    // R2 is no longer an orphan.
    for (unsigned j = 0, je = Orphans.size(); j != je; ++j)
      if (Orphans[j].SubReg == R2)
          Orphans[j].SubReg = 0;
  }

  // Now Orphans contains the inherited subregisters without a direct index.
  // Create inferred indexes for all missing entries.
  for (unsigned i = 0, e = Orphans.size(); i != e; ++i) {
    Orphan &O = Orphans[i];
    if (!O.SubReg)
      continue;
    SubRegs[RegBank.getCompositeSubRegIndex(O.First, O.Second, true)] =
      O.SubReg;
  }
  return SubRegs;
}

void
CodeGenRegister::addSubRegsPreOrder(SetVector<CodeGenRegister*> &OSet) const {
  assert(SubRegsComplete && "Must precompute sub-registers");
  std::vector<Record*> Indices = TheDef->getValueAsListOfDefs("SubRegIndices");
  for (unsigned i = 0, e = Indices.size(); i != e; ++i) {
    CodeGenRegister *SR = SubRegs.find(Indices[i])->second;
    if (OSet.insert(SR))
      SR->addSubRegsPreOrder(OSet);
  }
}

//===----------------------------------------------------------------------===//
//                               RegisterTuples
//===----------------------------------------------------------------------===//

// A RegisterTuples def is used to generate pseudo-registers from lists of
// sub-registers. We provide a SetTheory expander class that returns the new
// registers.
namespace {
struct TupleExpander : SetTheory::Expander {
  void expand(SetTheory &ST, Record *Def, SetTheory::RecSet &Elts) {
    std::vector<Record*> Indices = Def->getValueAsListOfDefs("SubRegIndices");
    unsigned Dim = Indices.size();
    ListInit *SubRegs = Def->getValueAsListInit("SubRegs");
    if (Dim != SubRegs->getSize())
      throw TGError(Def->getLoc(), "SubRegIndices and SubRegs size mismatch");
    if (Dim < 2)
      throw TGError(Def->getLoc(), "Tuples must have at least 2 sub-registers");

    // Evaluate the sub-register lists to be zipped.
    unsigned Length = ~0u;
    SmallVector<SetTheory::RecSet, 4> Lists(Dim);
    for (unsigned i = 0; i != Dim; ++i) {
      ST.evaluate(SubRegs->getElement(i), Lists[i]);
      Length = std::min(Length, unsigned(Lists[i].size()));
    }

    if (Length == 0)
      return;

    // Precompute some types.
    Record *RegisterCl = Def->getRecords().getClass("Register");
    RecTy *RegisterRecTy = RecordRecTy::get(RegisterCl);
    StringInit *BlankName = StringInit::get("");

    // Zip them up.
    for (unsigned n = 0; n != Length; ++n) {
      std::string Name;
      Record *Proto = Lists[0][n];
      std::vector<Init*> Tuple;
      unsigned CostPerUse = 0;
      for (unsigned i = 0; i != Dim; ++i) {
        Record *Reg = Lists[i][n];
        if (i) Name += '_';
        Name += Reg->getName();
        Tuple.push_back(DefInit::get(Reg));
        CostPerUse = std::max(CostPerUse,
                              unsigned(Reg->getValueAsInt("CostPerUse")));
      }

      // Create a new Record representing the synthesized register. This record
      // is only for consumption by CodeGenRegister, it is not added to the
      // RecordKeeper.
      Record *NewReg = new Record(Name, Def->getLoc(), Def->getRecords());
      Elts.insert(NewReg);

      // Copy Proto super-classes.
      for (unsigned i = 0, e = Proto->getSuperClasses().size(); i != e; ++i)
        NewReg->addSuperClass(Proto->getSuperClasses()[i]);

      // Copy Proto fields.
      for (unsigned i = 0, e = Proto->getValues().size(); i != e; ++i) {
        RecordVal RV = Proto->getValues()[i];

        // Replace the sub-register list with Tuple.
        if (RV.getName() == "SubRegs")
          RV.setValue(ListInit::get(Tuple, RegisterRecTy));

        // Provide a blank AsmName. MC hacks are required anyway.
        if (RV.getName() == "AsmName")
          RV.setValue(BlankName);

        // CostPerUse is aggregated from all Tuple members.
        if (RV.getName() == "CostPerUse")
          RV.setValue(IntInit::get(CostPerUse));

        // Copy fields from the RegisterTuples def.
        if (RV.getName() == "SubRegIndices" ||
            RV.getName() == "CompositeIndices") {
          NewReg->addValue(*Def->getValue(RV.getName()));
          continue;
        }

        // Some fields get their default uninitialized value.
        if (RV.getName() == "DwarfNumbers" ||
            RV.getName() == "DwarfAlias" ||
            RV.getName() == "Aliases") {
          if (const RecordVal *DefRV = RegisterCl->getValue(RV.getName()))
            NewReg->addValue(*DefRV);
          continue;
        }

        // Everything else is copied from Proto.
        NewReg->addValue(RV);
      }
    }
  }
};
}

//===----------------------------------------------------------------------===//
//                            CodeGenRegisterClass
//===----------------------------------------------------------------------===//

CodeGenRegisterClass::CodeGenRegisterClass(CodeGenRegBank &RegBank, Record *R)
  : TheDef(R), Name(R->getName()), EnumValue(-1) {
  // Rename anonymous register classes.
  if (R->getName().size() > 9 && R->getName()[9] == '.') {
    static unsigned AnonCounter = 0;
    R->setName("AnonRegClass_"+utostr(AnonCounter++));
  }

  std::vector<Record*> TypeList = R->getValueAsListOfDefs("RegTypes");
  for (unsigned i = 0, e = TypeList.size(); i != e; ++i) {
    Record *Type = TypeList[i];
    if (!Type->isSubClassOf("ValueType"))
      throw "RegTypes list member '" + Type->getName() +
        "' does not derive from the ValueType class!";
    VTs.push_back(getValueType(Type));
  }
  assert(!VTs.empty() && "RegisterClass must contain at least one ValueType!");

  // Default allocation order always contains all registers.
  Elements = RegBank.getSets().expand(R);
  for (unsigned i = 0, e = Elements->size(); i != e; ++i)
    Members.insert(RegBank.getReg((*Elements)[i]));

  // Alternative allocation orders may be subsets.
  ListInit *Alts = R->getValueAsListInit("AltOrders");
  AltOrders.resize(Alts->size());
  SetTheory::RecSet Order;
  for (unsigned i = 0, e = Alts->size(); i != e; ++i) {
    RegBank.getSets().evaluate(Alts->getElement(i), Order);
    AltOrders[i].append(Order.begin(), Order.end());
    // Verify that all altorder members are regclass members.
    while (!Order.empty()) {
      CodeGenRegister *Reg = RegBank.getReg(Order.back());
      Order.pop_back();
      if (!contains(Reg))
        throw TGError(R->getLoc(), " AltOrder register " + Reg->getName() +
                      " is not a class member");
    }
  }

  // SubRegClasses is a list<dag> containing (RC, subregindex, ...) dags.
  ListInit *SRC = R->getValueAsListInit("SubRegClasses");
  for (ListInit::const_iterator i = SRC->begin(), e = SRC->end(); i != e; ++i) {
    DagInit *DAG = dynamic_cast<DagInit*>(*i);
    if (!DAG) throw "SubRegClasses must contain DAGs";
    DefInit *DAGOp = dynamic_cast<DefInit*>(DAG->getOperator());
    Record *RCRec;
    if (!DAGOp || !(RCRec = DAGOp->getDef())->isSubClassOf("RegisterClass"))
      throw "Operator '" + DAG->getOperator()->getAsString() +
        "' in SubRegClasses is not a RegisterClass";
    // Iterate over args, all SubRegIndex instances.
    for (DagInit::const_arg_iterator ai = DAG->arg_begin(), ae = DAG->arg_end();
         ai != ae; ++ai) {
      DefInit *Idx = dynamic_cast<DefInit*>(*ai);
      Record *IdxRec;
      if (!Idx || !(IdxRec = Idx->getDef())->isSubClassOf("SubRegIndex"))
        throw "Argument '" + (*ai)->getAsString() +
          "' in SubRegClasses is not a SubRegIndex";
      if (!SubRegClasses.insert(std::make_pair(IdxRec, RCRec)).second)
        throw "SubRegIndex '" + IdxRec->getName() + "' mentioned twice";
    }
  }

  // Allow targets to override the size in bits of the RegisterClass.
  unsigned Size = R->getValueAsInt("Size");

  Namespace = R->getValueAsString("Namespace");
  SpillSize = Size ? Size : EVT(VTs[0]).getSizeInBits();
  SpillAlignment = R->getValueAsInt("Alignment");
  CopyCost = R->getValueAsInt("CopyCost");
  Allocatable = R->getValueAsBit("isAllocatable");
  AltOrderSelect = R->getValueAsCode("AltOrderSelect");
}

bool CodeGenRegisterClass::contains(const CodeGenRegister *Reg) const {
  return Members.count(Reg);
}

// Returns true if RC is a strict subclass.
// RC is a sub-class of this class if it is a valid replacement for any
// instruction operand where a register of this classis required. It must
// satisfy these conditions:
//
// 1. All RC registers are also in this.
// 2. The RC spill size must not be smaller than our spill size.
// 3. RC spill alignment must be compatible with ours.
//
static bool testSubClass(const CodeGenRegisterClass *A,
                         const CodeGenRegisterClass *B) {
  return A->SpillAlignment && B->SpillAlignment % A->SpillAlignment == 0 &&
    A->SpillSize <= B->SpillSize &&
    std::includes(A->getMembers().begin(), A->getMembers().end(),
                  B->getMembers().begin(), B->getMembers().end(),
                  CodeGenRegister::Less());
}

/// Sorting predicate for register classes.  This provides a topological
/// ordering that arranges all register classes before their sub-classes.
///
/// Register classes with the same registers, spill size, and alignment form a
/// clique.  They will be ordered alphabetically.
///
static int TopoOrderRC(const void *PA, const void *PB) {
  const CodeGenRegisterClass *A = *(const CodeGenRegisterClass* const*)PA;
  const CodeGenRegisterClass *B = *(const CodeGenRegisterClass* const*)PB;
  if (A == B)
    return 0;

  // Order by descending set size.
  if (A->getOrder().size() > B->getOrder().size())
    return -1;
  if (A->getOrder().size() < B->getOrder().size())
    return 1;

  // Order by ascending spill size.
  if (A->SpillSize < B->SpillSize)
    return -1;
  if (A->SpillSize > B->SpillSize)
    return 1;

  // Order by ascending spill alignment.
  if (A->SpillAlignment < B->SpillAlignment)
    return -1;
  if (A->SpillAlignment > B->SpillAlignment)
    return 1;

  // Finally order by name as a tie breaker.
  return A->getName() < B->getName();
}

std::string CodeGenRegisterClass::getQualifiedName() const {
  if (Namespace.empty())
    return getName();
  else
    return Namespace + "::" + getName();
}

// Compute sub-classes of all register classes.
// Assume the classes are ordered topologically.
void CodeGenRegisterClass::
computeSubClasses(ArrayRef<CodeGenRegisterClass*> RegClasses) {
  // Visit backwards so sub-classes are seen first.
  for (unsigned rci = RegClasses.size(); rci; --rci) {
    CodeGenRegisterClass &RC = *RegClasses[rci - 1];
    RC.SubClasses.resize(RegClasses.size());
    RC.SubClasses.set(RC.EnumValue);

    // Normally, all subclasses have IDs >= rci, unless RC is part of a clique.
    for (unsigned s = rci; s != RegClasses.size(); ++s) {
      if (RC.SubClasses.test(s))
        continue;
      CodeGenRegisterClass *SubRC = RegClasses[s];
      if (!testSubClass(&RC, SubRC))
        continue;
      // SubRC is a sub-class. Grap all its sub-classes so we won't have to
      // check them again.
      RC.SubClasses |= SubRC->SubClasses;
    }

    // Sweep up missed clique members.  They will be immediately preceeding RC.
    for (unsigned s = rci - 1; s && testSubClass(&RC, RegClasses[s - 1]); --s)
      RC.SubClasses.set(s - 1);
  }

  // Compute the SuperClasses lists from the SubClasses vectors.
  for (unsigned rci = 0; rci != RegClasses.size(); ++rci) {
    const BitVector &SC = RegClasses[rci]->getSubClasses();
    for (int s = SC.find_first(); s >= 0; s = SC.find_next(s)) {
      if (unsigned(s) == rci)
        continue;
      RegClasses[s]->SuperClasses.push_back(RegClasses[rci]);
    }
  }
}

//===----------------------------------------------------------------------===//
//                               CodeGenRegBank
//===----------------------------------------------------------------------===//

CodeGenRegBank::CodeGenRegBank(RecordKeeper &Records) : Records(Records) {
  // Configure register Sets to understand register classes and tuples.
  Sets.addFieldExpander("RegisterClass", "MemberList");
  Sets.addExpander("RegisterTuples", new TupleExpander());

  // Read in the user-defined (named) sub-register indices.
  // More indices will be synthesized later.
  SubRegIndices = Records.getAllDerivedDefinitions("SubRegIndex");
  std::sort(SubRegIndices.begin(), SubRegIndices.end(), LessRecord());
  NumNamedIndices = SubRegIndices.size();

  // Read in the register definitions.
  std::vector<Record*> Regs = Records.getAllDerivedDefinitions("Register");
  std::sort(Regs.begin(), Regs.end(), LessRecord());
  Registers.reserve(Regs.size());
  // Assign the enumeration values.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i)
    getReg(Regs[i]);

  // Expand tuples and number the new registers.
  std::vector<Record*> Tups =
    Records.getAllDerivedDefinitions("RegisterTuples");
  for (unsigned i = 0, e = Tups.size(); i != e; ++i) {
    const std::vector<Record*> *TupRegs = Sets.expand(Tups[i]);
    for (unsigned j = 0, je = TupRegs->size(); j != je; ++j)
      getReg((*TupRegs)[j]);
  }

  // Read in register class definitions.
  std::vector<Record*> RCs = Records.getAllDerivedDefinitions("RegisterClass");
  if (RCs.empty())
    throw std::string("No 'RegisterClass' subclasses defined!");

  RegClasses.reserve(RCs.size());
  for (unsigned i = 0, e = RCs.size(); i != e; ++i) {
    CodeGenRegisterClass *RC = new CodeGenRegisterClass(*this, RCs[i]);
    RegClasses.push_back(RC);
    Def2RC[RCs[i]] = RC;
  }
  // Order register classes topologically and assign enum values.
  array_pod_sort(RegClasses.begin(), RegClasses.end(), TopoOrderRC);
  for (unsigned i = 0, e = RegClasses.size(); i != e; ++i)
    RegClasses[i]->EnumValue = i;
  CodeGenRegisterClass::computeSubClasses(RegClasses);
}

CodeGenRegister *CodeGenRegBank::getReg(Record *Def) {
  CodeGenRegister *&Reg = Def2Reg[Def];
  if (Reg)
    return Reg;
  Reg = new CodeGenRegister(Def, Registers.size() + 1);
  Registers.push_back(Reg);
  return Reg;
}

CodeGenRegisterClass *CodeGenRegBank::getRegClass(Record *Def) {
  if (CodeGenRegisterClass *RC = Def2RC[Def])
    return RC;

  throw TGError(Def->getLoc(), "Not a known RegisterClass!");
}

Record *CodeGenRegBank::getCompositeSubRegIndex(Record *A, Record *B,
                                                bool create) {
  // Look for an existing entry.
  Record *&Comp = Composite[std::make_pair(A, B)];
  if (Comp || !create)
    return Comp;

  // None exists, synthesize one.
  std::string Name = A->getName() + "_then_" + B->getName();
  Comp = new Record(Name, SMLoc(), Records);
  SubRegIndices.push_back(Comp);
  return Comp;
}

unsigned CodeGenRegBank::getSubRegIndexNo(Record *idx) {
  std::vector<Record*>::const_iterator i =
    std::find(SubRegIndices.begin(), SubRegIndices.end(), idx);
  assert(i != SubRegIndices.end() && "Not a SubRegIndex");
  return (i - SubRegIndices.begin()) + 1;
}

void CodeGenRegBank::computeComposites() {
  // Precompute all sub-register maps. This will create Composite entries for
  // all inferred sub-register indices.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    Registers[i]->getSubRegs(*this);

  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    CodeGenRegister *Reg1 = Registers[i];
    const CodeGenRegister::SubRegMap &SRM1 = Reg1->getSubRegs();
    for (CodeGenRegister::SubRegMap::const_iterator i1 = SRM1.begin(),
         e1 = SRM1.end(); i1 != e1; ++i1) {
      Record *Idx1 = i1->first;
      CodeGenRegister *Reg2 = i1->second;
      // Ignore identity compositions.
      if (Reg1 == Reg2)
        continue;
      const CodeGenRegister::SubRegMap &SRM2 = Reg2->getSubRegs();
      // Try composing Idx1 with another SubRegIndex.
      for (CodeGenRegister::SubRegMap::const_iterator i2 = SRM2.begin(),
           e2 = SRM2.end(); i2 != e2; ++i2) {
        std::pair<Record*, Record*> IdxPair(Idx1, i2->first);
        CodeGenRegister *Reg3 = i2->second;
        // Ignore identity compositions.
        if (Reg2 == Reg3)
          continue;
        // OK Reg1:IdxPair == Reg3. Find the index with Reg:Idx == Reg3.
        for (CodeGenRegister::SubRegMap::const_iterator i1d = SRM1.begin(),
             e1d = SRM1.end(); i1d != e1d; ++i1d) {
          if (i1d->second == Reg3) {
            std::pair<CompositeMap::iterator, bool> Ins =
              Composite.insert(std::make_pair(IdxPair, i1d->first));
            // Conflicting composition? Emit a warning but allow it.
            if (!Ins.second && Ins.first->second != i1d->first) {
              errs() << "Warning: SubRegIndex " << getQualifiedName(Idx1)
                     << " and " << getQualifiedName(IdxPair.second)
                     << " compose ambiguously as "
                     << getQualifiedName(Ins.first->second) << " or "
                     << getQualifiedName(i1d->first) << "\n";
            }
          }
        }
      }
    }
  }

  // We don't care about the difference between (Idx1, Idx2) -> Idx2 and invalid
  // compositions, so remove any mappings of that form.
  for (CompositeMap::iterator i = Composite.begin(), e = Composite.end();
       i != e;) {
    CompositeMap::iterator j = i;
    ++i;
    if (j->first.second == j->second)
      Composite.erase(j);
  }
}

// Compute sets of overlapping registers.
//
// The standard set is all super-registers and all sub-registers, but the
// target description can add arbitrary overlapping registers via the 'Aliases'
// field. This complicates things, but we can compute overlapping sets using
// the following rules:
//
// 1. The relation overlap(A, B) is reflexive and symmetric but not transitive.
//
// 2. overlap(A, B) implies overlap(A, S) for all S in supers(B).
//
// Alternatively:
//
//    overlap(A, B) iff there exists:
//    A' in { A, subregs(A) } and B' in { B, subregs(B) } such that:
//    A' = B' or A' in aliases(B') or B' in aliases(A').
//
// Here subregs(A) is the full flattened sub-register set returned by
// A.getSubRegs() while aliases(A) is simply the special 'Aliases' field in the
// description of register A.
//
// This also implies that registers with a common sub-register are considered
// overlapping. This can happen when forming register pairs:
//
//    P0 = (R0, R1)
//    P1 = (R1, R2)
//    P2 = (R2, R3)
//
// In this case, we will infer an overlap between P0 and P1 because of the
// shared sub-register R1. There is no overlap between P0 and P2.
//
void CodeGenRegBank::
computeOverlaps(std::map<const CodeGenRegister*, CodeGenRegister::Set> &Map) {
  assert(Map.empty());

  // Collect overlaps that don't follow from rule 2.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    CodeGenRegister *Reg = Registers[i];
    CodeGenRegister::Set &Overlaps = Map[Reg];

    // Reg overlaps itself.
    Overlaps.insert(Reg);

    // All super-registers overlap.
    const CodeGenRegister::SuperRegList &Supers = Reg->getSuperRegs();
    Overlaps.insert(Supers.begin(), Supers.end());

    // Form symmetrical relations from the special Aliases[] lists.
    std::vector<Record*> RegList = Reg->TheDef->getValueAsListOfDefs("Aliases");
    for (unsigned i2 = 0, e2 = RegList.size(); i2 != e2; ++i2) {
      CodeGenRegister *Reg2 = getReg(RegList[i2]);
      CodeGenRegister::Set &Overlaps2 = Map[Reg2];
      const CodeGenRegister::SuperRegList &Supers2 = Reg2->getSuperRegs();
      // Reg overlaps Reg2 which implies it overlaps supers(Reg2).
      Overlaps.insert(Reg2);
      Overlaps.insert(Supers2.begin(), Supers2.end());
      Overlaps2.insert(Reg);
      Overlaps2.insert(Supers.begin(), Supers.end());
    }
  }

  // Apply rule 2. and inherit all sub-register overlaps.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    CodeGenRegister *Reg = Registers[i];
    CodeGenRegister::Set &Overlaps = Map[Reg];
    const CodeGenRegister::SubRegMap &SRM = Reg->getSubRegs();
    for (CodeGenRegister::SubRegMap::const_iterator i2 = SRM.begin(),
         e2 = SRM.end(); i2 != e2; ++i2) {
      CodeGenRegister::Set &Overlaps2 = Map[i2->second];
      Overlaps.insert(Overlaps2.begin(), Overlaps2.end());
    }
  }
}

void CodeGenRegBank::computeDerivedInfo() {
  computeComposites();
}

/// getRegisterClassForRegister - Find the register class that contains the
/// specified physical register.  If the register is not in a register class,
/// return null. If the register is in multiple classes, and the classes have a
/// superset-subset relationship and the same set of types, return the
/// superclass.  Otherwise return null.
const CodeGenRegisterClass*
CodeGenRegBank::getRegClassForRegister(Record *R) {
  const CodeGenRegister *Reg = getReg(R);
  ArrayRef<CodeGenRegisterClass*> RCs = getRegClasses();
  const CodeGenRegisterClass *FoundRC = 0;
  for (unsigned i = 0, e = RCs.size(); i != e; ++i) {
    const CodeGenRegisterClass &RC = *RCs[i];
    if (!RC.contains(Reg))
      continue;

    // If this is the first class that contains the register,
    // make a note of it and go on to the next class.
    if (!FoundRC) {
      FoundRC = &RC;
      continue;
    }

    // If a register's classes have different types, return null.
    if (RC.getValueTypes() != FoundRC->getValueTypes())
      return 0;

    // Check to see if the previously found class that contains
    // the register is a subclass of the current class. If so,
    // prefer the superclass.
    if (RC.hasSubClass(FoundRC)) {
      FoundRC = &RC;
      continue;
    }

    // Check to see if the previously found class that contains
    // the register is a superclass of the current class. If so,
    // prefer the superclass.
    if (FoundRC->hasSubClass(&RC))
      continue;

    // Multiple classes, and neither is a superclass of the other.
    // Return null.
    return 0;
  }
  return FoundRC;
}
