//===- CodeGenSchedule.cpp - Scheduling MachineModels ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate the machine model as decribed in
// the target description.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "subtarget-emitter"

#include "CodeGenSchedule.h"
#include "CodeGenTarget.h"
#include "llvm/TableGen/Error.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#ifndef NDEBUG
static void dumpIdxVec(const IdxVec &V) {
  for (unsigned i = 0, e = V.size(); i < e; ++i) {
    dbgs() << V[i] << ", ";
  }
}
static void dumpIdxVec(const SmallVectorImpl<unsigned> &V) {
  for (unsigned i = 0, e = V.size(); i < e; ++i) {
    dbgs() << V[i] << ", ";
  }
}
#endif

/// CodeGenModels ctor interprets machine model records and populates maps.
CodeGenSchedModels::CodeGenSchedModels(RecordKeeper &RK,
                                       const CodeGenTarget &TGT):
  Records(RK), Target(TGT), NumItineraryClasses(0) {

  // Instantiate a CodeGenProcModel for each SchedMachineModel with the values
  // that are explicitly referenced in tablegen records. Resources associated
  // with each processor will be derived later. Populate ProcModelMap with the
  // CodeGenProcModel instances.
  collectProcModels();

  // Instantiate a CodeGenSchedRW for each SchedReadWrite record explicitly
  // defined, and populate SchedReads and SchedWrites vectors. Implicit
  // SchedReadWrites that represent sequences derived from expanded variant will
  // be inferred later.
  collectSchedRW();

  // Instantiate a CodeGenSchedClass for each unique SchedRW signature directly
  // required by an instruction definition, and populate SchedClassIdxMap. Set
  // NumItineraryClasses to the number of explicit itinerary classes referenced
  // by instructions. Set NumInstrSchedClasses to the number of itinerary
  // classes plus any classes implied by instructions that derive from class
  // Sched and provide SchedRW list. This does not infer any new classes from
  // SchedVariant.
  collectSchedClasses();

  // Find instruction itineraries for each processor. Sort and populate
  // CodeGenProcMode::ItinDefList. (Cycle-to-cycle itineraries). This requires
  // all itinerary classes to be discovered.
  collectProcItins();

  // Find ItinRW records for each processor and itinerary class.
  // (For per-operand resources mapped to itinerary classes).
  collectProcItinRW();

  // Infer new SchedClasses from SchedVariant.
  inferSchedClasses();

  DEBUG(for (unsigned i = 0; i < SchedClasses.size(); ++i)
          SchedClasses[i].dump(this));

  // Populate each CodeGenProcModel's WriteResDefs, ReadAdvanceDefs, and
  // ProcResourceDefs.
  collectProcResources();
}

/// Gather all processor models.
void CodeGenSchedModels::collectProcModels() {
  RecVec ProcRecords = Records.getAllDerivedDefinitions("Processor");
  std::sort(ProcRecords.begin(), ProcRecords.end(), LessRecordFieldName());

  // Reserve space because we can. Reallocation would be ok.
  ProcModels.reserve(ProcRecords.size()+1);

  // Use idx=0 for NoModel/NoItineraries.
  Record *NoModelDef = Records.getDef("NoSchedModel");
  Record *NoItinsDef = Records.getDef("NoItineraries");
  ProcModels.push_back(CodeGenProcModel(0, "NoSchedModel",
                                        NoModelDef, NoItinsDef));
  ProcModelMap[NoModelDef] = 0;

  // For each processor, find a unique machine model.
  for (unsigned i = 0, N = ProcRecords.size(); i < N; ++i)
    addProcModel(ProcRecords[i]);
}

/// Get a unique processor model based on the defined MachineModel and
/// ProcessorItineraries.
void CodeGenSchedModels::addProcModel(Record *ProcDef) {
  Record *ModelKey = getModelOrItinDef(ProcDef);
  if (!ProcModelMap.insert(std::make_pair(ModelKey, ProcModels.size())).second)
    return;

  std::string Name = ModelKey->getName();
  if (ModelKey->isSubClassOf("SchedMachineModel")) {
    Record *ItinsDef = ModelKey->getValueAsDef("Itineraries");
    ProcModels.push_back(
      CodeGenProcModel(ProcModels.size(), Name, ModelKey, ItinsDef));
  }
  else {
    // An itinerary is defined without a machine model. Infer a new model.
    if (!ModelKey->getValueAsListOfDefs("IID").empty())
      Name = Name + "Model";
    ProcModels.push_back(
      CodeGenProcModel(ProcModels.size(), Name,
                       ProcDef->getValueAsDef("SchedModel"), ModelKey));
  }
  DEBUG(ProcModels.back().dump());
}

// Recursively find all reachable SchedReadWrite records.
static void scanSchedRW(Record *RWDef, RecVec &RWDefs,
                        SmallPtrSet<Record*, 16> &RWSet) {
  if (!RWSet.insert(RWDef))
    return;
  RWDefs.push_back(RWDef);
  // Reads don't current have sequence records, but it can be added later.
  if (RWDef->isSubClassOf("WriteSequence")) {
    RecVec Seq = RWDef->getValueAsListOfDefs("Writes");
    for (RecIter I = Seq.begin(), E = Seq.end(); I != E; ++I)
      scanSchedRW(*I, RWDefs, RWSet);
  }
  else if (RWDef->isSubClassOf("SchedVariant")) {
    // Visit each variant (guarded by a different predicate).
    RecVec Vars = RWDef->getValueAsListOfDefs("Variants");
    for (RecIter VI = Vars.begin(), VE = Vars.end(); VI != VE; ++VI) {
      // Visit each RW in the sequence selected by the current variant.
      RecVec Selected = (*VI)->getValueAsListOfDefs("Selected");
      for (RecIter I = Selected.begin(), E = Selected.end(); I != E; ++I)
        scanSchedRW(*I, RWDefs, RWSet);
    }
  }
}

// Collect and sort all SchedReadWrites reachable via tablegen records.
// More may be inferred later when inferring new SchedClasses from variants.
void CodeGenSchedModels::collectSchedRW() {
  // Reserve idx=0 for invalid writes/reads.
  SchedWrites.resize(1);
  SchedReads.resize(1);

  SmallPtrSet<Record*, 16> RWSet;

  // Find all SchedReadWrites referenced by instruction defs.
  RecVec SWDefs, SRDefs;
  for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
         E = Target.inst_end(); I != E; ++I) {
    Record *SchedDef = (*I)->TheDef;
    if (!SchedDef->isSubClassOf("Sched"))
      continue;
    RecVec RWs = SchedDef->getValueAsListOfDefs("SchedRW");
    for (RecIter RWI = RWs.begin(), RWE = RWs.end(); RWI != RWE; ++RWI) {
      if ((*RWI)->isSubClassOf("SchedWrite"))
        scanSchedRW(*RWI, SWDefs, RWSet);
      else {
        assert((*RWI)->isSubClassOf("SchedRead") && "Unknown SchedReadWrite");
        scanSchedRW(*RWI, SRDefs, RWSet);
      }
    }
  }
  // Find all ReadWrites referenced by InstRW.
  RecVec InstRWDefs = Records.getAllDerivedDefinitions("InstRW");
  for (RecIter OI = InstRWDefs.begin(), OE = InstRWDefs.end(); OI != OE; ++OI) {
    // For all OperandReadWrites.
    RecVec RWDefs = (*OI)->getValueAsListOfDefs("OperandReadWrites");
    for (RecIter RWI = RWDefs.begin(), RWE = RWDefs.end();
         RWI != RWE; ++RWI) {
      if ((*RWI)->isSubClassOf("SchedWrite"))
        scanSchedRW(*RWI, SWDefs, RWSet);
      else {
        assert((*RWI)->isSubClassOf("SchedRead") && "Unknown SchedReadWrite");
        scanSchedRW(*RWI, SRDefs, RWSet);
      }
    }
  }
  // Find all ReadWrites referenced by ItinRW.
  RecVec ItinRWDefs = Records.getAllDerivedDefinitions("ItinRW");
  for (RecIter II = ItinRWDefs.begin(), IE = ItinRWDefs.end(); II != IE; ++II) {
    // For all OperandReadWrites.
    RecVec RWDefs = (*II)->getValueAsListOfDefs("OperandReadWrites");
    for (RecIter RWI = RWDefs.begin(), RWE = RWDefs.end();
         RWI != RWE; ++RWI) {
      if ((*RWI)->isSubClassOf("SchedWrite"))
        scanSchedRW(*RWI, SWDefs, RWSet);
      else {
        assert((*RWI)->isSubClassOf("SchedRead") && "Unknown SchedReadWrite");
        scanSchedRW(*RWI, SRDefs, RWSet);
      }
    }
  }
  // Sort and add the SchedReadWrites directly referenced by instructions or
  // itinerary resources. Index reads and writes in separate domains.
  std::sort(SWDefs.begin(), SWDefs.end(), LessRecord());
  for (RecIter SWI = SWDefs.begin(), SWE = SWDefs.end(); SWI != SWE; ++SWI) {
    assert(!getSchedRWIdx(*SWI, /*IsRead=*/false) && "duplicate SchedWrite");
    SchedWrites.push_back(CodeGenSchedRW(*SWI));
  }
  std::sort(SRDefs.begin(), SRDefs.end(), LessRecord());
  for (RecIter SRI = SRDefs.begin(), SRE = SRDefs.end(); SRI != SRE; ++SRI) {
    assert(!getSchedRWIdx(*SRI, /*IsRead-*/true) && "duplicate SchedWrite");
    SchedReads.push_back(CodeGenSchedRW(*SRI));
  }
  // Initialize WriteSequence vectors.
  for (std::vector<CodeGenSchedRW>::iterator WI = SchedWrites.begin(),
         WE = SchedWrites.end(); WI != WE; ++WI) {
    if (!WI->IsSequence)
      continue;
    findRWs(WI->TheDef->getValueAsListOfDefs("Writes"), WI->Sequence,
            /*IsRead=*/false);
  }
  DEBUG(
    for (unsigned WIdx = 0, WEnd = SchedWrites.size(); WIdx != WEnd; ++WIdx) {
      dbgs() << WIdx << ": ";
      SchedWrites[WIdx].dump();
      dbgs() << '\n';
    }
    for (unsigned RIdx = 0, REnd = SchedReads.size(); RIdx != REnd; ++RIdx) {
      dbgs() << RIdx << ": ";
      SchedReads[RIdx].dump();
      dbgs() << '\n';
    }
    RecVec RWDefs = Records.getAllDerivedDefinitions("SchedReadWrite");
    for (RecIter RI = RWDefs.begin(), RE = RWDefs.end();
         RI != RE; ++RI) {
      if (!getSchedRWIdx(*RI, (*RI)->isSubClassOf("SchedRead"))) {
        const std::string &Name = (*RI)->getName();
        if (Name != "NoWrite" && Name != "ReadDefault")
          dbgs() << "Unused SchedReadWrite " << (*RI)->getName() << '\n';
      }
    });
}

/// Compute a SchedWrite name from a sequence of writes.
std::string CodeGenSchedModels::genRWName(const IdxVec& Seq, bool IsRead) {
  std::string Name("(");
  for (IdxIter I = Seq.begin(), E = Seq.end(); I != E; ++I) {
    if (I != Seq.begin())
      Name += '_';
    Name += getSchedRW(*I, IsRead).Name;
  }
  Name += ')';
  return Name;
}

unsigned CodeGenSchedModels::getSchedRWIdx(Record *Def, bool IsRead,
                                           unsigned After) const {
  const std::vector<CodeGenSchedRW> &RWVec = IsRead ? SchedReads : SchedWrites;
  assert(After < RWVec.size() && "start position out of bounds");
  for (std::vector<CodeGenSchedRW>::const_iterator I = RWVec.begin() + After,
         E = RWVec.end(); I != E; ++I) {
    if (I->TheDef == Def)
      return I - RWVec.begin();
  }
  return 0;
}

namespace llvm {
void splitSchedReadWrites(const RecVec &RWDefs,
                          RecVec &WriteDefs, RecVec &ReadDefs) {
  for (RecIter RWI = RWDefs.begin(), RWE = RWDefs.end(); RWI != RWE; ++RWI) {
    if ((*RWI)->isSubClassOf("SchedWrite"))
      WriteDefs.push_back(*RWI);
    else {
      assert((*RWI)->isSubClassOf("SchedRead") && "unknown SchedReadWrite");
      ReadDefs.push_back(*RWI);
    }
  }
}
} // namespace llvm

// Split the SchedReadWrites defs and call findRWs for each list.
void CodeGenSchedModels::findRWs(const RecVec &RWDefs,
                                 IdxVec &Writes, IdxVec &Reads) const {
    RecVec WriteDefs;
    RecVec ReadDefs;
    splitSchedReadWrites(RWDefs, WriteDefs, ReadDefs);
    findRWs(WriteDefs, Writes, false);
    findRWs(ReadDefs, Reads, true);
}

// Call getSchedRWIdx for all elements in a sequence of SchedRW defs.
void CodeGenSchedModels::findRWs(const RecVec &RWDefs, IdxVec &RWs,
                                 bool IsRead) const {
  for (RecIter RI = RWDefs.begin(), RE = RWDefs.end(); RI != RE; ++RI) {
    unsigned Idx = getSchedRWIdx(*RI, IsRead);
    assert(Idx && "failed to collect SchedReadWrite");
    RWs.push_back(Idx);
  }
}

void CodeGenSchedModels::expandRWSequence(unsigned RWIdx, IdxVec &RWSeq,
                                          bool IsRead) const {
  const CodeGenSchedRW &SchedRW = getSchedRW(RWIdx, IsRead);
  if (!SchedRW.IsSequence) {
    RWSeq.push_back(RWIdx);
    return;
  }
  int Repeat =
    SchedRW.TheDef ? SchedRW.TheDef->getValueAsInt("Repeat") : 1;
  for (int i = 0; i < Repeat; ++i) {
    for (IdxIter I = SchedRW.Sequence.begin(), E = SchedRW.Sequence.end();
         I != E; ++I) {
      expandRWSequence(*I, RWSeq, IsRead);
    }
  }
}

// Find the existing SchedWrite that models this sequence of writes.
unsigned CodeGenSchedModels::findRWForSequence(const IdxVec &Seq,
                                               bool IsRead) {
  std::vector<CodeGenSchedRW> &RWVec = IsRead ? SchedReads : SchedWrites;

  for (std::vector<CodeGenSchedRW>::iterator I = RWVec.begin(), E = RWVec.end();
       I != E; ++I) {
    if (I->Sequence == Seq)
      return I - RWVec.begin();
  }
  // Index zero reserved for invalid RW.
  return 0;
}

/// Add this ReadWrite if it doesn't already exist.
unsigned CodeGenSchedModels::findOrInsertRW(ArrayRef<unsigned> Seq,
                                            bool IsRead) {
  assert(!Seq.empty() && "cannot insert empty sequence");
  if (Seq.size() == 1)
    return Seq.back();

  unsigned Idx = findRWForSequence(Seq, IsRead);
  if (Idx)
    return Idx;

  CodeGenSchedRW SchedRW(Seq, genRWName(Seq, IsRead));
  if (IsRead) {
    SchedReads.push_back(SchedRW);
    return SchedReads.size() - 1;
  }
  SchedWrites.push_back(SchedRW);
  return SchedWrites.size() - 1;
}

/// Visit all the instruction definitions for this target to gather and
/// enumerate the itinerary classes. These are the explicitly specified
/// SchedClasses. More SchedClasses may be inferred.
void CodeGenSchedModels::collectSchedClasses() {

  // NoItinerary is always the first class at Idx=0
  SchedClasses.resize(1);
  SchedClasses.back().Name = "NoItinerary";
  SchedClasses.back().ProcIndices.push_back(0);
  SchedClassIdxMap[SchedClasses.back().Name] = 0;

  // Gather and sort all itinerary classes used by instruction descriptions.
  RecVec ItinClassList;
  for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
         E = Target.inst_end(); I != E; ++I) {
    Record *ItinDef = (*I)->TheDef->getValueAsDef("Itinerary");
    // Map a new SchedClass with no index.
    if (!SchedClassIdxMap.count(ItinDef->getName())) {
      SchedClassIdxMap[ItinDef->getName()] = 0;
      ItinClassList.push_back(ItinDef);
    }
  }
  // Assign each itinerary class unique number, skipping NoItinerary==0
  NumItineraryClasses = ItinClassList.size();
  std::sort(ItinClassList.begin(), ItinClassList.end(), LessRecord());
  for (unsigned i = 0, N = NumItineraryClasses; i < N; i++) {
    Record *ItinDef = ItinClassList[i];
    SchedClassIdxMap[ItinDef->getName()] = SchedClasses.size();
    SchedClasses.push_back(CodeGenSchedClass(ItinDef));
  }
  // Infer classes from SchedReadWrite resources listed for each
  // instruction definition that inherits from class Sched.
  for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
         E = Target.inst_end(); I != E; ++I) {
    if (!(*I)->TheDef->isSubClassOf("Sched"))
      continue;
    IdxVec Writes, Reads;
    findRWs((*I)->TheDef->getValueAsListOfDefs("SchedRW"), Writes, Reads);
    // ProcIdx == 0 indicates the class applies to all processors.
    IdxVec ProcIndices(1, 0);
    addSchedClass(Writes, Reads, ProcIndices);
  }
  // Create classes for InstReadWrite defs.
  RecVec InstRWDefs = Records.getAllDerivedDefinitions("InstRW");
  std::sort(InstRWDefs.begin(), InstRWDefs.end(), LessRecord());
  for (RecIter OI = InstRWDefs.begin(), OE = InstRWDefs.end(); OI != OE; ++OI)
    createInstRWClass(*OI);

  NumInstrSchedClasses = SchedClasses.size();

  bool EnableDump = false;
  DEBUG(EnableDump = true);
  if (!EnableDump)
    return;
  for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
         E = Target.inst_end(); I != E; ++I) {
    Record *SchedDef = (*I)->TheDef;
    std::string InstName = (*I)->TheDef->getName();
    if (SchedDef->isSubClassOf("Sched")) {
      IdxVec Writes;
      IdxVec Reads;
      findRWs((*I)->TheDef->getValueAsListOfDefs("SchedRW"), Writes, Reads);
      dbgs() << "SchedRW machine model for " << InstName;
      for (IdxIter WI = Writes.begin(), WE = Writes.end(); WI != WE; ++WI)
        dbgs() << " " << SchedWrites[*WI].Name;
      for (IdxIter RI = Reads.begin(), RE = Reads.end(); RI != RE; ++RI)
        dbgs() << " " << SchedReads[*RI].Name;
      dbgs() << '\n';
    }
    unsigned SCIdx = InstrClassMap.lookup((*I)->TheDef);
    if (SCIdx) {
      const RecVec &RWDefs = SchedClasses[SCIdx].InstRWs;
      for (RecIter RWI = RWDefs.begin(), RWE = RWDefs.end();
           RWI != RWE; ++RWI) {
        const CodeGenProcModel &ProcModel =
          getProcModel((*RWI)->getValueAsDef("SchedModel"));
        dbgs() << "InstrRW on " << ProcModel.ModelName << " for " << InstName;
        IdxVec Writes;
        IdxVec Reads;
        findRWs((*RWI)->getValueAsListOfDefs("OperandReadWrites"),
                Writes, Reads);
        for (IdxIter WI = Writes.begin(), WE = Writes.end(); WI != WE; ++WI)
          dbgs() << " " << SchedWrites[*WI].Name;
        for (IdxIter RI = Reads.begin(), RE = Reads.end(); RI != RE; ++RI)
          dbgs() << " " << SchedReads[*RI].Name;
        dbgs() << '\n';
      }
      continue;
    }
    if (!SchedDef->isSubClassOf("Sched")
        && (SchedDef->getValueAsDef("Itinerary")->getName() == "NoItinerary")) {
      dbgs() << "No machine model for " << (*I)->TheDef->getName() << '\n';
    }
  }
}

unsigned CodeGenSchedModels::getSchedClassIdx(
  const RecVec &RWDefs) const {

  IdxVec Writes, Reads;
  findRWs(RWDefs, Writes, Reads);
  return findSchedClassIdx(Writes, Reads);
}

/// Find an SchedClass that has been inferred from a per-operand list of
/// SchedWrites and SchedReads.
unsigned CodeGenSchedModels::findSchedClassIdx(const IdxVec &Writes,
                                               const IdxVec &Reads) const {
  for (SchedClassIter I = schedClassBegin(), E = schedClassEnd(); I != E; ++I) {
    // Classes with InstRWs may have the same Writes/Reads as a class originally
    // produced by a SchedRW definition. We need to be able to recover the
    // original class index for processors that don't match any InstRWs.
    if (I->ItinClassDef || !I->InstRWs.empty())
      continue;

    if (I->Writes == Writes && I->Reads == Reads) {
      return I - schedClassBegin();
    }
  }
  return 0;
}

// Get the SchedClass index for an instruction.
unsigned CodeGenSchedModels::getSchedClassIdx(
  const CodeGenInstruction &Inst) const {

  unsigned SCIdx = InstrClassMap.lookup(Inst.TheDef);
  if (SCIdx)
    return SCIdx;

  // If this opcode isn't mapped by the subtarget fallback to the instruction
  // definition's SchedRW or ItinDef values.
  if (Inst.TheDef->isSubClassOf("Sched")) {
    RecVec RWs = Inst.TheDef->getValueAsListOfDefs("SchedRW");
    return getSchedClassIdx(RWs);
  }
  Record *ItinDef = Inst.TheDef->getValueAsDef("Itinerary");
  assert(SchedClassIdxMap.count(ItinDef->getName()) && "missing ItinClass");
  unsigned Idx = SchedClassIdxMap.lookup(ItinDef->getName());
  assert(Idx <= NumItineraryClasses && "bad ItinClass index");
  return Idx;
}

std::string CodeGenSchedModels::createSchedClassName(
  const IdxVec &OperWrites, const IdxVec &OperReads) {

  std::string Name;
  for (IdxIter WI = OperWrites.begin(), WE = OperWrites.end(); WI != WE; ++WI) {
    if (WI != OperWrites.begin())
      Name += '_';
    Name += SchedWrites[*WI].Name;
  }
  for (IdxIter RI = OperReads.begin(), RE = OperReads.end(); RI != RE; ++RI) {
    Name += '_';
    Name += SchedReads[*RI].Name;
  }
  return Name;
}

std::string CodeGenSchedModels::createSchedClassName(const RecVec &InstDefs) {

  std::string Name;
  for (RecIter I = InstDefs.begin(), E = InstDefs.end(); I != E; ++I) {
    if (I != InstDefs.begin())
      Name += '_';
    Name += (*I)->getName();
  }
  return Name;
}

/// Add an inferred sched class from a per-operand list of SchedWrites and
/// SchedReads. ProcIndices contains the set of IDs of processors that may
/// utilize this class.
unsigned CodeGenSchedModels::addSchedClass(const IdxVec &OperWrites,
                                           const IdxVec &OperReads,
                                           const IdxVec &ProcIndices)
{
  assert(!ProcIndices.empty() && "expect at least one ProcIdx");

  unsigned Idx = findSchedClassIdx(OperWrites, OperReads);
  if (Idx) {
    IdxVec PI;
    std::set_union(SchedClasses[Idx].ProcIndices.begin(),
                   SchedClasses[Idx].ProcIndices.end(),
                   ProcIndices.begin(), ProcIndices.end(),
                   std::back_inserter(PI));
    SchedClasses[Idx].ProcIndices.swap(PI);
    return Idx;
  }
  Idx = SchedClasses.size();
  SchedClasses.resize(Idx+1);
  CodeGenSchedClass &SC = SchedClasses.back();
  SC.Name = createSchedClassName(OperWrites, OperReads);
  SC.Writes = OperWrites;
  SC.Reads = OperReads;
  SC.ProcIndices = ProcIndices;

  return Idx;
}

// Create classes for each set of opcodes that are in the same InstReadWrite
// definition across all processors.
void CodeGenSchedModels::createInstRWClass(Record *InstRWDef) {
  // ClassInstrs will hold an entry for each subset of Instrs in InstRWDef that
  // intersects with an existing class via a previous InstRWDef. Instrs that do
  // not intersect with an existing class refer back to their former class as
  // determined from ItinDef or SchedRW.
  SmallVector<std::pair<unsigned, SmallVector<Record *, 8> >, 4> ClassInstrs;
  // Sort Instrs into sets.
  RecVec InstDefs = InstRWDef->getValueAsListOfDefs("Instrs");
  std::sort(InstDefs.begin(), InstDefs.end(), LessRecord());
  for (RecIter I = InstDefs.begin(), E = InstDefs.end(); I != E; ++I) {
    unsigned SCIdx = 0;
    InstClassMapTy::const_iterator Pos = InstrClassMap.find(*I);
    if (Pos != InstrClassMap.end())
      SCIdx = Pos->second;
    else {
      // This instruction has not been mapped yet. Get the original class. All
      // instructions in the same InstrRW class must be from the same original
      // class because that is the fall-back class for other processors.
      Record *ItinDef = (*I)->getValueAsDef("Itinerary");
      SCIdx = SchedClassIdxMap.lookup(ItinDef->getName());
      if (!SCIdx && (*I)->isSubClassOf("Sched"))
        SCIdx = getSchedClassIdx((*I)->getValueAsListOfDefs("SchedRW"));
    }
    unsigned CIdx = 0, CEnd = ClassInstrs.size();
    for (; CIdx != CEnd; ++CIdx) {
      if (ClassInstrs[CIdx].first == SCIdx)
        break;
    }
    if (CIdx == CEnd) {
      ClassInstrs.resize(CEnd + 1);
      ClassInstrs[CIdx].first = SCIdx;
    }
    ClassInstrs[CIdx].second.push_back(*I);
  }
  // For each set of Instrs, create a new class if necessary, and map or remap
  // the Instrs to it.
  unsigned CIdx = 0, CEnd = ClassInstrs.size();
  for (; CIdx != CEnd; ++CIdx) {
    unsigned OldSCIdx = ClassInstrs[CIdx].first;
    ArrayRef<Record*> InstDefs = ClassInstrs[CIdx].second;
    // If the all instrs in the current class are accounted for, then leave
    // them mapped to their old class.
    if (SchedClasses[OldSCIdx].InstRWs.size() == InstDefs.size()) {
      assert(SchedClasses[OldSCIdx].ProcIndices[0] == 0 &&
             "expected a generic SchedClass");
      continue;
    }
    unsigned SCIdx = SchedClasses.size();
    SchedClasses.resize(SCIdx+1);
    CodeGenSchedClass &SC = SchedClasses.back();
    SC.Name = createSchedClassName(InstDefs);
    // Preserve ItinDef and Writes/Reads for processors without an InstRW entry.
    SC.ItinClassDef = SchedClasses[OldSCIdx].ItinClassDef;
    SC.Writes = SchedClasses[OldSCIdx].Writes;
    SC.Reads = SchedClasses[OldSCIdx].Reads;
    SC.ProcIndices.push_back(0);
    // Map each Instr to this new class.
    // Note that InstDefs may be a smaller list than InstRWDef's "Instrs".
    for (ArrayRef<Record*>::const_iterator
           II = InstDefs.begin(), IE = InstDefs.end(); II != IE; ++II) {
      unsigned OldSCIdx = InstrClassMap[*II];
      if (OldSCIdx) {
        SC.InstRWs.insert(SC.InstRWs.end(),
                          SchedClasses[OldSCIdx].InstRWs.begin(),
                          SchedClasses[OldSCIdx].InstRWs.end());
      }
      InstrClassMap[*II] = SCIdx;
    }
    SC.InstRWs.push_back(InstRWDef);
  }
}

// Gather the processor itineraries.
void CodeGenSchedModels::collectProcItins() {
  for (std::vector<CodeGenProcModel>::iterator PI = ProcModels.begin(),
         PE = ProcModels.end(); PI != PE; ++PI) {
    CodeGenProcModel &ProcModel = *PI;
    RecVec ItinRecords = ProcModel.ItinsDef->getValueAsListOfDefs("IID");
    // Skip empty itinerary.
    if (ItinRecords.empty())
      continue;

    ProcModel.ItinDefList.resize(NumItineraryClasses+1);

    // Insert each itinerary data record in the correct position within
    // the processor model's ItinDefList.
    for (unsigned i = 0, N = ItinRecords.size(); i < N; i++) {
      Record *ItinData = ItinRecords[i];
      Record *ItinDef = ItinData->getValueAsDef("TheClass");
      if (!SchedClassIdxMap.count(ItinDef->getName())) {
        DEBUG(dbgs() << ProcModel.ItinsDef->getName()
              << " has unused itinerary class " << ItinDef->getName() << '\n');
        continue;
      }
      assert(SchedClassIdxMap.count(ItinDef->getName()) && "missing ItinClass");
      unsigned Idx = SchedClassIdxMap.lookup(ItinDef->getName());
      assert(Idx <= NumItineraryClasses && "bad ItinClass index");
      ProcModel.ItinDefList[Idx] = ItinData;
    }
    // Check for missing itinerary entries.
    assert(!ProcModel.ItinDefList[0] && "NoItinerary class can't have rec");
    DEBUG(
      for (unsigned i = 1, N = ProcModel.ItinDefList.size(); i < N; ++i) {
        if (!ProcModel.ItinDefList[i])
          dbgs() << ProcModel.ItinsDef->getName()
                 << " missing itinerary for class "
                 << SchedClasses[i].Name << '\n';
      });
  }
}

// Gather the read/write types for each itinerary class.
void CodeGenSchedModels::collectProcItinRW() {
  RecVec ItinRWDefs = Records.getAllDerivedDefinitions("ItinRW");
  std::sort(ItinRWDefs.begin(), ItinRWDefs.end(), LessRecord());
  for (RecIter II = ItinRWDefs.begin(), IE = ItinRWDefs.end(); II != IE; ++II) {
    if (!(*II)->getValueInit("SchedModel")->isComplete())
      throw TGError((*II)->getLoc(), "SchedModel is undefined");
    Record *ModelDef = (*II)->getValueAsDef("SchedModel");
    ProcModelMapTy::const_iterator I = ProcModelMap.find(ModelDef);
    if (I == ProcModelMap.end()) {
      throw TGError((*II)->getLoc(), "Undefined SchedMachineModel "
                    + ModelDef->getName());
    }
    ProcModels[I->second].ItinRWDefs.push_back(*II);
  }
}

/// Infer new classes from existing classes. In the process, this may create new
/// SchedWrites from sequences of existing SchedWrites.
void CodeGenSchedModels::inferSchedClasses() {
  // Visit all existing classes and newly created classes.
  for (unsigned Idx = 0; Idx != SchedClasses.size(); ++Idx) {
    if (SchedClasses[Idx].ItinClassDef)
      inferFromItinClass(SchedClasses[Idx].ItinClassDef, Idx);
    else if (!SchedClasses[Idx].InstRWs.empty())
      inferFromInstRWs(Idx);
    else {
      inferFromRW(SchedClasses[Idx].Writes, SchedClasses[Idx].Reads,
                  Idx, SchedClasses[Idx].ProcIndices);
    }
    assert(SchedClasses.size() < (NumInstrSchedClasses*6) &&
           "too many SchedVariants");
  }
}

/// Infer classes from per-processor itinerary resources.
void CodeGenSchedModels::inferFromItinClass(Record *ItinClassDef,
                                            unsigned FromClassIdx) {
  for (unsigned PIdx = 0, PEnd = ProcModels.size(); PIdx != PEnd; ++PIdx) {
    const CodeGenProcModel &PM = ProcModels[PIdx];
    // For all ItinRW entries.
    bool HasMatch = false;
    for (RecIter II = PM.ItinRWDefs.begin(), IE = PM.ItinRWDefs.end();
         II != IE; ++II) {
      RecVec Matched = (*II)->getValueAsListOfDefs("MatchedItinClasses");
      if (!std::count(Matched.begin(), Matched.end(), ItinClassDef))
        continue;
      if (HasMatch)
        throw TGError((*II)->getLoc(), "Duplicate itinerary class "
                      + ItinClassDef->getName()
                      + " in ItinResources for " + PM.ModelName);
      HasMatch = true;
      IdxVec Writes, Reads;
      findRWs((*II)->getValueAsListOfDefs("OperandReadWrites"), Writes, Reads);
      IdxVec ProcIndices(1, PIdx);
      inferFromRW(Writes, Reads, FromClassIdx, ProcIndices);
    }
  }
}

/// Infer classes from per-processor InstReadWrite definitions.
void CodeGenSchedModels::inferFromInstRWs(unsigned SCIdx) {
  const RecVec &RWDefs = SchedClasses[SCIdx].InstRWs;
  for (RecIter RWI = RWDefs.begin(), RWE = RWDefs.end(); RWI != RWE; ++RWI) {
    RecVec Instrs = (*RWI)->getValueAsListOfDefs("Instrs");
    RecIter II = Instrs.begin(), IE = Instrs.end();
    for (; II != IE; ++II) {
      if (InstrClassMap[*II] == SCIdx)
        break;
    }
    // If this class no longer has any instructions mapped to it, it has become
    // irrelevant.
    if (II == IE)
      continue;
    IdxVec Writes, Reads;
    findRWs((*RWI)->getValueAsListOfDefs("OperandReadWrites"), Writes, Reads);
    unsigned PIdx = getProcModel((*RWI)->getValueAsDef("SchedModel")).Index;
    IdxVec ProcIndices(1, PIdx);
    inferFromRW(Writes, Reads, SCIdx, ProcIndices);
  }
}

namespace {
// Associate a predicate with the SchedReadWrite that it guards.
// RWIdx is the index of the read/write variant.
struct PredCheck {
  bool IsRead;
  unsigned RWIdx;
  Record *Predicate;

  PredCheck(bool r, unsigned w, Record *p): IsRead(r), RWIdx(w), Predicate(p) {}
};

// A Predicate transition is a list of RW sequences guarded by a PredTerm.
struct PredTransition {
  // A predicate term is a conjunction of PredChecks.
  SmallVector<PredCheck, 4> PredTerm;
  SmallVector<SmallVector<unsigned,4>, 16> WriteSequences;
  SmallVector<SmallVector<unsigned,4>, 16> ReadSequences;
};

// Encapsulate a set of partially constructed transitions.
// The results are built by repeated calls to substituteVariants.
class PredTransitions {
  CodeGenSchedModels &SchedModels;

public:
  std::vector<PredTransition> TransVec;

  PredTransitions(CodeGenSchedModels &sm): SchedModels(sm) {}

  void substituteVariantOperand(const SmallVectorImpl<unsigned> &RWSeq,
                                bool IsRead, unsigned StartIdx);

  void substituteVariants(const PredTransition &Trans);

#ifndef NDEBUG
  void dump() const;
#endif

private:
  bool mutuallyExclusive(Record *PredDef, ArrayRef<PredCheck> Term);
  void pushVariant(unsigned SchedRW, Record *Variant, PredTransition &Trans,
                   bool IsRead);
};
} // anonymous

// Return true if this predicate is mutually exclusive with a PredTerm. This
// degenerates into checking if the predicate is mutually exclusive with any
// predicate in the Term's conjunction.
//
// All predicates associated with a given SchedRW are considered mutually
// exclusive. This should work even if the conditions expressed by the
// predicates are not exclusive because the predicates for a given SchedWrite
// are always checked in the order they are defined in the .td file. Later
// conditions implicitly negate any prior condition.
bool PredTransitions::mutuallyExclusive(Record *PredDef,
                                        ArrayRef<PredCheck> Term) {

  for (ArrayRef<PredCheck>::iterator I = Term.begin(), E = Term.end();
       I != E; ++I) {
    if (I->Predicate == PredDef)
      return false;

    const CodeGenSchedRW &SchedRW = SchedModels.getSchedRW(I->RWIdx, I->IsRead);
    assert(SchedRW.HasVariants && "PredCheck must refer to a SchedVariant");
    RecVec Variants = SchedRW.TheDef->getValueAsListOfDefs("Variants");
    for (RecIter VI = Variants.begin(), VE = Variants.end(); VI != VE; ++VI) {
      if ((*VI)->getValueAsDef("Predicate") == PredDef)
        return true;
    }
  }
  return false;
}

// Push the Reads/Writes selected by this variant onto the given PredTransition.
void PredTransitions::pushVariant(unsigned RWIdx, Record *Variant,
                                  PredTransition &Trans, bool IsRead) {
  Trans.PredTerm.push_back(
    PredCheck(IsRead, RWIdx, Variant->getValueAsDef("Predicate")));
  RecVec SelectedDefs = Variant->getValueAsListOfDefs("Selected");
  IdxVec SelectedRWs;
  SchedModels.findRWs(SelectedDefs, SelectedRWs, IsRead);

  const CodeGenSchedRW &SchedRW = SchedModels.getSchedRW(RWIdx, IsRead);

  SmallVectorImpl<SmallVector<unsigned,4> > &RWSequences = IsRead
    ? Trans.ReadSequences : Trans.WriteSequences;
  if (SchedRW.IsVariadic) {
    unsigned OperIdx = RWSequences.size()-1;
    // Make N-1 copies of this transition's last sequence.
    for (unsigned i = 1, e = SelectedRWs.size(); i != e; ++i) {
      RWSequences.push_back(RWSequences[OperIdx]);
    }
    // Push each of the N elements of the SelectedRWs onto a copy of the last
    // sequence (split the current operand into N operands).
    // Note that write sequences should be expanded within this loop--the entire
    // sequence belongs to a single operand.
    for (IdxIter RWI = SelectedRWs.begin(), RWE = SelectedRWs.end();
         RWI != RWE; ++RWI, ++OperIdx) {
      IdxVec ExpandedRWs;
      if (IsRead)
        ExpandedRWs.push_back(*RWI);
      else
        SchedModels.expandRWSequence(*RWI, ExpandedRWs, IsRead);
      RWSequences[OperIdx].insert(RWSequences[OperIdx].end(),
                                  ExpandedRWs.begin(), ExpandedRWs.end());
    }
    assert(OperIdx == RWSequences.size() && "missed a sequence");
  }
  else {
    // Push this transition's expanded sequence onto this transition's last
    // sequence (add to the current operand's sequence).
    SmallVectorImpl<unsigned> &Seq = RWSequences.back();
    IdxVec ExpandedRWs;
    for (IdxIter RWI = SelectedRWs.begin(), RWE = SelectedRWs.end();
         RWI != RWE; ++RWI) {
      if (IsRead)
        ExpandedRWs.push_back(*RWI);
      else
        SchedModels.expandRWSequence(*RWI, ExpandedRWs, IsRead);
    }
    Seq.insert(Seq.end(), ExpandedRWs.begin(), ExpandedRWs.end());
  }
}

// RWSeq is a sequence of all Reads or all Writes for the next read or write
// operand. StartIdx is an index into TransVec where partial results
// starts. RWSeq must be applied to all tranistions between StartIdx and the end
// of TransVec.
void PredTransitions::substituteVariantOperand(
  const SmallVectorImpl<unsigned> &RWSeq, bool IsRead, unsigned StartIdx) {

  // Visit each original RW within the current sequence.
  for (SmallVectorImpl<unsigned>::const_iterator
         RWI = RWSeq.begin(), RWE = RWSeq.end(); RWI != RWE; ++RWI) {
    const CodeGenSchedRW &SchedRW = SchedModels.getSchedRW(*RWI, IsRead);
    // Push this RW on all partial PredTransitions or distribute variants.
    // New PredTransitions may be pushed within this loop which should not be
    // revisited (TransEnd must be loop invariant).
    for (unsigned TransIdx = StartIdx, TransEnd = TransVec.size();
         TransIdx != TransEnd; ++TransIdx) {
      // In the common case, push RW onto the current operand's sequence.
      if (!SchedRW.HasVariants) {
        if (IsRead)
          TransVec[TransIdx].ReadSequences.back().push_back(*RWI);
        else
          TransVec[TransIdx].WriteSequences.back().push_back(*RWI);
        continue;
      }
      // Distribute this partial PredTransition across intersecting variants.
      RecVec Variants = SchedRW.TheDef->getValueAsListOfDefs("Variants");
      std::vector<std::pair<Record*,unsigned> > IntersectingVariants;
      for (RecIter VI = Variants.begin(), VE = Variants.end(); VI != VE; ++VI) {
        Record *PredDef = (*VI)->getValueAsDef("Predicate");
        if (mutuallyExclusive(PredDef, TransVec[TransIdx].PredTerm))
          continue;
        if (IntersectingVariants.empty())
          // The first variant builds on the existing transition.
          IntersectingVariants.push_back(std::make_pair(*VI, TransIdx));
        else {
          // Push another copy of the current transition for more variants.
          IntersectingVariants.push_back(
            std::make_pair(*VI, TransVec.size()));
          TransVec.push_back(TransVec[TransIdx]);
        }
      }
      // Now expand each variant on top of its copy of the transition.
      for (std::vector<std::pair<Record*, unsigned> >::const_iterator
             IVI = IntersectingVariants.begin(),
             IVE = IntersectingVariants.end();
           IVI != IVE; ++IVI)
        pushVariant(*RWI, IVI->first, TransVec[IVI->second], IsRead);
    }
  }
}

// For each variant of a Read/Write in Trans, substitute the sequence of
// Read/Writes guarded by the variant. This is exponential in the number of
// variant Read/Writes, but in practice detection of mutually exclusive
// predicates should result in linear growth in the total number variants.
//
// This is one step in a breadth-first search of nested variants.
void PredTransitions::substituteVariants(const PredTransition &Trans) {
  // Build up a set of partial results starting at the back of
  // PredTransitions. Remember the first new transition.
  unsigned StartIdx = TransVec.size();
  TransVec.resize(TransVec.size() + 1);
  TransVec.back().PredTerm = Trans.PredTerm;

  // Visit each original write sequence.
  for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
         WSI = Trans.WriteSequences.begin(), WSE = Trans.WriteSequences.end();
       WSI != WSE; ++WSI) {
    // Push a new (empty) write sequence onto all partial Transitions.
    for (std::vector<PredTransition>::iterator I =
           TransVec.begin() + StartIdx, E = TransVec.end(); I != E; ++I) {
      I->WriteSequences.resize(I->WriteSequences.size() + 1);
    }
    substituteVariantOperand(*WSI, /*IsRead=*/false, StartIdx);
  }
  // Visit each original read sequence.
  for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
         RSI = Trans.ReadSequences.begin(), RSE = Trans.ReadSequences.end();
       RSI != RSE; ++RSI) {
    // Push a new (empty) read sequence onto all partial Transitions.
    for (std::vector<PredTransition>::iterator I =
           TransVec.begin() + StartIdx, E = TransVec.end(); I != E; ++I) {
      I->ReadSequences.resize(I->ReadSequences.size() + 1);
    }
    substituteVariantOperand(*RSI, /*IsRead=*/true, StartIdx);
  }
}

static bool hasVariant(ArrayRef<PredTransition> Transitions,
                       CodeGenSchedModels &SchedModels) {
  for (ArrayRef<PredTransition>::iterator
         PTI = Transitions.begin(), PTE = Transitions.end();
       PTI != PTE; ++PTI) {
    for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
           WSI = PTI->WriteSequences.begin(), WSE = PTI->WriteSequences.end();
         WSI != WSE; ++WSI) {
      for (SmallVectorImpl<unsigned>::const_iterator
             WI = WSI->begin(), WE = WSI->end(); WI != WE; ++WI) {
        if (SchedModels.getSchedWrite(*WI).HasVariants)
          return true;
      }
    }
    for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
           RSI = PTI->ReadSequences.begin(), RSE = PTI->ReadSequences.end();
         RSI != RSE; ++RSI) {
      for (SmallVectorImpl<unsigned>::const_iterator
             RI = RSI->begin(), RE = RSI->end(); RI != RE; ++RI) {
        if (SchedModels.getSchedRead(*RI).HasVariants)
          return true;
      }
    }
  }
  return false;
}

// Create a new SchedClass for each variant found by inferFromRW. Pass
// ProcIndices by copy to avoid referencing anything from SchedClasses.
static void inferFromTransitions(ArrayRef<PredTransition> LastTransitions,
                                 unsigned FromClassIdx, IdxVec ProcIndices,
                                 CodeGenSchedModels &SchedModels) {
  // For each PredTransition, create a new CodeGenSchedTransition, which usually
  // requires creating a new SchedClass.
  for (ArrayRef<PredTransition>::iterator
         I = LastTransitions.begin(), E = LastTransitions.end(); I != E; ++I) {
    IdxVec OperWritesVariant;
    for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
           WSI = I->WriteSequences.begin(), WSE = I->WriteSequences.end();
         WSI != WSE; ++WSI) {
      // Create a new write representing the expanded sequence.
      OperWritesVariant.push_back(
        SchedModels.findOrInsertRW(*WSI, /*IsRead=*/false));
    }
    IdxVec OperReadsVariant;
    for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
           RSI = I->ReadSequences.begin(), RSE = I->ReadSequences.end();
         RSI != RSE; ++RSI) {
      // Create a new write representing the expanded sequence.
      OperReadsVariant.push_back(
        SchedModels.findOrInsertRW(*RSI, /*IsRead=*/true));
    }
    CodeGenSchedTransition SCTrans;
    SCTrans.ToClassIdx =
      SchedModels.addSchedClass(OperWritesVariant, OperReadsVariant,
                                ProcIndices);
    SCTrans.ProcIndices = ProcIndices;
    // The final PredTerm is unique set of predicates guarding the transition.
    RecVec Preds;
    for (SmallVectorImpl<PredCheck>::const_iterator
           PI = I->PredTerm.begin(), PE = I->PredTerm.end(); PI != PE; ++PI) {
      Preds.push_back(PI->Predicate);
    }
    RecIter PredsEnd = std::unique(Preds.begin(), Preds.end());
    Preds.resize(PredsEnd - Preds.begin());
    SCTrans.PredTerm = Preds;
    SchedModels.getSchedClass(FromClassIdx).Transitions.push_back(SCTrans);
  }
}

/// Find each variant write that OperWrites or OperaReads refers to and create a
/// new SchedClass for each variant.
void CodeGenSchedModels::inferFromRW(const IdxVec &OperWrites,
                                     const IdxVec &OperReads,
                                     unsigned FromClassIdx,
                                     const IdxVec &ProcIndices) {
  DEBUG(dbgs() << "INFERRW Writes: ");

  // Create a seed transition with an empty PredTerm and the expanded sequences
  // of SchedWrites for the current SchedClass.
  std::vector<PredTransition> LastTransitions;
  LastTransitions.resize(1);
  for (IdxIter I = OperWrites.begin(), E = OperWrites.end(); I != E; ++I) {
    IdxVec WriteSeq;
    expandRWSequence(*I, WriteSeq, /*IsRead=*/false);
    unsigned Idx = LastTransitions[0].WriteSequences.size();
    LastTransitions[0].WriteSequences.resize(Idx + 1);
    SmallVectorImpl<unsigned> &Seq = LastTransitions[0].WriteSequences[Idx];
    for (IdxIter WI = WriteSeq.begin(), WE = WriteSeq.end(); WI != WE; ++WI)
      Seq.push_back(*WI);
    DEBUG(dbgs() << "("; dumpIdxVec(Seq); dbgs() << ") ");
  }
  DEBUG(dbgs() << " Reads: ");
  for (IdxIter I = OperReads.begin(), E = OperReads.end(); I != E; ++I) {
    IdxVec ReadSeq;
    expandRWSequence(*I, ReadSeq, /*IsRead=*/true);
    unsigned Idx = LastTransitions[0].ReadSequences.size();
    LastTransitions[0].ReadSequences.resize(Idx + 1);
    SmallVectorImpl<unsigned> &Seq = LastTransitions[0].ReadSequences[Idx];
    for (IdxIter RI = ReadSeq.begin(), RE = ReadSeq.end(); RI != RE; ++RI)
      Seq.push_back(*RI);
    DEBUG(dbgs() << "("; dumpIdxVec(Seq); dbgs() << ") ");
  }
  DEBUG(dbgs() << '\n');

  // Collect all PredTransitions for individual operands.
  // Iterate until no variant writes remain.
  while (hasVariant(LastTransitions, *this)) {
    PredTransitions Transitions(*this);
    for (std::vector<PredTransition>::const_iterator
           I = LastTransitions.begin(), E = LastTransitions.end();
         I != E; ++I) {
      Transitions.substituteVariants(*I);
    }
    DEBUG(Transitions.dump());
    LastTransitions.swap(Transitions.TransVec);
  }
  // If the first transition has no variants, nothing to do.
  if (LastTransitions[0].PredTerm.empty())
    return;

  // WARNING: We are about to mutate the SchedClasses vector. Do not refer to
  // OperWrites, OperReads, or ProcIndices after calling inferFromTransitions.
  inferFromTransitions(LastTransitions, FromClassIdx, ProcIndices, *this);
}

// Collect and sort WriteRes, ReadAdvance, and ProcResources.
void CodeGenSchedModels::collectProcResources() {
  // Add any subtarget-specific SchedReadWrites that are directly associated
  // with processor resources. Refer to the parent SchedClass's ProcIndices to
  // determine which processors they apply to.
  for (SchedClassIter SCI = schedClassBegin(), SCE = schedClassEnd();
       SCI != SCE; ++SCI) {
    if (SCI->ItinClassDef)
      collectItinProcResources(SCI->ItinClassDef);
    else
      collectRWResources(SCI->Writes, SCI->Reads, SCI->ProcIndices);
  }
  // Add resources separately defined by each subtarget.
  RecVec WRDefs = Records.getAllDerivedDefinitions("WriteRes");
  for (RecIter WRI = WRDefs.begin(), WRE = WRDefs.end(); WRI != WRE; ++WRI) {
    Record *ModelDef = (*WRI)->getValueAsDef("SchedModel");
    addWriteRes(*WRI, getProcModel(ModelDef).Index);
  }
  RecVec RADefs = Records.getAllDerivedDefinitions("ReadAdvance");
  for (RecIter RAI = RADefs.begin(), RAE = RADefs.end(); RAI != RAE; ++RAI) {
    Record *ModelDef = (*RAI)->getValueAsDef("SchedModel");
    addReadAdvance(*RAI, getProcModel(ModelDef).Index);
  }
  // Finalize each ProcModel by sorting the record arrays.
  for (unsigned PIdx = 0, PEnd = ProcModels.size(); PIdx != PEnd; ++PIdx) {
    CodeGenProcModel &PM = ProcModels[PIdx];
    std::sort(PM.WriteResDefs.begin(), PM.WriteResDefs.end(),
              LessRecord());
    std::sort(PM.ReadAdvanceDefs.begin(), PM.ReadAdvanceDefs.end(),
              LessRecord());
    std::sort(PM.ProcResourceDefs.begin(), PM.ProcResourceDefs.end(),
              LessRecord());
    DEBUG(
      PM.dump();
      dbgs() << "WriteResDefs: ";
      for (RecIter RI = PM.WriteResDefs.begin(),
             RE = PM.WriteResDefs.end(); RI != RE; ++RI) {
        if ((*RI)->isSubClassOf("WriteRes"))
          dbgs() << (*RI)->getValueAsDef("WriteType")->getName() << " ";
        else
          dbgs() << (*RI)->getName() << " ";
      }
      dbgs() << "\nReadAdvanceDefs: ";
      for (RecIter RI = PM.ReadAdvanceDefs.begin(),
             RE = PM.ReadAdvanceDefs.end(); RI != RE; ++RI) {
        if ((*RI)->isSubClassOf("ReadAdvance"))
          dbgs() << (*RI)->getValueAsDef("ReadType")->getName() << " ";
        else
          dbgs() << (*RI)->getName() << " ";
      }
      dbgs() << "\nProcResourceDefs: ";
      for (RecIter RI = PM.ProcResourceDefs.begin(),
             RE = PM.ProcResourceDefs.end(); RI != RE; ++RI) {
        dbgs() << (*RI)->getName() << " ";
      }
      dbgs() << '\n');
  }
}

// Collect itinerary class resources for each processor.
void CodeGenSchedModels::collectItinProcResources(Record *ItinClassDef) {
  for (unsigned PIdx = 0, PEnd = ProcModels.size(); PIdx != PEnd; ++PIdx) {
    const CodeGenProcModel &PM = ProcModels[PIdx];
    // For all ItinRW entries.
    bool HasMatch = false;
    for (RecIter II = PM.ItinRWDefs.begin(), IE = PM.ItinRWDefs.end();
         II != IE; ++II) {
      RecVec Matched = (*II)->getValueAsListOfDefs("MatchedItinClasses");
      if (!std::count(Matched.begin(), Matched.end(), ItinClassDef))
        continue;
      if (HasMatch)
        throw TGError((*II)->getLoc(), "Duplicate itinerary class "
                      + ItinClassDef->getName()
                      + " in ItinResources for " + PM.ModelName);
      HasMatch = true;
      IdxVec Writes, Reads;
      findRWs((*II)->getValueAsListOfDefs("OperandReadWrites"), Writes, Reads);
      IdxVec ProcIndices(1, PIdx);
      collectRWResources(Writes, Reads, ProcIndices);
    }
  }
}


// Collect resources for a set of read/write types and processor indices.
void CodeGenSchedModels::collectRWResources(const IdxVec &Writes,
                                            const IdxVec &Reads,
                                            const IdxVec &ProcIndices) {

  for (IdxIter WI = Writes.begin(), WE = Writes.end(); WI != WE; ++WI) {
    const CodeGenSchedRW &SchedRW = getSchedRW(*WI, /*IsRead=*/false);
    if (SchedRW.TheDef && SchedRW.TheDef->isSubClassOf("SchedWriteRes")) {
      for (IdxIter PI = ProcIndices.begin(), PE = ProcIndices.end();
           PI != PE; ++PI) {
        addWriteRes(SchedRW.TheDef, *PI);
      }
    }
  }
  for (IdxIter RI = Reads.begin(), RE = Reads.end(); RI != RE; ++RI) {
    const CodeGenSchedRW &SchedRW = getSchedRW(*RI, /*IsRead=*/true);
    if (SchedRW.TheDef && SchedRW.TheDef->isSubClassOf("SchedReadAdvance")) {
      for (IdxIter PI = ProcIndices.begin(), PE = ProcIndices.end();
           PI != PE; ++PI) {
        addReadAdvance(SchedRW.TheDef, *PI);
      }
    }
  }
}

// Find the processor's resource units for this kind of resource.
Record *CodeGenSchedModels::findProcResUnits(Record *ProcResKind,
                                             const CodeGenProcModel &PM) const {
  if (ProcResKind->isSubClassOf("ProcResourceUnits"))
    return ProcResKind;

  Record *ProcUnitDef = 0;
  RecVec ProcResourceDefs =
    Records.getAllDerivedDefinitions("ProcResourceUnits");

  for (RecIter RI = ProcResourceDefs.begin(), RE = ProcResourceDefs.end();
       RI != RE; ++RI) {

    if ((*RI)->getValueAsDef("Kind") == ProcResKind
        && (*RI)->getValueAsDef("SchedModel") == PM.ModelDef) {
      if (ProcUnitDef) {
        throw TGError((*RI)->getLoc(),
                      "Multiple ProcessorResourceUnits associated with "
                      + ProcResKind->getName());
      }
      ProcUnitDef = *RI;
    }
  }
  if (!ProcUnitDef) {
    throw TGError(ProcResKind->getLoc(),
                  "No ProcessorResources associated with "
                  + ProcResKind->getName());
  }
  return ProcUnitDef;
}

// Iteratively add a resource and its super resources.
void CodeGenSchedModels::addProcResource(Record *ProcResKind,
                                         CodeGenProcModel &PM) {
  for (;;) {
    Record *ProcResUnits = findProcResUnits(ProcResKind, PM);

    // See if this ProcResource is already associated with this processor.
    RecIter I = std::find(PM.ProcResourceDefs.begin(),
                          PM.ProcResourceDefs.end(), ProcResUnits);
    if (I != PM.ProcResourceDefs.end())
      return;

    PM.ProcResourceDefs.push_back(ProcResUnits);
    if (!ProcResUnits->getValueInit("Super")->isComplete())
      return;

    ProcResKind = ProcResUnits->getValueAsDef("Super");
  }
}

// Add resources for a SchedWrite to this processor if they don't exist.
void CodeGenSchedModels::addWriteRes(Record *ProcWriteResDef, unsigned PIdx) {
  RecVec &WRDefs = ProcModels[PIdx].WriteResDefs;
  RecIter WRI = std::find(WRDefs.begin(), WRDefs.end(), ProcWriteResDef);
  if (WRI != WRDefs.end())
    return;
  WRDefs.push_back(ProcWriteResDef);

  // Visit ProcResourceKinds referenced by the newly discovered WriteRes.
  RecVec ProcResDefs = ProcWriteResDef->getValueAsListOfDefs("ProcResources");
  for (RecIter WritePRI = ProcResDefs.begin(), WritePRE = ProcResDefs.end();
       WritePRI != WritePRE; ++WritePRI) {
    addProcResource(*WritePRI, ProcModels[PIdx]);
  }
}

// Add resources for a ReadAdvance to this processor if they don't exist.
void CodeGenSchedModels::addReadAdvance(Record *ProcReadAdvanceDef,
                                        unsigned PIdx) {
  RecVec &RADefs = ProcModels[PIdx].ReadAdvanceDefs;
  RecIter I = std::find(RADefs.begin(), RADefs.end(), ProcReadAdvanceDef);
  if (I != RADefs.end())
    return;
  RADefs.push_back(ProcReadAdvanceDef);
}

#ifndef NDEBUG
void CodeGenProcModel::dump() const {
  dbgs() << Index << ": " << ModelName << " "
         << (ModelDef ? ModelDef->getName() : "inferred") << " "
         << (ItinsDef ? ItinsDef->getName() : "no itinerary") << '\n';
}

void CodeGenSchedRW::dump() const {
  dbgs() << Name << (IsVariadic ? " (V) " : " ");
  if (IsSequence) {
    dbgs() << "(";
    dumpIdxVec(Sequence);
    dbgs() << ")";
  }
}

void CodeGenSchedClass::dump(const CodeGenSchedModels* SchedModels) const {
  dbgs() << "SCHEDCLASS " << Name << '\n'
         << "  Writes: ";
  for (unsigned i = 0, N = Writes.size(); i < N; ++i) {
    SchedModels->getSchedWrite(Writes[i]).dump();
    if (i < N-1) {
      dbgs() << '\n';
      dbgs().indent(10);
    }
  }
  dbgs() << "\n  Reads: ";
  for (unsigned i = 0, N = Reads.size(); i < N; ++i) {
    SchedModels->getSchedRead(Reads[i]).dump();
    if (i < N-1) {
      dbgs() << '\n';
      dbgs().indent(10);
    }
  }
  dbgs() << "\n  ProcIdx: "; dumpIdxVec(ProcIndices); dbgs() << '\n';
}

void PredTransitions::dump() const {
  dbgs() << "Expanded Variants:\n";
  for (std::vector<PredTransition>::const_iterator
         TI = TransVec.begin(), TE = TransVec.end(); TI != TE; ++TI) {
    dbgs() << "{";
    for (SmallVectorImpl<PredCheck>::const_iterator
           PCI = TI->PredTerm.begin(), PCE = TI->PredTerm.end();
         PCI != PCE; ++PCI) {
      if (PCI != TI->PredTerm.begin())
        dbgs() << ", ";
      dbgs() << SchedModels.getSchedRW(PCI->RWIdx, PCI->IsRead).Name
             << ":" << PCI->Predicate->getName();
    }
    dbgs() << "},\n  => {";
    for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
           WSI = TI->WriteSequences.begin(), WSE = TI->WriteSequences.end();
         WSI != WSE; ++WSI) {
      dbgs() << "(";
      for (SmallVectorImpl<unsigned>::const_iterator
             WI = WSI->begin(), WE = WSI->end(); WI != WE; ++WI) {
        if (WI != WSI->begin())
          dbgs() << ", ";
        dbgs() << SchedModels.getSchedWrite(*WI).Name;
      }
      dbgs() << "),";
    }
    dbgs() << "}\n";
  }
}
#endif // NDEBUG
