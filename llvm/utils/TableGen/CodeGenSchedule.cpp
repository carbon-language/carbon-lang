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
#endif // NDEBUG
