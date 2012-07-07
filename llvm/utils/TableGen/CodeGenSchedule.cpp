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
#include "llvm/Support/Debug.h"

using namespace llvm;

// CodeGenModels ctor interprets machine model records and populates maps.
CodeGenSchedModels::CodeGenSchedModels(RecordKeeper &RK,
                                       const CodeGenTarget &TGT):
  Records(RK), Target(TGT), NumItineraryClasses(0), HasProcItineraries(false) {

  // Populate SchedClassIdxMap and set NumItineraryClasses.
  CollectSchedClasses();

  // Populate ProcModelMap.
  CollectProcModels();
}

// Visit all the instruction definitions for this target to gather and enumerate
// the itinerary classes. These are the explicitly specified SchedClasses. More
// SchedClasses may be inferred.
void CodeGenSchedModels::CollectSchedClasses() {

  // NoItinerary is always the first class at Index=0
  SchedClasses.resize(1);
  SchedClasses.back().Name = "NoItinerary";
  SchedClassIdxMap[SchedClasses.back().Name] = 0;

  // Gather and sort all itinerary classes used by instruction descriptions.
  std::vector<Record*> ItinClassList;
  for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
         E = Target.inst_end(); I != E; ++I) {
    Record *SchedDef = (*I)->TheDef->getValueAsDef("Itinerary");
    // Map a new SchedClass with no index.
    if (!SchedClassIdxMap.count(SchedDef->getName())) {
      SchedClassIdxMap[SchedDef->getName()] = 0;
      ItinClassList.push_back(SchedDef);
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

  // TODO: Infer classes from non-itinerary scheduler resources.
}

// Gather all processor models.
void CodeGenSchedModels::CollectProcModels() {
  std::vector<Record*> ProcRecords =
    Records.getAllDerivedDefinitions("Processor");
  std::sort(ProcRecords.begin(), ProcRecords.end(), LessRecordFieldName());

  // Reserve space because we can. Reallocation would be ok.
  ProcModels.reserve(ProcRecords.size());

  // For each processor, find a unique machine model.
  for (unsigned i = 0, N = ProcRecords.size(); i < N; ++i)
    addProcModel(ProcRecords[i]);
}

// Get a unique processor model based on the defined MachineModel and
// ProcessorItineraries.
void CodeGenSchedModels::addProcModel(Record *ProcDef) {
  unsigned Idx = getProcModelIdx(ProcDef);
  if (Idx < ProcModels.size())
    return;

  Record *ModelDef = ProcDef->getValueAsDef("SchedModel");
  Record *ItinsDef = ProcDef->getValueAsDef("ProcItin");

  std::string ModelName = ModelDef->getName();
  const std::string &ItinName = ItinsDef->getName();

  bool NoModel = ModelDef->getValueAsBit("NoModel");
  bool hasTopLevelItin = !ItinsDef->getValueAsListOfDefs("IID").empty();
  if (NoModel) {
    // If an itinerary is defined without a machine model, infer a new model.
    if (NoModel && hasTopLevelItin) {
      ModelName = ItinName + "Model";
      ModelDef = NULL;
    }
  }
  else {
    // If a machine model is defined, the itinerary must be defined within it
    // rather than in the Processor definition itself.
    assert(!hasTopLevelItin && "Itinerary must be defined in SchedModel");
    ItinsDef = ModelDef->getValueAsDef("Itineraries");
  }

  ProcModelMap[getProcModelKey(ProcDef)]= ProcModels.size();

  ProcModels.push_back(CodeGenProcModel(ModelName, ModelDef, ItinsDef));

  std::vector<Record*> ItinRecords = ItinsDef->getValueAsListOfDefs("IID");
  CollectProcItin(ProcModels.back(), ItinRecords);
}

// Gather the processor itineraries.
void CodeGenSchedModels::CollectProcItin(CodeGenProcModel &ProcModel,
                                         std::vector<Record*> ItinRecords) {
  // Skip empty itinerary.
  if (ItinRecords.empty())
    return;

  HasProcItineraries = true;

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
    ProcModel.ItinDefList[getItinClassIdx(ItinDef)] = ItinData;
  }
#ifndef NDEBUG
  // Check for missing itinerary entries.
  assert(!ProcModel.ItinDefList[0] && "NoItinerary class can't have rec");
  for (unsigned i = 1, N = ProcModel.ItinDefList.size(); i < N; ++i) {
    if (!ProcModel.ItinDefList[i])
      DEBUG(dbgs() << ProcModel.ItinsDef->getName()
            << " missing itinerary for class " << SchedClasses[i].Name << '\n');
  }
#endif
}
