//===- CodeGenSchedule.h - Scheduling Machine Models ------------*- C++ -*-===//
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

#ifndef CODEGEN_SCHEDULE_H
#define CODEGEN_SCHEDULE_H

#include "llvm/TableGen/Record.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {

class CodeGenTarget;

// Scheduling class.
//
// Each instruction description will be mapped to a scheduling class. It may be
// an explicitly defined itinerary class, or an inferred class in which case
// ItinClassDef == NULL.
struct CodeGenSchedClass {
  std::string Name;
  unsigned Index;
  Record *ItinClassDef;

  CodeGenSchedClass(): Index(0), ItinClassDef(0) {}
  CodeGenSchedClass(Record *rec): Index(0), ItinClassDef(rec) {
    Name = rec->getName();
  }
};

// Processor model.
//
// ModelName is a unique name used to name an instantiation of MCSchedModel.
//
// ModelDef is NULL for inferred Models. This happens when a processor defines
// an itinerary but no machine model. If the processer defines neither a machine
// model nor itinerary, then ModelDef remains pointing to NoModel. NoModel has
// the special "NoModel" field set to true.
//
// ItinsDef always points to a valid record definition, but may point to the
// default NoItineraries. NoItineraries has an empty list of InstrItinData
// records.
//
// ItinDefList orders this processor's InstrItinData records by SchedClass idx.
struct CodeGenProcModel {
  std::string ModelName;
  Record *ModelDef;
  Record *ItinsDef;

  // Array of InstrItinData records indexed by CodeGenSchedClass::Index.
  // The list is empty if the subtarget has no itineraries.
  std::vector<Record *> ItinDefList;

  CodeGenProcModel(const std::string &Name, Record *MDef, Record *IDef):
    ModelName(Name), ModelDef(MDef), ItinsDef(IDef) {}
};

// Top level container for machine model data.
class CodeGenSchedModels {
  RecordKeeper &Records;
  const CodeGenTarget &Target;

  // List of unique SchedClasses.
  std::vector<CodeGenSchedClass> SchedClasses;

  // Map SchedClass name to itinerary index.
  // These are either explicit itinerary classes or inferred classes.
  StringMap<unsigned> SchedClassIdxMap;

  // SchedClass indices 1 up to and including NumItineraryClasses identify
  // itinerary classes that are explicitly used for this target's instruction
  // definitions. NoItinerary always has index 0 regardless of whether it is
  // explicitly referenced.
  //
  // Any inferred SchedClass have a index greater than NumItineraryClasses.
  unsigned NumItineraryClasses;

  // List of unique processor models.
  std::vector<CodeGenProcModel> ProcModels;

  // Map Processor's MachineModel + ProcItin fields to a CodeGenProcModel index.
  typedef DenseMap<std::pair<Record*, Record*>, unsigned> ProcModelMapTy;
  ProcModelMapTy ProcModelMap;

  // True if any processors have nonempty itineraries.
  bool HasProcItineraries;

public:
  CodeGenSchedModels(RecordKeeper& RK, const CodeGenTarget &TGT);

  // Check if any instructions are assigned to an explicit itinerary class other
  // than NoItinerary.
  bool hasItineraryClasses() const { return NumItineraryClasses > 0; }

  // Return the number of itinerary classes in use by this target's instruction
  // descriptions, not including "NoItinerary".
  unsigned numItineraryClasses() const {
    return NumItineraryClasses;
  }

  // Get a SchedClass from its index.
  const CodeGenSchedClass &getSchedClass(unsigned Idx) {
    assert(Idx < SchedClasses.size() && "bad SchedClass index");
    return SchedClasses[Idx];
  }

  // Get an itinerary class's index. Value indices are '0' for NoItinerary up to
  // and including numItineraryClasses().
  unsigned getItinClassIdx(Record *ItinDef) const {
    assert(SchedClassIdxMap.count(ItinDef->getName()) && "missing ItinClass");
    unsigned Idx = SchedClassIdxMap.lookup(ItinDef->getName());
    assert(Idx <= NumItineraryClasses && "bad ItinClass index");
    return Idx;
  }

  bool hasProcessorItineraries() const {
    return HasProcItineraries;
  }

  // Get an existing machine model for a processor definition.
  const CodeGenProcModel &getProcModel(Record *ProcDef) const {
    unsigned idx = getProcModelIdx(ProcDef);
    assert(idx < ProcModels.size() && "missing machine model");
    return ProcModels[idx];
  }

  // Iterate over the unique processor models.
  typedef std::vector<CodeGenProcModel>::const_iterator ProcIter;
  ProcIter procModelBegin() const { return ProcModels.begin(); }
  ProcIter procModelEnd() const { return ProcModels.end(); }

private:
  // Get a key that can uniquely identify a machine model.
  ProcModelMapTy::key_type getProcModelKey(Record *ProcDef) const {
    Record *ModelDef = ProcDef->getValueAsDef("SchedModel");
    Record *ItinsDef = ProcDef->getValueAsDef("ProcItin");
    return std::make_pair(ModelDef, ItinsDef);
  }

  // Get the unique index of a machine model.
  unsigned getProcModelIdx(Record *ProcDef) const {
    ProcModelMapTy::const_iterator I =
      ProcModelMap.find(getProcModelKey(ProcDef));
    if (I == ProcModelMap.end())
      return ProcModels.size();
    return I->second;
  }

  // Initialize a new processor model if it is unique.
  void addProcModel(Record *ProcDef);

  void CollectSchedClasses();
  void CollectProcModels();
  void CollectProcItin(CodeGenProcModel &ProcModel,
                       std::vector<Record*> ItinRecords);
};

} // namespace llvm

#endif
