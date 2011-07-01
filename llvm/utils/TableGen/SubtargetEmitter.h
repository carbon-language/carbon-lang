//===- SubtargetEmitter.h - Generate subtarget enumerations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits subtarget enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef SUBTARGET_EMITTER_H
#define SUBTARGET_EMITTER_H

#include "TableGenBackend.h"
#include "llvm/MC/MCInstrItineraries.h"
#include <vector>
#include <map>
#include <string>


namespace llvm {

class SubtargetEmitter : public TableGenBackend {

  RecordKeeper &Records;
  std::string Target;
  bool HasItineraries;

  void Enumeration(raw_ostream &OS, const char *ClassName, bool isBits);
  unsigned FeatureKeyValues(raw_ostream &OS);
  unsigned CPUKeyValues(raw_ostream &OS);
  unsigned CollectAllItinClasses(raw_ostream &OS,
                                 std::map<std::string,unsigned> &ItinClassesMap,
                                 std::vector<Record*> &ItinClassList);
  void FormItineraryStageString(const std::string &Names,
                                Record *ItinData, std::string &ItinString,
                                unsigned &NStages);
  void FormItineraryOperandCycleString(Record *ItinData, std::string &ItinString,
                                       unsigned &NOperandCycles);
  void FormItineraryBypassString(const std::string &Names,
                                 Record *ItinData,
                                 std::string &ItinString, unsigned NOperandCycles);
  void EmitStageAndOperandCycleData(raw_ostream &OS, unsigned NItinClasses,
                     std::map<std::string, unsigned> &ItinClassesMap,
                     std::vector<Record*> &ItinClassList,
                     std::vector<std::vector<InstrItinerary> > &ProcList);
  void EmitProcessorData(raw_ostream &OS,
                         std::vector<Record*> &ItinClassList,
                         std::vector<std::vector<InstrItinerary> > &ProcList);
  void EmitProcessorLookup(raw_ostream &OS);
  void EmitData(raw_ostream &OS);
  void ParseFeaturesFunction(raw_ostream &OS, unsigned NumFeatures,
                             unsigned NumProcs);

public:
  SubtargetEmitter(RecordKeeper &R) : Records(R), HasItineraries(false) {}

  // run - Output the subtarget enumerations, returning true on failure.
  void run(raw_ostream &o);

};


} // End llvm namespace

#endif



