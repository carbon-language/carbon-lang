//===- SubtargetEmitter.h - Generate subtarget enumerations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits subtarget enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef SUBTARGET_EMITTER_H
#define SUBTARGET_EMITTER_H

#include "TableGenBackend.h"
#include "llvm/Target/TargetInstrItineraries.h"
#include <vector>
#include <map>
#include <string>


namespace llvm {

//
// Convenience types.
//
typedef std::map<std::string, unsigned> IntMap;
typedef std::vector<InstrItinerary> IntineraryList;
typedef std::vector<IntineraryList> ProcessorList;

class SubtargetEmitter : public TableGenBackend {
  
  RecordKeeper &Records;
  std::string Target;
  
  void Enumeration(std::ostream &OS, const char *ClassName, bool isBits);
  void FeatureKeyValues(std::ostream &OS);
  void CPUKeyValues(std::ostream &OS);
  unsigned CollectAllItinClasses(IntMap &ItinClassesMap);
  void FormItineraryString(Record *ItinData, std::string &ItinString,
                           unsigned &N);
  void EmitStageData(std::ostream &OS, unsigned N,
                     IntMap &ItinClassesMap, ProcessorList &ProcList);
  void EmitProcessData(std::ostream &OS, ProcessorList &ProcList);
  void EmitData(std::ostream &OS);
  void ParseFeaturesFunction(std::ostream &OS);
  
public:
  SubtargetEmitter(RecordKeeper &R) : Records(R) {}

  // run - Output the subtarget enumerations, returning true on failure.
  void run(std::ostream &o);

};


} // End llvm namespace

#endif



