//===- SubtargetEmitter.cpp - Generate subtarget enumerations -------------===//
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

#include "CodeGenTarget.h"
#include "CodeGenSchedule.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/Debug.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <map>
#include <string>
#include <vector>
using namespace llvm;

namespace {
class SubtargetEmitter {

  RecordKeeper &Records;
  CodeGenSchedModels &SchedModels;
  std::string Target;

  void Enumeration(raw_ostream &OS, const char *ClassName, bool isBits);
  unsigned FeatureKeyValues(raw_ostream &OS);
  unsigned CPUKeyValues(raw_ostream &OS);
  void FormItineraryStageString(const std::string &Names,
                                Record *ItinData, std::string &ItinString,
                                unsigned &NStages);
  void FormItineraryOperandCycleString(Record *ItinData, std::string &ItinString,
                                       unsigned &NOperandCycles);
  void FormItineraryBypassString(const std::string &Names,
                                 Record *ItinData,
                                 std::string &ItinString, unsigned NOperandCycles);
  void EmitStageAndOperandCycleData(raw_ostream &OS,
                                    std::vector<std::vector<InstrItinerary> >
                                      &ProcItinLists);
  void EmitItineraries(raw_ostream &OS,
                       std::vector<std::vector<InstrItinerary> >
                         &ProcItinLists);
  void EmitProcessorProp(raw_ostream &OS, const Record *R, const char *Name,
                         char Separator);
  void EmitProcessorModels(raw_ostream &OS);
  void EmitProcessorLookup(raw_ostream &OS);
  void EmitSchedModel(raw_ostream &OS);
  void ParseFeaturesFunction(raw_ostream &OS, unsigned NumFeatures,
                             unsigned NumProcs);

public:
  SubtargetEmitter(RecordKeeper &R, CodeGenTarget &TGT):
    Records(R), SchedModels(TGT.getSchedModels()), Target(TGT.getName()) {}

  void run(raw_ostream &o);

};
} // End anonymous namespace

//
// Enumeration - Emit the specified class as an enumeration.
//
void SubtargetEmitter::Enumeration(raw_ostream &OS,
                                   const char *ClassName,
                                   bool isBits) {
  // Get all records of class and sort
  std::vector<Record*> DefList = Records.getAllDerivedDefinitions(ClassName);
  std::sort(DefList.begin(), DefList.end(), LessRecord());

  unsigned N = DefList.size();
  if (N == 0)
    return;
  if (N > 64) {
    errs() << "Too many (> 64) subtarget features!\n";
    exit(1);
  }

  OS << "namespace " << Target << " {\n";

  // For bit flag enumerations with more than 32 items, emit constants.
  // Emit an enum for everything else.
  if (isBits && N > 32) {
    // For each record
    for (unsigned i = 0; i < N; i++) {
      // Next record
      Record *Def = DefList[i];

      // Get and emit name and expression (1 << i)
      OS << "  const uint64_t " << Def->getName() << " = 1ULL << " << i << ";\n";
    }
  } else {
    // Open enumeration
    OS << "enum {\n";

    // For each record
    for (unsigned i = 0; i < N;) {
      // Next record
      Record *Def = DefList[i];

      // Get and emit name
      OS << "  " << Def->getName();

      // If bit flags then emit expression (1 << i)
      if (isBits)  OS << " = " << " 1ULL << " << i;

      // Depending on 'if more in the list' emit comma
      if (++i < N) OS << ",";

      OS << "\n";
    }

    // Close enumeration
    OS << "};\n";
  }

  OS << "}\n";
}

//
// FeatureKeyValues - Emit data of all the subtarget features.  Used by the
// command line.
//
unsigned SubtargetEmitter::FeatureKeyValues(raw_ostream &OS) {
  // Gather and sort all the features
  std::vector<Record*> FeatureList =
                           Records.getAllDerivedDefinitions("SubtargetFeature");

  if (FeatureList.empty())
    return 0;

  std::sort(FeatureList.begin(), FeatureList.end(), LessRecordFieldName());

  // Begin feature table
  OS << "// Sorted (by key) array of values for CPU features.\n"
     << "extern const llvm::SubtargetFeatureKV " << Target
     << "FeatureKV[] = {\n";

  // For each feature
  unsigned NumFeatures = 0;
  for (unsigned i = 0, N = FeatureList.size(); i < N; ++i) {
    // Next feature
    Record *Feature = FeatureList[i];

    const std::string &Name = Feature->getName();
    const std::string &CommandLineName = Feature->getValueAsString("Name");
    const std::string &Desc = Feature->getValueAsString("Desc");

    if (CommandLineName.empty()) continue;

    // Emit as { "feature", "description", featureEnum, i1 | i2 | ... | in }
    OS << "  { "
       << "\"" << CommandLineName << "\", "
       << "\"" << Desc << "\", "
       << Target << "::" << Name << ", ";

    const std::vector<Record*> &ImpliesList =
      Feature->getValueAsListOfDefs("Implies");

    if (ImpliesList.empty()) {
      OS << "0ULL";
    } else {
      for (unsigned j = 0, M = ImpliesList.size(); j < M;) {
        OS << Target << "::" << ImpliesList[j]->getName();
        if (++j < M) OS << " | ";
      }
    }

    OS << " }";
    ++NumFeatures;

    // Depending on 'if more in the list' emit comma
    if ((i + 1) < N) OS << ",";

    OS << "\n";
  }

  // End feature table
  OS << "};\n";

  return NumFeatures;
}

//
// CPUKeyValues - Emit data of all the subtarget processors.  Used by command
// line.
//
unsigned SubtargetEmitter::CPUKeyValues(raw_ostream &OS) {
  // Gather and sort processor information
  std::vector<Record*> ProcessorList =
                          Records.getAllDerivedDefinitions("Processor");
  std::sort(ProcessorList.begin(), ProcessorList.end(), LessRecordFieldName());

  // Begin processor table
  OS << "// Sorted (by key) array of values for CPU subtype.\n"
     << "extern const llvm::SubtargetFeatureKV " << Target
     << "SubTypeKV[] = {\n";

  // For each processor
  for (unsigned i = 0, N = ProcessorList.size(); i < N;) {
    // Next processor
    Record *Processor = ProcessorList[i];

    const std::string &Name = Processor->getValueAsString("Name");
    const std::vector<Record*> &FeatureList =
      Processor->getValueAsListOfDefs("Features");

    // Emit as { "cpu", "description", f1 | f2 | ... fn },
    OS << "  { "
       << "\"" << Name << "\", "
       << "\"Select the " << Name << " processor\", ";

    if (FeatureList.empty()) {
      OS << "0ULL";
    } else {
      for (unsigned j = 0, M = FeatureList.size(); j < M;) {
        OS << Target << "::" << FeatureList[j]->getName();
        if (++j < M) OS << " | ";
      }
    }

    // The "0" is for the "implies" section of this data structure.
    OS << ", 0ULL }";

    // Depending on 'if more in the list' emit comma
    if (++i < N) OS << ",";

    OS << "\n";
  }

  // End processor table
  OS << "};\n";

  return ProcessorList.size();
}

//
// FormItineraryStageString - Compose a string containing the stage
// data initialization for the specified itinerary.  N is the number
// of stages.
//
void SubtargetEmitter::FormItineraryStageString(const std::string &Name,
                                                Record *ItinData,
                                                std::string &ItinString,
                                                unsigned &NStages) {
  // Get states list
  const std::vector<Record*> &StageList =
    ItinData->getValueAsListOfDefs("Stages");

  // For each stage
  unsigned N = NStages = StageList.size();
  for (unsigned i = 0; i < N;) {
    // Next stage
    const Record *Stage = StageList[i];

    // Form string as ,{ cycles, u1 | u2 | ... | un, timeinc, kind }
    int Cycles = Stage->getValueAsInt("Cycles");
    ItinString += "  { " + itostr(Cycles) + ", ";

    // Get unit list
    const std::vector<Record*> &UnitList = Stage->getValueAsListOfDefs("Units");

    // For each unit
    for (unsigned j = 0, M = UnitList.size(); j < M;) {
      // Add name and bitwise or
      ItinString += Name + "FU::" + UnitList[j]->getName();
      if (++j < M) ItinString += " | ";
    }

    int TimeInc = Stage->getValueAsInt("TimeInc");
    ItinString += ", " + itostr(TimeInc);

    int Kind = Stage->getValueAsInt("Kind");
    ItinString += ", (llvm::InstrStage::ReservationKinds)" + itostr(Kind);

    // Close off stage
    ItinString += " }";
    if (++i < N) ItinString += ", ";
  }
}

//
// FormItineraryOperandCycleString - Compose a string containing the
// operand cycle initialization for the specified itinerary.  N is the
// number of operands that has cycles specified.
//
void SubtargetEmitter::FormItineraryOperandCycleString(Record *ItinData,
                         std::string &ItinString, unsigned &NOperandCycles) {
  // Get operand cycle list
  const std::vector<int64_t> &OperandCycleList =
    ItinData->getValueAsListOfInts("OperandCycles");

  // For each operand cycle
  unsigned N = NOperandCycles = OperandCycleList.size();
  for (unsigned i = 0; i < N;) {
    // Next operand cycle
    const int OCycle = OperandCycleList[i];

    ItinString += "  " + itostr(OCycle);
    if (++i < N) ItinString += ", ";
  }
}

void SubtargetEmitter::FormItineraryBypassString(const std::string &Name,
                                                 Record *ItinData,
                                                 std::string &ItinString,
                                                 unsigned NOperandCycles) {
  const std::vector<Record*> &BypassList =
    ItinData->getValueAsListOfDefs("Bypasses");
  unsigned N = BypassList.size();
  unsigned i = 0;
  for (; i < N;) {
    ItinString += Name + "Bypass::" + BypassList[i]->getName();
    if (++i < NOperandCycles) ItinString += ", ";
  }
  for (; i < NOperandCycles;) {
    ItinString += " 0";
    if (++i < NOperandCycles) ItinString += ", ";
  }
}

//
// EmitStageAndOperandCycleData - Generate unique itinerary stages and operand
// cycle tables. Create a list of InstrItinerary objects (ProcItinLists) indexed
// by CodeGenSchedClass::Index.
//
void SubtargetEmitter::
EmitStageAndOperandCycleData(raw_ostream &OS,
                             std::vector<std::vector<InstrItinerary> >
                               &ProcItinLists) {

  // Multiple processor models may share an itinerary record. Emit it once.
  SmallPtrSet<Record*, 8> ItinsDefSet;

  // Emit functional units for all the itineraries.
  for (CodeGenSchedModels::ProcIter PI = SchedModels.procModelBegin(),
         PE = SchedModels.procModelEnd(); PI != PE; ++PI) {

    if (!ItinsDefSet.insert(PI->ItinsDef))
      continue;

    std::vector<Record*> FUs = PI->ItinsDef->getValueAsListOfDefs("FU");
    if (FUs.empty())
      continue;

    const std::string &Name = PI->ItinsDef->getName();
    OS << "\n// Functional units for \"" << Name << "\"\n"
       << "namespace " << Name << "FU {\n";

    for (unsigned j = 0, FUN = FUs.size(); j < FUN; ++j)
      OS << "  const unsigned " << FUs[j]->getName()
         << " = 1 << " << j << ";\n";

    OS << "}\n";

    std::vector<Record*> BPs = PI->ItinsDef->getValueAsListOfDefs("BP");
    if (BPs.size()) {
      OS << "\n// Pipeline forwarding pathes for itineraries \"" << Name
         << "\"\n" << "namespace " << Name << "Bypass {\n";

      OS << "  const unsigned NoBypass = 0;\n";
      for (unsigned j = 0, BPN = BPs.size(); j < BPN; ++j)
        OS << "  const unsigned " << BPs[j]->getName()
           << " = 1 << " << j << ";\n";

      OS << "}\n";
    }
  }

  // Begin stages table
  std::string StageTable = "\nextern const llvm::InstrStage " + Target +
                           "Stages[] = {\n";
  StageTable += "  { 0, 0, 0, llvm::InstrStage::Required }, // No itinerary\n";

  // Begin operand cycle table
  std::string OperandCycleTable = "extern const unsigned " + Target +
    "OperandCycles[] = {\n";
  OperandCycleTable += "  0, // No itinerary\n";

  // Begin pipeline bypass table
  std::string BypassTable = "extern const unsigned " + Target +
    "ForwardingPaths[] = {\n";
  BypassTable += " 0, // No itinerary\n";

  // For each Itinerary across all processors, add a unique entry to the stages,
  // operand cycles, and pipepine bypess tables. Then add the new Itinerary
  // object with computed offsets to the ProcItinLists result.
  unsigned StageCount = 1, OperandCycleCount = 1;
  std::map<std::string, unsigned> ItinStageMap, ItinOperandMap;
  for (CodeGenSchedModels::ProcIter PI = SchedModels.procModelBegin(),
         PE = SchedModels.procModelEnd(); PI != PE; ++PI) {
    const CodeGenProcModel &ProcModel = *PI;

    // Add process itinerary to the list.
    ProcItinLists.resize(ProcItinLists.size()+1);

    // If this processor defines no itineraries, then leave the itinerary list
    // empty.
    std::vector<InstrItinerary> &ItinList = ProcItinLists.back();
    if (ProcModel.ItinDefList.empty())
      continue;

    // Reserve index==0 for NoItinerary.
    ItinList.resize(SchedModels.numItineraryClasses()+1);

    const std::string &Name = ProcModel.ItinsDef->getName();

    // For each itinerary data
    for (unsigned SchedClassIdx = 0,
           SchedClassEnd = ProcModel.ItinDefList.size();
         SchedClassIdx < SchedClassEnd; ++SchedClassIdx) {

      // Next itinerary data
      Record *ItinData = ProcModel.ItinDefList[SchedClassIdx];

      // Get string and stage count
      std::string ItinStageString;
      unsigned NStages = 0;
      if (ItinData)
        FormItineraryStageString(Name, ItinData, ItinStageString, NStages);

      // Get string and operand cycle count
      std::string ItinOperandCycleString;
      unsigned NOperandCycles = 0;
      std::string ItinBypassString;
      if (ItinData) {
        FormItineraryOperandCycleString(ItinData, ItinOperandCycleString,
                                        NOperandCycles);

        FormItineraryBypassString(Name, ItinData, ItinBypassString,
                                  NOperandCycles);
      }

      // Check to see if stage already exists and create if it doesn't
      unsigned FindStage = 0;
      if (NStages > 0) {
        FindStage = ItinStageMap[ItinStageString];
        if (FindStage == 0) {
          // Emit as { cycles, u1 | u2 | ... | un, timeinc }, // indices
          StageTable += ItinStageString + ", // " + itostr(StageCount);
          if (NStages > 1)
            StageTable += "-" + itostr(StageCount + NStages - 1);
          StageTable += "\n";
          // Record Itin class number.
          ItinStageMap[ItinStageString] = FindStage = StageCount;
          StageCount += NStages;
        }
      }

      // Check to see if operand cycle already exists and create if it doesn't
      unsigned FindOperandCycle = 0;
      if (NOperandCycles > 0) {
        std::string ItinOperandString = ItinOperandCycleString+ItinBypassString;
        FindOperandCycle = ItinOperandMap[ItinOperandString];
        if (FindOperandCycle == 0) {
          // Emit as  cycle, // index
          OperandCycleTable += ItinOperandCycleString + ", // ";
          std::string OperandIdxComment = itostr(OperandCycleCount);
          if (NOperandCycles > 1)
            OperandIdxComment += "-"
              + itostr(OperandCycleCount + NOperandCycles - 1);
          OperandCycleTable += OperandIdxComment + "\n";
          // Record Itin class number.
          ItinOperandMap[ItinOperandCycleString] =
            FindOperandCycle = OperandCycleCount;
          // Emit as bypass, // index
          BypassTable += ItinBypassString + ", // " + OperandIdxComment + "\n";
          OperandCycleCount += NOperandCycles;
        }
      }

      // Set up itinerary as location and location + stage count
      int NumUOps = ItinData ? ItinData->getValueAsInt("NumMicroOps") : 0;
      InstrItinerary Intinerary = { NumUOps, FindStage, FindStage + NStages,
                                    FindOperandCycle,
                                    FindOperandCycle + NOperandCycles};

      // Inject - empty slots will be 0, 0
      ItinList[SchedClassIdx] = Intinerary;
    }
  }

  // Closing stage
  StageTable += "  { 0, 0, 0, llvm::InstrStage::Required } // End stages\n";
  StageTable += "};\n";

  // Closing operand cycles
  OperandCycleTable += "  0 // End operand cycles\n";
  OperandCycleTable += "};\n";

  BypassTable += " 0 // End bypass tables\n";
  BypassTable += "};\n";

  // Emit tables.
  OS << StageTable;
  OS << OperandCycleTable;
  OS << BypassTable;
}

//
// EmitProcessorData - Generate data for processor itineraries that were
// computed during EmitStageAndOperandCycleData(). ProcItinLists lists all
// Itineraries for each processor. The Itinerary lists are indexed on
// CodeGenSchedClass::Index.
//
void SubtargetEmitter::
EmitItineraries(raw_ostream &OS,
                std::vector<std::vector<InstrItinerary> > &ProcItinLists) {

  // Multiple processor models may share an itinerary record. Emit it once.
  SmallPtrSet<Record*, 8> ItinsDefSet;

  // For each processor's machine model
  std::vector<std::vector<InstrItinerary> >::iterator
      ProcItinListsIter = ProcItinLists.begin();
  for (CodeGenSchedModels::ProcIter PI = SchedModels.procModelBegin(),
         PE = SchedModels.procModelEnd(); PI != PE; ++PI) {

    Record *ItinsDef = PI->ItinsDef;
    if (!ItinsDefSet.insert(ItinsDef))
      continue;

    // Get processor itinerary name
    const std::string &Name = ItinsDef->getName();

    // Get the itinerary list for the processor.
    assert(ProcItinListsIter != ProcItinLists.end() && "bad iterator");
    std::vector<InstrItinerary> &ItinList = *ProcItinListsIter++;

    OS << "\n";
    OS << "static const llvm::InstrItinerary ";
    if (ItinList.empty()) {
      OS << '*' << Name << " = 0;\n";
      continue;
    }

    // Begin processor itinerary table
    OS << Name << "[] = {\n";

    // For each itinerary class in CodeGenSchedClass::Index order.
    for (unsigned j = 0, M = ItinList.size(); j < M; ++j) {
      InstrItinerary &Intinerary = ItinList[j];

      // Emit Itinerary in the form of
      // { firstStage, lastStage, firstCycle, lastCycle } // index
      OS << "  { " <<
        Intinerary.NumMicroOps << ", " <<
        Intinerary.FirstStage << ", " <<
        Intinerary.LastStage << ", " <<
        Intinerary.FirstOperandCycle << ", " <<
        Intinerary.LastOperandCycle << " }" <<
        ", // " << j << " " << SchedModels.getSchedClass(j).Name << "\n";
    }
    // End processor itinerary table
    OS << "  { 0, ~0U, ~0U, ~0U, ~0U } // end marker\n";
    OS << "};\n";
  }
}

// Emit either the value defined in the TableGen Record, or the default
// value defined in the C++ header. The Record is null if the processor does not
// define a model.
void SubtargetEmitter::EmitProcessorProp(raw_ostream &OS, const Record *R,
                                         const char *Name, char Separator) {
  OS << "  ";
  int V = R ? R->getValueAsInt(Name) : -1;
  if (V >= 0)
    OS << V << Separator << " // " << Name;
  else
    OS << "MCSchedModel::Default" << Name << Separator;
  OS << '\n';
}

void SubtargetEmitter::EmitProcessorModels(raw_ostream &OS) {
  // For each processor model.
  for (CodeGenSchedModels::ProcIter PI = SchedModels.procModelBegin(),
         PE = SchedModels.procModelEnd(); PI != PE; ++PI) {
    // Skip default
    // Begin processor itinerary properties
    OS << "\n";
    OS << "static const llvm::MCSchedModel " << PI->ModelName << "(\n";
    EmitProcessorProp(OS, PI->ModelDef, "IssueWidth", ',');
    EmitProcessorProp(OS, PI->ModelDef, "MinLatency", ',');
    EmitProcessorProp(OS, PI->ModelDef, "LoadLatency", ',');
    EmitProcessorProp(OS, PI->ModelDef, "HighLatency", ',');
    EmitProcessorProp(OS, PI->ModelDef, "MispredictPenalty", ',');
    if (SchedModels.hasItineraryClasses())
      OS << "  " << PI->ItinsDef->getName();
    else
      OS << "  0";
    OS << ");\n";
  }
}

//
// EmitProcessorLookup - generate cpu name to itinerary lookup table.
//
void SubtargetEmitter::EmitProcessorLookup(raw_ostream &OS) {
  // Gather and sort processor information
  std::vector<Record*> ProcessorList =
                          Records.getAllDerivedDefinitions("Processor");
  std::sort(ProcessorList.begin(), ProcessorList.end(), LessRecordFieldName());

  // Begin processor table
  OS << "\n";
  OS << "// Sorted (by key) array of itineraries for CPU subtype.\n"
     << "extern const llvm::SubtargetInfoKV "
     << Target << "ProcSchedKV[] = {\n";

  // For each processor
  for (unsigned i = 0, N = ProcessorList.size(); i < N;) {
    // Next processor
    Record *Processor = ProcessorList[i];

    const std::string &Name = Processor->getValueAsString("Name");
    const std::string &ProcModelName =
      SchedModels.getProcModel(Processor).ModelName;

    // Emit as { "cpu", procinit },
    OS << "  { "
       << "\"" << Name << "\", "
       << "(const void *)&" << ProcModelName;

    OS << " }";

    // Depending on ''if more in the list'' emit comma
    if (++i < N) OS << ",";

    OS << "\n";
  }

  // End processor table
  OS << "};\n";
}

//
// EmitSchedModel - Emits all scheduling model tables, folding common patterns.
//
void SubtargetEmitter::EmitSchedModel(raw_ostream &OS) {
  if (SchedModels.hasItineraryClasses()) {
    std::vector<std::vector<InstrItinerary> > ProcItinLists;
    // Emit the stage data
    EmitStageAndOperandCycleData(OS, ProcItinLists);
    EmitItineraries(OS, ProcItinLists);
  }
  // Emit the processor machine model
  EmitProcessorModels(OS);
  // Emit the processor lookup data
  EmitProcessorLookup(OS);
}

//
// ParseFeaturesFunction - Produces a subtarget specific function for parsing
// the subtarget features string.
//
void SubtargetEmitter::ParseFeaturesFunction(raw_ostream &OS,
                                             unsigned NumFeatures,
                                             unsigned NumProcs) {
  std::vector<Record*> Features =
                       Records.getAllDerivedDefinitions("SubtargetFeature");
  std::sort(Features.begin(), Features.end(), LessRecord());

  OS << "// ParseSubtargetFeatures - Parses features string setting specified\n"
     << "// subtarget options.\n"
     << "void llvm::";
  OS << Target;
  OS << "Subtarget::ParseSubtargetFeatures(StringRef CPU, StringRef FS) {\n"
     << "  DEBUG(dbgs() << \"\\nFeatures:\" << FS);\n"
     << "  DEBUG(dbgs() << \"\\nCPU:\" << CPU << \"\\n\\n\");\n";

  if (Features.empty()) {
    OS << "}\n";
    return;
  }

  OS << "  uint64_t Bits = ReInitMCSubtargetInfo(CPU, FS);\n";

  for (unsigned i = 0; i < Features.size(); i++) {
    // Next record
    Record *R = Features[i];
    const std::string &Instance = R->getName();
    const std::string &Value = R->getValueAsString("Value");
    const std::string &Attribute = R->getValueAsString("Attribute");

    if (Value=="true" || Value=="false")
      OS << "  if ((Bits & " << Target << "::"
         << Instance << ") != 0) "
         << Attribute << " = " << Value << ";\n";
    else
      OS << "  if ((Bits & " << Target << "::"
         << Instance << ") != 0 && "
         << Attribute << " < " << Value << ") "
         << Attribute << " = " << Value << ";\n";
  }

  OS << "}\n";
}

//
// SubtargetEmitter::run - Main subtarget enumeration emitter.
//
void SubtargetEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Subtarget Enumeration Source Fragment", OS);

  OS << "\n#ifdef GET_SUBTARGETINFO_ENUM\n";
  OS << "#undef GET_SUBTARGETINFO_ENUM\n";

  OS << "namespace llvm {\n";
  Enumeration(OS, "SubtargetFeature", true);
  OS << "} // End llvm namespace \n";
  OS << "#endif // GET_SUBTARGETINFO_ENUM\n\n";

  OS << "\n#ifdef GET_SUBTARGETINFO_MC_DESC\n";
  OS << "#undef GET_SUBTARGETINFO_MC_DESC\n";

  OS << "namespace llvm {\n";
#if 0
  OS << "namespace {\n";
#endif
  unsigned NumFeatures = FeatureKeyValues(OS);
  OS << "\n";
  unsigned NumProcs = CPUKeyValues(OS);
  OS << "\n";
  EmitSchedModel(OS);
  OS << "\n";
#if 0
  OS << "}\n";
#endif

  // MCInstrInfo initialization routine.
  OS << "static inline void Init" << Target
     << "MCSubtargetInfo(MCSubtargetInfo *II, "
     << "StringRef TT, StringRef CPU, StringRef FS) {\n";
  OS << "  II->InitMCSubtargetInfo(TT, CPU, FS, ";
  if (NumFeatures)
    OS << Target << "FeatureKV, ";
  else
    OS << "0, ";
  if (NumProcs)
    OS << Target << "SubTypeKV, ";
  else
    OS << "0, ";
  if (SchedModels.hasItineraryClasses()) {
    OS << Target << "ProcSchedKV, "
       << Target << "Stages, "
       << Target << "OperandCycles, "
       << Target << "ForwardingPaths, ";
  } else
    OS << "0, 0, 0, 0, ";
  OS << NumFeatures << ", " << NumProcs << ");\n}\n\n";

  OS << "} // End llvm namespace \n";

  OS << "#endif // GET_SUBTARGETINFO_MC_DESC\n\n";

  OS << "\n#ifdef GET_SUBTARGETINFO_TARGET_DESC\n";
  OS << "#undef GET_SUBTARGETINFO_TARGET_DESC\n";

  OS << "#include \"llvm/Support/Debug.h\"\n";
  OS << "#include \"llvm/Support/raw_ostream.h\"\n";
  ParseFeaturesFunction(OS, NumFeatures, NumProcs);

  OS << "#endif // GET_SUBTARGETINFO_TARGET_DESC\n\n";

  // Create a TargetSubtargetInfo subclass to hide the MC layer initialization.
  OS << "\n#ifdef GET_SUBTARGETINFO_HEADER\n";
  OS << "#undef GET_SUBTARGETINFO_HEADER\n";

  std::string ClassName = Target + "GenSubtargetInfo";
  OS << "namespace llvm {\n";
  OS << "class DFAPacketizer;\n";
  OS << "struct " << ClassName << " : public TargetSubtargetInfo {\n"
     << "  explicit " << ClassName << "(StringRef TT, StringRef CPU, "
     << "StringRef FS);\n"
     << "public:\n"
     << "  DFAPacketizer *createDFAPacketizer(const InstrItineraryData *IID)"
     << " const;\n"
     << "};\n";
  OS << "} // End llvm namespace \n";

  OS << "#endif // GET_SUBTARGETINFO_HEADER\n\n";

  OS << "\n#ifdef GET_SUBTARGETINFO_CTOR\n";
  OS << "#undef GET_SUBTARGETINFO_CTOR\n";

  OS << "namespace llvm {\n";
  OS << "extern const llvm::SubtargetFeatureKV " << Target << "FeatureKV[];\n";
  OS << "extern const llvm::SubtargetFeatureKV " << Target << "SubTypeKV[];\n";
  if (SchedModels.hasItineraryClasses()) {
    OS << "extern const llvm::SubtargetInfoKV " << Target << "ProcSchedKV[];\n";
    OS << "extern const llvm::InstrStage " << Target << "Stages[];\n";
    OS << "extern const unsigned " << Target << "OperandCycles[];\n";
    OS << "extern const unsigned " << Target << "ForwardingPaths[];\n";
  }

  OS << ClassName << "::" << ClassName << "(StringRef TT, StringRef CPU, "
     << "StringRef FS)\n"
     << "  : TargetSubtargetInfo() {\n"
     << "  InitMCSubtargetInfo(TT, CPU, FS, ";
  if (NumFeatures)
    OS << Target << "FeatureKV, ";
  else
    OS << "0, ";
  if (NumProcs)
    OS << Target << "SubTypeKV, ";
  else
    OS << "0, ";
  if (SchedModels.hasItineraryClasses()) {
    OS << Target << "ProcSchedKV, "
       << Target << "Stages, "
       << Target << "OperandCycles, "
       << Target << "ForwardingPaths, ";
  } else
    OS << "0, 0, 0, 0, ";
  OS << NumFeatures << ", " << NumProcs << ");\n}\n\n";
  OS << "} // End llvm namespace \n";

  OS << "#endif // GET_SUBTARGETINFO_CTOR\n\n";
}

namespace llvm {

void EmitSubtarget(RecordKeeper &RK, raw_ostream &OS) {
  CodeGenTarget CGTarget(RK);
  SubtargetEmitter(RK, CGTarget).run(OS);
}

} // End llvm namespace
