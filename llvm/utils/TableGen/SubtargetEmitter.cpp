//===- SubtargetEmitter.cpp - Generate subtarget enumerations -------------===//
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

#include "SubtargetEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
using namespace llvm;

//
// Convenience types.
//
typedef std::vector<Record*> RecordList;

struct RecordListIter {
  std::vector<Record*>::iterator RI;
  std::vector<Record*>::iterator E;
  
  RecordListIter(RecordList &RL)
      : RI(RL.begin()), E(RL.end())
  {}
  
  bool isMore() const { return RI != E; }
  
  Record *next() { return isMore() ? *RI++ : NULL; }
};

struct DefListIter {
  ListInit *List;
  unsigned N;
  unsigned i;

  DefListIter(Record *R, const std::string &Name)
      : List(R->getValueAsListInit(Name)), N(List->getSize()), i(0)
  {}
  
  bool isMore() const { return i < N; }
  
  Record *next() {
    if (isMore()) {
      if (DefInit *DI = dynamic_cast<DefInit*>(List->getElement(i++))) {
        return DI->getDef();
      }
    }
    return NULL;
  }
};

//
// Record sort by name function.
//
struct LessRecord {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return Rec1->getName() < Rec2->getName();
  }
};

//
// Record sort by field "Name" function.
//
struct LessRecordFieldName {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return Rec1->getValueAsString("Name") < Rec2->getValueAsString("Name");
  }
};

//
// Enumeration - Emit the specified class as an enumeration.
//
void SubtargetEmitter::Enumeration(std::ostream &OS,
                                   const char *ClassName,
                                   bool isBits) {
  RecordList Defs = Records.getAllDerivedDefinitions(ClassName);
  sort(Defs.begin(), Defs.end(), LessRecord());

  int i = 0;
  
  OS << "enum {\n";
  
  RecordListIter DI(Defs);
  while (Record *R = DI.next()) {
    std::string Instance = R->getName();
    OS << "  "
       << Instance;
    if (isBits) {
      OS << " = "
         << " 1 << " << i++;
    }
    OS << (DI.isMore() ? ",\n" : "\n");
  }
  
  OS << "};\n";
}

//
// FeatureKeyValues - Emit data of all the subtarget features.  Used by command
// line.
//
void SubtargetEmitter::FeatureKeyValues(std::ostream &OS) {
  RecordList Features = Records.getAllDerivedDefinitions("SubtargetFeature");
  sort(Features.begin(), Features.end(), LessRecord());

  OS << "// Sorted (by key) array of values for CPU features.\n"
     << "static llvm::SubtargetFeatureKV FeatureKV[] = {\n";
  RecordListIter FI(Features);
  while (Record *R = FI.next()) {
    std::string Instance = R->getName();
    std::string Name = R->getValueAsString("Name");
    std::string Desc = R->getValueAsString("Desc");
    OS << "  { "
       << "\"" << Name << "\", "
       << "\"" << Desc << "\", "
       << Instance
       << (FI.isMore() ? " },\n" : " }\n");
  }
  OS << "};\n";

  OS<<"\nenum {\n";
  OS<<"  FeatureKVSize = sizeof(FeatureKV)/sizeof(llvm::SubtargetFeatureKV)\n";
  OS<<"};\n";
}

//
// CPUKeyValues - Emit data of all the subtarget processors.  Used by command
// line.
//
void SubtargetEmitter::CPUKeyValues(std::ostream &OS) {
  RecordList Processors = Records.getAllDerivedDefinitions("Processor");
  sort(Processors.begin(), Processors.end(), LessRecordFieldName());

  OS << "// Sorted (by key) array of values for CPU subtype.\n"
     << "static const llvm::SubtargetFeatureKV SubTypeKV[] = {\n";
  RecordListIter PI(Processors);
  while (Record *R = PI.next()) {
    std::string Name = R->getValueAsString("Name");
    DefListIter FI(R, "Features");
    
    OS << "  { "
       << "\"" << Name << "\", "
       << "\"Select the " << Name << " processor\", ";
    
    if (!FI.isMore()) {
      OS << "0";
    } else {
      while (Record *Feature = FI.next()) {
        std::string Name = Feature->getName();
        OS << Name;
        if (FI.isMore()) OS << " | ";
      }
    }
    
    OS << (PI.isMore() ? " },\n" : " }\n");
  }
  OS << "};\n";

  OS<<"\nenum {\n";
  OS<<"  SubTypeKVSize = sizeof(SubTypeKV)/sizeof(llvm::SubtargetFeatureKV)\n";
  OS<<"};\n";
}

//
// CollectAllItinClasses - Gathers and enumerates all the itinerary classes.
//
unsigned SubtargetEmitter::CollectAllItinClasses(IntMap &ItinClassesMap) {
  RecordList ICL = Records.getAllDerivedDefinitions("InstrItinClass");
  sort(ICL.begin(), ICL.end(), LessRecord());
  
  RecordListIter ICI(ICL);
  unsigned Index = 0;
  while (Record *ItinClass = ICI.next()) {
    std::string Name = ItinClass->getName();
    ItinClassesMap[Name] = Index++;
  }
  
  return Index;
}

//
// FormItineraryString - Compose a string containing the data initialization
// for the specified itinerary.  N is the number of stages.
//
void SubtargetEmitter::FormItineraryString(Record *ItinData,
                                           std::string &ItinString,
                                           unsigned &N) {
  DefListIter SLI(ItinData, "Stages");
  N = SLI.N;
  while (Record *Stage = SLI.next()) {
    int Cycles = Stage->getValueAsInt("Cycles");
    ItinString += "  ,{ " + itostr(Cycles) + ", ";
    
    DefListIter ULI(Stage, "Units");
    while (Record *Unit = ULI.next()) {
      std::string Name = Unit->getName();
      ItinString += Name;
      if (ULI.isMore())ItinString += " | ";
    }
  }
  
  ItinString += " }";
}

//
// EmitStageData - Generate unique itinerary stages.  Record itineraries for 
// processors.
//
void SubtargetEmitter::EmitStageData(std::ostream &OS,
                                     unsigned N,
                                     IntMap &ItinClassesMap, 
                                     ProcessorList &ProcList) {
  OS << "static llvm::InstrStage Stages[] = {\n"
        "  { 0, 0 } // No itinerary\n";
        
  IntMap ItinMap;
  unsigned Index  = 1;
  RecordList Itins = Records.getAllDerivedDefinitions("ProcessorItineraries");
  RecordListIter II(Itins);
  while (Record *Itin = II.next()) {
    std::string Name = Itin->getName();
    if (Name == "NoItineraries") continue;
    
    IntineraryList IL;
    IL.resize(N);
    
    DefListIter IDLI(Itin, "IID");
    while (Record *ItinData = IDLI.next()) {
      std::string ItinString;
      unsigned M;
      FormItineraryString(ItinData, ItinString, M);

      unsigned Find = ItinMap[ItinString];
      
      if (Find == 0) {
        OS << ItinString << " // " << Index << "\n";
        ItinMap[ItinString] = Find = Index++;
      }
      
      InstrItinerary Intinerary = { Find, Find + M };

      std::string Name = ItinData->getValueAsDef("TheClass")->getName();
      Find = ItinClassesMap[Name];
      IL[Find] = Intinerary;
    }
    
    ProcList.push_back(IL);
  }
  
  OS << "};\n";
}

//
// EmitProcessData - Generate data for processor itineraries.
//
void SubtargetEmitter::EmitProcessData(std::ostream &OS,
                                       ProcessorList &ProcList) {
  ProcessorList::iterator PLI = ProcList.begin();
  RecordList Itins = Records.getAllDerivedDefinitions("ProcessorItineraries");
  RecordListIter II(Itins);
  while (Record *Itin = II.next()) {
    std::string Name = Itin->getName();
    if (Name == "NoItineraries") continue;

    OS << "\n";
    OS << "static llvm::InstrItinerary " << Name << "[] = {\n";
    
    IntineraryList &IL = *PLI++;
    unsigned Index = 0;
    for (IntineraryList::iterator ILI = IL.begin(), E = IL.end(); ILI != E;) {
      InstrItinerary &Intinerary = *ILI++;
      
      if (Intinerary.First == 0) {
        OS << "  { 0, 0 }";
      } else {
        OS << "  { " << Intinerary.First << ", " << Intinerary.Last << " }";
      }
      
      if (ILI != E) OS << ",";
      OS << " // " << Index++ << "\n";
    }
    OS << "};\n";
  }
}

//
// EmitData - Emits all stages and itineries, folding common patterns.
//
void SubtargetEmitter::EmitData(std::ostream &OS) {
  IntMap ItinClassesMap;
  ProcessorList ProcList;
  
  unsigned N = CollectAllItinClasses(ItinClassesMap);
  EmitStageData(OS, N, ItinClassesMap, ProcList);
  EmitProcessData(OS, ProcList);
}

//
// ParseFeaturesFunction - Produces a subtarget specific function for parsing
// the subtarget features string.
//
void SubtargetEmitter::ParseFeaturesFunction(std::ostream &OS) {
  RecordList Features = Records.getAllDerivedDefinitions("SubtargetFeature");
  sort(Features.begin(), Features.end(), LessRecord());

  OS << "// ParseSubtargetFeatures - Parses features string setting specified\n" 
        "// subtarget options.\n" 
        "void llvm::";
  OS << Target;
  OS << "Subtarget::ParseSubtargetFeatures(const std::string &FS,\n"
        "                                  const std::string &CPU) {\n"
        "  SubtargetFeatures Features(FS);\n"
        "  Features.setCPUIfNone(CPU);\n"
        "  uint32_t Bits =  Features.getBits(SubTypeKV, SubTypeKVSize,\n"
        "                                    FeatureKV, FeatureKVSize);\n";
        
  RecordListIter FI(Features);
  while (Record *R = FI.next()) {
    std::string Instance = R->getName();
    std::string Name = R->getValueAsString("Name");
    std::string Type = R->getValueAsString("Type");
    std::string Attribute = R->getValueAsString("Attribute");
    
    OS << "  " << Attribute << " = (Bits & " << Instance << ") != 0;\n";
  }
  OS << "}\n";
}

// 
// SubtargetEmitter::run - Main subtarget enumeration emitter.
//
void SubtargetEmitter::run(std::ostream &OS) {
  Target = CodeGenTarget().getName();

  EmitSourceFileHeader("Subtarget Enumeration Source Fragment", OS);

  OS << "#include \"llvm/Target/SubtargetFeature.h\"\n";
  OS << "#include \"llvm/Target/TargetInstrItineraries.h\"\n\n";
  
  Enumeration(OS, "FuncUnit", true);
  OS<<"\n";
  Enumeration(OS, "InstrItinClass", false);
  OS<<"\n";
  Enumeration(OS, "SubtargetFeature", true);
  OS<<"\n";
  FeatureKeyValues(OS);
  OS<<"\n";
  CPUKeyValues(OS);
  OS<<"\n";
  EmitData(OS);
  OS<<"\n";
  ParseFeaturesFunction(OS);
}
