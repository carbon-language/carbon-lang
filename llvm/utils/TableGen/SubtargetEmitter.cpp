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

//
// RecordListIter - Simplify iterating through a std::vector of records.
// 
class RecordListIter {
  std::vector<Record*>::iterator RI;  // Currect cursor
  std::vector<Record*>::iterator E;   // End point

public:
  
  //
  // Ctor.
  //
  RecordListIter(RecordList &RL)
      : RI(RL.begin()), E(RL.end())
  {}
  
  
  //
  // isMore - Return true if more records are available.
  //
  bool isMore() const { return RI != E; }
  
  //
  // next - Return the next record or NULL if none.
  //
  Record *next() { return isMore() ? *RI++ : NULL; }
};

//
// DefListIter - Simplify iterating through a field which is a list of records.
// 
struct DefListIter {
  ListInit *List;  // List of DefInit
  unsigned N;      // Number of elements in list
  unsigned i;      // Current index in list

  //
  // Ctor - Lookup field and get list and length.
  //
  DefListIter(Record *R, const std::string &Name)
      : List(R->getValueAsListInit(Name)), N(List->getSize()), i(0)
  {}
  
  //
  // isMore - Return true if more records are available.
  //
  bool isMore() const { return i < N; }
  
  //
  // next - Return the next record or NULL if none.
  //
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
  // Get all records of class and sort
  RecordList Defs = Records.getAllDerivedDefinitions(ClassName);
  sort(Defs.begin(), Defs.end(), LessRecord());

  // Track position if isBits
  int i = 0;
  
  // Open enumeration
  OS << "enum {\n";
  
  // For each record
  RecordListIter DI(Defs);
  while (Record *R = DI.next()) {
    // Get and emit name
    std::string Name = R->getName();
    OS << "  " << Name;
    
    // If bit flags then emit expression (1 << i)
    if (isBits)  OS << " = " << " 1 << " << i++;

    // Depending on if more in the list, emit comma and new line
    OS << (DI.isMore() ? ",\n" : "\n");
  }
  
  // Close enumeration
  OS << "};\n";
}

//
// FeatureKeyValues - Emit data of all the subtarget features.  Used by command
// line.
//
void SubtargetEmitter::FeatureKeyValues(std::ostream &OS) {
  // Gather and sort all the features
  RecordList Features = Records.getAllDerivedDefinitions("SubtargetFeature");
  sort(Features.begin(), Features.end(), LessRecord());

  // Begin feature table
  OS << "// Sorted (by key) array of values for CPU features.\n"
     << "static llvm::SubtargetFeatureKV FeatureKV[] = {\n";
  
  // For each feature
  RecordListIter FI(Features);
  while (Record *R = FI.next()) {
    std::string Instance = R->getName();
    std::string Name = R->getValueAsString("Name");
    std::string Desc = R->getValueAsString("Desc");
    
    // Emit as { "feature", "decription", feactureEnum }
    OS << "  { "
       << "\"" << Name << "\", "
       << "\"" << Desc << "\", "
       << Instance
       << (FI.isMore() ? " },\n" : " }\n");
  }
  
  // End feature table
  OS << "};\n";

  // Emit size of table
  OS<<"\nenum {\n";
  OS<<"  FeatureKVSize = sizeof(FeatureKV)/sizeof(llvm::SubtargetFeatureKV)\n";
  OS<<"};\n";
}

//
// CPUKeyValues - Emit data of all the subtarget processors.  Used by command
// line.
//
void SubtargetEmitter::CPUKeyValues(std::ostream &OS) {
  // Gather and sort processor information
  RecordList Processors = Records.getAllDerivedDefinitions("Processor");
  sort(Processors.begin(), Processors.end(), LessRecordFieldName());

  // Begin processor table
  OS << "// Sorted (by key) array of values for CPU subtype.\n"
     << "static const llvm::SubtargetFeatureKV SubTypeKV[] = {\n";
     
  // For each processor
  RecordListIter PI(Processors);
  while (Record *R = PI.next()) {
    std::string Name = R->getValueAsString("Name");
    DefListIter FI(R, "Features");
    
    // Emit as { "cpu", "description", f1 | f2 | ... fn },
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
  
  // End processor table
  OS << "};\n";

  // Emit size of table
  OS<<"\nenum {\n";
  OS<<"  SubTypeKVSize = sizeof(SubTypeKV)/sizeof(llvm::SubtargetFeatureKV)\n";
  OS<<"};\n";
}

//
// CollectAllItinClasses - Gathers and enumerates all the itinerary classes.
// Returns itinerary class count.
//
unsigned SubtargetEmitter::CollectAllItinClasses(IntMap &ItinClassesMap) {
  // Gather and sort all itinerary classes
  RecordList ICL = Records.getAllDerivedDefinitions("InstrItinClass");
  sort(ICL.begin(), ICL.end(), LessRecord());

  // Track enumeration
  unsigned Index = 0;
  
  // For each class
  RecordListIter ICI(ICL);
  while (Record *ItinClass = ICI.next()) {
    // Get name of itinerary class
    std::string Name = ItinClass->getName();
    // Assign itinerary class a unique number
    ItinClassesMap[Name] = Index++;
  }
  
  // Return itinerary class count
  return Index;
}

//
// FormItineraryString - Compose a string containing the data initialization
// for the specified itinerary.  N is the number of stages.
//
void SubtargetEmitter::FormItineraryString(Record *ItinData,
                                           std::string &ItinString,
                                           unsigned &N) {
  // Set up stages iterator
  DefListIter SLI(ItinData, "Stages");
  // Get stage count
  N = SLI.N;

  // For each stage
  while (Record *Stage = SLI.next()) {
    // Form string as ,{ cycles, u1 | u2 | ... | un }
    int Cycles = Stage->getValueAsInt("Cycles");
    ItinString += "  ,{ " + itostr(Cycles) + ", ";
    
    // For each unit
    DefListIter ULI(Stage, "Units");
    while (Record *Unit = ULI.next()) {
      std::string Name = Unit->getName();
      ItinString += Name;
      if (ULI.isMore())ItinString += " | ";
    }
    
    // Close off stage
    ItinString += " }";
  }
}

//
// EmitStageData - Generate unique itinerary stages.  Record itineraries for 
// processors.
//
void SubtargetEmitter::EmitStageData(std::ostream &OS,
                                     unsigned N,
                                     IntMap &ItinClassesMap, 
                                     ProcessorList &ProcList) {
  // Gather processor iteraries
  RecordList Itins = Records.getAllDerivedDefinitions("ProcessorItineraries");
  
  // If just no itinerary then don't bother
  if (Itins.size() < 2) return;

  // Begin stages table
  OS << "static llvm::InstrStage Stages[] = {\n"
        "  { 0, 0 } // No itinerary\n";
        
  IntMap ItinMap;
  unsigned Index  = 1;
  RecordListIter II(Itins);
  while (Record *Itin = II.next()) {
    // Get processor itinerary name
    std::string Name = Itin->getName();
    
    // Skip default
    if (Name == "NoItineraries") continue;
    
    // Create and expand processor itinerary to cover all itinerary classes
    IntineraryList IL;
    IL.resize(N);
    
    // For each itinerary
    DefListIter IDLI(Itin, "IID");
    while (Record *ItinData = IDLI.next()) {
      // Get string and stage count
      std::string ItinString;
      unsigned M;
      FormItineraryString(ItinData, ItinString, M);

      // Check to see if it already exists
      unsigned Find = ItinMap[ItinString];
      
      // If new itinerary
      if (Find == 0) {
        // Emit as ,{ cycles, u1 | u2 | ... | un } // index
        OS << ItinString << " // " << Index << "\n";
        ItinMap[ItinString] = Find = Index++;
      }
      
      // Set up itinerary as location and location + stage count
      InstrItinerary Intinerary = { Find, Find + M };

      // Locate where to inject into processor itinerary table
      std::string Name = ItinData->getValueAsDef("TheClass")->getName();
      Find = ItinClassesMap[Name];
      
      // Inject - empty slots will be 0, 0
      IL[Find] = Intinerary;
    }
    
    // Add process itinerary to list
    ProcList.push_back(IL);
  }
  
  // End stages table
  OS << "};\n";
}

//
// EmitProcessData - Generate data for processor itineraries.
//
void SubtargetEmitter::EmitProcessData(std::ostream &OS,
                                       ProcessorList &ProcList) {
  // Get an iterator for processor itinerary stages
  ProcessorList::iterator PLI = ProcList.begin();
  
  // For each processor itinerary
  RecordList Itins = Records.getAllDerivedDefinitions("ProcessorItineraries");
  RecordListIter II(Itins);
  while (Record *Itin = II.next()) {
    // Get processor itinerary name
    std::string Name = Itin->getName();
    
    // Skip default
    if (Name == "NoItineraries") continue;

    // Begin processor itinerary table
    OS << "\n";
    OS << "static llvm::InstrItinerary " << Name << "[] = {\n";
    
    // For each itinerary class
    IntineraryList &IL = *PLI++;
    unsigned Index = 0;
    for (IntineraryList::iterator ILI = IL.begin(), E = IL.end(); ILI != E;) {
      InstrItinerary &Intinerary = *ILI++;
      
      // Emit in the form of { first, last } // index
      if (Intinerary.First == 0) {
        OS << "  { 0, 0 }";
      } else {
        OS << "  { " << Intinerary.First << ", " << Intinerary.Last << " }";
      }
      
      if (ILI != E) OS << ",";
      OS << " // " << Index++ << "\n";
    }
    
    // End processor itinerary table
    OS << "};\n";
  }
}

//
// EmitData - Emits all stages and itineries, folding common patterns.
//
void SubtargetEmitter::EmitData(std::ostream &OS) {
  IntMap ItinClassesMap;
  ProcessorList ProcList;
  
  // Enumerate all the itinerary classes
  unsigned N = CollectAllItinClasses(ItinClassesMap);
  // Emit the stage data
  EmitStageData(OS, N, ItinClassesMap, ProcList);
  // Emit the processor itinerary data
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
//  Enumeration(OS, "InstrItinClass", false);
//  OS<<"\n";
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
