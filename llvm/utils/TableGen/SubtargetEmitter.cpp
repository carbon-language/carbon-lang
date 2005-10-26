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
#include <set>
using namespace llvm;

//
// Convenience types.
//
typedef std::vector<Record*> RecordList;
typedef std::vector<Record*>::iterator RecordListIter;

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
  
  for (RecordListIter RI = Defs.begin(), E = Defs.end(); RI != E;) {
    Record *R = *RI++;
    std::string Instance = R->getName();
    OS << "  "
       << Instance;
    if (isBits) {
      OS << " = "
         << " 1 << " << i++;
    }
    OS << ((RI != E) ? ",\n" : "\n");
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
  for (RecordListIter RI = Features.begin(), E = Features.end(); RI != E;) {
    Record *R = *RI++;
    std::string Instance = R->getName();
    std::string Name = R->getValueAsString("Name");
    std::string Desc = R->getValueAsString("Desc");
    OS << "  { "
       << "\"" << Name << "\", "
       << "\"" << Desc << "\", "
       << Instance
       << ((RI != E) ? " },\n" : " }\n");
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
  for (RecordListIter RI = Processors.begin(), E = Processors.end();
       RI != E;) {
    Record *R = *RI++;
    std::string Name = R->getValueAsString("Name");
    Record *ProcItin = R->getValueAsDef("ProcItin");
    ListInit *Features = R->getValueAsListInit("Features");
    unsigned N = Features->getSize();
    OS << "  { "
       << "\"" << Name << "\", "
       << "\"Select the " << Name << " processor\", ";
       
    
    if (N == 0) {
      OS << "0";
    } else {
      for (unsigned i = 0; i < N; ) {
        if (DefInit *DI = dynamic_cast<DefInit*>(Features->getElement(i++))) {
          Record *Feature = DI->getDef();
          std::string Name = Feature->getName();
          OS << Name;
          if (i != N) OS << " | ";
        } else {
          throw "Feature: " + Name +
                " expected feature in processor feature list!";
        }
      }
    }
    
    OS << ((RI != E) ? " },\n" : " }\n");
  }
  OS << "};\n";

  OS<<"\nenum {\n";
  OS<<"  SubTypeKVSize = sizeof(SubTypeKV)/sizeof(llvm::SubtargetFeatureKV)\n";
  OS<<"};\n";
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
        
  for (RecordListIter RI = Features.begin(), E = Features.end(); RI != E;) {
    Record *R = *RI++;
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
  std::vector<Record*> Targets = Records.getAllDerivedDefinitions("Target");
  if (Targets.size() == 0)
    throw std::string("ERROR: No 'Target' subclasses defined!");
  if (Targets.size() != 1)
    throw std::string("ERROR: Multiple subclasses of Target defined!");
  Target = Targets[0]->getName();

  EmitSourceFileHeader("Subtarget Enumeration Source Fragment", OS);

  OS << "#include \"llvm/Target/SubtargetFeature.h\"\n\n";
  
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
  ParseFeaturesFunction(OS);
}
