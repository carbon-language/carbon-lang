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

// Convenience types
typedef std::vector<Record*> RecordList;
typedef std::vector<Record*>::iterator RecordListIter;


// SubtargetEmitter::run - Main subtarget enumeration emitter.
//
void SubtargetEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("Subtarget Enumeration Source Fragment", OS);
  RecordList Features = Records.getAllDerivedDefinitions("SubtargetFeature");
  RecordList Processors = Records.getAllDerivedDefinitions("Processor");

  OS << "namespace llvm {\n\n";
  
  { // Feature enumeration
    int i = 0;
    
    OS << "enum {\n";
    
    for (RecordListIter RI = Features.begin(), E = Features.end(); RI != E;){
      Record *R = *RI++;
      std::string Instance = R->getName();
      OS << "  "
         << Instance
         << " = "
         << " 1 << " << i++
         << ((RI != E) ? ",\n" : "\n");
    }
    
    OS << "};\n";
  }
  
  { // Feature key values
    OS << "\n\n"
       << "/// Sorted (by key) array of values for CPU features.\n"
       << "static SubtargetFeatureKV FeatureKV[] = {\n";
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
  }
  
  { // Feature key values
    OS << "\n\n"
       << "/// Sorted (by key) array of values for CPU subtype.\n"
       << "static const SubtargetFeatureKV SubTypeKV[] = {\n";
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
  }

  OS << "\n} // End llvm namespace \n";
}
