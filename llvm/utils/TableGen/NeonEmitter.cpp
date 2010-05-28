//===- NeonEmitter.cpp - Generate arm_neon.h for use with clang -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting arm_neon.h, which includes
// a declaration and definition of each function specified by the ARM NEON 
// compiler interface.  See ARM document DUI0348B.
//
//===----------------------------------------------------------------------===//

#include "NeonEmitter.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include <string>

using namespace llvm;

void NeonEmitter::run(raw_ostream &OS) {
  EmitSourceFileHeader("ARM NEON Header", OS);
  
  // FIXME: emit license into file?
  
  OS << "#ifndef __ARM_NEON_H\n";
  OS << "#define __ARM_NEON_H\n\n";
  
  OS << "#ifndef __ARM_NEON__\n";
  OS << "#error \"NEON support not enabled\"\n";
  OS << "#endif\n\n";

  OS << "#include <stdint.h>\n\n";
  
  // EmitTypedefs(OS);
  
  // Process Records
  
  std::vector<Record*> RV = Records.getAllDerivedDefinitions("Inst");
  
  // Unique the pattern types, and assign them 
  
  // emit #define directives for uniq'd prototypes
  
  // emit record directives
  
  for (unsigned i = 0, e = RV.size(); i != e; ++i) {
    Record *R = RV[i];
    
    OS << LowercaseString(R->getName()) << "\n";

    std::string Types = R->getValueAsString("Types");
    std::string Pattern = R->getValueAsString("Pattern");
    
    OS << Types << "\n" << Pattern << "\n\n";
  }
  
  OS << "#endif /* __ARM_NEON_H */\n";
}
