//===- IntrinsicEmitter.cpp - Generate intrinsic information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits information about intrinsic functions.
//
//===----------------------------------------------------------------------===//

#include "IntrinsicEmitter.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
// IntrinsicEmitter Implementation
//===----------------------------------------------------------------------===//

void IntrinsicEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("Intrinsic Function Source Fragment", OS);
  
  std::vector<CodeGenIntrinsic> Ints = LoadIntrinsics(Records);

  // Emit the enum information.
  EmitEnumInfo(Ints, OS);

  // Emit the intrinsic ID -> name table.
  EmitIntrinsicToNameTable(Ints, OS);
  
  // Emit the function name recognizer.
  EmitFnNameRecognizer(Ints, OS);
  
  // Emit the intrinsic verifier.
  EmitVerifier(Ints, OS);
  
  // Emit mod/ref info for each function.
  EmitModRefInfo(Ints, OS);
  
  // Emit table of non-memory accessing intrinsics.
  EmitNoMemoryInfo(Ints, OS);
  
  // Emit side effect info for each intrinsic.
  EmitSideEffectInfo(Ints, OS);

  // Emit a list of intrinsics with corresponding GCC builtins.
  EmitGCCBuiltinList(Ints, OS);

  // Emit code to translate GCC builtins into LLVM intrinsics.
  EmitIntrinsicToGCCBuiltinMap(Ints, OS);
}

void IntrinsicEmitter::EmitEnumInfo(const std::vector<CodeGenIntrinsic> &Ints,
                                    std::ostream &OS) {
  OS << "// Enum values for Intrinsics.h\n";
  OS << "#ifdef GET_INTRINSIC_ENUM_VALUES\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    OS << "    " << Ints[i].EnumName;
    OS << ((i != e-1) ? ", " : "  ");
    OS << std::string(40-Ints[i].EnumName.size(), ' ') 
      << "// " << Ints[i].Name << "\n";
  }
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitFnNameRecognizer(const std::vector<CodeGenIntrinsic> &Ints, 
                     std::ostream &OS) {
  // Build a function name -> intrinsic name mapping.
  std::map<std::string, std::string> IntMapping;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    IntMapping[Ints[i].Name] = Ints[i].EnumName;
    
  OS << "// Function name -> enum value recognizer code.\n";
  OS << "#ifdef GET_FUNCTION_RECOGNIZER\n";
  OS << "  switch (Name[5]) {\n";
  OS << "  default: break;\n";
  // Emit the intrinsics in sorted order.
  char LastChar = 0;
  for (std::map<std::string, std::string>::iterator I = IntMapping.begin(),
       E = IntMapping.end(); I != E; ++I) {
    if (I->first[5] != LastChar) {
      LastChar = I->first[5];
      OS << "  case '" << LastChar << "':\n";
    }
    
    OS << "    if (Name == \"" << I->first << "\") return Intrinsic::"
       << I->second << ";\n";
  }
  OS << "  }\n";
  OS << "  // The 'llvm.' namespace is reserved!\n";
  OS << "  assert(0 && \"Unknown LLVM intrinsic function!\");\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitIntrinsicToNameTable(const std::vector<CodeGenIntrinsic> &Ints, 
                         std::ostream &OS) {
  std::vector<std::string> Names;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    Names.push_back(Ints[i].Name);
  std::sort(Names.begin(), Names.end());
  
  OS << "// Intrinsic ID to name table\n";
  OS << "#ifdef GET_INTRINSIC_NAME_TABLE\n";
  OS << "  // Note that entry #0 is the invalid intrinsic!\n";
  for (unsigned i = 0, e = Names.size(); i != e; ++i)
    OS << "  \"" << Names[i] << "\",\n";
  OS << "#endif\n\n";
}

static void EmitTypeVerify(std::ostream &OS, const std::string &Val,
                           Record *ArgType) {
  OS << "    Assert1(" << Val << "->getTypeID() == "
     << ArgType->getValueAsString("TypeVal") << ",\n"
     << "            \"Illegal intrinsic type!\", IF);\n";

  // If this is a packed type, check that the subtype and size are correct.
  if (ArgType->isSubClassOf("LLVMPackedType")) {
    Record *SubType = ArgType->getValueAsDef("ElTy");
    OS << "    Assert1(cast<PackedType>(" << Val
       << ")->getElementType()->getTypeID() == "
       << SubType->getValueAsString("TypeVal") << ",\n"
       << "            \"Illegal intrinsic type!\", IF);\n";
    OS << "    Assert1(cast<PackedType>(" << Val << ")->getNumElements() == "
       << ArgType->getValueAsInt("NumElts") << ",\n"
       << "            \"Illegal intrinsic type!\", IF);\n";
  }
}

void IntrinsicEmitter::EmitVerifier(const std::vector<CodeGenIntrinsic> &Ints, 
                                    std::ostream &OS) {
  OS << "// Verifier::visitIntrinsicFunctionCall code.\n";
  OS << "#ifdef GET_INTRINSIC_VERIFIER\n";
  OS << "  switch (ID) {\n";
  OS << "  default: assert(0 && \"Invalid intrinsic!\");\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    OS << "  case Intrinsic::" << Ints[i].EnumName << ":\t\t// "
       << Ints[i].Name << "\n";
    OS << "    Assert1(FTy->getNumParams() == " << Ints[i].ArgTypes.size()-1
       << ",\n"
       << "            \"Illegal # arguments for intrinsic function!\", IF);\n";
    EmitTypeVerify(OS, "FTy->getReturnType()", Ints[i].ArgTypeDefs[0]);
    for (unsigned j = 1; j != Ints[i].ArgTypes.size(); ++j)
      EmitTypeVerify(OS, "FTy->getParamType(" + utostr(j-1) + ")",
                     Ints[i].ArgTypeDefs[j]);
    OS << "    break;\n";
  }
  OS << "  }\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::EmitModRefInfo(const std::vector<CodeGenIntrinsic> &Ints,
                                      std::ostream &OS) {
  OS << "// BasicAliasAnalysis code.\n";
  OS << "#ifdef GET_MODREF_BEHAVIOR\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    switch (Ints[i].ModRef) {
    default: break;
    case CodeGenIntrinsic::NoMem:
      OS << "  NoMemoryTable.push_back(\"" << Ints[i].Name << "\");\n";
      break;
    case CodeGenIntrinsic::ReadArgMem:
    case CodeGenIntrinsic::ReadMem:
      OS << "  OnlyReadsMemoryTable.push_back(\"" << Ints[i].Name << "\");\n";
      break;
    }
  }
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitNoMemoryInfo(const std::vector<CodeGenIntrinsic> &Ints, std::ostream &OS) {
  OS << "// SelectionDAGIsel code.\n";
  OS << "#ifdef GET_NO_MEMORY_INTRINSICS\n";
  OS << "  switch (IntrinsicID) {\n";
  OS << "  default: break;\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    switch (Ints[i].ModRef) {
    default: break;
    case CodeGenIntrinsic::NoMem:
      OS << "  case Intrinsic::" << Ints[i].EnumName << ":\n";
      break;
    }
  }
  OS << "    return true; // These intrinsics have no side effects.\n";
  OS << "  }\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitSideEffectInfo(const std::vector<CodeGenIntrinsic> &Ints, std::ostream &OS){
  OS << "// isInstructionTriviallyDead code.\n";
  OS << "#ifdef GET_SIDE_EFFECT_INFO\n";
  OS << "  switch (F->getIntrinsicID()) {\n";
  OS << "  default: break;\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    switch (Ints[i].ModRef) {
    default: break;
    case CodeGenIntrinsic::NoMem:
    case CodeGenIntrinsic::ReadArgMem:
    case CodeGenIntrinsic::ReadMem:
      OS << "  case Intrinsic::" << Ints[i].EnumName << ":\n";
      break;
    }
  }
  OS << "    return true; // These intrinsics have no side effects.\n";
  OS << "  }\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitGCCBuiltinList(const std::vector<CodeGenIntrinsic> &Ints, std::ostream &OS){
  OS << "// Get the GCC builtin that corresponds to an LLVM intrinsic.\n";
  OS << "#ifdef GET_GCC_BUILTIN_NAME\n";
  OS << "  switch (F->getIntrinsicID()) {\n";
  OS << "  default: BuiltinName = \"\"; break;\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    if (!Ints[i].GCCBuiltinName.empty()) {
      OS << "  case Intrinsic::" << Ints[i].EnumName << ": BuiltinName = \""
         << Ints[i].GCCBuiltinName << "\"; break;\n";
    }
  }
  OS << "  }\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitIntrinsicToGCCBuiltinMap(const std::vector<CodeGenIntrinsic> &Ints, 
                             std::ostream &OS) {
  typedef std::map<std::pair<std::string, std::string>, std::string> BIMTy;
  BIMTy BuiltinMap;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    if (!Ints[i].GCCBuiltinName.empty()) {
      std::pair<std::string, std::string> Key(Ints[i].GCCBuiltinName,
                                              Ints[i].TargetPrefix);
      if (!BuiltinMap.insert(std::make_pair(Key, Ints[i].EnumName)).second)
        throw "Intrinsic '" + Ints[i].TheDef->getName() +
              "': duplicate GCC builtin name!";
    }
  }
  
  OS << "// Get the LLVM intrinsic that corresponds to a GCC builtin.\n";
  OS << "// This is used by the C front-end.  The GCC builtin name is passed\n";
  OS << "// in as BuiltinName, and a target prefix (e.g. 'ppc') is passed\n";
  OS << "// in as TargetPrefix.  The result is assigned to 'IntrinsicID'.\n";
  OS << "#ifdef GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN\n";
  OS << "  if (0);\n";
  // Note: this could emit significantly better code if we cared.
  for (BIMTy::iterator I = BuiltinMap.begin(), E = BuiltinMap.end();I != E;++I){
    OS << "  else if (";
    if (!I->first.second.empty()) {
      // Emit this as a strcmp, so it can be constant folded by the FE.
      OS << "!strcmp(TargetPrefix, \"" << I->first.second << "\") &&\n"
         << "           ";
    }
    OS << "!strcmp(BuiltinName, \"" << I->first.first << "\"))\n";
    OS << "    IntrinsicID = Intrinsic::" << I->second << ";\n";
  }
  OS << "  else\n";
  OS << "    IntrinsicID = Intrinsic::not_intrinsic;\n";
  OS << "#endif\n\n";
}
