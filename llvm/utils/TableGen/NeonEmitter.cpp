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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include <string>

using namespace llvm;

static void ParseTypes(Record *r, std::string &s,
                       SmallVectorImpl<StringRef> &TV) {
  const char *data = s.data();
  int len = 0;
  
  for (unsigned i = 0, e = s.size(); i != e; ++i, ++len) {
    if (data[len] == 'P' || data[len] == 'Q' || data[len] == 'U')
      continue;
    
    switch (data[len]) {
      case 'c':
      case 's':
      case 'i':
      case 'l':
      case 'h':
      case 'f':
        break;
      default:
        throw TGError(r->getLoc(),
                      "Unexpected letter: " + std::string(data + len, 1));
        break;
    }
    TV.push_back(StringRef(data, len + 1));
    data += len + 1;
    len = -1;
  }
}

static const char Widen(const char t) {
  switch (t) {
    case 'c':
      return 's';
    case 's':
      return 'i';
    case 'i':
      return 'l';
    default: throw "unhandled type in widen!";
  }
  return '\0';
}

static std::string TypeString(const char mod, StringRef typestr) {
  unsigned off = 0;
  
  bool quad = false;
  bool poly = false;
  bool usgn = false;
  bool scal = false;
  bool cnst = false;
  bool pntr = false;
  
  // remember quad.
  if (typestr[off] == 'Q') {
    quad = true;
    ++off;
  }
    
  // remember poly.
  if (typestr[off] == 'P') {
    poly = true;
    ++off;
  }
  
  // remember unsigned.
  if (typestr[off] == 'U') {
    usgn = true;
    ++off;
  }
  
  // base type to get the type string for.
  char type = typestr[off];
  
  // Based on the modifying character, change the type and width if necessary.
  switch (mod) {
    case 'v':
      type = 'v';
      scal = true;
      usgn = false;
      break;
    case 't':
      if (poly) {
        poly = false;
        usgn = true;
      }
      break;
    case 'x':
      usgn = true;
      if (type == 'f')
        type = 'i';
      break;
    case 'f':
      type = 'f';
      break;
    case 'w':
      type = Widen(type);
      quad = true;
      break;
    case 'n':
      type = Widen(type);
      break;
    case 'i':
      type = 'i';
      scal = true;
      usgn = false;
      break;
    case 'l':
      type = 'l';
      scal = true;
      usgn = true;
      break;
    case 's':
      scal = true;
      break;
    case 'k':
      quad = true;
      break;
    case 'c':
      cnst = true;
    case 'p':
      pntr = true;
      scal = true;
      break;
    default:
      break;
  }
  
  SmallString<128> s;
  
  if (usgn)
    s.push_back('u');
  
  switch (type) {
    case 'c':
      s += poly ? "poly8" : "int8";
      if (scal)
        break;
      s += quad ? "x16" : "x8";
      break;
    case 's':
      s += poly ? "poly16" : "int16";
      if (scal)
        break;
      s += quad ? "x8" : "x4";
      break;
    case 'i':
      s += "int32";
      if (scal)
        break;
      s += quad ? "x4" : "x2";
      break;
    case 'l':
      s += "int64";
      if (scal)
        break;
      s += quad ? "x2" : "x1";
      break;
    case 'h':
      s += "float16";
      if (scal)
        break;
      s += quad ? "x8" : "x4";
      break;
    case 'f':
      s += "float32";
      if (scal)
        break;
      s += quad ? "x4" : "x2";
      break;
    case 'v':
      s += "void";
      break;
    default:
      throw "unhandled type!";
      break;
  }

  if (mod == '2')
    s += "x2";
  if (mod == '3')
    s += "x3";
  if (mod == '4')
    s += "x4";
  
  // Append _t, finishing the type string typedef type.
  s += "_t";
  
  if (cnst)
    s += " const";
  
  if (pntr)
    s += " *";
  
  return s.str();
}

// Turn "vst2_lane" into "vst2q_lane_f32", etc.
static std::string MangleName(const std::string &name, StringRef typestr) {
  return "";
}

// 
static std::string GenArgs(const std::string &proto, StringRef typestr) {
  return "";
}

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
  
  std::vector<Record*> RV = Records.getAllDerivedDefinitions("Inst");
  
  // Initialize Type Map
  
  // Unique the return+pattern types, and assign them.
  for (unsigned i = 0, e = RV.size(); i != e; ++i) {
    Record *R = RV[i];
    std::string name = LowercaseString(R->getName());
    std::string Proto = R->getValueAsString("Prototype");
    std::string Types = R->getValueAsString("Types");
    
    SmallVector<StringRef, 16> TypeVec;
    ParseTypes(R, Types, TypeVec);
    
    for (unsigned ti = 0, te = TypeVec.size(); ti != te; ++ti) {
      assert(!Proto.empty() && "");
      
      SmallString<128> Prototype;
      Prototype += TypeString(Proto[0], TypeVec[ti]);
      Prototype += " ";
      Prototype += MangleName(name, TypeVec[ti]);
      Prototype += GenArgs(Proto, TypeVec[ti]);
      
      OS << Prototype << ";\n";
      
      // gen definition
    
        // if (opcode)
      
          // gen opstring
      
        // gen builtin (args)
    }
    OS << "\n";
  }

  // TODO: 
  // Unique the return+pattern types, and assign them to each record
  // Emit a #define for each unique "type" of intrinsic declaring all variants.
  // Emit a #define for each intrinsic mapping it to a particular type.
  
  OS << "\n#endif /* __ARM_NEON_H */\n";
}
