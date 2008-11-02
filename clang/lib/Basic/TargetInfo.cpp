//===--- TargetInfo.cpp - Information about Target machine ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the TargetInfo and TargetInfoImpl interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdlib>
using namespace clang;

// TargetInfo Constructor.
TargetInfo::TargetInfo(const std::string &T) : Triple(T) {
  // Set defaults.  Defaults are set for a 32-bit RISC platform,
  // like PPC or SPARC.
  // These should be overridden by concrete targets as needed.
  CharIsSigned = true;
  PointerWidth = PointerAlign = 32;
  WCharWidth = WCharAlign = 32;
  IntWidth = IntAlign = 32;
  LongWidth = LongAlign = 32;
  LongLongWidth = LongLongAlign = 64;
  FloatWidth = 32;
  FloatAlign = 32;
  DoubleWidth = 64;
  DoubleAlign = 64;
  LongDoubleWidth = 64;
  LongDoubleAlign = 64;
  SizeType = UnsignedLong;
  PtrDiffType = SignedLong;
  IntMaxType = SignedLongLong;
  UIntMaxType = UnsignedLongLong;
  WCharType = SignedInt;
  FloatFormat = &llvm::APFloat::IEEEsingle;
  DoubleFormat = &llvm::APFloat::IEEEdouble;
  LongDoubleFormat = &llvm::APFloat::IEEEdouble;
  DescriptionString = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                      "i64:64:64-f32:32:32-f64:64:64";
  UserLabelPrefix = "_";
}

// Out of line virtual dtor for TargetInfo.
TargetInfo::~TargetInfo() {}

//===----------------------------------------------------------------------===//


static void removeGCCRegisterPrefix(const char *&Name) {
  if (Name[0] == '%' || Name[0] == '#')
    Name++;
}

/// isValidGCCRegisterName - Returns whether the passed in string
/// is a valid register name according to GCC. This is used by Sema for
/// inline asm statements.
bool TargetInfo::isValidGCCRegisterName(const char *Name) const {
  const char * const *Names;
  unsigned NumNames;
  
  // Get rid of any register prefix.
  removeGCCRegisterPrefix(Name);

  
  if (strcmp(Name, "memory") == 0 ||
      strcmp(Name, "cc") == 0)
    return true;
  
  getGCCRegNames(Names, NumNames);
  
  // If we have a number it maps to an entry in the register name array.
  if (isdigit(Name[0])) {
    char *End;
    int n = (int)strtol(Name, &End, 0);
    if (*End == 0)
      return n >= 0 && (unsigned)n < NumNames;
  }

  // Check register names.
  for (unsigned i = 0; i < NumNames; i++) {
    if (strcmp(Name, Names[i]) == 0)
      return true;
  }
  
  // Now check aliases.
  const GCCRegAlias *Aliases;
  unsigned NumAliases;
  
  getGCCRegAliases(Aliases, NumAliases);
  for (unsigned i = 0; i < NumAliases; i++) {
    for (unsigned j = 0 ; j < llvm::array_lengthof(Aliases[i].Aliases); j++) {
      if (!Aliases[i].Aliases[j])
        break;
      if (strcmp(Aliases[i].Aliases[j], Name) == 0)
        return true;
    }
  }
  
  return false;
}

const char *TargetInfo::getNormalizedGCCRegisterName(const char *Name) const {
  assert(isValidGCCRegisterName(Name) && "Invalid register passed in");
  
  removeGCCRegisterPrefix(Name);
    
  const char * const *Names;
  unsigned NumNames;

  getGCCRegNames(Names, NumNames);

  // First, check if we have a number.
  if (isdigit(Name[0])) {
    char *End;
    int n = (int)strtol(Name, &End, 0);
    if (*End == 0) {
      assert(n >= 0 && (unsigned)n < NumNames && 
             "Out of bounds register number!");
      return Names[n];
    }
  }
  
  // Now check aliases.
  const GCCRegAlias *Aliases;
  unsigned NumAliases;
  
  getGCCRegAliases(Aliases, NumAliases);
  for (unsigned i = 0; i < NumAliases; i++) {
    for (unsigned j = 0 ; j < llvm::array_lengthof(Aliases[i].Aliases); j++) {
      if (!Aliases[i].Aliases[j])
        break;
      if (strcmp(Aliases[i].Aliases[j], Name) == 0)
        return Aliases[i].Register;
    }
  }
  
  return Name;
}

bool TargetInfo::validateOutputConstraint(const char *Name, 
                                          ConstraintInfo &info) const
{
  // An output constraint must start with '=' or '+'
  if (*Name != '=' && *Name != '+')
    return false;

  if (*Name == '+')
    info = CI_ReadWrite;
  else
    info = CI_None;

  Name++;
  while (*Name) {
    switch (*Name) {
    default:
      if (!validateAsmConstraint(*Name, info)) {
        // FIXME: We temporarily return false
        // so we can add more constraints as we hit it.
        // Eventually, an unknown constraint should just be treated as 'g'.
        return false;
      }
    case '&': // early clobber.
      break;
    case 'r': // general register.
      info = (ConstraintInfo)(info|CI_AllowsRegister);
      break;
    case 'm': // memory operand.
      info = (ConstraintInfo)(info|CI_AllowsMemory);
      break;
    case 'g': // general register, memory operand or immediate integer.
      info = (ConstraintInfo)(info|CI_AllowsMemory|CI_AllowsRegister);
      break;
    }
    
    Name++;
  }
  
  return true;
}

bool TargetInfo::validateInputConstraint(const char *Name,
                                         unsigned NumOutputs,
                                         ConstraintInfo &info) const {
  while (*Name) {
    switch (*Name) {
    default:
      // Check if we have a matching constraint
      if (*Name >= '0' && *Name <= '9') {
        unsigned i = *Name - '0';
        
        // Check if matching constraint is out of bounds.
        if (i >= NumOutputs)
          return false;
      } else if (!validateAsmConstraint(*Name, info)) {
        // FIXME: This error return is in place temporarily so we can
        // add more constraints as we hit it.  Eventually, an unknown
        // constraint should just be treated as 'g'.
        return false;
      }        
    case '%': // commutative
      // FIXME: Fail if % is used with the last operand.
      break;
    case 'i': // immediate integer.
    case 'I':
    case 'n': // immediate integer with a known value.
      break;
    case 'r': // general register.
      info = (ConstraintInfo)(info|CI_AllowsRegister);
      break;
    case 'm': // memory operand.
      info = (ConstraintInfo)(info|CI_AllowsMemory);
      break;
    case 'g': // general register, memory operand or immediate integer.
      info = (ConstraintInfo)(info|CI_AllowsMemory|CI_AllowsRegister);
      break;
    }
    
    Name++;
  }
  
  return true;
}
