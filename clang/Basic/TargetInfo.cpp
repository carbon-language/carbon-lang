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
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/Builtins.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include <set>
using namespace clang;

void TargetInfoImpl::ANCHOR() {} // out-of-line virtual method for class.


//===----------------------------------------------------------------------===//
// FIXME: These are temporary hacks, they should revector into the
// TargetInfoImpl.

void TargetInfo::getFloatInfo(uint64_t &Size, unsigned &Align,
                              const llvm::fltSemantics *&Format) {
  Align = 32;  // FIXME: implement correctly.
  Size = 32;
  Format = &llvm::APFloat::IEEEsingle;
}
void TargetInfo::getDoubleInfo(uint64_t &Size, unsigned &Align,
                               const llvm::fltSemantics *&Format) {
  Size = 64; // FIXME: implement correctly.
  Align = 32;
  Format = &llvm::APFloat::IEEEdouble;
}
void TargetInfo::getLongDoubleInfo(uint64_t &Size, unsigned &Align,
                                   const llvm::fltSemantics *&Format) {
  Size = Align = 64;  // FIXME: implement correctly.
  Format = &llvm::APFloat::IEEEdouble;
  //Size = 80; Align = 32;  // FIXME: implement correctly.
  //Format = &llvm::APFloat::x87DoubleExtended;
}


//===----------------------------------------------------------------------===//

TargetInfo::~TargetInfo() {
  delete Target;
}

const char* TargetInfo::getTargetTriple() const {
  return Target->getTargetTriple();
}

const char *TargetInfo::getTargetPrefix() const {
 return Target->getTargetPrefix();
}

/// getTargetDefines - Appends the target-specific #define values for this
/// target set to the specified buffer.
void TargetInfo::getTargetDefines(std::vector<char> &Buffer) {
  Target->getTargetDefines(Buffer);
}

/// ComputeWCharWidth - Determine the width of the wchar_t type for the primary
/// target, diagnosing whether this is non-portable across the secondary
/// targets.
void TargetInfo::ComputeWCharInfo() {
  Target->getWCharInfo(WCharWidth, WCharAlign);
}


/// getTargetBuiltins - Return information about target-specific builtins for
/// the current primary target, and info about which builtins are non-portable
/// across the current set of primary and secondary targets.
void TargetInfo::getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {
  // Get info about what actual builtins we will expose.
  Target->getTargetBuiltins(Records, NumRecords);
}

/// getVAListDeclaration - Return the declaration to use for
/// __builtin_va_list, which is target-specific.
const char *TargetInfo::getVAListDeclaration() const {
  return Target->getVAListDeclaration();
}

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
  
  Target->getGCCRegNames(Names, NumNames);
  
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
  const TargetInfoImpl::GCCRegAlias *Aliases;
  unsigned NumAliases;
  
  Target->getGCCRegAliases(Aliases, NumAliases);
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

const char *TargetInfo::getNormalizedGCCRegisterName(const char *Name) const
{
  assert(isValidGCCRegisterName(Name) && "Invalid register passed in");
  
  removeGCCRegisterPrefix(Name);
    
  const char * const *Names;
  unsigned NumNames;

  Target->getGCCRegNames(Names, NumNames);

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
  const TargetInfoImpl::GCCRegAlias *Aliases;
  unsigned NumAliases;
  
  Target->getGCCRegAliases(Aliases, NumAliases);
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
      if (!Target->validateAsmConstraint(*Name, info)) {
        // FIXME: This assert is in place temporarily 
        // so we can add more constraints as we hit it.
        // Eventually, an unknown constraint should just be treated as 'g'.
        assert(0 && "Unknown output constraint type!");
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
                                         ConstraintInfo &info) const
{
  while (*Name) {
    switch (*Name) {
    default:
      // Check if we have a matching constraint
      if (*Name >= '0' && *Name <= '9') {
        unsigned i = *Name - '0';
        
        // Check if matching constraint is out of bounds.
        if (i >= NumOutputs)
          return false;
      } else if (!Target->validateAsmConstraint(*Name, info)) {
        // FIXME: This assert is in place temporarily 
        // so we can add more constraints as we hit it.
        // Eventually, an unknown constraint should just be treated as 'g'.
        assert(0 && "Unknown input constraint type!");
      }        
    case '%': // commutative
      // FIXME: Fail if % is used with the last operand.
      break;
    case 'i': // immediate integer.
    case 'I':
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

std::string TargetInfo::convertConstraint(const char Constraint) const {
  return Target->convertConstraint(Constraint);
}

const char *TargetInfo::getClobbers() const {
  return Target->getClobbers();
}


