//===--- TargetInfo.cpp - Information about Target machine ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the TargetInfo and TargetInfoImpl interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/Builtins.h"
#include "llvm/ADT/StringMap.h"
#include <set>
using namespace clang;

void TargetInfoImpl::ANCHOR() {} // out-of-line virtual method for class.


/// DiagnoseNonPortability - When a use of a non-portable target feature is
/// used, this method emits the diagnostic and marks the translation unit as
/// non-portable.
void TargetInfo::DiagnoseNonPortability(SourceLocation Loc, unsigned DiagKind) {
  NonPortable = true;
  if (Diag && Loc.isValid()) Diag->Report(Loc, DiagKind);
}

/// GetTargetDefineMap - Get the set of target #defines in an associative
/// collection for easy lookup.
static void GetTargetDefineMap(const TargetInfoImpl *Target,
                               llvm::StringMap<std::string> &Map) {
  std::vector<std::string> PrimaryDefines;
  Target->getTargetDefines(PrimaryDefines);

  while (!PrimaryDefines.empty()) {
    std::string &PrimDefineStr = PrimaryDefines.back();
    const char *Str    = PrimDefineStr.c_str();
    const char *StrEnd = Str+PrimDefineStr.size();
    
    if (const char *Equal = strchr(Str, '=')) {
      // Split at the '='.
      
      std::string &Entry = Map.GetOrCreateValue(Str, Equal).getValue();
      Entry = std::string(Equal+1, StrEnd);
    } else {
      // Remember "macroname=1".
      std::string &Entry = Map.GetOrCreateValue(Str, StrEnd).getValue();
      Entry = "1";
    }
    PrimaryDefines.pop_back();
  }
}

/// getTargetDefines - Appends the target-specific #define values for this
/// target set to the specified buffer.
void TargetInfo::getTargetDefines(std::vector<char> &Buffer) {
  // This is tricky in the face of secondary targets.  Specifically, 
  // target-specific #defines that are present and identical across all
  // secondary targets are turned into #defines, #defines that are present in
  // the primary target but are missing or different in the secondary targets
  // are turned into #define_target, and #defines that are not defined in the
  // primary, but are defined in a secondary are turned into
  // #define_other_target.  This allows the preprocessor to correctly track uses
  // of target-specific macros.
  
  // Get the set of primary #defines.
  llvm::StringMap<std::string> PrimaryDefines;
  GetTargetDefineMap(PrimaryTarget, PrimaryDefines);
  
  // If we have no secondary targets, be a bit more efficient.
  if (SecondaryTargets.empty()) {
    for (llvm::StringMap<std::string>::iterator I = 
           PrimaryDefines.begin(), E = PrimaryDefines.end(); I != E; ++I) {
      // If this define is non-portable, turn it into #define_target, otherwise
      // just use #define.
      const char *Command = "#define ";
      Buffer.insert(Buffer.end(), Command, Command+strlen(Command));
      
      // Insert "defname defvalue\n".
      const char *KeyStart = I->getKeyData();
      const char *KeyEnd = KeyStart + I->getKeyLength();
      
      Buffer.insert(Buffer.end(), KeyStart, KeyEnd);
      Buffer.push_back(' ');
      Buffer.insert(Buffer.end(), I->getValue().begin(), I->getValue().end());
      Buffer.push_back('\n');
    }
    return;
  }
  
  // Get the sets of secondary #defines.
  llvm::StringMap<std::string> *SecondaryDefines
    = new llvm::StringMap<std::string>[SecondaryTargets.size()];
  for (unsigned i = 0, e = SecondaryTargets.size(); i != e; ++i)
    GetTargetDefineMap(SecondaryTargets[i], SecondaryDefines[i]);

  // Loop over all defines in the primary target, processing them until we run
  // out.
  for (llvm::StringMap<std::string>::iterator PDI = 
         PrimaryDefines.begin(), E = PrimaryDefines.end(); PDI != E; ++PDI) {
    std::string DefineName(PDI->getKeyData(),
                           PDI->getKeyData() + PDI->getKeyLength());
    std::string DefineValue = PDI->getValue();
    
    // Check to see whether all secondary targets have this #define and whether
    // it is to the same value.  Remember if not, but remove the #define from
    // their collection in any case if they have it.
    bool isPortable = true;
    
    for (unsigned i = 0, e = SecondaryTargets.size(); i != e; ++i) {
      llvm::StringMap<std::string>::iterator I = 
        SecondaryDefines[i].find(&DefineName[0],
                                 &DefineName[0]+DefineName.size());
      if (I == SecondaryDefines[i].end()) {
        // Secondary target doesn't have this #define.
        isPortable = false;
      } else {
        // Secondary target has this define, remember if it disagrees.
        if (isPortable)
          isPortable = I->getValue() == DefineValue;
        // Remove it from the secondary target unconditionally.
        SecondaryDefines[i].erase(I);
      }
    }
    
    // If this define is non-portable, turn it into #define_target, otherwise
    // just use #define.
    const char *Command = isPortable ? "#define " : "#define_target ";
    Buffer.insert(Buffer.end(), Command, Command+strlen(Command));

    // Insert "defname defvalue\n".
    Buffer.insert(Buffer.end(), DefineName.begin(), DefineName.end());
    Buffer.push_back(' ');
    Buffer.insert(Buffer.end(), DefineValue.begin(), DefineValue.end());
    Buffer.push_back('\n');
  }
  
  // Now that all of the primary target's defines have been handled and removed
  // from the secondary target's define sets, go through the remaining secondary
  // target's #defines and taint them.
  for (unsigned i = 0, e = SecondaryTargets.size(); i != e; ++i) {
    llvm::StringMap<std::string> &Defs = SecondaryDefines[i];
    while (!Defs.empty()) {
      const char *DefStart = Defs.begin()->getKeyData();
      const char *DefEnd = DefStart + Defs.begin()->getKeyLength();
      
      // Insert "#define_other_target defname".
      const char *Command = "#define_other_target ";
      Buffer.insert(Buffer.end(), Command, Command+strlen(Command));
      Buffer.insert(Buffer.end(), DefStart, DefEnd);
      Buffer.push_back('\n');
      
      // If any other secondary targets have this same define, remove it from
      // them to avoid duplicate #define_other_target directives.
      for (unsigned j = i+1; j != e; ++j) {
        llvm::StringMap<std::string>::iterator I =
          SecondaryDefines[j].find(DefStart, DefEnd);
        if (I != SecondaryDefines[j].end())
          SecondaryDefines[j].erase(I);
      }
      Defs.erase(Defs.begin());
    }
  }
  
  delete[] SecondaryDefines;
}

/// ComputeWCharWidth - Determine the width of the wchar_t type for the primary
/// target, diagnosing whether this is non-portable across the secondary
/// targets.
void TargetInfo::ComputeWCharInfo(SourceLocation Loc) {
  PrimaryTarget->getWCharInfo(WCharWidth, WCharAlign);
  
  // Check whether this is portable across the secondary targets if the T-U is
  // portable so far.
  for (unsigned i = 0, e = SecondaryTargets.size(); i != e; ++i) {
    unsigned Width, Align;
    SecondaryTargets[i]->getWCharInfo(Width, Align);
    if (Width != WCharWidth || Align != WCharAlign)
      return DiagnoseNonPortability(Loc, diag::port_wchar_t);
  }
}


/// getTargetBuiltins - Return information about target-specific builtins for
/// the current primary target, and info about which builtins are non-portable
/// across the current set of primary and secondary targets.
void TargetInfo::getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords,
                                   std::vector<const char *> &NPortable) const {
  // Get info about what actual builtins we will expose.
  PrimaryTarget->getTargetBuiltins(Records, NumRecords);
  if (SecondaryTargets.empty()) return;
 
  // Compute the set of non-portable builtins.
  
  // Start by computing a mapping from the primary target's builtins to their
  // info records for efficient lookup.
  llvm::StringMap<const Builtin::Info*> PrimaryRecs;
  for (unsigned i = 0, e = NumRecords; i != e; ++i) {
    const char *BIName = Records[i].Name;
    PrimaryRecs.GetOrCreateValue(BIName, BIName+strlen(BIName)).getValue()
      = Records+i;
  }
  
  for (unsigned i = 0, e = SecondaryTargets.size(); i != e; ++i) {
    // Get the builtins for this secondary target.
    const Builtin::Info *Records2nd;
    unsigned NumRecords2nd;
    SecondaryTargets[i]->getTargetBuiltins(Records2nd, NumRecords2nd);
    
    // Remember all of the secondary builtin names.
    std::set<std::string> BuiltinNames2nd;

    for (unsigned j = 0, e = NumRecords2nd; j != e; ++j) {
      BuiltinNames2nd.insert(Records2nd[j].Name);
      
      // Check to see if the primary target has this builtin.
      llvm::StringMap<const Builtin::Info*>::iterator I =
        PrimaryRecs.find(Records2nd[j].Name,
                         Records2nd[j].Name+strlen(Records2nd[j].Name));
      if (I != PrimaryRecs.end()) {
        const Builtin::Info *PrimBI = I->getValue();
        // If does.  If they are not identical, mark the builtin as being
        // non-portable.
        if (Records2nd[j] != *PrimBI)
          NPortable.push_back(PrimBI->Name);
      } else {
        // The primary target doesn't have this, it is non-portable.
        NPortable.push_back(Records2nd[j].Name);
      }
    }
    
    // Now that we checked all the secondary builtins, check to see if the
    // primary target has any builtins that the secondary one doesn't.  If so,
    // then those are non-portable.
    for (unsigned j = 0, e = NumRecords; j != e; ++j) {
      if (!BuiltinNames2nd.count(Records[j].Name))
        NPortable.push_back(Records[j].Name);
    }
  }
}


