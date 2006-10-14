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
#include <map>
using namespace llvm;
using namespace clang;

void TargetInfoImpl::ANCHOR() {} // out-of-line virtual method for class.


/// DiagnoseNonPortability - When a use of a non-portable target feature is
/// used, this method emits the diagnostic and marks the translation unit as
/// non-portable.
void TargetInfo::DiagnoseNonPortability(SourceLocation Loc, unsigned DiagKind) {
  NonPortable = true;
  if (Diag) Diag->Report(Loc, DiagKind);
}

/// GetTargetDefineMap - Get the set of target #defines in an associative
/// collection for easy lookup.
static void GetTargetDefineMap(const TargetInfoImpl *Target,
                               std::map<std::string, std::string> &Map) {
  std::vector<std::string> PrimaryDefines;
  Target->getTargetDefines(PrimaryDefines);

  while (!PrimaryDefines.empty()) {
    const char *Str = PrimaryDefines.back().c_str();
    if (const char *Equal = strchr(Str, '=')) {
      // Split at the '='.
      Map.insert(std::make_pair(std::string(Str, Equal),
                                std::string(Equal+1,
                                            Str+PrimaryDefines.back().size())));
    } else {
      // Remember "macroname=1".
      Map.insert(std::make_pair(PrimaryDefines.back(), std::string("1")));
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
  std::map<std::string, std::string> PrimaryDefines;
  GetTargetDefineMap(PrimaryTarget, PrimaryDefines);
  
  // Get the sets of secondary #defines.
  std::vector<std::map<std::string, std::string> > SecondaryDefines;
  SecondaryDefines.resize(SecondaryTargets.size());
  for (unsigned i = 0, e = SecondaryTargets.size(); i != e; ++i)
    GetTargetDefineMap(SecondaryTargets[i], SecondaryDefines[i]);

  // Loop over all defines in the primary target, processing them until we run
  // out.
  while (!PrimaryDefines.empty()) {
    std::string DefineName  = PrimaryDefines.begin()->first;
    std::string DefineValue = PrimaryDefines.begin()->second;
    PrimaryDefines.erase(PrimaryDefines.begin());
    
    // Check to see whether all secondary targets have this #define and whether
    // it is to the same value.  Remember if not, but remove the #define from
    // their collection in any case if they have it.
    bool isPortable = true;
    
    for (unsigned i = 0, e = SecondaryDefines.size(); i != e; ++i) {
      std::map<std::string, std::string>::iterator I = 
        SecondaryDefines[i].find(DefineName);
      if (I == SecondaryDefines[i].end()) {
        // Secondary target doesn't have this #define.
        isPortable = false;
      } else {
        // Secondary target has this define, remember if it disagrees.
        if (isPortable)
          isPortable = I->second == DefineValue;
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
  for (unsigned i = 0, e = SecondaryDefines.size(); i != e; ++i) {
    std::map<std::string, std::string> &Defs = SecondaryDefines[i];
    while (!Defs.empty()) {
      const std::string &DefName = Defs.begin()->first;
      
      // Insert "#define_other_target defname".
      const char *Command = "#define_other_target ";
      Buffer.insert(Buffer.end(), Command, Command+strlen(Command));
      Buffer.insert(Buffer.end(), DefName.begin(), DefName.end());
      Buffer.push_back('\n');
      
      // If any other secondary targets have this same define, remove it from
      // them to avoid duplicate #define_other_target directives.
      for (unsigned j = i+1; j != e; ++j)
        SecondaryDefines[j].erase(DefName);
      
      Defs.erase(Defs.begin());
    }
  }
}

/// ComputeWCharWidth - Determine the width of the wchar_t type for the primary
/// target, diagnosing whether this is non-portable across the secondary
/// targets.
void TargetInfo::ComputeWCharWidth(SourceLocation Loc) {
  WCharWidth = PrimaryTarget->getWCharWidth();
  
  // Check whether this is portable across the secondary targets if the T-U is
  // portable so far.
  for (unsigned i = 0, e = SecondaryTargets.size(); i != e; ++i)
    if (SecondaryTargets[i]->getWCharWidth() != WCharWidth)
      return DiagnoseNonPortability(Loc, diag::port_wchar_t);
}

