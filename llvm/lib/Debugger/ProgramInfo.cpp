//===-- ProgramInfo.cpp - Compute and cache info about a program ----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file implements the ProgramInfo and related classes, by sorting through
// the loaded Module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Debugger/ProgramInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Intrinsics.h"
#include "llvm/iOther.h"
#include "llvm/Module.h"
#include "llvm/Debugger/SourceFile.h"
#include "llvm/Debugger/SourceLanguage.h"
#include "Support/FileUtilities.h"
#include "Support/SlowOperationInformer.h"
#include "Support/STLExtras.h"
using namespace llvm;

/// getGlobalVariablesUsing - Return all of the global variables which have the
/// specified value in their initializer somewhere.
static void getGlobalVariablesUsing(Value *V,
                                    std::vector<GlobalVariable*> &Found) {
  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(*I))
      Found.push_back(GV);
    else if (Constant *C = dyn_cast<Constant>(*I))
      getGlobalVariablesUsing(C, Found);
  }
}

/// getStringValue - Turn an LLVM constant pointer that eventually points to a
/// global into a string value.  Return an empty string if we can't do it.
///
static std::string getStringValue(Value *V, unsigned Offset = 0) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    if (GV->hasInitializer() && isa<ConstantArray>(GV->getInitializer())) {
      ConstantArray *Init = cast<ConstantArray>(GV->getInitializer());
      if (Init->getType()->getElementType() == Type::SByteTy ||
          Init->getType()->getElementType() == Type::UByteTy) {
        std::string Result = Init->getAsString();
        if (Offset < Result.size()) {
          // If we are pointing INTO The string, erase the beginning...
          Result.erase(Result.begin(), Result.begin()+Offset);

          // Take off the null terminator, and any string fragments after it.
          std::string::size_type NullPos = Result.find_first_of((char)0);
          if (NullPos != std::string::npos)
            Result.erase(Result.begin()+NullPos, Result.end());
          return Result;
        }
      }
    }
  } else if (Constant *C = dyn_cast<Constant>(V)) {
    if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C))
      return getStringValue(CPR->getValue(), Offset);
    else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        // Turn a gep into the specified offset.
        if (CE->getNumOperands() == 3 &&
            cast<Constant>(CE->getOperand(1))->isNullValue() &&
            isa<ConstantInt>(CE->getOperand(2))) {
          return getStringValue(CE->getOperand(0),
                   Offset+cast<ConstantInt>(CE->getOperand(2))->getRawValue());
        }
      }
    }
  }
  return "";
}

/// getNextStopPoint - Follow the def-use chains of the specified LLVM value,
/// traversing the use chains until we get to a stoppoint.  When we do, return
/// the source location of the stoppoint.  If we don't find a stoppoint, return
/// null.
static const GlobalVariable *getNextStopPoint(const Value *V, unsigned &LineNo,
                                              unsigned &ColNo) {
  // The use-def chains can fork.  As such, we pick the lowest numbered one we
  // find.
  const GlobalVariable *LastDesc = 0;
  unsigned LastLineNo = ~0;
  unsigned LastColNo = ~0;

  for (Value::use_const_iterator UI = V->use_begin(), E = V->use_end();
       UI != E; ++UI) {
    bool ShouldRecurse = true;
    if (cast<Instruction>(*UI)->getOpcode() == Instruction::PHI) {
      // Infinite loops == bad, ignore PHI nodes.
      ShouldRecurse = false;
    } else if (const CallInst *CI = dyn_cast<CallInst>(*UI)) {
      // If we found a stop point, check to see if it is earlier than what we
      // already have.  If so, remember it.
      if (const Function *F = CI->getCalledFunction())
        if (F->getIntrinsicID() == Intrinsic::dbg_stoppoint) {
          unsigned CurLineNo = ~0, CurColNo = ~0;
          const GlobalVariable *CurDesc = 0;
          if (const ConstantInt *C = dyn_cast<ConstantInt>(CI->getOperand(2)))
            CurLineNo = C->getRawValue();
          if (const ConstantInt *C = dyn_cast<ConstantInt>(CI->getOperand(3)))
            CurColNo = C->getRawValue();
          const Value *Op = CI->getOperand(4);
          if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Op))
            Op = CPR->getValue();
          
          if ((CurDesc = dyn_cast<GlobalVariable>(Op)) &&
              (LineNo < LastLineNo ||
               (LineNo == LastLineNo && ColNo < LastColNo))) {
            LastDesc = CurDesc;
            LastLineNo = CurLineNo;
            LastColNo = CurColNo;            
          }
          ShouldRecurse = false;
        }

    }

    // If this is not a phi node or a stopping point, recursively scan the users
    // of this instruction to skip over region.begin's and the like.
    if (ShouldRecurse) {
      unsigned CurLineNo, CurColNo;
      if (const GlobalVariable *GV = getNextStopPoint(*UI, CurLineNo,CurColNo)){
        if (LineNo < LastLineNo || (LineNo == LastLineNo && ColNo < LastColNo)){
          LastDesc = GV;
          LastLineNo = CurLineNo;
          LastColNo = CurColNo;            
        }
      }
    }
  }
  
  if (LastDesc) {
    LineNo = LastLineNo != ~0U ? LastLineNo : 0;
    ColNo  = LastColNo  != ~0U ? LastColNo : 0;
  }
  return LastDesc;
}


//===----------------------------------------------------------------------===//
// SourceFileInfo implementation
//

SourceFileInfo::SourceFileInfo(const GlobalVariable *Desc,
                               const SourceLanguage &Lang)
  : Language(&Lang), Descriptor(Desc) {
  Version = 0;
  SourceText = 0;

  if (Desc && Desc->hasInitializer())
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(Desc->getInitializer()))
      if (CS->getNumOperands() > 4) {
        if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(CS->getOperand(1)))
          Version = CUI->getValue();
        
        BaseName  = getStringValue(CS->getOperand(3));
        Directory = getStringValue(CS->getOperand(4));
      }
}

SourceFileInfo::~SourceFileInfo() {
  delete SourceText;
}

SourceFile &SourceFileInfo::getSourceText() const {
  // FIXME: this should take into account the source search directories!
  if (SourceText == 0)  // Read the file in if we haven't already.
    if (!Directory.empty() && FileOpenable(Directory+"/"+BaseName))
      SourceText = new SourceFile(Directory+"/"+BaseName, Descriptor);
    else
      SourceText = new SourceFile(BaseName, Descriptor);
  return *SourceText;
}


//===----------------------------------------------------------------------===//
// SourceFunctionInfo implementation
//
SourceFunctionInfo::SourceFunctionInfo(ProgramInfo &PI,
                                       const GlobalVariable *Desc)
  : Descriptor(Desc) {
  LineNo = ColNo = 0;
  if (Desc && Desc->hasInitializer())
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(Desc->getInitializer()))
      if (CS->getNumOperands() > 2) {
        // Entry #1 is the file descriptor.
        if (const ConstantPointerRef *CPR =
            dyn_cast<ConstantPointerRef>(CS->getOperand(1)))
          if (const GlobalVariable *GV =
              dyn_cast<GlobalVariable>(CPR->getValue()))
            SourceFile = &PI.getSourceFile(GV);

        // Entry #2 is the function name.
        Name = getStringValue(CS->getOperand(2));
      }
}

/// getSourceLocation - This method returns the location of the first stopping
/// point in the function.
void SourceFunctionInfo::getSourceLocation(unsigned &RetLineNo,
                                           unsigned &RetColNo) const {
  // If we haven't computed this yet...
  if (!LineNo) {
    // Look at all of the users of the function descriptor, looking for calls to
    // %llvm.dbg.func.start.
    for (Value::use_const_iterator UI = Descriptor->use_begin(),
           E = Descriptor->use_end(); UI != E; ++UI)
      if (const CallInst *CI = dyn_cast<CallInst>(*UI))
        if (const Function *F = CI->getCalledFunction())
          if (F->getIntrinsicID() == Intrinsic::dbg_func_start) {
            // We found the start of the function.  Check to see if there are
            // any stop points on the use-list of the function start.
            const GlobalVariable *SD = getNextStopPoint(CI, LineNo, ColNo);
            if (SD) {             // We found the first stop point!
              // This is just a sanity check.
              if (getSourceFile().getDescriptor() != SD)
                std::cout << "WARNING: first line of function is not in the"
                  " file that the function descriptor claims it is in.\n";
              break;
            }
          }
  }
  RetLineNo = LineNo; RetColNo = ColNo;
}

//===----------------------------------------------------------------------===//
// ProgramInfo implementation
//

ProgramInfo::ProgramInfo(Module *m) : M(m) {
  assert(M && "Cannot create program information with a null module!");
  ProgramTimeStamp = getFileTimestamp(M->getModuleIdentifier());

  SourceFilesIsComplete = false;
  SourceFunctionsIsComplete = false;
}

ProgramInfo::~ProgramInfo() {
  // Delete cached information about source program objects...
  for (std::map<const GlobalVariable*, SourceFileInfo*>::iterator
         I = SourceFiles.begin(), E = SourceFiles.end(); I != E; ++I)
    delete I->second;
  for (std::map<const GlobalVariable*, SourceFunctionInfo*>::iterator
         I = SourceFunctions.begin(), E = SourceFunctions.end(); I != E; ++I)
    delete I->second;

  // Delete the source language caches.
  for (unsigned i = 0, e = LanguageCaches.size(); i != e; ++i)
    delete LanguageCaches[i].second;
}


//===----------------------------------------------------------------------===//
// SourceFileInfo tracking...
//

/// getSourceFile - Return source file information for the specified source file
/// descriptor object, adding it to the collection as needed.  This method
/// always succeeds (is unambiguous), and is always efficient.
///
const SourceFileInfo &
ProgramInfo::getSourceFile(const GlobalVariable *Desc) {
  SourceFileInfo *&Result = SourceFiles[Desc];
  if (Result) return *Result;

  // Figure out what language this source file comes from...
  unsigned LangID = 0;   // Zero is unknown language
  if (Desc && Desc->hasInitializer())
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(Desc->getInitializer()))
      if (CS->getNumOperands() > 2)
        if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(CS->getOperand(2)))
          LangID = CUI->getValue();

  const SourceLanguage &Lang = SourceLanguage::get(LangID);
  SourceFileInfo *New = Lang.createSourceFileInfo(Desc, *this);

  // FIXME: this should check to see if there is already a Filename/WorkingDir
  // pair that matches this one.  If so, we shouldn't create the duplicate!
  //
  SourceFileIndex.insert(std::make_pair(New->getBaseName(), New));
  return *(Result = New);
}


/// getSourceFiles - Index all of the source files in the program and return
/// a mapping of it.  This information is lazily computed the first time
/// that it is requested.  Since this information can take a long time to
/// compute, the user is given a chance to cancel it.  If this occurs, an
/// exception is thrown.
const std::map<const GlobalVariable*, SourceFileInfo*> &
ProgramInfo::getSourceFiles(bool RequiresCompleteMap) {
  // If we have a fully populated map, or if the client doesn't need one, just
  // return what we have.
  if (SourceFilesIsComplete || !RequiresCompleteMap)
    return SourceFiles;

  // Ok, all of the source file descriptors (compile_unit in dwarf terms),
  // should be on the use list of the llvm.dbg.translation_units global.
  //
  GlobalVariable *Units =
    M->getGlobalVariable("llvm.dbg.translation_units",
                         StructType::get(std::vector<const Type*>()));
  if (Units == 0)
    throw "Program contains no debugging information!";

  std::vector<GlobalVariable*> TranslationUnits;
  getGlobalVariablesUsing(Units, TranslationUnits);

  SlowOperationInformer SOI("building source files index");

  // Loop over all of the translation units found, building the SourceFiles
  // mapping.
  for (unsigned i = 0, e = TranslationUnits.size(); i != e; ++i) {
    getSourceFile(TranslationUnits[i]);
    SOI.progress(i+1, e);
  }

  // Ok, if we got this far, then we indexed the whole program.
  SourceFilesIsComplete = true;
  return SourceFiles;
}

/// getSourceFile - Look up the file with the specified name.  If there is
/// more than one match for the specified filename, prompt the user to pick
/// one.  If there is no source file that matches the specified name, throw
/// an exception indicating that we can't find the file.  Otherwise, return
/// the file information for that file.
const SourceFileInfo &ProgramInfo::getSourceFile(const std::string &Filename) {
  std::multimap<std::string, SourceFileInfo*>::const_iterator Start, End;
  getSourceFiles();
  tie(Start, End) = SourceFileIndex.equal_range(Filename);
  
  if (Start == End) throw "Could not find source file '" + Filename + "'!";
  const SourceFileInfo &SFI = *Start->second;
  ++Start;
  if (Start == End) return SFI;

  throw "FIXME: Multiple source files with the same name not implemented!";
}


//===----------------------------------------------------------------------===//
// SourceFunctionInfo tracking...
//


/// getFunction - Return function information for the specified function
/// descriptor object, adding it to the collection as needed.  This method
/// always succeeds (is unambiguous), and is always efficient.
///
const SourceFunctionInfo &
ProgramInfo::getFunction(const GlobalVariable *Desc) {
  SourceFunctionInfo *&Result = SourceFunctions[Desc];
  if (Result) return *Result;

  // Figure out what language this function comes from...
  const GlobalVariable *SourceFileDesc = 0;
  if (Desc && Desc->hasInitializer())
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(Desc->getInitializer()))
      if (CS->getNumOperands() > 0)
        if (const ConstantPointerRef *CPR =
            dyn_cast<ConstantPointerRef>(CS->getOperand(1)))
          SourceFileDesc = dyn_cast<GlobalVariable>(CPR->getValue());

  const SourceLanguage &Lang = getSourceFile(SourceFileDesc).getLanguage();
  return *(Result = Lang.createSourceFunctionInfo(Desc, *this));
}


// getSourceFunctions - Index all of the functions in the program and return
// them.  This information is lazily computed the first time that it is
// requested.  Since this information can take a long time to compute, the user
// is given a chance to cancel it.  If this occurs, an exception is thrown.
const std::map<const GlobalVariable*, SourceFunctionInfo*> &
ProgramInfo::getSourceFunctions(bool RequiresCompleteMap) {
  if (SourceFunctionsIsComplete || !RequiresCompleteMap)
    return SourceFunctions;

  // Ok, all of the source function descriptors (subprogram in dwarf terms),
  // should be on the use list of the llvm.dbg.translation_units global.
  //
  GlobalVariable *Units =
    M->getGlobalVariable("llvm.dbg.globals",
                         StructType::get(std::vector<const Type*>()));
  if (Units == 0)
    throw "Program contains no debugging information!";

  std::vector<GlobalVariable*> Functions;
  getGlobalVariablesUsing(Units, Functions);

  SlowOperationInformer SOI("building functions index");

  // Loop over all of the functions found, building the SourceFunctions mapping.
  for (unsigned i = 0, e = Functions.size(); i != e; ++i) {
    getFunction(Functions[i]);
    SOI.progress(i+1, e);
  }

  // Ok, if we got this far, then we indexed the whole program.
  SourceFunctionsIsComplete = true;
  return SourceFunctions;
}
