//===-- LTOPartition.cpp - Parition Merged Module --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LTOPartition.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace lto;

// /////////////////////////////////////////////////////////////////////////////
//
//   Implementation of IPOPartition and IPOPartMgr
//
// /////////////////////////////////////////////////////////////////////////////
//
IPOPartition::IPOPartition(Module *M, const char *NameWoExt, IPOFileMgr &FM) :
  Mod(0), Ctx(0), IRFile(0), ObjFile(0), FileNameWoExt(NameWoExt), FileMgr(FM) {
}

IPOFile &IPOPartition::getIRFile() const {
  if (IRFile)
    return *IRFile;
  else {
    std::string FN(FileNameWoExt + ".bc");
    return *(IRFile = FileMgr.createIRFile(FN.c_str()));
  }
}

IPOFile &IPOPartition::getObjFile() const {
  if (ObjFile)
    return *ObjFile;
  else {
    std::string FN(FileNameWoExt + ".o");
    return *(ObjFile = FileMgr.createObjFile(FN.c_str()));
  }
}

bool IPOPartition::saveBitCode() {
  if (!Mod) {
    // The bit-code have already saved in disk.
    return true;
  }

  IPOFile &F = getIRFile();
  if (F.errOccur())
    return false;

  raw_fd_ostream OF(F.getPath().c_str(), F.getLastErrStr(),
                    sys::fs::F_Binary);
  WriteBitcodeToFile(Mod, OF);
  OF.close();

  Mod = 0;
  delete Ctx;
  Ctx = 0;
 
  return !F.errOccur();
}

bool IPOPartition::loadBitCode() {
  if (Mod)
    return true;

  IPOFile &F = getIRFile();
  if (F.errOccur())
    return false;

  Ctx = new LLVMContext;

  error_code &EC = F.getLastErrCode();
  std::string &ErrMsg = F.getLastErrStr();

  OwningPtr<MemoryBuffer> Buf;
  if (error_code ec = MemoryBuffer::getFile(F.getPath(), Buf, -1, false)) {
    EC = ec; 
    ErrMsg += ec.message();
    return false;
  }

  Mod = ParseBitcodeFile(Buf.get(), *Ctx, &ErrMsg);

  return Mod != 0;
}

IPOPartition *IPOPartMgr::createIPOPart(Module *M) {
  std::string PartName;
  raw_string_ostream OS(PartName); 
  OS << "part" << NextPartId++;

  IPOPartition *P = new IPOPartition(M, OS.str().c_str(), FileMgr);
  P->Mod = M;
  IPOParts.push_back(P);
  return P;
}

// ///////////////////////////////////////////////////////////////////////////
//
//      Implementation of IPOFile and IPOFileMgr 
//  
// ///////////////////////////////////////////////////////////////////////////
//
IPOFile::IPOFile(const char *DirName, const char *BaseName, bool KeepFile)
  : Fname(BaseName), Keep(KeepFile) {
  // Concatenate dirname and basename
  StringRef D(DirName);
  SmallVector<char, 64> Path(D.begin(), D.end());
  sys::path::append(Path, Twine(BaseName));
  Fpath = StringRef(Path.data(), Path.size());
}

IPOFileMgr::IPOFileMgr() {
  IRFiles.reserve(20);
  ObjFiles.reserve(20);
  OtherFiles.reserve(8);
  KeepWorkDir = false;
  WorkDirCreated = false;
}

bool IPOFileMgr::createWorkDir(std::string &ErrorInfo) {
  if (WorkDirCreated)
    return true;

  error_code EC;
  if (WorkDir.empty()) {
    // If the workdir is not specified, then create workdir under current
    // directory.
    //
    SmallString<128> D;
    if (sys::fs::current_path(D) != error_code::success()) {
      ErrorInfo += "fail to get current directory";
      return false;
    }
    sys::path::append(D, "llvmipo");
    sys::fs::make_absolute(D);

    SmallVector<char, 64> ResPath;
    EC = sys::fs::createUniqueDirectory(Twine(StringRef(D.data(), D.size())),
                                        ResPath);
    WorkDir = StringRef(ResPath.data(), ResPath.size());
  } else {
    bool Exist;
    EC = sys::fs::create_directory(Twine(WorkDir), Exist);
  }

  if (EC == error_code::success()) {
    WorkDirCreated = true;
    return true;
  }
 
  return false;
}

IPOFile *IPOFileMgr::createIRFile(const char *Name) {
  IPOFile *F = CreateFile(Name);
  IRFiles.push_back(F);
  return F;
}

IPOFile *IPOFileMgr::createObjFile(const char *Name) {
  IPOFile *F = CreateFile(Name);
  ObjFiles.push_back(F);
  return F;
}

IPOFile *IPOFileMgr::createMakefile(const char *Name) {
  IPOFile *F = CreateFile(Name);
  OtherFiles.push_back(F);
  return F;
}

void IPOFileMgr::removeAllUnneededFiles() {
  FileNameVect ToRm;
  getFilesNeedToRemove(ToRm);

  for (SmallVector<const char *, 4>::iterator I = ToRm.begin(), E = ToRm.end();
       I != E; I++) {
    const char *FN = *I;
    sys::fs::file_status Stat;
    if (sys::fs::status(Twine(FN), Stat) != error_code::success())
      continue;

    uint32_t Dummy;
    if (sys::fs::is_directory(FN))
      sys::fs::remove_all(Twine(FN), Dummy);
    else
      sys::fs::remove(Twine(FN));
  }
}
