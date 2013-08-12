//===---------- LTOPostIPODriver.h - PostIPO Driver -----------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PostIPODriver class which is the driver for Post-IPO
// compilation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/ObjCARC.h"
#include "LTOPartition.h"
#include "LTOPostIPODriver.h"

using namespace llvm;
using namespace lto;

// /////////////////////////////////////////////////////////////////////////////
//
//      Declare all variants of Post-IPO drivers
//
// /////////////////////////////////////////////////////////////////////////////
//
namespace {
  /// \breif Base class for all driver variants.
  ///
  class PostIPODrvBaseImpl {
  public:
    PostIPODrvBaseImpl(TargetMachine *Targ, IPOPartMgr &IPM, IPOFileMgr &IFM,
                       bool ToMergeObjs):
      PartMgr(IPM), FileMgr(IFM), MergedObjFile(0), Target(Targ),
      MergeObjs(ToMergeObjs) {}
  
    virtual ~PostIPODrvBaseImpl() {};
  
    IPOPartMgr &getPartitionMgr() { return PartMgr; }
    IPOFileMgr &getFileMgr() { return FileMgr; }

    // Implement the PostIPODriver::getSingleObjFile()
    virtual IPOFile *getSingleObjFile() const = 0;

    bool IsToMergeObj() const { return MergeObjs; }
  
    virtual bool Compile(std::string &ErrMsg) = 0;
  
  protected:
    // Populate post-IPO scalar optimization pass manager
    bool PopulatePostIPOOptPM(PassManager &PM);

    // Populate post-IPO machine-specific CodeGen pass manager
    bool PopulateCodeGenPM(PassManager &PM, formatted_raw_ostream &OutFile,
                           std::string &Err);

  protected:
    IPOPartMgr &PartMgr;
    IPOFileMgr &FileMgr;
    IPOFile *MergedObjFile;
    TargetMachine *Target;
    bool MergeObjs;
  };
  
  /// \breif PostIPO driver for the compiling the entire program without
  ///    partition.
  class PostIPODrvSerial : public PostIPODrvBaseImpl {
  public:
    PostIPODrvSerial(TargetMachine *T, IPOPartMgr &IPM, IPOFileMgr &IFM,
                        bool ToMergeObjs) :
      PostIPODrvBaseImpl(T, IPM, IFM, ToMergeObjs) {}

    virtual bool Compile(std::string &ErrMsg);
    virtual IPOFile *getSingleObjFile() const;

  private:
    Module *getModule() const { return (*PartMgr.begin())->getModule(); }
  };
}

// ////////////////////////////////////////////////////////////////////////////
//
//              Implemetation of PostIPODriver
//
// ////////////////////////////////////////////////////////////////////////////
//
PostIPODriver::PostIPODriver(VariantTy V, TargetMachine *TM, IPOPartMgr &IPM,
                             IPOFileMgr &IFM, bool ToMergeObjs) {
  if (V == PIDV_SERIAL) 
    DrvImpl = new PostIPODrvSerial(TM, IPM, IFM, ToMergeObjs);
  else 
    assert(false && "TBD");
}

bool PostIPODriver::Compile(std::string &ErrMsg) {
  PostIPODrvBaseImpl *P = static_cast<PostIPODrvBaseImpl *>(DrvImpl);
  return P->Compile(ErrMsg);
}

IPOFile *PostIPODriver::getSingleObjFile() const {
  PostIPODrvBaseImpl *P = static_cast<PostIPODrvBaseImpl *>(DrvImpl);
  return P->getSingleObjFile();
}

// ////////////////////////////////////////////////////////////////////////////
//
//              Implemetation of PostIPODrvBaseImpl
//
// ////////////////////////////////////////////////////////////////////////////
//
bool PostIPODrvBaseImpl::PopulatePostIPOOptPM(PassManager &PM) {
  (void)PM;
  return true;
}

bool PostIPODrvBaseImpl::PopulateCodeGenPM(PassManager &PM,
                                           formatted_raw_ostream &OutFile,
                                           std::string &Err) {
  PM.add(new DataLayout(*Target->getDataLayout()));
  Target->addAnalysisPasses(PM);

  // If the bitcode files contain ARC code and were compiled with optimization,
  // the ObjCARCContractPass must be run, so do it unconditionally here.
  PM.add(createObjCARCContractPass());

  if (Target->addPassesToEmitFile(PM, OutFile,
                                  TargetMachine::CGFT_ObjectFile)) {
    Err = "target file type not supported";
    return false;
  }
  return true;
}

// ////////////////////////////////////////////////////////////////////////////
//
//              Implemetation of PostIPODrvSerial
//
// ////////////////////////////////////////////////////////////////////////////
//
bool PostIPODrvSerial::Compile(std::string &ErrMsg) {
  Module *M = getModule();

  // Step 1: Run the post-IPO scalar optimizations
  {
    PassManager SoptPM;
    PopulatePostIPOOptPM(SoptPM);
    SoptPM.run(*M);
  }

  // Step 2: Run the post-IPO machine-specific code-generation passes
  {
    IPOFile &Obj = (*PartMgr.begin())->getObjFile();
    raw_fd_ostream ros(Obj.getPath().c_str(), Obj.getLastErrStr(),
                       sys::fs::F_Binary);
    formatted_raw_ostream OutFile(ros);
    
    PassManager CodGenPM;
    if (!PopulateCodeGenPM(CodGenPM, OutFile, ErrMsg)) {
      ErrMsg += Obj.getLastErrStr();
      return false;
    }

    CodGenPM.run(*M);
  }

  return true;
}

IPOFile *PostIPODrvSerial::getSingleObjFile() const {
  assert(!MergedObjFile && "No need to *merge* a single object file");
  IPOPartition *P = *PartMgr.begin();
  return &P->getObjFile();
}
