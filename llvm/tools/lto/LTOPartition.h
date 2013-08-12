//===-------- LTOPartition.h - Partition related classes and functions ---===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file declare the partition related classes and functions. A partition
// is a portion of the merged module. In case partition is disabled, the entire
// merged module is considered as a degenerated partition.
//
//   The classes declared in this file are:
//   o. IPOPartition : to depicit a partition
//   o. IPOFile: It is a "container" collecting miscellaneous information about
//        an intermeidate file, including file name, path, last-err-message etc.
//   o. IPOPartMgr, IPOFileMgr: as the name suggests, it's the manager of 
//        IPOPartitions and IPOFiles, respectively.
//        
//===----------------------------------------------------------------------===//

#ifndef LTO_PARTITION_H
#define LTO_PARTITION_H

#include "llvm/Pass.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/system_error.h"

using namespace llvm;

namespace lto {
  /// \brief To collect miscellaneous information about an intermdiate file.
  ///  
  /// These informration include file name, path, last error message etc.
  ///
  class IPOFile {
  public:
    const std::string &getName() { return Fname; }
    const std::string &getPath() { return Fpath; }

    error_code &getLastErrCode() { return LastErr; }
    std::string &getLastErrStr() { return LastErrStr; }

    bool errOccur() const {
      return LastErr != error_code::success() || !LastErrStr.empty();
    }

    // To keep this file after compilation finish. 
    void setKeep() { Keep = true; }
    bool isKept() const { return Keep; }

  private:
    friend class IPOFileMgr;
    IPOFile(const char* DirName, const char *BaseName, bool Keep=false);
    ~IPOFile();
  
  private:
    std::string Fname;
    std::string Fpath;
    error_code LastErr;
    std::string LastErrStr;
    bool Keep;
  };

  /// \brief To manage IPOFiles, create and remove work-directory.
  ///
  class IPOFileMgr {
  public:
    typedef SmallVector<const char *, 4> FileNameVect;

    IPOFileMgr();

    // NOTE: Do not delete intermeidate in the destructor as we never know
    //   if these files out-last the class or not. It is safe to let linker's
    //   clean-up hook to take care these files.
    ~IPOFileMgr() {};

    void setWorkDir(const char* WD) {
      assert(!WorkDirCreated /* Too late to change mind */ &&
             WorkDir.empty() /* don't change back and forth */ &&
             "Cannot change work dir");
      WorkDir = WD;
    }
    void setKeepWorkDir(bool Keep) { KeepWorkDir = Keep; }
    bool IsToKeepWorkDir() const { return KeepWorkDir; }
    const std::string &getWorkDir() { return WorkDir; }

    bool createWorkDir(std::string &ErrorInfo);
    
    IPOFile *createIRFile(const char *Name);
    IPOFile *createObjFile(const char *Name);
    IPOFile *createMakefile(const char *Name);

    typedef std::vector<IPOFile *> FileVect;
    FileVect &getIRFiles() { return IRFiles; }
    FileVect &getObjFiles() { return ObjFiles; }

    // Get all files/dirs that need to removed after the LTO complete.
    void getFilesNeedToRemove(FileNameVect &ToRm) {
      ToRm.clear();
      if (!IsToKeepWorkDir() && WorkDirCreated)
        ToRm.push_back(WorkDir.c_str());
    }

    // Remove all files/dirs returned from getFilesNeedToRemove().
    void removeAllUnneededFiles();

  private:
    IPOFile *CreateFile(const char *Name) {
      return new IPOFile(WorkDir.c_str(), Name);
    }

  private:
    FileVect IRFiles;
    FileVect ObjFiles;
    FileVect OtherFiles;
    std::string WorkDir;
    bool KeepWorkDir;
    bool WorkDirCreated;
  };

  /// \brief Describe a partition of the merged module.
  ///
  class IPOPartition {
  public:
    llvm::Module *getModule() const { return Mod; }
    IPOFile &getIRFile() const;
    IPOFile &getObjFile() const;
    const std::string &getIRFilePath() const { return getIRFile().getPath(); }
    const std::string &getObjFilePath() const { return getObjFile().getPath(); }

    // If the bitcode reside in memory or disk
    bool isInMemory() const { return Mod != 0; }

    // Load/store bitcode from/to disk file.
    bool saveBitCode();
    bool loadBitCode();

  private:
    friend class IPOPartMgr;
    IPOPartition(llvm::Module *M, const char *FileNameWoExt, IPOFileMgr &FM);

    // The module associated with this partition
    Module *Mod;
    LLVMContext *Ctx;

    // The bitcode file and its corresponding object file associated with
    // this partition. The names of these two files are different only in
    // extension; the "FileNameWoExt" record their (common) name without 
    // extension.
    //
    mutable IPOFile *IRFile;
    mutable IPOFile *ObjFile;
    std::string FileNameWoExt;

    IPOFileMgr &FileMgr;
  };
  
  /// \brief To manage IPOPartitions
  ///
  class IPOPartMgr {
  public:
    IPOPartMgr(IPOFileMgr &IFM) : FileMgr(IFM), NextPartId(1) {}

    typedef std::vector<IPOPartition *> IPOPartsTy;
    typedef IPOPartsTy::iterator iterator;
    typedef IPOPartsTy::const_iterator const_iterator;

    iterator begin() { return IPOParts.begin(); }
    iterator end() { return IPOParts.end(); }
    const_iterator begin() const { return IPOParts.begin(); }
    const_iterator end() const { return IPOParts.end(); }

    IPOPartition *createIPOPart(Module *);
    IPOPartition *getSinglePartition() {
      assert(IPOParts.size() == 1 && "Has multiple partition");
      return IPOParts[0];
    }

  private:
    IPOPartsTy IPOParts;
    IPOFileMgr &FileMgr;
    int NextPartId;
  };

}

#endif //LTO_PARTITION_H
