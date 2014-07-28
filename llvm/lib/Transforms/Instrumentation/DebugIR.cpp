//===--- DebugIR.cpp - Transform debug metadata to allow debugging IR -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A Module transform pass that emits a succinct version of the IR and replaces
// the source file metadata to allow debuggers to step through the IR.
//
// FIXME: instead of replacing debug metadata, this pass should allow for
// additional metadata to be used to point capable debuggers to the IR file
// without destroying the mapping to the original source file.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ValueMap.h"
#include "DebugIR.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <string>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

using namespace llvm;

#define DEBUG_TYPE "debug-ir"

namespace {

/// Builds a map of Value* to line numbers on which the Value appears in a
/// textual representation of the IR by plugging into the AssemblyWriter by
/// masquerading as an AssemblyAnnotationWriter.
class ValueToLineMap : public AssemblyAnnotationWriter {
  ValueMap<const Value *, unsigned int> Lines;
  typedef ValueMap<const Value *, unsigned int>::const_iterator LineIter;

  void addEntry(const Value *V, formatted_raw_ostream &Out) {
    Out.flush();
    Lines.insert(std::make_pair(V, Out.getLine() + 1));
  }

public:

  /// Prints Module to a null buffer in order to build the map of Value pointers
  /// to line numbers.
  ValueToLineMap(const Module *M) {
    raw_null_ostream ThrowAway;
    M->print(ThrowAway, this);
  }

  // This function is called after an Instruction, GlobalValue, or GlobalAlias
  // is printed.
  void printInfoComment(const Value &V, formatted_raw_ostream &Out) override {
    addEntry(&V, Out);
  }

  void emitFunctionAnnot(const Function *F,
                         formatted_raw_ostream &Out) override {
    addEntry(F, Out);
  }

  /// If V appears on a line in the textual IR representation, sets Line to the
  /// line number and returns true, otherwise returns false.
  bool getLine(const Value *V, unsigned int &Line) const {
    LineIter i = Lines.find(V);
    if (i != Lines.end()) {
      Line = i->second;
      return true;
    }
    return false;
  }
};

/// Removes debug intrisncs like llvm.dbg.declare and llvm.dbg.value.
class DebugIntrinsicsRemover : public InstVisitor<DebugIntrinsicsRemover> {
  void remove(Instruction &I) { I.eraseFromParent(); }

public:
  static void process(Module &M) {
    DebugIntrinsicsRemover Remover;
    Remover.visit(&M);
  }
  void visitDbgDeclareInst(DbgDeclareInst &I) { remove(I); }
  void visitDbgValueInst(DbgValueInst &I) { remove(I); }
  void visitDbgInfoIntrinsic(DbgInfoIntrinsic &I) { remove(I); }
};

/// Removes debug metadata (!dbg) nodes from all instructions, and optionally
/// metadata named "llvm.dbg.cu" if RemoveNamedInfo is true.
class DebugMetadataRemover : public InstVisitor<DebugMetadataRemover> {
  bool RemoveNamedInfo;

public:
  static void process(Module &M, bool RemoveNamedInfo = true) {
    DebugMetadataRemover Remover(RemoveNamedInfo);
    Remover.run(&M);
  }

  DebugMetadataRemover(bool RemoveNamedInfo)
      : RemoveNamedInfo(RemoveNamedInfo) {}

  void visitInstruction(Instruction &I) {
    if (I.getMetadata(LLVMContext::MD_dbg))
      I.setMetadata(LLVMContext::MD_dbg, nullptr);
  }

  void run(Module *M) {
    // Remove debug metadata attached to instructions
    visit(M);

    if (RemoveNamedInfo) {
      // Remove CU named metadata (and all children nodes)
      NamedMDNode *Node = M->getNamedMetadata("llvm.dbg.cu");
      if (Node)
        M->eraseNamedMetadata(Node);
    }
  }
};

/// Updates debug metadata in a Module:
///   - changes Filename/Directory to values provided on construction
///   - adds/updates line number (DebugLoc) entries associated with each
///     instruction to reflect the instruction's location in an LLVM IR file
class DIUpdater : public InstVisitor<DIUpdater> {
  /// Builder of debug information
  DIBuilder Builder;

  /// Helper for type attributes/sizes/etc
  DataLayout Layout;

  /// Map of Value* to line numbers
  const ValueToLineMap LineTable;

  /// Map of Value* (in original Module) to Value* (in optional cloned Module)
  const ValueToValueMapTy *VMap;

  /// Directory of debug metadata
  DebugInfoFinder Finder;

  /// Source filename and directory
  StringRef Filename;
  StringRef Directory;

  // CU nodes needed when creating DI subprograms
  MDNode *FileNode;
  MDNode *LexicalBlockFileNode;
  const MDNode *CUNode;

  ValueMap<const Function *, MDNode *> SubprogramDescriptors;
  DenseMap<const Type *, MDNode *> TypeDescriptors;

public:
  DIUpdater(Module &M, StringRef Filename = StringRef(),
            StringRef Directory = StringRef(), const Module *DisplayM = nullptr,
            const ValueToValueMapTy *VMap = nullptr)
      : Builder(M), Layout(&M), LineTable(DisplayM ? DisplayM : &M), VMap(VMap),
        Finder(), Filename(Filename), Directory(Directory), FileNode(nullptr),
        LexicalBlockFileNode(nullptr), CUNode(nullptr) {
    Finder.processModule(M);
    visit(&M);
  }

  ~DIUpdater() { Builder.finalize(); }

  void visitModule(Module &M) {
    if (Finder.compile_unit_count() > 1)
      report_fatal_error("DebugIR pass supports only a signle compile unit per "
                         "Module.");
    createCompileUnit(Finder.compile_unit_count() == 1 ?
                      (MDNode*)*Finder.compile_units().begin() : nullptr);
  }

  void visitFunction(Function &F) {
    if (F.isDeclaration() || findDISubprogram(&F))
      return;

    StringRef MangledName = F.getName();
    DICompositeType Sig = createFunctionSignature(&F);

    // find line of function declaration
    unsigned Line = 0;
    if (!findLine(&F, Line)) {
      DEBUG(dbgs() << "WARNING: No line for Function " << F.getName().str()
                   << "\n");
      return;
    }

    Instruction *FirstInst = F.begin()->begin();
    unsigned ScopeLine = 0;
    if (!findLine(FirstInst, ScopeLine)) {
      DEBUG(dbgs() << "WARNING: No line for 1st Instruction in Function "
                   << F.getName().str() << "\n");
      return;
    }

    bool Local = F.hasInternalLinkage();
    bool IsDefinition = !F.isDeclaration();
    bool IsOptimized = false;

    int FuncFlags = llvm::DIDescriptor::FlagPrototyped;
    assert(CUNode && FileNode);
    DISubprogram Sub = Builder.createFunction(
        DICompileUnit(CUNode), F.getName(), MangledName, DIFile(FileNode), Line,
        Sig, Local, IsDefinition, ScopeLine, FuncFlags, IsOptimized, &F);
    assert(Sub.isSubprogram());
    DEBUG(dbgs() << "create subprogram mdnode " << *Sub << ": "
                 << "\n");

    SubprogramDescriptors.insert(std::make_pair(&F, Sub));
  }

  void visitInstruction(Instruction &I) {
    DebugLoc Loc(I.getDebugLoc());

    /// If a ValueToValueMap is provided, use it to get the real instruction as
    /// the line table was generated on a clone of the module on which we are
    /// operating.
    Value *RealInst = nullptr;
    if (VMap)
      RealInst = VMap->lookup(&I);

    if (!RealInst)
      RealInst = &I;

    unsigned Col = 0; // FIXME: support columns
    unsigned Line;
    if (!LineTable.getLine(RealInst, Line)) {
      // Instruction has no line, it may have been removed (in the module that
      // will be passed to the debugger) so there is nothing to do here.
      DEBUG(dbgs() << "WARNING: no LineTable entry for instruction " << RealInst
                   << "\n");
      DEBUG(RealInst->dump());
      return;
    }

    DebugLoc NewLoc;
    if (!Loc.isUnknown())
      // I had a previous debug location: re-use the DebugLoc
      NewLoc = DebugLoc::get(Line, Col, Loc.getScope(RealInst->getContext()),
                             Loc.getInlinedAt(RealInst->getContext()));
    else if (MDNode *scope = findScope(&I))
      NewLoc = DebugLoc::get(Line, Col, scope, nullptr);
    else {
      DEBUG(dbgs() << "WARNING: no valid scope for instruction " << &I
                   << ". no DebugLoc will be present."
                   << "\n");
      return;
    }

    addDebugLocation(I, NewLoc);
  }

private:

  void createCompileUnit(MDNode *CUToReplace) {
    std::string Flags;
    bool IsOptimized = false;
    StringRef Producer;
    unsigned RuntimeVersion(0);
    StringRef SplitName;

    if (CUToReplace) {
      // save fields from existing CU to re-use in the new CU
      DICompileUnit ExistingCU(CUToReplace);
      Producer = ExistingCU.getProducer();
      IsOptimized = ExistingCU.isOptimized();
      Flags = ExistingCU.getFlags();
      RuntimeVersion = ExistingCU.getRunTimeVersion();
      SplitName = ExistingCU.getSplitDebugFilename();
    } else {
      Producer =
          "LLVM Version " STR(LLVM_VERSION_MAJOR) "." STR(LLVM_VERSION_MINOR);
    }

    CUNode =
        Builder.createCompileUnit(dwarf::DW_LANG_C99, Filename, Directory,
                                  Producer, IsOptimized, Flags, RuntimeVersion);

    if (CUToReplace)
      CUToReplace->replaceAllUsesWith(const_cast<MDNode *>(CUNode));

    DICompileUnit CU(CUNode);
    FileNode = Builder.createFile(Filename, Directory);
    LexicalBlockFileNode = Builder.createLexicalBlockFile(CU, DIFile(FileNode));
  }

  /// Returns the MDNode* that represents the DI scope to associate with I
  MDNode *findScope(const Instruction *I) {
    const Function *F = I->getParent()->getParent();
    if (MDNode *ret = findDISubprogram(F))
      return ret;

    DEBUG(dbgs() << "WARNING: Using fallback lexical block file scope "
                 << LexicalBlockFileNode << " as scope for instruction " << I
                 << "\n");
    return LexicalBlockFileNode;
  }

  /// Returns the MDNode* that is the descriptor for F
  MDNode *findDISubprogram(const Function *F) {
    typedef ValueMap<const Function *, MDNode *>::const_iterator FuncNodeIter;
    FuncNodeIter i = SubprogramDescriptors.find(F);
    if (i != SubprogramDescriptors.end())
      return i->second;

    DEBUG(dbgs() << "searching for DI scope node for Function " << F
                 << " in a list of " << Finder.subprogram_count()
                 << " subprogram nodes"
                 << "\n");

    for (DISubprogram S : Finder.subprograms()) {
      if (S.getFunction() == F) {
        DEBUG(dbgs() << "Found DISubprogram " << S << " for function "
                     << S.getFunction() << "\n");
        return S;
      }
    }
    DEBUG(dbgs() << "unable to find DISubprogram node for function "
                 << F->getName().str() << "\n");
    return nullptr;
  }

  /// Sets Line to the line number on which V appears and returns true. If a
  /// line location for V is not found, returns false.
  bool findLine(const Value *V, unsigned &Line) {
    if (LineTable.getLine(V, Line))
      return true;

    if (VMap) {
      Value *mapped = VMap->lookup(V);
      if (mapped && LineTable.getLine(mapped, Line))
        return true;
    }
    return false;
  }

  std::string getTypeName(Type *T) {
    std::string TypeName;
    raw_string_ostream TypeStream(TypeName);
    if (T)
      T->print(TypeStream);
    else
      TypeStream << "Printing <null> Type";
    TypeStream.flush();
    return TypeName;
  }

  /// Returns the MDNode that represents type T if it is already created, or 0
  /// if it is not.
  MDNode *getType(const Type *T) {
    typedef DenseMap<const Type *, MDNode *>::const_iterator TypeNodeIter;
    TypeNodeIter i = TypeDescriptors.find(T);
    if (i != TypeDescriptors.end())
      return i->second;
    return nullptr;
  }

  /// Returns a DebugInfo type from an LLVM type T.
  DIDerivedType getOrCreateType(Type *T) {
    MDNode *N = getType(T);
    if (N)
      return DIDerivedType(N);
    else if (T->isVoidTy())
      return DIDerivedType(nullptr);
    else if (T->isStructTy()) {
      N = Builder.createStructType(
          DIScope(LexicalBlockFileNode), T->getStructName(), DIFile(FileNode),
          0, Layout.getTypeSizeInBits(T), Layout.getABITypeAlignment(T), 0,
          DIType(nullptr), DIArray(nullptr)); // filled in later

      // N is added to the map (early) so that element search below can find it,
      // so as to avoid infinite recursion for structs that contain pointers to
      // their own type.
      TypeDescriptors[T] = N;
      DICompositeType StructDescriptor(N);

      SmallVector<Value *, 4> Elements;
      for (unsigned i = 0; i < T->getStructNumElements(); ++i)
        Elements.push_back(getOrCreateType(T->getStructElementType(i)));

      // set struct elements
      StructDescriptor.setArrays(Builder.getOrCreateArray(Elements));
    } else if (T->isPointerTy()) {
      Type *PointeeTy = T->getPointerElementType();
      if (!(N = getType(PointeeTy)))
        N = Builder.createPointerType(
            getOrCreateType(PointeeTy), Layout.getPointerTypeSizeInBits(T),
            Layout.getPrefTypeAlignment(T), getTypeName(T));
    } else if (T->isArrayTy()) {
      SmallVector<Value *, 1> Subrange;
      Subrange.push_back(
          Builder.getOrCreateSubrange(0, T->getArrayNumElements() - 1));

      N = Builder.createArrayType(Layout.getTypeSizeInBits(T),
                                  Layout.getPrefTypeAlignment(T),
                                  getOrCreateType(T->getArrayElementType()),
                                  Builder.getOrCreateArray(Subrange));
    } else {
      int encoding = llvm::dwarf::DW_ATE_signed;
      if (T->isIntegerTy())
        encoding = llvm::dwarf::DW_ATE_unsigned;
      else if (T->isFloatingPointTy())
        encoding = llvm::dwarf::DW_ATE_float;

      N = Builder.createBasicType(getTypeName(T), T->getPrimitiveSizeInBits(),
                                  0, encoding);
    }
    TypeDescriptors[T] = N;
    return DIDerivedType(N);
  }

  /// Returns a DebugInfo type that represents a function signature for Func.
  DICompositeType createFunctionSignature(const Function *Func) {
    SmallVector<Value *, 4> Params;
    DIDerivedType ReturnType(getOrCreateType(Func->getReturnType()));
    Params.push_back(ReturnType);

    const Function::ArgumentListType &Args(Func->getArgumentList());
    for (Function::ArgumentListType::const_iterator i = Args.begin(),
                                                    e = Args.end();
         i != e; ++i) {
      Type *T(i->getType());
      Params.push_back(getOrCreateType(T));
    }

    DITypeArray ParamArray = Builder.getOrCreateTypeArray(Params);
    return Builder.createSubroutineType(DIFile(FileNode), ParamArray);
  }

  /// Associates Instruction I with debug location Loc.
  void addDebugLocation(Instruction &I, DebugLoc Loc) {
    MDNode *MD = Loc.getAsMDNode(I.getContext());
    I.setMetadata(LLVMContext::MD_dbg, MD);
  }
};

/// Sets Filename/Directory from the Module identifier and returns true, or
/// false if source information is not present.
bool getSourceInfoFromModule(const Module &M, std::string &Directory,
                             std::string &Filename) {
  std::string PathStr(M.getModuleIdentifier());
  if (PathStr.length() == 0 || PathStr == "<stdin>")
    return false;

  Filename = sys::path::filename(PathStr);
  SmallVector<char, 16> Path(PathStr.begin(), PathStr.end());
  sys::path::remove_filename(Path);
  Directory = StringRef(Path.data(), Path.size());
  return true;
}

// Sets Filename/Directory from debug information in M and returns true, or
// false if no debug information available, or cannot be parsed.
bool getSourceInfoFromDI(const Module &M, std::string &Directory,
                         std::string &Filename) {
  NamedMDNode *CUNode = M.getNamedMetadata("llvm.dbg.cu");
  if (!CUNode || CUNode->getNumOperands() == 0)
    return false;

  DICompileUnit CU(CUNode->getOperand(0));
  if (!CU.Verify())
    return false;

  Filename = CU.getFilename();
  Directory = CU.getDirectory();
  return true;
}

} // anonymous namespace

namespace llvm {

bool DebugIR::getSourceInfo(const Module &M) {
  ParsedPath = getSourceInfoFromDI(M, Directory, Filename) ||
               getSourceInfoFromModule(M, Directory, Filename);
  return ParsedPath;
}

bool DebugIR::updateExtension(StringRef NewExtension) {
  size_t dot = Filename.find_last_of(".");
  if (dot == std::string::npos)
    return false;

  Filename.erase(dot);
  Filename += NewExtension.str();
  return true;
}

void DebugIR::generateFilename(std::unique_ptr<int> &fd) {
  SmallVector<char, 16> PathVec;
  fd.reset(new int);
  sys::fs::createTemporaryFile("debug-ir", "ll", *fd, PathVec);
  StringRef Path(PathVec.data(), PathVec.size());
  Filename = sys::path::filename(Path);
  sys::path::remove_filename(PathVec);
  Directory = StringRef(PathVec.data(), PathVec.size());

  GeneratedPath = true;
}

std::string DebugIR::getPath() {
  SmallVector<char, 16> Path;
  sys::path::append(Path, Directory, Filename);
  Path.resize(Filename.size() + Directory.size() + 2);
  Path[Filename.size() + Directory.size() + 1] = '\0';
  return std::string(Path.data());
}

void DebugIR::writeDebugBitcode(const Module *M, int *fd) {
  std::unique_ptr<raw_fd_ostream> Out;
  std::string error;

  if (!fd) {
    std::string Path = getPath();
    Out.reset(new raw_fd_ostream(Path.c_str(), error, sys::fs::F_Text));
    DEBUG(dbgs() << "WRITING debug bitcode from Module " << M << " to file "
                 << Path << "\n");
  } else {
    DEBUG(dbgs() << "WRITING debug bitcode from Module " << M << " to fd "
                 << *fd << "\n");
    Out.reset(new raw_fd_ostream(*fd, true));
  }

  M->print(*Out, nullptr);
  Out->close();
}

void DebugIR::createDebugInfo(Module &M, std::unique_ptr<Module> &DisplayM) {
  if (M.getFunctionList().size() == 0)
    // no functions -- no debug info needed
    return;

  std::unique_ptr<ValueToValueMapTy> VMap;

  if (WriteSourceToDisk && (HideDebugIntrinsics || HideDebugMetadata)) {
    VMap.reset(new ValueToValueMapTy);
    DisplayM.reset(CloneModule(&M, *VMap));

    if (HideDebugIntrinsics)
      DebugIntrinsicsRemover::process(*DisplayM);

    if (HideDebugMetadata)
      DebugMetadataRemover::process(*DisplayM);
  }

  DIUpdater R(M, Filename, Directory, DisplayM.get(), VMap.get());
}

bool DebugIR::isMissingPath() { return Filename.empty() || Directory.empty(); }

bool DebugIR::runOnModule(Module &M) {
  std::unique_ptr<int> fd;

  if (isMissingPath() && !getSourceInfo(M)) {
    if (!WriteSourceToDisk)
      report_fatal_error("DebugIR unable to determine file name in input. "
                         "Ensure Module contains an identifier, a valid "
                         "DICompileUnit, or construct DebugIR with "
                         "non-empty Filename/Directory parameters.");
    else
      generateFilename(fd);
  }

  if (!GeneratedPath && WriteSourceToDisk)
    updateExtension(".debug-ll");

  // Clear line numbers. Keep debug info (if any) if we were able to read the
  // file name from the DICompileUnit descriptor.
  DebugMetadataRemover::process(M, !ParsedPath);

  std::unique_ptr<Module> DisplayM;
  createDebugInfo(M, DisplayM);
  if (WriteSourceToDisk) {
    Module *OutputM = DisplayM.get() ? DisplayM.get() : &M;
    writeDebugBitcode(OutputM, fd.get());
  }

  DEBUG(M.dump());
  return true;
}

bool DebugIR::runOnModule(Module &M, std::string &Path) {
  bool result = runOnModule(M);
  Path = getPath();
  return result;
}

} // llvm namespace

char DebugIR::ID = 0;
INITIALIZE_PASS(DebugIR, "debug-ir", "Enable debugging IR", false, false)

ModulePass *llvm::createDebugIRPass(bool HideDebugIntrinsics,
                                    bool HideDebugMetadata, StringRef Directory,
                                    StringRef Filename) {
  return new DebugIR(HideDebugIntrinsics, HideDebugMetadata, Directory,
                     Filename);
}

ModulePass *llvm::createDebugIRPass() { return new DebugIR(); }
