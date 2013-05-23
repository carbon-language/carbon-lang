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
// The location where the IR file is emitted is the same as the directory
// operand of the !llvm.dbg.cu metadata node present in the input module. The
// file name is constructed from the original file name by stripping the
// extension and replacing it with "-debug-ll" or the Postfix string specified
// at construction.
//
// FIXME: instead of replacing debug metadata, additional metadata should be
// used to point capable debuggers to the IR file without destroying the
// mapping to the original source file.
//
// FIXME: this pass should not depend on the existance of debug metadata in
// the module as it does now. Instead, it should use DIBuilder to create the
// required metadata.
//
//===----------------------------------------------------------------------===//

#include <string>

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/ValueMap.h"
#include "llvm/Assembly/AssemblyAnnotationWriter.h"
#include "llvm/DebugInfo.h"
#include "llvm/DIBuilder.h"
#include "llvm/InstVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

namespace {

/// Builds a map of Value* to line numbers on which the Value appears in a
/// textual representation of the IR by plugging into the AssemblyWriter by
/// masquerading as an AssemblyAnnotationWriter.
class ValueToLineMap : public AssemblyAnnotationWriter {
  ValueMap<const Value *, unsigned int> Lines;
  typedef ValueMap<const Value *, unsigned int>::const_iterator LineIter;

public:

  /// Prints Module to a null buffer in order to build the map of Value pointers
  /// to line numbers.
  ValueToLineMap(Module *M) {
    raw_null_ostream ThrowAway;
    M->print(ThrowAway, this);
  }

  // This function is called after an Instruction, GlobalValue, or GlobalAlias
  // is printed.
  void printInfoComment(const Value &V, formatted_raw_ostream &Out) {
    Out.flush();
    Lines.insert(std::make_pair(&V, Out.getLine() + 1));
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
  void visitDbgDeclareInst(DbgDeclareInst &I) { remove(I); }
  void visitDbgValueInst(DbgValueInst &I) { remove(I); }
  void visitDbgInfoIntrinsic(DbgInfoIntrinsic &I) { remove(I); }
};

/// Removes debug metadata (!dbg) nodes from all instructions as well as
/// metadata named "llvm.dbg.cu" in the Module.
class DebugMetadataRemover : public InstVisitor<DebugMetadataRemover> {
public:
  void visitInstruction(Instruction &I) {
    if (I.getMetadata(LLVMContext::MD_dbg))
      I.setMetadata(LLVMContext::MD_dbg, 0);
  }

  void run(Module *M) {
    // Remove debug metadata attached to instructions
    visit(M);

    // Remove CU named metadata (and all children nodes)
    NamedMDNode *Node = M->getNamedMetadata("llvm.dbg.cu");
    M->eraseNamedMetadata(Node);
  }
};

/// Replaces line number metadata attached to Instruction nodes with new line
/// numbers provided by the ValueToLineMap.
class LineNumberReplacer : public InstVisitor<LineNumberReplacer> {
  /// Table of line numbers
  const ValueToLineMap &LineTable;

  /// Table of cloned values
  const ValueToValueMapTy &VMap;

  /// Directory of debug metadata
  const DebugInfoFinder &Finder;

public:
  LineNumberReplacer(const ValueToLineMap &VLM, const DebugInfoFinder &Finder,
                     const ValueToValueMapTy &VMap)
      : LineTable(VLM), VMap(VMap), Finder(Finder) {}

  void visitInstruction(Instruction &I) {
    DebugLoc Loc(I.getDebugLoc());

    unsigned Col = 0; // FIXME: support columns
    unsigned Line;
    if (!LineTable.getLine(VMap.lookup(&I), Line))
      // Instruction has no line, it may have been removed (in the module that
      // will be passed to the debugger) so there is nothing to do here.
      return;

    DebugLoc NewLoc;
    if (!Loc.isUnknown())
      // I had a previous debug location: re-use the DebugLoc
      NewLoc = DebugLoc::get(Line, Col, Loc.getScope(I.getContext()),
                             Loc.getInlinedAt(I.getContext()));
    else if (MDNode *scope = findFunctionMD(I.getParent()->getParent()))
      // I had no previous debug location, but M has some debug information
      NewLoc =
          DebugLoc::get(Line, Col, scope, /*FIXME: inlined instructions*/ 0);
    else
      // Neither I nor M has any debug information -- nothing to do here.
      // FIXME: support debugging of undecorated IR (generated by clang without
      //        the -g option)
      return;

    addDebugLocation(const_cast<Instruction &>(I), NewLoc);
  }

private:

  /// Returns the MDNode that corresponds with F
  MDNode *findFunctionMD(const Function *F) {
    for (DebugInfoFinder::iterator i = Finder.subprogram_begin(),
                                   e = Finder.subprogram_end();
         i != e; ++i) {
      DISubprogram S(*i);
      if (S.getFunction() == F)
        return *i;
    }
    // cannot find F -- likely means there is no debug information
    return 0;
  }

  void addDebugLocation(Instruction &I, DebugLoc Loc) {
    MDNode *MD = Loc.getAsMDNode(I.getContext());
    I.setMetadata(LLVMContext::MD_dbg, MD);
  }
};

class DebugIR : public ModulePass {
  std::string Postfix;
  std::string Filename;

  /// Flags to control the verbosity of the generated IR file
  bool hideDebugIntrinsics;
  bool hideDebugMetadata;

public:
  static char ID;

  const char *getPassName() const { return "DebugIR"; }

  // FIXME: figure out if we are compiling something that already exists on disk
  // in text IR form, in which case we can omit outputting a new IR file, or if
  // we're building something from memory where we actually need to emit a new
  // IR file for the debugger.

  /// Output a file with the same base name as the original, but with the
  /// postfix "-debug-ll" appended.
  DebugIR()
      : ModulePass(ID), Postfix("-debug-ll"), hideDebugIntrinsics(true),
        hideDebugMetadata(true) {}

  /// Customize the postfix string used to replace the extension of the
  /// original filename that appears in the !llvm.dbg.cu metadata node.
  DebugIR(StringRef postfix, bool hideDebugIntrinsics, bool hideDebugMetadata)
      : ModulePass(ID), Postfix(postfix),
        hideDebugIntrinsics(hideDebugIntrinsics),
        hideDebugMetadata(hideDebugMetadata) {}

private:
  // Modify the filename embedded in the Compilation-Unit debug information of M
  bool replaceFilename(Module &M, const DebugInfoFinder &Finder) {
    bool changed = false;

    // Sanity check -- if llvm.dbg.cu node exists, the DebugInfoFinder
    // better have found at least one CU!
    if (M.getNamedMetadata("llvm.dbg.cu"))
      assert(Finder.compile_unit_count() > 0 &&
             "Found no compile units but llvm.dbg.cu node exists");

    for (DebugInfoFinder::iterator i = Finder.compile_unit_begin(),
                                   e = Finder.compile_unit_end();
         i != e; ++i) {
      DICompileUnit CU(*i);
      Filename = CU.getFilename();

      // Replace extension with postfix
      size_t dot = Filename.find_last_of(".");
      if (dot != std::string::npos)
        Filename.erase(dot);
      Filename += Postfix;

      CU.setFilename(Filename, M.getContext());
      changed = true;
    }
    return changed;
  }

  /// Replace existing line number metadata with line numbers that correspond
  /// with the IR file that is seen by the debugger.
  void addLineNumberMetadata(Module *M, const ValueToLineMap &VLM,
                             const ValueToValueMapTy &VMap,
                             const DebugInfoFinder &Finder) {
    LineNumberReplacer Replacer(VLM, Finder, VMap);
    Replacer.visit(M);
  }

  void writeDebugBitcode(Module *M) {
    std::string error;
    tool_output_file OutFile(Filename.c_str(), error);
    OutFile.keep();
    formatted_raw_ostream OS;
    OS.setStream(OutFile.os());
    M->print(OS, 0);
  }

  void removeDebugIntrinsics(Module *M) {
    DebugIntrinsicsRemover Remover;
    Remover.visit(M);
  }

  void removeDebugMetadata(Module *M) {
    DebugMetadataRemover Remover;
    Remover.run(M);
  }

  void updateAndWriteDebugIRFile(Module *M, const DebugInfoFinder &Finder) {
    // The module we output in text form for a debugger to open is stripped of
    // 'extras' like debug intrinsics that end up in DWARF anyways and just
    // clutter the debug experience.

    ValueToValueMapTy VMap;
    Module *DebuggerM = CloneModule(M, VMap);

    if (hideDebugIntrinsics)
      removeDebugIntrinsics(DebuggerM);

    if (hideDebugMetadata)
      removeDebugMetadata(DebuggerM);

    // FIXME: remove all debug metadata from M once we support generating DWARF
    // subprogram attributes.

    ValueToLineMap LineTable(DebuggerM);
    addLineNumberMetadata(M, LineTable, VMap, Finder);
    writeDebugBitcode(DebuggerM);
  }

  bool runOnModule(Module &M) {
    // Stores existing debug info needed when creating new line number entries.
    DebugInfoFinder Finder;
    Finder.processModule(M);

    bool changed = replaceFilename(M, Finder);
    if (changed)
      updateAndWriteDebugIRFile(&M, Finder);
    return changed;
  }
};

} // anonymous namespace

char DebugIR::ID = 0;
INITIALIZE_PASS(DebugIR, "debug-ir", "Enable debugging IR", false, false)

ModulePass *llvm::createDebugIRPass(StringRef FilenamePostfix,
                                    bool hideDebugIntrinsics,
                                    bool hideDebugMetadata) {
  return new DebugIR(FilenamePostfix, hideDebugIntrinsics, hideDebugMetadata);
}
