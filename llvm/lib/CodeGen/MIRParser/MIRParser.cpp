//===- MIRParser.cpp - MIR serialization format parser implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that parses the optional LLVM IR and machine
// functions that are stored in MIR files.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include <memory>

using namespace llvm;

namespace llvm {

/// This class implements the parsing of LLVM IR that's embedded inside a MIR
/// file.
class MIRParserImpl {
  SourceMgr SM;
  StringRef Filename;
  LLVMContext &Context;
  StringMap<std::unique_ptr<yaml::MachineFunction>> Functions;

public:
  MIRParserImpl(std::unique_ptr<MemoryBuffer> Contents, StringRef Filename,
                LLVMContext &Context);

  void reportDiagnostic(const SMDiagnostic &Diag);

  /// Report an error with the given message at unknown location.
  ///
  /// Always returns true.
  bool error(const Twine &Message);

  /// Try to parse the optional LLVM module and the machine functions in the MIR
  /// file.
  ///
  /// Return null if an error occurred.
  std::unique_ptr<Module> parse();

  /// Parse the machine function in the current YAML document.
  ///
  /// \param NoLLVMIR - set to true when the MIR file doesn't have LLVM IR.
  /// A dummy IR function is created and inserted into the given module when
  /// this parameter is true.
  ///
  /// Return true if an error occurred.
  bool parseMachineFunction(yaml::Input &In, Module &M, bool NoLLVMIR);

  /// Initialize the machine function to the state that's described in the MIR
  /// file.
  ///
  /// Return true if error occurred.
  bool initializeMachineFunction(MachineFunction &MF);

  /// Initialize the machine basic block using it's YAML representation.
  ///
  /// Return true if an error occurred.
  bool initializeMachineBasicBlock(MachineBasicBlock &MBB,
                                   const yaml::MachineBasicBlock &YamlMBB);

private:
  /// Return a MIR diagnostic converted from an LLVM assembly diagnostic.
  SMDiagnostic diagFromLLVMAssemblyDiag(const SMDiagnostic &Error,
                                        SMRange SourceRange);

  /// Create an empty function with the given name.
  void createDummyFunction(StringRef Name, Module &M);
};

} // end namespace llvm

MIRParserImpl::MIRParserImpl(std::unique_ptr<MemoryBuffer> Contents,
                             StringRef Filename, LLVMContext &Context)
    : SM(), Filename(Filename), Context(Context) {
  SM.AddNewSourceBuffer(std::move(Contents), SMLoc());
}

bool MIRParserImpl::error(const Twine &Message) {
  Context.diagnose(DiagnosticInfoMIRParser(
      DS_Error, SMDiagnostic(Filename, SourceMgr::DK_Error, Message.str())));
  return true;
}

void MIRParserImpl::reportDiagnostic(const SMDiagnostic &Diag) {
  DiagnosticSeverity Kind;
  switch (Diag.getKind()) {
  case SourceMgr::DK_Error:
    Kind = DS_Error;
    break;
  case SourceMgr::DK_Warning:
    Kind = DS_Warning;
    break;
  case SourceMgr::DK_Note:
    Kind = DS_Note;
    break;
  }
  Context.diagnose(DiagnosticInfoMIRParser(Kind, Diag));
}

static void handleYAMLDiag(const SMDiagnostic &Diag, void *Context) {
  reinterpret_cast<MIRParserImpl *>(Context)->reportDiagnostic(Diag);
}

std::unique_ptr<Module> MIRParserImpl::parse() {
  yaml::Input In(SM.getMemoryBuffer(SM.getMainFileID())->getBuffer(),
                 /*Ctxt=*/nullptr, handleYAMLDiag, this);

  if (!In.setCurrentDocument()) {
    if (In.error())
      return nullptr;
    // Create an empty module when the MIR file is empty.
    return llvm::make_unique<Module>(Filename, Context);
  }

  std::unique_ptr<Module> M;
  bool NoLLVMIR = false;
  // Parse the block scalar manually so that we can return unique pointer
  // without having to go trough YAML traits.
  if (const auto *BSN =
          dyn_cast_or_null<yaml::BlockScalarNode>(In.getCurrentNode())) {
    SMDiagnostic Error;
    M = parseAssembly(MemoryBufferRef(BSN->getValue(), Filename), Error,
                      Context);
    if (!M) {
      reportDiagnostic(diagFromLLVMAssemblyDiag(Error, BSN->getSourceRange()));
      return M;
    }
    In.nextDocument();
    if (!In.setCurrentDocument())
      return M;
  } else {
    // Create an new, empty module.
    M = llvm::make_unique<Module>(Filename, Context);
    NoLLVMIR = true;
  }

  // Parse the machine functions.
  do {
    if (parseMachineFunction(In, *M, NoLLVMIR))
      return nullptr;
    In.nextDocument();
  } while (In.setCurrentDocument());

  return M;
}

bool MIRParserImpl::parseMachineFunction(yaml::Input &In, Module &M,
                                         bool NoLLVMIR) {
  auto MF = llvm::make_unique<yaml::MachineFunction>();
  yaml::yamlize(In, *MF, false);
  if (In.error())
    return true;
  auto FunctionName = MF->Name;
  if (Functions.find(FunctionName) != Functions.end())
    return error(Twine("redefinition of machine function '") + FunctionName +
                 "'");
  Functions.insert(std::make_pair(FunctionName, std::move(MF)));
  if (NoLLVMIR)
    createDummyFunction(FunctionName, M);
  else if (!M.getFunction(FunctionName))
    return error(Twine("function '") + FunctionName +
                 "' isn't defined in the provided LLVM IR");
  return false;
}

void MIRParserImpl::createDummyFunction(StringRef Name, Module &M) {
  auto &Context = M.getContext();
  Function *F = cast<Function>(M.getOrInsertFunction(
      Name, FunctionType::get(Type::getVoidTy(Context), false)));
  BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
  new UnreachableInst(Context, BB);
}

bool MIRParserImpl::initializeMachineFunction(MachineFunction &MF) {
  auto It = Functions.find(MF.getName());
  if (It == Functions.end())
    return error(Twine("no machine function information for function '") +
                 MF.getName() + "' in the MIR file");
  // TODO: Recreate the machine function.
  const yaml::MachineFunction &YamlMF = *It->getValue();
  if (YamlMF.Alignment)
    MF.setAlignment(YamlMF.Alignment);
  MF.setExposesReturnsTwice(YamlMF.ExposesReturnsTwice);
  MF.setHasInlineAsm(YamlMF.HasInlineAsm);
  const auto &F = *MF.getFunction();
  for (const auto &YamlMBB : YamlMF.BasicBlocks) {
    const BasicBlock *BB = nullptr;
    if (!YamlMBB.Name.empty()) {
      BB = dyn_cast_or_null<BasicBlock>(
          F.getValueSymbolTable().lookup(YamlMBB.Name));
      if (!BB)
        return error(Twine("basic block '") + YamlMBB.Name +
                     "' is not defined in the function '" + MF.getName() + "'");
    }
    auto *MBB = MF.CreateMachineBasicBlock(BB);
    MF.insert(MF.end(), MBB);
    if (initializeMachineBasicBlock(*MBB, YamlMBB))
      return true;
  }
  return false;
}

bool MIRParserImpl::initializeMachineBasicBlock(
    MachineBasicBlock &MBB, const yaml::MachineBasicBlock &YamlMBB) {
  MBB.setAlignment(YamlMBB.Alignment);
  if (YamlMBB.AddressTaken)
    MBB.setHasAddressTaken();
  MBB.setIsLandingPad(YamlMBB.IsLandingPad);
  return false;
}

SMDiagnostic MIRParserImpl::diagFromLLVMAssemblyDiag(const SMDiagnostic &Error,
                                                     SMRange SourceRange) {
  assert(SourceRange.isValid());

  // Translate the location of the error from the location in the llvm IR string
  // to the corresponding location in the MIR file.
  auto LineAndColumn = SM.getLineAndColumn(SourceRange.Start);
  unsigned Line = LineAndColumn.first + Error.getLineNo() - 1;
  unsigned Column = Error.getColumnNo();
  StringRef LineStr = Error.getLineContents();
  SMLoc Loc = Error.getLoc();

  // Get the full line and adjust the column number by taking the indentation of
  // LLVM IR into account.
  for (line_iterator L(*SM.getMemoryBuffer(SM.getMainFileID()), false), E;
       L != E; ++L) {
    if (L.line_number() == Line) {
      LineStr = *L;
      Loc = SMLoc::getFromPointer(LineStr.data());
      auto Indent = LineStr.find(Error.getLineContents());
      if (Indent != StringRef::npos)
        Column += Indent;
      break;
    }
  }

  return SMDiagnostic(SM, Loc, Filename, Line, Column, Error.getKind(),
                      Error.getMessage(), LineStr, Error.getRanges(),
                      Error.getFixIts());
}

MIRParser::MIRParser(std::unique_ptr<MIRParserImpl> Impl)
    : Impl(std::move(Impl)) {}

MIRParser::~MIRParser() {}

std::unique_ptr<Module> MIRParser::parseLLVMModule() { return Impl->parse(); }

bool MIRParser::initializeMachineFunction(MachineFunction &MF) {
  return Impl->initializeMachineFunction(MF);
}

std::unique_ptr<MIRParser> llvm::createMIRParserFromFile(StringRef Filename,
                                                         SMDiagnostic &Error,
                                                         LLVMContext &Context) {
  auto FileOrErr = MemoryBuffer::getFile(Filename);
  if (std::error_code EC = FileOrErr.getError()) {
    Error = SMDiagnostic(Filename, SourceMgr::DK_Error,
                         "Could not open input file: " + EC.message());
    return nullptr;
  }
  return createMIRParser(std::move(FileOrErr.get()), Context);
}

std::unique_ptr<MIRParser>
llvm::createMIRParser(std::unique_ptr<MemoryBuffer> Contents,
                      LLVMContext &Context) {
  auto Filename = Contents->getBufferIdentifier();
  return llvm::make_unique<MIRParser>(
      llvm::make_unique<MIRParserImpl>(std::move(Contents), Filename, Context));
}
