//===-- Assembler.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Assembler.h"

#include "SnippetRepetitor.h"
#include "Target.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace exegesis {

static constexpr const char ModuleID[] = "ExegesisInfoTest";
static constexpr const char FunctionID[] = "foo";
static const Align kFunctionAlignment(4096);

// Fills the given basic block with register setup code, and returns true if
// all registers could be setup correctly.
static bool generateSnippetSetupCode(
    const ExegesisTarget &ET, const MCSubtargetInfo *const MSI,
    ArrayRef<RegisterValue> RegisterInitialValues, BasicBlockFiller &BBF) {
  bool IsSnippetSetupComplete = true;
  for (const RegisterValue &RV : RegisterInitialValues) {
    // Load a constant in the register.
    const auto SetRegisterCode = ET.setRegTo(*MSI, RV.Register, RV.Value);
    if (SetRegisterCode.empty())
      IsSnippetSetupComplete = false;
    BBF.addInstructions(SetRegisterCode);
  }
  return IsSnippetSetupComplete;
}

// Small utility function to add named passes.
static bool addPass(PassManagerBase &PM, StringRef PassName,
                    TargetPassConfig &TPC) {
  const PassRegistry *PR = PassRegistry::getPassRegistry();
  const PassInfo *PI = PR->getPassInfo(PassName);
  if (!PI) {
    errs() << " run-pass " << PassName << " is not registered.\n";
    return true;
  }

  if (!PI->getNormalCtor()) {
    errs() << " cannot create pass: " << PI->getPassName() << "\n";
    return true;
  }
  Pass *P = PI->getNormalCtor()();
  std::string Banner = std::string("After ") + std::string(P->getPassName());
  PM.add(P);
  TPC.printAndVerify(Banner);

  return false;
}

MachineFunction &createVoidVoidPtrMachineFunction(StringRef FunctionName,
                                                  Module *Module,
                                                  MachineModuleInfo *MMI) {
  Type *const ReturnType = Type::getInt32Ty(Module->getContext());
  Type *const MemParamType = PointerType::get(
      Type::getInt8Ty(Module->getContext()), 0 /*default address space*/);
  FunctionType *FunctionType =
      FunctionType::get(ReturnType, {MemParamType}, false);
  Function *const F = Function::Create(
      FunctionType, GlobalValue::InternalLinkage, FunctionName, Module);
  // Making sure we can create a MachineFunction out of this Function even if it
  // contains no IR.
  F->setIsMaterializable(true);
  return MMI->getOrCreateMachineFunction(*F);
}

BasicBlockFiller::BasicBlockFiller(MachineFunction &MF, MachineBasicBlock *MBB,
                                   const MCInstrInfo *MCII)
    : MF(MF), MBB(MBB), MCII(MCII) {}

void BasicBlockFiller::addInstruction(const MCInst &Inst, const DebugLoc &DL) {
  const unsigned Opcode = Inst.getOpcode();
  const MCInstrDesc &MCID = MCII->get(Opcode);
  MachineInstrBuilder Builder = BuildMI(MBB, DL, MCID);
  for (unsigned OpIndex = 0, E = Inst.getNumOperands(); OpIndex < E;
       ++OpIndex) {
    const MCOperand &Op = Inst.getOperand(OpIndex);
    if (Op.isReg()) {
      const bool IsDef = OpIndex < MCID.getNumDefs();
      unsigned Flags = 0;
      const MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
      if (IsDef && !OpInfo.isOptionalDef())
        Flags |= RegState::Define;
      Builder.addReg(Op.getReg(), Flags);
    } else if (Op.isImm()) {
      Builder.addImm(Op.getImm());
    } else if (!Op.isValid()) {
      llvm_unreachable("Operand is not set");
    } else {
      llvm_unreachable("Not yet implemented");
    }
  }
}

void BasicBlockFiller::addInstructions(ArrayRef<MCInst> Insts,
                                       const DebugLoc &DL) {
  for (const MCInst &Inst : Insts)
    addInstruction(Inst, DL);
}

void BasicBlockFiller::addReturn(const DebugLoc &DL) {
  // Insert the return code.
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  if (TII->getReturnOpcode() < TII->getNumOpcodes()) {
    BuildMI(MBB, DL, TII->get(TII->getReturnOpcode()));
  } else {
    MachineIRBuilder MIB(MF);
    MIB.setMBB(*MBB);

    FunctionLoweringInfo FuncInfo;
    FuncInfo.CanLowerReturn = true;
    MF.getSubtarget().getCallLowering()->lowerReturn(MIB, nullptr, {},
                                                     FuncInfo);
  }
}

FunctionFiller::FunctionFiller(MachineFunction &MF,
                               std::vector<unsigned> RegistersSetUp)
    : MF(MF), MCII(MF.getTarget().getMCInstrInfo()), Entry(addBasicBlock()),
      RegistersSetUp(std::move(RegistersSetUp)) {}

BasicBlockFiller FunctionFiller::addBasicBlock() {
  MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
  MF.push_back(MBB);
  return BasicBlockFiller(MF, MBB, MCII);
}

ArrayRef<unsigned> FunctionFiller::getRegistersSetUp() const {
  return RegistersSetUp;
}

static std::unique_ptr<Module>
createModule(const std::unique_ptr<LLVMContext> &Context, const DataLayout &DL) {
  auto Mod = std::make_unique<Module>(ModuleID, *Context);
  Mod->setDataLayout(DL);
  return Mod;
}

BitVector getFunctionReservedRegs(const TargetMachine &TM) {
  std::unique_ptr<LLVMContext> Context = std::make_unique<LLVMContext>();
  std::unique_ptr<Module> Module = createModule(Context, TM.createDataLayout());
  // TODO: This only works for targets implementing LLVMTargetMachine.
  const LLVMTargetMachine &LLVMTM = static_cast<const LLVMTargetMachine &>(TM);
  std::unique_ptr<MachineModuleInfoWrapperPass> MMIWP =
      std::make_unique<MachineModuleInfoWrapperPass>(&LLVMTM);
  MachineFunction &MF = createVoidVoidPtrMachineFunction(
      FunctionID, Module.get(), &MMIWP.get()->getMMI());
  // Saving reserved registers for client.
  return MF.getSubtarget().getRegisterInfo()->getReservedRegs(MF);
}

Error assembleToStream(const ExegesisTarget &ET,
                       std::unique_ptr<LLVMTargetMachine> TM,
                       ArrayRef<unsigned> LiveIns,
                       ArrayRef<RegisterValue> RegisterInitialValues,
                       const FillFunction &Fill, raw_pwrite_stream &AsmStream) {
  auto Context = std::make_unique<LLVMContext>();
  std::unique_ptr<Module> Module =
      createModule(Context, TM->createDataLayout());
  auto MMIWP = std::make_unique<MachineModuleInfoWrapperPass>(TM.get());
  MachineFunction &MF = createVoidVoidPtrMachineFunction(
      FunctionID, Module.get(), &MMIWP.get()->getMMI());
  MF.ensureAlignment(kFunctionAlignment);

  // We need to instruct the passes that we're done with SSA and virtual
  // registers.
  auto &Properties = MF.getProperties();
  Properties.set(MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(MachineFunctionProperties::Property::IsSSA);
  Properties.set(MachineFunctionProperties::Property::NoPHIs);

  for (const unsigned Reg : LiveIns)
    MF.getRegInfo().addLiveIn(Reg);

  std::vector<unsigned> RegistersSetUp;
  for (const auto &InitValue : RegisterInitialValues) {
    RegistersSetUp.push_back(InitValue.Register);
  }
  FunctionFiller Sink(MF, std::move(RegistersSetUp));
  auto Entry = Sink.getEntry();
  for (const unsigned Reg : LiveIns)
    Entry.MBB->addLiveIn(Reg);

  const bool IsSnippetSetupComplete = generateSnippetSetupCode(
      ET, TM->getMCSubtargetInfo(), RegisterInitialValues, Entry);

  // If the snippet setup is not complete, we disable liveliness tracking. This
  // means that we won't know what values are in the registers.
  if (!IsSnippetSetupComplete)
    Properties.reset(MachineFunctionProperties::Property::TracksLiveness);

  Fill(Sink);

  // prologue/epilogue pass needs the reserved registers to be frozen, this
  // is usually done by the SelectionDAGISel pass.
  MF.getRegInfo().freezeReservedRegs(MF);

  // We create the pass manager, run the passes to populate AsmBuffer.
  MCContext &MCContext = MMIWP->getMMI().getContext();
  legacy::PassManager PM;

  TargetLibraryInfoImpl TLII(Triple(Module->getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  TargetPassConfig *TPC = TM->createPassConfig(PM);
  PM.add(TPC);
  PM.add(MMIWP.release());
  TPC->printAndVerify("MachineFunctionGenerator::assemble");
  // Add target-specific passes.
  ET.addTargetSpecificPasses(PM);
  TPC->printAndVerify("After ExegesisTarget::addTargetSpecificPasses");
  // Adding the following passes:
  // - postrapseudos: expands pseudo return instructions used on some targets.
  // - machineverifier: checks that the MachineFunction is well formed.
  // - prologepilog: saves and restore callee saved registers.
  for (const char *PassName :
       {"postrapseudos", "machineverifier", "prologepilog"})
    if (addPass(PM, PassName, *TPC))
      return make_error<Failure>("Unable to add a mandatory pass");
  TPC->setInitialized();

  // AsmPrinter is responsible for generating the assembly into AsmBuffer.
  if (TM->addAsmPrinter(PM, AsmStream, nullptr, CGFT_ObjectFile, MCContext))
    return make_error<Failure>("Cannot add AsmPrinter passes");

  PM.run(*Module); // Run all the passes
  return Error::success();
}

object::OwningBinary<object::ObjectFile>
getObjectFromBuffer(StringRef InputData) {
  // Storing the generated assembly into a MemoryBuffer that owns the memory.
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBufferCopy(InputData);
  // Create the ObjectFile from the MemoryBuffer.
  std::unique_ptr<object::ObjectFile> Obj =
      cantFail(object::ObjectFile::createObjectFile(Buffer->getMemBufferRef()));
  // Returning both the MemoryBuffer and the ObjectFile.
  return object::OwningBinary<object::ObjectFile>(std::move(Obj),
                                                  std::move(Buffer));
}

object::OwningBinary<object::ObjectFile> getObjectFromFile(StringRef Filename) {
  return cantFail(object::ObjectFile::createObjectFile(Filename));
}

namespace {

// Implementation of this class relies on the fact that a single object with a
// single function will be loaded into memory.
class TrackingSectionMemoryManager : public SectionMemoryManager {
public:
  explicit TrackingSectionMemoryManager(uintptr_t *CodeSize)
      : CodeSize(CodeSize) {}

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override {
    *CodeSize = Size;
    return SectionMemoryManager::allocateCodeSection(Size, Alignment, SectionID,
                                                     SectionName);
  }

private:
  uintptr_t *const CodeSize = nullptr;
};

} // namespace

ExecutableFunction::ExecutableFunction(
    std::unique_ptr<LLVMTargetMachine> TM,
    object::OwningBinary<object::ObjectFile> &&ObjectFileHolder)
    : Context(std::make_unique<LLVMContext>()) {
  assert(ObjectFileHolder.getBinary() && "cannot create object file");
  // Initializing the execution engine.
  // We need to use the JIT EngineKind to be able to add an object file.
  LLVMLinkInMCJIT();
  uintptr_t CodeSize = 0;
  std::string Error;
  ExecEngine.reset(
      EngineBuilder(createModule(Context, TM->createDataLayout()))
          .setErrorStr(&Error)
          .setMCPU(TM->getTargetCPU())
          .setEngineKind(EngineKind::JIT)
          .setMCJITMemoryManager(
              std::make_unique<TrackingSectionMemoryManager>(&CodeSize))
          .create(TM.release()));
  if (!ExecEngine)
    report_fatal_error(Twine(Error));
  // Adding the generated object file containing the assembled function.
  // The ExecutionEngine makes sure the object file is copied into an
  // executable page.
  ExecEngine->addObjectFile(std::move(ObjectFileHolder));
  // Fetching function bytes.
  const uint64_t FunctionAddress = ExecEngine->getFunctionAddress(FunctionID);
  assert(isAligned(kFunctionAlignment, FunctionAddress) &&
         "function is not properly aligned");
  FunctionBytes =
      StringRef(reinterpret_cast<const char *>(FunctionAddress), CodeSize);
}

} // namespace exegesis
} // namespace llvm
