//===-- MipsConstantIslandPass.cpp - Emit Pc Relative loads----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
// This pass is used to make Pc relative loads of constants.
// For now, only Mips16 will use this. While it has the same name and
// uses many ideas from the LLVM ARM Constant Island Pass, it's not intended
// to reuse any of the code from the ARM version.
//
// Loading constants inline is expensive on Mips16 and it's in general better
// to place the constant nearby in code space and then it can be loaded with a
// simple 16 bit load instruction.
//
// The constants can be not just numbers but addresses of functions and labels.
// This can be particularly helpful in static relocation mode for embedded
// non linux targets.
//
//

#define DEBUG_TYPE "mips-constant-islands"

#include "Mips.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <algorithm>

using namespace llvm;

STATISTIC(NumCPEs,       "Number of constpool entries");

// FIXME: This option should be removed once it has received sufficient testing.
static cl::opt<bool>
AlignConstantIslands("mips-align-constant-islands", cl::Hidden, cl::init(true),
          cl::desc("Align constant islands in code"));

namespace {
  typedef MachineBasicBlock::iterator Iter;
  typedef MachineBasicBlock::reverse_iterator ReverseIter;

  class MipsConstantIslands : public MachineFunctionPass {

  const TargetMachine &TM;
  bool IsPIC;
  unsigned ABI;
  const MipsSubtarget *STI;
  const MipsInstrInfo *TII;
  MachineFunction *MF;
  MachineConstantPool *MCP;

  /// CPEntry - One per constant pool entry, keeping the machine instruction
  /// pointer, the constpool index, and the number of CPUser's which
  /// reference this entry.
  struct CPEntry {
    MachineInstr *CPEMI;
    unsigned CPI;
    unsigned RefCount;
    CPEntry(MachineInstr *cpemi, unsigned cpi, unsigned rc = 0)
      : CPEMI(cpemi), CPI(cpi), RefCount(rc) {}
  };

  /// CPEntries - Keep track of all of the constant pool entry machine
  /// instructions. For each original constpool index (i.e. those that
  /// existed upon entry to this pass), it keeps a vector of entries.
  /// Original elements are cloned as we go along; the clones are
  /// put in the vector of the original element, but have distinct CPIs.
  std::vector<std::vector<CPEntry> > CPEntries;

  public:
    static char ID;
    MipsConstantIslands(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm),
        IsPIC(TM.getRelocationModel() == Reloc::PIC_),
        ABI(TM.getSubtarget<MipsSubtarget>().getTargetABI()),
        STI(&TM.getSubtarget<MipsSubtarget>()), MF(0), MCP(0){}

    virtual const char *getPassName() const {
      return "Mips Constant Islands";
    }

    bool runOnMachineFunction(MachineFunction &F);

    void doInitialPlacement(std::vector<MachineInstr*> &CPEMIs);

    void prescanForConstants();

  private:

  };

  char MipsConstantIslands::ID = 0;
} // end of anonymous namespace

/// createMipsLongBranchPass - Returns a pass that converts branches to long
/// branches.
FunctionPass *llvm::createMipsConstantIslandPass(MipsTargetMachine &tm) {
  return new MipsConstantIslands(tm);
}

bool MipsConstantIslands::runOnMachineFunction(MachineFunction &mf) {
  // The intention is for this to be a mips16 only pass for now
  // FIXME:
  MF = &mf;
  MCP = mf.getConstantPool();
  DEBUG(dbgs() << "constant island machine function " << "\n");
  if (!TM.getSubtarget<MipsSubtarget>().inMips16Mode() ||
      !MipsSubtarget::useConstantIslands()) {
    return false;
  }
  TII = (const MipsInstrInfo*)MF->getTarget().getInstrInfo();
  DEBUG(dbgs() << "constant island processing " << "\n");
  //
  // will need to make predermination if there is any constants we need to
  // put in constant islands. TBD.
  //
  prescanForConstants();

  // This pass invalidates liveness information when it splits basic blocks.
  MF->getRegInfo().invalidateLiveness();

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  MF->RenumberBlocks();

  // Perform the initial placement of the constant pool entries.  To start with,
  // we put them all at the end of the function.
  std::vector<MachineInstr*> CPEMIs;
  if (!MCP->isEmpty())
    doInitialPlacement(CPEMIs);

  return true;
}

/// doInitialPlacement - Perform the initial placement of the constant pool
/// entries.  To start with, we put them all at the end of the function.
void
MipsConstantIslands::doInitialPlacement(std::vector<MachineInstr*> &CPEMIs) {
  // Create the basic block to hold the CPE's.
  MachineBasicBlock *BB = MF->CreateMachineBasicBlock();
  MF->push_back(BB);


  // MachineConstantPool measures alignment in bytes. We measure in log2(bytes).
  unsigned MaxAlign = Log2_32(MCP->getConstantPoolAlignment());

  // Mark the basic block as required by the const-pool.
  // If AlignConstantIslands isn't set, use 4-byte alignment for everything.
  BB->setAlignment(AlignConstantIslands ? MaxAlign : 2);

  // The function needs to be as aligned as the basic blocks. The linker may
  // move functions around based on their alignment.
  MF->ensureAlignment(BB->getAlignment());

  // Order the entries in BB by descending alignment.  That ensures correct
  // alignment of all entries as long as BB is sufficiently aligned.  Keep
  // track of the insertion point for each alignment.  We are going to bucket
  // sort the entries as they are created.
  SmallVector<MachineBasicBlock::iterator, 8> InsPoint(MaxAlign + 1, BB->end());

  // Add all of the constants from the constant pool to the end block, use an
  // identity mapping of CPI's to CPE's.
  const std::vector<MachineConstantPoolEntry> &CPs = MCP->getConstants();

  const DataLayout &TD = *MF->getTarget().getDataLayout();
  for (unsigned i = 0, e = CPs.size(); i != e; ++i) {
    unsigned Size = TD.getTypeAllocSize(CPs[i].getType());
    assert(Size >= 4 && "Too small constant pool entry");
    unsigned Align = CPs[i].getAlignment();
    assert(isPowerOf2_32(Align) && "Invalid alignment");
    // Verify that all constant pool entries are a multiple of their alignment.
    // If not, we would have to pad them out so that instructions stay aligned.
    assert((Size % Align) == 0 && "CP Entry not multiple of 4 bytes!");

    // Insert CONSTPOOL_ENTRY before entries with a smaller alignment.
    unsigned LogAlign = Log2_32(Align);
    MachineBasicBlock::iterator InsAt = InsPoint[LogAlign];

    MachineInstr *CPEMI =
      BuildMI(*BB, InsAt, DebugLoc(), TII->get(Mips::CONSTPOOL_ENTRY))
        .addImm(i).addConstantPoolIndex(i).addImm(Size);

    CPEMIs.push_back(CPEMI);

    // Ensure that future entries with higher alignment get inserted before
    // CPEMI. This is bucket sort with iterators.
    for (unsigned a = LogAlign + 1; a <= MaxAlign; ++a)
      if (InsPoint[a] == InsAt)
        InsPoint[a] = CPEMI;
    // Add a new CPEntry, but no corresponding CPUser yet.
    std::vector<CPEntry> CPEs;
    CPEs.push_back(CPEntry(CPEMI, i));
    CPEntries.push_back(CPEs);
    ++NumCPEs;
    DEBUG(dbgs() << "Moved CPI#" << i << " to end of function, size = "
                 << Size << ", align = " << Align <<'\n');
  }
  DEBUG(BB->dump());
}


void MipsConstantIslands::prescanForConstants() {
  unsigned int J;
  for (MachineFunction::iterator B =
         MF->begin(), E = MF->end(); B != E; ++B) {
    for (MachineBasicBlock::instr_iterator I =
        B->instr_begin(), EB = B->instr_end(); I != EB; ++I) {
      switch(I->getDesc().getOpcode()) {
        case Mips::LwConstant32: {
          DEBUG(dbgs() << "constant island constant " << *I << "\n");
          J = I->getNumOperands();
          DEBUG(dbgs() << "num operands " << J  << "\n");
          MachineOperand& Literal = I->getOperand(1);
          if (Literal.isImm()) {
            int64_t V = Literal.getImm();
            DEBUG(dbgs() << "literal " << V  << "\n");
            Type *Int32Ty =
              Type::getInt32Ty(MF->getFunction()->getContext());
            const Constant *C = ConstantInt::get(Int32Ty, V);
            unsigned index = MCP->getConstantPoolIndex(C, 4);
            I->getOperand(2).ChangeToImmediate(index);
            DEBUG(dbgs() << "constant island constant " << *I << "\n");
            I->setDesc(TII->get(Mips::LwRxPcTcpX16));
            I->RemoveOperand(1);
            I->RemoveOperand(1);
            I->addOperand(MachineOperand::CreateCPI(index, 0));
          }
          break;
        }
        default:
          break;
      }
    }
  }
}
