#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "SparcInternals.h"
#include "SparcV9CodeEmitter.h"

bool UltraSparc::addPassesToEmitMachineCode(PassManager &PM,
                                            MachineCodeEmitter &MCE) {
  //PM.add(new SparcV9CodeEmitter(MCE));
  //MachineCodeEmitter *M = MachineCodeEmitter::createDebugMachineCodeEmitter();
  MachineCodeEmitter *M = 
    MachineCodeEmitter::createFilePrinterMachineCodeEmitter(MCE);
  PM.add(new SparcV9CodeEmitter(*M));
  return false;
}

void SparcV9CodeEmitter::emitConstant(unsigned Val, unsigned Size) {
  // Output the constant in big endian byte order...
  unsigned byteVal;
  for (int i = Size-1; i >= 0; --i) {
    byteVal = Val >> 8*i;
    MCE.emitByte(byteVal & 255);
  }
}

int64_t SparcV9CodeEmitter::getMachineOpValue(MachineOperand &MO) {
  if (MO.isPhysicalRegister()) {
    return MO.getReg();
  } else if (MO.isImmediate()) {
    return MO.getImmedValue();
  } else if (MO.isPCRelativeDisp()) {
    // FIXME!!!
    //return MO.getPCRelativeDisp();
    return 0;
  } else {
    assert(0 && "Unknown type of MachineOperand");
    return 0;
  }
}

unsigned SparcV9CodeEmitter::getValueBit(int64_t Val, unsigned bit) {
  Val >>= bit;
  return (Val & 1);
}


bool SparcV9CodeEmitter::runOnMachineFunction(MachineFunction &MF) {
  MCE.startFunction(MF);
  MCE.emitConstantPool(MF.getConstantPool());
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    emitBasicBlock(*I);
  MCE.finishFunction(MF);
  return false;
}

void SparcV9CodeEmitter::emitBasicBlock(MachineBasicBlock &MBB) {
  MCE.startBasicBlock(MBB);
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ++I)
    emitInstruction(**I);
}

void SparcV9CodeEmitter::emitInstruction(MachineInstr &MI) {
  emitConstant(getBinaryCodeForInstr(MI), 4);
}

#include "SparcV9CodeEmitter.inc"
