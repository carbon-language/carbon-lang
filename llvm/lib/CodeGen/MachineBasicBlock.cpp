//===-- MachineCodeForBasicBlock.cpp --------------------------------------===//
// 
// Collect the sequence of machine instructions for a basic block.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineCodeForBasicBlock.h"

AnnotationID MCFBB_AID(
             AnnotationManager::getID("CodeGen::MachineBasicBlock"));

static Annotation *CreateMCFBB(AnnotationID AID, const Annotable *, void *) {
  assert(AID == MCFBB_AID);
  return new MachineBasicBlock();  // Invoke constructor!
}

// Register the annotation with the annotation factory
static struct MCFBBInitializer {
  MCFBBInitializer() {
    AnnotationManager::registerAnnotationFactory(MCFBB_AID, &CreateMCFBB);
  }
} RegisterCreateMCFBB;

