//===-- ExecutionAnnotations.h ---------------------------------*- C++ -*--===//
//
// This header file defines annotations used by the execution engine.
//
//===----------------------------------------------------------------------===//

#ifndef LLI_EXECUTION_ANNOTATIONS_H
#define LLI_EXECUTION_ANNOTATIONS_H

//===----------------------------------------------------------------------===//
// Support for FunctionInfo annotations
//===----------------------------------------------------------------------===//

// This annotation (attached only to Function objects) is used to cache useful
// information about the function, including the number of types present in the
// function, and the number of values for each type.
//
// This annotation object is created on demand, and attaches other annotation
// objects to the instructions in the function when it's created.
//
static AnnotationID FunctionInfoAID(
	            AnnotationManager::getID("Interpreter::FunctionInfo"));

struct FunctionInfo : public Annotation {
  FunctionInfo(Function *F);
  std::vector<unsigned> NumPlaneElements;

  // Create - Factory function to allow FunctionInfo annotations to be
  // created on demand.
  //
  static Annotation *Create(AnnotationID AID, const Annotable *O, void *) {
    assert(AID == FunctionInfoAID);
    return new FunctionInfo(cast<Function>((Value*)O));
  }

private:
  unsigned getValueSlot(const Value *V);
};

//===----------------------------------------------------------------------===//
// Support for the SlotNumber annotation
//===----------------------------------------------------------------------===//

// This annotation (attached only to Argument & Instruction objects) is used to
// hold the the slot number for the value in its type plane.
//
// Entities have this annotation attached to them when the containing
// function has it's FunctionInfo created (by the FunctionInfo ctor).
//
static AnnotationID SlotNumberAID(
	            AnnotationManager::getID("Interpreter::SlotNumber"));

struct SlotNumber : public Annotation {
  unsigned SlotNum;   // Ranges from 0->

  SlotNumber(unsigned sn) : Annotation(SlotNumberAID), 
			    SlotNum(sn) {}
};

//===----------------------------------------------------------------------===//
// Support for the InstNumber annotation
//===----------------------------------------------------------------------===//

// This annotation (attached only to Instruction objects) is used to hold the
// instruction number of the instruction, and the slot number for the value in
// its type plane.  InstNumber's are used for user interaction, and for
// calculating which value slot to store the result of the instruction in.
//
// Instructions have this annotation attached to them when the containing
// function has it's FunctionInfo created (by the FunctionInfo ctor).
//
struct InstNumber : public SlotNumber {
  unsigned InstNum;   // Ranges from 1->

  InstNumber(unsigned in, unsigned sn) : SlotNumber(sn), InstNum(in) {}
};

#endif
