//===-- ExecutionAnnotations.h ---------------------------------*- C++ -*--===//
//
// This header file defines annotations used by the execution engine.
//
//===----------------------------------------------------------------------===//

#ifndef LLI_EXECUTION_ANNOTATIONS_H
#define LLI_EXECUTION_ANNOTATIONS_H

//===----------------------------------------------------------------------===//
// Support for MethodInfo annotations
//===----------------------------------------------------------------------===//

// This annotation (attached only to Method objects) is used to cache useful
// information about the method, including the number of types present in the
// method, and the number of values for each type.
//
// This annotation object is created on demand, and attaches other annotation
// objects to the instructions in the method when it's created.
//
static AnnotationID MethodInfoAID(
	            AnnotationManager::getID("Interpreter::MethodInfo"));

struct MethodInfo : public Annotation {
  MethodInfo(Method *M);
  vector<unsigned> NumPlaneElements;

private:
  unsigned getValueSlot(const Value *V);
};

// CreateMethodInfo - Factory function to allow MethodInfo annotations to be
// created on demand.
//
inline static Annotation *CreateMethodInfo(AnnotationID AID, Annotable *O) {
  assert(AID == MethodInfoAID);
  return new MethodInfo((Method*)O);  // Simply invoke the ctor
}


//===----------------------------------------------------------------------===//
// Support for the SlotNumber annotation
//===----------------------------------------------------------------------===//

// This annotation (attached only to MethodArgument & Instruction objects) is
// used to hold the the slot number for the value in its type plane.
//
// Entities have this annotation attached to them when the containing
// method has it's MethodInfo created (by the MethodInfo ctor).
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
// Instructions have this annotation attached to them when the containing method
// has it's MethodInfo created (by the MethodInfo ctor).
//
struct InstNumber : public SlotNumber {
  unsigned InstNum;   // Ranges from 1->

  InstNumber(unsigned in, unsigned sn) : SlotNumber(sn), InstNum(in) {}
};


//===----------------------------------------------------------------------===//
// Support for the Breakpoint annotation
//===----------------------------------------------------------------------===//

static AnnotationID BreakpointAID(
	            AnnotationManager::getID("Interpreter::Breakpoint"));
// Just use an Annotation directly, Breakpoint is currently just a marker

#endif
