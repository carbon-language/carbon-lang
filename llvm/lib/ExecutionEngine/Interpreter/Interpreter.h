//===-- Interpreter.h ------------------------------------------*- C++ -*--===//
//
// This header file defines the interpreter structure
//
//===----------------------------------------------------------------------===//

#ifndef LLI_INTERPRETER_H
#define LLI_INTERPRETER_H

#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/Support/DataTypes.h"

struct MethodInfo;          // Defined in ExecutionAnnotations.h
class CallInst;
class ReturnInst;
class BranchInst;
class AllocationInst;

union GenericValue {
  bool            BoolVal;
  unsigned char   UByteVal;
  signed   char   SByteVal;
  unsigned short  UShortVal;
  signed   short  ShortVal;
  unsigned int    UIntVal;
  signed   int    IntVal;
  uint64_t        ULongVal;
  int64_t         LongVal;
  double          DoubleVal;
  float           FloatVal;
  GenericValue *PointerVal;
};

typedef vector<GenericValue> ValuePlaneTy;

// ExecutionContext struct - This struct represents one stack frame currently
// executing.
//
struct ExecutionContext {
  Method               *CurMethod;  // The currently executing method
  BasicBlock           *CurBB;      // The currently executing BB
  BasicBlock::iterator  CurInst;    // The next instruction to execute
  MethodInfo           *MethInfo;   // The MethInfo annotation for the method
  vector<ValuePlaneTy>  Values;     // ValuePlanes for each type

  BasicBlock           *PrevBB;     // The previous BB or null if in first BB
  CallInst             *Caller;     // Holds the call that called subframes.
                                    // NULL if main func or debugger invoked fn
};


// Interpreter - This class represents the entirety of the interpreter.
//
class Interpreter {
  Module *CurMod;              // The current Module being executed (0 if none)
  int ExitCode;                // The exit code to be returned by the lli util
  bool Profile;                // Profiling enabled?
  int CurFrame;                // The current stack frame being inspected

  // The runtime stack of executing code.  The top of the stack is the current
  // method record.
  vector<ExecutionContext> ECStack;

public:
  Interpreter();
  inline ~Interpreter() { delete CurMod; }

  // getExitCode - return the code that should be the exit code for the lli
  // utility.
  inline int getExitCode() const { return ExitCode; }

  // enableProfiling() - Turn profiling on, clear stats?
  void enableProfiling() { Profile = true; }

  void initializeExecutionEngine();
  void handleUserInput();

  // User Interation Methods...
  void loadModule(const string &Filename);
  bool flushModule();
  bool callMethod(const string &Name);      // return true on failure
  void setBreakpoint(const string &Name);
  void infoValue(const string &Name);
  void print(const string &Name);
  static void print(const Type *Ty, GenericValue V);
  static void printValue(const Type *Ty, GenericValue V);

  // Hack until we can parse command line args...
  bool callMainMethod(const string &MainName,
                      const vector<string> &InputFilename);

  void list();             // Do the 'list' command
  void printStackTrace();  // Do the 'backtrace' command

  // Code execution methods...
  void callMethod        (Method *Meth, const vector<GenericValue> &ArgVals);
  void callExternalMethod(Method *Meth, const vector<GenericValue> &ArgVals);
  bool executeInstruction(); // Execute one instruction...

  void stepInstruction();  // Do the 'step' command
  void nextInstruction();  // Do the 'next' command
  void run();              // Do the 'run' command
  void finish();           // Do the 'finish' command

  // Opcode Implementations
  void executeCallInst(CallInst *I, ExecutionContext &SF);
  void executeRetInst(ReturnInst *I, ExecutionContext &SF);
  void executeBrInst(BranchInst *I, ExecutionContext &SF);
  void executeAllocInst(AllocationInst *I, ExecutionContext &SF);
  void exitCalled(GenericValue GV);

  // getCurrentMethod - Return the currently executing method
  inline Method *getCurrentMethod() const {
    return CurFrame < 0 ? 0 : ECStack[CurFrame].CurMethod;
  }

  // isStopped - Return true if a program is stopped.  Return false if no
  // program is running.
  //
  inline bool isStopped() const { return !ECStack.empty(); }

private:  // Helper functions
  // getCurrentExecutablePath() - Return the directory that the lli executable
  // lives in.
  //
  string getCurrentExecutablePath() const;

  // printCurrentInstruction - Print out the instruction that the virtual PC is
  // at, or fail silently if no program is running.
  //
  void printCurrentInstruction();

  // LookupMatchingNames - Search the current method namespace, then the global
  // namespace looking for values that match the specified name.  Return ALL
  // matches to that name.  This is obviously slow, and should only be used for
  // user interaction.
  //
  vector<Value*> LookupMatchingNames(const string &Name);

  // ChooseOneOption - Prompt the user to choose among the specified options to
  // pick one value.  If no options are provided, emit an error.  If a single 
  // option is provided, just return that option.
  //
  Value *ChooseOneOption(const string &Name, const vector<Value*> &Opts);
};

#endif
