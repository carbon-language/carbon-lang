//===-- Interpreter.h ------------------------------------------*- C++ -*--===//
//
// This header file defines the interpreter structure
//
//===----------------------------------------------------------------------===//

#ifndef LLI_INTERPRETER_H
#define LLI_INTERPRETER_H

#include "../ExecutionEngine.h"
#include "../GenericValue.h"
#include "Support/DataTypes.h"
#include "llvm/Assembly/CachedWriter.h"
#include "llvm/Target/TargetData.h"
#include "llvm/BasicBlock.h"
#include "llvm/Support/InstVisitor.h"

extern CachedWriter CW;     // Object to accelerate printing of LLVM

struct FunctionInfo;        // Defined in ExecutionAnnotations.h

// AllocaHolder - Object to track all of the blocks of memory allocated by
// alloca.  When the function returns, this object is poped off the execution
// stack, which causes the dtor to be run, which frees all the alloca'd memory.
//
class AllocaHolder {
  friend class AllocaHolderHandle;
  std::vector<void*> Allocations;
  unsigned RefCnt;
public:
  AllocaHolder() : RefCnt(0) {}
  void add(void *mem) { Allocations.push_back(mem); }
  ~AllocaHolder() {
    for (unsigned i = 0; i < Allocations.size(); ++i)
      free(Allocations[i]);
  }
};

// AllocaHolderHandle gives AllocaHolder value semantics so we can stick it into
// a vector...
//
class AllocaHolderHandle {
  AllocaHolder *H;
public:
  AllocaHolderHandle() : H(new AllocaHolder()) { H->RefCnt++; }
  AllocaHolderHandle(const AllocaHolderHandle &AH) : H(AH.H) { H->RefCnt++; }
  ~AllocaHolderHandle() { if (--H->RefCnt == 0) delete H; }

  void add(void *mem) { H->add(mem); }
};

typedef std::vector<GenericValue> ValuePlaneTy;

// ExecutionContext struct - This struct represents one stack frame currently
// executing.
//
struct ExecutionContext {
  Function             *CurFunction;// The currently executing function
  BasicBlock           *CurBB;      // The currently executing BB
  BasicBlock::iterator  CurInst;    // The next instruction to execute
  FunctionInfo         *FuncInfo;   // The FuncInfo annotation for the function
  std::vector<ValuePlaneTy>  Values;// ValuePlanes for each type
  std::vector<GenericValue>  VarArgs; // Values passed through an ellipsis

  CallInst             *Caller;     // Holds the call that called subframes.
                                    // NULL if main func or debugger invoked fn
  AllocaHolderHandle    Allocas;    // Track memory allocated by alloca
};

// Interpreter - This class represents the entirety of the interpreter.
//
class Interpreter : public ExecutionEngine, public InstVisitor<Interpreter> {
  int ExitCode;                // The exit code to be returned by the lli util
  bool Profile;                // Profiling enabled?
  bool Trace;                  // Tracing enabled?
  int CurFrame;                // The current stack frame being inspected
  TargetData TD;

  // The runtime stack of executing code.  The top of the stack is the current
  // function record.
  std::vector<ExecutionContext> ECStack;

  // AtExitHandlers - List of functions to call when the program exits.
  std::vector<Function*> AtExitHandlers;
public:
  Interpreter(Module *M, bool isLittleEndian, bool isLongPointer,
              bool TraceMode);
  inline ~Interpreter() { CW.setModule(0); }

  /// create - Create an interpreter ExecutionEngine. This can never fail.
  ///
  static ExecutionEngine *create(Module *M, bool TraceMode);

  /// getExitCode - return the code that should be the exit code for the lli
  /// utility.
  ///
  inline int getExitCode() const { return ExitCode; }

  /// run - Start execution with the specified function and arguments.
  ///
  virtual int run(const std::string &FnName,
		  const std::vector<std::string> &Args,
                  const char ** envp);
 

  // enableProfiling() - Turn profiling on, clear stats?
  void enableProfiling() { Profile = true; }
  void enableTracing() { Trace = true; }

  void handleUserInput();

  // User Interation Methods...
  bool callFunction(const std::string &Name);      // return true on failure
  void infoValue(const std::string &Name);
  void print(const std::string &Name);
  static void print(const Type *Ty, GenericValue V);
  static void printValue(const Type *Ty, GenericValue V);

  bool callMainFunction(const std::string &MainName,
                        const std::vector<std::string> &InputFilename);

  void list();             // Do the 'list' command
  void printStackTrace();  // Do the 'backtrace' command

  // Code execution methods...
  void callFunction(Function *F, const std::vector<GenericValue> &ArgVals);
  void executeInstruction(); // Execute one instruction...

  void stepInstruction();  // Do the 'step' command
  void nextInstruction();  // Do the 'next' command
  void run();              // Do the 'run' command
  void finish();           // Do the 'finish' command

  // Opcode Implementations
  void visitReturnInst(ReturnInst &I);
  void visitBranchInst(BranchInst &I);
  void visitSwitchInst(SwitchInst &I);

  void visitBinaryOperator(BinaryOperator &I);
  void visitAllocationInst(AllocationInst &I);
  void visitFreeInst(FreeInst &I);
  void visitLoadInst(LoadInst &I);
  void visitStoreInst(StoreInst &I);
  void visitGetElementPtrInst(GetElementPtrInst &I);

  void visitPHINode(PHINode &PN) { assert(0 && "PHI nodes already handled!"); }
  void visitCastInst(CastInst &I);
  void visitCallInst(CallInst &I);
  void visitShl(ShiftInst &I);
  void visitShr(ShiftInst &I);
  void visitVarArgInst(VarArgInst &I);
  void visitInstruction(Instruction &I) {
    std::cerr << I;
    assert(0 && "Instruction not interpretable yet!");
  }

  GenericValue callExternalFunction(Function *F, 
                                    const std::vector<GenericValue> &ArgVals);
  void exitCalled(GenericValue GV);

  // getCurrentFunction - Return the currently executing function
  inline Function *getCurrentFunction() const {
    return CurFrame < 0 ? 0 : ECStack[CurFrame].CurFunction;
  }

  // isStopped - Return true if a program is stopped.  Return false if no
  // program is running.
  //
  inline bool isStopped() const { return !ECStack.empty(); }

  void addAtExitHandler(Function *F) {
    AtExitHandlers.push_back(F);
  }

  //FIXME: private:
public:
  GenericValue executeGEPOperation(Value *Ptr, User::op_iterator I,
				   User::op_iterator E, ExecutionContext &SF);

private:  // Helper functions
  // SwitchToNewBasicBlock - Start execution in a new basic block and run any
  // PHI nodes in the top of the block.  This is used for intraprocedural
  // control flow.
  // 
  void SwitchToNewBasicBlock(BasicBlock *Dest, ExecutionContext &SF);

  void *getPointerToFunction(Function *F) { return (void*)F; }

  // getCurrentExecutablePath() - Return the directory that the lli executable
  // lives in.
  //
  std::string getCurrentExecutablePath() const;

  // printCurrentInstruction - Print out the instruction that the virtual PC is
  // at, or fail silently if no program is running.
  //
  void printCurrentInstruction();

  // printStackFrame - Print information about the specified stack frame, or -1
  // for the default one.
  //
  void printStackFrame(int FrameNo = -1);

  // LookupMatchingNames - Search the current function namespace, then the
  // global namespace looking for values that match the specified name.  Return
  // ALL matches to that name.  This is obviously slow, and should only be used
  // for user interaction.
  //
  std::vector<Value*> LookupMatchingNames(const std::string &Name);

  // ChooseOneOption - Prompt the user to choose among the specified options to
  // pick one value.  If no options are provided, emit an error.  If a single 
  // option is provided, just return that option.
  //
  Value *ChooseOneOption(const std::string &Name,
                         const std::vector<Value*> &Opts);

  void initializeExecutionEngine();
  void initializeExternalFunctions();
};

#endif
