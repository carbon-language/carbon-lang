//===-- UserInput.cpp - Interpreter Input Loop support --------------------===//
// 
//  This file implements the interpreter Input I/O loop.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Assembly/Writer.h"
#include <algorithm>

enum CommandID {
  Quit, Help,                                 // Basics
  Print, Info, List, StackTrace, Up, Down,    // Inspection
  Next, Step, Run, Finish, Call,              // Control flow changes
  Break, Watch,                               // Debugging
  Load, Flush
};

// CommandTable - Build a lookup table for the commands available to the user...
static struct CommandTableElement {
  const char *Name;
  enum CommandID CID;

  inline bool operator<(const CommandTableElement &E) const {
    return string(Name) < string(E.Name);
  }
  inline bool operator==(const string &S) const { 
    return string(Name) == S;
  }
} CommandTable[] = {
  { "quit"     , Quit       }, { "q", Quit }, { "", Quit }, // Empty str = eof
  { "help"     , Help       }, { "h", Help },

  { "print"    , Print      }, { "p", Print },
  { "list"     , List       },
  { "info"     , Info       },
  { "backtrace", StackTrace }, { "bt", StackTrace }, { "where", StackTrace },
  { "up"       , Up         },
  { "down"     , Down       },

  { "next"     , Next       }, { "n", Next },
  { "step"     , Step       }, { "s", Step },
  { "run"      , Run        },
  { "finish"   , Finish     },
  { "call"     , Call       },

  { "break"    , Break      }, { "b", Break },
  { "watch"    , Watch      },

  { "load"     , Load       },
  { "flush"    , Flush      },
};
static CommandTableElement *CommandTableEnd = 
   CommandTable+sizeof(CommandTable)/sizeof(CommandTable[0]);


//===----------------------------------------------------------------------===//
// handleUserInput - Enter the input loop for the interpreter.  This function
// returns when the user quits the interpreter.
//
void Interpreter::handleUserInput() {
  bool UserQuit = false;

  // Sort the table...
  sort(CommandTable, CommandTableEnd);

  // Print the instruction that we are stopped at...
  printCurrentInstruction();

  do {
    string Command;
    cout << "lli> " << flush;
    cin >> Command;

    CommandTableElement *E = find(CommandTable, CommandTableEnd, Command);

    if (E == CommandTableEnd) {
      cout << "Error: '" << Command << "' not recognized!\n";
      continue;
    }

    switch (E->CID) {
    case Quit:       UserQuit = true;   break;
    case Load:
      cin >> Command;
      loadModule(Command);
      break;
    case Flush: flushModule(); break;
    case Print:
      cin >> Command;
      print(Command);
      break;
    case Info:
      cin >> Command;
      infoValue(Command);
      break;
     
    case List:       list();            break;
    case StackTrace: printStackTrace(); break;
    case Up: 
      if (CurFrame > 0) --CurFrame;
      else cout << "Error: Already at root of stack!\n";
      break;
    case Down:
      if ((unsigned)CurFrame < ECStack.size()-1) ++CurFrame;
      else cout << "Error: Already at bottom of stack!\n";
      break;
    case Next:       nextInstruction(); break;
    case Step:       stepInstruction(); break;
    case Run:        run();             break;
    case Finish:     finish();          break;
    case Call:
      cin >> Command;
      callMethod(Command);    // Enter the specified method
      finish();               // Run until it's complete
      break;

    default:
      cout << "Command '" << Command << "' unimplemented!\n";
      break;
    }

  } while (!UserQuit);
}

//===----------------------------------------------------------------------===//
// loadModule - Load a new module to execute...
//
void Interpreter::loadModule(const string &Filename) {
  if (CurMod && !flushModule()) return;  // Kill current execution

  CurMod = ParseBytecodeFile(Filename);
  if (CurMod == 0) {
    cout << "Error parsing '" << Filename << "': No module loaded.\n";
    return;
  }

  // TODO: link in support library...
}


//===----------------------------------------------------------------------===//
// flushModule - Return true if the current program has been unloaded.
//
bool Interpreter::flushModule() {
  if (CurMod == 0) {
    cout << "Error flushing: No module loaded!\n";
    return false;
  }

  if (!ECStack.empty()) {
    // TODO: if use is not sure, return false
    cout << "Killing current execution!\n";
    ECStack.clear();
    CurFrame = -1;
  }

  delete CurMod;
  CurMod = 0;
  ExitCode = 0;
  return true;
}

//===----------------------------------------------------------------------===//
// setBreakpoint - Enable a breakpoint at the specified location
//
void Interpreter::setBreakpoint(const string &Name) {
  Value *PickedVal = ChooseOneOption(Name, LookupMatchingNames(Name));
  // TODO: Set a breakpoint on PickedVal
}

//===----------------------------------------------------------------------===//
// callMethod - Enter the specified method...
//
bool Interpreter::callMethod(const string &Name) {
  vector<Value*> Options = LookupMatchingNames(Name);

  for (unsigned i = 0; i < Options.size(); ++i) { // Remove nonmethod matches...
    if (!isa<Method>(Options[i])) {
      Options.erase(Options.begin()+i);
      --i;
    }
  }

  Value *PickedMeth = ChooseOneOption(Name, Options);
  if (PickedMeth == 0)
    return true;

  Method *M = cast<Method>(PickedMeth);

  vector<GenericValue> Args;
  // TODO, get args from user...

  callMethod(M, Args);  // Start executing it...

  // Reset the current frame location to the top of stack
  CurFrame = ECStack.size()-1;

  return false;
}
