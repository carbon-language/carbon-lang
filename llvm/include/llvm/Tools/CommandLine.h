//===-- llvm/Tools/CommandLine.h - Command line parser for tools -*- C++ -*--=//
//
// This class implements a command line argument processor that is useful when
// creating a tool.
//
// This class is defined entirely inline so that you don't have to link to any
// libraries to use this.
//
// TODO: make this extensible by passing in arguments to be read.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_COMMANDLINE_H
#define LLVM_TOOLS_COMMANDLINE_H

#include <string>

class ToolCommandLine {
public:
  inline ToolCommandLine(int &argc, char **argv, bool OutputBytecode = true);
  inline ToolCommandLine(const string &infn, const string &outfn = "-");
  inline ToolCommandLine(const ToolCommandLine &O);
  inline ToolCommandLine &operator=(const ToolCommandLine &O);

  inline bool getForce() const { return Force; }
  inline const string getInputFilename() const { return InputFilename; }
  inline const string getOutputFilename() const { return OutputFilename; }

private:
  void calculateOutputFilename(bool OutputBytecode) {
    OutputFilename = InputFilename;
    unsigned Len = OutputFilename.length();

    if (Len <= 3) { 
      OutputFilename += (OutputBytecode ? ".bc" : ".ll"); 
      return; 
    }

    if (OutputBytecode) {
      if (OutputFilename[Len-3] == '.' &&
	  OutputFilename[Len-2] == 'l' &&
	  OutputFilename[Len-1] == 'l') {   // .ll -> .bc	  
	OutputFilename[Len-2] = 'b'; 
	OutputFilename[Len-1] = 'c';
      } else {
	OutputFilename += ".bc";
      }
    } else {
      if (OutputFilename[Len-3] == '.' &&
	  OutputFilename[Len-2] == 'b' &&
	  OutputFilename[Len-1] == 'c') {   // .ll -> .bc	  
	OutputFilename[Len-2] = 'l'; 
	OutputFilename[Len-1] = 'l';
      } else {
	OutputFilename += ".ll";
      }
    }
  }

private:
  string InputFilename;           // Filename to read from.  If "-", use stdin.
  string OutputFilename;          // Filename to write to.   If "-", use stdout.
  bool   Force;                   // Force output (-f argument)
};

inline ToolCommandLine::ToolCommandLine(int &argc, char **argv, bool OutBC) 
  : InputFilename("-"), OutputFilename("-"), Force(false) {
  bool FoundInputFN  = false;
  bool FoundOutputFN = false;
  bool FoundForce    = false;

  for (int i = 1; i < argc; i++) {
    int RemoveArg = 0;
    
    if (argv[i][0] == '-') {
      if (!FoundInputFN && argv[i][1] == 0) { // Is the current argument '-'
	InputFilename = argv[i];
	FoundInputFN = true;
	RemoveArg = 1;
      } else if (!FoundOutputFN && (argv[i][1] == 'o' && argv[i][2] == 0)) {
	// Is the argument -o?
	if (i+1 < argc) {        // Next arg is output fn
	  OutputFilename = argv[i+1];
	  FoundOutputFN = true;
	  RemoveArg = 2;
	}
      } else if (!FoundForce && (argv[i][1] == 'f' && argv[i][2] == 0)) {
	Force = true;
	FoundForce = true;
	RemoveArg = 1;
      }
    } else if (!FoundInputFN) {     // Is the current argument '[^-].*'?
      InputFilename = argv[i];
      FoundInputFN = true;
      RemoveArg = 1;
    }
					    
    if (RemoveArg) {
      argc -= RemoveArg;                           // Shift args over...
      memmove(argv+i, argv+i+RemoveArg, (argc-i)*sizeof(char*));
      i--;                                         // Reprocess this argument...
    }
  }

  if (!FoundOutputFN && InputFilename != "-")
    calculateOutputFilename(OutBC);
}

inline ToolCommandLine::ToolCommandLine(const string &inf, 
                                        const string &outf) 
  : InputFilename(inf), OutputFilename(outf), Force(false) {
}

inline ToolCommandLine::ToolCommandLine(const ToolCommandLine &Opts) 
  : InputFilename(Opts.InputFilename), OutputFilename(Opts.OutputFilename),
    Force(Opts.Force) {
}

inline ToolCommandLine &ToolCommandLine::operator=(const ToolCommandLine &Opts){
  InputFilename  = Opts.InputFilename;
  OutputFilename = Opts.OutputFilename;
  Force          = Opts.Force;
  return *this;
}

#endif
