// $Id$ -*-c++-*-
//**************************************************************************/
// File:
//	LLCOptions.h
// 
// Purpose:
//	Options for the llc compiler.
// 
// History:
//	7/15/01	 -  Vikram Adve  -  Created
// 
//**************************************************************************/

#ifndef LLVM_LLC_LLCOPTIONS_H
#define LLVM_LLC_LLCOPTIONS_H

//************************** System Include Files **************************/

#include <iostream.h>
#include <unistd.h>


//*************************** User Include Files ***************************/

#include "llvm/Support/ProgramOptions.h"
#include "llvm/Support/ProgramOption.h"

//************************ Option Name Definitions *************************/

const char* const HELP_OPT		= "help";
const char* const DEBUG_OPT		= "d";
const char* const QUIET_OPT		= "q";
const char* const DEBUG_INSTR_SELECT_OPT= "debug_select";
const char* const OUTFILENAME_OPT	= "o";


//---------------------------------------------------------------------------
// class LLCOptions
//---------------------------------------------------------------------------

class LLCOptions : public ProgramOptions {
public:
  /*ctor*/		LLCOptions	(int _argc,
					 const char* _argv[],
					 const char* _envp[]); 
  /*dtor*/ virtual	~LLCOptions	();

  const string&		getInputFileName() const  { return inputFileName; }
  
  const string&		getOutputFileName() const { return outputFileName; }
  
protected:

  //--------------------------------------------------------------------
  // Initialize for all our compiler options (called by constructors).
  //--------------------------------------------------------------------
  void InitializeOptions();
  
  //--------------------------------------------------------------------
  // Make sure the parse went ok.
  //--------------------------------------------------------------------
  void CheckParse();

  //--------------------------------------------------------------------
  // Parse arguments after all options are consumed.
  // This is called after a successful ParseArgs.
  //--------------------------------------------------------------------
  virtual void ParseExtraArgs(); 
  
  //--------------------------------------------------------------------
  // Print message describing which arguments and options are 
  // required, optional, mutually exclusive, ...
  // called in ProgramOptions::Usage() method
  //--------------------------------------------------------------------
  virtual void PrintUsage(ostream& stream) const;

private:
  string	  inputFileName;
  string	  outputFileName;
};

//**************************************************************************/

#endif
