// $Id$ -*-c++-*-
//***************************************************************************
//
// File:
//	ProgramOptions.h
//
// Purpose:
//	A representation of options for any program.
//
// History:
//	08/08/95 - adve  - Created in the dHPF compiler
//      10/10/96 - mpal, dbaker - converted to const member functions.
//      10/19/96 - meven - slightly changed interface to accomodate 
//                         arguments other than -X type options
//	07/15/01 - vadve - Copied to LLVM system and modified
//
//**************************************************************************/

#ifndef LLVM_SUPPORT_PROGRAMOPTIONS_H
#define LLVM_SUPPORT_PROGRAMOPTIONS_H

//************************** System Include Files **************************/

#include <iostream.h>

//*************************** User Include Files ***************************/

#include "llvm/Support/Unique.h"
#include "llvm/Support/StringUtils.h"

//************************ Forward Declarations ****************************/

class ProgramOption;

//************************* Main Driver Routine ****************************/

//---------------------------------------------------------------------------
//
//  Class: ProgramOptions
//
//  Base Classes: none
//
//  Class Data Members:
//	ProgramOptionsRepr*	Internal representation of program options,
//				accessible to derived classes.
//  Purpose:
//     Base class for representing the set of options for a program.
//
//---------------------------------------------------------------------------

class ProgramOptions: public Unique {
public:
  /*ctor*/	ProgramOptions	(int _argc,
				 const char* _argv[],
				 const char* _envp[]);
  /*dtor*/	~ProgramOptions	()	{}	
  
  //--------------------------------------------------------------------
  // Retrieving different kinds of arguments.
  // The required argument is specified by the optionString.
  //--------------------------------------------------------------------
    
  const char*	StringOptionValue(const char* optionString) const;
  bool		FlagOptionValue	 (const char* optionString) const;
  double	RealOptionValue	 (const char* optionString) const;
  int		IntOptionValue	 (const char* optionString) const;
  
  bool		OptionSpecified	 (const char* optionString) const;
    
  //--------------------------------------------------------------------
  // The name used to invoke this program.
  //--------------------------------------------------------------------
  const char* ProgramName	 () const;
    
  //--------------------------------------------------------------------
  // Access to unparsed arguments
  //--------------------------------------------------------------------
  int         NumberOfOtherOptions() const;
  const char* OtherOption(int i) const;

  //--------------------------------------------------------------------
  // Access to the original arguments
  //--------------------------------------------------------------------
  const char**		GetOriginalArgs() const;
  void PrintArgs(ostream &out) const;

  //--------------------------------------------------------------------
  // Derived classes may use PrintOptions in their own PrintUsage() fct 
  // to print information about optional, required, or additional
  // arguments 
  //--------------------------------------------------------------------
  virtual void PrintOptions    (ostream& stream) const;
  virtual void Usage           () const;

  //--------------------------------------------------------------------
  // Generate a human-friendly description of the options actually set.
  // The vector returned contains a multiple of 3 of entries, entry 3n is
  // the name of the option, entry 3n + 1 contains the description of
  // the option and entry 3n + 2 contains the ascii value of the option.
  // All entries are allocated using malloc and can be freed with 'free'.
  //--------------------------------------------------------------------
  virtual vector<char*> GetDescription	() const;
  
protected:
  //--------------------------------------------------------------------
  // Called by the subclass to register each possible option
  // used by the program.  Assumes ownership of the ProgramOption.
  //--------------------------------------------------------------------
  void				Register	(ProgramOption* option);

  //--------------------------------------------------------------------
  // Parses the options.
  //--------------------------------------------------------------------
  void	ParseArgs	(int argc,
			 const char* argv[],
			 const char* envp[]);
  
  inline ProgramOption* OptionHandler(const char* optString) {
     ProgramOption** poPtr = optionRegistry.query(optString);
     return poPtr? *poPtr : NULL;
  }

  inline const ProgramOption* OptionHandler(const char* optString) const {
     const ProgramOption* const* poPtr = optionRegistry.query(optString);
     return poPtr? *poPtr : NULL;
  }
  
protected:
  //--------------------------------------------------------------------
  // Functions that must be overridden by the subclass.
  //--------------------------------------------------------------------
  
  virtual void	ParseExtraArgs	() = 0; // called after successful ParseArgs
  
  virtual void	PrintUsage	(ostream& stream) const = 0;
  
protected:
  StringMap<ProgramOption*> optionRegistry;
  int			argc;
  const char**		argv;
  const char**		envp;
  int			argsConsumed;
};

//**************************************************************************/

#endif
