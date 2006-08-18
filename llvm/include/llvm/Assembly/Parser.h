//===-- llvm/Assembly/Parser.h - Parser for VM assembly files ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  These classes are implemented by the lib/AsmParser library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_PARSER_H
#define LLVM_ASSEMBLY_PARSER_H

#include <string>

namespace llvm {

class Module;
class ParseError;


/// This function is the main interface to the LLVM Assembly Parse. It parses 
/// an ascii file that (presumably) contains LLVM Assembly code. It returns a
/// Module (intermediate representation) with the corresponding features. Note
/// that this does not verify that the generated Module is valid, so you should
/// run the verifier after parsing the file to check that it is okay.
/// @brief Parse LLVM Assembly from a file
Module *ParseAssemblyFile(
  const std::string &Filename, ///< The name of the file to parse
  ParseError* Error = 0        ///< If not null, an object to return errors in.
);

/// The function is a secondary interface to the LLVM Assembly Parse. It parses 
/// an ascii string that (presumably) contains LLVM Assembly code. It returns a
/// Module (intermediate representation) with the corresponding features. Note
/// that this does not verify that the generated Module is valid, so you should
/// run the verifier after parsing the file to check that it is okay.
/// @brief Parse LLVM Assembly from a string
Module *ParseAssemblyString(
  const char * AsmString, ///< The string containing assembly
  Module * M,             ///< A module to add the assembly too.
  ParseError* Error = 0   ///< If not null, an object to return errors in.
);

//===------------------------------------------------------------------------===
//                              Helper Classes
//===------------------------------------------------------------------------===

/// An instance of this class can be passed to ParseAssemblyFile or 
/// ParseAssemblyString functions in order to capture error information from
/// the parser.  It provides a standard way to print out the error message
/// including the file name and line number where the error occurred.
/// @brief An LLVM Assembly Parsing Error Object
class ParseError {
public:
  ParseError() : Filename("unknown"), Message("none"), LineNo(0), ColumnNo(0) {}
  ParseError(const ParseError &E);

  // getMessage - Return the message passed in at construction time plus extra
  // information extracted from the options used to parse with...
  //
  const std::string getMessage() const;

  inline const std::string &getRawMessage() const {   // Just the raw message...
    return Message;
  }

  inline const std::string &getFilename() const {
    return Filename;
  }

  void setError(const std::string &filename, const std::string &message,
                 int LineNo = -1, int ColNo = -1);

  // getErrorLocation - Return the line and column number of the error in the
  // input source file.  The source filename can be derived from the
  // ParserOptions in effect.  If positional information is not applicable,
  // these will return a value of -1.
  //
  inline const void getErrorLocation(int &Line, int &Column) const {
    Line = LineNo; Column = ColumnNo;
  }

private :
  std::string Filename;
  std::string Message;
  int LineNo, ColumnNo;                               // -1 if not relevant

  ParseError &operator=(const ParseError &E); // objects by reference
};

} // End llvm namespace

#endif
