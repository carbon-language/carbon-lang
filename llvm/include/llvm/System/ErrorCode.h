//===- ErrorCode.h - Declare ErrorCode class --------------------*- C++ -*-===//
//
// Copyright (C) 2004 eXtensible Systems, Inc. All Rights Reserved.
//
// This program is open source software; you can redistribute it and/or modify
// it under the terms of the University of Illinois Open Source License. See
// LICENSE.TXT (distributed with this software) for details.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.
//
//===----------------------------------------------------------------------===//
/// @file lib/System/ErrorCode.h
/// @author Reid Spencer <raspencer@x10sys.com> (original author)
/// @version \verbatim $Id$ \endverbatim
/// @date 2004/08/14
/// @since 1.4
/// @brief Declares the llvm::sys::ErrorCode class.
//===----------------------------------------------------------------------===//
#ifndef LLVM_SYSTEM_ERRORCODE_H
#define LLVM_SYSTEM_ERRORCODE_H

#include <string>

/// @brief Computes an errorcode value from its domain and index values.
#define LLVM_ERROR_CODE( domain, index ) \
    ( ( domain << ::llvm::sys::ERROR_DOMAIN_SHIFT ) + \
      ( index & ::llvm::sys::ERROR_INDEX_MASK ))

namespace llvm {
namespace sys {

  /// @brief The number of bits to shift right to get the domain part of an error code.
  const uint32_t ERROR_DOMAIN_SHIFT = (sizeof(uint32_t)*8)*3/4;

  const uint32_t ERROR_DOMAIN_MASK =
    ((1<<((sizeof(uint32_t)*8)-ERROR_DOMAIN_SHIFT))-1)<<ERROR_DOMAIN_SHIFT;

  /// @brief The mask to get only the index part of an error code.
  const uint32_t ERROR_INDEX_MASK = ( 1 << ERROR_DOMAIN_SHIFT) - 1;

  /// @brief The maximum value for the index part of an error code.
  const uint32_t MAX_ERROR_INDEX= ERROR_INDEX_MASK;

  /// @brief The minimum value for the index part of an error code.
  const uint32_t MIN_ERROR_INDEX= 1;

  /// @brief A constant to represent the non error condition.
  const uint32_t NOT_AN_ERROR = 0;

  /// @brief An enumeration of the possible error code domains
  enum ErrorDomains {
    OSDomain = 0,     ///< The domain of operating system specific error codes
    SystemDomain = 1, ///< The domain of lib/System specific error codes
  };

  /// @brief An enumeration of the error codes defined by lib/System.
  enum SystemErrors {
    ERR_SYS_INVALID_ARG = LLVM_ERROR_CODE(SystemDomain,1),
  };

  /// This class provides the error code value for every error that the System
  /// library can produce. It simply partitions a uint32_t into a domain part
  /// (top 8 bits) and an index part (low 24 bits). This is necessary since 
  /// lib/System can generate its own error codes and there needs to be a way
  /// to distinguish them from the operating system errors.
  ///
  /// Note that ErrorCode is the only way to determine if an error occurred
  /// resulting from a lib/System call. lib/System will (by design) never throw
  /// an exception. This is done to make lib/System calls very efficient and to
  /// avoid the overhead of exception processing since most of the time the
  /// calls will succeed.
  ///
  /// There are various methods on the ErrorCode class to translate the error
  /// code into a readable string, should the need arise. There are also methods
  /// to distinguish the kind of error and this works in a platform agnostic
  /// way for most of the common cases.
  ///
  /// @since 1.4
  /// @brief Provides a 32 bit error code that encapsulates error domain and 
  /// index.
  class ErrorCode
  {
  /// @name Constructors
  /// @{
  public:

    /// This constructor instantiates a new ErrorCode object
    /// with a "no error" error code.
    /// @brief Constructor.
    ErrorCode() throw() : Code(NOT_AN_ERROR) {}

    /// This constructor instantiates a new ErrorCode object
    /// with an integer code.
    /// @brief Constructor.
    ErrorCode (uint32_t code) throw() : Code(code) {}

    /// This constructor instantiates a new ErrorCode object
    /// with an integer code.
    /// @brief Constructor.
    ErrorCode(int code) throw()
      : Code ( static_cast<uint32_t>( code ) ) { }

    /// Copies one ErrorCode to another.
    /// @brief Copy Constructor.
    ErrorCode(const ErrorCode& that) throw() : Code ( that.Code ) { }

    /// Nothing much to do.
    /// @brief Destructor
    ~ErrorCode(void) throw() {}

  /// @}
  /// @name Operators
  /// @{
  public:
    /// Assigns one ErrorCode object to another
    /// @brief Assignment Operator.
    ErrorCode & operator = (const ErrorCode& that) throw() {
      Code = that.Code;
      return *this;
    }

    /// Returns true if \p this and \p that refer to the same type of error
    /// @brief Equality Operator.
    bool operator == (const ErrorCode& that) const throw() {
      return Code == that.Code;
    }

    /// Returns true if \p this and \p that do not refer to the same type of error
    /// @brief Inequality Operator.
    bool operator != (const ErrorCode& that) const throw() {
      return Code != that.Code;
    }

    /// @return a non zero value if an error condition exists
    /// @brief Test Operator
    operator bool() const throw() { return Code == NOT_AN_ERROR; }

    /// @return a non zero value if an error condition exists
    /// @brief Test Operator.
    operator int() const throw() { return Code; }

    /// @brief unsigned conversion operator.
    operator unsigned int() const throw() { return Code; }

    /// @brief long conversion operator.
    operator long() const throw() { return Code; }

    /// @brief unsigned long conversion operator
    operator unsigned long() const throw() { return Code; }

  /// @}
  /// @name Accessors
  /// @{
  public:
    /// @returns the integer error code value.
    /// @brief Provides the integer error code value.
    uint32_t code() const throw() { return Code; }

    /// @returns the integer domain number for the error number
    /// @brief Provides the domain of the error
    uint32_t domain() const throw() {
      return (Code & ERROR_DOMAIN_MASK) >> ERROR_DOMAIN_SHIFT;
    }

    /// @brief Provides the index of the error
    uint32_t index() const throw() {
      return Code & ERROR_INDEX_MASK;
    }

    /// @brief Provides a readable string related to the error code
    std::string description() const throw();

    /// If \p this ErrorCode and \p ec have the same error code then return 
    /// true, otherwise false.
    /// @param ec An errorcode value to compare against this ErrorCode
    /// @returns true if \p this ErrorCode has the same error code as \p errcode
    /// @brief Determines identity of the error code.
    bool is( const ErrorCode& ec ) const throw() { return Code == ec.Code; }

  /// @}
  /// @name Mutators
  /// @{
  public:
    /// @returns the previous integer error code value.
    /// @brief allows the integer error code to be set.
    uint32_t code(uint32_t code) throw() {
      uint32_t old_code = Code;
      Code = code;
      return old_code;
    }

  /// @}
  /// @name Data
  /// @{
  private:
    uint32_t Code; ///< The error code value
  /// @}

  };

}
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab

#endif
