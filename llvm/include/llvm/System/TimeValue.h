//===-- TimeValue.h - Declare OS TimeValue Concept ---------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file declares the operating system TimeValue concept.
//
//===----------------------------------------------------------------------===//

#include <llvm/Support/DataTypes.h>

#ifndef LLVM_SYSTEM_TIMEVALUE_H
#define LLVM_SYSTEM_TIMEVALUE_H

namespace llvm {
namespace sys {
  /// This class is used where a precise fixed point in time is required. The 
  /// range of TimeValue spans many hundreds of billions of years both past and 
  /// present.  The precision of TimeValue is to the nanosecond. However, actual 
  /// precision of values will be determined by the resolution of the system clock. 
  /// The TimeValue class is used in conjunction with several other lib/System 
  /// interfaces to specify the time at which a call should timeout, etc.
  /// @since 1.4
  /// @brief Provides an abstraction for a fixed point in time.
  class TimeValue {
  /// @name Constants
  /// @{
  public:

    /// A constant TimeValue representing the smallest time
    /// value permissable by the class. min_time is some point
    /// in the distant past, about 300 billion years BC.
    /// @brief The smallest possible time value.
    static const TimeValue MinTime;

    /// A constant TimeValue representing the largest time
    /// value permissable by the class. max_time is some point
    /// in the distant future, about 300 billion years AD.
    /// @brief The largest possible time value.
    static const TimeValue MaxTime;

    /// A constant TimeValue representing the base time,
    /// or zero time of 00:00:00 (midnight) January 1st, 2000.
    /// @brief 00:00:00 Jan 1, 2000 UTC.
    static const TimeValue ZeroTime;

    /// A constant TimeValue for the posix base time which is
    /// 00:00:00 (midnight) January 1st, 1970.
    /// @brief 00:00:00 Jan 1, 1970 UTC.
    static const TimeValue PosixZeroTime;

    /// A constant TimeValue for the win32 base time which is
    /// 00:00:00 (midnight) January 1st, 1601.
    /// @brief 00:00:00 Jan 1, 1601 UTC.
    static const TimeValue Win32ZeroTime;

  /// @}
  /// @name Types
  /// @{
  public:
    typedef int64_t SecondsType;        ///< Type used for representing seconds.
    typedef int32_t NanoSecondsType;    ///< Type used for representing nanoseconds.

    enum TimeConversions {
      NANOSECONDS_PER_SECOND = 1000000000,
      MICROSECONDS_PER_SECOND = 1000000,
      MILLISECONDS_PER_SECOND = 1000,
      NANOSECONDS_PER_MICROSECOND = 1000,
      NANOSECONDS_PER_MILLISECOND = 1000000,
      NANOSECONDS_PER_POSIX_TICK = 100,
      NANOSECONDS_PER_WIN32_TICK = 100,
    };

  /// @}
  /// @name Constructors
  /// @{
  public:
    /// Value is initialized to zero_time.
    /// @brief Default Constructor
    TimeValue () 
      : seconds_(0), nanos_(0) {}

    /// Caller provides the exact value in seconds and
    /// nano-seconds. The \p nsec argument defaults to
    /// zero for convenience.
    /// @brief Explicit Constructor.
    TimeValue (SecondsType seconds, NanoSecondsType nanos = 0)
      : seconds_( seconds )
      , nanos_( nanos )
    {
      this->normalize();
    }

    /// Caller provides the exact value in in seconds with the
    /// fractional part represengin nanoseconds.
    /// @brief Double Constructor.
    TimeValue( double time )
      : seconds_( 0 ) , nanos_ ( 0 )
    {
      this->set( time );
    }

    /// Copies one TimeValue to another.
    /// @brief Copy Constructor.
    TimeValue( const TimeValue & that ) 
      : seconds_( that.seconds_ ) , nanos_( that.nanos_ ) { }

  //
  /// @}
  /// @name Operators
  /// @{
  public:
    /// Assigns the value of \p that TimeValue to \p this
    /// @brief Assignment operator.
    TimeValue& operator = ( const TimeValue& that ) {
      this->set( that );
      return *this;
    }

    /// Assigns the value of \p that floating point value to \p this.
    /// The \p that vlue is assumed to be in seconds format with
    /// the fraction indicating the number of nanoseconds.
    /// @brief Assignment operator.
    TimeValue& operator = ( double that ) {
      this->set( that );
      return *this;
    }

    /// Add \p that to \p this.
    /// @returns this
    /// @brief Incrementing assignment operator.
    TimeValue& operator += (const TimeValue& that ) {
      this->seconds_ += that.seconds_  ;
      this->nanos_ += that.nanos_ ;
      this->normalize();
      return *this;
    }

    /// Add \p addend to \p this. \p addend is assumed to be in seconds
    /// format with the fraction providing nanoseconds.
    /// @returns this
    /// @brief Incrementing assignment operator.
    TimeValue& operator += ( double addend ) {
      SecondsType seconds_part = static_cast<SecondsType>( addend );
      NanoSecondsType nanos_part = static_cast<NanoSecondsType>(
          (addend - static_cast<double>(seconds_part)) * NANOSECONDS_PER_SECOND );
      this->seconds_ += seconds_part;
      this->nanos_ += nanos_part;
      this->normalize();
      return *this;
    }

    /// Subtract \p that from \p this.
    /// @returns this
    /// @brief Decrementing assignment operator.
    TimeValue& operator -= (const TimeValue &that ) {
      this->seconds_ -= that.seconds_ ;
      this->nanos_ -= that.nanos_ ;
      this->normalize();
      return *this;
    }

    /// Add \p that to \p this. \p that is assumed to be in seconds
    /// format with the fraction providing nanoseconds.
    /// @returns this
    /// @brief Decrementing assignment operator.
    TimeValue& operator -= ( double subtrahend ) {
      SecondsType seconds_part = static_cast<SecondsType>( subtrahend );
      NanoSecondsType nanos_part = static_cast<NanoSecondsType>(
          (subtrahend - static_cast<double>(seconds_part)) * NANOSECONDS_PER_SECOND );
      this->seconds_ -= seconds_part;
      this->nanos_ -= nanos_part;
      this->normalize();
      return *this;
    }

    /// @brief True if this < that.
    int operator < (const TimeValue &that) const { return that > *this; }

    /// @brief True if this > that.
    int operator > (const TimeValue &that) const {
      if ( this->seconds_ > that.seconds_ )
      {
          return 1;
      }
      else if ( this->seconds_ == that.seconds_ )
      {
          if ( this->nanos_ > that.nanos_ ) return 1;
      }
      return 0;
    }

    /// @brief True if this <= that.
    int operator <= (const TimeValue &that) const { return that >= *this; }

    /// @brief True if this >= that.
    int operator >= (const TimeValue &that) const {
      if ( this->seconds_ > that.seconds_ )
      {
          return 1;
      }
      else if ( this->seconds_ == that.seconds_ )
      {
          if ( this->nanos_ >= that.nanos_ ) return 1;
      }
      return 0;
    }

    /// @brief True if this == that.
    int operator == (const TimeValue &that) const {
      return (this->seconds_ == that.seconds_) && 
             (this->nanos_ == that.nanos_);
    }

    /// @brief True if this != that.
    int operator != (const TimeValue &that) const { return !(*this == that); }

    /// Adds two TimeValue objects together.
    /// @returns The sum of the two operands as a new TimeValue
    /// @brief Addition operator.
    friend TimeValue operator + (const TimeValue &tv1, const TimeValue &tv2);

    /// Subtracts two TimeValue objects.
    /// @returns The difference of the two operands as a new TimeValue
    /// @brief Subtraction operator.
    friend TimeValue operator - (const TimeValue &tv1, const TimeValue &tv2);

  /// @}
  /// @name Accessors
  /// @{
  public:

      /// @brief Retrieve the seconds component
      SecondsType seconds( void ) const { return seconds_; }

      /// @brief Retrieve the nanoseconds component.
      NanoSecondsType nanoseconds( void ) const { return nanos_; }

      /// @brief Retrieve the fractional part as microseconds;
      uint32_t microseconds( void ) const { 
        return nanos_ / NANOSECONDS_PER_MICROSECOND;
      }

      /// @brief Retrieve the fractional part as milliseconds;
      uint32_t milliseconds( void ) const {
        return nanos_ / NANOSECONDS_PER_MILLISECOND;
      }

      /// @brief Convert to a number of microseconds (can overflow)
      uint64_t usec( void ) const {
        return seconds_ * MICROSECONDS_PER_SECOND + 
               ( nanos_ / NANOSECONDS_PER_MICROSECOND );
      }

      /// @brief Convert to a number of milliseconds (can overflow)
      uint64_t msec( void ) const {
        return seconds_ * MILLISECONDS_PER_SECOND + ( nanos_ / NANOSECONDS_PER_MILLISECOND );
      }

      /// @brief Convert to unix time (100 nanoseconds since 12:00:00a Jan 1, 1970)
      uint64_t posix_time( void ) const {
        uint64_t result = seconds_ - PosixZeroTime.seconds_;
        result += nanos_ / NANOSECONDS_PER_POSIX_TICK;
        return result;
      }

      /// @brief Convert to windows time (seconds since 12:00:00a Jan 1, 1601)
      uint64_t win32_time( void ) const {
        uint64_t result = seconds_ - Win32ZeroTime.seconds_;
        result += nanos_ / NANOSECONDS_PER_WIN32_TICK;
        return result;
      }

      /// @brief Convert to timespec time (ala POSIX.1b)
      void timespecTime( uint64_t& seconds, uint32_t& nanos ) const {
        nanos = nanos_;
        seconds = seconds_ - PosixZeroTime.seconds_;
      }

  /// @}
  /// @name Mutators
  /// @{
      /// @brief Set a TimeValue from the two component values.
      void set (SecondsType secs, NanoSecondsType nanos) {
        this->seconds_ = secs;
        this->nanos_ = nanos;
        this->normalize();
      }

      /// @brief Set a TimeValue from another
      void set ( const TimeValue & that ) {
        this->seconds_ = that.seconds_;
        this->nanos_ = that.nanos_;
      }

      /// The double value is assumed to be in seconds format, with any 
      /// remainder treated as nanoseconds.
      /// @brief Set a TimeValue from a double.
      void set (double new_time) {
        SecondsType integer_part = static_cast<SecondsType>( new_time );
        seconds_ = integer_part;
        nanos_ = static_cast<NanoSecondsType>( (new_time - static_cast<double>(integer_part)) * NANOSECONDS_PER_SECOND );
        this->normalize();
      }

      /// The seconds component of the timevalue is set to \p sec without
      /// modifying the nanoseconds part.  This is useful for whole second arithmetic.
      /// @brief Set the seconds component.
      void seconds (SecondsType sec ) {
        this->seconds_ = sec;
        this->normalize();
      }

      /// The seconds component remains unchanged.
      /// @brief Set the nanoseconds component using a number of nanoseconds.
      void nanoseconds ( NanoSecondsType nanos ) {
        this->nanos_ = nanos;
        this->normalize();
      }

      /// The seconds component remains unchanged.
      /// @brief Set the nanoseconds component using a number of microseconds.
      void microseconds ( int32_t micros ) {
        this->nanos_ = micros * NANOSECONDS_PER_MICROSECOND;
        this->normalize();
      };

      /// The seconds component remains unchanged.
      /// @brief Set the nanoseconds component using a number of milliseconds.
      void milliseconds ( int32_t millis ) {
        this->nanos_ = millis * NANOSECONDS_PER_MILLISECOND;
        this->normalize();
      };

      /// @brief Converts from microsecond format to TimeValue format
      void usec( int64_t microseconds ) {
        this->seconds_ = microseconds / MICROSECONDS_PER_SECOND;
        this->nanos_ = (microseconds % MICROSECONDS_PER_SECOND) * 
          NANOSECONDS_PER_MICROSECOND;
        this->normalize();
      }

      /// @brief Converts from millisecond format to TimeValue format
      void msec( int64_t milliseconds ) {
        this->seconds_ = milliseconds / MILLISECONDS_PER_SECOND;
        this->nanos_ = (milliseconds % MILLISECONDS_PER_SECOND) * 
          NANOSECONDS_PER_MILLISECOND;
        this->normalize();
      }

      /// This causes the values to be represented so that the fractional
      /// part is minimized, possibly incrementing the seconds part.
      /// @brief Normalize to canonical form.
      void normalize (void);

      /// @brief Sets \p this to the current time (UTC).
      void now( void );

  /// @}
  /// @name Data
  /// @{
  private:
      /// Store the values as a <timeval>.
      SecondsType      seconds_;       ///< Stores the seconds component of the TimeVal
      NanoSecondsType  nanos_;         ///< Stores the nanoseconds component of the TimeVal

  /// @}

  };

inline TimeValue operator + (const TimeValue &tv1, const TimeValue &tv2) {
  TimeValue sum (tv1.seconds_ + tv2.seconds_, tv1.nanos_ + tv2.nanos_);
  sum.normalize ();
  return sum;
}

inline TimeValue operator - (const TimeValue &tv1, const TimeValue &tv2) {
  TimeValue difference (tv1.seconds_ - tv2.seconds_, tv1.nanos_ - tv2.nanos_ );
  difference.normalize ();
  return difference;
}

}
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
#endif
