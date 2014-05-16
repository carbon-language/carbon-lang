//===-- MIUtilDateTimeStd.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIUtilDateTimeStd.cpp
//
// Overview:	CMIUtilDateTimeStd implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// Include compiler configuration
#include "MICmnConfig.h"

// Third party headers
#include <time.h>

// In-house headers:
#include "MIUtilDateTimeStd.h"
#include "MICmnResources.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMIUtilDateTimeStd constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMIUtilDateTimeStd::CMIUtilDateTimeStd( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMIUtilDateTimeStd destructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMIUtilDateTimeStd::~CMIUtilDateTimeStd( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve system local current date. Format is DD/MM/YYYY.
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString - Text description.
// Throws:	None.
//--
CMIUtilString CMIUtilDateTimeStd::GetDate( void ) const
{
	CMIUtilString strDate( MIRSRC( IDS_WORD_INVALIDBRKTS ) );
	CMIUtilString localDate;
	CMIUtilString localTime;
	if( GetDateTimeShort( localDate, localTime ) )
		strDate = localDate;
	
	return strDate;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve system local current time. Format is HH:MM:SS 24 hour clock.
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString - Text description.
// Throws:	None.
//--
CMIUtilString CMIUtilDateTimeStd::GetTime( void ) const 
{
	CMIUtilString strTime( MIRSRC( IDS_WORD_INVALIDBRKTS ) );
	CMIUtilString localDate;
	CMIUtilString localTime;
	if( GetDateTimeShort( localDate, localTime ) )
		strTime = localTime;
	
	return strTime;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve system local time and date, short version.
// Type:	Method.
// Args:	vrwLocalDate	- (W) Text date. Format is DD/MM/YYYY.
//			vrwLocalTime	- (W) Text time. Format is HH:MM:SS 24 hour clock.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIUtilDateTimeStd::GetDateTimeShort( CMIUtilString & vrwLocalDate, CMIUtilString & vrwLocalTime ) const
{
	time_t rawTime;
	::time( &rawTime );
	struct tm * pTi = ::localtime( &rawTime );

	vrwLocalDate = CMIUtilString::Format( "%d/%d/%d", pTi->tm_wday, pTi->tm_mon, pTi->tm_year );
	vrwLocalTime = CMIUtilString::Format( "%d:%d:%d", pTi->tm_hour, pTi->tm_min, pTi->tm_sec  );

	return MIstatus::success;
}
