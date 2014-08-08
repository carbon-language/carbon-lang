//===-- MICmnStreamStdinWindows.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIUtilStreamStdin.cpp
//
// Overview:	CMICmnStreamStdinWindows implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// Third Party Headers:
#if defined( _MSC_VER )
#include <stdio.h>
#include <Windows.h>
#include <io.h>
#include <conio.h>
#endif // defined( _MSC_VER )
#include <string.h>

// In-house headers:
#include "MICmnStreamStdinWindows.h"
#include "MICmnLog.h"
#include "MICmnResources.h"
#include "MIUtilSystemWindows.h"
#include "MIUtilSingletonHelper.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnStreamStdinWindows constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnStreamStdinWindows::CMICmnStreamStdinWindows( void )
:	m_constBufferSize( 1024 )
,	m_pStdin( nullptr )
,	m_pCmdBuffer( nullptr )
,	m_pStdinBuffer( nullptr )
,	m_nBytesToBeRead( 0 )
,	m_bRunningInConsoleWin( false )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnStreamStdinWindows destructor.
// Type:	Overridable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnStreamStdinWindows::~CMICmnStreamStdinWindows( void )
{
	Shutdown();
}

//++ ------------------------------------------------------------------------------------
// Details:	Initialize resources for *this Stdin stream.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdinWindows::Initialize( void )
{
	if( m_bInitialized )
		return MIstatus::success;

	bool bOk = MIstatus::success;
	CMIUtilString errMsg;
 
	// Note initialisation order is important here as some resources depend on previous
	MI::ModuleInit< CMICmnLog >      ( IDS_MI_INIT_ERR_LOG      , bOk, errMsg );
	MI::ModuleInit< CMICmnResources >( IDS_MI_INIT_ERR_RESOURCES, bOk, errMsg );

	// Other resources required
	if( bOk )
	{
		m_pCmdBuffer = new MIchar[ m_constBufferSize ];
		m_pStdin = stdin;

#if MICONFIG_CREATE_OWN_STDIN_BUFFER
		// Give stdinput a user defined buffer
		m_pStdinBuffer = new char[ 1024 ];
		::setbuf( stdin, m_pStdinBuffer );
#endif // MICONFIG_CREATE_OWN_STDIN_BUFFER

		// Clear error indicators for std input
		::clearerr( stdin );

#if defined( _MSC_VER )
		m_bRunningInConsoleWin = ::_isatty( ::fileno( stdin ) );
#endif // #if defined( _MSC_VER )
	}

	m_bInitialized = bOk;

	if( !bOk )
	{
		CMIUtilString strInitError( CMIUtilString::Format( MIRSRC( IDS_MI_INIT_ERR_OS_STDIN_HANDLER ), errMsg.c_str() ) );
		SetErrorDescription( strInitError );
		return MIstatus::failure;
	}

	return  MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Release resources for *this Stdin stream.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdinWindows::Shutdown( void )
{
	if( !m_bInitialized )
		return MIstatus::success;

	m_bInitialized = false;

	ClrErrorDescription();

	bool bOk = MIstatus::success;
	CMIUtilString errMsg;

	// Tidy up
	if( m_pCmdBuffer != nullptr )
	{
		delete [] m_pCmdBuffer;
		m_pCmdBuffer = nullptr;
	}
	m_pStdin = nullptr;

#if MICONFIG_CREATE_OWN_STDIN_BUFFER
	if ( m_pStdinBuffer )
		delete [] m_pStdinBuffer;
	m_pStdinBuffer = nullptr;
#endif // MICONFIG_CREATE_OWN_STDIN_BUFFER

	// Note shutdown order is important here 	
	MI::ModuleShutdown< CMICmnResources >( IDS_MI_INIT_ERR_RESOURCES, bOk, errMsg );
	MI::ModuleShutdown< CMICmnLog >      ( IDS_MI_INIT_ERR_LOG      , bOk, errMsg );

	if( !bOk )
	{
		SetErrorDescriptionn( MIRSRC( IDS_MI_SHTDWN_ERR_OS_STDIN_HANDLER ), errMsg.c_str() );
	}

	return MIstatus::success;
}	

//++ ------------------------------------------------------------------------------------
// Details:	Determine if stdin has any characters present in its buffer.
// Type:	Method.
// Args:	vwbAvail	- (W) True = There is chars available, false = nothing there.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdinWindows::InputAvailable( bool & vwbAvail )
{
	return m_bRunningInConsoleWin ? InputAvailableConsoleWin( vwbAvail ) : InputAvailableApplication( vwbAvail );
}

//++ ------------------------------------------------------------------------------------
// Details:	Determine if stdin has any characters present in its buffer. If running in a 
//			terminal use _kbhit().
// Type:	Method.
// Args:	vwbAvail	- (W) True = There is chars available, false = nothing there.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdinWindows::InputAvailableConsoleWin( bool & vwbAvail )
{
#if defined( _MSC_VER )
  if( m_nBytesToBeRead == 0 )
   {
        // Get a windows handle to std input stream
        HANDLE handle = ::GetStdHandle( STD_INPUT_HANDLE );
        DWORD nBytesWaiting = ::_kbhit();
        
		// Save the number of bytes to be read so that we can check if input is available to be read
        m_nBytesToBeRead = nBytesWaiting;

        // Return state of whether bytes are waiting or not
        vwbAvail = (nBytesWaiting > 0);
    }
#endif // #if defined( _MSC_VER )

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Determine if stdin has any characters present in its buffer.
// Type:	Method.
// Args:	vwbAvail	- (W) True = There is chars available, false = nothing there.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdinWindows::InputAvailableApplication( bool & vwbAvail )
{
 #if defined( _MSC_VER )
  if( m_nBytesToBeRead == 0 )
   {
        // Get a windows handle to std input stream
        HANDLE handle = ::GetStdHandle( STD_INPUT_HANDLE );
        DWORD nBytesWaiting = 0;

        // Ask how many bytes are available
        if( ::PeekNamedPipe( handle, nullptr, 0, nullptr, &nBytesWaiting, nullptr ) == FALSE )
        {
            // This can occur when the client i.e. Eclipse closes the stdin stream 'cause it deems its work is finished
            // for that debug session. May be we should be handling SIGKILL somehow?
            const CMIUtilString osErrMsg( CMIUtilSystemWindows().GetOSLastError().StripCRAll() );
            SetErrorDescription( CMIUtilString::Format( MIRSRC( IDS_STDIN_ERR_CHKING_BYTE_AVAILABLE ), osErrMsg.c_str() ) );
            return MIstatus::failure;
        }
 
        // Save the number of bytes to be read so that we can check if input is available to be read
        m_nBytesToBeRead = nBytesWaiting;

        // Return state of whether bytes are waiting or not
        vwbAvail = (nBytesWaiting > 0);
    }
#endif // #if defined( _MSC_VER )

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Wait on new line of data from stdin stream (completed by '\n' or '\r').
// Type:	Method.
// Args:	vwErrMsg	- (W) Empty string ok or error description.			
// Return:	MIchar * - text buffer pointer or NULL on failure.
// Throws:	None.
//--
const MIchar * CMICmnStreamStdinWindows::ReadLine( CMIUtilString & vwErrMsg )
{
	vwErrMsg.clear();
	
	// Read user input
	const MIchar * pText = ::fgets( &m_pCmdBuffer[ 0 ], m_constBufferSize, stdin );
	if( pText == nullptr )
	{
		if( ::ferror( m_pStdin ) != 0 )
			vwErrMsg = ::strerror( errno );
		return nullptr;
	}

    // Subtract the number of bytes read so that we can check if input is available to be read
    m_nBytesToBeRead = m_nBytesToBeRead - ::strlen( pText );
	
	// Strip off new line characters
	for( MIchar * pI = m_pCmdBuffer; *pI != '\0'; pI++ )
	{
		if( (*pI == '\n') || (*pI == '\r') )
		{
			*pI = '\0';
			break;
		}
	}

	return pText;
}