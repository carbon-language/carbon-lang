//===-- MIDriver.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIDriver.cpp
//
// Overview:	CMIDriver implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// Third party headers:
#include <stdarg.h>		// va_list, va_start, var_end
#include <iostream>
#include <lldb/API/SBError.h>

// In-house headers:
#include "Driver.h"
#include "MIDriver.h"
#include "MICmnResources.h"
#include "MICmnLog.h"
#include "MICmdMgr.h"
#include "MICmnLLDBDebugger.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"
#include "MICmnThreadMgrStd.h"
#include "MIUtilDebug.h"
#include "MIUtilSingletonHelper.h"
#include "MICmnStreamStdout.h"
#include "MICmnStreamStderr.h"
#include "MICmdArgValFile.h"
#include "MICmdArgValString.h"
#include "MICmnConfig.h"

// Instantiations:
#if _DEBUG
	const CMIUtilString	CMIDriver::ms_constMIVersion = MIRSRC( IDS_MI_VERSION_DESCRIPTION_DEBUG );	
#else
	const CMIUtilString	CMIDriver::ms_constMIVersion = MIRSRC( IDS_MI_VERSION_DESCRIPTION );	// Matches version in resources file
#endif // _DEBUG
const CMIUtilString	CMIDriver::ms_constAppNameShort( MIRSRC( IDS_MI_APPNAME_SHORT ) );
const CMIUtilString	CMIDriver::ms_constAppNameLong( MIRSRC( IDS_MI_APPNAME_LONG ) );

//++ ------------------------------------------------------------------------------------
// Details:	CMIDriver constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMIDriver::CMIDriver( void )
:	m_bFallThruToOtherDriverEnabled( false )
,	m_bDriverIsExiting( false )
,	m_handleMainThread( 0 )
,	m_rStdin( CMICmnStreamStdin::Instance() )
,	m_rLldbDebugger( CMICmnLLDBDebugger::Instance() )
,	m_rStdOut( CMICmnStreamStdout::Instance() )
,	m_eCurrentDriverState( eDriverState_NotRunning )
,	m_bHaveExecutableFileNamePathOnCmdLine( false )
,	m_bDriverDebuggingArgExecutable( false )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMIDriver destructor.
// Type:	Overridden.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMIDriver::~CMIDriver( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	Set whether *this driver (the parent) is enabled to pass a command to its 
//			fall through (child) driver to interpret the command and do work instead
//			(if *this driver decides it can't hanled the command).
// Type:	Method.
// Args:	vbYes	- (R) True = yes fall through, false = do not pass on command.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::SetEnableFallThru( const bool vbYes )
{
	m_bFallThruToOtherDriverEnabled = vbYes;
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Get whether *this driver (the parent) is enabled to pass a command to its 
//			fall through (child) driver to interpret the command and do work instead
//			(if *this driver decides it can't hanled the command).
// Type:	Method.
// Args:	None.
// Return:	bool - True = yes fall through, false = do not pass on command.
// Throws:	None.
//--
bool CMIDriver::GetEnableFallThru( void ) const
{
	return m_bFallThruToOtherDriverEnabled;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve MI's application name of itself.
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString & - Text description.
// Throws:	None.
//--
const CMIUtilString & CMIDriver::GetAppNameShort( void ) const
{
	return ms_constAppNameShort;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve MI's application name of itself.
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString & - Text description.
// Throws:	None.
//--
const CMIUtilString & CMIDriver::GetAppNameLong( void ) const
{
	return ms_constAppNameLong;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve MI's version description of itself.
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString & - Text description.
// Throws:	None.
//--
const CMIUtilString & CMIDriver::GetVersionDescription( void ) const
{
	return ms_constMIVersion;
}

//++ ------------------------------------------------------------------------------------
// Details:	Initialize setup *this driver ready for use.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::Initialize( void )
{
	m_eCurrentDriverState = eDriverState_Initialising;
	m_clientUsageRefCnt++;

	ClrErrorDescription();

	if( m_bInitialized )
		return MIstatus::success;

	bool bOk = MIstatus::success;
	CMIUtilString errMsg;

	// Initialize all of the modules we depend on
	MI::ModuleInit< CMICmnLog >         ( IDS_MI_INIT_ERR_LOG          , bOk, errMsg );
	MI::ModuleInit< CMICmnStreamStdout >( IDS_MI_INIT_ERR_STREAMSTDOUT , bOk, errMsg );
	MI::ModuleInit< CMICmnStreamStderr >( IDS_MI_INIT_ERR_STREAMSTDERR , bOk, errMsg );
	MI::ModuleInit< CMICmnResources >   ( IDS_MI_INIT_ERR_RESOURCES    , bOk, errMsg );
	MI::ModuleInit< CMICmnThreadMgrStd >( IDS_MI_INIT_ERR_THREADMANAGER, bOk, errMsg );
	MI::ModuleInit< CMICmnStreamStdin > ( IDS_MI_INIT_ERR_STREAMSTDIN  , bOk, errMsg );
	MI::ModuleInit< CMICmdMgr >         ( IDS_MI_INIT_ERR_CMDMGR       , bOk, errMsg );
	bOk &= m_rLldbDebugger.SetDriver( *this );
	MI::ModuleInit< CMICmnLLDBDebugger >( IDS_MI_INIT_ERR_LLDBDEBUGGER , bOk, errMsg );

#if MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER
	CMIDriverMgr & rDrvMgr = CMIDriverMgr::Instance();
	bOk = bOk && rDrvMgr.RegisterDriver( *g_driver, "LLDB driver" );	// Will be pass thru driver
	if( bOk )
	{
		bOk = SetEnableFallThru( false ); // This is intentional at this time - yet to be fully implemented
		bOk = bOk && SetDriverToFallThruTo( *g_driver );
		CMIUtilString strOtherDrvErrMsg;
		if( bOk && GetEnableFallThru() && !g_driver->MISetup( strOtherDrvErrMsg ) )
		{
			bOk = false;
			errMsg = CMIUtilString::Format( MIRSRC( IDS_MI_INIT_ERR_FALLTHRUDRIVER ), strOtherDrvErrMsg.c_str()  );
		}
	}
#endif // MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER

	m_bExitApp = false;
		
	m_bInitialized = bOk;

	if( !bOk )
	{
		const CMIUtilString msg = CMIUtilString::Format( MIRSRC( IDS_MI_INIT_ERR_DRIVER ), errMsg.c_str() );
		SetErrorDescription( msg );
		return MIstatus::failure;
	}

	m_eCurrentDriverState = eDriverState_RunningNotDebugging;
	
	return bOk;
}

//++ ------------------------------------------------------------------------------------
// Details:	Unbind detach or release resources used by *this driver.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::Shutdown( void )
{
	if( --m_clientUsageRefCnt > 0 )
		return MIstatus::success;
	
	if( !m_bInitialized )
		return MIstatus::success;

	m_eCurrentDriverState = eDriverState_ShuttingDown;

	ClrErrorDescription();

	bool bOk = MIstatus::success;
	CMIUtilString errMsg;

	// Shutdown all of the modules we depend on
	MI::ModuleShutdown< CMICmnLLDBDebugger >( IDS_MI_INIT_ERR_LLDBDEBUGGER , bOk, errMsg );
	MI::ModuleShutdown< CMICmdMgr >         ( IDS_MI_INIT_ERR_CMDMGR       , bOk, errMsg );
	MI::ModuleShutdown< CMICmnStreamStdin > ( IDS_MI_INIT_ERR_STREAMSTDIN  , bOk, errMsg );
	MI::ModuleShutdown< CMICmnThreadMgrStd >( IDS_MI_INIT_ERR_THREADMANAGER, bOk, errMsg );
	MI::ModuleShutdown< CMICmnResources >   ( IDS_MI_INIT_ERR_RESOURCES    , bOk, errMsg );
	MI::ModuleShutdown< CMICmnStreamStderr >( IDS_MI_INIT_ERR_STREAMSTDERR , bOk, errMsg );
	MI::ModuleShutdown< CMICmnStreamStdout >( IDS_MI_INIT_ERR_STREAMSTDOUT , bOk, errMsg );
	MI::ModuleShutdown< CMICmnLog >         ( IDS_MI_INIT_ERR_LOG          , bOk, errMsg );
																					 
	if( !bOk )
	{
		SetErrorDescriptionn( MIRSRC( IDS_MI_SHUTDOWN_ERR ), errMsg.c_str() );
	}

	m_eCurrentDriverState = eDriverState_NotRunning;

	return bOk;
}

//++ ------------------------------------------------------------------------------------
// Details:	Work function. Client (the driver's user) is able to append their own message 
//			in to the MI's Log trace file.
// Type:	Method.
// Args:	vMessage		- (R) Client's text message.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::WriteMessageToLog( const CMIUtilString & vMessage )
{
	CMIUtilString msg;
	msg = CMIUtilString::Format( MIRSRC( IDS_MI_CLIENT_MSG ), vMessage.c_str() );
	return m_pLog->Write( msg, CMICmnLog::eLogVerbosity_ClientMsg );
}

//++ ------------------------------------------------------------------------------------
// Details: CDriverMgr calls *this driver initialize setup ready for use.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::DoInitialize( void )
{
	return CMIDriver::Instance().Initialize();
}

//++ ------------------------------------------------------------------------------------
// Details:	CDriverMgr calls *this driver to unbind detach or release resources used by 
//			*this driver.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::DoShutdown( void )
{
	return CMIDriver::Instance().Shutdown();
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the name for *this driver.
// Type:	Overridden.
// Args:	None.
// Return:	CMIUtilString & - Driver name.
// Throws:	None.
//--
const CMIUtilString & CMIDriver::GetName( void ) const
{
	const CMIUtilString & rName = GetAppNameLong();
	const CMIUtilString & rVsn = GetVersionDescription();
	static CMIUtilString strName = CMIUtilString::Format( "%s %s", rName.c_str(), rVsn.c_str() );
	
	return strName;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve *this driver's last error condition.
// Type:	Overridden.
// Args:	None.
// Return:	CMIUtilString - Text description.
// Throws:	None.
//--
CMIUtilString CMIDriver::GetError( void ) const
{
	return GetErrorDescription();
}

//++ ------------------------------------------------------------------------------------
// Details:	Call *this driver to resize the console window.
// Type:	Overridden.
// Args:	vTermWidth	- (R) New window column size.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
void CMIDriver::DoResizeWindow( const uint32_t vTermWidth )
{
	GetTheDebugger().SetTerminalWidth( vTermWidth );
}

//++ ------------------------------------------------------------------------------------
// Details:	Call *this driver to return it's debugger.
// Type:	Overridden.
// Args:	None.
// Return:	lldb::SBDebugger & - LLDB debugger object reference.
// Throws:	None.
//--
lldb::SBDebugger & CMIDriver::GetTheDebugger( void )
{
	return m_rLldbDebugger.GetTheDebugger();
}

//++ ------------------------------------------------------------------------------------
// Details:	Specify another driver *this driver can call should this driver not be able 
//			to handle the client data input. DoFallThruToAnotherDriver() makes the call.
// Type:	Overridden.
// Args:	vrOtherDriver	- (R) Reference to another driver object.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::SetDriverToFallThruTo( const CMIDriverBase & vrOtherDriver )
{
	m_pDriverFallThru = const_cast< CMIDriverBase * >( &vrOtherDriver );

	return m_pDriverFallThru->SetDriverParent( *this );
}

//++ ------------------------------------------------------------------------------------
// Details:	Proxy function CMIDriverMgr IDriver interface implementation. *this driver's
//			implementation called from here to match the existing function name of the 
//			original LLDb driver class (the extra indirection is not necessarily required).
//			Check the arguments that were passed to this program to make sure they are 
//			valid and to get their argument values (if any).
// Type:	Overridden.
// Args:	argc		- (R)	An integer that contains the count of arguments that follow in 
//								argv. The argc parameter is always greater than or equal to 1.
//			argv		- (R)	An array of null-terminated strings representing command-line 
//								arguments entered by the user of the program. By convention, 
//								argv[0] is the command with which the program is invoked.
//			vpStdOut	- (R)	Pointer to a standard output stream. 
//			vwbExiting	- (W)	True = *this want to exit, Reasons: help, invalid arg(s),
//								version information only.
//								False = Continue to work, start debugger i.e. Command 
//								interpreter. 
// Return:	lldb::SBError - LLDB current error status.
// Throws:	None.
//--
lldb::SBError CMIDriver::DoParseArgs( const int argc, const char * argv[], FILE * vpStdOut, bool & vwbExiting )
{
	return ParseArgs( argc, argv, vpStdOut, vwbExiting );
}

//++ ------------------------------------------------------------------------------------
// Details:	Check the arguments that were passed to this program to make sure they are 
//			valid and to get their argument values (if any). The following are options 
//			that are only handled by *this driver: 
//				--executable 
//			The application's options --interpreter and --executable in code act very similar.
//			The --executable is necessary to differentiate whither the MI Driver is being
//			using by a client i.e. Eclipse or from the command line. Eclipse issues the option
//			--interpreter and also passes additional arguments which can be interpreted as an
//			executable if called from the command line. Using --executable tells the MI 
//			Driver is being called the command line and that the executable argument is indeed
//			a specified executable an so actions commands to set up the executable for a 
//			debug session. Using --interpreter on the commnd line does not action additional
//			commands to initialise a debug session and so be able to launch the process.
// Type:	Overridden.
// Args:	argc		- (R)	An integer that contains the count of arguments that follow in 
//								argv. The argc parameter is always greater than or equal to 1.
//			argv		- (R)	An array of null-terminated strings representing command-line 
//								arguments entered by the user of the program. By convention, 
//								argv[0] is the command with which the program is invoked.
//			vpStdOut	- (R)	Pointer to a standard output stream. 
//			vwbExiting	- (W)	True = *this want to exit, Reasons: help, invalid arg(s),
//								version information only.
//								False = Continue to work, start debugger i.e. Command 
//								interpreter. 
// Return:	lldb::SBError - LLDB current error status.
// Throws:	None.
//--
lldb::SBError CMIDriver::ParseArgs( const int argc, const char * argv[], FILE * vpStdOut, bool & vwbExiting )
{
	lldb::SBError errStatus;
	const bool bHaveArgs( argc >= 2 );
	
	// *** Add any args handled here to GetHelpOnCmdLineArgOptions() ***
	
	// CODETAG_MIDRIVE_CMD_LINE_ARG_HANDLING
	// Look for the command line options		
	bool bHaveExecutableFileNamePath = false;
	bool bHaveExecutableLongOption = false;
	
	if( bHaveArgs )
	{
		// Search right to left to look for the executable
		for( MIint i = argc - 1; i > 0; i-- ) 
		{ 
			const CMIUtilString strArg( argv[ i ] );
			const CMICmdArgValFile argFile;
			if( argFile.IsFilePath( strArg  ) || 
				CMICmdArgValString( true, false, true ).IsStringArg( strArg ))
			{
				bHaveExecutableFileNamePath = true;
				m_strCmdLineArgExecuteableFileNamePath = argFile.GetFileNamePath( strArg );
				m_bHaveExecutableFileNamePathOnCmdLine = true;
			}
			// This argument is also check for in CMIDriverMgr::ParseArgs()
			if( 0 == strArg.compare( "--executable" ) )	// Used to specify that there is executable argument also on the command line 
			{											// See fn description.
				   bHaveExecutableLongOption = true;
			}
		}
	}

	if( bHaveExecutableFileNamePath && bHaveExecutableLongOption )
	{
		// CODETAG_CMDLINE_ARG_EXECUTABLE_DEBUG_SESSION
#if MICONFIG_ENABLE_MI_DRIVER_MI_MODE_CMDLINE_ARG_EXECUTABLE_DEBUG_SESSION
		SetDriverDebuggingArgExecutable();
#else
		vwbExiting = true;
		errStatus.SetErrorString( MIRSRC( IDS_DRIVER_ERR_LOCAL_DEBUG_NOT_IMPL ) );
#endif // MICONFIG_ENABLE_MI_DRIVER_MI_MODE_CMDLINE_ARG_EXECUTABLE_DEBUG_SESSION
	}

	return errStatus;
}

//++ ------------------------------------------------------------------------------------
// Details:	A client can ask if *this driver is GDB/MI compatible.
// Type:	Overridden.
// Args:	None.
// Return:	True - GBD/MI compatible LLDB front end.
//			False - Not GBD/MI compatible LLDB front end.
// Throws:	None.
//--
bool CMIDriver::GetDriverIsGDBMICompatibleDriver( void ) const
{
	return true;
}

//++ ------------------------------------------------------------------------------------
// Details:	Callback function for monitoring stream stdin object. Part of the visitor 
//			pattern. 
//			This function is called by the CMICmnStreamStdin::CThreadStdin
//			"stdin monitor" thread (ID).
// Type:	Overridden.
// Args:	vStdInBuffer	- (R) Copy of the current stdin line data.
//			vrbYesExit		- (RW) True = yes exit stdin monitoring, false = continue monitor.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::ReadLine( const CMIUtilString & vStdInBuffer, bool & vrwbYesExit )
{
	// For debugging. Update prompt show stdin is working
	//printf( "%s\n", vStdInBuffer.c_str() );
	//fflush( stdout );

	// Special case look for the quit command here so stop monitoring stdin stream
	// So we do not go back to fgetc() and wait and hang thread on exit
	if( vStdInBuffer == "quit" )
		vrwbYesExit = true;

	// 1. Put new line in the queue container by stdin monitor thread
	// 2. Then *this driver calls ReadStdinLineQueue() when ready to read the queue in its
	// own thread
	const bool bOk = QueueMICommand( vStdInBuffer );

	// Check to see if the *this driver is shutting down (exit application)
	if( !vrwbYesExit )
		vrwbYesExit = m_bDriverIsExiting;

	return bOk;
}

//++ ------------------------------------------------------------------------------------
// Details:	Start worker threads for the driver.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::StartWorkerThreads( void )
{
	bool bOk = MIstatus::success;
	
	// Grab the thread manager
	CMICmnThreadMgrStd & rThreadMgr = CMICmnThreadMgrStd::Instance();

	// Start the stdin thread
	bOk &= m_rStdin.SetVisitor( *this );
	if( bOk && !rThreadMgr.ThreadStart< CMICmnStreamStdin >( m_rStdin ))
	{
		const CMIUtilString errMsg = CMIUtilString::Format( MIRSRC( IDS_THREADMGR_ERR_THREAD_FAIL_CREATE ), CMICmnThreadMgrStd::Instance().GetErrorDescription().c_str() );
		SetErrorDescriptionn( errMsg );
		return MIstatus::failure;
	}

	// Start the event polling thread
	if( bOk && !rThreadMgr.ThreadStart< CMICmnLLDBDebugger >( m_rLldbDebugger ) )
	{
		const CMIUtilString errMsg = CMIUtilString::Format( MIRSRC( IDS_THREADMGR_ERR_THREAD_FAIL_CREATE ), CMICmnThreadMgrStd::Instance().GetErrorDescription().c_str() );
		SetErrorDescriptionn( errMsg );
		return MIstatus::failure;
	}

	return bOk;
}

//++ ------------------------------------------------------------------------------------
// Details:	Stop worker threads for the driver.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::StopWorkerThreads( void )
{
	CMICmnThreadMgrStd & rThreadMgr = CMICmnThreadMgrStd::Instance();
	return rThreadMgr.ThreadAllTerminate();
}

//++ ------------------------------------------------------------------------------------
// Details:	Call this function puts *this driver to work.
//			This function is used by the application's main thread.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::DoMainLoop( void )
{
	if( !InitClientIDEToMIDriver() ) // Init Eclipse IDE
	{
		SetErrorDescriptionn( MIRSRC( IDS_MI_INIT_ERR_CLIENT_USING_DRIVER ) );
		return MIstatus::failure;
	}

	if( !StartWorkerThreads() )
		return MIstatus::failure;
	
	// App is not quitting currently
	m_bExitApp = false;

	// CODETAG_CMDLINE_ARG_EXECUTABLE_DEBUG_SESSION
#if MICONFIG_ENABLE_MI_DRIVER_MI_MODE_CMDLINE_ARG_EXECUTABLE_DEBUG_SESSION
	if( HaveExecutableFileNamePathOnCmdLine() )
	{
		if( !LocalDebugSessionStartupInjectCommands() )
		{
			SetErrorDescription( MIRSRC( IDS_MI_INIT_ERR_LOCAL_DEBUG_SESSION ) );
			return MIstatus::failure;
		}
	}
#endif // MICONFIG_ENABLE_MI_DRIVER_MI_MODE_CMDLINE_ARG_EXECUTABLE_DEBUG_SESSION

	// While the app is active
	while( !m_bExitApp )
	{
		// Poll stdin queue and dispatch
		if( !ReadStdinLineQueue() )
		{
			// Something went wrong
			break;
		}
	}

	// Signal that the application is shutting down
	DoAppQuit();

	// Close and wait for the workers to stop
	StopWorkerThreads();

	// Ensure that a new line is sent as the last act of the dying driver
	m_rStdOut.WriteMIResponse( "\n", false );

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	*this driver sits and waits for input to the stdin line queue shared by *this
//			driver and the stdin monitor thread, it queues, *this reads, interprets and
//			reacts.
//			This function is used by the application's main thread.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::ReadStdinLineQueue( void )
{
	// True when queue contains input
	bool bHaveInput = false;

	// Stores the current input line
	CMIUtilString lineText;
	{
		// Lock while we access the queue
		CMIUtilThreadLock lock( m_threadMutex );
		if( !m_queueStdinLine.empty() )
		{
			lineText = m_queueStdinLine.front();
			m_queueStdinLine.pop();
			bHaveInput = !lineText.empty();
		}
	}

	// Process while we have input
	if( bHaveInput )
	{
		if( lineText == "quit" )
		{
			// We want to be exiting when receiving a quit command
			m_bExitApp = true;
			return MIstatus::success;
		}

		// Process the command
		const bool bOk = InterpretCommand( lineText );

		// Draw prompt if desired
		if( bOk && m_rStdin.GetEnablePrompt() )
			m_rStdOut.WriteMIResponse( m_rStdin.GetPrompt() );

		// Input has been processed
		bHaveInput = false;
	}
	else
	{
		// Give resources back to the OS
		const std::chrono::milliseconds time( 1 );
		std::this_thread::sleep_for( time );
	}
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Set things in motion, set state etc that brings *this driver (and the 
//			application) to a tidy shutdown.
//			This function is used by the application's main thread.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::DoAppQuit( void )
{
	bool bYesQuit = true;

	// Shutdown stuff, ready app for exit
	{
		CMIUtilThreadLock lock( m_threadMutex );
		m_bDriverIsExiting = true;
	}

	return bYesQuit;
}

//++ ------------------------------------------------------------------------------------
// Details:	*this driver passes text commands to a fall through driver is it does not
//			understand them (the LLDB driver).
//			This function is used by the application's main thread.
// Type:	Method.
// Args:	vTextLine			- (R) Text data representing a possible command.
//			vwbCmdYesValid		- (W) True = Command valid, false = command not handled.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::InterpretCommandFallThruDriver( const CMIUtilString & vTextLine, bool & vwbCmdYesValid )
{
	MIunused( vTextLine );
	MIunused( vwbCmdYesValid );

	// ToDo: Implement when less urgent work to be done or decide remove as not required
	//bool bOk = MIstatus::success;
	//bool bCmdNotUnderstood = true;
	//if( bCmdNotUnderstood && GetEnableFallThru() )
	//{
	//	CMIUtilString errMsg;
	//	bOk = DoFallThruToAnotherDriver( vStdInBuffer, errMsg );
	//	if( !bOk )
	//	{
	//		errMsg = errMsg.StripCREndOfLine();
	//		errMsg = errMsg.StripCRAll();
	//		const CMIDriverBase * pOtherDriver = GetDriverToFallThruTo();
	//		const MIchar * pName = pOtherDriver->GetDriverName().c_str();
	//		const MIchar * pId = pOtherDriver->GetDriverId().c_str();
	//		const CMIUtilString msg( CMIUtilString::Format( MIRSRC( IDS_DRIVER_ERR_FALLTHRU_DRIVER_ERR ), pName, pId, errMsg.c_str() ) );
	//		m_pLog->WriteMsg( msg );
	//	}
	//}
	//
	//vwbCmdYesValid = bOk;
	//CMIUtilString strNot;
	//if( vwbCmdYesValid)
	//	strNot = CMIUtilString::Format( "%s ", MIRSRC( IDS_WORD_NOT ) );
	//const CMIUtilString msg( CMIUtilString::Format( MIRSRC( IDS_FALLTHRU_DRIVER_CMD_RECEIVED ), vTextLine.c_str(), strNot.c_str() ) );
	//m_pLog->WriteLog( msg );
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the name for *this driver.
// Type:	Overridden.
// Args:	None.
// Return:	CMIUtilString & - Driver name.
// Throws:	None.
//--
const CMIUtilString & CMIDriver::GetDriverName( void ) const
{
	return GetName();
}

//++ ------------------------------------------------------------------------------------
// Details:	Get the unique ID for *this driver.
// Type:	Overridden.
// Args:	None.
// Return:	CMIUtilString & - Text description.
// Throws:	None.
//--
const CMIUtilString & CMIDriver::GetDriverId( void ) const
{
	return GetId();
}

//++ ------------------------------------------------------------------------------------
// Details:	This function allows *this driver to call on another driver to perform work
//			should this driver not be able to handle the client data input.
//			SetDriverToFallThruTo() specifies the fall through to driver.
//			Check the error message if the function returns a failure.
// Type:	Overridden.
// Args:	vCmd		- (R) Command instruction to interpret.
//			vwErrMsg	- (W) Error description on command failing.
// Return:	MIstatus::success - Command succeeded.
//			MIstatus::failure - Command failed.
// Throws:	None.
//--
bool CMIDriver::DoFallThruToAnotherDriver( const CMIUtilString & vCmd, CMIUtilString & vwErrMsg )
{
	bool bOk = MIstatus::success;

	CMIDriverBase * pOtherDriver = GetDriverToFallThruTo();
	if( pOtherDriver == nullptr )
		return bOk;

	return pOtherDriver->DoFallThruToAnotherDriver( vCmd, vwErrMsg );
}

//++ ------------------------------------------------------------------------------------
// Details:	*this driver provides a file stream to other drivers on which *this driver
//			write's out to and they read as expected input. *this driver is passing
//			through commands to the (child) pass through assigned driver.
// Type:	Overrdidden.
// Args:	None.
// Return:	FILE * - Pointer to stream.
// Throws:	None.
//--
FILE * CMIDriver::GetStdin( void ) const
{
	// Note this fn is called on CMIDriverMgr register driver so stream has to be
	// available before *this driver has been initialized! Flaw?

	// This very likely to change later to a stream that the pass thru driver
	// will read and we write to give it 'input'
	return stdin;
}

//++ ------------------------------------------------------------------------------------
// Details:	*this driver provides a file stream to other pass through assigned drivers 
//			so they know what to write to.
// Type:	Overidden.
// Args:	None.
// Return:	FILE * - Pointer to stream.
// Throws:	None.
//--
FILE * CMIDriver::GetStdout( void ) const
{
	// Note this fn is called on CMIDriverMgr register driver so stream has to be
	// available before *this driver has been initialized! Flaw?

	// Do not want to pass through driver to write to stdout
	return NULL;
}

//++ ------------------------------------------------------------------------------------
// Details:	*this driver provides a error file stream to other pass through assigned drivers 
//			so they know what to write to.
// Type:	Overidden.
// Args:	None.
// Return:	FILE * - Pointer to stream.
// Throws:	None.
//--
FILE * CMIDriver::GetStderr( void ) const
{
	// Note this fn is called on CMIDriverMgr register driver so stream has to be
	// available before *this driver has been initialized! Flaw?

	// This very likely to change later to a stream that the pass thru driver
	// will write to and *this driver reads from to pass on the the CMICmnLog object
	return stderr;
}

//++ ------------------------------------------------------------------------------------
// Details:	Set a unique ID for *this driver. It cannot be empty.
// Type:	Overridden.
// Args:	vId	- (R) Text description.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::SetId( const CMIUtilString & vId )
{
	if( vId.empty() )
	{
		SetErrorDescriptionn( MIRSRC( IDS_DRIVER_ERR_ID_INVALID ), GetName().c_str(), vId.c_str() );
		return MIstatus::failure;
	}

	m_strDriverId = vId;
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Get the unique ID for *this driver.
// Type:	Overridden.
// Args:	None.
// Return:	CMIUtilString & - Text description.
// Throws:	None.
//--
const CMIUtilString & CMIDriver::GetId( void ) const
{
	return m_strDriverId;
}

//++ ------------------------------------------------------------------------------------
// Details:	Inject a command into the command processing system to be interpreted as a
//			command read from stdin. The text representing the command is also written
//			out to stdout as the command did not come from via stdin.
// Type:	Method.
// Args:	vMICmd	- (R) Text data representing a possible command.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::InjectMICommand( const CMIUtilString & vMICmd )
{
	const bool bOk = m_rStdOut.WriteMIResponse( vMICmd );

	return bOk && QueueMICommand( vMICmd );
}

//++ ------------------------------------------------------------------------------------
// Details:	Add a new command candidate to the command queue to be processed by the 
//			command system.
// Type:	Method.
// Args:	vMICmd	- (R) Text data representing a possible command.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::QueueMICommand( const CMIUtilString & vMICmd )
{
	CMIUtilThreadLock lock( m_threadMutex );
	m_queueStdinLine.push( vMICmd );
	
	return MIstatus::success;
}
	
//++ ------------------------------------------------------------------------------------
// Details:	Interpret the text data and match against current commands to see if there 
//			is a match. If a match then the command is issued and actioned on. The 
//			text data if not understood by *this driver is past on to the Fall Thru
//			driver.
//			This function is used by the application's main thread.
// Type:	Method.
// Args:	vTextLine	- (R) Text data representing a possible command.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::InterpretCommand( const CMIUtilString & vTextLine )
{
	bool bCmdYesValid = false;
	bool bOk = InterpretCommandThisDriver( vTextLine, bCmdYesValid );
	if( bOk && !bCmdYesValid )
		bOk = InterpretCommandFallThruDriver( vTextLine, bCmdYesValid );

	return bOk;
}

//++ ------------------------------------------------------------------------------------
// Details:	Interpret the text data and match against current commands to see if there 
//			is a match. If a match then the command is issued and actioned on. If a
//			command cannot be found to match then vwbCmdYesValid is set to false and
//			nothing else is done here.
//			This function is used by the application's main thread.
// Type:	Method.
// Args:	vTextLine			- (R) Text data representing a possible command.
//			vwbCmdYesValid		- (W) True = Command invalid, false = command acted on.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::InterpretCommandThisDriver( const CMIUtilString & vTextLine, bool & vwbCmdYesValid )
{
	vwbCmdYesValid = false;

	bool bCmdNotInCmdFactor = false;
	SMICmdData cmdData;
	CMICmdMgr & rCmdMgr = CMICmdMgr::Instance();
	if( !rCmdMgr.CmdInterpret( vTextLine, vwbCmdYesValid, bCmdNotInCmdFactor, cmdData ) )
		return MIstatus::failure;
	
	if( vwbCmdYesValid )
	{
		// For debugging only
		//m_pLog->WriteLog( cmdData.strMiCmdAll.c_str() );
		
		return ExecuteCommand( cmdData );
	}

	// Check for escape character, may be cursor control characters
	// This code is not necessary for application operation, just want to keep tabs on what 
	// is been given to the driver to try and intepret.
	if( vTextLine.at( 0 ) == 27 )
	{
		CMIUtilString logInput( MIRSRC( IDS_STDIN_INPUT_CTRL_CHARS ) );
		for( MIuint i = 0; i < vTextLine.length(); i++ )
		{
			logInput += CMIUtilString::Format( "%d ", vTextLine.at( i ) );
		}
		m_pLog->WriteLog( logInput );
		return MIstatus::success;
	}

	// Write to the Log that a 'command' was not valid. 
	// Report back to the MI client via MI result record.
	CMIUtilString strNotInCmdFactory;
	if( bCmdNotInCmdFactor )
		strNotInCmdFactory = CMIUtilString::Format( MIRSRC( IDS_DRIVER_CMD_NOT_IN_FACTORY ), cmdData.strMiCmd.c_str() );
	const CMIUtilString strNot( CMIUtilString::Format( "%s ", MIRSRC( IDS_WORD_NOT ) ) );
	const CMIUtilString msg( CMIUtilString::Format( MIRSRC( IDS_DRIVER_CMD_RECEIVED ), vTextLine.c_str(), strNot.c_str(), strNotInCmdFactory.c_str() ) );
	const CMICmnMIValueConst vconst = CMICmnMIValueConst( msg );
	const CMICmnMIValueResult valueResult( "msg", vconst );
	const CMICmnMIResultRecord miResultRecord( cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, valueResult );
	m_rStdOut.WriteMIResponse( miResultRecord.GetString() );
	
	// Proceed to wait for or execute next command
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Having previously had the potential command validated and found valid now
//			get the command executed.
//			This function is used by the application's main thread.
// Type:	Method.
// Args:	vCmdData	- (RW) Command meta data.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIDriver::ExecuteCommand( const SMICmdData & vCmdData )
{
	CMICmdMgr & rCmdMgr = CMICmdMgr::Instance();
	return rCmdMgr.CmdExecute( vCmdData );
}

//++ ------------------------------------------------------------------------------------
// Details:	Set the MI Driver's exit application flag. The application checks this flag 
//			after every stdin line is read so the exit may not be instantious.
//			If vbForceExit is false the MI Driver queries its state and determines if is
//			should exit or continue operating depending on that running state.
//			This is related to the running state of the MI driver.
// Type:	Overridden.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
void CMIDriver::SetExitApplicationFlag( const bool vbForceExit )
{
	if( vbForceExit )
	{
		CMIUtilThreadLock lock( m_threadMutex );
		m_bExitApp = true;
		return;
	}

	// CODETAG_DEBUG_SESSION_RUNNING_PROG_RECEIVED_SIGINT_PAUSE_PROGRAM
	// Did we receive a SIGINT from the client during a running debug program, if
	// so then SIGINT is not to be taken as meaning kill the MI driver application
	// but halt the inferior program being debugged instead
	if( m_eCurrentDriverState == eDriverState_RunningDebugging )
	{
		InjectMICommand( "-exec-interrupt" );
		return;
	}

	m_bExitApp = true;
}

//++ ------------------------------------------------------------------------------------
// Details:	Get the  MI Driver's exit exit application flag. 
//			This is related to the running state of the MI driver.
// Type:	Method.
// Args:	None.
// Return:	bool	- True = MI Driver is shutting down, false = MI driver is running.
// Throws:	None.
//--
bool CMIDriver::GetExitApplicationFlag( void ) const
{
	return m_bExitApp;
}

//++ ------------------------------------------------------------------------------------
// Details:	Get the current running state of the MI Driver. 
// Type:	Method.
// Args:	None.
// Return:	DriverState_e	- The current running state of the application.
// Throws:	None.
//--
CMIDriver::DriverState_e CMIDriver::GetCurrentDriverState( void ) const
{
	return m_eCurrentDriverState;
}

//++ ------------------------------------------------------------------------------------
// Details:	Set the current running state of the MI Driver to running and currently not in
//			a debug session. 
// Type:	Method.
// Return:	MIstatus::success - Functionality succeeded.
//			MIstatus::failure - Functionality failed.
// Return:	DriverState_e	- The current running state of the application.
// Throws:	None.
//--
bool CMIDriver::SetDriverStateRunningNotDebugging( void )
{
	// CODETAG_DEBUG_SESSION_RUNNING_PROG_RECEIVED_SIGINT_PAUSE_PROGRAM
		
	if( m_eCurrentDriverState == eDriverState_RunningNotDebugging )
		return MIstatus::success;

	// Driver cannot be in the following states to set eDriverState_RunningNotDebugging
	switch( m_eCurrentDriverState )
	{
	case eDriverState_NotRunning:
	case eDriverState_Initialising:
	case eDriverState_ShuttingDown:
	{
		SetErrorDescription( MIRSRC( IDS_DRIVER_ERR_DRIVER_STATE_ERROR ) );
		return MIstatus::failure;
	}
	case eDriverState_RunningDebugging:
	case eDriverState_RunningNotDebugging:
		break;
	case eDriverState_count:
	default:
		SetErrorDescription( CMIUtilString::Format( MIRSRC( IDS_CODE_ERR_INVALID_ENUMERATION_VALUE ), "SetDriverStateRunningNotDebugging()" ) );
		return MIstatus::failure;
	}

	// Driver must be in this state to set eDriverState_RunningNotDebugging
	if( m_eCurrentDriverState != eDriverState_RunningDebugging )
	{
		SetErrorDescription( MIRSRC( IDS_DRIVER_ERR_DRIVER_STATE_ERROR ) );
		return MIstatus::failure;
	}

	m_eCurrentDriverState = eDriverState_RunningNotDebugging;

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Set the current running state of the MI Driver to running and currently not in
//			a debug session. The driver's state must in the state running and in a
//			debug session to set this new state.
// Type:	Method.
// Return:	MIstatus::success - Functionality succeeded.
//			MIstatus::failure - Functionality failed.
// Return:	DriverState_e	- The current running state of the application.
// Throws:	None.
//--
bool CMIDriver::SetDriverStateRunningDebugging( void )
{
	// CODETAG_DEBUG_SESSION_RUNNING_PROG_RECEIVED_SIGINT_PAUSE_PROGRAM
		
	if( m_eCurrentDriverState == eDriverState_RunningDebugging )
		return MIstatus::success;

	// Driver cannot be in the following states to set eDriverState_RunningDebugging
	switch( m_eCurrentDriverState )
	{
	case eDriverState_NotRunning:
	case eDriverState_Initialising:
	case eDriverState_ShuttingDown:
	{
		SetErrorDescription( MIRSRC( IDS_DRIVER_ERR_DRIVER_STATE_ERROR ) );
		return MIstatus::failure;
	}
	case eDriverState_RunningDebugging:
	case eDriverState_RunningNotDebugging:
		break;
	case eDriverState_count:
	default:
		SetErrorDescription( CMIUtilString::Format( MIRSRC( IDS_CODE_ERR_INVALID_ENUMERATION_VALUE ), "SetDriverStateRunningDebugging()" ) );
		return MIstatus::failure;
	}

	// Driver must be in this state to set eDriverState_RunningDebugging
	if( m_eCurrentDriverState != eDriverState_RunningNotDebugging )
	{
		SetErrorDescription( MIRSRC( IDS_DRIVER_ERR_DRIVER_STATE_ERROR ) );
		return MIstatus::failure;
	}

	m_eCurrentDriverState = eDriverState_RunningDebugging;

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Prepare the client IDE so it will start working/communicating with *this MI 
//			driver.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functionality succeeded.
//			MIstatus::failure - Functionality failed.
// Throws:	None.
//--
bool CMIDriver::InitClientIDEToMIDriver( void ) const
{
	// Put other IDE init functions here
	return InitClientIDEEclipse();
}

//++ ------------------------------------------------------------------------------------
// Details: The IDE Eclipse when debugging locally expects "(gdb)\n" character
//			sequence otherwise it refuses to communicate and times out. This should be
//			sent to Eclipse before anything else.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functionality succeeded.
//			MIstatus::failure - Functionality failed.
// Throws:	None.
//--
bool CMIDriver::InitClientIDEEclipse( void ) const
{
	std::cout << "(gdb)" << std::endl;

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Ask *this driver whether it found an executable in the MI Driver's list of 
//			arguments which to open and debug. If so instigate commands to set up a debug
//			session for that executable.
// Type:	Method.
// Args:	None.
// Return:	bool - True = True = Yes executable given as one of the parameters to the MI 
//				   Driver.
//				   False = not found.
// Throws:	None.
//--
bool CMIDriver::HaveExecutableFileNamePathOnCmdLine( void ) const
{
	return m_bHaveExecutableFileNamePathOnCmdLine;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve from *this driver executable file name path to start a debug session
//			with (if present see HaveExecutableFileNamePathOnCmdLine()).
// Type:	Method.
// Args:	None.
// Return:	CMIUtilString & - Executeable file name path or empty string.
// Throws:	None.
//--
const CMIUtilString & CMIDriver::GetExecutableFileNamePathOnCmdLine( void ) const
{
	return m_strCmdLineArgExecuteableFileNamePath;
}

//++ ------------------------------------------------------------------------------------
// Details: Execute commands (by injecting them into the stdin line queue container) and
//			other code to set up the MI Driver such that is can take the executable 
//			argument passed on the command and create a debug session for it.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functionality succeeded.
//			MIstatus::failure - Functionality failed.
// Throws:	None.
//--
bool CMIDriver::LocalDebugSessionStartupInjectCommands( void )
{
	const CMIUtilString strCmd( CMIUtilString::Format( "-file-exec-and-symbols %s", m_strCmdLineArgExecuteableFileNamePath.c_str() ) );
	
	return InjectMICommand( strCmd );
}

//++ ------------------------------------------------------------------------------------
// Details: Set the MI Driver into "its debugging an executable passed as an argument"
//			mode as against running via a client like Eclipse.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
void CMIDriver::SetDriverDebuggingArgExecutable( void )
{
	m_bDriverDebuggingArgExecutable = true;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the MI Driver state indicating if it is operating in "its debugging 
//			an executable passed as an argument" mode as against running via a client 
//			like Eclipse.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
bool CMIDriver::IsDriverDebuggingArgExecutable( void ) const
{
	return m_bDriverDebuggingArgExecutable;
}
